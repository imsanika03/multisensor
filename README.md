# modelcascade

A resolution cascade: an ordered list of image classifiers `[f1, ..., fM]` run
at increasing input resolutions (cheap → expensive). The goal is to learn when a
cheap, low-resolution model is good enough vs. when an input needs the expensive
high-resolution model.

## Running

```bash
# run from the parent directory (modelcascade resolves as a namespace package)
cd /home/sanikabhar
python -m modelcascade.main --cfg modelcascade/cfgs/resolution_cascade.yaml
```

### Multi-GPU

With `data_parallel: true` in the config (and `num_workers` set), each batch is
split across all **visible** GPUs via `nn.DataParallel`. `batch_size` is the
global batch — it is divided among the GPUs, so size it for the total (e.g. 64
across 4 GPUs = 16/GPU). Control which GPUs are used with `CUDA_VISIBLE_DEVICES`:

```bash
# use all 4 GPUs
python -m modelcascade.main --cfg modelcascade/cfgs/resolution_cascade.yaml

# pin to GPUs 0 and 1 only
CUDA_VISIBLE_DEVICES=0,1 python -m modelcascade.main --cfg modelcascade/cfgs/resolution_cascade.yaml
```

Checkpoints are saved unwrapped, so they load fine in either single- or
multi-GPU runs.

## Config (`cfgs/resolution_cascade.yaml`)

```yaml
dataset: cifar100

cascade_id: resnet50_res_cascade_v1   # run identifier (see Checkpoints)
checkpoint_dir: checkpoints           # root for all runs' checkpoints

train:
  epochs: 5          # max epochs; training stops early on val plateau
  batch_size: 64
  lr: 0.001
  patience: 2        # epochs with no val-loss improvement before stopping

probe:               # tiny routing probe g_phi
  resolution: 96     # cheap low-res view the probe runs on
  z_dim: 128
  epochs: 10
  batch_size: 64
  lr: 0.001

routing:
  epsilon: 0.02      # target harmful-routing risk tolerance (thresholds calibrated to this)
  delta: 0.05        # 1 - delta = confidence level of the binomial UCB (95%)

cascade:             # ordered stages, cheap -> expensive
  f1: {backbone: resnet50, resolution: 112}
  f2: {backbone: resnet50, resolution: 224}
  f3: {backbone: resnet50, resolution: 448}   # this is fM (the expensive oracle)
```

Nothing is hardcoded: the number of stages, each `backbone` (any model in
`torchvision.models`), and each `resolution` all come from the config.

## Checkpoint convention (`cascade_id`)

Checkpoints are namespaced by `cascade_id`. Each run's weights live in:

```
<checkpoint_dir>/<cascade_id>_chkpts/<stage>_<backbone>_<resolution>px.pt
```

The trained probe is checkpointed in the same folder as `probe.pt`.

Behaviour in `build_cascade` (and likewise for the probe):

- If a stage's checkpoint already exists in that folder, it is **loaded** and
  training is **skipped**.
- Otherwise the stage is **trained and saved** there.

The checkpoint filename captures only the architecture (`backbone`,
`resolution`) — **not** the training recipe (`lr`, `epochs`, `batch_size`,
`patience`, dataset preprocessing). `cascade_id` is how you version the recipe:

- **Changed dataset / lr / epochs / anything you want fresh results for →
  bump `cascade_id`.** This starts a new `<cascade_id>_chkpts/` folder and
  trains from scratch.
- **Want to reuse or extend an earlier run's weights →
  keep the same `cascade_id`.** Its saved weights are loaded instead of
  retraining.

> Caveat: this is a convention, not an enforced check. If you change `lr` but
> forget to bump `cascade_id`, the old weights are still loaded silently. Treat
> `cascade_id` as the version of the training recipe.

## Data splits (no leakage)

`main.py` splits the dataset sequentially 70 / 10 / 10 / 10, each role disjoint:

| split | fraction | role |
|-------|----------|------|
| `train`        | 70% | train stage weights |
| `val`          | 10% | early-stopping / model selection **and** routing evaluation |
| `probe_calib`  | 10% | train the probe (its harm labels) |
| `thresh_calib` | 10% | calibrate thresholds |

The probe is trained on `probe_calib` and thresholds are calibrated on a
**separate** split (`thresh_calib`) the probe never trained on — so the binomial
upper confidence bound used in calibration is honest (it accounts for finite
samples *and* is not optimistic from the probe overfitting its own training
data). Routing is evaluated on `val`; the cascade saw `val` only for
early-stopping selection, and the routed-vs-`fM` comparison uses the same data
and models, so the comparison is fair.

## Harm labels (probe targets)

For each held-out image and each cheap route `k` (the first M-1 stages):

```
Hk(x) = 1  if  fk is wrong  and  fM is correct
Hk(x) = 0  otherwise
```

`Hk` flags inputs where routing to cheap model `k` would hurt accuracy relative
to the expensive model `fM`. `compute_harm_labels` returns `H` of shape
`[N, M-1]` (plus ground-truth `labels`); these are the training targets for the
probe's route-safety head. See `score/harm.py`.

## Probe and routing

The probe `g_phi` (`models/probe.py`) is a tiny CNN that runs on a cheap
low-resolution view of the input (`probe.resolution`, default 96px). It is
trained on the calibration set: the route head learns **safety**
`s_k = sigmoid(route_logit_k)` against the target `1 - Hk` (a route is "safe"
when it is not harmful) via BCE; the class head is an auxiliary cross-entropy
target that regularizes the embedding and is not used for routing.

At test time, routing walks the cheap routes in order:

```
run g_phi(x) -> s_1, ..., s_{M-1}
if   s_1 >= tau_1: choose f1
elif s_2 >= tau_2: choose f2
...
else:              choose fM
```

### Threshold calibration (rigorous, conformal-style)

Thresholds `tau_k` are **calibrated on the calibration set**, not hand-set. For
each cheap route we use the harm labels `Lk = Hk` (1 = routing to `fk` loses a
correction `fM` would have made) and the probe's safety scores `s_k`:

1. Calibrate routes **sequentially**. Route `k` is calibrated only on the
   examples that did **not** pass any earlier threshold (`B_k`, the test-time
   fall-through pool), so the thresholds compose with the decision rule.
2. For a candidate `tau`, the accepted set is `{x in B_k : s_k(x) >= tau}`. We
   compute the **lost-correction rate** among accepted examples and its
   **binomial upper confidence bound** (one-sided Clopper–Pearson at confidence
   `1 - delta`).
3. Select the **least conservative** (lowest) `tau` whose UCB stays
   `<= epsilon` — the most aggressive threshold that is still certified safe.

Using the UCB (not the raw empirical rate) makes calibration conservative when
few examples pass: e.g. 0 harmful in 40 still has a ~7% UCB, so with `epsilon`
2% a route is left disabled (`tau = inf`, everything falls through to `fM`)
until enough calibration data certifies it. `epsilon` and `delta` are config.
See `calibrate_thresholds` in `score/routing.py`.

## Routing metrics

`evaluate_routing` reports, on the `val` split, the routed cascade against the
**always-run-fM** reference (the most important row):

| metric | meaning |
|--------|---------|
| **Top-1 accuracy** | correct only if the single highest-confidence class matches the true label — did routing preserve predictive performance? |
| **Accuracy drop** | `fM_acc - routed_acc` — how much was lost vs always running `fM` |
| **Average FLOPs** | mean per-image forward FLOPs: probe (always) + the chosen stage; theoretical compute savings |

It also prints the routing distribution (what fraction went to each stage).
FLOPs are counted with `fvcore` (multiply-accumulates, per image at each stage's
resolution); use them as a consistent *relative* measure, not exact hardware
cost.

## Layout

```
main.py                 # entry: split -> train cascade -> harm labels -> train probe -> route+eval
cfgs/                   # run configs
cascade/stage.py        # CascadeStage, build_cascade (training, early stopping, checkpoints)
models/probe.py         # the probe network + train_probe (embedding + class + route-safety heads)
score/harm.py           # calibration split + harm-label construction + predict_stage
score/routing.py        # decision rule, probe safety, FLOPs, evaluate_routing
utils.py                # get_dataset + shared resize/normalize helpers
```
