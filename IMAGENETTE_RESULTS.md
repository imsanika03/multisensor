# Resolution-cascade routing — Imagenette results

Resolution cascade `f1@112 → f2@224 → f3@448` (ResNet50, ImageNet-pretrained,
fine-tuned per stage), evaluated on **Imagenette** (full-size, 10 classes). Native
high-resolution photos, so 112/224/448 are genuine *downsampled* views and higher
resolution genuinely helps — unlike CIFAR-100 (native 32px), where upsampling made
the "expensive" stage the *worst* model and inverted the whole premise.

## Setup
- Per-stage fine-tune: max 30 epochs, early stop on **val top-1 accuracy**, patience 8.
- Data: single dataset, fixed-seed shuffle, split 70% train / 10% val / 10% probe-calib / 10% thresh-calib.
- Routing thresholds calibrated on the held-out thresh-calib split via a binomial (Clopper-Pearson) UCB ≤ ε at 95% confidence; evaluated on val.
- Harm label `H_k = 1` iff cheap stage k is wrong **and** fM (f3) is right.

## Stages are right-side-up (fM is a real oracle)

| stage | resolution | val top-1 |
|-------|-----------|-----------|
| f1 | 112px | 0.9229 |
| f2 | 224px | 0.9430 |
| f3 | 448px | **0.9451** |

Monotonically increasing → fM is the best model and the harm framework is valid.
Base harm rates: **f1 = 7.1%, f2 = 4.65%** (f2 ≈ f3, so routing to f2 rarely loses a correction).

## Routing strategies (n=947 eval)

References (ε-independent):

| strategy | acc | drop vs f3 | avg GFLOPs | ×f3 |
|----------|-----|-----------|-----------|-----|
| Always f1 | 0.9229 | +0.0222 | 1.08 | 0.07× |
| Always f2 | 0.9430 | +0.0021 | 4.11 | 0.25× |
| Always f3 | 0.9451 | 0.0000 | 16.44 | 1.00× |
| **Oracle cheapest-correct** | **0.9863** | −0.0412 | 1.67 | **0.10×** |

Calibrated strategies:

| ε | strategy | acc | drop | ×f3 | cheap-routed | harm@routed | distribution |
|---|----------|-----|------|-----|-------------|-------------|--------------|
| 0.05 | A probe one-shot | 0.9451 | +0.000 | 0.95× | 6%  | 0.036 | f1=6% f2=0% f3=94% |
| 0.05 | **B conf cascade** | 0.9430 | +0.002 | **0.14×** | 94% | 0.033 ✅ | f1=94% f2=0% f3=6% |
| 0.05 | C hybrid | 0.9430 | +0.002 | 0.14× | 94% | 0.034 ✅ | f1=94% f2=0% f3=6% |
| 0.10 | A probe one-shot | 0.9430 | +0.002 | 0.88× | 15% | 0.029 | f1=7% f2=7% f3=85% |
| 0.10 | B conf cascade | 0.9240 | +0.021 | 0.07× | 100% | 0.056 | f1=100% |
| 0.10 | C hybrid | 0.9250 | +0.020 | 0.08× | 100% | f1=93% f2=7% |

(Strategies: **A** = probe predicts the single stage, cost `gϕ + flops[chosen]`; **B** =
sequential confidence cascade on each stage's max-softmax, cost `cumsum(flops)`; **C** =
probe first, else fall into B.)

## Findings

1. **Confidence ≫ probe.** At ε=0.05, **B/C reach 0.14× FLOPs (≈7× speedup) for a 0.2% accuracy drop**, realized harm 3.3% (honestly under budget). The pixel probe (A) manages only 0.95× — its signal (AUC ≈ 0.57) can't certify cheap routes. The model's own confidence can.

2. **B beats "just use f2."** Always-f2 is 0.9430 @ 0.25×; B matches that accuracy at **0.14×** — roughly half the cost again, by sending easy inputs to f1 and only escalating when f1 is unsure.

3. **f2 and the 448px stage barely earn their keep.** B/C route **0% to f2** (f1's confidence either accepts or jumps past f2 to f3), and f2 (0.9430) ≈ f3 (0.9451). A 2-stage `f1→f3` (or `f1→f2`) cascade would likely capture nearly all the value at lower complexity.

4. **Large oracle headroom.** Perfect cheapest-correct routing hits **0.9863 at 0.10×** (above f3's 0.9451) — substantial error diversity across stages. B captures part of it; a fused/learned router could go further.

## Reproduce
```bash
cd ~/modelcascade
python main.py  --cfg cfgs/imagenette.yaml   # train/load cascade+probe, strategy A
python bench.py --cfg cfgs/imagenette.yaml   # full A-F frontier
```
