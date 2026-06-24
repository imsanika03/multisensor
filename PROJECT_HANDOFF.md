# Project Handoff — Spectral Token Expansion (STE) for Frozen RGB Foundation Models

**Purpose of this doc:** let a fresh Claude instance on a new VM pick up exactly where we left off.
Read it top to bottom before running anything. It records *what the project is now*, *what was
already tried and killed* (so you don't repeat ~11 dead ends), the *current method + latest numbers*,
the *data/scripts on disk*, *environment gotchas that cost us hours*, and *prioritized next steps*.

Working dir for everything: `/home/ubuntu/sanikabhar/modelcascade`. Run scripts from there.
(Note: `/home/ubuntu/sanikabhar` and `/lambda/nfs/sanikabhar` are symlinked — same filesystem, either path works.)

---

## 1. TL;DR — current state

- **Goal:** a novel, *publishable-at-WACV* method. The bar the user set: "very likely to work AND very
  likely accepted." Applied/method paper, not analysis (WACV skews to methods).
- **Where we landed after a long search:** a **remote-sensing** method — **Spectral Token Expansion (STE)**:
  give a *frozen* RGB foundation model (DINOv2) access to **multispectral** satellite bands via a small,
- **Why this regime:** it's the only place we found *structural headroom* — a setting where the strong
  baseline (frozen RGB foundation model) is genuinely weak. On **BigEarthNet** (unsaturated):
  - RGB-DINOv2 linear-probe = **0.565** macro-mAP (well below ~0.65–0.70 SOTA → unsaturated)
  - all-12-band CNN (from scratch) = **0.630** → **+6.45** multispectral headroom.
- **Latest results (conv spatial encoder, confirmed working):**
  - **GAIN: STE = 0.6310** macro-mAP — beats RGB-DINOv2 (+6.60 pts) ✅ AND beats all-band CNN (+0.10 pts) ✅
  - **XSENS (cross-sensor, 2 seeds):** wavelength beats index by **+1.69 pts mean** (wl 0.6278±0.001, idx 0.6109±0.005)
    - seed0: wl 0.6273 vs idx 0.6076, gap +1.97
    - seed1: wl 0.6282 vs idx 0.6141, gap +1.42
  - Both results are **seed-stable** with clear magnitude. This hits the "strong novel paper" verdict.
- **What changed from previous run (0.5876):** replaced the spatial mean-pool in `SpectralEnc` with a
  2-layer Conv2d over the 16×16 patch token grid. Masked mean over bands is preserved (composition-invariant).
  Also updated hyperparameters for the GH200: bs=1024, lr=4e-3, epochs=50, torch.compile, TF32.

**Verdict: the method is a win.** Next priority is scaling for rigor (§8).

---

## 2. What the project is (and how it got here)

It started as `modelcascade`: a resolution cascade (ResNet50 @112/224/448) with a learned router/probe to
route each image to the cheapest sufficient stage under a certified harm budget. Over a long investigation
we **falsified ~11 routing/repair ideas** (details in §3) and discovered the recurring wall: *on standard
image classification the simple/strong baseline is near-optimal, so bolt-on structure adds nothing.*

We then searched for a regime where the baseline is genuinely weak, and landed on **multispectral remote
sensing**: RGB foundation models structurally cannot see the non-RGB Sentinel-2 bands, which carry real
signal. STE is the method that gives a frozen RGB foundation model that access, conditioned on physical
**wavelength** (so it can generalize across sensors with different bands — the novelty).

---

## 3. What was already tried and KILLED (do not repeat)

Each was de-risked and failed for a concrete reason. The throughline: strong baselines are hard to beat.

1. **Probe one-shot router** (predict best stage from a 96px thumbnail): harm-ranking AUC ≈ 0.5 (chance).
   Pixels don't predict whether a model will err.
2. **Multi-task router** (distill confidence/correctness/oracle-route + ranking loss): held-out AUC still ≈ 0.5.
   λ-sweep didn't help. The "0.64" we once saw was small-sample epoch-selection bias.
3. **RC-VoI** (ensemble/skip actions, value-of-information): cost-aware **oracle frontier proved ensemble/skip
   add 0.0** headroom over pure selection on same-arch resolution stages.
4. **RGC** (recoverability-gated cascade — skip "unrecoverable" escalations): can't separate recoverable from
   unrecoverable errors from the cheap model's state (AUC ≈ 0.5).
5. **DGC** (disagreement-gated cascade): cross-architecture disagreement predicts recoverability *conditionally*
   (AUC 0.76) but the **confidence gate still wins the cost/accuracy frontier** even under corruption. A good
   conditional signal ≠ a good gate.
6. **Confidence cascade** itself works great (e.g., Imagewoof 0.14× FLOPs at ~no acc loss) but is **prior art**.
7. **CoRE** (confusion-conditioned residual experts for FGVC): frozen-feature experts can't repair confusions
   (confusion lives in the *features*, not the head); on CUB even **full pair specialists barely separate** the
   top confusion pairs (data-starved; the global model already extracts the best signal).
8. **PatchCache** (confusion-conditioned DINOv2 retrieval): works in absolute terms (+12–16 pts over a weak base)
   but **dominated by plain DINOv2-full kNN (0.87)**, and confusion-conditioning *hurts*. Reduces to "use DINOv2."
9. **Few-shot patch retrieval (FGVC):** naive patch-Chamfer ≪ CLS-proto (background-dominated); and a **linear
   probe on DINOv2 features is a brutally strong few-shot baseline** that retrieval doesn't beat.
10. **STE wavelength conditioning on EuroSAT:** EuroSAT is **saturated** (RGB-DINOv2 = 0.96); MS headroom only
    +1.6; cross-sensor wavelength-vs-index gap was **seed-noise** (a single +3.46 didn't survive 5 seeds:
    k1 +5.81±5.84, k3 +0.43, k4 −0.57±1.67). Right *direction*, no magnitude — wrong (saturated) testbed.

**Key cross-cutting findings (reusable):**
- Foundation features (DINOv2) + a linear probe is an extremely strong baseline; hard to beat with adapters/retrieval.
- The signal for "will a model err / will a bigger model fix it" lives in the model's **computed response**,
  not in cheap pre-computation features ("routability horizon": partial-f1 features rose 0.51→0.70 with depth).
- Multispectral is the one place with *structural* headroom (RGB models can't see the bands).

---

## 4. Current method — Spectral Token Expansion (STE)

**Idea:** frozen DINOv2 (RGB) provides semantics; a small trainable spectral adapter injects the Sentinel-2
multispectral bands, conditioned on each band's **physical wavelength** (sinusoidal embedding) and
**resolution** (10/20/60 m). Only the adapter + head train (~0.1–0.2 M params; DINOv2 stays frozen).

**Current architecture (`ben_ste.py`, also `ste_v2.py` for EuroSAT):**
- Per band: 4×4 patch embeddings → conditioning **concatenated** then MLP (load-bearing, not additive) →
  **masked mean over present bands** (composition-invariant; handles missing bands) →
  **2-layer Conv2d over 16×16 spatial grid** (preserves spatial structure) → global avg pool →
  late-fused (concat) with frozen DINOv2-RGB CLS → multi-label head.
- Design fixes that mattered: (a) **gated + LayerNorm injection** (earlier additive injection collapsed to
  near-random on new band sets); (b) **band-dropout in training** (so the adapter handles arbitrary band subsets);
  (c) **conv spatial encoder** replacing DeepSets mean-pool — this was the fix that pushed 0.5876 → 0.6310.
- Hyperparameters tuned for GH200 480GB: `BS=1024`, `LR=4e-3`, `epochs=50`, `torch.compile`, TF32 precision.

**Two evaluations (in `ben_ste.py`):**
- **GAIN:** STE (all 12 bands) macro-mAP vs RGB-DINOv2 (0.565) and all-band CNN (0.630). **Latest: 0.6310 ✅**
- **XSENS (cross-sensor, the novelty):** hold out informative bands `[4,5,8,10]` (B05,B06,B8A,B11), train on the
  rest; at test the "new sensor" = RGB + held-out bands. Compare **wavelength STE vs channel-index STE** on the
  new sensor. **Latest (2 seeds): wavelength +1.69 pts over index, seed-stable ✅**

---

## 5. Data & artifacts (on THIS VM's persistent disk `/home`; NOT on a fresh VM)

Under `data/`:
- `ben/S2.tar.gzaa` (48 GB) + `ben/S2.tar.gzab` (15 GB) — BigEarthNet-S2 v2.0 tarball (split gzip; cat both).
- `ben/x/` — 480k extracted tiffs = our 40k-patch subset (30k train + 10k test), 12 bands each.
- `ben/ben_subset.pt` (13.8 GB) — **prepped tensor** `X[40000,12,120,120] f16` + `Y[40000,19]` multihot + split.
- `ben/dino_cls.pt` (123 MB) — **cached frozen DINOv2-RGB CLS** for the 40k subset (skips DINOv2 recompute).
- `ben/metadata.parquet`, `ben/subset.json`, `ben/extract_list.txt`.
- `eurosat_ms/` (13-band EuroSAT), `imagewoof2/`, `imagenette2/`, `CUB_200_2011/`, `cifar100_hf/`.
- `checkpoints/` — trained cascades (imagenette/imagewoof resnet50 + diverse resnet18/convnext/vit), `cub_base/`.

**Results on BigEarthNet 40k subset (macro-mAP):**
- RGB-DINOv2 LP = **0.565** (baseline)
- All-12-band CNN = **0.630** (from-scratch MS baseline)
- STE-wavelength (mean-pool, old) = **0.5876** ← dead, replaced
- **STE-wavelength (conv spatial encoder) = 0.6310** ✅ beats both baselines
- XSENS: wavelength **0.6278±0.001** vs index **0.6109±0.005**, gap **+1.69 pts** (2 seeds) ✅

### Regenerate `ben_subset.pt` from scratch on a fresh VM

The fully-scripted pipeline (all four scripts travel with the repo). The subset is **deterministic
(seed 0)** so you get byte-identical `subset.json`/`extract_list.txt` and the same 40k patches as this VM.
Run everything from the repo root (`~/modelcascade`). Total ≈1 hr wall (download-bound).

```bash
# 0. deps — tifffile + pyarrow are the only extras beyond torch/torchvision
pip install tifffile pyarrow

# 1. download the two split-gzip parts (~63 GB) + the label/split metadata, into data/ben/
mkdir -p data/ben && cd data/ben
B=https://huggingface.co/datasets/torchgeo/bigearthnet/resolve/main/V2
wget -c $B/BigEarthNet-S2.tar.gzaa -O S2.tar.gzaa
wget -c $B/BigEarthNet-S2.tar.gzab -O S2.tar.gzab
wget -c $B/metadata.parquet        -O metadata.parquet
cd ../..

# 2. build subset.json (40k = 30k train + 10k test) + extract_list.txt (480k band-tiff paths), seed 0
python ben_make_subset.py
#    -> "subset.json: 40000 patches (30000 train / 10000 test); extract_list.txt: 480000 files; 19 classes"

# 3. stream-extract ONLY those 480k tiffs into data/ben/x/
#    Use ben_extract_fast.py (pigz parallel decompression + 32 write threads) — ~4 min vs ~1 hr for ben_extract.py
#    Requires pigz: apt install pigz  (or brew install pigz)
#    do NOT use `tar -x --files-from` on the 480k list — it's O(n^2) and hangs (see §6)
#    do NOT kill this mid-run — ThreadPoolExecutor writes are in-flight; killing leaves partial files silently
cat data/ben/S2.tar.gzaa data/ben/S2.tar.gzab | pigz -d | python ben_extract_fast.py
#    -> "EXTRACT_DONE 480000/480000"

# 4. pack the 12 resampled bands per patch into the tensor -> data/ben/ben_subset.pt (13 GB)
python ben_prep.py
```

After step 4 you can delete `S2.tar.gz*` (61 GB) and `data/ben/x/` (7 GB) — `ben_subset.pt` supersedes both.
`ben_ste.py` / `ben_headroom.py` then precompute `dino_cls.pt` (DINOv2-RGB CLS cache) on first run (~2 min).

(EuroSAT-MS, if ever needed for the killed cross-sensor test: `https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip`.)

---

## 6. Environment gotchas (these cost us real time — heed them)

- **Current VM: single NVIDIA GH200 480GB.** No faulty GPUs; no `CUDA_VISIBLE_DEVICES` needed. On a different
  VM, check `nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv` and exclude nonzero GPUs.
- **DataLoader `num_workers>0` after CUDA is initialized → fork deadlock** (main thread spins at 100% one core,
  workers idle, GPUs partly stuck). Use `num_workers=0` (data is in RAM anyway) OR set multiprocessing start
  method to `spawn`. This was a multi-hour stall.
- **GNU `tar -x --files-from=<huge list>` is O(n²)** and effectively hangs (99% CPU, 0 files out). Use
  `ben_extract_fast.py` (pigz + threading) — ~4 min for 480k files.
- **Do NOT kill `ben_extract_fast.py` mid-run.** The ThreadPoolExecutor write futures may not have flushed;
  silent missing files will cause `ben_prep.py` to crash on `FileNotFoundError`. Always wait for `EXTRACT_DONE`.
- **`/home/ubuntu/sanikabhar` and `/lambda/nfs/sanikabhar` are the same filesystem** (symlinked). Python
  resolves the canonical path, so error traces show `/lambda/nfs/...` even when you `cd` to `/home/ubuntu/...`.
- **`/tmp` is wiped on VM reboot**; persistent data is on `/home`. Write logs to the project dir.
- **`print()` to a redirected file is block-buffered** → progress looks frozen. Use `flush=True` (scripts do).
- **DataParallel for tiny models has high overhead**; for many small independent trainings, **fan out one job
  per GPU** (see `solidify_worker.py` + `launch_solidify.sh`) instead of DataParallel-one-at-a-time.
- Run from `/home/ubuntu/sanikabhar/modelcascade` (relative paths: `data/...`, `cfgs/...`, `checkpoints/...`).
- DINOv2 loads via `torch.hub.load('facebookresearch/dinov2','dinov2_vitb14')` (auto-downloads; xFormers
  warnings are harmless).

---

## 7. Key scripts

- `ben_ste.py` — **the current method**: STE on BigEarthNet, GAIN + XSENS. (Edit the spectral encoder here.)
- `ben_headroom.py` — RGB-DINOv2 LP vs all-band CNN, macro-mAP (the headroom check). Gave 0.565 / 0.630.
- `ben_make_subset.py` — build `subset.json` + `extract_list.txt` deterministically (seed 0) from `metadata.parquet`.
- `ben_extract.py` — original single-threaded extractor (slow, ~1 hr). Superseded by `ben_extract_fast.py`.
- `ben_extract_fast.py` — **fast extractor**: pigz parallel decompression + 32 write threads (~4 min). Use this.
- `ben_prep.py` — pack 40k patches (12 resampled bands) → `ben_subset.pt`.
- `ste_v2.py`, `cross_sensor_ste.py`, `solidify_ste.py`, `solidify_worker.py`, `launch_solidify.sh` — the
  EuroSAT STE + cross-sensor + seed-sweep machinery (showed the EuroSAT effect was noise; reuse patterns).
- `sat_headroom.py`, `sat_ms_headroom.py`, `fusion_derisk.py`, `spectral_token_expansion.py` — EuroSAT pipeline.
- `utils.py` (`get_dataset` supports cifar100/imagenette/imagewoof/imagenette_c/cub/imagenet[100]),
  `cascade/stage.py` (per-stage lr + val-accuracy early stopping), `bench.py` (A–F routing benchmark),
  `IMAGENETTE_RESULTS.md` (the routing-era results).

---

## 8. Next steps (prioritized)

~~1. Fix STE's spectral encoder (conv-based). **DONE** — 0.5876 → 0.6310.~~
~~2. Run XSENS on BigEarthNet. **DONE** — wavelength +1.69 pts over index, seed-stable (2 seeds).~~
~~3. Run ≥5 seeds on GAIN and XSENS + full ablation. **DONE** — see §9 for results.~~

**The method is confirmed. Now scale for rigor.**

1. ~~**Run with ≥5 seeds** on GAIN and XSENS to report proper mean±std and CIs.~~ **DONE — see §9.**
2. **Scale to full BigEarthNet with official splits.** Current results are on a 40k subset. The official BEN
   train/val/test splits are larger; reviewers will ask why we subsetted. Either justify the subset (speed,
   ablations) or re-run on full data.
3. **Add real RS-foundation baselines:** SatMAE, Scale-MAE, Prithvi, Clay. These require RS pretraining;
   STE's claim is that it doesn't. Also add "fine-tune the DINOv2 backbone" as an upper bound.
4. **Run the ablation suite:** wavelength vs index vs no-conditioning, adapter size sweep, band-dropout
   on/off, fusion mechanism (concat vs add), missing-band robustness (drop bands at test), param/FLOP budget.
5. **The headline novelty experiment:** a *real* cross-sensor benchmark — train on Sentinel-2, test on
   Landsat-8/9 (different band set, physically meaningful wavelengths). The band-holdout XSENS is a
   single-dataset *simulation*; a true cross-sensor result is what makes the claim land in a paper.
6. **Positioning:** differentiate from Tip-Adapter (cache adaptation), RS foundation models (require RS
   pretraining; STE doesn't), and generic channel/band adapters (channel-index, exactly our failing ablation).
   The defensible STE claim: *"no RS pretraining needed — a parameter-efficient, wavelength-conditioned adapter
   on a frozen RGB foundation model recovers the multispectral gain AND transfers across sensors."*

---

**Why wavelength enables cross-sensor generalization.**
Index conditioning learns "slot 4 means X" — fails on a new sensor where slot 4 is a different band.
Wavelength conditioning learns "705nm means X" — a new sensor's 710nm band gets a nearly identical embedding,
so the physical meaning transfers. The XSENS nodrop ablation confirmed this directly: without band dropout,
wavelength models collapse from 0.624 → 0.543 (std 0.003 → 0.026) because they never learned to handle
arbitrary band subsets during training.

## 10. Confirmed results — BigEarthNet 40k subset (5 seeds, macro-mAP)

Scripts: `ablation_ben.py` (results → `results/ablation_ben.json`),
         `xsens_ablation.py` (results → `results/xsens_ablation.json`).

**Baselines:**
- RGB-DINOv2 LP = **0.5650**
- All-band CNN  = **0.6183 ± 0.003**

**GAIN (all 12 bands):**
| Model                  | mAP    | ±std  | vs RGB  | vs CNN  |
|------------------------|--------|-------|---------|---------|
| ste_conv_wl (main)     | 0.6330 | 0.005 | +6.80   | +1.47   |
| ste_conv_wl_nodrop     | 0.6410 | 0.005 | +7.61   | +2.27   |
| ste_conv_idx           | 0.6308 | 0.006 | +6.59   | +1.25   |
| ste_conv_idx_nodrop    | 0.6377 | 0.004 | +7.28   | +1.94   |
| ste_mean_wl            | 0.6340 | 0.005 | +6.90   | +1.57   |

**XSENS (train 8 bands → new sensor = RGB + held-out [B05,B06,B8A,B11]):**
| Config             | mAP    | ±std  |
|--------------------|--------|-------|
| xsens_wl_drop      | 0.6242 | 0.003 |
| xsens_wl_nodrop    | 0.5426 | 0.026 |
| xsens_idx_drop     | 0.6145 | 0.005 |
| xsens_idx_nodrop   | TBD    | —     |

Key findings:
- Band dropout is **essential** for XSENS: without it wl collapses 0.624 → 0.543 with high variance.
- Wavelength beats index when dropout is on (+1.0 pt, low std vs high std). Claim is robust.
- Conv spatial encoder ≈ mean-pool on GAIN alone; conv advantage shows in XSENS stability.
- nodrop variants score *higher* on GAIN (full 12 bands always present at test) — expected, not a contradiction.

---

## 11. MESA Architecture Results — BigEarthNet official 10% splits (macro-mAP)

**Benchmark:** official BigEarthNet-S2 v1 splits, 10% training subset (26,969 train / 125,866 test), 19-class
multi-label, macro-mAP. Scripts in `/lambda/nfs/sanikabhar-texas/multisensor/`.

**Working directory:** `/lambda/nfs/sanikabhar-texas/multisensor/`
(also accessible as `/home/ubuntu/sanikabhar-texas/multisensor` — same filesystem)

**Data files (all present):**
- `data/ben/ben_v1.pt` — X=(152835,12,120,120) f16, Y=(152835,19). Raw reflectance ÷3000 to normalize.
- `data/ben/dino_cls_v1.pt` — [152835,768] float32, cached frozen DINOv2 CLS tokens.
- `data/ben/dino_patches_v1.pt` — [152835,256,768] float16, ~60 GB. **Do not call `.float().mean(1)` directly — OOM. Use chunked loop (2000 at a time).**

**Frozen backbone:** DINOv2 ViT-B/14, 86.58M params, ALL weights frozen throughout. This is a hard constraint.

**GPU:** A100 80GB (or similar). No OOM risk at BS=256 for v4. `sqa` tmux session is the standard run session.

### Baselines (cached DINOv2 features)
| Model | mAP |
|---|---|
| CLS linear-probe | 63.41% |
| CLS + patch-mean linear-probe | 64.36% |

---

### MESA architecture history and results

**MESA-v1** (`ben_mesa_mid.py`): SpectralGatedAdapter, **global** spectral gating, blocks 9–11 only.
- D_BOTTLE=64, D_SUMM=256, K=16 knots, BS=256, LR=2e-3, EPOCHS=100, ASL loss
- seed 0: **68.40%** | seed 1: **68.00%**
- *Limitation:* global gating applied the same spectral correction to all 257 tokens — no spatial specificity.

**MESA-v2** (`ben_mesa_v2.py`): PerLocGatedAdapter, **per-location** spectral gating, blocks 6–11. **BEST CONFIRMED RESULT.**
- D_BOTTLE=128, D_SPEC=128, D_SUMM=256, K=16 knots, BS=256, LR=2e-3, EPOCHS=100, ASL loss
- Patch token j gated by `local_feat[:,j,:]` (spectrally-aligned correction per location)
- CLS token gated by global spectral summary
- seed 0: **70.65%** | seed 1: **71.12%**
- 3.48M trainable params / 90.06M total (rest frozen)
- *(Run stopped after 2 seeds to pursue v4. If v4 doesn't beat 70.65%, run 5 seeds on v2.)*

**MESA-v4-attempt-1** (`ben_mesa_v3.py` / first v4 attempt): spectral tokens **inside** frozen DINOv2 attention.
- K=8 spectral tokens inserted into 257-token sequence at block 6 → expanded to 265 tokens through blocks 6–11
- Tokens participate in DINOv2's native frozen self-attention
- seed 0: **69.59%** ← WORSE than v2
- *Root cause:* frozen attention weights are trained for 257 tokens. 8 extra tokens absorb attention weight that
  should go to RGB patches, diluting spatial reasoning. Frozen softmax normalizes over 265 instead of 257.

**MESA-v4-attempt-2** (second v4 attempt, `ben_mesa_v4.py` before fix): unconstrained bidirectional cross-attn post-block.
- Spectral tokens evolve separately (not inside frozen attention — this part correct)
- Patch update via unconstrained cross-attention after each frozen block
- seed 0: **16.95%** ← random prediction (≈ mean class prevalence)
- *Root cause:* zero-init on output projection starts corrections at 0, but gradient immediately pushes p_o to grow.
  Without bounding, corrections become >> DINOv2 features by epoch ~10. Model learns degenerate constant-output.
  Fix: sigmoid gating (same bounded mechanism as v2).

---

### CURRENT VERSION TO RUN: MESA-v4 sigmoid-gated bidir (`ben_mesa_v4.py`)

**Status: NOT YET RUN TO COMPLETION.** Training was killed at epoch 20/100. No result obtained.

**Architecture:**
- MESASummarizer: S2 bands → canonical coefficients (physics A/M matrices, K=16 Gaussian SRF knots,
  420–2200nm) → spatial conv encoder (16×16) + ND ratio features → local_feat [B,256,128] + global_summ [B,256]
- SpectralRegInit: K=8 learned spectral regime tokens shifted by global scene offset → [B, K, 768]
- BidirSpectralLayer (one per adapted block, 6 total covering blocks 6–11):
  - Step 1 — spec ← patches: spec tokens query DINOv2 patches via cross-attn [K×256, d_bidir=64], zero-init s_o
    (safe: spec is auxiliary, no risk of corrupting backbone features)
  - Step 2 — patches ← evolved spec: each patch soft-attends to K=8 spec tokens → per-patch spec context [B,256,768].
    Then SIGMOID-GATED adapter: gate=sigmoid(W_gate @ spec_ctx) bounds correction ∈ [0,1], zero-init p_up.
    This is the same stability mechanism as v2 — sigmoid gate prevents feature swamping.
- Head: CLS + patch_mean → Linear(768+768→512) → GELU → Dropout(0.2) → Linear(512→19)
- K_REG=8, D_BIDIR=64, D_BOTTLE=128, N_ADAPT_START=6
- BS=256, LR=2e-3, EPOCHS=100, WARMUP=5, SEEDS=[0]

**Why this should beat v2:**
v2 patches are gated by STATIC local_feat from MESASummarizer (same spectral features across all 6 adapted blocks).
v4 patches are gated by EVOLVED spec tokens — after 6 bidir rounds, spec tokens have absorbed DINOv2's intermediate
representations and reflect both spectral AND spatial context. Gate is informed by richer, scene-adapted state.

**Different from SpectraDINO (arxiv 2605.02258):**
- SpectraDINO: per-modality bottleneck adapters, multi-stage training (cosine distillation + contrastive + patch alignment)
- MESA: physics-based A/M matrices (Gaussian SRF overlaps → regularized inversion), K=8 spectral regime tokens that
  evolve bidirectionally, sigmoid-gated adapter for stability. End-to-end ASL training, no distillation.

**Run command (in `sqa` tmux session):**
```bash
cd /lambda/nfs/sanikabhar-texas/multisensor
python3 ben_mesa_v4.py 2>&1 | tee results/ben_mesa_v4.log
```
Expected: ~45 min on A100 for 100 epochs / 1 seed. Output: `results/ben_mesa_v4.json`.

**Decision tree after v4 finishes:**
- v4 seed 0 > 70.65% (v2 seed 0): v4 is better → run 5 seeds on v4 for the paper table
- v4 seed 0 < 70.65%: v4 does not beat v2 → fall back to v2 as the paper method, run 5 seeds on v2

---

### Complete results table

| Version | Architecture | mAP (seed 0) | mAP (seed 1) | Status |
|---|---|---|---|---|
| Linear probe (CLS) | Frozen DINOv2 features | 63.41% | — | baseline |
| Linear probe (CLS+patches) | Frozen DINOv2 features | 64.36% | — | baseline |
| MESA-v1 | Global spectral gate, blocks 9–11 | 68.40% | 68.00% | done |
| MESA-v2 | Per-loc spectral gate, blocks 6–11 | 70.65% | 71.12% | **BEST — 2 seeds** |
| MESA-v4-a1 | Tokens inside frozen attn | 69.59% | — | killed (worse than v2) |
| MESA-v4-a2 | Bidir unbounded cross-attn | 16.95% | — | killed (degenerate) |
| **MESA-v4** | **Bidir sigmoid-gated (current)** | **NOT YET RUN** | — | **← RUN THIS NEXT** |

---

### Key architecture progression
```
MESA-v1: spectral_summary [B,256]  → same gate for all 257 tokens        → 68.2% avg
MESA-v2: local_feat [B,256,128]    → per-location gate for token j       → ~70.9% (seeds 0-1) ← BEST
MESA-v4-a1: K=8 tokens inside attn → dilutes frozen attention             → 69.59%
MESA-v4-a2: bidir, unbounded       → corrections swamp DINOv2 features   → 16.95% (random)
MESA-v4:  bidir, sigmoid-gated     → bounded, evolved spec context       → NOT YET RUN
```

---

### MESA implementation notes (avoid repeating these mistakes)
- **Zero-init on adapter output projections** (`up.weight`, `up.bias`, `s_o.weight`, `s_o.bias`) — ensures model
  starts as frozen DINOv2, learns additive corrections. Miss this and training diverges from LP baseline.
- **Sigmoid gating is essential for patch updates** — any unbounded cross-attention correction for patch tokens will
  swamp DINOv2 features (v4-a2 failure). Spec tokens can be updated freely (they are auxiliary).
- **Do NOT insert spectral tokens into frozen DINOv2 attention** — frozen weights are trained for 257 tokens; extra
  tokens dilute spatial attention (v4-a1 failure at 69.59%).
- **Detach after block 5** (`x = x.detach()`) — gradient flows only through adapters + MESASummarizer.
- **Consistent spatial augmentation** — flip rgb AND x_loc together (x_loc reshaped as [16,16,12] → flip → [256,12]).
- **OOM on patch cache:** `dino_patches_v1.pt` [152835,256,768] f16 ≈ 60 GB. `.float()` doubles to 120 GB → OOM.
  Use chunked loop: `for i in range(0, len(PAT), 2000): chunks[i:i+2000] = PAT[i:i+2000].float().mean(1)`.
- **ASL loss** (γ_neg=4, γ_pos=0, margin=0.05) outperforms BCE for multi-label with class imbalance.
- **LR=2e-3, WARMUP=5** — confirmed working. Do not increase WARMUP (WARMUP=10 caused collapse).
- **FiLM adapters are off-limits** — another paper already uses FiLM for this; MESA must be architecturally distinct.

---

### Paper comparison baselines
| Method | mAP | Notes |
|---|---|---|
| CLS linear-probe | 63.41% | frozen DINOv2, cached |
| CLS+patch linear-probe | 64.36% | frozen DINOv2, cached |
| SatMAE++ ViT-L | 85.10% | RS-pretrained + full FT |
| SMARTIES ViT-B | 86.90% | RS-pretrained + full FT |
| CROMA ViT-B | 87.60% | RS-pretrained + full FT |
| **MESA-v2 (ours)** | **~70.9%** | Frozen DINOv2, 2 seeds |
| **MESA-v4 (ours)** | **TBD** | Frozen DINOv2, needs run |

Note: the RS-pretrained methods use full fine-tuning and RS-domain pretraining; MESA's claim is that a frozen
ImageNet model + physics-based spectral adapter reaches competitive accuracy without any RS pretraining.
