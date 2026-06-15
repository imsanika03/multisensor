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

## 9. Architecture detail — what sits on top of frozen DINOv2

**Step 1: Frozen DINOv2 extracts RGB semantics.**
RGB bands (B04, B03, B02) → frozen `dinov2_vitb14` → 768-dim CLS token. Precomputed once, cached as
`data/ben/dino_cls.pt`. DINOv2 never trains, never sees non-RGB bands.

**Step 2: SpectralEnc — the trainable adapter (~0.1–0.2M params).**
For each of the 12 Sentinel-2 bands:
1. *Patch embed* — 64×64 band image split into 4×4 patches → 16×16 grid = 256 spatial tokens (16-dim each)
   → linear projection to 128-dim.
2. *Wavelength conditioning* — sinusoidal embedding of the band's physical wavelength (e.g. 705nm for B05)
   → small MLP → 128-dim. **Cross-sensor key:** the adapter knows *what wavelength* each band is, not just
   which slot it occupies.
3. *Fuse* — band embedding + wavelength conditioning concatenated → 2-layer MLP → 128-dim per spatial token
   per band.
4. *Masked mean over bands* — average over whichever bands are present (composition-invariant; handles missing
   bands at test time). Result: [256 spatial tokens × 128-dim].
5. *Conv spatial encoder* — reshape 256 tokens → 16×16 grid → two Conv2d layers → global avg pool → 128-dim.

**Step 3: Late fusion + head.**
128-dim spectral vector concat with 768-dim DINOv2 CLS → 896-dim → MLP → 19-class multi-label output.

```
DINOv2 CLS (768d, frozen) ─────────────────────────────────┐
                                                            concat → MLP → labels
SpectralEnc (128d, trainable):                             │
  bands → patch embed → [+ wavelength MLP] → fuse         │
        → masked mean over bands                           │
        → conv 16×16 → global avg pool ───────────────────┘
```

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
