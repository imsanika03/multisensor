# MESA: Multispectral Encoder with Spectral Adaptation

## Overview

MESA adapts a frozen DINOv2 ViT-B/14 backbone to multispectral satellite imagery without modifying any backbone weights. It introduces a lightweight spectral front-end that computes per-location spectral features from raw multispectral bands, then uses those features to gate a bottleneck adapter injected into the last 6 transformer blocks.

## Method

### 1. Spectral Front-End (MESASummarizer)

Given a multispectral image with B bands, each pixel's spectrum is first projected onto a set of K=16 physically-motivated knot wavelengths via a regularized inversion:

```
alpha = x_loc @ M^T        # [B, 256, K_knots]
```

where M is the Moore-Penrose-style pseudoinverse of the Gaussian spectral response function matrix A (band center wavelengths convolved with instrument FWHM + a smoothness prior via finite-difference regularization).

Two complementary features are computed per spatial location:

- **Spatial spectral encoding**: alpha reshaped to a spatial grid, processed by a depthwise-separable CNN → [B, 256, D_SPEC]
- **Normalized difference features**: N_ND=16 learnable spectral ratio pairs (analogous to NDVI but generalized) → [B, 256, D_SPEC]

These are fused and layer-normalized to produce `local_feat` [B, 256, D_SPEC=128] — one feature vector per patch location.

### 2. Spectral Register Tokens (SpectralRegInit)

K_REG=8 learnable spectral register tokens are initialized from a global scene summary (mean-pooled `local_feat` → MLP → 768-dim), giving the model a scene-level spectral context to carry through the adapted blocks.

### 3. Bidirectional Spectral Layers (BidirSpectralLayer × 6)

Applied after each of the last 6 DINOv2 blocks (blocks 6–11). Each layer performs two steps:

**Step 1 — Spec ← Patches** (spec tokens absorb DINOv2 context):
```
spec = spec + s_o( softmax(s_q(spec) @ s_k(patches)^T / sqrt(d)) @ s_v(patches) )
```
s_o is zero-initialized so the spec tokens start as identity and learn gradually.

**Step 2 — Patches ← Local spectral gate** (key design choice):
```
g = sigmoid( p_gate(local_feat) )        # per-location gate from raw spectral features
h = GELU( p_down(patches) )
patches = patches + p_up(h * g)          # zero-init p_up
```

The gate is conditioned on `local_feat` — the static per-location spectral features from the MESASummarizer — rather than on the dynamically aggregated spec tokens. This preserves spatial specificity: the spectral correction at patch position (i,j) is conditioned only on the spectrum observed at (i,j), not a global average.

### 4. Classification Head

The final DINOv2 LayerNorm is applied, then CLS token and mean-pooled patch tokens are concatenated → MLP head:
```
head( cat([cls, mean(patches)]) )        # [B, 1536] -> [B, 512] -> GELU -> Dropout(0.2) -> [B, NC]
```

## Parameter Count

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| DINOv2 ViT-B/14 backbone | 86.58M | No (frozen) |
| MESASummarizer | 0.106M | Yes |
| SpectralRegInit | 0.205M | Yes |
| BidirSpectralLayers × 6 | 2.468M | Yes |
| Classification head | 0.797M | Yes |
| **Total extra (trainable)** | **3.576M** | **Yes** |

MESA adds **3.6M trainable parameters** on top of a frozen 86.6M DINOv2 backbone — **90.16M total**, 4.1% parameter overhead.

## Training

- Optimizer: AdamW (lr=5e-4, weight_decay=5e-4)
- Schedule: cosine decay with 5-epoch linear warmup, max 50 epochs + early stopping (patience=3 eval intervals)
- Batch size: 256
- Loss: ASL (gamma_neg=4, gamma_pos=0, margin=0.05) for multi-label; CrossEntropy for single-label
- Augmentation: random horizontal and vertical flips
- Dropout: 0.3 in classification head
- Mixed precision: bfloat16

## Results

### BigEarthNet-S2 (19-class multi-label, official 10% train split, macro-mAP)

| Model | mAP | Seeds |
|-------|-----|-------|
| MESA-v1 (global gate) | 68.20% | 2 |
| MESA-v4 (bidir, global spec_ctx gate) | 69.78% | 1 |
| MESA-v2 (per-loc gate, no bidir) | 70.65% / 71.12% | 2 |
| **MESA (bidir + local_feat gate)** | **75.19%** | 1 |

### EuroSAT (10-class single-label, top-1 accuracy)

| Model | Accuracy | Seeds |
|-------|----------|-------|
| DINOv2 LP baseline | 96.78% | 1 |
| **MESA (bidir + local_feat gate)** | **99.13% ± 0.07%** | 3 (99.06 / 99.19 / 99.15) |
