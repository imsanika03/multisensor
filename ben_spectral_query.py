"""Spectral Query Attention (SQA) for BigEarthNet 10% benchmark.

Architecture:
  - Frozen DINOv2 ViT-B/14: CLS + 256 patch tokens (precomputed, cached)
  - Spectral indices computed from raw S2 bands (NDVI, NDWI, NDBI, NBR, SAVI, MSI, BSI)
  - MLP(indices) → query vector → attention over DINOv2 patch tokens
    (spectral signature says WHICH spatial regions to look at in the RGB feature map)
  - Logit-level spectral correction: small head on indices added to main logits
  - No feature modulation, no backbone modification

Backbone is completely frozen. RGB contribution fully preserved.
"""
import json, math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"

RGB_IDX = [3, 2, 1]   # B04, B03, B02 in our band ordering
# Band ordering: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
#                  0   1   2   3   4   5   6   7   8   9  10  11

BS      = 4096   # larger batch — model is small, GPU has 80GB
LR      = 4e-3   # linear scale with BS (2e-3 @ 2048 → 4e-3 @ 4096)
EPOCHS  = 200    # compensate for fewer steps/epoch at larger BS
SEEDS   = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Spectral indices (physics-informed, dimensionality-reduced spectral features)
# ---------------------------------------------------------------------------

def compute_indices(x):
    """x: [B, 12, H, W] raw reflectance — returns [B, 7] global spectral indices."""
    b02 = x[:, 1]   # blue
    b03 = x[:, 2]   # green
    b04 = x[:, 3]   # red
    b08 = x[:, 7]   # NIR
    b11 = x[:, 10]  # SWIR1
    b12 = x[:, 11]  # SWIR2

    eps = 1e-6

    ndvi = (b08 - b04) / (b08 + b04 + eps)                         # vegetation
    ndwi = (b03 - b08) / (b03 + b08 + eps)                         # water
    ndbi = (b11 - b08) / (b11 + b08 + eps)                         # built-up
    nbr  = (b08 - b12) / (b08 + b12 + eps)                         # burn ratio
    savi = 1.5 * (b08 - b04) / (b08 + b04 + 0.5 + eps)            # soil-adj veg
    msi  = b11 / (b08 + eps)                                        # moisture stress
    bsi  = ((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02) + eps)  # bare soil

    # spatial mean over H×W → [B, 7]
    idx = torch.stack([ndvi, ndwi, ndbi, nbr, savi, msi, bsi], dim=1)
    return idx.mean(dim=(-2, -1)).clamp(-5, 5)   # clamp MSI which can be large


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def macro_mAP(scores, targets):
    aps = []
    for c in range(scores.shape[1]):
        t = targets[:, c]
        if t.sum() == 0:
            continue
        tt = t[scores[:, c].argsort(descending=True)]
        prec = tt.cumsum(0) / torch.arange(1, len(tt) + 1, dtype=torch.float)
        aps.append((prec * tt).sum().item() / tt.sum().item())
    return sum(aps) / len(aps)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpectralQueryAttention(nn.Module):
    """Spectral indices → query → attention over DINOv2 patch tokens.

    The spectral signature determines WHICH spatial regions of the RGB
    feature map to attend to, rather than modifying the features themselves.
    """
    def __init__(self, nc, d=768):
        super().__init__()
        # 7 spectral indices → attention query in DINOv2 patch feature space
        self.query_mlp = nn.Sequential(
            nn.Linear(7, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, d),
        )
        # logit-level correction: spectral indices directly predict class adjustments
        self.spec_correction = nn.Sequential(
            nn.Linear(7, 64), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, nc),
        )
        # main head: CLS + spectrally-pooled patches
        self.head = nn.Sequential(
            nn.Linear(d + d, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, nc),
        )

    def forward(self, cls, patches, indices):
        # cls:     [B, 768]   — frozen DINOv2 CLS
        # patches: [B, 256, 768] — frozen DINOv2 patch tokens (float16 → cast inside)
        # indices: [B, 7]    — spectral indices (NDVI, NDWI, ...)

        pat = patches.float()                              # [B, 256, 768]

        # spectral-conditioned query
        q = self.query_mlp(indices)                        # [B, 768]

        # attention: which patch tokens does this spectral signature focus on?
        # [B, 256] = [B, 256, 768] @ [B, 768, 1] squeezed
        attn = torch.bmm(pat, q.unsqueeze(2)).squeeze(2)  # [B, 256]
        attn = torch.softmax(attn / (768 ** 0.5), dim=1)  # [B, 256]

        # spectrally-guided spatial pool
        z_pool = torch.bmm(attn.unsqueeze(1), pat).squeeze(1)  # [B, 768]

        # main logits from CLS + spectrally-pooled patches
        logits = self.head(torch.cat([cls, z_pool], dim=1))

        # logit-level spectral correction (additive)
        logits = logits + self.spec_correction(indices)

        return logits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BENDataset(Dataset):
    """X removed from DataLoader — spectral indices precomputed once at start."""
    def __init__(self, CLS, PAT, IDX, Y, idx):
        self.CLS, self.PAT = CLS, PAT
        self.IDX, self.Y = IDX, Y
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.CLS[j], self.PAT[j], self.IDX[j], self.Y[j]


# ---------------------------------------------------------------------------
# Train + eval
# ---------------------------------------------------------------------------

def train_eval(CLS, PAT, IDX, Y, tr, te, nc, seed):
    torch.manual_seed(seed)
    net   = SpectralQueryAttention(nc).to(DEV)
    opt   = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    trl = DataLoader(
        BENDataset(CLS, PAT, IDX, Y, tr),
        batch_size=BS, shuffle=True, num_workers=8, drop_last=True,
        pin_memory=True, persistent_workers=True,
    )
    for ep in range(EPOCHS):
        net.train()
        for c, p, idx, y in trl:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(
                    net(c.to(DEV), p.to(DEV), idx.to(DEV)), y.to(DEV)
                )
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"    epoch {ep+1}/{EPOCHS}", flush=True)

    net.eval()
    tel = DataLoader(
        BENDataset(CLS, PAT, IDX, Y, te),
        batch_size=BS * 2, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True,
    )
    S = []
    with torch.no_grad():
        for c, p, idx, y in tel:
            S.append(torch.sigmoid(net(c.to(DEV), p.to(DEV), idx.to(DEV))).cpu().float())
    return macro_mAP(torch.cat(S), Y[torch.tensor(te)])


def precompute_indices(X):
    """Compute 7 spectral indices for all patches. Eliminates X from training loop."""
    print("  precomputing spectral indices ...", flush=True)
    IDX = torch.zeros(len(X), 7)
    for i in range(0, len(X), 2000):
        x = F.interpolate(X[i:i+2000].float(), size=64, mode="bilinear", align_corners=False)
        IDX[i:i+2000] = compute_indices(x)
    print(f"  IDX {tuple(IDX.shape)}", flush=True)
    return IDX


def linprobe_indices(IDX, Y, tr, te):
    """LP baseline: 7 spectral indices only, no DINOv2."""
    head = nn.Linear(7, Y.shape[1]).to(DEV)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt, Yt = IDX[tr].to(DEV), Y[tr].to(DEV)
    for _ in range(2000):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(head(Xt), Yt).backward()
        opt.step()
    with torch.no_grad():
        return macro_mAP(torch.sigmoid(head(IDX[te].to(DEV))).cpu(), Y[te])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("high")

    data_path     = "data/ben/ben_v1.pt"
    cache_cls     = "data/ben/dino_cls_v1.pt"
    cache_patches = "data/ben/dino_patches_v1.pt"
    out_path      = "results/ben_spectral_query.json"

    print(f"loading {data_path} ...", flush=True)
    d = torch.load(data_path, weights_only=False)
    X, Y, split = d["X"], d["Y"], d["split"]
    nc = Y.shape[1]

    tr = [i for i, s in enumerate(split) if s == "train"]
    te = [i for i, s in enumerate(split) if s == "test"]
    print(f"  train={len(tr)}  test={len(te)}  classes={nc}", flush=True)

    print("loading DINOv2 features from cache ...", flush=True)
    CLS = torch.load(cache_cls, weights_only=False)
    PAT = torch.load(cache_patches, weights_only=False)

    IDX = precompute_indices(X)
    del X  # free 52GB — no longer needed in training loop

    results = {}

    # --- spectral indices LP baseline ---
    print("\n[LP] spectral indices only (7 features) ...", flush=True)
    lp_idx = linprobe_indices(IDX, Y, tr, te)
    results["spectral_indices_lp"] = {"mAP": lp_idx}
    print(f"  spectral indices LP = {lp_idx*100:.2f}%", flush=True)

    # --- SQA ---
    print(f"\n[SQA] spectral query attention, {len(SEEDS)} seeds ...", flush=True)
    scores = []
    for s in SEEDS:
        print(f"  seed {s} ...", flush=True)
        v = train_eval(CLS, PAT, IDX, Y, tr, te, nc, s)
        scores.append(v)
        print(f"    seed {s} mAP = {v*100:.2f}%", flush=True)

    t = torch.tensor(scores)
    results["sqa"] = {"seeds": scores, "mean": t.mean().item(), "std": t.std().item()}

    os.makedirs("results", exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n--- SUMMARY ---", flush=True)
    print(f"  DINOv2 CLS LP          : 63.41%  (from ben_standard.py)", flush=True)
    print(f"  Spectral indices LP    : {lp_idx*100:.2f}%", flush=True)
    print(f"  SQA (ours)             : {t.mean()*100:.2f} ± {t.std()*100:.2f}%", flush=True)
    print(f"\n  Paper baselines (RS-pretrained + FT):", flush=True)
    print(f"    SatMAE++ ViT-L : 85.10%", flush=True)
    print(f"    SMARTIES ViT-B : 86.90%", flush=True)
    print(f"    CROMA    ViT-B : 87.60%", flush=True)


if __name__ == "__main__":
    main()
