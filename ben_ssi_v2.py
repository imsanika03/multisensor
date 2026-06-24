"""Spatial Spectral Injection v2 (SSI-v2) for BigEarthNet 10% benchmark.

Improvements over SSI-v1 (66.07% seed 0):
  1. Learned band attention: soft-weighted pooling over 12 bands (vs hard mean)
  2. Resolution-aware conditioning: 10m/20m/60m encoded with nn.Embedding (fixes unused buffer bug)
  3. Spatial spectral indices: NDVI/NDWI/etc. at each patch location (not global)
  4. Post-injection spatial refiner: 1-layer Transformer on augmented tokens
  5. Consistent augmentation: flip both x_loc AND patch token grid together
  6. d=128 spectral dim (vs 64)

All output projections zero-initialized → starts as pure DINOv2, learns additive corrections.
"""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Sentinel-2 band metadata
WL      = torch.tensor([443., 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190])
RES_IDX = torch.tensor([2, 0, 0, 0, 1, 1, 1, 0, 1, 2, 1, 1])  # 0=10m 1=20m 2=60m

BS      = 2048
LR      = 2e-3
EPOCHS  = 150
WARMUP  = 10
SEEDS   = [0, 1, 2, 3, 4]
D_SPEC  = 128


# ---------------------------------------------------------------------------
# Metrics
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


def sinusoid(vals, d):
    pos = (vals / 100.0).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe  = torch.zeros(len(vals), d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# Physics-based spectral indices (computed per patch location, not globally)
# ---------------------------------------------------------------------------

def spatial_indices(x_loc):
    """
    x_loc: [B, 256, 12] normalized band values per patch location
    returns: [B, 256, 7] physics-based indices per location

    Using spatial (per-patch) indices rather than global means preserves
    the texture/structure that discriminates e.g. coniferous vs broad-leaved.
    """
    b02 = x_loc[..., 1];  b03 = x_loc[..., 2];  b04 = x_loc[..., 3]
    b08 = x_loc[..., 7];  b11 = x_loc[..., 10]; b12 = x_loc[..., 11]
    eps = 1e-6
    ndvi = (b08 - b04) / (b08 + b04 + eps)
    ndwi = (b03 - b08) / (b03 + b08 + eps)
    ndbi = (b11 - b08) / (b11 + b08 + eps)
    nbr  = (b08 - b12) / (b08 + b12 + eps)
    savi = 1.5 * (b08 - b04) / (b08 + b04 + 0.5 + eps)
    msi  = (b11 / (b08 + eps)).clamp(-5, 5)
    bsi  = ((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02) + eps)
    return torch.stack([ndvi, ndwi, ndbi, nbr, savi, msi, bsi], dim=-1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ImprovedSpectralAdapter(nn.Module):
    """Per-location spectral adapter:
      - wavelength + resolution conditioning
      - learned band attention (which of 12 bands matter here?)
      - physics indices per patch location combined with raw bands
    """
    def __init__(self, d=D_SPEC):
        super().__init__()
        self.register_buffer("wl_emb",  sinusoid(WL, d))  # [12, d] fixed
        self.register_buffer("res_idx", RES_IDX)           # [12] resolution group
        self.res_emb = nn.Embedding(3, d)                  # learned 10m/20m/60m embed

        # Band embedding: value [1] + wavelength [d] + resolution [d] → d
        self.band_proj = nn.Linear(1 + d + d, d)

        # Learned band attention: which bands matter at each spatial location?
        self.band_score = nn.Linear(d, 1)

        # Physics spectral indices → d
        self.idx_proj = nn.Sequential(nn.Linear(7, d), nn.GELU(), nn.Linear(d, d))

        # Fuse raw-band features + physics features
        self.combine = nn.Sequential(
            nn.Linear(d + d, d), nn.GELU(), nn.LayerNorm(d)
        )

        # Zero-init: injection starts at 0 so model begins as pure DINOv2
        self.up = nn.Linear(d, 768)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x_loc):
        # x_loc: [B, 256, 12] normalized band values
        B, N, C = x_loc.shape

        # Per-band embeddings with wavelength + resolution conditioning
        v   = x_loc.unsqueeze(-1)                                           # [B,256,12,1]
        wl  = self.wl_emb.view(1, 1, C, -1).expand(B, N, C, -1)           # [B,256,12,d]
        res = self.res_emb(self.res_idx).view(1, 1, C, -1).expand(B, N, C, -1)  # [B,256,12,d]
        feat = self.band_proj(torch.cat([v, wl, res], dim=-1))              # [B,256,12,d]

        # Learned band attention (soft-weighted pooling over 12 bands)
        band_w   = torch.softmax(self.band_score(feat).squeeze(-1), dim=-1) # [B,256,12]
        raw_spec = (band_w.unsqueeze(-1) * feat).sum(2)                     # [B,256,d]

        # Physics-based spatial spectral indices
        idx_feat = self.idx_proj(spatial_indices(x_loc))                    # [B,256,d]

        # Combine and project up to DINOv2 dim
        spec = self.combine(torch.cat([raw_spec, idx_feat], dim=-1))        # [B,256,d]
        return self.up(spec)                                                 # [B,256,768]


class SpatialRefiner(nn.Module):
    """1-layer Transformer on spectral-augmented patch tokens.

    Lets spatially adjacent tokens communicate AFTER spectral injection,
    so the model can learn spatial patterns of spectral signatures (e.g.,
    urban grid patterns in NDBI, farm field boundaries in NDVI gradient).

    Output projections zero-initialized so refiner starts as identity.
    """
    def __init__(self, dim=768, n_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim), nn.Dropout(0.1)
        )
        # Zero-init output projections → identity at initialization
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ff[2].weight)
        nn.init.zeros_(self.ff[2].bias)

    def forward(self, x):
        # x: [B, 256, 768]
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class SSIv2Model(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.adapter = ImprovedSpectralAdapter()
        self.refiner = SpatialRefiner()
        self.head = nn.Sequential(
            nn.Linear(768 + 768, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, nc),
        )

    def forward(self, cls, patches, x_loc):
        pat      = patches.float()           # [B, 256, 768]
        augmented = pat + self.adapter(x_loc)  # [B, 256, 768]  zero-init → pat at init
        refined   = self.refiner(augmented)    # [B, 256, 768]  identity at init
        z         = refined.mean(1)            # [B, 768]
        return self.head(torch.cat([cls, z], dim=1))


# ---------------------------------------------------------------------------
# Dataset — consistent geometric augmentation on both x_loc and patch grid
# ---------------------------------------------------------------------------

class BENDataset(Dataset):
    def __init__(self, CLS, PAT, X, Y, idx, train):
        self.CLS, self.PAT = CLS, PAT
        self.X, self.Y = X, Y
        self.idx = idx
        self.train = train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        # Resize all 12 S2 bands to 16×16 to match DINOv2 patch grid
        x16 = F.interpolate(self.X[j].float()[None], size=(16, 16),
                             mode="bilinear", align_corners=False)[0]  # [12, 16, 16]
        # Patch tokens as spatial grid for consistent augmentation
        pat = self.PAT[j].view(16, 16, 768)                           # [16, 16, 768]

        if self.train:
            # Horizontal flip — valid for top-down RS imagery
            if torch.rand(1).item() < 0.5:
                x16 = x16.flip(-1)   # flip columns of band maps
                pat = pat.flip(1)    # flip columns of patch grid
            # Vertical flip — also valid for aerial/satellite views
            if torch.rand(1).item() < 0.5:
                x16 = x16.flip(-2)   # flip rows of band maps
                pat = pat.flip(0)    # flip rows of patch grid

        x_loc   = (x16 / 3000.0).clamp(0, 1).permute(1, 2, 0).reshape(256, 12)
        pat_flat = pat.reshape(256, 768)
        return self.CLS[j], pat_flat, x_loc, self.Y[j]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def linprobe(Xtr, Ytr, Xte, Yte, steps=2000):
    head = nn.Linear(Xtr.shape[1], Ytr.shape[1]).to(DEV)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt, Yt = Xtr.to(DEV), Ytr.to(DEV)
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(head(Xt), Yt).backward()
        opt.step()
    with torch.no_grad():
        return macro_mAP(torch.sigmoid(head(Xte.to(DEV))).cpu(), Yte)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cosine_with_warmup(opt, warmup, total):
    def fn(ep):
        if ep < warmup:
            return ep / max(warmup, 1)
        p = (ep - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train_eval(CLS, PAT, X, Y, tr, te, nc, seed):
    torch.manual_seed(seed)
    net   = SSIv2Model(nc).to(DEV)
    opt   = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = cosine_with_warmup(opt, WARMUP, EPOCHS)

    trl = DataLoader(
        BENDataset(CLS, PAT, X, Y, tr, train=True),
        batch_size=BS, shuffle=True, num_workers=8,
        drop_last=True, pin_memory=True, persistent_workers=True,
    )
    for ep in range(EPOCHS):
        net.train()
        for c, p, x, y in trl:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(
                    net(c.to(DEV), p.to(DEV), x.to(DEV)), y.to(DEV)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % 15 == 0:
            print(f"    epoch {ep+1}/{EPOCHS}", flush=True)

    net.eval()
    tel = DataLoader(
        BENDataset(CLS, PAT, X, Y, te, train=False),
        batch_size=BS * 2, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True,
    )
    S = []
    with torch.no_grad():
        for c, p, x, y in tel:
            S.append(torch.sigmoid(net(c.to(DEV), p.to(DEV), x.to(DEV))).cpu().float())
    return macro_mAP(torch.cat(S), Y[torch.tensor(te)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("high")

    data_path     = "data/ben/ben_v1.pt"
    cache_cls     = "data/ben/dino_cls_v1.pt"
    cache_patches = "data/ben/dino_patches_v1.pt"
    out_path      = "results/ben_ssi_v2.json"

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

    results = {}

    # LP baselines (run once for reference)
    print("\n[LP] CLS-only ...", flush=True)
    lp_cls = linprobe(CLS[tr], Y[tr], CLS[te], Y[te])
    results["cls_lp"] = {"mAP": lp_cls}
    print(f"  CLS LP = {lp_cls*100:.2f}%", flush=True)

    print("\n[LP] CLS + patch-pool ...", flush=True)
    chunks = torch.zeros(len(PAT), 768)
    for i in range(0, len(PAT), 2000):
        chunks[i:i+2000] = PAT[i:i+2000].float().mean(1)
    lp_patch = linprobe(torch.cat([CLS, chunks], 1)[tr], Y[tr],
                        torch.cat([CLS, chunks], 1)[te], Y[te])
    del chunks
    results["cls_patch_lp"] = {"mAP": lp_patch}
    print(f"  CLS+patch LP = {lp_patch*100:.2f}%", flush=True)

    # SSI-v2
    print(f"\n[SSI-v2] {len(SEEDS)} seeds ...", flush=True)
    scores = []
    for s in SEEDS:
        print(f"  seed {s} ...", flush=True)
        v = train_eval(CLS, PAT, X, Y, tr, te, nc, s)
        scores.append(v)
        print(f"    seed {s} mAP = {v*100:.2f}%", flush=True)

    t = torch.tensor(scores)
    results["ssi_v2"] = {"seeds": scores, "mean": t.mean().item(), "std": t.std().item()}

    os.makedirs("results", exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n--- SUMMARY ---", flush=True)
    print(f"  CLS LP             : {lp_cls*100:.2f}%", flush=True)
    print(f"  CLS + patch-pool   : {lp_patch*100:.2f}%", flush=True)
    print(f"  SSI-v1 seed 0      : 66.07%  (reference)", flush=True)
    print(f"  SSI-v2 (ours)      : {t.mean()*100:.2f} ± {t.std()*100:.2f}%", flush=True)
    print(f"\n  Paper baselines (RS-pretrained + FT):", flush=True)
    print(f"    SatMAE++ ViT-L   : 85.10%", flush=True)
    print(f"    SMARTIES ViT-B   : 86.90%", flush=True)
    print(f"    CROMA    ViT-B   : 87.60%", flush=True)


if __name__ == "__main__":
    main()
