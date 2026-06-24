"""Spatial Spectral Injection (SSI) for BigEarthNet 10% benchmark.

Key insight from failure of global spectral indices (30% mAP):
  Global spectral means lose spatial structure. DINOv2 gets 63% from RGB
  spatial texture — guiding it with a weaker global signal hurts.

Fix: spatial correspondence.
  1. Resize all 12 S2 bands to 16x16 (matching DINOv2's patch grid at 224px input)
  2. For each of the 256 patch locations, compute a wavelength-conditioned
     spectral embedding from the 12 band values at that location
  3. Inject it directly into the frozen DINOv2 patch token at the same location
  4. Pool the augmented 256 patch tokens + CLS → classify

The frozen DINOv2 patch token at (i,j) encodes "what RGB texture is here."
The spectral injection adds "and the physical spectral properties here are X."
Together they are much more informative than either alone.

Backbone is completely frozen.
"""
import json, math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Sentinel-2 band wavelengths (nm) — for wavelength conditioning
WL  = torch.tensor([443., 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190])
RES = torch.tensor([2,    0,   0,   0,   1,   1,   1,   0,   1,   2,   1,    1   ])

BS      = 2048
LR      = 3e-3
EPOCHS  = 150
SEEDS   = [0, 1, 2, 3, 4]
D_SPEC  = 64   # spectral embedding dim per location


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
# Model
# ---------------------------------------------------------------------------

class SpectralInjector(nn.Module):
    """Per-location spectral injection into frozen DINOv2 patch tokens.

    For each of the 256 DINOv2 patch locations:
      - Take the 12 S2 band values at that location
      - Embed each band value with its wavelength conditioning
      - Average across bands → local spectral token [D_SPEC]
      - Project up to DINOv2 dim [768] and add to frozen patch token

    This preserves spatial correspondence: spectral info at location (i,j)
    enriches the RGB patch token at the same location (i,j).
    """
    def __init__(self, d=D_SPEC):
        super().__init__()
        # Fixed wavelength embeddings (no params — encodes physical prior)
        self.register_buffer("wl_emb", sinusoid(WL, d))   # [12, d]
        self.register_buffer("res_emb_w", torch.randn(3, d) * 0.02)  # resolution embed

        # Per-band: (normalized_value [1] + wavelength_emb [d]) → d
        self.band_proj = nn.Linear(1 + d, d)

        # Across-band aggregation
        self.agg = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

        # Project spectral token up to DINOv2 patch dim for injection
        self.up = nn.Linear(d, 768)
        nn.init.zeros_(self.up.weight)   # start as identity injection (zero)
        nn.init.zeros_(self.up.bias)

    def forward(self, x_loc, mask=None):
        # x_loc: [B, 256, 12] — raw band values per patch location (normalized to [0,1])
        B, N, C = x_loc.shape
        v = x_loc.unsqueeze(-1)                             # [B, 256, 12, 1]
        w = self.wl_emb.unsqueeze(0).unsqueeze(0).expand(B, N, C, -1)  # [B, 256, 12, d]
        feat = self.band_proj(torch.cat([v, w], dim=-1))    # [B, 256, 12, d]
        spec = self.agg(feat.mean(2))                       # [B, 256, d]
        return self.up(spec)                                 # [B, 256, 768]


class SSIModel(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.injector = SpectralInjector()
        self.head = nn.Sequential(
            nn.Linear(768 + 768, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, nc),
        )

    def forward(self, cls, patches, x_loc):
        # cls:     [B, 768]
        # patches: [B, 256, 768] float16 → cast inside
        # x_loc:   [B, 256, 12] normalized band values per location
        pat = patches.float()                                # [B, 256, 768]
        injection = self.injector(x_loc)                    # [B, 256, 768]
        augmented = pat + injection                          # [B, 256, 768]
        z = augmented.mean(1)                               # [B, 768]
        return self.head(torch.cat([cls, z], dim=1))


# ---------------------------------------------------------------------------
# Dataset
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
        # Resize all 12 bands to 16×16 to match DINOv2 patch grid
        x16 = F.interpolate(self.X[j].float()[None], size=(16, 16),
                             mode="bilinear", align_corners=False)[0]  # [12, 16, 16]
        if self.train and torch.rand(1).item() < 0.5:
            x16 = x16.flip(-1)
        x_loc = (x16 / 3000.0).clamp(0, 1).permute(1, 2, 0).reshape(256, 12)  # [256, 12]
        return self.CLS[j], self.PAT[j], x_loc, self.Y[j]


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
# Train + eval
# ---------------------------------------------------------------------------

def train_eval(CLS, PAT, X, Y, tr, te, nc, seed):
    torch.manual_seed(seed)
    net   = SSIModel(nc).to(DEV)
    opt   = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

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
    out_path      = "results/ben_spatial_spectral.json"

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

    # LP baselines
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

    # SSI
    print(f"\n[SSI] spatial spectral injection, {len(SEEDS)} seeds ...", flush=True)
    scores = []
    for s in SEEDS:
        print(f"  seed {s} ...", flush=True)
        v = train_eval(CLS, PAT, X, Y, tr, te, nc, s)
        scores.append(v)
        print(f"    seed {s} mAP = {v*100:.2f}%", flush=True)

    t = torch.tensor(scores)
    results["ssi"] = {"seeds": scores, "mean": t.mean().item(), "std": t.std().item()}

    os.makedirs("results", exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n--- SUMMARY ---", flush=True)
    print(f"  CLS LP             : {lp_cls*100:.2f}%", flush=True)
    print(f"  CLS + patch-pool   : {lp_patch*100:.2f}%", flush=True)
    print(f"  SSI (ours)         : {t.mean()*100:.2f} ± {t.std()*100:.2f}%", flush=True)
    print(f"\n  Paper baselines (RS-pretrained + FT):", flush=True)
    print(f"    SatMAE++ ViT-L : 85.10%", flush=True)
    print(f"    SMARTIES ViT-B : 86.90%", flush=True)
    print(f"    CROMA    ViT-B : 87.60%", flush=True)


if __name__ == "__main__":
    main()
