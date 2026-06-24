"""STE on the standard BigEarthNet 10% benchmark.

Architecture v2 — frozen DINOv2 ViT-B/14 + spectral adapter:
  - DINOv2 CLS token [768] + all 256 patch tokens [256×768] (precomputed, frozen)
  - SpectralEnc: per-band patch embeddings → wavelength conditioning → band self-attention
  - STEFusion: cross-attention (spectral band tokens attend to DINOv2 patch tokens)
               + spatial pool + spectral summary → MLP head
  - Backbone remains completely frozen throughout

Protocol matches SMARTIES/CROMA/SatMAE++ (BEN 10% column):
  - 10% of official train split (26,969 patches, seed 0)
  - Full official test split (125,866 patches)
  - 5 seeds, report mean ± std macro-mAP
"""
import json, math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Sentinel-2 band wavelengths (nm) and resolution class (0=10m, 1=20m, 2=60m)
WL  = torch.tensor([443., 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190])
RES = torch.tensor([2,    0,   0,   0,   1,   1,   1,   0,   1,   2,   1,    1   ])
RGB_IDX = [3, 2, 1]   # B04, B03, B02
IMN_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

BS      = 2048   # H100 80GB
LR      = 1e-3   # attention layers need lower LR than pure MLP
EPOCHS  = 150    # more steps to compensate for lower LR
DINO_BS = 1024   # forward_features uses more memory than forward-CLS-only
SEEDS   = [0, 1, 2, 3, 4]
D       = 256    # spectral encoder dim
N_HEADS = 4


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
# DINOv2 precomputation — CLS + all 256 patch tokens (frozen, one-time)
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_features(X, cache_cls, cache_patches, bs=DINO_BS):
    cls_exists = os.path.exists(cache_cls)
    pat_exists = os.path.exists(cache_patches)
    if cls_exists and pat_exists:
        print("  loading DINOv2 features from cache", flush=True)
        return (torch.load(cache_cls, weights_only=False),
                torch.load(cache_patches, weights_only=False))

    print(f"  computing DINOv2-RGB CLS + patch tokens (bs={bs}) ...", flush=True)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',
                          verbose=False).to(DEV).eval()
    cls_out, patch_out = [], []
    for i in range(0, len(X), bs):
        r = (X[i:i + bs][:, RGB_IDX].float() / 3000.0).clamp(0, 1).to(DEV)
        r = F.interpolate(r, size=224, mode="bilinear", align_corners=False)
        r = (r - IMN_M.to(DEV)) / IMN_S.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats  = dino.forward_features(r)
            cls    = F.normalize(feats['x_norm_clstoken'].float(), dim=1)    # [B, 768]
            patches = feats['x_norm_patchtokens'].float()                    # [B, 256, 768]
        cls_out.append(cls.cpu())
        patch_out.append(patches.cpu().half())
        if (i // bs) % 10 == 0:
            print(f"    {i}/{len(X)}", flush=True)

    cls_t = torch.cat(cls_out)                   # [N, 768] float32
    pat_t = torch.cat(patch_out)                 # [N, 256, 768] float16  (~60GB)
    if not cls_exists:
        torch.save(cls_t, cache_cls)
    torch.save(pat_t, cache_patches)
    print(f"  CLS {tuple(cls_t.shape)}, patches {tuple(pat_t.shape)}", flush=True)
    return cls_t, pat_t


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def sinusoid(vals, d):
    pos = (vals / 100.0).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe  = torch.zeros(len(vals), d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SpectralEnc(nn.Module):
    """Per-band patch embeddings → wavelength conditioning → band self-attention.

    Returns:
        summary  [B, d]   — global spectral summary (for concatenation to head)
        band_tok [B, 12, d] — per-band tokens (for cross-attention with DINOv2 patches)
    """
    def __init__(self, d=D, use_wl=True):
        super().__init__()
        self.use_wl = use_wl
        self.band_embed = nn.Linear(16, d)
        if use_wl:
            self.register_buffer("wl", sinusoid(WL, d))
            self.wl_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, d))
        else:
            self.band_idx = nn.Embedding(12, d)
        self.res_emb = nn.Embedding(3, d)
        self.register_buffer("res_cls", RES)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(True), nn.Linear(d, d))
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU(True),
        )
        # Band self-attention: 12 bands attend to each other
        self.band_attn  = nn.MultiheadAttention(d, N_HEADS, batch_first=True, dropout=0.1)
        self.band_norm  = nn.LayerNorm(d)
        self.band_ff    = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.band_norm2 = nn.LayerNorm(d)
        self.out        = nn.Linear(d, d)

    def cond(self):
        c = self.wl_mlp(self.wl) if self.use_wl else self.band_idx.weight
        return c + self.res_emb(self.res_cls)

    def forward(self, x, mask):                          # x[B,12,64,64], mask[B,12]
        B = x.shape[0]
        p = x.reshape(B, 12, 16, 4, 16, 4).permute(0, 1, 2, 4, 3, 5).reshape(B, 12, 256, 16)
        v = self.band_embed(p)
        cond = self.cond()[None, :, None, :].expand(B, 12, 256, -1)
        u = self.fuse(torch.cat([v, cond], dim=-1))      # [B, 12, 256, d]

        m = mask[:, :, None, None]
        # per-band spatial mean → [B, 12, d]
        u_band = (u * m).mean(2)

        # band self-attention — masked-out bands not attended to
        kpm = ~mask.bool()                               # True = ignore this key
        attn_out, _ = self.band_attn(u_band, u_band, u_band, key_padding_mask=kpm)
        u_band = self.band_norm(u_band + attn_out)
        u_band = self.band_norm2(u_band + self.band_ff(u_band))  # [B, 12, d]

        # spatial conv summary
        u_global = (u * m).sum(1) / m.sum(1).clamp(min=1)       # [B, 256, d]
        u_global = u_global.permute(0, 2, 1).reshape(B, -1, 16, 16)
        u_global = self.spatial_conv(u_global).mean(dim=(2, 3))  # [B, d]

        return self.out(u_global), u_band


class STEFusion(nn.Module):
    """CLS + DINOv2 patch tokens (spatial) + spectral cross-attention + spectral summary → head."""
    def __init__(self, d, nc, use_wl):
        super().__init__()
        self.enc = SpectralEnc(d, use_wl)
        # project DINOv2 patch tokens (768) → d for cross-attention K/V
        self.patch_proj   = nn.Linear(768, d)
        # cross-attention: spectral band tokens (Q) attend to DINOv2 patches (K, V)
        self.cross_attn   = nn.MultiheadAttention(d, N_HEADS, batch_first=True, dropout=0.1)
        self.cross_norm   = nn.LayerNorm(d)
        # project pooled DINOv2 patches for concatenation
        self.spatial_proj = nn.Linear(768, d)
        # head: CLS(768) + spatial_pool(d) + cross_summary(d) + spectral_summary(d)
        self.head = nn.Sequential(
            nn.Linear(768 + 3 * d, 512), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(512, 256),         nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(256, nc),
        )

    def forward(self, cls, patches, x, mask):
        # cls:     [B, 768]
        # patches: [B, 256, 768]  float16 → cast to float inside
        # x:       [B, 12, 64, 64]
        # mask:    [B, 12]
        spectral_summary, band_tokens = self.enc(x, mask)    # [B,d], [B,12,d]
        pat = patches.float()
        pat_d = self.patch_proj(pat)                         # [B, 256, d]
        cross, _ = self.cross_attn(band_tokens, pat_d, pat_d)
        cross_summary = self.cross_norm(band_tokens + cross).mean(1)  # [B, d]
        spatial_pool  = self.spatial_proj(pat.mean(1))       # [B, d]
        feat = torch.cat([cls, spatial_pool, cross_summary, spectral_summary], dim=1)
        return self.head(feat)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BENDataset(Dataset):
    def __init__(self, CLS, PAT, X, Y, idx, mean, std, present, train):
        self.CLS, self.PAT = CLS, PAT
        self.X, self.Y = X, Y
        self.idx, self.mean, self.std = idx, mean, std
        self.present, self.train = present, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x  = F.interpolate(self.X[j].float()[None], size=64,
                            mode="bilinear", align_corners=False)[0]
        xn = (x - self.mean) / self.std
        mask = torch.zeros(12)
        mask[self.present] = 1
        if self.train:
            for b in self.present:
                if b not in RGB_IDX and torch.rand(1).item() < 0.4:
                    mask[b] = 0
        xn = xn * mask[:, None, None]
        if self.train and torch.rand(1).item() < 0.5:
            xn = xn.flip(-1)
        return self.CLS[j], self.PAT[j], xn, mask, self.Y[j]


# ---------------------------------------------------------------------------
# Train + evaluate one run
# ---------------------------------------------------------------------------

def train_eval(model_cls, CLS, PAT, X, Y, tr, te, mean, std, nc, present, seed):
    torch.manual_seed(seed)
    net = model_cls(nc).to(DEV)
    opt   = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    trl = DataLoader(
        BENDataset(CLS, PAT, X, Y, tr, mean, std, present, True),
        batch_size=BS, shuffle=True, num_workers=4, drop_last=True, pin_memory=True,
    )
    for ep in range(EPOCHS):
        net.train()
        for c, p, x, m, y in trl:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(
                    net(c.to(DEV), p.to(DEV), x.to(DEV), m.to(DEV)), y.to(DEV)
                )
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{EPOCHS}", flush=True)

    net.eval()
    tel = DataLoader(
        BENDataset(CLS, PAT, X, Y, te, mean, std, present, False),
        batch_size=BS, shuffle=False, num_workers=4, pin_memory=True,
    )
    S = []
    with torch.no_grad():
        for c, p, x, m, y in tel:
            S.append(torch.sigmoid(net(c.to(DEV), p.to(DEV), x.to(DEV), m.to(DEV))).cpu().float())
    return macro_mAP(torch.cat(S), Y[torch.tensor(te)])


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


def make_ste(nc):
    return STEFusion(D, nc, use_wl=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("high")

    data_path     = "data/ben/ben_v1.pt"
    cache_cls     = "data/ben/dino_cls_v1.pt"
    cache_patches = "data/ben/dino_patches_v1.pt"
    out_path      = "results/ben_standard.json"
    if "--data"    in sys.argv: data_path     = sys.argv[sys.argv.index("--data")    + 1]
    if "--cache"   in sys.argv: cache_cls     = sys.argv[sys.argv.index("--cache")   + 1]
    if "--patches" in sys.argv: cache_patches = sys.argv[sys.argv.index("--patches") + 1]
    if "--out"     in sys.argv: out_path      = sys.argv[sys.argv.index("--out")     + 1]

    print(f"loading {data_path} ...", flush=True)
    d = torch.load(data_path, weights_only=False)
    X, Y, split = d["X"], d["Y"], d["split"]
    nc = Y.shape[1]

    tr = [i for i, s in enumerate(split) if s == "train"]
    te = [i for i, s in enumerate(split) if s == "test"]
    print(f"  train={len(tr)}  test={len(te)}  classes={nc}", flush=True)

    samp = torch.tensor(tr[:5000])
    x5   = F.interpolate(X[samp].float(), size=64, mode="bilinear", align_corners=False)
    mean = x5.mean(dim=(0, 2, 3), keepdim=True)[0]
    std  = x5.std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6

    CLS, PAT = precompute_features(X, cache_cls, cache_patches)

    allb    = list(range(12))
    results = {}

    # --- LP baselines ---
    print("\n[LP] RGB-DINOv2 CLS-only linear probe ...", flush=True)
    lp_cls = linprobe(CLS[tr], Y[tr], CLS[te], Y[te])
    results["rgb_dino_lp_cls"] = {"mAP": lp_cls}
    print(f"  CLS-only LP        = {lp_cls*100:.2f}%", flush=True)

    # CLS+patch-pool LP — stream one chunk at a time to avoid 120GB fp32 peak
    print("\n[LP] RGB-DINOv2 CLS + patch-pool linear probe ...", flush=True)
    patch_mean = torch.zeros(len(PAT), 768)
    for i in range(0, len(PAT), 2000):
        patch_mean[i:i + 2000] = PAT[i:i + 2000].float().mean(1)
    CLS_PAT = torch.cat([CLS, patch_mean], dim=1)
    del patch_mean
    lp_full = linprobe(CLS_PAT[tr], Y[tr], CLS_PAT[te], Y[te])
    results["rgb_dino_lp_full"] = {"mAP": lp_full}
    print(f"  CLS+patch-pool LP  = {lp_full*100:.2f}%", flush=True)
    del CLS_PAT

    # --- STE ---
    print(f"\n[STE] cross-attn + band-self-attn, {len(SEEDS)} seeds ...", flush=True)
    ste_scores = []
    for s in SEEDS:
        print(f"  seed {s} ...", flush=True)
        v = train_eval(make_ste, CLS, PAT, X, Y, tr, te, mean, std, nc, allb, s)
        ste_scores.append(v)
        print(f"    seed {s} mAP = {v*100:.2f}%", flush=True)
    ste_t = torch.tensor(ste_scores)
    results["ste_crossattn_wl"] = {
        "seeds": ste_scores,
        "mean":  ste_t.mean().item(),
        "std":   ste_t.std().item(),
    }

    os.makedirs("results", exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n--- SUMMARY (standard BEN 10% protocol) ---", flush=True)
    print(f"  DINOv2 LP (CLS only)    : {lp_cls*100:.2f}%", flush=True)
    print(f"  DINOv2 LP (CLS+patches) : {lp_full*100:.2f}%", flush=True)
    print(f"  STE (ours)              : {ste_t.mean()*100:.2f} ± {ste_t.std()*100:.2f}%", flush=True)
    print(f"\n  Paper baselines (S2 FT, RS-pretrained backbones):", flush=True)
    print(f"    SatMAE++ ViT-L : 85.10%", flush=True)
    print(f"    SMARTIES ViT-B : 86.90% (S2)  /  78.90% (S1)", flush=True)
    print(f"    CROMA    ViT-B : 87.60%", flush=True)


if __name__ == "__main__":
    main()
