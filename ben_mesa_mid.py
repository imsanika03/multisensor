"""MESA with Mid-Level Spectral-Gated Adapters (MESA-MLA) for BigEarthNet.

Architecture:
  - MESA spectral front-end: S2 bands → canonical spectral coefficients (K=16)
    → spatial conv encoder (16×16 field) + learned ND features → summary [B, D_SUMM]

  - DINOv2 ViT-B/14 (completely frozen weights), running on-the-fly:
    * Blocks 0–8: no-grad context → detach (efficient, no backprop through here)
    * Blocks 9–11: grad-enabled for adapter backprop
    * Each of the 3 last blocks followed by a SpectralGatedAdapter

  - SpectralGatedAdapter [768→D_BOTTLE→768]:
    h = GELU(down(x))
    h = h * sigmoid(gate(summary)) + bias(summary)   ← spectral gating
    out = x + up(h)                                    ← zero-init residual

    The spectral summary lets spectral evidence reshape the token geometry
    INSIDE DINOv2 before the CLS token is finalized — strictly stronger
    than pure late fusion.

  - Head: concat[adapted_cls, mean(adapted_patches)] → 512 → nc

Preprocessing matches the cached features exactly:
  rgb = (X[:, [3,2,1]] / 3000.0).clamp(0,1) → resize 224 → ImageNet norm

Loss: Asymmetric Loss (ASL) for multilabel class imbalance.
"""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Band metadata
# ---------------------------------------------------------------------------
RGB_IDX = [3, 2, 1]   # B04, B03, B02 in our 12-band ordering
IMN_M   = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_S   = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

WL_C = [443., 490., 560., 665., 705., 740., 783., 842., 865., 945., 1610., 2190.]
FWHM = [ 20.,  65.,  35.,  30.,  15.,  15.,  20., 115.,  20.,  20.,   90.,  180.]
KNOTS = torch.tensor([
    420., 460., 510., 560., 610., 650., 680., 720.,
    760., 800., 870., 950., 1200., 1400., 1620., 2200.
])
K = len(KNOTS)  # 16

BS            = 256    # DINOv2 on-the-fly — larger than 128 is fine on H100 80GB
LR            = 2e-3
EPOCHS        = 100
WARMUP        = 5
SEEDS         = [0, 1, 2, 3, 4]
D_SPEC        = 128    # spectral feature dim
N_ND          = 16     # learned ND features
D_SUMM        = 256    # spectral summary dim fed into adapters
D_BOTTLE      = 64     # adapter bottleneck dim
ADAPTER_BLOCKS = [9, 10, 11]  # which DINOv2 blocks get adapters (last 3 of 12)


# ---------------------------------------------------------------------------
# Spectral measurement matrix (precomputed from ESA band SRFs)
# ---------------------------------------------------------------------------

def build_A():
    wl    = torch.tensor(WL_C)
    sig_b = torch.tensor(FWHM) / 2.355
    sig_k = 80.0
    delta = wl.unsqueeze(1) - KNOTS.unsqueeze(0)           # [12, K]
    sig_t = (sig_b.unsqueeze(1) ** 2 + sig_k ** 2).sqrt() # [12, K]
    A = torch.exp(-delta ** 2 / (2 * sig_t ** 2))
    return A / (A.sum(1, keepdim=True) + 1e-8)             # row-normalize [12, K]


def build_M(A, lam=0.1):
    """Regularized pseudoinverse M = (A^T A + λ D^T D)^{-1} A^T  [K, 12]."""
    Kn = A.shape[1]
    D  = torch.zeros(Kn - 1, Kn)
    for i in range(Kn - 1):
        D[i, i] = -1.; D[i, i + 1] = 1.
    return torch.linalg.solve(A.T @ A + lam * D.T @ D, A.T)


# ---------------------------------------------------------------------------
# Asymmetric Loss
# ---------------------------------------------------------------------------

def asl_loss(logits, targets, gamma_neg=4, gamma_pos=0, margin=0.05):
    p   = torch.sigmoid(logits)
    lp  = targets       * torch.log(p.clamp(min=1e-8))
    pm  = (p - margin).clamp(min=0)
    ln  = (1 - targets) * torch.log((1 - pm).clamp(min=1e-8))
    if gamma_pos > 0: lp = lp * (1 - p)  ** gamma_pos
    if gamma_neg > 0: ln = ln * pm        ** gamma_neg
    return -(lp + ln).mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def macro_mAP(scores, targets):
    aps = []
    for c in range(scores.shape[1]):
        t = targets[:, c]
        if t.sum() == 0: continue
        tt = t[scores[:, c].argsort(descending=True)]
        prec = tt.cumsum(0) / torch.arange(1, len(tt) + 1, dtype=torch.float)
        aps.append((prec * tt).sum().item() / tt.sum().item())
    return sum(aps) / len(aps)


# ---------------------------------------------------------------------------
# MESA spectral front-end → global summary token
# ---------------------------------------------------------------------------

class MESASummarizer(nn.Module):
    """S2 bands → canonical spectral coefficients → spectral summary [B, D_SUMM].

    This summary conditions the mid-level adapters in DINOv2 — it is the
    'spectral evidence' that reshapes the frozen backbone's token geometry.
    """
    def __init__(self, d=D_SPEC, n_nd=N_ND, d_summ=D_SUMM):
        super().__init__()
        A = build_A(); M = build_M(A)
        self.register_buffer("A", A)  # [12, K]
        self.register_buffer("M", M)  # [K, 12]

        # Spatial encoder over 16×16 canonical coefficient field
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(K, d, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d), nn.GELU(),  # depthwise
            nn.Conv2d(d, d, kernel_size=1),
        )

        # Learned normalized-difference features over canonical coefficients
        self.nd_a    = nn.Parameter(torch.randn(n_nd, K) * 0.1)
        self.nd_b    = nn.Parameter(torch.randn(n_nd, K) * 0.1)
        self.nd_proj = nn.Linear(n_nd, d)

        self.combine   = nn.Sequential(nn.Linear(d + d, d), nn.GELU(), nn.LayerNorm(d))
        self.summarize = nn.Sequential(nn.Linear(d, d_summ), nn.GELU(), nn.LayerNorm(d_summ))

    def forward(self, x_loc):
        # x_loc: [B, 256, 12] — normalized S2 band values per patch location
        B, N, _ = x_loc.shape
        alpha = x_loc @ self.M.T                                     # [B, 256, K]

        # Spatial features from 16×16 canonical field
        ag = alpha.permute(0, 2, 1).reshape(B, K, 16, 16)
        sf = self.spatial_enc(ag).flatten(2).permute(0, 2, 1)        # [B, 256, d]

        # Learned ND features
        a_n, b_n = F.normalize(self.nd_a, dim=1), F.normalize(self.nd_b, dim=1)
        num = alpha @ (a_n - b_n).T
        den = (alpha @ a_n.T).abs() + (alpha @ b_n.T).abs() + 1e-6
        nf  = self.nd_proj((num / den).tanh())                       # [B, 256, d]

        feat = self.combine(torch.cat([sf, nf], dim=-1))             # [B, 256, d]
        return self.summarize(feat.mean(1))                           # [B, D_SUMM]


# ---------------------------------------------------------------------------
# Spectral-gated bottleneck adapter
# ---------------------------------------------------------------------------

class SpectralGatedAdapter(nn.Module):
    """Bottleneck adapter conditioned on MESA spectral summary.

    h = GELU(down(x))
    h = h * sigmoid(gate(summary)) + bias(summary)  ← spectral gating
    out = x + up(h)                                  ← residual, zero-init

    up.weight zero-initialized → adapter starts as identity.
    """
    def __init__(self, dim=768, d_bottle=D_BOTTLE, d_summ=D_SUMM):
        super().__init__()
        self.down = nn.Linear(dim, d_bottle)
        self.gate = nn.Linear(d_summ, d_bottle)   # scale gate from spectral summary
        self.bias = nn.Linear(d_summ, d_bottle)   # bias from spectral summary
        self.up   = nn.Linear(d_bottle, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, summary):
        # x:       [B, N, dim]    — DINOv2 patch + CLS tokens
        # summary: [B, d_summ]   — MESA spectral summary
        h = F.gelu(self.down(x))                                # [B, N, d_bottle]
        g = torch.sigmoid(self.gate(summary)).unsqueeze(1)      # [B, 1, d_bottle]
        b = self.bias(summary).unsqueeze(1)                     # [B, 1, d_bottle]
        return x + self.up(h * g + b)                           # [B, N, dim]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MESAMidModel(nn.Module):
    def __init__(self, nc):
        super().__init__()
        # MESA spectral front-end
        self.summarizer = MESASummarizer()

        # Frozen DINOv2 ViT-B/14
        self.dino = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14',
            pretrained=True, verbose=False
        ).to(DEV)
        for p in self.dino.parameters():
            p.requires_grad_(False)

        # Mid-level adapters: one after each of the last 3 DINOv2 blocks
        self.adapters = nn.ModuleList([
            SpectralGatedAdapter() for _ in ADAPTER_BLOCKS
        ])

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(768 + 768, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, nc),
        )

    def train(self, mode=True):
        """DINOv2 always stays in eval mode (frozen weights, no dropout)."""
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, rgb, x_loc):
        B = rgb.shape[0]

        # 1. MESA spectral summary (conditions the mid-level adapters)
        summary = self.summarizer(x_loc)            # [B, D_SUMM]

        # 2. DINOv2 blocks 0–8: no grad (no adapter, no backprop here)
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(rgb)   # [B, 257, 768]
            for blk in self.dino.blocks[:9]:
                x = blk(x)

        # 3. Detach at block 8/9 boundary — gradients only flow through
        #    the adapter parameters and summary from this point on.
        #    DINOv2 blocks 9–11 are traversed for the chain rule but
        #    their parameters (requires_grad=False) don't accumulate grad.
        x = x.detach()

        # 4. DINOv2 blocks 9–11 + spectral-gated adapters
        for i, blk in enumerate(self.dino.blocks[9:]):
            x = blk(x)                              # frozen block (grad flows through)
            x = self.adapters[i](x, summary)        # trainable adapter

        # 5. Final DINOv2 LayerNorm
        x = self.dino.norm(x)                       # [B, 257, 768]
        cls = x[:, 0]                               # [B, 768]
        pat = x[:, 1:]                              # [B, 256, 768]
        z   = pat.mean(1)                           # [B, 768]

        return self.head(torch.cat([cls, z], dim=1))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BENDataset(Dataset):
    """Loads S2 bands; computes RGB (for DINOv2) and x_loc (for MESA) on-the-fly."""

    def __init__(self, X, Y, idx, train):
        self.X, self.Y = X, Y
        self.idx = idx
        self.train = train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j   = self.idx[i]
        x   = self.X[j].float()                    # [12, 120, 120] raw reflectance

        # RGB for DINOv2: B04, B03, B02 — matches cached feature preprocessing exactly
        rgb = x[RGB_IDX]                            # [3, 120, 120]
        rgb = (rgb / 3000.0).clamp(0, 1)
        rgb = F.interpolate(rgb[None], size=224, mode='bilinear', align_corners=False)[0]
        rgb = (rgb - IMN_M.squeeze(0)) / IMN_S.squeeze(0)  # ImageNet normalization

        # S2 for MESA: all 12 bands resized to 16×16 (DINOv2 patch grid)
        x16  = F.interpolate(x[None], size=16, mode='bilinear', align_corners=False)[0]
        x_loc = (x16 / 3000.0).clamp(0, 1).permute(1, 2, 0).reshape(256, 12)

        # Consistent geometric augmentation (both rgb and x_loc spatial grids)
        if self.train:
            if torch.rand(1).item() < 0.5:
                rgb   = rgb.flip(-1)                                     # horizontal
                x_loc = x_loc.view(16, 16, 12).flip(1).reshape(256, 12)
            if torch.rand(1).item() < 0.5:
                rgb   = rgb.flip(-2)                                     # vertical
                x_loc = x_loc.view(16, 16, 12).flip(0).reshape(256, 12)

        return rgb, x_loc, self.Y[j]


# ---------------------------------------------------------------------------
# Baselines (using cached features for comparability)
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


def train_eval(X, Y, tr, te, nc, seed):
    torch.manual_seed(seed)
    net   = MESAMidModel(nc).to(DEV)
    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=LR, weight_decay=1e-4
    )
    sched = cosine_with_warmup(opt, WARMUP, EPOCHS)

    trl = DataLoader(
        BENDataset(X, Y, tr, train=True),
        batch_size=BS, shuffle=True, num_workers=8,
        drop_last=True, pin_memory=True, persistent_workers=True,
    )
    for ep in range(EPOCHS):
        net.train()
        for rgb, x_loc, y in trl:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = asl_loss(
                    net(rgb.to(DEV), x_loc.to(DEV)), y.to(DEV)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{EPOCHS}", flush=True)

    net.eval()
    tel = DataLoader(
        BENDataset(X, Y, te, train=False),
        batch_size=BS * 2, shuffle=False, num_workers=8,
        pin_memory=True, persistent_workers=True,
    )
    S = []
    with torch.no_grad():
        for rgb, x_loc, y in tel:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(rgb.to(DEV), x_loc.to(DEV))
            S.append(torch.sigmoid(logits).float().cpu())
    return macro_mAP(torch.cat(S), Y[torch.tensor(te)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("high")

    print(f"MESA-MLA: K={K} knots, D={D_SPEC}, D_SUMM={D_SUMM}, "
          f"D_BOTTLE={D_BOTTLE}, adapter_blocks={ADAPTER_BLOCKS}", flush=True)
    print(f"Training: LR={LR}, BS={BS}, EPOCHS={EPOCHS}, warmup={WARMUP}", flush=True)
    print(f"DINOv2: blocks 0–8 frozen+no_grad, blocks 9–11 frozen+adapters", flush=True)

    data_path     = "data/ben/ben_v1.pt"
    cache_cls     = "data/ben/dino_cls_v1.pt"
    cache_patches = "data/ben/dino_patches_v1.pt"
    out_path      = "results/ben_mesa_mid.json"

    print(f"\nloading {data_path} ...", flush=True)
    d = torch.load(data_path, weights_only=False)
    X, Y, split = d["X"], d["Y"], d["split"]
    nc = Y.shape[1]

    tr = [i for i, s in enumerate(split) if s == "train"]
    te = [i for i, s in enumerate(split) if s == "test"]
    print(f"  train={len(tr)}  test={len(te)}  classes={nc}", flush=True)

    results = {}

    # LP baselines from cached features (for consistent comparison)
    if os.path.exists(cache_cls) and os.path.exists(cache_patches):
        print("\nloading cached DINOv2 features for LP baselines ...", flush=True)
        CLS = torch.load(cache_cls, weights_only=False)
        PAT = torch.load(cache_patches, weights_only=False)

        print("[LP] CLS-only ...", flush=True)
        lp_cls = linprobe(CLS[tr], Y[tr], CLS[te], Y[te])
        results["cls_lp"] = {"mAP": lp_cls}
        print(f"  CLS LP = {lp_cls*100:.2f}%", flush=True)

        print("[LP] CLS + patch-pool ...", flush=True)
        chunks = torch.zeros(len(PAT), 768)
        for i in range(0, len(PAT), 2000):
            chunks[i:i+2000] = PAT[i:i+2000].float().mean(1)
        lp_patch = linprobe(torch.cat([CLS, chunks], 1)[tr], Y[tr],
                            torch.cat([CLS, chunks], 1)[te], Y[te])
        del CLS, PAT, chunks
        results["cls_patch_lp"] = {"mAP": lp_patch}
        print(f"  CLS+patch LP = {lp_patch*100:.2f}%", flush=True)
    else:
        lp_cls = lp_patch = 0.0
        print("  (cached features not found — skipping LP baselines)", flush=True)

    # MESA-MLA
    print(f"\n[MESA-MLA] {len(SEEDS)} seeds ...", flush=True)
    scores = []
    for s in SEEDS:
        print(f"  seed {s} ...", flush=True)
        v = train_eval(X, Y, tr, te, nc, s)
        scores.append(v)
        print(f"    seed {s} mAP = {v*100:.2f}%", flush=True)

    t = torch.tensor(scores)
    results["mesa_mla"] = {
        "seeds": scores, "mean": t.mean().item(), "std": t.std().item()
    }

    os.makedirs("results", exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n--- SUMMARY ---", flush=True)
    print(f"  CLS LP              : {lp_cls*100:.2f}%", flush=True)
    print(f"  CLS + patch-pool LP : {lp_patch*100:.2f}%", flush=True)
    print(f"  SSI-v1  seed 0      : 66.07%  (late fusion reference)", flush=True)
    print(f"  MESA-MLA (ours)     : {t.mean()*100:.2f} ± {t.std()*100:.2f}%", flush=True)
    print(f"\n  Paper baselines (RS-pretrained + full FT):", flush=True)
    print(f"    SatMAE++ ViT-L    : 85.10%", flush=True)
    print(f"    SMARTIES ViT-B    : 86.90%", flush=True)
    print(f"    CROMA    ViT-B    : 87.60%", flush=True)


if __name__ == "__main__":
    main()
