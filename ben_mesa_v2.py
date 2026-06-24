"""MESA with Per-Location Mid-Level Adapters (MESA-v2) for BigEarthNet.

Key fix over v1 (68.4%): per-location spectral gating.

v1 error: the spectral summary was a single global vector [B, D_SUMM]
applied uniformly to all 257 tokens. This means every patch token —
regardless of spatial position — received the same spectral correction.
An adapter at (0,0) that's over bare soil got the same gate as (15,15)
that's over deep water. That's not mid-level adaptation; it's just a
global bias that's slightly more expressive than late fusion.

v2 fix: per-location spectral gating.
  - MESASummarizer returns local_feat [B, 256, D_SPEC] (not globally pooled)
  - PerLocGatedAdapter uses local_feat[:,j,:] to gate patch token j
  - CLS token still uses global spectral summary [B, D_SUMM]

This is the correct spatial correspondence: the spectral content at
patch position (i,j) conditions the adapter gate for the DINOv2 patch
token at the same (i,j). It respects the spatial alignment principle
stated in the MESA proposal.

Additional improvements:
  - D_BOTTLE 64 → 128 (more adapter capacity)
  - Adapter blocks 9-11 → 6-11 (last 6 of 12 blocks; detach after block 5)
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
RGB_IDX = [3, 2, 1]
IMN_M   = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMN_S   = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

WL_C = [443., 490., 560., 665., 705., 740., 783., 842., 865., 945., 1610., 2190.]
FWHM = [ 20.,  65.,  35.,  30.,  15.,  15.,  20., 115.,  20.,  20.,   90.,  180.]
KNOTS = torch.tensor([
    420., 460., 510., 560., 610., 650., 680., 720.,
    760., 800., 870., 950., 1200., 1400., 1620., 2200.
])
K = len(KNOTS)

BS             = 256
LR             = 2e-3
EPOCHS         = 100
WARMUP         = 5
SEEDS          = [0, 1, 2, 3, 4]
D_SPEC         = 128    # local spectral feature dim (from MESASummarizer)
D_SUMM         = 256    # global spectral summary dim (for CLS gating)
D_BOTTLE       = 128    # adapter bottleneck (doubled from v1)
ADAPTER_BLOCKS = list(range(6, 12))  # blocks 6-11 (last 6 of 12); detach after block 5
N_ND           = 16


# ---------------------------------------------------------------------------
# Spectral measurement matrix
# ---------------------------------------------------------------------------

def build_A():
    wl    = torch.tensor(WL_C)
    sig_b = torch.tensor(FWHM) / 2.355
    delta = wl.unsqueeze(1) - KNOTS.unsqueeze(0)
    sig_t = (sig_b.unsqueeze(1) ** 2 + 80.0 ** 2).sqrt()
    A = torch.exp(-delta ** 2 / (2 * sig_t ** 2))
    return A / (A.sum(1, keepdim=True) + 1e-8)

def build_M(A, lam=0.1):
    Kn = A.shape[1]
    D  = torch.zeros(Kn - 1, Kn)
    for i in range(Kn - 1): D[i, i] = -1.; D[i, i + 1] = 1.
    return torch.linalg.solve(A.T @ A + lam * D.T @ D, A.T)


# ---------------------------------------------------------------------------
# Asymmetric Loss + metrics
# ---------------------------------------------------------------------------

def asl_loss(logits, targets, gamma_neg=4, gamma_pos=0, margin=0.05):
    p  = torch.sigmoid(logits)
    lp = targets       * torch.log(p.clamp(min=1e-8))
    pm = (p - margin).clamp(min=0)
    ln = (1 - targets) * torch.log((1 - pm).clamp(min=1e-8))
    if gamma_pos > 0: lp = lp * (1 - p)  ** gamma_pos
    if gamma_neg > 0: ln = ln * pm        ** gamma_neg
    return -(lp + ln).mean()

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
# MESA spectral front-end — returns per-location AND global features
# ---------------------------------------------------------------------------

class MESASummarizer(nn.Module):
    """S2 bands → canonical spectral coefficients → two outputs:
       local_feat  [B, 256, D_SPEC]  per-location spectral features
       global_summ [B, D_SUMM]       global summary for CLS gating
    """
    def __init__(self, d=D_SPEC, n_nd=N_ND, d_summ=D_SUMM):
        super().__init__()
        A = build_A(); M = build_M(A)
        self.register_buffer("A", A)
        self.register_buffer("M", M)

        self.spatial_enc = nn.Sequential(
            nn.Conv2d(K, d, 3, padding=1), nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1, groups=d), nn.GELU(),  # depthwise
            nn.Conv2d(d, d, 1),
        )
        self.nd_a    = nn.Parameter(torch.randn(n_nd, K) * 0.1)
        self.nd_b    = nn.Parameter(torch.randn(n_nd, K) * 0.1)
        self.nd_proj = nn.Linear(n_nd, d)
        self.combine  = nn.Sequential(nn.Linear(d + d, d), nn.GELU(), nn.LayerNorm(d))
        # Global summary head (for CLS token gating)
        self.summarize = nn.Sequential(nn.Linear(d, d_summ), nn.GELU(), nn.LayerNorm(d_summ))

    def forward(self, x_loc):
        B, N, _ = x_loc.shape
        alpha = x_loc @ self.M.T                                       # [B, 256, K]

        ag = alpha.permute(0, 2, 1).reshape(B, K, 16, 16)
        sf = self.spatial_enc(ag).flatten(2).permute(0, 2, 1)          # [B, 256, d]

        a_n, b_n = F.normalize(self.nd_a, dim=1), F.normalize(self.nd_b, dim=1)
        num = alpha @ (a_n - b_n).T
        den = (alpha @ a_n.T).abs() + (alpha @ b_n.T).abs() + 1e-6
        nf  = self.nd_proj((num / den).tanh())                         # [B, 256, d]

        local_feat  = self.combine(torch.cat([sf, nf], dim=-1))        # [B, 256, D_SPEC]
        global_summ = self.summarize(local_feat.mean(1))               # [B, D_SUMM]
        return local_feat, global_summ


# ---------------------------------------------------------------------------
# Per-location spectral-gated adapter
# ---------------------------------------------------------------------------

class PerLocGatedAdapter(nn.Module):
    """Mid-level adapter with per-location spectral gating.

    Patch token at position j is gated by local spectral features at j.
    CLS token is gated by the global spectral summary.

    This is the key fix over v1: the spectral correction is spatially
    specific — bare soil patches get a different correction than forest
    patches, even within the same image.
    """
    def __init__(self, dim=768, d_bottle=D_BOTTLE, d_spec=D_SPEC, d_summ=D_SUMM):
        super().__init__()
        self.down = nn.Linear(dim, d_bottle)

        # Per-location gating for the 256 patch tokens
        self.loc_gate = nn.Linear(d_spec, d_bottle)
        self.loc_bias = nn.Linear(d_spec, d_bottle)

        # Global gating for the CLS token
        self.cls_gate = nn.Linear(d_summ, d_bottle)
        self.cls_bias = nn.Linear(d_summ, d_bottle)

        self.up = nn.Linear(d_bottle, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, local_feat, global_summ):
        # x:           [B, 257, dim]
        # local_feat:  [B, 256, D_SPEC]  — spectral features per patch location
        # global_summ: [B, D_SUMM]       — global spectral summary for CLS
        h = F.gelu(self.down(x))                                     # [B, 257, d_bottle]

        # CLS token: global spectral gating
        cg = torch.sigmoid(self.cls_gate(global_summ)).unsqueeze(1)  # [B, 1, d_bottle]
        cb = self.cls_bias(global_summ).unsqueeze(1)
        h_cls = h[:, :1] * cg + cb                                   # [B, 1, d_bottle]

        # Patch tokens: per-location spectral gating
        pg = torch.sigmoid(self.loc_gate(local_feat))                # [B, 256, d_bottle]
        pb = self.loc_bias(local_feat)
        h_pat = h[:, 1:] * pg + pb                                   # [B, 256, d_bottle]

        h_gated = torch.cat([h_cls, h_pat], dim=1)                  # [B, 257, d_bottle]
        return x + self.up(h_gated)                                  # [B, 257, dim]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MESAv2Model(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.summarizer = MESASummarizer()

        self.dino = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14',
            pretrained=True, verbose=False
        ).to(DEV)
        for p in self.dino.parameters():
            p.requires_grad_(False)

        self.adapters = nn.ModuleList([
            PerLocGatedAdapter() for _ in ADAPTER_BLOCKS
        ])

        self.head = nn.Sequential(
            nn.Linear(768 + 768, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, nc),
        )

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()  # backbone always in eval
        return self

    def forward(self, rgb, x_loc):
        # 1. Per-location + global spectral features
        local_feat, global_summ = self.summarizer(x_loc)              # [B,256,D_SPEC], [B,D_SUMM]

        # 2. DINOv2 blocks 0 – (ADAPTER_BLOCKS[0]-1): no grad
        n_frozen = ADAPTER_BLOCKS[0]
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(rgb)               # [B, 257, 768]
            for blk in self.dino.blocks[:n_frozen]:
                x = blk(x)

        # 3. Detach: gradient flows only through adapter params + summarizer from here
        x = x.detach()

        # 4. Adapted blocks: frozen DINOv2 block + per-location spectral adapter
        for i, blk in enumerate(self.dino.blocks[n_frozen:]):
            x = blk(x)
            x = self.adapters[i](x, local_feat, global_summ)

        # 5. Final DINOv2 LayerNorm
        x     = self.dino.norm(x)
        cls   = x[:, 0]                                                # [B, 768]
        z     = x[:, 1:].mean(1)                                       # [B, 768]
        return self.head(torch.cat([cls, z], dim=1))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BENDataset(Dataset):
    def __init__(self, X, Y, idx, train):
        self.X, self.Y, self.idx, self.train = X, Y, idx, train

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        j   = self.idx[i]
        x   = self.X[j].float()

        rgb  = (x[RGB_IDX] / 3000.0).clamp(0, 1)
        rgb  = F.interpolate(rgb[None], size=224, mode='bilinear', align_corners=False)[0]
        rgb  = (rgb - IMN_M) / IMN_S

        x16  = F.interpolate(x[None], size=16, mode='bilinear', align_corners=False)[0]
        x_loc = (x16 / 3000.0).clamp(0, 1).permute(1, 2, 0).reshape(256, 12)

        if self.train:
            if torch.rand(1).item() < 0.5:
                rgb   = rgb.flip(-1)
                x_loc = x_loc.view(16, 16, 12).flip(1).reshape(256, 12)
            if torch.rand(1).item() < 0.5:
                rgb   = rgb.flip(-2)
                x_loc = x_loc.view(16, 16, 12).flip(0).reshape(256, 12)

        return rgb, x_loc, self.Y[j]


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
        if ep < warmup: return ep / max(warmup, 1)
        p = (ep - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def train_eval(X, Y, tr, te, nc, seed):
    torch.manual_seed(seed)
    net   = MESAv2Model(nc).to(DEV)
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
                loss = asl_loss(net(rgb.to(DEV), x_loc.to(DEV)), y.to(DEV))
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
        for rgb, x_loc, _ in tel:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(rgb.to(DEV), x_loc.to(DEV))
            S.append(torch.sigmoid(logits).float().cpu())
    return macro_mAP(torch.cat(S), Y[torch.tensor(te)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("high")

    print(f"MESA-v2: K={K}, D_SPEC={D_SPEC}, D_SUMM={D_SUMM}, "
          f"D_BOTTLE={D_BOTTLE}, blocks={ADAPTER_BLOCKS}", flush=True)
    print(f"  per-location gating: patch token j gated by spectral feat at j", flush=True)
    print(f"  CLS token gated by global spectral summary", flush=True)
    print(f"Training: LR={LR}, BS={BS}, EPOCHS={EPOCHS}, warmup={WARMUP}", flush=True)

    data_path     = "data/ben/ben_v1.pt"
    cache_cls     = "data/ben/dino_cls_v1.pt"
    cache_patches = "data/ben/dino_patches_v1.pt"
    out_path      = "results/ben_mesa_v2.json"

    print(f"\nloading {data_path} ...", flush=True)
    d = torch.load(data_path, weights_only=False)
    X, Y, split = d["X"], d["Y"], d["split"]
    nc = Y.shape[1]
    tr = [i for i, s in enumerate(split) if s == "train"]
    te = [i for i, s in enumerate(split) if s == "test"]
    print(f"  train={len(tr)}  test={len(te)}  classes={nc}", flush=True)

    results = {}

    # LP baselines from cached features
    if os.path.exists(cache_cls) and os.path.exists(cache_patches):
        print("\nloading cached features for LP baselines ...", flush=True)
        CLS = torch.load(cache_cls, weights_only=False)
        PAT = torch.load(cache_patches, weights_only=False)

        lp_cls = linprobe(CLS[tr], Y[tr], CLS[te], Y[te])
        results["cls_lp"] = {"mAP": lp_cls}
        print(f"  CLS LP = {lp_cls*100:.2f}%", flush=True)

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

    print(f"\n[MESA-v2] {len(SEEDS)} seeds ...", flush=True)
    scores = []
    for s in SEEDS:
        print(f"  seed {s} ...", flush=True)
        v = train_eval(X, Y, tr, te, nc, s)
        scores.append(v)
        print(f"    seed {s} mAP = {v*100:.2f}%", flush=True)

    t = torch.tensor(scores)
    results["mesa_v2"] = {"seeds": scores, "mean": t.mean().item(), "std": t.std().item()}

    os.makedirs("results", exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n--- SUMMARY ---", flush=True)
    print(f"  CLS LP              : {lp_cls*100:.2f}%", flush=True)
    print(f"  CLS + patch-pool LP : {lp_patch*100:.2f}%", flush=True)
    print(f"  MESA-v1 (global gate, blocks 9-11)  : 68.20% avg  [seeds 0-1]", flush=True)
    print(f"  MESA-v2 (per-loc gate, blocks 6-11) : {t.mean()*100:.2f} ± {t.std()*100:.2f}%", flush=True)
    print(f"\n  Paper baselines (RS-pretrained + full FT):", flush=True)
    print(f"    SatMAE++ ViT-L    : 85.10%", flush=True)
    print(f"    SMARTIES ViT-B    : 86.90%", flush=True)
    print(f"    CROMA    ViT-B    : 87.60%", flush=True)


if __name__ == "__main__":
    main()
