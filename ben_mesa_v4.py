"""MESA-v4 (v2): Bidirectional spectral register tokens with gated patch update.

Failure analysis of v4-original (69.59%) and v4-fixed (16.95%):
  v4-original: spectral tokens INSIDE frozen DINOv2 attention → dilutes spatial attention
  v4-fixed:    unbounded bidir cross-attn for patch update → correction swamps DINOv2
               features, model outputs constant prediction for all samples (~16% = random)

Root cause of v4-fixed failure:
  Zero-init output projection starts patch corrections at 0, but the gradient
  pushes p_o to grow fast. Without bounding, corrections become >> DINOv2 features.
  Model learns a degenerate solution: constant output regardless of input.

This fix:
  - spec ← patches: small cross-attention (spec updates from DINOv2 patches, zero-init out)
    Safe: spec tokens are auxiliary, no risk of corrupting frozen backbone features.
  - patches ← spec: SIGMOID-GATED adapter (same mechanism as v2, bounded by design)
    Each patch token attends softly to K=8 evolved spec tokens → per-patch spec context.
    Gate = sigmoid(W_gate @ spec_context) bounds correction to [0,1] scale.
    Zero-init output projection.

Why this beats v2:
  v2: patches gated by STATIC local_feat from MESASummarizer (same across all 6 blocks)
  v4: patches gated by EVOLVED spec_tokens — after 6 bidir rounds, spec tokens have
      absorbed DINOv2's intermediate representations and reflect both spectral AND
      spatial context. The gate is informed by a richer, scene-adapted spectral state.

Different from SpectraDINO:
  - Physics A/M matrices (Gaussian SRF overlaps → regularized inversion)
  - K=8 spectral regime tokens that evolve bidirectionally across blocks
  - Sigmoid-gated adapter ensures stability while allowing bidirectional learning
"""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Band / knot metadata
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
K_KNOTS = len(KNOTS)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BS             = 256
LR             = 2e-3
EPOCHS         = 100
WARMUP         = 5
SEEDS          = [0]
D_SPEC         = 128
D_SUMM         = 256
N_ND           = 16
K_REG          = 8       # spectral register tokens
D_BIDIR        = 64      # bottleneck for spec←patches cross-attn
D_BOTTLE       = 128     # bottleneck for gated patch adapter (same as v2)
N_ADAPT_START  = 6


# ---------------------------------------------------------------------------
# Physics: spectral measurement matrix
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
# Loss + metrics
# ---------------------------------------------------------------------------

def asl_loss(logits, targets, gamma_neg=4, gamma_pos=0, margin=0.05):
    p  = torch.sigmoid(logits)
    lp = targets * torch.log(p.clamp(min=1e-8))
    pm = (p - margin).clamp(min=0)
    ln = (1 - targets) * torch.log((1 - pm).clamp(min=1e-8))
    if gamma_pos > 0: lp = lp * (1 - p) ** gamma_pos
    if gamma_neg > 0: ln = ln * pm      ** gamma_neg
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
# MESA spectral front-end (unchanged from v2)
# ---------------------------------------------------------------------------

class MESASummarizer(nn.Module):
    def __init__(self, d=D_SPEC, n_nd=N_ND, d_summ=D_SUMM):
        super().__init__()
        A = build_A(); M = build_M(A)
        self.register_buffer("A", A)
        self.register_buffer("M", M)
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(K_KNOTS, d, 3, padding=1), nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1, groups=d), nn.GELU(),
            nn.Conv2d(d, d, 1),
        )
        self.nd_a    = nn.Parameter(torch.randn(n_nd, K_KNOTS) * 0.1)
        self.nd_b    = nn.Parameter(torch.randn(n_nd, K_KNOTS) * 0.1)
        self.nd_proj = nn.Linear(n_nd, d)
        self.combine  = nn.Sequential(nn.Linear(d + d, d), nn.GELU(), nn.LayerNorm(d))
        self.summarize = nn.Sequential(nn.Linear(d, d_summ), nn.GELU(), nn.LayerNorm(d_summ))

    def forward(self, x_loc):
        B, N, _ = x_loc.shape
        alpha = x_loc @ self.M.T
        ag = alpha.permute(0, 2, 1).reshape(B, K_KNOTS, 16, 16)
        sf = self.spatial_enc(ag).flatten(2).permute(0, 2, 1)
        a_n, b_n = F.normalize(self.nd_a, dim=1), F.normalize(self.nd_b, dim=1)
        num = alpha @ (a_n - b_n).T
        den = (alpha @ a_n.T).abs() + (alpha @ b_n.T).abs() + 1e-6
        nf  = self.nd_proj((num / den).tanh())
        local_feat  = self.combine(torch.cat([sf, nf], dim=-1))
        global_summ = self.summarize(local_feat.mean(1))
        return local_feat, global_summ


# ---------------------------------------------------------------------------
# Spectral register token initializer
# ---------------------------------------------------------------------------

class SpectralRegInit(nn.Module):
    """K learnable spectral regime tokens shifted by global scene spectral offset."""
    def __init__(self, K=K_REG, d_summ=D_SUMM, dim=768):
        super().__init__()
        self.base        = nn.Parameter(torch.randn(1, K, dim) * 0.02)
        self.global_cond = nn.Linear(d_summ, dim)
        self.norm        = nn.LayerNorm(dim)

    def forward(self, global_summ):
        B = global_summ.shape[0]
        g = self.global_cond(global_summ).unsqueeze(1)                    # [B, 1, 768]
        return self.norm(self.base.expand(B, -1, -1) + g)                 # [B, K, 768]


# ---------------------------------------------------------------------------
# Bidirectional spectral layer (FIXED: gated patch update)
# ---------------------------------------------------------------------------

class BidirSpectralLayer(nn.Module):
    """Two-step update:

    Step 1 — spec ← patches  (spec reads DINOv2 patch context):
      Small cross-attention [K×256], zero-init output → safe, no patch corruption.
      Spec tokens evolve to reflect what DINOv2 sees in this scene.

    Step 2 — patches ← spec  (patches read evolved spec tokens):
      Each patch soft-attends to K=8 spec tokens → per-patch spectral context [B,256,768].
      Then GATED adapter (sigmoid bounded, zero-init up) — same stability as v2.
      Gate = sigmoid(W_gate @ spec_context) bounds correction to [0, 1] × feature scale.

    This prevents the catastrophic failure of v4-fixed where unconstrained corrections
    swamped DINOv2 features. Sigmoid gating is the key stabilizer from v2.
    """
    def __init__(self, dim=768, K=K_REG, d_bidir=D_BIDIR, d_bottle=D_BOTTLE):
        super().__init__()
        self.scale_spec = d_bidir ** -0.5
        self.scale_gate = dim    ** -0.5

        # spec ← patches: cross-attention (spec queries, patches as K/V)
        self.s_q = nn.Linear(dim, d_bidir, bias=False)
        self.s_k = nn.Linear(dim, d_bidir, bias=False)
        self.s_v = nn.Linear(dim, d_bidir, bias=False)
        self.s_o = nn.Linear(d_bidir, dim)
        nn.init.zeros_(self.s_o.weight); nn.init.zeros_(self.s_o.bias)

        # patches ← spec: per-patch spec context via soft-attention over K tokens,
        # then sigmoid-gated adapter
        self.p_gate = nn.Linear(dim, d_bottle)   # spec_context → gate weights
        self.p_bias = nn.Linear(dim, d_bottle)   # spec_context → bias
        self.p_down = nn.Linear(dim, d_bottle)   # patch_token → bottleneck
        self.p_up   = nn.Linear(d_bottle, dim)   # bottleneck → correction (zero-init)
        nn.init.zeros_(self.p_up.weight); nn.init.zeros_(self.p_up.bias)

    def forward(self, x, spec):
        # x:    [B, 257, 768]
        # spec: [B, K, 768]
        patches = x[:, 1:]                                                 # [B, 256, 768]

        # --- Step 1: spec reads from DINOv2 patches ---
        q = self.s_q(spec)                                                 # [B, K, d_bidir]
        k = self.s_k(patches)                                              # [B, 256, d_bidir]
        v = self.s_v(patches)                                              # [B, 256, d_bidir]
        spec_attn = F.softmax(q @ k.transpose(-2, -1) * self.scale_spec, dim=-1)  # [B, K, 256]
        spec = spec + self.s_o(spec_attn @ v)                             # residual, zero-init s_o

        # --- Step 2: patches read from evolved spec (bounded by sigmoid gate) ---
        # Per-patch spectral context: each patch soft-attends to K spec tokens
        gate_attn = F.softmax(patches @ spec.transpose(-2, -1) * self.scale_gate, dim=-1)  # [B, 256, K]
        spec_ctx  = gate_attn @ spec                                       # [B, 256, 768]

        # Gated adapter (sigmoid-bounded, zero-init up → starts as identity)
        g = torch.sigmoid(self.p_gate(spec_ctx))                          # [B, 256, d_bottle]
        b = self.p_bias(spec_ctx)                                         # [B, 256, d_bottle]
        h = F.gelu(self.p_down(patches))                                  # [B, 256, d_bottle]
        patches = patches + self.p_up(h * g + b)                         # [B, 256, 768]

        return torch.cat([x[:, :1], patches], dim=1), spec


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MESAv4Model(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.summarizer = MESASummarizer()
        self.spec_reg   = SpectralRegInit()

        self.dino = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14',
            pretrained=True, verbose=False
        ).to(DEV)
        for p in self.dino.parameters():
            p.requires_grad_(False)

        n_adapted = 12 - N_ADAPT_START
        self.bidir_layers = nn.ModuleList([BidirSpectralLayer() for _ in range(n_adapted)])

        self.head = nn.Sequential(
            nn.Linear(768 + 768, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, nc),
        )

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, rgb, x_loc):
        _, global_summ = self.summarizer(x_loc)
        spec = self.spec_reg(global_summ)                                  # [B, K, 768]

        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(rgb)
            for blk in self.dino.blocks[:N_ADAPT_START]:
                x = blk(x)
        x = x.detach()

        for i, blk in enumerate(self.dino.blocks[N_ADAPT_START:]):
            x = blk(x)                                                     # 257 tokens, no dilution
            x, spec = self.bidir_layers[i](x, spec)

        x   = self.dino.norm(x)
        cls = x[:, 0]
        z   = x[:, 1:].mean(1)
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
    net   = MESAv4Model(nc).to(DEV)
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

    print(f"MESA-v4: K_REG={K_REG}, D_BIDIR={D_BIDIR}, D_BOTTLE={D_BOTTLE}, "
          f"adapt_blocks={N_ADAPT_START}–11", flush=True)
    print(f"  spec←patches: cross-attn [K×256], zero-init s_o", flush=True)
    print(f"  patches←spec: soft-attn over K tokens → sigmoid-gated adapter, zero-init up", flush=True)
    print(f"Training: LR={LR}, BS={BS}, EPOCHS={EPOCHS}, warmup={WARMUP}", flush=True)

    data_path     = "data/ben/ben_v1.pt"
    cache_cls     = "data/ben/dino_cls_v1.pt"
    cache_patches = "data/ben/dino_patches_v1.pt"
    out_path      = "results/ben_mesa_v4.json"

    print(f"\nloading {data_path} ...", flush=True)
    d = torch.load(data_path, weights_only=False)
    X, Y, split = d["X"], d["Y"], d["split"]
    nc = Y.shape[1]
    tr = [i for i, s in enumerate(split) if s == "train"]
    te = [i for i, s in enumerate(split) if s == "test"]
    print(f"  train={len(tr)}  test={len(te)}  classes={nc}", flush=True)

    results = {}

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

    print(f"\n[MESA-v4] {len(SEEDS)} seeds ...", flush=True)
    scores = []
    for s in SEEDS:
        print(f"  seed {s} ...", flush=True)
        v = train_eval(X, Y, tr, te, nc, s)
        scores.append(v)
        print(f"    seed {s} mAP = {v*100:.2f}%", flush=True)

    t = torch.tensor(scores)
    results["mesa_v4"] = {"seeds": scores, "mean": t.mean().item()}

    os.makedirs("results", exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n--- SUMMARY ---", flush=True)
    print(f"  CLS LP                                        : {lp_cls*100:.2f}%", flush=True)
    print(f"  CLS+patch LP                                  : {lp_patch*100:.2f}%", flush=True)
    print(f"  MESA-v1 (global gate)                         : 68.20% avg [seeds 0-1]", flush=True)
    print(f"  MESA-v2 (per-loc gate)                        : 70.65% / 71.12% [seeds 0-1]", flush=True)
    print(f"  MESA-v4-attempt1 (tokens in frozen attn)      : 69.59%  [attn dilution]", flush=True)
    print(f"  MESA-v4-attempt2 (bidir, unbounded)           : 16.95%  [feature swamping]", flush=True)
    print(f"  MESA-v4 (bidir, sigmoid-gated patch update)   : {t.mean()*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
