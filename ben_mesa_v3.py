"""MESA-v3: Spectral Cross-Attention Adapters for BigEarthNet.

Architecture change over v2 (per-loc gating → 70.65% seed 0):

  SpectralCrossAttnAdapter:
    Query  = DINOv2 patch token (after frozen block) — what the backbone SEES
    Key, V = canonical spectral field (all 256 locations) — what the sensors MEASURE

    Each patch token can pull spectral information from ANY spatial location,
    not just its own position. A forest patch at a field boundary attends to
    neighboring land pixels' SWIR absorption. A water patch attends to
    spectrally-similar water patches elsewhere in the scene.

    This is the correct asymmetry: DINOv2 RGB representations are the consumers
    of spectral information (Q), not the producers. Spectral features are the
    providers (K, V). Contrast with SpectraDINO which likely inserts spectral
    tokens as additional inputs (on the Q side).

  AttentionPool head:
    Replace patch mean-pooling with a learned attention query over 256 patch tokens.
    The model learns which spatial locations are most discriminative per scene.

Physics front-end (unchanged from v2):
  - Measurement matrix A: Gaussian SRF × Gaussian basis overlaps (S2 physics)
  - Regularized inverse M = (A^T A + λ D^T D)^{-1} A^T  (Tikhonov)
  - K=16 canonical spectral knots covering 420–2200 nm
  - Learned ND ratio features: n_nd=16 learned (a_n, b_n) pairs in spectral space

Unique vs. SpectraDINO:
  1. Physics-derived A/M (not learned band tokenizer)
  2. ND ratio features in canonical reflectance space
  3. Spectral serves as K/V only (DINOv2 patch tokens are Q)
  4. Mid-level injection inside frozen DINOv2 blocks (not prefix/suffix)
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
SEEDS          = [0]            # single seed for architecture comparison
D_SPEC         = 128    # local spectral feature dim (MESASummarizer output)
D_SUMM         = 256    # global spectral summary for CLS
N_HEADS        = 4      # cross-attention heads in adapter
D_KV           = 64     # key-value dim in cross-attention (16 per head)
ADAPTER_BLOCKS = list(range(6, 12))   # last 6 of 12 blocks; detach after block 5
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
# MESA spectral front-end
# ---------------------------------------------------------------------------

class MESASummarizer(nn.Module):
    """S2 bands → canonical spectral coefficients → local + global features.

    local_feat  [B, 256, D_SPEC]  per-location spectral features (K/V in adapter)
    global_summ [B, D_SUMM]       global summary for CLS token correction
    """
    def __init__(self, d=D_SPEC, n_nd=N_ND, d_summ=D_SUMM):
        super().__init__()
        A = build_A(); M = build_M(A)
        self.register_buffer("A", A)
        self.register_buffer("M", M)

        self.spatial_enc = nn.Sequential(
            nn.Conv2d(K, d, 3, padding=1), nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1, groups=d), nn.GELU(),
            nn.Conv2d(d, d, 1),
        )
        self.nd_a    = nn.Parameter(torch.randn(n_nd, K) * 0.1)
        self.nd_b    = nn.Parameter(torch.randn(n_nd, K) * 0.1)
        self.nd_proj = nn.Linear(n_nd, d)
        self.combine  = nn.Sequential(nn.Linear(d + d, d), nn.GELU(), nn.LayerNorm(d))
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
# Spectral cross-attention adapter
# ---------------------------------------------------------------------------

class SpectralCrossAttnAdapter(nn.Module):
    """Mid-level adapter: patch tokens (Q) attend to spectral field (K, V).

    Direction: RGB features consume spectral information, not vice versa.
    Flash attention via scaled_dot_product_attention — no explicit attention matrix.
    Zero-init on output projections: model starts as frozen DINOv2.
    """
    def __init__(self, dim=768, d_spec=D_SPEC, d_summ=D_SUMM,
                 n_heads=N_HEADS, d_kv=D_KV):
        super().__init__()
        assert d_kv % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_kv // n_heads

        self.q_proj  = nn.Linear(dim, d_kv)
        self.k_proj  = nn.Linear(d_spec, d_kv)
        self.v_proj  = nn.Linear(d_spec, d_kv)
        self.out     = nn.Linear(d_kv, dim)

        # CLS token: direct correction from global spectral summary
        self.cls_proj = nn.Linear(d_summ, dim)

        nn.init.zeros_(self.out.weight);  nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.cls_proj.weight); nn.init.zeros_(self.cls_proj.bias)

    def forward(self, x, local_feat, global_summ):
        # x:           [B, 257, 768]
        # local_feat:  [B, 256, D_SPEC]  — spectral keys/values
        # global_summ: [B, D_SUMM]       — CLS correction
        B = x.shape[0]
        H = self.n_heads; dh = self.d_head

        # Patch tokens as queries; spectral field as keys + values
        q = self.q_proj(x[:, 1:]).reshape(B, 256, H, dh).transpose(1, 2)   # [B,H,256,dh]
        k = self.k_proj(local_feat).reshape(B, 256, H, dh).transpose(1, 2) # [B,H,256,dh]
        v = self.v_proj(local_feat).reshape(B, 256, H, dh).transpose(1, 2) # [B,H,256,dh]

        # Flash attention — avoids materialising full [B,H,256,256] matrix
        out = F.scaled_dot_product_attention(q, k, v)                       # [B,H,256,dh]
        out = out.transpose(1, 2).reshape(B, 256, H * dh)
        patch_correction = self.out(out)                                     # [B,256,768]

        # CLS: zero-init projection from global spectral summary
        cls_correction = self.cls_proj(global_summ).unsqueeze(1)            # [B,1,768]

        return x + torch.cat([cls_correction, patch_correction], dim=1)


# ---------------------------------------------------------------------------
# Attention pooling head
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """Single learned query attending over 256 patch tokens.
    Replaces mean pooling — lets model focus on discriminative locations.
    """
    def __init__(self, dim=768):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, patches):  # [B, 256, 768]
        q = self.q.expand(patches.shape[0], -1, -1)           # [B, 1, 768]
        attn = F.softmax(q @ patches.transpose(-2, -1) * self.scale, dim=-1)  # [B,1,256]
        return (attn @ patches).squeeze(1)                     # [B, 768]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MESAv3Model(nn.Module):
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
            SpectralCrossAttnAdapter() for _ in ADAPTER_BLOCKS
        ])

        self.attn_pool = AttentionPool()

        # Head: CLS + attention-pooled patches
        self.head = nn.Sequential(
            nn.Linear(768 + 768, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, nc),
        )

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, rgb, x_loc):
        local_feat, global_summ = self.summarizer(x_loc)

        n_frozen = ADAPTER_BLOCKS[0]
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(rgb)
            for blk in self.dino.blocks[:n_frozen]:
                x = blk(x)

        x = x.detach()

        for i, blk in enumerate(self.dino.blocks[n_frozen:]):
            x = blk(x)
            x = self.adapters[i](x, local_feat, global_summ)

        x   = self.dino.norm(x)
        cls    = x[:, 0]                      # [B, 768]
        pooled = self.attn_pool(x[:, 1:])     # [B, 768]  learned attention pool
        return self.head(torch.cat([cls, pooled], dim=1))


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
            # Spectral jitter: per-band scale [0.9,1.1] + small additive noise
            # Simulates sensor calibration uncertainty & atmospheric correction errors
            if torch.rand(1).item() < 0.5:
                scale = torch.empty(12).uniform_(0.9, 1.1)
                x_loc = (x_loc * scale + torch.randn(256, 12) * 0.01).clamp(0, 1)

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
    net   = MESAv3Model(nc).to(DEV)
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

    print(f"MESA-v3: K={K}, D_SPEC={D_SPEC}, D_SUMM={D_SUMM}, "
          f"N_HEADS={N_HEADS}, D_KV={D_KV}, blocks={ADAPTER_BLOCKS}", flush=True)
    print(f"  adapter: SpectralCrossAttnAdapter (Q=patch, K/V=spectral field)", flush=True)
    print(f"  head: AttentionPool over 256 patch tokens", flush=True)
    print(f"Training: LR={LR}, BS={BS}, EPOCHS={EPOCHS}, warmup={WARMUP}", flush=True)

    data_path     = "data/ben/ben_v1.pt"
    cache_cls     = "data/ben/dino_cls_v1.pt"
    cache_patches = "data/ben/dino_patches_v1.pt"
    out_path      = "results/ben_mesa_v3.json"

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

    print(f"\n[MESA-v3] {len(SEEDS)} seeds ...", flush=True)
    scores = []
    for s in SEEDS:
        print(f"  seed {s} ...", flush=True)
        v = train_eval(X, Y, tr, te, nc, s)
        scores.append(v)
        print(f"    seed {s} mAP = {v*100:.2f}%", flush=True)

    t = torch.tensor(scores)
    results["mesa_v3"] = {"seeds": scores, "mean": t.mean().item(), "std": t.std().item()}

    os.makedirs("results", exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n--- SUMMARY ---", flush=True)
    print(f"  CLS LP                                   : {lp_cls*100:.2f}%", flush=True)
    print(f"  CLS + patch-pool LP                      : {lp_patch*100:.2f}%", flush=True)
    print(f"  MESA-v1 (global gate, blocks 9-11)       : 68.20% avg  [seeds 0-1]", flush=True)
    print(f"  MESA-v2 (per-loc gate, blocks 6-11)      : 70.65% seed 0", flush=True)
    print(f"  MESA-v3 (spectral XA, attn-pool)         : "
          f"{t.mean()*100:.2f} ± {t.std()*100:.2f}%", flush=True)
    print(f"\n  Paper baselines (RS-pretrained + full FT):", flush=True)
    print(f"    SatMAE++ ViT-L    : 85.10%", flush=True)
    print(f"    SMARTIES ViT-B    : 86.90%", flush=True)
    print(f"    CROMA    ViT-B    : 87.60%", flush=True)


if __name__ == "__main__":
    main()
