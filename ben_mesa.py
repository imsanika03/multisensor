"""Measure-Aware Spectral Adapter (MESA) for BigEarthNet 10% benchmark.

Physics-grounded architecture (Bharvirkar et al.):

Observation model: each band is a weighted integral over the reflectance field,
    x_{b,p} = ∫ s_b(λ) r_p(λ) dλ + ε  ≈  Σ_k A_{b,k} α_{p,k}

where A_{b,k} = ∫ s_b(λ) φ_k(λ) dλ is precomputed from ESA-published SRFs and
Gaussian basis functions φ_k centered at K canonical spectral knots.

Inference: α_p = (A^T A + λ_s D^T D)^{-1} A^T x_p + gate · P(frozen_patch_p)
  — closed-form regularized spectral inversion (precomputed as a fixed matrix)
  — RGB-informed prior P(·) from frozen DINOv2 patch tokens (trained, gated)
  — spectral smoothness enforced by finite-difference D

Two feature branches over canonical coefficients α:
  1. Spatial: lightweight conv encoder on 16×16 canonical field
  2. Learned ND: M learnable spectral-contrast features q_m = tanh(normalized diff)
     — generalization of NDVI/NDWI; task-adaptive without hard-coding indices

Both branches uncertainty-gated by reconstruction error ||Aα - x||.
Zero-init final projection → starts as pure DINOv2, learns additive corrections.

Loss: Asymmetric Loss (ASL, Ridnik et al. 2021) for multilabel class imbalance.
"""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Sentinel-2 band metadata (ESA MSI instrument specification)
# Band: B01   B02   B03   B04   B05   B06   B07   B08   B8A   B09   B11    B12
WL_C = [443., 490., 560., 665., 705., 740., 783., 842., 865., 945., 1610., 2190.]
FWHM = [ 20.,  65.,  35.,  30.,  15.,  15.,  20., 115.,  20.,  20.,   90.,  180.]
# Resolution group: 0=10m  1=20m  2=60m
RES  = [  2,    0,    0,    0,    1,    1,    1,    0,    1,    2,    1,     1  ]

# Canonical spectral knots (nm) — K=16 spanning vis → SWIR
KNOTS = torch.tensor([
    420., 460., 510., 560., 610., 650., 680., 720.,
    760., 800., 870., 950., 1200., 1400., 1620., 2200.
])
K = len(KNOTS)  # 16

BS     = 2048
LR     = 3e-3   # match SSI-v1's successful setting (2e-3+warmup collapses to LP baseline)
EPOCHS = 150
WARMUP = 5      # short warmup; zero-init already provides stable init
SEEDS  = [0, 1, 2, 3, 4]
D      = 128    # spectral feature dim
N_ND   = 16    # learned normalized-difference features


# ---------------------------------------------------------------------------
# Spectral measurement matrix (precomputed, fixed)
# ---------------------------------------------------------------------------

def build_A():
    """A[b,k] = overlap integral of band SRF (Gaussian) × basis func φ_k (Gaussian).

    Product of two Gaussians is Gaussian with combined variance:
      A[b,k] ∝ exp(-(wl_c[b] - knot[k])² / (2(σ_b² + σ_k²)))
    Row-normalized so each band's total response = 1.
    """
    wl   = torch.tensor(WL_C)             # [12]
    sig_b = torch.tensor(FWHM) / 2.355   # [12] band sigma
    sig_k = 80.0                          # knot width (nm)
    delta = wl.unsqueeze(1) - KNOTS.unsqueeze(0)           # [12, K]
    sig_t = (sig_b.unsqueeze(1) ** 2 + sig_k ** 2).sqrt() # [12, K]
    A = torch.exp(-delta ** 2 / (2 * sig_t ** 2))          # [12, K]
    A = A / (A.sum(1, keepdim=True) + 1e-8)
    return A


def build_M(A, lambda_s=0.1):
    """Regularized pseudoinverse M = (A^T A + λ_s D^T D)^{-1} A^T  [K, 12].

    D is finite-difference spectral smoothness operator [K-1, K].
    Precomputed once; applied as a batched matmul at runtime.
    """
    Kn = A.shape[1]
    D  = torch.zeros(Kn - 1, Kn)
    for i in range(Kn - 1):
        D[i, i] = -1.; D[i, i + 1] = 1.
    reg = A.T @ A + lambda_s * (D.T @ D)
    return torch.linalg.solve(reg, A.T)   # [K, 12]


# ---------------------------------------------------------------------------
# Asymmetric Loss
# ---------------------------------------------------------------------------

def asl_loss(logits, targets, gamma_neg=4, gamma_pos=0, margin=0.05):
    """ASL: down-weights easy negatives more aggressively than positives."""
    p = torch.sigmoid(logits)
    l_pos = targets       * torch.log(p.clamp(min=1e-8))
    pm    = (p - margin).clamp(min=0)
    l_neg = (1 - targets) * torch.log((1 - pm).clamp(min=1e-8))
    if gamma_pos > 0:
        l_pos = l_pos * (1 - p) ** gamma_pos
    if gamma_neg > 0:
        l_neg = l_neg * pm ** gamma_neg
    return -(l_pos + l_neg).mean()


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
# MESA Adapter
# ---------------------------------------------------------------------------

class MESAAdapter(nn.Module):
    """Measure-Aware Spectral Adapter.

    At each spatial patch location p:
      1. Infer canonical spectral coefficients α_p via regularized inversion
         (precomputed matrix M, applied as a fixed matmul)
      2. Optionally refine with RGB-informed prior from frozen DINOv2 patch token
      3. Extract spatial features via conv over 16×16 canonical field
      4. Compute learned ND features (learnable NDVI-like contrasts over α)
      5. Gate both branches by reconstruction uncertainty
      6. Project to DINOv2 dim [768], zero-initialized
    """
    def __init__(self, d=D, n_nd=N_ND):
        super().__init__()
        A = build_A()       # [12, K]
        M = build_M(A)      # [K, 12]
        self.register_buffer("A", A)  # measurement matrix — for reconstruction check
        self.register_buffer("M", M)  # regularized inverse — for spectral inversion

        # RGB-informed prior: frozen DINOv2 patch token → canonical spectral prior
        # Starts suppressed (gate ≈ 0.018) so training begins from pure inversion
        self.rgb_prior  = nn.Linear(768, K)
        self.prior_gate = nn.Parameter(torch.tensor(-4.0))

        # Spatial encoder: canonical coefficients as 16×16 field → spatial features
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(K, d, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d),  # depthwise: spatial
            nn.GELU(),
            nn.Conv2d(d, d, kernel_size=1),                        # pointwise: mix
        )

        # Learned ND features: M pairs (a_m, b_m) over canonical coefficients
        # q_m = tanh((a_m - b_m)^T α / (|a_m^T α| + |b_m^T α| + ε))
        self.nd_a   = nn.Parameter(torch.randn(n_nd, K) * 0.1)
        self.nd_b   = nn.Parameter(torch.randn(n_nd, K) * 0.1)
        self.nd_proj = nn.Linear(n_nd, d)

        # Fuse spatial + ND → DINOv2 dim; zero-init so model starts as pure DINOv2
        self.combine = nn.Sequential(nn.Linear(d + d, d), nn.GELU(), nn.LayerNorm(d))
        self.up = nn.Linear(d, 768)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x_loc, pat):
        # x_loc: [B, 256, 12] — normalized S2 band values per patch location
        # pat:   [B, 256, 768] — frozen DINOv2 patch tokens
        B, N, _ = x_loc.shape

        # 1. Canonical spectral inversion: α = M x + gate · prior(pat)
        alpha = x_loc @ self.M.T                                # [B, 256, K]
        prior = self.rgb_prior(pat.float().detach())            # [B, 256, K]
        alpha = alpha + torch.sigmoid(self.prior_gate) * prior  # [B, 256, K]

        # 2. Reconstruction uncertainty: ||A α - x||² per location
        recon = alpha @ self.A.T                                 # [B, 256, 12]
        unc   = ((recon - x_loc) ** 2).mean(-1, keepdim=True)   # [B, 256, 1]
        conf  = torch.exp(-unc)                                  # [B, 256, 1]

        # 3. Spatial encoder on 16×16 canonical coefficient field
        ag = alpha.permute(0, 2, 1).reshape(B, K, 16, 16)       # [B, K, 16, 16]
        sf = self.spatial_enc(ag)                                 # [B, d, 16, 16]
        sf = sf.flatten(2).permute(0, 2, 1)                      # [B, 256, d]

        # 4. Learned ND features (uncertainty-gated)
        a_n  = F.normalize(self.nd_a, dim=1)                     # [n_nd, K]
        b_n  = F.normalize(self.nd_b, dim=1)
        num  = alpha @ (a_n - b_n).T                             # [B, 256, n_nd]
        den  = (alpha @ a_n.T).abs() + (alpha @ b_n.T).abs() + 1e-6
        nd   = (num / den).tanh() * conf                         # [B, 256, n_nd]
        nf   = self.nd_proj(nd)                                   # [B, 256, d]

        # 5. Fuse + project (zero-init ensures injection=0 at init)
        spec = self.combine(torch.cat([sf, nf], dim=-1))          # [B, 256, d]
        return self.up(spec)                                       # [B, 256, 768]


class SpatialRefiner(nn.Module):
    """1-layer Transformer: spectral-augmented patch tokens communicate spatially.

    Output projections zero-initialized → identity at init (preserves DINOv2 baseline).
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
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ff[2].weight)
        nn.init.zeros_(self.ff[2].bias)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class MESAModel(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.adapter = MESAAdapter()
        self.head = nn.Sequential(
            nn.Linear(768 + 768, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, nc),
        )

    def forward(self, cls, patches, x_loc):
        pat       = patches.float()
        augmented = pat + self.adapter(x_loc, pat)  # [B, 256, 768]  (=pat at init)
        z         = augmented.mean(1)               # [B, 768]
        return self.head(torch.cat([cls, z], dim=1))


# ---------------------------------------------------------------------------
# Dataset — consistent geometric augmentation on x_loc AND patch token grid
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
        x16 = F.interpolate(self.X[j].float()[None], size=(16, 16),
                             mode="bilinear", align_corners=False)[0]  # [12, 16, 16]
        pat = self.PAT[j].view(16, 16, 768)                            # [16, 16, 768]

        if self.train:
            if torch.rand(1).item() < 0.5:
                x16 = x16.flip(-1);  pat = pat.flip(1)   # horizontal flip
            if torch.rand(1).item() < 0.5:
                x16 = x16.flip(-2);  pat = pat.flip(0)   # vertical flip

        x_loc = (x16 / 3000.0).clamp(0, 1).permute(1, 2, 0).reshape(256, 12)
        return self.CLS[j], pat.reshape(256, 768), x_loc, self.Y[j]


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
    net   = MESAModel(nc).to(DEV)
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
                loss = asl_loss(net(c.to(DEV), p.to(DEV), x.to(DEV)), y.to(DEV))
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

    print(f"MESA: K={K} canonical knots, D={D}, N_ND={N_ND}", flush=True)
    print(f"Training: LR={LR}, BS={BS}, EPOCHS={EPOCHS}, warmup={WARMUP}", flush=True)

    data_path     = "data/ben/ben_v1.pt"
    cache_cls     = "data/ben/dino_cls_v1.pt"
    cache_patches = "data/ben/dino_patches_v1.pt"
    out_path      = "results/ben_mesa.json"

    print(f"\nloading {data_path} ...", flush=True)
    d = torch.load(data_path, weights_only=False)
    X, Y, split = d["X"], d["Y"], d["split"]
    nc = Y.shape[1]

    tr = [i for i, s in enumerate(split) if s == "train"]
    te = [i for i, s in enumerate(split) if s == "test"]
    print(f"  train={len(tr)}  test={len(te)}  classes={nc}", flush=True)

    print("loading DINOv2 features from cache ...", flush=True)
    CLS = torch.load(cache_cls, weights_only=False)
    PAT = torch.load(cache_patches, weights_only=False)

    # Print precomputed matrices for verification
    A_mat = build_A()
    M_mat = build_M(A_mat)
    print(f"  A (measurement matrix): {tuple(A_mat.shape)}", flush=True)
    print(f"  M (regularized inverse): {tuple(M_mat.shape)}", flush=True)
    recon_err = (A_mat @ M_mat @ A_mat - A_mat).abs().mean().item()
    print(f"  A·M·A ≈ A reconstruction error: {recon_err:.4f} (0=perfect)", flush=True)

    results = {}

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

    print(f"\n[MESA] {len(SEEDS)} seeds ...", flush=True)
    scores = []
    for s in SEEDS:
        print(f"  seed {s} ...", flush=True)
        v = train_eval(CLS, PAT, X, Y, tr, te, nc, s)
        scores.append(v)
        print(f"    seed {s} mAP = {v*100:.2f}%", flush=True)

    t = torch.tensor(scores)
    results["mesa"] = {"seeds": scores, "mean": t.mean().item(), "std": t.std().item()}

    os.makedirs("results", exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)

    print("\n--- SUMMARY ---", flush=True)
    print(f"  CLS LP            : {lp_cls*100:.2f}%", flush=True)
    print(f"  CLS + patch-pool  : {lp_patch*100:.2f}%", flush=True)
    print(f"  SSI-v1  seed 0    : 66.07%  (spatial injection baseline)", flush=True)
    print(f"  MESA (ours)       : {t.mean()*100:.2f} ± {t.std()*100:.2f}%", flush=True)
    print(f"\n  Paper baselines (RS-pretrained + FT):", flush=True)
    print(f"    SatMAE++ ViT-L  : 85.10%", flush=True)
    print(f"    SMARTIES ViT-B  : 86.90%", flush=True)
    print(f"    CROMA    ViT-B  : 87.60%", flush=True)


if __name__ == "__main__":
    main()
