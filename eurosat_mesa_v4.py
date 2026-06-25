"""MESA-v4 on EuroSAT (13-band Sentinel-2, 10-class single-label).

Same bidirectional spectral register token architecture as ben_mesa_v4.py.
Key differences from BEN:
  - 13 bands (adds B10 Cirrus at 1375nm)
  - 64x64 images
  - 10-class single-label (CrossEntropy)
  - Accuracy metric
"""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sat_ms_headroom import ensure_unzipped, load_all

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Band metadata (13 bands: B01-B09, B8A, B10, B11, B12)
# ---------------------------------------------------------------------------
RGB_IDX = [3, 2, 1]
IMN_M   = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMN_S   = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

WL_C = [443., 490., 560., 665., 705., 740., 783., 842., 865., 945., 1375., 1610., 2190.]
FWHM = [ 20.,  65.,  35.,  30.,  15.,  15.,  20., 115.,  20.,  20.,   30.,   90.,  180.]
KNOTS = torch.tensor([
    420., 460., 510., 560., 610., 650., 680., 720.,
    760., 800., 870., 950., 1200., 1400., 1620., 2200.
])
K_KNOTS = len(KNOTS)
N_BANDS = 13

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BS             = 256
LR             = 2e-3
EPOCHS         = 100
WARMUP         = 5
SEEDS          = [0, 1, 2]
D_SPEC         = 128
D_SUMM         = 256
N_ND           = 16
K_REG          = 8
D_BIDIR        = 64
D_BOTTLE       = 128
N_ADAPT_START  = 6
TRAIN_FRAC     = 0.8


# ---------------------------------------------------------------------------
# Physics
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
# MESASummarizer (adapted for 13 bands)
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
# SpectralRegInit + BidirSpectralLayer (identical to ben_mesa_v4.py)
# ---------------------------------------------------------------------------

class SpectralRegInit(nn.Module):
    def __init__(self, K=K_REG, d_summ=D_SUMM, dim=768):
        super().__init__()
        self.base        = nn.Parameter(torch.randn(1, K, dim) * 0.02)
        self.global_cond = nn.Linear(d_summ, dim)
        self.norm        = nn.LayerNorm(dim)

    def forward(self, global_summ):
        B = global_summ.shape[0]
        g = self.global_cond(global_summ).unsqueeze(1)
        return self.norm(self.base.expand(B, -1, -1) + g)


class BidirSpectralLayer(nn.Module):
    def __init__(self, dim=768, K=K_REG, d_bidir=D_BIDIR, d_bottle=D_BOTTLE):
        super().__init__()
        self.scale_spec = d_bidir ** -0.5
        self.scale_gate = dim    ** -0.5
        self.s_q = nn.Linear(dim, d_bidir, bias=False)
        self.s_k = nn.Linear(dim, d_bidir, bias=False)
        self.s_v = nn.Linear(dim, d_bidir, bias=False)
        self.s_o = nn.Linear(d_bidir, dim)
        nn.init.zeros_(self.s_o.weight); nn.init.zeros_(self.s_o.bias)
        self.p_gate = nn.Linear(dim, d_bottle)
        self.p_down = nn.Linear(dim, d_bottle)
        self.p_up   = nn.Linear(d_bottle, dim)
        nn.init.zeros_(self.p_up.weight); nn.init.zeros_(self.p_up.bias)

    def forward(self, x, spec):
        patches = x[:, 1:]
        q = self.s_q(spec); k = self.s_k(patches); v = self.s_v(patches)
        spec_attn = F.softmax(q @ k.transpose(-2, -1) * self.scale_spec, dim=-1)
        spec = spec + self.s_o(spec_attn @ v)
        gate_attn = F.softmax(patches @ spec.transpose(-2, -1) * self.scale_gate, dim=-1)
        spec_ctx  = gate_attn @ spec
        g = torch.sigmoid(self.p_gate(spec_ctx))
        h = F.gelu(self.p_down(patches))
        patches = patches + self.p_up(h * g)
        return torch.cat([x[:, :1], patches], dim=1), spec


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MESAv4EuroSAT(nn.Module):
    def __init__(self, nc=10):
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
        spec = self.spec_reg(global_summ)
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(rgb)
            for blk in self.dino.blocks[:N_ADAPT_START]:
                x = blk(x)
        x = x.detach()
        for i, blk in enumerate(self.dino.blocks[N_ADAPT_START:]):
            x = blk(x)
            x, spec = self.bidir_layers[i](x, spec)
        x   = self.dino.norm(x)
        cls = x[:, 0]
        z   = x[:, 1:].mean(1)
        return self.head(torch.cat([cls, z], dim=1))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EuroSATDataset(Dataset):
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
        x_loc = (x16 / 3000.0).clamp(0, 1).permute(1, 2, 0).reshape(256, N_BANDS)

        if self.train:
            if torch.rand(1).item() < 0.5:
                rgb   = rgb.flip(-1)
                x_loc = x_loc.view(16, 16, N_BANDS).flip(1).reshape(256, N_BANDS)
            if torch.rand(1).item() < 0.5:
                rgb   = rgb.flip(-2)
                x_loc = x_loc.view(16, 16, N_BANDS).flip(0).reshape(256, N_BANDS)

        return rgb, x_loc, self.Y[j]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cosine_with_warmup(opt, warmup, total):
    def fn(ep):
        if ep < warmup: return ep / max(warmup, 1)
        p = (ep - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def train_eval(X, Y, tr, te, seed):
    torch.manual_seed(seed)
    net   = MESAv4EuroSAT(nc=10).to(DEV)
    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=LR, weight_decay=1e-4
    )
    sched = cosine_with_warmup(opt, WARMUP, EPOCHS)

    trl = DataLoader(
        EuroSATDataset(X, Y, tr, train=True),
        batch_size=BS, shuffle=True, num_workers=0, drop_last=True,
    )
    for ep in range(EPOCHS):
        net.train()
        for rgb, x_loc, y in trl:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net(rgb.to(DEV), x_loc.to(DEV)), y.to(DEV))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{EPOCHS}", flush=True)

    net.eval()
    tel = DataLoader(
        EuroSATDataset(X, Y, te, train=False),
        batch_size=BS * 2, shuffle=False, num_workers=0,
    )
    correct = total = 0
    with torch.no_grad():
        for rgb, x_loc, y in tel:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(rgb.to(DEV), x_loc.to(DEV))
            correct += (logits.argmax(1).cpu() == y).sum().item()
            total   += len(y)
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("high")

    print(f"MESA-v4 EuroSAT: K_REG={K_REG}, D_BIDIR={D_BIDIR}, D_BOTTLE={D_BOTTLE}, "
          f"adapt_blocks={N_ADAPT_START}-11, bands={N_BANDS}", flush=True)
    print(f"Training: LR={LR}, BS={BS}, EPOCHS={EPOCHS}, seeds={SEEDS}", flush=True)

    tifs = ensure_unzipped()
    print(f"loading {len(tifs)} tifs ...", flush=True)
    X, Y, classes = load_all(tifs)
    print(f"  X={tuple(X.shape)}  classes={classes}", flush=True)

    N = len(X)
    torch.manual_seed(0)
    perm = torch.randperm(N).tolist()
    n_tr = int(N * TRAIN_FRAC)
    tr, te = perm[:n_tr], perm[n_tr:]
    print(f"  train={len(tr)}  test={len(te)}", flush=True)

    results = {"seeds": [], "mean": 0.0}
    for s in SEEDS:
        print(f"\n  seed {s} ...", flush=True)
        acc = train_eval(X, Y, tr, te, s)
        results["seeds"].append(acc)
        print(f"    seed {s} acc = {acc*100:.2f}%", flush=True)

    t = torch.tensor(results["seeds"])
    results["mean"] = t.mean().item()
    results["std"]  = t.std().item()

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/eurosat_mesa_v4.json", "w"), indent=2)

    print("\n--- SUMMARY ---", flush=True)
    print(f"  RGB-DINOv2 LP (prior)            : ~96%", flush=True)
    print(f"  MESA-v4 EuroSAT ({len(SEEDS)} seeds)       : "
          f"{t.mean()*100:.2f}% ± {t.std()*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
