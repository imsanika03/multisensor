"""EuroSAT evaluation: LP vs FT for MESA-v4.

LP  — Freeze DINOv2 backbone + spectral adapter (random init, zero-patch-update),
      train only a single linear head. Since p_up / s_o are zero-init the adapter
      contributes nothing at init, so LP measures DINOv2 feature quality.

FT  — Freeze DINOv2, train spectral adapter + MLP head (full MESA-v4).

Both use the same train/test split (80/20, seed=0 permutation).
Reports top-1 accuracy for each protocol across SEEDS.
"""
import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sat_ms_headroom import ensure_unzipped, load_all

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Band metadata (13 bands)
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
NC             = 10


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
# Architecture (same as ben_mesa_v4, adapted for 13 bands)
# ---------------------------------------------------------------------------
class MESASummarizer(nn.Module):
    def __init__(self):
        super().__init__()
        A = build_A(); M = build_M(A)
        self.register_buffer("A", A); self.register_buffer("M", M)
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(K_KNOTS, D_SPEC, 3, padding=1), nn.GELU(),
            nn.Conv2d(D_SPEC, D_SPEC, 3, padding=1, groups=D_SPEC), nn.GELU(),
            nn.Conv2d(D_SPEC, D_SPEC, 1),
        )
        self.nd_a = nn.Parameter(torch.randn(N_ND, K_KNOTS) * 0.1)
        self.nd_b = nn.Parameter(torch.randn(N_ND, K_KNOTS) * 0.1)
        self.nd_proj = nn.Linear(N_ND, D_SPEC)
        self.combine   = nn.Sequential(nn.Linear(D_SPEC + D_SPEC, D_SPEC), nn.GELU(), nn.LayerNorm(D_SPEC))
        self.summarize = nn.Sequential(nn.Linear(D_SPEC, D_SUMM), nn.GELU(), nn.LayerNorm(D_SUMM))

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


class SpectralRegInit(nn.Module):
    def __init__(self):
        super().__init__()
        self.base        = nn.Parameter(torch.randn(1, K_REG, 768) * 0.02)
        self.global_cond = nn.Linear(D_SUMM, 768)
        self.norm        = nn.LayerNorm(768)

    def forward(self, global_summ):
        B = global_summ.shape[0]
        g = self.global_cond(global_summ).unsqueeze(1)
        return self.norm(self.base.expand(B, -1, -1) + g)


class BidirSpectralLayer(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.scale_spec = D_BIDIR ** -0.5
        self.scale_gate = dim    ** -0.5
        self.s_q = nn.Linear(dim, D_BIDIR, bias=False)
        self.s_k = nn.Linear(dim, D_BIDIR, bias=False)
        self.s_v = nn.Linear(dim, D_BIDIR, bias=False)
        self.s_o = nn.Linear(D_BIDIR, dim)
        nn.init.zeros_(self.s_o.weight); nn.init.zeros_(self.s_o.bias)
        self.p_gate = nn.Linear(dim, D_BOTTLE)
        self.p_down = nn.Linear(dim, D_BOTTLE)
        self.p_up   = nn.Linear(D_BOTTLE, dim)
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


class MESAv4EuroSAT(nn.Module):
    def __init__(self, linear_head=False):
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
        if linear_head:
            self.head = nn.Linear(768 + 768, NC)
        else:
            self.head = nn.Sequential(
                nn.Linear(768 + 768, 512), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(512, NC),
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


def run_one(X, Y, tr, te, seed, freeze_adapter):
    """
    freeze_adapter=True  → LP: adapter params frozen (all zero at init), linear head
    freeze_adapter=False → FT: adapter params trained, MLP head
    """
    torch.manual_seed(seed)
    net = MESAv4EuroSAT(linear_head=freeze_adapter).to(DEV)

    if freeze_adapter:
        # Freeze adapter weights — DINOv2 already frozen, also freeze spectral layers
        for p in net.summarizer.parameters():  p.requires_grad_(False)
        for p in net.spec_reg.parameters():    p.requires_grad_(False)
        for p in net.bidir_layers.parameters(): p.requires_grad_(False)

    trainable = list(filter(lambda p: p.requires_grad, net.parameters()))
    opt   = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    sched = cosine_with_warmup(opt, WARMUP, EPOCHS)

    trl = DataLoader(
        EuroSATDataset(X, Y, tr, train=True),
        batch_size=BS, shuffle=True, num_workers=0, drop_last=True,
    )
    tel = DataLoader(
        EuroSATDataset(X, Y, te, train=False),
        batch_size=BS * 2, shuffle=False, num_workers=0,
    )

    def eval_acc():
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for rgb, x_loc, y in tel:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = net(rgb.to(DEV), x_loc.to(DEV))
                correct += (logits.argmax(1).cpu() == y).sum().item()
                total   += len(y)
        net.train()
        return correct / total

    for ep in range(EPOCHS):
        net.train()
        ep_loss = 0.0
        n_batches = 0
        for rgb, x_loc, y in trl:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net(rgb.to(DEV), x_loc.to(DEV)), y.to(DEV))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            n_batches += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            acc = eval_acc()
            print(f"      epoch {ep+1}/{EPOCHS}  loss={ep_loss/n_batches:.4f}  acc={acc*100:.2f}%", flush=True)

    return eval_acc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.set_float32_matmul_precision("high")
    print(f"EuroSAT LP vs FT  |  seeds={SEEDS}  BS={BS}  LR={LR}  EPOCHS={EPOCHS}", flush=True)

    tifs = ensure_unzipped()
    print(f"loading {len(tifs)} tifs ...", flush=True)
    X, Y, classes = load_all(tifs)
    print(f"  X={tuple(X.shape)}  classes={classes}", flush=True)

    torch.manual_seed(0)
    perm  = torch.randperm(len(X)).tolist()
    n_tr  = int(len(X) * TRAIN_FRAC)
    tr, te = perm[:n_tr], perm[n_tr:]
    print(f"  train={len(tr)}  test={len(te)}", flush=True)

    lp_accs, ft_accs = [], []

    for s in SEEDS:
        print(f"\n  === seed {s} ===", flush=True)

        print(f"  [LP] frozen adapter + linear head ...", flush=True)
        lp = run_one(X, Y, tr, te, s, freeze_adapter=True)
        lp_accs.append(lp)
        print(f"    LP seed {s}: {lp*100:.2f}%", flush=True)

        print(f"  [FT] frozen DINOv2, train adapter + MLP head ...", flush=True)
        ft = run_one(X, Y, tr, te, s, freeze_adapter=False)
        ft_accs.append(ft)
        print(f"    FT seed {s}: {ft*100:.2f}%", flush=True)

    lp_t = torch.tensor(lp_accs)
    ft_t = torch.tensor(ft_accs)

    results = {
        "LP": {"seeds": lp_accs, "mean": lp_t.mean().item(), "std": lp_t.std().item()},
        "FT": {"seeds": ft_accs, "mean": ft_t.mean().item(), "std": ft_t.std().item()},
    }
    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/eurosat_lp_ft.json", "w"), indent=2)

    print("\n=== EUROSAT RESULTS ===", flush=True)
    print(f"  LP (frozen DINOv2 + adapter, linear head) : {lp_t.mean()*100:.2f}% ± {lp_t.std()*100:.2f}%", flush=True)
    print(f"  FT (frozen DINOv2, adapter + MLP head)    : {ft_t.mean()*100:.2f}% ± {ft_t.std()*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
