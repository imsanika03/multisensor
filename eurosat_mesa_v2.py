"""MESA-v2 on EuroSAT (13-band Sentinel-2, 10-class single-label).

Per-location spectral gating — port of ben_mesa_v2.py to EuroSAT.
Key differences from BEN version:
  - 13 bands (adds B10 Cirrus at 1375nm, FWHM=30nm)
  - CrossEntropy loss, top-1 accuracy metric
  - 80/20 random split (seed=0 permutation)
  - num_workers=0 (CUDA fork deadlock)
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
NC      = 10

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
D_BOTTLE       = 128
N_ND           = 16
ADAPTER_BLOCKS = list(range(6, 12))
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
# MESASummarizer
# ---------------------------------------------------------------------------
class MESASummarizer(nn.Module):
    def __init__(self):
        super().__init__()
        A = build_A(); M = build_M(A)
        self.register_buffer("A", A)
        self.register_buffer("M", M)
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(K_KNOTS, D_SPEC, 3, padding=1), nn.GELU(),
            nn.Conv2d(D_SPEC, D_SPEC, 3, padding=1, groups=D_SPEC), nn.GELU(),
            nn.Conv2d(D_SPEC, D_SPEC, 1),
        )
        self.nd_a    = nn.Parameter(torch.randn(N_ND, K_KNOTS) * 0.1)
        self.nd_b    = nn.Parameter(torch.randn(N_ND, K_KNOTS) * 0.1)
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


# ---------------------------------------------------------------------------
# Per-location gated adapter (identical to ben_mesa_v2)
# ---------------------------------------------------------------------------
class PerLocGatedAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down     = nn.Linear(768, D_BOTTLE)
        self.loc_gate = nn.Linear(D_SPEC, D_BOTTLE)
        self.loc_bias = nn.Linear(D_SPEC, D_BOTTLE)
        self.cls_gate = nn.Linear(D_SUMM, D_BOTTLE)
        self.cls_bias = nn.Linear(D_SUMM, D_BOTTLE)
        self.up = nn.Linear(D_BOTTLE, 768)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, local_feat, global_summ):
        h = F.gelu(self.down(x))
        cg = torch.sigmoid(self.cls_gate(global_summ)).unsqueeze(1)
        cb = self.cls_bias(global_summ).unsqueeze(1)
        h_cls = h[:, :1] * cg + cb
        pg = torch.sigmoid(self.loc_gate(local_feat))
        pb = self.loc_bias(local_feat)
        h_pat = h[:, 1:] * pg + pb
        return x + self.up(torch.cat([h_cls, h_pat], dim=1))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class MESAv2EuroSAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.summarizer = MESASummarizer()
        self.dino = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14',
            pretrained=True, verbose=False
        ).to(DEV)
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.adapters = nn.ModuleList([PerLocGatedAdapter() for _ in ADAPTER_BLOCKS])
        self.head = nn.Sequential(
            nn.Linear(768 + 768, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, NC),
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
        j    = self.idx[i]
        x    = self.X[j].float()
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


def train_eval(X, Y, tr, te, seed):
    torch.manual_seed(seed)
    net   = MESAv2EuroSAT().to(DEV)
    opt   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    sched = cosine_with_warmup(opt, WARMUP, EPOCHS)

    trl = DataLoader(EuroSATDataset(X, Y, tr, train=True),
                     batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
    tel = DataLoader(EuroSATDataset(X, Y, te, train=False),
                     batch_size=BS * 2, shuffle=False, num_workers=0)

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
        ep_loss = 0.0; n_batches = 0
        for rgb, x_loc, y in trl:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(net(rgb.to(DEV), x_loc.to(DEV)), y.to(DEV))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_batches += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            acc = eval_acc()
            print(f"    epoch {ep+1}/{EPOCHS}  loss={ep_loss/n_batches:.4f}  acc={acc*100:.2f}%", flush=True)

    return eval_acc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.set_float32_matmul_precision("high")
    print(f"MESA-v2 EuroSAT: bands={N_BANDS}, D_SPEC={D_SPEC}, D_BOTTLE={D_BOTTLE}, "
          f"blocks={ADAPTER_BLOCKS}", flush=True)
    print(f"Training: LR={LR}, BS={BS}, EPOCHS={EPOCHS}, seeds={SEEDS}", flush=True)

    tifs = ensure_unzipped()
    print(f"loading {len(tifs)} tifs ...", flush=True)
    X, Y, classes = load_all(tifs)
    print(f"  X={tuple(X.shape)}  classes={classes}", flush=True)

    torch.manual_seed(0)
    perm = torch.randperm(len(X)).tolist()
    n_tr = int(len(X) * TRAIN_FRAC)
    tr, te = perm[:n_tr], perm[n_tr:]
    print(f"  train={len(tr)}  test={len(te)}", flush=True)

    accs = []
    for s in SEEDS:
        print(f"\n  seed {s} ...", flush=True)
        acc = train_eval(X, Y, tr, te, s)
        accs.append(acc)
        print(f"    seed {s} acc = {acc*100:.2f}%", flush=True)

    t = torch.tensor(accs)
    results = {"seeds": accs, "mean": t.mean().item(), "std": t.std().item()}
    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/eurosat_mesa_v2.json", "w"), indent=2)

    print("\n=== EUROSAT SUMMARY ===", flush=True)
    print(f"  DINOv2 LP (baseline)  : 96.78%", flush=True)
    print(f"  MESA-v2 ({len(SEEDS)} seeds)    : {t.mean()*100:.2f}% +/- {t.std()*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
