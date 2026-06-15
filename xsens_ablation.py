"""
XSENS ablation: wavelength vs index, dropout on vs off.
4 configs x 5 seeds = 20 runs (~2.6 hrs on GH200).

Results saved to results/xsens_ablation.json
Run: python xsens_ablation.py 2>&1 | tee results/xsens_ablation.log
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WL = torch.tensor([443., 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190])
RES = torch.tensor([2, 0, 0, 0, 1, 1, 1, 0, 1, 2, 1, 1])
RGB_IDX = [3, 2, 1]
IMN_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
HELDOUT = [4, 5, 8, 10]
TRAIN_BANDS = [b for b in range(12) if b not in HELDOUT]
NEW_SENSOR = sorted(set(RGB_IDX) | set(HELDOUT))
SEEDS = [0, 1, 2, 3, 4]
BS = 1024
LR = 4e-3
EPOCHS = 50

CONFIGS = [
    ("xsens_wl_drop",   dict(use_wl=True,  band_dropout=True)),
    ("xsens_wl_nodrop", dict(use_wl=True,  band_dropout=False)),
    ("xsens_idx_drop",  dict(use_wl=False, band_dropout=True)),
    ("xsens_idx_nodrop",dict(use_wl=False, band_dropout=False)),
]


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
    pe = torch.zeros(len(vals), d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


@torch.no_grad()
def precompute_cls(X, cache="data/ben/dino_cls.pt", bs=256):
    if os.path.exists(cache):
        print("  loading cached DINOv2 CLS ...", flush=True)
        return torch.load(cache)
    print("  computing DINOv2 CLS (first time) ...", flush=True)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(DEV).eval()
    out = []
    for i in range(0, len(X), bs):
        r = (X[i:i + bs][:, RGB_IDX].float() / 3000.0).clamp(0, 1).to(DEV)
        r = F.interpolate(r, size=224, mode='bilinear', align_corners=False)
        r = (r - IMN_M.to(DEV)) / IMN_S.to(DEV)
        out.append(F.normalize(dino(r), dim=1).cpu())
    cls = torch.cat(out)
    torch.save(cls, cache)
    return cls


class SpectralEnc(nn.Module):
    def __init__(self, d=128, use_wl=True):
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
        self.out = nn.Linear(d, d)

    def cond(self):
        c = self.wl_mlp(self.wl) if self.use_wl else self.band_idx.weight
        return c + self.res_emb(self.res_cls)

    def forward(self, x, mask):
        B = x.shape[0]
        p = x.reshape(B, 12, 16, 4, 16, 4).permute(0, 1, 2, 4, 3, 5).reshape(B, 12, 256, 16)
        v = self.band_embed(p)
        cond = self.cond()[None, :, None, :].expand(B, 12, 256, -1)
        u = self.fuse(torch.cat([v, cond], dim=-1))
        m = mask[:, :, None, None]
        u = (u * m).sum(1) / m.sum(1).clamp(min=1)
        u = u.permute(0, 2, 1).reshape(B, -1, 16, 16)
        u = self.spatial_conv(u).mean(dim=(2, 3))
        return self.out(u)


class Fusion(nn.Module):
    def __init__(self, d, nc, use_wl):
        super().__init__()
        self.enc = SpectralEnc(d, use_wl)
        self.head = nn.Sequential(
            nn.Linear(768 + d, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, nc)
        )

    def forward(self, cls, x, mask):
        return self.head(torch.cat([cls, self.enc(x, mask)], dim=1))


class STE_DS(Dataset):
    def __init__(self, CLS, X, Y, idx, mean, std, present, train, band_dropout=True):
        self.CLS, self.X, self.Y, self.idx = CLS, X, Y, idx
        self.mean, self.std, self.present = mean, std, present
        self.train, self.band_dropout = train, band_dropout

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = F.interpolate(self.X[j].float()[None], size=64, mode='bilinear', align_corners=False)[0]
        xn = (x - self.mean) / self.std
        mask = torch.zeros(12)
        mask[self.present] = 1
        if self.train and self.band_dropout:
            for b in self.present:
                if b not in RGB_IDX and torch.rand(1).item() < 0.4:
                    mask[b] = 0
        xn = xn * mask[:, None, None]
        if self.train and torch.rand(1).item() < 0.5:
            xn = xn.flip(-1)
        return self.CLS[j], xn, mask, self.Y[j]


def train_eval(CLS, X, Y, tr, te, mean, std, nc, use_wl, band_dropout, seed):
    torch.manual_seed(seed)
    net = torch.compile(Fusion(128, nc, use_wl).to(DEV))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    trl = DataLoader(
        STE_DS(CLS, X, Y, tr, mean, std, TRAIN_BANDS, True, band_dropout),
        batch_size=BS, shuffle=True, num_workers=0, drop_last=True, pin_memory=True
    )
    for _ in range(EPOCHS):
        net.train()
        for c, x, m, y in trl:
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(
                net(c.to(DEV), x.to(DEV), m.to(DEV)), y.to(DEV)
            ).backward()
            opt.step()
        sched.step()
    net.eval()
    tel = DataLoader(
        STE_DS(CLS, X, Y, te, mean, std, NEW_SENSOR, False, False),
        batch_size=BS, shuffle=False, num_workers=0, pin_memory=True
    )
    S = []
    with torch.no_grad():
        for c, x, m, y in tel:
            S.append(torch.sigmoid(net(c.to(DEV), x.to(DEV), m.to(DEV))).cpu())
    return macro_mAP(torch.cat(S), Y[torch.tensor(te)])


def main():
    torch.set_float32_matmul_precision("high")
    os.makedirs("results", exist_ok=True)
    t_start = time.time()

    print("=" * 70, flush=True)
    print("XSENS Ablation: wavelength/index x dropout on/off (5 seeds each)", flush=True)
    print(f"held-out={HELDOUT}  train_bands={TRAIN_BANDS}  new_sensor={NEW_SENSOR}", flush=True)
    print("=" * 70, flush=True)

    print("\nloading data...", flush=True)
    d = torch.load("data/ben/ben_subset.pt")
    X, Y, split = d["X"], d["Y"], d["split"]
    nc = Y.shape[1]
    tr = [i for i, s in enumerate(split) if s == "train"]
    te = [i for i, s in enumerate(split) if s == "test"]

    samp = torch.tensor(tr[:5000])
    x5 = F.interpolate(X[samp].float(), size=64, mode='bilinear', align_corners=False)
    mean = x5.mean(dim=(0, 2, 3), keepdim=True)[0]
    std  = x5.std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6

    CLS = precompute_cls(X)
    results = {}

    for i, (name, cfg) in enumerate(CONFIGS, 1):
        print(f"\n=== [{i}/4] {name} (5 seeds) ===", flush=True)
        vals = []
        for s in SEEDS:
            t0 = time.time()
            v = train_eval(CLS, X, Y, tr, te, mean, std, nc,
                           cfg["use_wl"], cfg["band_dropout"], s)
            elapsed = time.time() - t0
            vals.append(v)
            t = torch.tensor(vals)
            std_val = float(t.std()) if len(vals) > 1 else 0.0
            results[name] = {"seeds": vals, "mean": float(t.mean()), "std": std_val}
            with open("results/xsens_ablation.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [{name}] seed={s}  mAP={v:.4f}  ({elapsed/60:.1f} min)", flush=True)
        r = results[name]
        print(f"  {name}: {r['mean']:.4f}±{r['std']:.3f}", flush=True)

    # ── Summary table ──────────────────────────────────────────────────────
    rgb_ref = 0.5650
    total_min = (time.time() - t_start) / 60

    print("\n" + "=" * 65, flush=True)
    print("XSENS ABLATION RESULTS — BigEarthNet 40k (macro-mAP)", flush=True)
    print("=" * 65, flush=True)
    print(f"  RGB-DINOv2 LP reference: {rgb_ref:.4f}", flush=True)
    print(f"{'Config':<24} {'mean':>8} {'±std':>7} {'vs RGB':>8}", flush=True)
    print("-" * 65, flush=True)
    for name, _ in CONFIGS:
        r = results[name]
        print(f"  {name:<22} {r['mean']:>8.4f} {r['std']:>7.3f} {(r['mean']-rgb_ref)*100:>+8.2f}", flush=True)

    wl_drop   = results["xsens_wl_drop"]
    wl_nodrop = results["xsens_wl_nodrop"]
    idx_drop  = results["xsens_idx_drop"]
    idx_nodrop= results["xsens_idx_nodrop"]
    print(f"\n  wl   dropout effect:  {(wl_drop['mean']  - wl_nodrop['mean'])*100:+.2f} pts  (drop={wl_drop['mean']:.4f}  nodrop={wl_nodrop['mean']:.4f})", flush=True)
    print(f"  idx  dropout effect:  {(idx_drop['mean'] - idx_nodrop['mean'])*100:+.2f} pts  (drop={idx_drop['mean']:.4f}  nodrop={idx_nodrop['mean']:.4f})", flush=True)
    print(f"  wl vs idx (drop ON):  {(wl_drop['mean']  - idx_drop['mean'])*100:+.2f} pts", flush=True)
    print(f"  wl vs idx (drop OFF): {(wl_nodrop['mean']- idx_nodrop['mean'])*100:+.2f} pts", flush=True)
    print("=" * 65, flush=True)
    print(f"\nTotal wall time: {total_min:.0f} min", flush=True)
    print("Results: results/xsens_ablation.json  |  log: results/xsens_ablation.log", flush=True)


if __name__ == "__main__":
    main()
