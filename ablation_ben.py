"""
BigEarthNet STE ablation study — 40k subset.

GAIN (all 12 bands, 5 seeds each):
  rgb_dino            : DINOv2-RGB linear probe (deterministic)
  cnn_allband         : all-12-band CNN from scratch
  ste_mean_wl         : STE mean-pool + wavelength cond + dropout ON
  ste_conv_wl         : STE conv spatial + wavelength cond + dropout ON  [main]
  ste_conv_wl_nodrop  : STE conv spatial + wavelength cond + dropout OFF
  ste_conv_idx        : STE conv spatial + index cond + dropout ON
  ste_conv_idx_nodrop : STE conv spatial + index cond + dropout OFF

XSENS (train 8 bands -> new sensor, 5 seeds each):
  xsens_wl            : STE conv + wavelength cond + dropout ON
  xsens_idx           : STE conv + index cond + dropout ON
  (RGB-DINOv2 = reference, doesn't change with band set)

Results saved to results/ablation_ben.json
Run: python ablation_ben.py 2>&1 | tee results/ablation_ben.log
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


# ── utils ──────────────────────────────────────────────────────────────────────

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


def gpu_mem():
    if torch.cuda.is_available():
        return f"  GPU {torch.cuda.memory_allocated()/1e9:.1f}/{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB"
    return ""


# ── models ─────────────────────────────────────────────────────────────────────

class SpectralEnc(nn.Module):
    def __init__(self, d=128, use_wl=True, use_conv=True):
        super().__init__()
        self.use_wl = use_wl
        self.use_conv = use_conv
        self.band_embed = nn.Linear(16, d)
        if use_wl:
            self.register_buffer("wl", sinusoid(WL, d))
            self.wl_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, d))
        else:
            self.band_idx = nn.Embedding(12, d)
        self.res_emb = nn.Embedding(3, d)
        self.register_buffer("res_cls", RES)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(True), nn.Linear(d, d))
        if use_conv:
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
        u = (u * m).sum(1) / m.sum(1).clamp(min=1)   # masked mean over bands -> [B, 256, d]
        if self.use_conv:
            u = u.permute(0, 2, 1).reshape(B, -1, 16, 16)
            u = self.spatial_conv(u).mean(dim=(2, 3))  # conv + global avg pool -> [B, d]
        else:
            u = u.mean(dim=1)                          # mean-pool over spatial tokens -> [B, d]
        return self.out(u)


class Fusion(nn.Module):
    def __init__(self, d, nc, use_wl, use_conv):
        super().__init__()
        self.enc = SpectralEnc(d, use_wl, use_conv)
        self.head = nn.Sequential(
            nn.Linear(768 + d, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, nc)
        )

    def forward(self, cls, x, mask):
        return self.head(torch.cat([cls, self.enc(x, mask)], dim=1))


class CNN(nn.Module):
    def __init__(self, in_ch, nc):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2)
            )
        self.net = nn.Sequential(
            blk(in_ch, 32), blk(32, 64), blk(64, 128), blk(128, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, nc)
        )

    def forward(self, x): return self.net(x)


# ── datasets ───────────────────────────────────────────────────────────────────

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


class CNN_DS(Dataset):
    def __init__(self, X, Y, idx, bands, mean, std, train=False):
        self.X, self.Y, self.idx, self.bands = X, Y, idx, bands
        self.mean, self.std, self.train = mean, std, train

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        x = self.X[self.idx[i]].float()
        xn = (x[self.bands] - self.mean) / self.std
        if self.train:
            if torch.rand(1).item() < 0.5: xn = xn.flip(-1)
            if torch.rand(1).item() < 0.5: xn = xn.flip(-2)
        return xn, self.Y[self.idx[i]]


# ── training functions ─────────────────────────────────────────────────────────

def train_ste(CLS, X, Y, tr, te, mean, std, nc,
              use_wl, use_conv, band_dropout,
              train_bands, eval_bands, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    net = torch.compile(Fusion(128, nc, use_wl, use_conv).to(DEV))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    trl = DataLoader(
        STE_DS(CLS, X, Y, tr, mean, std, train_bands, True, band_dropout),
        batch_size=BS, shuffle=True, num_workers=0, drop_last=True, pin_memory=True
    )
    for _ in range(epochs):
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
        STE_DS(CLS, X, Y, te, mean, std, eval_bands, False, False),
        batch_size=BS, shuffle=False, num_workers=0, pin_memory=True
    )
    S = []
    with torch.no_grad():
        for c, x, m, y in tel:
            S.append(torch.sigmoid(net(c.to(DEV), x.to(DEV), m.to(DEV))).cpu())
    return macro_mAP(torch.cat(S), Y[torch.tensor(te)])


def train_cnn(X, Y, tr, te, bands, mean, std, nc, seed, epochs=25):
    torch.manual_seed(seed)
    net = torch.compile(CNN(len(bands), nc).to(DEV))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    trl = DataLoader(
        CNN_DS(X, Y, tr, bands, mean, std, train=True),
        batch_size=BS, shuffle=True, num_workers=0, drop_last=True, pin_memory=True
    )
    for _ in range(epochs):
        net.train()
        for x, y in trl:
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(net(x.to(DEV)), y.to(DEV)).backward()
            opt.step()
        sched.step()
    net.eval()
    tel = DataLoader(
        CNN_DS(X, Y, te, bands, mean, std),
        batch_size=BS, shuffle=False, num_workers=0, pin_memory=True
    )
    S = []
    with torch.no_grad():
        for x, y in tel:
            S.append(torch.sigmoid(net(x.to(DEV))).cpu())
    return macro_mAP(torch.cat(S), Y[torch.tensor(te)])


def rgb_dino_linprobe(CLS, Y, tr, te, steps=800):
    head = nn.Linear(768, Y.shape[1]).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    Xtr = CLS[torch.tensor(tr)].to(DEV)
    Ytr = Y[torch.tensor(tr)].to(DEV)
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(head(Xtr), Ytr).backward()
        opt.step()
    with torch.no_grad():
        S = torch.sigmoid(head(CLS[torch.tensor(te)].to(DEV))).cpu()
    return macro_mAP(S, Y[torch.tensor(te)])


# ── orchestration ──────────────────────────────────────────────────────────────

def run_seeds(label, fn, results_dict, seeds=SEEDS):
    vals = []
    for s in seeds:
        t0 = time.time()
        v = fn(s)
        elapsed = time.time() - t0
        vals.append(v)
        print(f"  [{label}] seed={s}  mAP={v:.4f}  ({elapsed/60:.1f} min){gpu_mem()}", flush=True)
        # checkpoint after every seed
        results_dict[label] = {"seeds": vals, "mean": float(torch.tensor(vals).mean()), "std": float(torch.tensor(vals).std())}
        with open("results/ablation_ben.json", "w") as f:
            json.dump(results_dict, f, indent=2)
    t = torch.tensor(vals)
    return {"seeds": vals, "mean": float(t.mean()), "std": float(t.std())}


def main():
    torch.set_float32_matmul_precision("high")   # TF32 on GH200
    os.makedirs("results", exist_ok=True)

    t_start = time.time()
    print("=" * 70, flush=True)
    print("BigEarthNet STE Ablation Study", flush=True)
    print("=" * 70, flush=True)

    print("\nloading data...", flush=True)
    d = torch.load("data/ben/ben_subset.pt")
    X, Y, split = d["X"], d["Y"], d["split"]
    nc = Y.shape[1]
    tr = [i for i, s in enumerate(split) if s == "train"]
    te = [i for i, s in enumerate(split) if s == "test"]
    print(f"  {len(X)} patches, {nc} classes, train={len(tr)} test={len(te)}", flush=True)

    # STE normalization: 5k-sample subset at 64x64 (fast, matches ben_ste.py)
    samp = torch.tensor(tr[:5000])
    x5 = F.interpolate(X[samp].float(), size=64, mode='bilinear', align_corners=False)
    ste_mean = x5.mean(dim=(0, 2, 3), keepdim=True)[0]
    ste_std  = x5.std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6

    # CNN normalization: 5k-sample subset at native 120x120
    cnn_x5 = X[samp].float()
    cnn_mean = cnn_x5.mean(dim=(0, 2, 3), keepdim=True)[0]
    cnn_std  = cnn_x5.std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6

    print("precomputing DINOv2-RGB CLS ...", flush=True)
    CLS = precompute_cls(X)

    results = {}

    # ── RGB-DINOv2 linear probe ─────────────────────────────────────────────
    print("\n=== [1/9] RGB-DINOv2 LP ===", flush=True)
    t0 = time.time()
    rgb_map = rgb_dino_linprobe(CLS, Y, tr, te)
    results["rgb_dino"] = {"mAP": rgb_map}
    with open("results/ablation_ben.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  RGB-DINOv2 LP  mAP={rgb_map:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    allb = list(range(12))

    # ── GAIN: main STE result (5 seeds) — FIRST ────────────────────────────
    print("\n=== [2/9] GAIN: ste_conv_wl (5 seeds, all 12 bands) ===", flush=True)
    results["ste_conv_wl"] = run_seeds(
        "ste_conv_wl",
        lambda s: train_ste(CLS, X, Y, tr, te, ste_mean, ste_std, nc,
                            True, True, True, allb, allb, s),
        results,
    )
    r = results["ste_conv_wl"]
    print(f"  ste_conv_wl (GAIN): {r['mean']:.4f}±{r['std']:.3f}  "
          f"vs RGB {(r['mean']-rgb_map)*100:+.2f}  vs CNN (TBD)", flush=True)

    # ── XSENS: wavelength vs index (5 seeds each) — SECOND ────────────────
    print(f"\n[XSENS] held-out={HELDOUT}  train_bands={TRAIN_BANDS}  new_sensor={NEW_SENSOR}", flush=True)
    print("\n=== [3/9] XSENS: xsens_wl (5 seeds) ===", flush=True)
    results["xsens_wl"] = run_seeds(
        "xsens_wl",
        lambda s: train_ste(CLS, X, Y, tr, te, ste_mean, ste_std, nc,
                            True, True, True, TRAIN_BANDS, NEW_SENSOR, s),
        results,
    )
    r = results["xsens_wl"]
    print(f"  xsens_wl: {r['mean']:.4f}±{r['std']:.3f}", flush=True)

    print("\n=== [4/9] XSENS: xsens_idx (5 seeds) ===", flush=True)
    results["xsens_idx"] = run_seeds(
        "xsens_idx",
        lambda s: train_ste(CLS, X, Y, tr, te, ste_mean, ste_std, nc,
                            False, True, True, TRAIN_BANDS, NEW_SENSOR, s),
        results,
    )
    r = results["xsens_idx"]
    print(f"  xsens_idx: {r['mean']:.4f}±{r['std']:.3f}", flush=True)

    wl_xsens = results["xsens_wl"]
    idx_xsens = results["xsens_idx"]
    print(f"\n  *** XSENS gap (wl - idx): {(wl_xsens['mean']-idx_xsens['mean'])*100:+.2f} pts  "
          f"wl={wl_xsens['mean']:.4f}±{wl_xsens['std']:.3f}  "
          f"idx={idx_xsens['mean']:.4f}±{idx_xsens['std']:.3f} ***", flush=True)

    # ── Remaining ablation configs ─────────────────────────────────────────
    print("\n=== [5/9] All-band CNN (5 seeds) ===", flush=True)
    results["cnn_allband"] = run_seeds(
        "cnn_allband",
        lambda s: train_cnn(X, Y, tr, te, allb, cnn_mean, cnn_std, nc, s),
        results,
    )
    r = results["cnn_allband"]
    print(f"  cnn_allband: {r['mean']:.4f}±{r['std']:.3f}", flush=True)

    ablation_configs = [
        ("ste_mean_wl",         6,  dict(use_wl=True,  use_conv=False, band_dropout=True)),
        ("ste_conv_wl_nodrop",  7,  dict(use_wl=True,  use_conv=True,  band_dropout=False)),
        ("ste_conv_idx",        8,  dict(use_wl=False, use_conv=True,  band_dropout=True)),
        ("ste_conv_idx_nodrop", 9,  dict(use_wl=False, use_conv=True,  band_dropout=False)),
    ]
    for name, step, cfg in ablation_configs:
        print(f"\n=== [{step}/9] {name} (5 seeds, GAIN) ===", flush=True)
        results[name] = run_seeds(
            name,
            lambda s, cfg=cfg: train_ste(
                CLS, X, Y, tr, te, ste_mean, ste_std, nc,
                cfg["use_wl"], cfg["use_conv"], cfg["band_dropout"],
                allb, allb, s
            ),
            results,
        )
        r = results[name]
        print(f"  {name}: {r['mean']:.4f}±{r['std']:.3f}", flush=True)

    # ── Final table ────────────────────────────────────────────────────────
    rgb = results["rgb_dino"]["mAP"]
    cnn = results["cnn_allband"]["mean"]
    wl  = results["xsens_wl"]
    idx = results["xsens_idx"]
    total_min = (time.time() - t_start) / 60

    gain_order = [
        "ste_conv_wl", "ste_conv_wl_nodrop",
        "ste_conv_idx", "ste_conv_idx_nodrop",
        "ste_mean_wl",
    ]

    print("\n" + "=" * 72, flush=True)
    print("ABLATION RESULTS — BigEarthNet 40k subset (macro-mAP)", flush=True)
    print("=" * 72, flush=True)
    print(f"{'Model':<30} {'mean':>8} {'±std':>7} {'vs RGB':>8} {'vs CNN':>8}", flush=True)
    print("-" * 72, flush=True)
    print(f"{'RGB-DINOv2 LP':<30} {rgb:>8.4f} {'—':>7} {'—':>8} {'—':>8}", flush=True)
    r = results["cnn_allband"]
    print(f"{'All-band CNN':<30} {r['mean']:>8.4f} {r['std']:>7.3f} {(r['mean']-rgb)*100:>+8.2f} {'—':>8}", flush=True)
    print(flush=True)
    print("── GAIN (all 12 bands) ──────────────────────────────────────────────", flush=True)
    for name in gain_order:
        r = results[name]
        print(f"  {name:<28} {r['mean']:>8.4f} {r['std']:>7.3f} {(r['mean']-rgb)*100:>+8.2f} {(r['mean']-cnn)*100:>+8.2f}", flush=True)
    print(flush=True)
    print("── XSENS (train 8 bands -> new sensor) ─────────────────────────────", flush=True)
    for name in ["xsens_wl", "xsens_idx"]:
        r = results[name]
        print(f"  {name:<28} {r['mean']:>8.4f} {r['std']:>7.3f} {(r['mean']-rgb)*100:>+8.2f}", flush=True)
    print(f"\n  XSENS gap (wl - idx): {(wl['mean']-idx['mean'])*100:+.2f} pts  "
          f"(wl {wl['mean']:.4f}±{wl['std']:.3f}  idx {idx['mean']:.4f}±{idx['std']:.3f})", flush=True)
    print("=" * 72, flush=True)
    print(f"\nTotal wall time: {total_min:.0f} min", flush=True)
    print(f"Results: results/ablation_ben.json  |  log: results/ablation_ben.log", flush=True)


if __name__ == "__main__":
    main()
