"""BigEarthNet STE: gain test + cross-sensor, multi-label, on the unsaturated regime.

Frozen DINOv2-RGB CLS (precomputed) (+) wavelength/resolution-conditioned spectral
adapter over the 12 bands -> multi-label head. SpectralEnc v2: masked-mean over bands
(composition-invariant) then 2-layer conv over 16x16 spatial grid (preserves spatial
structure) instead of mean-pooling. Baselines: RGB-DINOv2 LP = 0.565, all-12-band CNN = 0.630.

Tests:
  GAIN  : STE (all bands) macro-mAP vs 0.565 / 0.630  -> does it beat both?
  XSENS : hold out informative bands; new-sensor wavelength vs index gain.

Run:  python ben_ste.py
"""
import math
import os
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
HELDOUT = [4, 5, 8, 10]                                   # B05,B06,B8A,B11 (informative, interpolatable)
TRAIN_BANDS = [b for b in range(12) if b not in HELDOUT]
NEW_SENSOR = sorted(set(RGB_IDX) | set(HELDOUT))


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
    pos = (vals / 100.0).unsqueeze(1); div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(len(vals), d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


@torch.no_grad()
def precompute_cls(X, cache="data/ben/dino_cls.pt", bs=256):
    if os.path.exists(cache):
        return torch.load(cache)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(DEV).eval()
    out = []
    for i in range(0, len(X), bs):
        r = (X[i:i + bs][:, RGB_IDX].float() / 3000.0).clamp(0, 1).to(DEV)
        r = F.interpolate(r, size=224, mode="bilinear", align_corners=False)
        r = (r - IMN_M.to(DEV)) / IMN_S.to(DEV)
        out.append(F.normalize(dino(r), dim=1).cpu())
    cls = torch.cat(out); torch.save(cls, cache); return cls


class SpectralEnc(nn.Module):
    def __init__(self, d=128, use_wl=True):
        super().__init__(); self.use_wl = use_wl
        self.band_embed = nn.Linear(16, d)
        if use_wl:
            self.register_buffer("wl", sinusoid(WL, d)); self.wl_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, d))
        else:
            self.band_idx = nn.Embedding(12, d)
        self.res_emb = nn.Embedding(3, d); self.register_buffer("res_cls", RES)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(True), nn.Linear(d, d))
        # conv over the 16×16 spatial grid of patch tokens (replaces mean-pool — preserves spatial structure)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU(True),
        )
        self.out = nn.Linear(d, d)

    def cond(self):
        c = self.wl_mlp(self.wl) if self.use_wl else self.band_idx.weight
        return c + self.res_emb(self.res_cls)

    def forward(self, x, mask):                               # x[B,12,64,64] mask[B,12]
        B = x.shape[0]
        p = x.reshape(B, 12, 16, 4, 16, 4).permute(0, 1, 2, 4, 3, 5).reshape(B, 12, 256, 16)
        v = self.band_embed(p)
        cond = self.cond()[None, :, None, :].expand(B, 12, 256, -1)
        u = self.fuse(torch.cat([v, cond], dim=-1))
        m = mask[:, :, None, None]
        u = (u * m).sum(1) / m.sum(1).clamp(min=1)           # masked mean over bands → [B, 256, d]
        u = u.permute(0, 2, 1).reshape(B, -1, 16, 16)        # [B, d, 16, 16]
        u = self.spatial_conv(u).mean(dim=(2, 3))             # conv then global avg pool → [B, d]
        return self.out(u)


class Fusion(nn.Module):
    def __init__(self, d, nc, use_wl):
        super().__init__(); self.enc = SpectralEnc(d, use_wl)
        self.head = nn.Sequential(nn.Linear(768 + d, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, nc))

    def forward(self, cls, x, mask):
        return self.head(torch.cat([cls, self.enc(x, mask)], dim=1))


class DS(Dataset):
    def __init__(self, CLS, X, Y, idx, mean, std, present, train):
        self.CLS, self.X, self.Y, self.idx, self.mean, self.std, self.present, self.train = CLS, X, Y, idx, mean, std, present, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        x = F.interpolate(self.X[j].float()[None], size=64, mode="bilinear", align_corners=False)[0]  # 120->64
        xn = (x - self.mean) / self.std
        mask = torch.zeros(12); mask[self.present] = 1
        if self.train:
            for b in self.present:
                if b not in RGB_IDX and torch.rand(1).item() < 0.4: mask[b] = 0
        xn = xn * mask[:, None, None]
        if self.train and torch.rand(1).item() < 0.5: xn = xn.flip(-1)
        return self.CLS[j], xn, mask, self.Y[j]


BS = 1024       # GH200 480 GB — batch size tuned for single large-HBM GPU
LR = 4e-3       # conservative 2× linear scale from original lr=2e-3 at bs=128
EPOCHS = 50     # 50 × (30k/1024) ≈ 1465 updates; larger bs + higher lr compensates


def train_eval(CLS, X, Y, tr, te, mean, std, nc, use_wl, train_bands, eval_present, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    net = torch.compile(Fusion(128, nc, use_wl).to(DEV))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    trl = DataLoader(DS(CLS, X, Y, tr, mean, std, train_bands, True), batch_size=BS, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    for _ in range(epochs):
        net.train()
        for c, x, m, y in trl:
            opt.zero_grad(); F.binary_cross_entropy_with_logits(net(c.to(DEV), x.to(DEV), m.to(DEV)), y.to(DEV)).backward(); opt.step()
        sched.step()
    net.eval(); S = []
    tel = DataLoader(DS(CLS, X, Y, te, mean, std, eval_present, False), batch_size=BS, shuffle=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        for c, x, m, y in tel:
            S.append(torch.sigmoid(net(c.to(DEV), x.to(DEV), m.to(DEV))).cpu())
    return macro_mAP(torch.cat(S), Y[torch.tensor(te)])


def main():
    torch.set_float32_matmul_precision("high")  # TF32 on GH200
    d = torch.load("data/ben/ben_subset.pt"); X, Y, split = d["X"], d["Y"], d["split"]; nc = Y.shape[1]
    tr = [i for i, s in enumerate(split) if s == "train"]; te = [i for i, s in enumerate(split) if s == "test"]
    samp = torch.tensor(tr[:5000])
    x5 = F.interpolate(X[samp].float(), size=64, mode="bilinear", align_corners=False)
    mean = x5.mean(dim=(0, 2, 3), keepdim=True)[0]; std = x5.std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6
    print("precompute DINOv2-RGB CLS ...", flush=True); CLS = precompute_cls(X)
    print(f"BEN-STE: train={len(tr)} test={len(te)} | baselines RGB-DINOv2=0.565 all-band=0.630", flush=True)

    allb = list(range(12))
    g = train_eval(CLS, X, Y, tr, te, mean, std, nc, True, allb, allb, 0)
    print(f"\n[GAIN] STE-wavelength (all 12 bands) macro-mAP = {g:.4f}  (vs RGB 0.565 = {(g-0.565)*100:+.2f}, vs all-band-CNN 0.630 = {(g-0.630)*100:+.2f})", flush=True)

    print(f"\n[XSENS] held out {HELDOUT}; new-sensor = {NEW_SENSOR}", flush=True)
    wl, idx = [], []
    for seed in [0, 1]:
        nw = train_eval(CLS, X, Y, tr, te, mean, std, nc, True, TRAIN_BANDS, NEW_SENSOR, seed)
        ni = train_eval(CLS, X, Y, tr, te, mean, std, nc, False, TRAIN_BANDS, NEW_SENSOR, seed)
        wl.append(nw); idx.append(ni)
        print(f"  seed{seed}: wl_new={nw:.4f} idx_new={ni:.4f} gap={(nw-ni)*100:+.2f}", flush=True)
    wl = torch.tensor(wl); idx = torch.tensor(idx)
    print(f"\n  cross-sensor wavelength - index = {(wl.mean()-idx.mean())*100:+.2f} pts (wl {wl.mean():.4f}±{wl.std():.3f}, idx {idx.mean():.4f}±{idx.std():.3f})")


if __name__ == "__main__":
    main()
