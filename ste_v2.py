"""STE v2 — fix the cross-sensor mechanism.

Diagnosis of v1: on new band compositions the spectral pathway injected net NOISE
(new-sensor acc < RGB baseline for BOTH wavelength and index) -> nothing to
differentiate. Causes: (1) additive conditioning leaks pixels (conditioning not
load-bearing), (2) competitive softmax pool is not composition-invariant,
(3) fragile token injection.

Fix: DeepSets spectral encoder -- per band  u_b = MLP([band_embed(pixels_b) ;
cond_b]) (conditioning CONCATENATED -> load-bearing), masked MEAN over present
bands (composition-invariant, additive), late-fused with frozen DINOv2-RGB CLS
(precomputed -> fast, no ViT backprop).

Success: new-sensor wavelength > RGB-only ref (extracts held-out signal) AND
new-sensor wavelength > index (reliably over seeds).
Run:  python ste_v2.py
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sat_ms_headroom import ensure_unzipped, load_all

DEV = "cuda" if torch.cuda.is_available() else "cpu"
WL = torch.tensor([443., 490, 560, 665, 705, 740, 783, 842, 865, 945, 1375, 1610, 2190])
RES = torch.tensor([2, 0, 0, 0, 1, 1, 1, 0, 1, 2, 2, 1, 1])
RGB_IDX = [3, 2, 1]
IMN_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
HELDOUT = [4, 5, 8, 11]
TRAIN_BANDS = [b for b in range(13) if b not in HELDOUT]
NEW_SENSOR = sorted(set(RGB_IDX) | set(HELDOUT))


def sinusoid(vals, d):
    pos = (vals / 100.0).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(len(vals), d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


@torch.no_grad()
def precompute_cls(X, cache="data/eurosat_ms/dino_rgb_cls.pt", bs=256):
    if os.path.exists(cache):
        return torch.load(cache)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(DEV).eval()
    out = []
    for i in range(0, len(X), bs):
        rgb = (X[i:i + bs][:, RGB_IDX] / 3000.0).clamp(0, 1).to(DEV)
        rgb = F.interpolate(rgb, size=224, mode="bilinear", align_corners=False)
        rgb = (rgb - IMN_M.to(DEV)) / IMN_S.to(DEV)
        out.append(F.normalize(dino(rgb), dim=1).cpu())
    cls = torch.cat(out); torch.save(cls, cache); return cls


class SpectralEnc(nn.Module):
    def __init__(self, d=128, use_wl=True):
        super().__init__()
        self.use_wl = use_wl
        self.band_embed = nn.Linear(16, d)
        if use_wl:
            self.register_buffer("wl", sinusoid(WL, d)); self.wl_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, d))
        else:
            self.band_idx = nn.Embedding(13, d)
        self.res_emb = nn.Embedding(3, d); self.register_buffer("res_cls", RES)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(True), nn.Linear(d, d))
        self.out = nn.Linear(d, d)

    def cond(self):
        c = self.wl_mlp(self.wl) if self.use_wl else self.band_idx.weight
        return c + self.res_emb(self.res_cls)                       # [13,d]

    def forward(self, x, mask):                                     # x[B,13,64,64] mask[B,13]
        B = x.shape[0]
        p = x.reshape(B, 13, 16, 4, 16, 4).permute(0, 1, 2, 4, 3, 5).reshape(B, 13, 256, 16)
        v = self.band_embed(p)                                      # [B,13,256,d]
        cond = self.cond()[None, :, None, :].expand(B, 13, 256, -1)
        u = self.fuse(torch.cat([v, cond], dim=-1))                 # conditioning load-bearing
        m = mask[:, :, None, None]
        u = (u * m).sum(1) / m.sum(1).clamp(min=1)                  # masked MEAN over bands (composition-invariant)
        return self.out(u.mean(1))                                  # mean over patches -> [B,d]


class Fusion(nn.Module):
    def __init__(self, d, nc, use_wl):
        super().__init__()
        self.enc = SpectralEnc(d, use_wl)
        self.head = nn.Sequential(nn.Linear(768 + d, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, nc))

    def forward(self, cls, x, mask):
        return self.head(torch.cat([cls, self.enc(x, mask)], dim=1))


class DS(Dataset):
    def __init__(self, CLS, X, Y, idx, mean, std, present, train):
        self.CLS, self.X, self.Y, self.idx, self.mean, self.std, self.present, self.train = CLS, X, Y, idx, mean, std, present, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        xn = (self.X[j] - self.mean) / self.std
        mask = torch.zeros(13); mask[self.present] = 1
        if self.train:
            for b in self.present:
                if b not in RGB_IDX and torch.rand(1).item() < 0.4: mask[b] = 0
        xn = xn * mask[:, None, None]
        if self.train and torch.rand(1).item() < 0.5: xn = xn.flip(-1)
        if self.train and torch.rand(1).item() < 0.5: xn = xn.flip(-2)
        return self.CLS[j], xn, mask, self.Y[j]


def train_eval(CLS, X, Y, tr, te, mean, std, nc, use_wl, seed, epochs=40):
    torch.manual_seed(seed)
    net = Fusion(128, nc, use_wl).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    trl = DataLoader(DS(CLS, X, Y, tr, mean, std, TRAIN_BANDS, True), batch_size=512, shuffle=True, num_workers=12, drop_last=True)

    def ev(present):
        net.eval(); cor = tot = 0
        tel = DataLoader(DS(CLS, X, Y, te, mean, std, present, False), batch_size=512, shuffle=False, num_workers=12)
        with torch.no_grad():
            for c, xn, m, y in tel:
                p = net(c.to(DEV), xn.to(DEV), m.to(DEV)).argmax(1).cpu()
                cor += (p == y).sum().item(); tot += y.numel()
        return cor / tot

    for _ in range(epochs):
        net.train()
        for c, xn, m, y in trl:
            opt.zero_grad(); F.cross_entropy(net(c.to(DEV), xn.to(DEV), m.to(DEV)), y.to(DEV)).backward(); opt.step()
        sched.step()
    return ev(TRAIN_BANDS), ev(NEW_SENSOR), ev(RGB_IDX)   # same-sensor, new-sensor, rgb-only ref


def main():
    tifs = ensure_unzipped(); X, Y, classes = load_all(tifs); nc = len(classes); N = len(Y)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    tr, te = perm[:int(.8 * N)], perm[int(.8 * N):]
    mean = X[tr].mean(dim=(0, 2, 3), keepdim=True)[0]; std = X[tr].std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6
    print("precomputing frozen DINOv2-RGB CLS ...")
    CLS = precompute_cls(X)
    print(f"STE v2 (DeepSets, late-fusion). held out {HELDOUT}; new sensor {NEW_SENSOR}")

    res = {"wl": {"same": [], "new": [], "rgb": []}, "idx": {"same": [], "new": [], "rgb": []}}
    for seed in [0, 1, 2]:
        for use_wl, key in [(True, "wl"), (False, "idx")]:
            s, nw, rgb = train_eval(CLS, X, Y, tr, te, mean, std, nc, use_wl, seed)
            res[key]["same"].append(s); res[key]["new"].append(nw); res[key]["rgb"].append(rgb)
            print(f"  seed{seed} {key}: same={s:.4f} new={nw:.4f} rgb-only={rgb:.4f}", flush=True)

    def m(a): t = torch.tensor(a); return f"{t.mean():.4f}±{t.std():.3f}"
    print("\n#### STE v2 (mean±std over 3 seeds) ####")
    print(f"  wavelength : same={m(res['wl']['same'])}  new={m(res['wl']['new'])}  rgb-only={m(res['wl']['rgb'])}")
    print(f"  index      : same={m(res['idx']['same'])}  new={m(res['idx']['new'])}  rgb-only={m(res['idx']['rgb'])}")
    nw = torch.tensor(res['wl']['new']); ni = torch.tensor(res['idx']['new']); rg = torch.tensor(res['wl']['rgb'])
    print(f"\n  new-sensor wavelength - index   = {(nw.mean()-ni.mean())*100:+.2f} pts (the novelty)")
    print(f"  new-sensor wavelength - RGB-only = {(nw.mean()-rg.mean())*100:+.2f} pts (does it USE the new bands?)")


if __name__ == "__main__":
    main()
