"""Multispectral headroom check on EuroSAT (13-band Sentinel-2).

Same small CNN trained from scratch on RGB-only (B4,B3,B2) vs all-13-bands, same
split/epochs -> isolates whether the non-RGB bands carry exploitable signal.
Compared against the strong RGB-DINOv2 linear-probe (0.96) for context.

  13-band >> RGB         -> multispectral signal is real; RGB foundation models
                            leave it on the table -> headroom for an "inject MS
                            into a frozen RGB foundation model" method.
  13-band ~= RGB         -> even MS is saturated on EuroSAT -> go to BigEarthNet.

Run:  python sat_ms_headroom.py
"""

import glob
import os
import zipfile

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "data/eurosat_ms"
RGB_IDX = [3, 2, 1]   # Sentinel-2 B04,B03,B02 within the 13-band stack


def ensure_unzipped():
    tifs = glob.glob(f"{ROOT}/**/*.tif", recursive=True)
    if len(tifs) < 1000:
        print("unzipping EuroSATallBands.zip ...")
        with zipfile.ZipFile(f"{ROOT}/EuroSATallBands.zip") as z:
            z.extractall(ROOT)
        tifs = glob.glob(f"{ROOT}/**/*.tif", recursive=True)
    return sorted(tifs)


def load_all(tifs):
    classes = sorted({os.path.basename(os.path.dirname(t)) for t in tifs})
    c2i = {c: i for i, c in enumerate(classes)}
    X = np.zeros((len(tifs), 13, 64, 64), dtype=np.float32)
    Y = np.zeros(len(tifs), dtype=np.int64)
    for k, t in enumerate(tifs):
        a = tifffile.imread(t).astype(np.float32)        # [64,64,13] or [13,64,64]
        if a.shape[-1] == 13:
            a = np.transpose(a, (2, 0, 1))
        X[k] = a[:13]
        Y[k] = c2i[os.path.basename(os.path.dirname(t))]
    return torch.from_numpy(X), torch.from_numpy(Y), classes


class BandDS(Dataset):
    def __init__(self, X, Y, idx, bands, mean, std, train):
        self.X, self.Y, self.idx, self.bands = X, Y, idx, bands
        self.mean, self.std, self.train = mean, std, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        x = self.X[self.idx[i]][self.bands]
        x = (x - self.mean) / self.std
        if self.train:
            if torch.rand(1).item() < 0.5: x = x.flip(-1)
            if torch.rand(1).item() < 0.5: x = x.flip(-2)
        return x, self.Y[self.idx[i]]


class CNN(nn.Module):
    def __init__(self, in_ch, nc):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                            nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(in_ch, 32), blk(32, 64), blk(64, 128), blk(128, 256),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, nc))
    def forward(self, x): return self.net(x)


def run(X, Y, tr, te, bands, nc, epochs=25):
    sel = X[tr][:, bands]
    mean = sel.mean(dim=(0, 2, 3), keepdim=True)[0]
    std = sel.std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6
    trl = DataLoader(BandDS(X, Y, tr, bands, mean, std, True), batch_size=256, shuffle=True, num_workers=8, drop_last=True)
    tel = DataLoader(BandDS(X, Y, te, bands, mean, std, False), batch_size=256, shuffle=False, num_workers=8)
    net = CNN(len(bands), nc).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best = 0.0
    for ep in range(epochs):
        net.train()
        for x, y in trl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(); F.cross_entropy(net(x), y).backward(); opt.step()
        sched.step()
        net.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in tel:
                p = net(x.to(DEV)).argmax(1).cpu()
                correct += (p == y).sum().item(); total += y.numel()
        best = max(best, correct / total)
    return best


def main():
    tifs = ensure_unzipped()
    print(f"found {len(tifs)} tif tiles; loading ...")
    X, Y, classes = load_all(tifs)
    nc = len(classes); N = len(Y)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    tr, te = perm[:int(.8 * N)], perm[int(.8 * N):]
    print(f"EuroSAT-MS: {N} tiles, {nc} classes, 13 bands, train={len(tr)} test={len(te)}")

    rgb = run(X, Y, tr, te, RGB_IDX, nc)
    print(f"  from-scratch CNN, RGB (3 band)  test acc = {rgb:.4f}")
    allb = run(X, Y, tr, te, list(range(13)), nc)
    print(f"  from-scratch CNN, all 13 bands  test acc = {allb:.4f}   ({(allb-rgb)*100:+.2f} vs RGB)")
    print(f"\n  reference: RGB-DINOv2 linear-probe = 0.9600 (frozen foundation, RGB only)")
    print(f"  MS gain over RGB (same CNN) = {(allb-rgb)*100:+.2f} pts")
    print(f"  13-band CNN vs RGB-DINOv2   = {(allb-0.96)*100:+.2f} pts")


if __name__ == "__main__":
    main()
