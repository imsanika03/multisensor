"""Fusion de-risk: does (frozen DINOv2-RGB) + (small MS encoder) beat BOTH alone?

On EuroSAT-MS (same 80/20 split):
  (a) DINOv2-RGB linear-probe      frozen foundation, RGB bands only
  (b) MS-CNN (13 band, scratch)    spectral signal, no foundation semantics
  (c) Fusion                       frozen DINOv2 feat (cid) + trainable MS encoder -> head

Fusion > max(a,b)  -> the method adds value (foundation semantics + spectral) -> build
                      the proper spectral adapter and scale to BigEarthNet.
Fusion ~= max(a,b) -> late fusion is trivial; redesign (spectral tokens into DINOv2).

Run:  python fusion_derisk.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sat_ms_headroom import ensure_unzipped, load_all, run, RGB_IDX

DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMNORM_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNORM_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def dino_rgb_feats(dino, X, idx, bs=256):
    dino = dino.to(DEV).eval()
    out = []
    for i in range(0, len(idx), bs):
        chunk = X[idx[i:i + bs]][:, RGB_IDX]                 # [b,3,64,64] reflectance
        rgb = (chunk / 3000.0).clamp(0, 1).to(DEV)
        rgb = F.interpolate(rgb, size=224, mode="bilinear", align_corners=False)
        rgb = (rgb - IMNORM_M.to(DEV)) / IMNORM_S.to(DEV)
        out.append(F.normalize(dino(rgb), dim=1).cpu())
    return torch.cat(out)


def linprobe(Xtr, Ytr, Xte, Yte, nc, steps=500):
    head = nn.Linear(Xtr.shape[1], nc).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    X, Y = Xtr.to(DEV), Ytr.to(DEV)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(head(X), Y).backward(); opt.step()
    with torch.no_grad():
        return (head(Xte.to(DEV)).argmax(1).cpu() == Yte).float().mean().item()


class MSEnc(nn.Module):
    def __init__(self, in_ch=13, out=256):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                            nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(in_ch, 32), blk(32, 64), blk(64, 128), blk(128, out),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())
    def forward(self, x): return self.net(x)


class Fusion(nn.Module):
    def __init__(self, dino_dim, ms_dim, nc):
        super().__init__()
        self.enc = MSEnc(out=ms_dim)
        self.head = nn.Sequential(nn.Linear(dino_dim + ms_dim, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, nc))
    def forward(self, ms, df): return self.head(torch.cat([self.enc(ms), df], dim=1))


class FuseDS(Dataset):
    def __init__(self, X, D, Y, idx, mean, std, train):
        self.X, self.D, self.Y, self.idx, self.mean, self.std, self.train = X, D, Y, idx, mean, std, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        x = (self.X[j] - self.mean) / self.std
        if self.train:
            if torch.rand(1).item() < 0.5: x = x.flip(-1)
            if torch.rand(1).item() < 0.5: x = x.flip(-2)
        return x, self.D[j], self.Y[j]


def train_fusion(X, D, Y, tr, te, mean, std, nc, epochs=25):
    net = Fusion(D.shape[1], 256, nc).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    trl = DataLoader(FuseDS(X, D, Y, tr, mean, std, True), batch_size=256, shuffle=True, num_workers=8, drop_last=True)
    tel = DataLoader(FuseDS(X, D, Y, te, mean, std, False), batch_size=256, shuffle=False, num_workers=8)
    best = 0.0
    for ep in range(epochs):
        net.train()
        for ms, df, y in trl:
            ms, df, y = ms.to(DEV), df.to(DEV), y.to(DEV)
            opt.zero_grad(); F.cross_entropy(net(ms, df), y).backward(); opt.step()
        sched.step()
        net.eval(); cor = tot = 0
        with torch.no_grad():
            for ms, df, y in tel:
                p = net(ms.to(DEV), df.to(DEV)).argmax(1).cpu()
                cor += (p == y).sum().item(); tot += y.numel()
        best = max(best, cor / tot)
    return best


def main():
    tifs = ensure_unzipped()
    X, Y, classes = load_all(tifs); nc = len(classes); N = len(Y)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    tr, te = perm[:int(.8 * N)], perm[int(.8 * N):]
    Ytr, Yte = Y[tr], Y[te]
    print(f"EuroSAT-MS fusion de-risk: N={N} nc={nc} train={len(tr)} test={len(te)}")

    print("DINOv2-RGB features (frozen) ...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
    D = dino_rgb_feats(dino, X, list(range(N)))
    a = linprobe(D[tr], Ytr, D[te], Yte, nc)
    print(f"  (a) DINOv2-RGB linear-probe = {a:.4f}")

    b = run(X, Y, tr, te, list(range(13)), nc)
    print(f"  (b) MS-CNN (13 band)        = {b:.4f}")

    mean = X[tr].mean(dim=(0, 2, 3), keepdim=True)[0]
    std = X[tr].std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6
    c = train_fusion(X, D, Y, tr, te, mean, std, nc)
    print(f"  (c) Fusion (DINOv2 + MS enc)= {c:.4f}")

    print(f"\n  fusion vs DINOv2-RGB = {(c-a)*100:+.2f} pts;  fusion vs MS-CNN = {(c-b)*100:+.2f} pts")
    print("  VERDICT: " + ("fusion beats both -> method adds value" if c > max(a, b) + 0.002
                           else "fusion ~= max(both) -> late fusion trivial, redesign"))


if __name__ == "__main__":
    main()
