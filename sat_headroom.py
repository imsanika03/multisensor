"""Satellite headroom check: is DINOv2 actually WEAK on RGB remote-sensing?

Measures linear-probe / kNN accuracy of frozen DINOv2 vs an ImageNet-resnet50
baseline on EuroSAT, vs known supervised SOTA (~98.6%). If DINOv2 is already
near-SOTA, RGB scene classification is saturated -> the real headroom is
multispectral (DINOv2 ignores 10 of Sentinel-2's 13 bands). If DINOv2 lags,
there's room for an RGB adaptation method.

Run:  python sat_headroom.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset

from utils import ResizedDataset, resize_transform

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RES = 224


class RGB:
    def __init__(self, base): self.base = base; self.classes = base.classes
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, y = self.base[i]; return img.convert("RGB"), y


@torch.no_grad()
def feats(model, ds, idx, kind, nw=8):
    loader = DataLoader(ResizedDataset(Subset(ds, idx), resize_transform(RES)),
                        batch_size=128, shuffle=False, num_workers=nw, pin_memory=True)
    model = model.to(DEV).eval(); out = []
    for x, _ in loader:
        x = x.to(DEV)
        e = model(x) if kind == "dino" else torch.flatten(model(x), 1)
        out.append(F.normalize(e, dim=1).cpu())
    return torch.cat(out)


def linprobe(Xtr, Ytr, Xte, Yte, nc, steps=500):
    head = nn.Linear(Xtr.shape[1], nc).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    X, Y = Xtr.to(DEV), Ytr.to(DEV)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(head(X), Y).backward(); opt.step()
    with torch.no_grad():
        return (head(Xte.to(DEV)).argmax(1).cpu() == Yte).float().mean().item()


def proto_knn(Xtr, Ytr, Xte, Yte, nc):
    protos = torch.stack([F.normalize(Xtr[Ytr == c].mean(0), dim=0) for c in range(nc)])
    return ((Xte @ protos.t()).argmax(1) == Yte).float().mean().item()


def main():
    base = torchvision.datasets.EuroSAT(root="data", download=True)
    ds = RGB(base); nc = len(ds.classes); N = len(ds)
    Y = torch.tensor([base.samples[i][1] for i in range(N)])
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    tr, te = perm[:int(.8 * N)].tolist(), perm[int(.8 * N):].tolist()
    Ytr, Yte = Y[torch.tensor(tr)], Y[torch.tensor(te)]
    print(f"EuroSAT: {N} imgs, {nc} classes, train={len(tr)} test={len(te)}")

    print("DINOv2 vitb14 ...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False)
    Dtr, Dte = feats(dino, ds, tr, "dino"), feats(dino, ds, te, "dino")
    print(f"  DINOv2  linear-probe = {linprobe(Dtr, Ytr, Dte, Yte, nc):.4f}   kNN-proto = {proto_knn(Dtr, Ytr, Dte, Yte, nc):.4f}")

    print("ImageNet resnet50 ...")
    rn = torchvision.models.resnet50(weights="DEFAULT")
    feat_net = nn.Sequential(*list(rn.children())[:-1])
    Rtr, Rte = feats(feat_net, ds, tr, "rn"), feats(feat_net, ds, te, "rn")
    print(f"  resnet50 linear-probe = {linprobe(Rtr, Ytr, Rte, Yte, nc):.4f}   kNN-proto = {proto_knn(Rtr, Ytr, Rte, Yte, nc):.4f}")

    print("\n  reference: EuroSAT supervised SOTA ~0.986")
    print("  headroom = SOTA - DINOv2 linear-probe  (large -> RGB method possible; ~0 -> go multispectral)")


if __name__ == "__main__":
    main()
