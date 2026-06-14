"""BigEarthNet headroom: is RGB-DINOv2 unsaturated, and does multispectral add a lot?

Multi-label (19-class) macro-mAP on the official train/test subset:
  (a) DINOv2-RGB linear probe   (frozen foundation, RGB bands)
  (b) all-12-band CNN (scratch)
If all-band mAP >> RGB-DINOv2 mAP and RGB is well below saturation, BigEarthNet
gives the headroom EuroSAT lacked -> the STE "gain" claim can be shown here.
DataParallel across all GPUs.

Run:  python ben_headroom.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RGB_IDX = [3, 2, 1]                                       # B04,B03,B02 within the 12-band stack
IMN_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMN_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def macro_mAP(scores, targets):
    aps = []
    for c in range(scores.shape[1]):
        s, t = scores[:, c], targets[:, c]
        if t.sum() == 0:
            continue
        order = s.argsort(descending=True); tt = t[order]
        prec = tt.cumsum(0) / torch.arange(1, len(tt) + 1, dtype=torch.float)
        aps.append((prec * tt).sum().item() / tt.sum().item())
    return sum(aps) / len(aps)


class BandDS(Dataset):
    def __init__(self, X, Y, idx, bands, mean, std, rgb=False, train=False):
        self.X, self.Y, self.idx, self.bands, self.mean, self.std, self.rgb, self.train = X, Y, idx, bands, mean, std, rgb, train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        x = self.X[self.idx[i]].float()
        if self.rgb:
            r = (x[RGB_IDX] / 3000.0).clamp(0, 1)
            r = F.interpolate(r[None], size=224, mode="bilinear", align_corners=False)[0]
            r = (r - IMN_M[0]) / IMN_S[0]
            return r, self.Y[self.idx[i]]
        xn = (x[self.bands] - self.mean) / self.std
        if self.train:
            if torch.rand(1).item() < 0.5: xn = xn.flip(-1)
            if torch.rand(1).item() < 0.5: xn = xn.flip(-2)
        return xn, self.Y[self.idx[i]]


@torch.no_grad()
def dino_feats(X, idx, bs=256):
    # No DataLoader (avoids fork-after-CUDA deadlock); batched interpolate on GPU.
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(DEV).eval()
    idx = list(idx); out = []
    for i in range(0, len(idx), bs):
        chunk = X[idx[i:i + bs]][:, RGB_IDX].float().to(DEV)        # [b,3,120,120]
        r = (chunk / 3000.0).clamp(0, 1)
        r = F.interpolate(r, size=224, mode="bilinear", align_corners=False)
        r = (r - IMN_M.to(DEV)) / IMN_S.to(DEV)
        out.append(F.normalize(dino(r), dim=1).cpu())
    return torch.cat(out)


def linprobe(Xtr, Ytr, Xte, Yte, steps=800):
    head = nn.Linear(Xtr.shape[1], Ytr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
    X, Y = Xtr.to(DEV), Ytr.to(DEV)
    for _ in range(steps):
        opt.zero_grad(); F.binary_cross_entropy_with_logits(head(X), Y).backward(); opt.step()
    with torch.no_grad():
        return macro_mAP(torch.sigmoid(head(Xte.to(DEV))).cpu(), Yte)


class CNN(nn.Module):
    def __init__(self, in_ch, nc):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(in_ch, 32), blk(32, 64), blk(64, 128), blk(128, 256),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, nc))
    def forward(self, x): return self.net(x)


def train_cnn(X, Y, tr, te, bands, mean, std, nc, epochs=25):
    ng = torch.cuda.device_count()
    net = CNN(len(bands), nc).to(DEV); dp = nn.DataParallel(net) if ng > 1 else net
    bs = 128 * max(ng, 1)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    trl = DataLoader(BandDS(X, Y, tr, bands, mean, std, train=True), batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    for _ in range(epochs):
        net.train()
        for x, y in trl:
            opt.zero_grad(); F.binary_cross_entropy_with_logits(dp(x.to(DEV)), y.to(DEV)).backward(); opt.step()
        sched.step()
    net.eval(); S = []
    tel = DataLoader(BandDS(X, Y, te, bands, mean, std), batch_size=bs, shuffle=False, num_workers=0)
    with torch.no_grad():
        for x, y in tel:
            S.append(torch.sigmoid(dp(x.to(DEV))).cpu())
    return macro_mAP(torch.cat(S), Y[torch.tensor(te)])


def main():
    d = torch.load("data/ben/ben_subset.pt")
    X, Y, split = d["X"], d["Y"], d["split"]
    tr = [i for i, s in enumerate(split) if s == "train"]; te = [i for i, s in enumerate(split) if s == "test"]
    nc = Y.shape[1]
    print(f"BigEarthNet subset: {len(X)} patches, {nc} classes, train={len(tr)} test={len(te)}")
    mean = X[torch.tensor(tr)].float().mean(dim=(0, 2, 3), keepdim=True)[0]
    std = X[torch.tensor(tr)].float().std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6

    print("DINOv2-RGB features ..."); D = dino_feats(X, list(range(len(X))))
    rgb_map = linprobe(D[torch.tensor(tr)], Y[torch.tensor(tr)], D[torch.tensor(te)], Y[torch.tensor(te)])
    print(f"  (a) DINOv2-RGB linear-probe  macro-mAP = {rgb_map:.4f}")
    allb = train_cnn(X, Y, tr, te, list(range(12)), mean, std, nc)
    print(f"  (b) all-12-band CNN          macro-mAP = {allb:.4f}   ({(allb-rgb_map)*100:+.2f} vs RGB)")
    print(f"\n  RGB saturated? (BEN SOTA ~0.65-0.70 macro-mAP). headroom = all-band - RGB-DINOv2 = {(allb-rgb_map)*100:+.2f} pts")


if __name__ == "__main__":
    main()
