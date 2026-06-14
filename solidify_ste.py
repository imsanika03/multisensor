"""Solidify the cross-sensor robustness claim for STE v2 on EuroSAT.

4 held-out configs (k=1..4 informative bands) x 5 seeds, wavelength vs index.
Shows: wavelength new-sensor is stable (low std, ~RGB), index is erratic (high std,
often < RGB). DINOv2 CLS precomputed -> fast.

Run:  python solidify_ste.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ste_v2 import Fusion, DS, precompute_cls, RGB_IDX
from sat_ms_headroom import ensure_unzipped, load_all

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CONFIGS = {
    "k1 [B11]":            [11],
    "k2 [B8A,B11]":        [8, 11],
    "k3 [B6,B8A,B11]":     [5, 8, 11],
    "k4 [B5,B6,B8A,B11]":  [4, 5, 8, 11],
}
SEEDS = [0, 1, 2, 3, 4]


def train_eval(CLS, X, Y, tr, te, mean, std, nc, train_bands, new_sensor, use_wl, seed, epochs=40):
    torch.manual_seed(seed)
    net = Fusion(128, nc, use_wl).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    trl = DataLoader(DS(CLS, X, Y, tr, mean, std, train_bands, True), batch_size=512, shuffle=True, num_workers=12, drop_last=True)

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
    return ev(new_sensor), ev(RGB_IDX)


def main():
    tifs = ensure_unzipped(); X, Y, classes = load_all(tifs); nc = len(classes); N = len(Y)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    tr, te = perm[:int(.8 * N)], perm[int(.8 * N):]
    mean = X[tr].mean(dim=(0, 2, 3), keepdim=True)[0]; std = X[tr].std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6
    CLS = precompute_cls(X)
    print("solidify STE v2 cross-sensor robustness (5 seeds/config)\n")

    rows = []
    for name, heldout in CONFIGS.items():
        tb = [b for b in range(13) if b not in heldout]
        ns = sorted(set(RGB_IDX) | set(heldout))
        wl_new, idx_new, rgb_ref = [], [], []
        for seed in SEEDS:
            nw, rg = train_eval(CLS, X, Y, tr, te, mean, std, nc, tb, ns, True, seed)
            ni, _ = train_eval(CLS, X, Y, tr, te, mean, std, nc, tb, ns, False, seed)
            wl_new.append(nw); idx_new.append(ni); rgb_ref.append(rg)
            print(f"  [{name}] seed{seed}: wl_new={nw:.4f} idx_new={ni:.4f} rgb={rg:.4f}", flush=True)
        rows.append((name, len(heldout), torch.tensor(wl_new), torch.tensor(idx_new), torch.tensor(rgb_ref)))

    print("\n#### cross-sensor robustness (5 seeds) ####")
    print(f"{'held-out':<22}{'k':>3}{'wl new':>16}{'idx new':>16}{'gap':>8}{'wl std':>8}{'idx std':>9}")
    for name, k, wl, idx, rg in rows:
        print(f"{name:<22}{k:>3}{f'{wl.mean():.4f}±{wl.std():.3f}':>16}{f'{idx.mean():.4f}±{idx.std():.3f}':>16}"
              f"{(wl.mean()-idx.mean())*100:>+8.2f}{wl.std():>8.3f}{idx.std():>9.3f}")
    print("\n  claim: wl new-sensor stable (low std) across all k; idx erratic (high std); gap > 0 throughout")


if __name__ == "__main__":
    main()
