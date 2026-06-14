"""Seed sweep + held-out-informativeness curve for wavelength-conditioned STE.

For held-out sets of increasing informativeness (k=1,3,4 informative bands), over
3 seeds each, train wavelength-STE and index-STE and report the cross-sensor
wavelength-index gap (mean +/- std). Confirms the +3.46 is (a) seed-robust and
(b) scales with how many novel informative bands the new sensor brings.

Run:  python seed_sweep.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cross_sensor_ste import STE, DS, rgb224, RGB_IDX
from sat_ms_headroom import ensure_unzipped, load_all

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CONFIGS = {                                    # held-out informative bands, increasing
    "k1 [B11]":          [11],
    "k3 [B6,B8A,B11]":   [5, 8, 11],
    "k4 [B5,B6,B8A,B11]": [4, 5, 8, 11],
}
SEEDS = [0, 1, 2]


def train_eval(X, Y, tr, te, mean, std, nc, train_bands, new_sensor, use_wl, seed, epochs=20):
    torch.manual_seed(seed)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).eval()
    net = STE(dino, nc, use_wavelength=use_wl).to(DEV)
    ng = torch.cuda.device_count(); dp = nn.DataParallel(net) if ng > 1 else net
    bs = 96 * max(ng, 1)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    trl = DataLoader(DS(X, Y, tr, mean, std, train_bands, True), batch_size=bs, shuffle=True, num_workers=16, drop_last=True)

    def evaluate(present):
        net.eval(); cor = tot = 0
        tel = DataLoader(DS(X, Y, te, mean, std, present, False), batch_size=bs, shuffle=False, num_workers=16)
        with torch.no_grad():
            for xn, rgb, mask, y in tel:
                p = dp(xn.to(DEV), rgb224(rgb.to(DEV)), mask.to(DEV)).argmax(1).cpu()
                cor += (p == y).sum().item(); tot += y.numel()
        return cor / tot

    for ep in range(epochs):
        net.train(); net.dino.eval()
        for xn, rgb, mask, y in trl:
            opt.zero_grad()
            F.cross_entropy(dp(xn.to(DEV), rgb224(rgb.to(DEV)), mask.to(DEV)), y.to(DEV)).backward(); opt.step()
        sched.step()
    return evaluate(train_bands), evaluate(new_sensor)


def main():
    tifs = ensure_unzipped(); X, Y, classes = load_all(tifs); nc = len(classes); N = len(Y)
    g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
    tr, te = perm[:int(.8 * N)], perm[int(.8 * N):]
    mean = X[tr].mean(dim=(0, 2, 3), keepdim=True)[0]; std = X[tr].std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6

    results = {name: {"new_gap": [], "ctrl_gap": []} for name in CONFIGS}
    for name, heldout in CONFIGS.items():
        tb = [b for b in range(13) if b not in heldout]
        ns = sorted(set(RGB_IDX) | set(heldout))
        for seed in SEEDS:
            sw, nw = train_eval(X, Y, tr, te, mean, std, nc, tb, ns, True, seed)
            si, ni = train_eval(X, Y, tr, te, mean, std, nc, tb, ns, False, seed)
            results[name]["new_gap"].append((nw - ni) * 100)
            results[name]["ctrl_gap"].append((sw - si) * 100)
            print(f"  [{name}] seed{seed}: wl(new={nw:.4f}) idx(new={ni:.4f}) "
                  f"NEW-gap={(nw-ni)*100:+.2f}  ctrl-gap={(sw-si)*100:+.2f}", flush=True)

    print("\n#### informativeness curve (cross-sensor wavelength-index gap, mean +/- std over seeds) ####")
    print(f"{'held-out':<22}{'#bands':>7}{'NEW-sensor gap':>20}{'control gap':>16}")
    for name, heldout in CONFIGS.items():
        ng = torch.tensor(results[name]["new_gap"]); cg = torch.tensor(results[name]["ctrl_gap"])
        print(f"{name:<22}{len(heldout):>7}{f'{ng.mean():+.2f} ± {ng.std():.2f}':>20}{f'{cg.mean():+.2f} ± {cg.std():.2f}':>16}")
    print("\n  expect: NEW-sensor gap grows with #held-out informative bands; control ~0")


if __name__ == "__main__":
    main()
