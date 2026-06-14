"""Worker: run assigned (config_index:seed) cells on the single visible GPU.
Prints `RESULT <ci> <seed> <wl_new> <idx_new> <rgb>` per cell.
Launched 8x (one per GPU) by launch_solidify.sh.
"""
import argparse
import torch

from solidify_ste import train_eval, CONFIGS
from ste_v2 import precompute_cls, RGB_IDX
from sat_ms_headroom import ensure_unzipped, load_all

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)   # "ci:seed,ci:seed,..."
jobs = [tuple(map(int, j.split(":"))) for j in ap.parse_args().jobs.split(",") if j]

tifs = ensure_unzipped(); X, Y, classes = load_all(tifs); nc = len(classes); N = len(Y)
g = torch.Generator().manual_seed(0); perm = torch.randperm(N, generator=g)
tr, te = perm[:int(.8 * N)], perm[int(.8 * N):]
mean = X[tr].mean(dim=(0, 2, 3), keepdim=True)[0]; std = X[tr].std(dim=(0, 2, 3), keepdim=True)[0] + 1e-6
CLS = precompute_cls(X)
cfgs = list(CONFIGS.items())

for ci, seed in jobs:
    name, heldout = cfgs[ci]
    tb = [b for b in range(13) if b not in heldout]
    ns = sorted(set(RGB_IDX) | set(heldout))
    nw, rg = train_eval(CLS, X, Y, tr, te, mean, std, nc, tb, ns, True, seed)
    ni, _ = train_eval(CLS, X, Y, tr, te, mean, std, nc, tb, ns, False, seed)
    print(f"RESULT {ci} {seed} {nw:.4f} {ni:.4f} {rg:.4f}", flush=True)
