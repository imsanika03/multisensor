"""Preprocess the BigEarthNet-S2 subset into one tensor.

Reads the 12 band-tiffs per patch (10m: B02/03/04/08 @120; 20m: B05/06/07/8A/11/12
@60; 60m: B01/09 @20), bilinearly resamples each to 120x120, stacks to [12,120,120].
Saves X (uint16) + Y (19-way multi-hot) + split. Run after extraction.

Run:
  python ben_prep.py                                          # original 40k subset
  python ben_prep.py --subset data/ben/standard_subset.json --out data/ben/ben_standard.pt
"""
import json
import sys
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
ROOT = "data/ben/x/BigEarthNet-S2"


def load_patch(pid):
    tile = pid.rsplit("_", 2)[0]
    arrs = []
    for b in BANDS:
        a = tifffile.imread(f"{ROOT}/{tile}/{pid}/{pid}_{b}.tif").astype(np.float32)
        t = torch.from_numpy(a)[None, None]
        t = F.interpolate(t, size=(120, 120), mode="bilinear", align_corners=False)[0, 0]
        arrs.append(t)
    return torch.stack(arrs)                              # [12,120,120]


def main():
    subset_path = "data/ben/subset.json"
    out_path    = "data/ben/ben_subset.pt"
    if "--subset" in sys.argv:
        subset_path = sys.argv[sys.argv.index("--subset") + 1]
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]

    d = json.load(open(subset_path))
    pids = list(d["patches"].keys())
    N = len(pids)
    X = torch.zeros(N, 12, 120, 120, dtype=torch.float16)
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, arr in enumerate(ex.map(load_patch, pids)):
            X[i] = arr.half()
            if i % 5000 == 0:
                print(f"  loaded {i}/{N}", flush=True)
    Y = torch.zeros(N, 19)
    split = []
    for i, pid in enumerate(pids):
        for c in d["patches"][pid]["y"]:
            Y[i, c] = 1
        split.append(d["patches"][pid]["split"])
    torch.save({"X": X, "Y": Y, "split": split, "pids": pids, "vocab": d["vocab"]}, out_path)
    print(f"saved {out_path}  X={tuple(X.shape)}  Y={tuple(Y.shape)}")


if __name__ == "__main__":
    main()
