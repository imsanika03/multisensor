"""Build the standard BigEarthNet-S2 v1.0 10% benchmark subset.

Uses official train/test CSVs from TU Berlin (already in data/ben/splits/).
Protocol matches SMARTIES/CROMA/SatMAE++ (BEN 10% column):
  - 10% of official train split (deterministic, seed 0)
  - Full official test split

Outputs:
  data/ben/v1_subset.json        -- patch list + split (labels read in prep from JSON)
  data/ben/v1_extract_list.txt   -- tiff paths for ben_extract_fast.py

Run: python ben_v1_subset.py
"""
import json
import random

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def read_s2_ids(csv_path):
    with open(csv_path) as f:
        return [line.split(",")[0].strip() for line in f if line.strip()]


tr_all = read_s2_ids("data/ben/splits/train.csv")
te_all = read_s2_ids("data/ben/splits/test.csv")

random.seed(0)
random.shuffle(tr_all)
tr_10 = tr_all[: len(tr_all) // 10]

patches = {}
for pid in tr_10:
    patches[pid] = {"split": "train"}
for pid in te_all:
    patches[pid] = {"split": "test"}

json.dump({"patches": patches}, open("data/ben/v1_subset.json", "w"))

with open("data/ben/v1_extract_list.txt", "w") as f:
    for pid in patches:
        for b in BANDS:
            f.write(f"BigEarthNet-v1.0/{pid}/{pid}_{b}.tif\n")

n_tr = sum(1 for v in patches.values() if v["split"] == "train")
n_te = sum(1 for v in patches.values() if v["split"] == "test")
print(f"v1_subset.json: {len(patches)} patches ({n_tr} train/10% + {n_te} test/full)")
print(f"  full train: {len(tr_all)}  →  10%: {len(tr_10)}")
print(f"  full test:  {len(te_all)}")
print(f"v1_extract_list.txt: {len(patches) * len(BANDS)} tiff paths")
