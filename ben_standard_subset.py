"""Build the standard BigEarthNet 10% benchmark subset.

Protocol matches SMARTIES/CROMA/SatMAE++ papers (BEN 10% column):
  - 10% of official train split (deterministic, seed 0)
  - Full official test split

Outputs:
  data/ben/standard_subset.json        -- patch metadata + split labels
  data/ben/standard_extract_list.txt   -- band-tiff paths for ben_extract_fast.py

Run: python ben_standard_subset.py
"""
import json
import random
import pyarrow.parquet as pq

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

t = pq.read_table("data/ben/metadata.parquet")
pid  = t.column("patch_id").to_pylist()
labs = t.column("labels").to_pylist()
spl  = t.column("split").to_pylist()

vocab = sorted({l for ls in labs for l in ls})
v2i   = {c: i for i, c in enumerate(vocab)}

tr_all = [i for i, s in enumerate(spl) if s == "train"]
te_all = [i for i, s in enumerate(spl) if s == "test"]

random.seed(0)
random.shuffle(tr_all)
tr_10 = tr_all[: len(tr_all) // 10]          # 10% of train split

sel = tr_10 + te_all
patches = {
    pid[i]: {"y": sorted(v2i[l] for l in labs[i]), "split": spl[i]}
    for i in sel
}

json.dump({"vocab": vocab, "patches": patches}, open("data/ben/standard_subset.json", "w"))

with open("data/ben/standard_extract_list.txt", "w") as f:
    for p in patches:
        tile = p.rsplit("_", 2)[0]
        for b in BANDS:
            f.write(f"BigEarthNet-S2/{tile}/{p}/{p}_{b}.tif\n")

n_tr = sum(v["split"] == "train" for v in patches.values())
n_te = sum(v["split"] == "test"  for v in patches.values())
print(f"standard_subset.json: {len(patches)} patches "
      f"({n_tr} train/10% + {n_te} test/full); "
      f"{len(BANDS) * len(patches)} tiff paths; {len(vocab)} classes")
print(f"  full train size: {len(tr_all)}  →  10%: {len(tr_10)}")
print(f"  full test  size: {len(te_all)}")
