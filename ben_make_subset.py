"""Build the BigEarthNet subset definition deterministically (seed 0):
  data/ben/subset.json       -- 40k patches (30k train + 10k test), 19-class multihot + split
  data/ben/extract_list.txt  -- the 480k band-tiff member paths for ben_extract.py

Reproduces exactly the subset used to make ben_subset.pt. Run after downloading
data/ben/metadata.parquet; before ben_extract.py.
Run:  python ben_make_subset.py
"""
import json
import random
import pyarrow.parquet as pq

# BigEarthNet v2.0 S2 band order used throughout the pipeline (note: no B10 in reBEN)
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

t = pq.read_table("data/ben/metadata.parquet")
pid = t.column("patch_id").to_pylist()
labs = t.column("labels").to_pylist()
spl = t.column("split").to_pylist()

vocab = sorted({l for ls in labs for l in ls})          # 19 classes
v2i = {c: i for i, c in enumerate(vocab)}

random.seed(0)                                          # deterministic subset
tr = [i for i, s in enumerate(spl) if s == "train"]
te = [i for i, s in enumerate(spl) if s == "test"]
random.shuffle(tr); random.shuffle(te)
sel = tr[:30000] + te[:10000]
patches = {pid[i]: {"y": sorted(v2i[l] for l in labs[i]), "split": spl[i]} for i in sel}

json.dump({"vocab": vocab, "patches": patches}, open("data/ben/subset.json", "w"))
with open("data/ben/extract_list.txt", "w") as f:
    for p in patches:
        tile = p.rsplit("_", 2)[0]                      # tile folder = patch_id minus "_X_Y"
        for b in BANDS:
            f.write(f"BigEarthNet-S2/{tile}/{p}/{p}_{b}.tif\n")

print(f"subset.json: {len(patches)} patches ({sum(v['split']=='train' for v in patches.values())} train / "
      f"{sum(v['split']=='test' for v in patches.values())} test); "
      f"extract_list.txt: {len(patches) * len(BANDS)} files; {len(vocab)} classes")
