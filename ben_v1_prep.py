"""Pack extracted BigEarthNet-S2 v1.0 tiffs into a tensor.

Reads labels from each patch's *_labels_metadata.json (43-class names →
19-class multi-hot via torchgeo's label_converter).
Resamples all bands to 120×120.

Run: python ben_v1_prep.py
"""
import json
import glob
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
ROOT  = "data/ben/x/BigEarthNet-v1.0"

# 43-class name list (torchgeo ordering)
CLASSES_43 = [
    "Continuous urban fabric", "Discontinuous urban fabric", "Industrial or commercial units",
    "Road and rail networks and associated land", "Port areas", "Airports",
    "Mineral extraction sites", "Dump sites", "Construction sites",
    "Green urban areas", "Sport and leisure facilities",
    "Non-irrigated arable land", "Permanently irrigated land", "Rice fields",
    "Vineyards", "Fruit trees and berry plantations", "Olive groves", "Pastures",
    "Annual crops associated with permanent crops", "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest",
    "Natural grassland", "Moors and heathland", "Sclerophyllous vegetation",
    "Transitional woodland/shrub", "Beaches, dunes, sands", "Bare rock",
    "Sparsely vegetated areas", "Burnt areas", "Inland marshes", "Peatbogs",
    "Salt marshes", "Saline marshes", "Intertidal flats",
    "Water courses", "Water bodies", "Coastal lagoons", "Estuaries", "Sea and ocean",
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES_43)}

# 43 → 19 class mapping (torchgeo label_converter)
LABEL_CONV = {
    0:0, 1:0, 2:1, 11:2, 12:2, 13:2, 14:3, 15:3, 16:3, 18:3,
    17:4, 19:5, 20:6, 21:7, 22:8, 23:9, 24:10, 25:11, 31:11,
    26:12, 27:12, 28:13, 29:14, 33:15, 34:15, 35:16, 36:16,
    38:17, 39:17, 40:18, 41:18, 42:18,
}
NC = 19


def load_patch(pid):
    arrs = []
    for b in BANDS:
        path = f"{ROOT}/{pid}/{pid}_{b}.tif"
        a = tifffile.imread(path).astype(np.float32)
        t = torch.from_numpy(a)[None, None]
        t = F.interpolate(t, size=(120, 120), mode="bilinear", align_corners=False)[0, 0]
        arrs.append(t)
    return torch.stack(arrs)   # [12, 120, 120]


def load_label(pid):
    j = glob.glob(f"{ROOT}/{pid}/*_labels_metadata.json")[0]
    labels = json.load(open(j))["labels"]
    idxs43 = [CLASS2IDX[l] for l in labels if l in CLASS2IDX]
    idxs19 = list({LABEL_CONV[i] for i in idxs43 if i in LABEL_CONV})
    y = torch.zeros(NC)
    for i in idxs19:
        y[i] = 1
    return y


def main():
    d     = json.load(open("data/ben/v1_subset.json"))
    pids  = list(d["patches"].keys())
    split = [d["patches"][p]["split"] for p in pids]
    N     = len(pids)

    X = torch.zeros(N, 12, 120, 120, dtype=torch.float16)
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, arr in enumerate(ex.map(load_patch, pids)):
            X[i] = arr.half()
            if i % 5000 == 0:
                print(f"  images {i}/{N}", flush=True)

    Y = torch.zeros(N, NC)
    with ThreadPoolExecutor(max_workers=32) as ex:
        for i, y in enumerate(ex.map(load_label, pids)):
            Y[i] = y
            if i % 5000 == 0:
                print(f"  labels {i}/{N}", flush=True)

    torch.save({"X": X, "Y": Y, "split": split, "pids": pids}, "data/ben/ben_v1.pt")
    print(f"saved data/ben/ben_v1.pt  X={tuple(X.shape)}  Y={tuple(Y.shape)}")


if __name__ == "__main__":
    main()
