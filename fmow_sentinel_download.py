"""Download fMoW-Sentinel from Stanford Stacks.

Image path: <split>/<category>/<category>_<location_id>/<category>_<image_id>_<location_id>.tif
URL: https://stacks.stanford.edu/file/vg497cb6002/<split>/<category>/<category>_<location_id>/<category>_<image_id>_<location_id>.tif

Usage:
  python3 fmow_sentinel_download.py            # all splits
  python3 fmow_sentinel_download.py train      # single split
"""
import csv, io, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

BASE_URL  = "https://stacks.stanford.edu/file/vg497cb6002"
OUT_DIR   = "/home/ubuntu/fmow_sentinel"  # local SSD — avoids NFS contention during training
WORKERS   = 64
SPLITS    = ["train", "val", "test_gt"]
RETRY     = 3

split_arg = sys.argv[1] if len(sys.argv) > 1 else None
if split_arg:
    SPLITS = [split_arg]


def csv_url(split):
    return f"{BASE_URL}/{split}.csv"


def img_url(split, cat, loc_id, img_id):
    folder = f"{cat}_{loc_id}"
    fname  = f"{cat}_{loc_id}_{img_id}.tif"
    return f"{BASE_URL}/{split}/{cat}/{folder}/{fname}"


def img_path(split, cat, loc_id, img_id):
    folder = f"{cat}_{loc_id}"
    fname  = f"{cat}_{loc_id}_{img_id}.tif"
    return os.path.join(OUT_DIR, split, cat, folder, fname)


def download_one(url, dest):
    if os.path.exists(dest):
        return "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(RETRY):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            with open(dest + ".tmp", "wb") as f:
                f.write(data)
            os.rename(dest + ".tmp", dest)
            return "ok"
        except Exception as e:
            if attempt == RETRY - 1:
                return f"err:{e}"
            time.sleep(2 ** attempt)


def load_csv(split):
    url = csv_url(split)
    with urllib.request.urlopen(url) as r:
        text = r.read().decode()
    reader = csv.DictReader(io.StringIO(text))
    return [(row["category"], row["location_id"], row["image_id"]) for row in reader]


total_ok = total_skip = total_err = 0

for split in SPLITS:
    print(f"\n[{split}] loading CSV...", flush=True)
    # save CSV locally
    csv_dest = os.path.join(OUT_DIR, f"{split}.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(csv_dest):
        import shutil
        with urllib.request.urlopen(csv_url(split)) as r:
            with open(csv_dest, "wb") as f:
                shutil.copyfileobj(r, f)

    rows = load_csv(split)
    print(f"[{split}] {len(rows)} images, downloading with {WORKERS} workers...", flush=True)

    tasks = [(img_url(split, cat, lid, iid), img_path(split, cat, lid, iid))
             for cat, lid, iid in rows]

    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(download_one, url, dest): (url, dest) for url, dest in tasks}
        for i, fut in enumerate(as_completed(futs)):
            res = fut.result()
            if res == "ok":    ok   += 1
            elif res == "skip": skip += 1
            else:               err  += 1
            if (i + 1) % 10000 == 0:
                print(f"  [{split}] {i+1}/{len(tasks)} ok={ok} skip={skip} err={err}", flush=True)

    print(f"[{split}] DONE ok={ok} skip={skip} err={err}", flush=True)
    total_ok += ok; total_skip += skip; total_err += err

print(f"\nALL DONE: ok={total_ok} skip={total_skip} err={total_err}", flush=True)
