"""Extract S2-patch → labels mapping from the BigEarthNet-S1 tarball.

Pipe through pigz for fast parallel decompression:
  cat /home/ubuntu/BigEarthNet-S1-v1.0.tar.gz | pigz -d | python3 ben_build_label_map.py

Each S1 patch JSON contains `corresponding_s2_patch` and `labels` fields.
Outputs data/ben/s2_label_map.json.
"""
import json
import sys
import tarfile

OUT = "data/ben/s2_label_map.json"

label_map = {}   # s2_patch_id -> list[str] labels
done = 0

print("Streaming S1 tarball (via stdin) ...", flush=True)
tf = tarfile.open(fileobj=sys.stdin.buffer, mode="r|")
for m in tf:
    if not m.name.endswith("_labels_metadata.json"):
        continue
    f = tf.extractfile(m)
    if f is None:
        continue
    d = json.loads(f.read())
    s2_pid = d.get("corresponding_s2_patch")
    labels = d.get("labels", [])
    if s2_pid:
        label_map[s2_pid] = labels
    done += 1
    if done % 50000 == 0:
        print(f"  processed {done} S1 patches, {len(label_map)} s2 entries", flush=True)

print(f"Total S1 patches: {done}, unique S2 refs: {len(label_map)}", flush=True)

# Check coverage against v1_subset
subset = json.load(open("data/ben/v1_subset.json"))
pids   = list(subset["patches"].keys())
missing = [p for p in pids if p not in label_map]
print(f"Coverage: {len(pids) - len(missing)}/{len(pids)} S2 patches found", flush=True)
if missing:
    print(f"  MISSING {len(missing)} patches:", flush=True)
    for p in missing[:10]:
        print(f"    {p}", flush=True)

json.dump(label_map, open(OUT, "w"))
print(f"Saved {OUT}", flush=True)
