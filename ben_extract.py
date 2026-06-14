"""Stream-extract the BigEarthNet subset via Python tarfile (O(1) hash lookup
per member, early-stop) -- avoids GNU tar's O(n^2) --files-from pathology.
Usage:  cat data/ben/S2.tar.gzaa data/ben/S2.tar.gzab | python ben_extract.py
"""
import sys
import tarfile

wanted = set(l.strip() for l in open("data/ben/extract_list.txt"))
total = len(wanted); done = 0
tf = tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz")
for m in tf:
    if m.name in wanted:
        tf.extract(m, "data/ben/x")
        wanted.discard(m.name); done += 1
        if done % 20000 == 0:
            print(f"  extracted {done}/{total}", flush=True)
        if not wanted:
            break
print(f"EXTRACT_DONE {done}/{total}", flush=True)
