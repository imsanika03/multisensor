"""Fast extraction: pigz handles parallel gzip decompression, threads handle writes.
Usage: cat data/ben/S2.tar.gzaa data/ben/S2.tar.gzab | pigz -d | python ben_extract_fast.py
"""
import sys
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor

WRITE_WORKERS = 32
OUT = "data/ben/x"

wanted = set(l.strip() for l in open("data/ben/extract_list.txt"))
total = len(wanted); done = 0


def write_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


tf = tarfile.open(fileobj=sys.stdin.buffer, mode="r|")  # raw stream — pigz decompresses
with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as ex:
    for m in tf:
        if m.name in wanted:
            data = tf.extractfile(m).read()
            ex.submit(write_file, os.path.join(OUT, m.name), data)
            wanted.discard(m.name); done += 1
            if done % 20000 == 0:
                print(f"  extracted {done}/{total}", flush=True)
            if not wanted:
                break

print(f"EXTRACT_DONE {done}/{total}", flush=True)
