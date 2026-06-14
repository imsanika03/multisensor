#!/usr/bin/env bash
# Fan the 20 (config x seed) cells across all 8 GPUs, then aggregate.
cd "$(dirname "$0")"
rm -f /tmp/sol_gpu_*.log

# build 20 jobs: ci in 0..3, seed in 0..4
JOBS=(); for ci in 0 1 2 3; do for s in 0 1 2 3 4; do JOBS+=("$ci:$s"); done; done
NG=8
# round-robin partition into NG groups
for g in $(seq 0 $((NG-1))); do
  grp=""
  for i in "${!JOBS[@]}"; do [ $((i % NG)) -eq "$g" ] && grp="$grp,${JOBS[$i]}"; done
  grp="${grp#,}"
  [ -z "$grp" ] && continue
  CUDA_VISIBLE_DEVICES=$g python solidify_worker.py --jobs "$grp" > "/tmp/sol_gpu_$g.log" 2>&1 &
done
wait
echo "=== all workers done; aggregating ==="
python - <<'PY'
import glob, torch
from solidify_ste import CONFIGS
rows = {ci: {"wl": [], "idx": [], "rgb": []} for ci in range(len(CONFIGS))}
for f in glob.glob("/tmp/sol_gpu_*.log"):
    for line in open(f):
        if line.startswith("RESULT"):
            _, ci, seed, wl, idx, rgb = line.split()
            ci = int(ci); rows[ci]["wl"].append(float(wl)); rows[ci]["idx"].append(float(idx)); rows[ci]["rgb"].append(float(rgb))
names = list(CONFIGS.keys()); ks = [len(v) for v in CONFIGS.values()]
print(f"{'held-out':<22}{'k':>3}{'n':>3}{'wl new':>16}{'idx new':>16}{'gap':>8}{'wl std':>8}{'idx std':>9}")
for ci in range(len(CONFIGS)):
    wl = torch.tensor(rows[ci]["wl"]); idx = torch.tensor(rows[ci]["idx"])
    if len(wl) == 0: continue
    print(f"{names[ci]:<22}{ks[ci]:>3}{len(wl):>3}{f'{wl.mean():.4f}±{wl.std():.3f}':>16}"
          f"{f'{idx.mean():.4f}±{idx.std():.3f}':>16}{(wl.mean()-idx.mean())*100:>+8.2f}{wl.std():>8.3f}{idx.std():>9.3f}")
print("\nclaim: wl new-sensor stable (low std) across k; idx erratic (high std); gap grows with k")
PY
