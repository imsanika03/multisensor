#!/usr/bin/env bash
# Run the resolution-cascade pipeline (train) + A-F routing benchmark for one or
# more dataset configs, logging everything under results/<name>/.
#
# Usage:
#   bash run_benchmarks.sh                       # defaults: imagenette imagewoof imagenette_c
#   bash run_benchmarks.sh imagenette imagewoof  # specific configs (names = cfgs/<name>.yaml)
#   GPUS=0,1 bash run_benchmarks.sh imagenet100  # choose GPUs; ImageNet needs data_root set in its cfg
#
# Each config runs:  python main.py --cfg cfgs/<name>.yaml   (trains/loads cascade+probe, strategy A)
#              then:  python bench.py --cfg cfgs/<name>.yaml  (full A-F frontier)
set -u

cd "$(dirname "$0")"                       # always run from the project dir
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1}"

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(imagenette imagewoof imagenette_c)   # auto-downloadable defaults
fi

echo "GPUs: $CUDA_VISIBLE_DEVICES | configs: ${CONFIGS[*]}"
declare -A STATUS

for name in "${CONFIGS[@]}"; do
  cfg="cfgs/${name}.yaml"
  outdir="results/${name}"
  mkdir -p "$outdir"
  if [ ! -f "$cfg" ]; then
    echo "!! missing $cfg -- skipping"; STATUS[$name]="missing-config"; continue
  fi

  echo "==================================================================="
  echo ">> [$name] training pipeline  ($(date '+%H:%M:%S'))  -> $outdir/main.log"
  echo "==================================================================="
  if python main.py --cfg "$cfg" 2>&1 | tee "$outdir/main.log"; then
    echo ">> [$name] benchmark A-F     ($(date '+%H:%M:%S'))  -> $outdir/bench.log"
    if python bench.py --cfg "$cfg" 2>&1 | tee "$outdir/bench.log"; then
      STATUS[$name]="ok"
    else
      STATUS[$name]="bench-failed"
    fi
  else
    STATUS[$name]="train-failed"
  fi
done

echo
echo "===================== SUMMARY ====================="
for name in "${CONFIGS[@]}"; do
  printf "  %-16s %s\n" "$name" "${STATUS[$name]:-?}"
done
echo "Logs + per-config results under: results/<name>/{main,bench}.log"
