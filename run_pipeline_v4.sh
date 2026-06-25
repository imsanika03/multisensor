#!/usr/bin/env bash
# Full pipeline: wait for download → extract → prep → MESA-v4 training
# Run from: /lambda/nfs/sanikabhar-georgia/multisensor/
set -e

TARBALL=/home/ubuntu/BigEarthNet-S2-v1.0.tar.gz
TARBALL_SIZE=70032164579  # 65.2 GB, from HTTP content-length

echo "[pipeline] starting at $(date)" | tee results/pipeline_v4.log

# ---------- 1. Wait for tarball download to finish ----------
echo "[pipeline] waiting for download to complete..." | tee -a results/pipeline_v4.log
PREV_SIZE=0
STALL_COUNT=0
while true; do
    CURRENT=$(stat -c%s "$TARBALL" 2>/dev/null || echo 0)
    pct=$(( CURRENT * 100 / TARBALL_SIZE ))
    echo "  downloaded: ${CURRENT} / ${TARBALL_SIZE} bytes (${pct}%)" | tee -a results/pipeline_v4.log
    if [ "$CURRENT" -ge "$TARBALL_SIZE" ]; then
        echo "[pipeline] download complete: $CURRENT bytes" | tee -a results/pipeline_v4.log
        break
    fi
    # Stall detection: if size hasn't changed for 5 consecutive checks (5 min), try to resume
    if [ "$CURRENT" -eq "$PREV_SIZE" ]; then
        STALL_COUNT=$(( STALL_COUNT + 1 ))
        if [ "$STALL_COUNT" -ge 5 ]; then
            echo "[pipeline] download stalled, resuming wget..." | tee -a results/pipeline_v4.log
            wget -c "https://zenodo.org/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz" \
                -O "$TARBALL" --progress=dot:giga >> results/pipeline_v4.log 2>&1 &
            STALL_COUNT=0
        fi
    else
        STALL_COUNT=0
    fi
    PREV_SIZE=$CURRENT
    sleep 60
done

# ---------- 2. Extract the required tiffs ----------
echo "[pipeline] starting extraction at $(date)" | tee -a results/pipeline_v4.log
cat "$TARBALL" | pigz -d | python3 ben_extract_fast.py --list data/ben/v1_extract_list.txt 2>&1 | tee -a results/pipeline_v4.log
echo "[pipeline] extraction done at $(date)" | tee -a results/pipeline_v4.log

# ---------- 3. Pack into ben_v1.pt ----------
echo "[pipeline] running ben_v1_prep.py at $(date)" | tee -a results/pipeline_v4.log
python3 ben_v1_prep.py 2>&1 | tee -a results/pipeline_v4.log
echo "[pipeline] prep done at $(date)" | tee -a results/pipeline_v4.log

# Free up local disk space
echo "[pipeline] removing tarball to free disk..." | tee -a results/pipeline_v4.log
rm -f "$TARBALL"

# ---------- 4. Run MESA-v4 training ----------
echo "[pipeline] starting MESA-v4 at $(date)" | tee -a results/pipeline_v4.log
python3 ben_mesa_v4.py 2>&1 | tee results/ben_mesa_v4.log
echo "[pipeline] MESA-v4 done at $(date)" | tee -a results/pipeline_v4.log

echo "[pipeline] ALL DONE at $(date)" | tee -a results/pipeline_v4.log
