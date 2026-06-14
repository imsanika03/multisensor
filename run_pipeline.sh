#!/bin/bash
# Pipeline: extract (pigz + threaded writes) → prep → train
set -e
cd /home/ubuntu/sanikabhar/modelcascade
LOG=pipeline.log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

log "=== Pipeline restarted (fast extraction with pigz) ==="

# Extract — pigz decompresses on all cores, python writes in threads
log "=== Extracting tiffs (pigz + 32 write threads) ==="
cat data/ben/S2.tar.gzaa data/ben/S2.tar.gzab | pigz -d | python ben_extract_fast.py 2>&1 | tee -a $LOG

# Prep tensor
log "=== Building ben_subset.pt ==="
python ben_prep.py 2>&1 | tee -a $LOG

# Run STE
log "=== Running ben_ste.py (conv SpectralEnc) ==="
python ben_ste.py 2>&1 | tee -a $LOG

log "=== Pipeline complete ==="
