#!/usr/bin/env bash
# Product B matrix runner: for each (kernel, shape) run quick-autotune SEEDED vs
# UNSEEDED, N random seeds each, COLD CACHE per run (fresh Triton + inductor
# cache dirs), one GPU, separate autotune_log CSVs.
#
# Usage: productB_run.sh <GPU_IDX>
# Outputs:
#   logs/productB/<kernel>_<M>_<N>_<mode>_s<seed>.csv   (convergence trace)
#   logs/productB/<kernel>_<M>_<N>_<mode>_s<seed>.driver.log  (verify block)
set -euo pipefail

GPU="${1:-2}"
GROUP="${2:-all}"   # all | A (shapes 1-2) | B (shapes 3-4)
WT=/home/calebkim/helion-new-heuristics/wt-reduction
PY=/home/calebkim/.conda/envs/helion/bin/python
OUT="$WT/logs/productB"
CACHE_ROOT=/tmp/productB_cache_$GPU
mkdir -p "$OUT"

SEEDS=(0 1 2)
# kernel M N
if [ "$GROUP" = "A" ]; then
  SHAPES=( "rms_norm 2048 16384" "rms_norm 8192 8192" )
elif [ "$GROUP" = "B" ]; then
  SHAPES=( "long_sum 8 131072" "sum 2048 16384" )
else
  SHAPES=(
    "rms_norm 2048 16384"
    "rms_norm 8192 8192"
    "long_sum 8 131072"
    "sum 2048 16384"
  )
fi

cd "$WT"
for triple in "${SHAPES[@]}"; do
  read -r KERN M N <<< "$triple"
  for MODE in seeded unseeded; do
    for S in "${SEEDS[@]}"; do
      TAG="${KERN}_${M}_${N}_${MODE}_s${S}"
      LOGBASE="$OUT/$TAG"
      CACHE="$CACHE_ROOT/$TAG"
      rm -rf "$CACHE"; mkdir -p "$CACHE/triton" "$CACHE/inductor"
      EXTRA=""
      if [ "$MODE" = "unseeded" ]; then EXTRA="HELION_DISABLE_AUTOTUNER_HEURISTICS=1"; fi
      echo "=== RUN $TAG (GPU $GPU) ==="
      env \
        TRITON_CACHE_DIR="$CACHE/triton" \
        TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        PYTHONPATH="$WT" \
        HELION_AUTOTUNE_EFFORT=quick \
        HELION_FORCE_AUTOTUNE=1 \
        HELION_AUTOTUNE_RANDOM_SEED="$S" \
        HELION_AUTOTUNE_LOG="$LOGBASE" \
        $EXTRA \
        "$PY" _lab/harness/productB_driver.py \
          --kernel "$KERN" --M "$M" --N "$N" --mode "$MODE" --rand-seed "$S" \
          --log "$LOGBASE" > "$LOGBASE.driver.log" 2>&1 || {
            echo "  FAILED $TAG (see $LOGBASE.driver.log)"; tail -5 "$LOGBASE.driver.log"; }
      # echo the verify lines for live monitoring
      grep -E "^\[verify\]" "$LOGBASE.driver.log" || true
      rm -rf "$CACHE"
    done
  done
done
echo "ALL DONE"
