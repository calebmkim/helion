#!/usr/bin/env bash
# Product B POST-FIX matrix runner (after the persistent-seed round-trip fix in
# ReductionLoopSpec._encode_flat_value). Identical protocol/budget to
# productB_run.sh; only the output dir (logs/productB_postfix) and the shape set
# (the 3 shapes where the persistent lever matters) differ.
#
# Usage: productB_postfix_run.sh <GPU_IDX>
set -euo pipefail

GPU="${1:-3}"
WT=/home/calebkim/helion-new-heuristics/wt-reduction
PY=/home/calebkim/.conda/envs/helion/bin/python
OUT="$WT/logs/productB_postfix"
CACHE_ROOT=/tmp/productB_postfix_cache_$GPU
mkdir -p "$OUT"

SEEDS=(0 1 2)
SHAPES=(
  "rms_norm 2048 16384"
  "rms_norm 8192 8192"
  "long_sum 8 131072"
)

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
      grep -E "^\[verify\]" "$LOGBASE.driver.log" || true
      rm -rf "$CACHE"
    done
  done
done
echo "ALL DONE"
