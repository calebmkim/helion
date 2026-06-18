#!/usr/bin/env bash
# Product B COMPREHENSIVE matrix runner across the full 8-kernel active curriculum
# (after the persistent-seed round-trip fix in ReductionLoopSpec._encode_flat_value,
# commit 664a9524). Identical protocol/budget to productB_postfix_run.sh:
#   quick effort, HELION_FORCE_AUTOTUNE=1, COLD cache per run (fresh Triton +
#   inductor cache dirs), one GPU, seeds {0,1,2}, NOT HELION_SKIP_CACHE, full
#   max_generations=5 (quick default). Only difference per mode = presence of the
#   compiler seed in gen0.
#
# Usage: productB_full_run.sh <GPU_IDX> <GROUP>
#   GROUP = A (T1 norm/reduce kernels) | B (T2 + loss kernels) | all
# Split A on one GPU and B on another to parallelize WITHOUT two timing runs on
# one GPU.
#
# Outputs:
#   logs/productB_full/<kernel>_<M>_<N>_<mode>_s<seed>.csv         (convergence trace)
#   logs/productB_full/<kernel>_<M>_<N>_<mode>_s<seed>.driver.log  (verify block)
set -euo pipefail

GPU="${1:-2}"
GROUP="${2:-all}"
WT=/home/calebkim/helion-new-heuristics/wt-reduction
PY=/home/calebkim/.conda/envs/helion/bin/python
OUT="$WT/logs/productB_full"
CACHE_ROOT=/tmp/productB_full_cache_$GPU
mkdir -p "$OUT"

SEEDS=(0 1 2)

# kernel M N  (M,N for layer_norm/rms_norm/sum/long_sum; N rows x V vocab for the
# loss kernels and softmax/kl_div/jsd -- the driver build fns handle the layout).
# Group A = T1 norm/reduce (rms_norm x2, layer_norm, long_sum lifted to M=256, sum).
# Group B = T1 cross_entropy + the 3 T2 kernels (softmax_two_pass, kl_div, jsd).
if [ "$GROUP" = "A" ]; then
  SHAPES=(
    "rms_norm 2048 16384"
    "rms_norm 8192 8192"
    "layer_norm 4096 15872"
    "long_sum 256 131072"
    "sum 2048 16384"
  )
elif [ "$GROUP" = "B" ]; then
  SHAPES=(
    "cross_entropy 8192 65536"
    "softmax_two_pass 4096 16384"
    "kl_div 4096 65536"
    "jsd 8192 65536"
  )
else
  SHAPES=(
    "rms_norm 2048 16384"
    "rms_norm 8192 8192"
    "layer_norm 4096 15872"
    "long_sum 256 131072"
    "sum 2048 16384"
    "cross_entropy 8192 65536"
    "softmax_two_pass 4096 16384"
    "kl_div 4096 65536"
    "jsd 8192 65536"
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
            echo "  FAILED $TAG (see $LOGBASE.driver.log)"; tail -8 "$LOGBASE.driver.log"; }
      grep -E "^\[verify\]" "$LOGBASE.driver.log" || true
      rm -rf "$CACHE"
    done
  done
done
rm -rf "$CACHE_ROOT"
echo "ALL DONE (GROUP=$GROUP GPU=$GPU)"
