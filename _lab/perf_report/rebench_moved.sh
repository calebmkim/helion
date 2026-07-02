#!/usr/bin/env bash
# Targeted rebench of ONLY the (kernel,dtype,shape) cells whose config moved after the
# NARROW_W1 removal. One fresh foreground process per kernel; merges into existing JSON.
set -u
RESULTS="${1:?results dir}"
WT=/home/dev/local/helion-redesign
PY=/home/dev/helion/.venv/bin/python
BENCH="$WT/_lab/perf_report/perf_report_bench.py"

run() {  # corpus kernel only-shapes
  echo "==================== $1/$2 ===================="
  cd /tmp && HELION_AUTOTUNE_EFFORT=none PERF_COMPILE_TIMEOUT_S=180 HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH="$WT" "$PY" "$BENCH" --corpus "$1" --kernel "$2" \
    --only-shapes "$3" --out-dir "$RESULTS" 2>&1 | grep -E "^\[cell|^\[ERR|MERGED|DONE"
}

run curriculum layer_norm            "16384x896:bf16"
run curriculum rms_norm              "16384x896:bf16"
run curriculum softmax               "131072x128:bf16,8192x896:bf16"
run curriculum welford               "16384x896:bf16"
run transfer   dynamic_quant         "16384x1024:bf16,8192x768:bf16"
run transfer   fused_add_rmsnorm     "8192x768:bf16,8192x1024:bf16"
run transfer   gated_rmsnorm         "8192x768:bf16,8192x1024:bf16"
run transfer   scaled_masked_softmax "16384x1024:bf16"
run vllm       per_token_group_fp8_quant "128x4096x128:native,8192x4096x128:native,2048x8192x128:native"
echo "REBENCH REAL-CORPORA COMPLETE"
