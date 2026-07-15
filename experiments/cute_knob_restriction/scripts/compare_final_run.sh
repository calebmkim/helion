#!/usr/bin/env bash
# Final 5-cell x 2-shape comparison. GPU serial -> sequential. Resumable: skip a cell
# whose out.json already has status "ok".
set -u
PY=/home/dev/helion-env/bin/python
LAB=/home/dev/local/helion-rank0/_lab/matmul-autotune
PROBE=$LAB/scripts/compare_final.py
OUT=$LAB/rank0/final_compare
mkdir -p $OUT/tallies

export PYTHONPATH=/home/dev/local/helion-rank0
export HELION_FORCE_AUTOTUNE=1
export HELION_AUTOTUNE_EFFORT=full
export HELION_AUTOTUNE_BUDGET_SECONDS=600
export HELION_AUTOTUNE_RANDOM_SEED=2000

# cell: name shape_m shape_n shape_k kind  [env overrides...]
run_cell(){
  local name="$1" m="$2" n="$3" k="$4" kind="$5"; shift 5
  local outjson=$OUT/${name}.json
  if [ -f "$outjson" ] && grep -q '"status": "ok"' "$outjson"; then
    echo "[skip] $name already ok"; return
  fi
  export HELION_CACHE_DIR=$OUT/cache_${name}
  export HELION_AUTOTUNE_LOG=$OUT/tallies/${name}
  rm -rf "$HELION_CACHE_DIR"
  # backend depends on kind
  if [ "$kind" = "triton_bf16" ]; then export HELION_BACKEND=triton; else export HELION_BACKEND=cute; fi
  echo "=== [$(date +%H:%M:%S)] START $name  m$m k$k n$n  kind=$kind  ($*) ==="
  env "$@" $PY $PROBE --m $m --n $n --k $k --kind $kind \
      --log-csv "$HELION_AUTOTUNE_LOG" --out-json "$outjson" 2>&1 | grep -E "^DONE|Error|WRONG|Traceback"
  echo "=== [$(date +%H:%M:%S)] END $name ==="
}

# ---- Shape 1: 4096^3 (compute-bound canonical) ----
for shp in "4096 4096 4096 sq4096" "64 24576 4096 m64k4096n24576"; do
  set -- $shp; M=$1; N=$2; K=$3; TAG=$4
  # cute 4 cells
  run_cell ${TAG}__c1_sub1_fix0  $M $N $K cute_fp8    HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1
  run_cell ${TAG}__c2_sub1_fix1  $M $N $K cute_fp8    HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1 HELION_FUSED_ACCURACY_CHECK=1
  run_cell ${TAG}__c3_sub0_fix0  $M $N $K cute_fp8    HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0
  run_cell ${TAG}__c4_sub0_fix1  $M $N $K cute_fp8    HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0 HELION_FUSED_ACCURACY_CHECK=1
  # triton baseline
  run_cell ${TAG}__c5_triton     $M $N $K triton_bf16
done
echo "=== ALL FINAL COMPARE CELLS DONE ==="
