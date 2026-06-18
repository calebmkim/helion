#!/bin/bash
# Fresh subprocess PER shape. Collects REFEREE_JSON lines.
set -u
PY=/home/calebkim/.conda/envs/helion/bin/python
WT=/home/calebkim/helion-new-heuristics/wt-reduction
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$WT
cd "$WT" || exit 1

run() {  # kernel m n nruns
  $PY _lab/harness/referee_verify.py --kernel "$1" --m "$2" --n "$3" --n-runs "$4" --seed 0 2>/dev/null | grep REFEREE_JSON
}

KERNEL=$1
case "$KERNEL" in
  long_sum)
    for shp in "1 32768" "2 65536" "4 130000" "8 131072" "16 262144"; do
      set -- $shp; run long_sum "$1" "$2" 15
    done
    ;;
  long_sum_bigM)  # robustness: same N, larger M to lift latency above noise floor
    for shp in "256 131072" "512 131072" "256 262144"; do
      set -- $shp; run long_sum "$1" "$2" 11
    done
    ;;
  sum)
    for shp in "2048 1024" "2048 4096" "2048 16384" "4096 1536" "4096 5120" \
               "8192 256" "8192 4096" "32768 256" "32768 1024"; do
      set -- $shp; run sum "$1" "$2" 9
    done
    ;;
  rms_norm)
    for shp in "2048 16384" "4096 5120" "8192 4096" "32768 256"; do
      set -- $shp; run rms_norm "$1" "$2" 9
    done
    ;;
esac
