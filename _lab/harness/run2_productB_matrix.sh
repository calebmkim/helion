#!/bin/bash
# Run-2 Product-B matrix for ONE (kernel,M,N) on ONE GPU: seeded|unseeded x quick|full x N rand-seeds.
# CSVs -> $OUTDIR/<kernel>_<M>x<N>_<mode>_<effort>_s<seed>.csv  (cold cache per run = fresh subprocess).
# Usage: run2_productB_matrix.sh KERNEL M N GPU OUTDIR NREPS
set -u
K=$1; M=$2; N=$3; GPU=$4; OUT=$5; NREPS=${6:-3}
WT=/home/calebkim/helion-new-heuristics/wt-reduction-2
PY=/home/calebkim/.conda/envs/helion/bin/python
mkdir -p "$OUT"
for effort in quick full; do
  for mode in seeded unseeded; do
    for s in $(seq 0 $((NREPS-1))); do
      tag="${K}_${M}x${N}_${mode}_${effort}_s${s}"
      log="$OUT/$tag"
      disable=""
      [ "$mode" = "unseeded" ] && disable="HELION_DISABLE_AUTOTUNER_HEURISTICS=1"
      echo "[matrix] $tag ..."
      env CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WT $disable \
        HELION_FORCE_AUTOTUNE=1 HELION_AUTOTUNE_EFFORT=$effort \
        HELION_AUTOTUNE_RANDOM_SEED=$s HELION_AUTOTUNE_LOG="$log" \
        $PY $WT/_lab/harness/run2_productB_driver.py \
        --kernel $K --M $M --N $N --mode $mode --rand-seed $s --log "$log" \
        > "$OUT/$tag.out" 2> "$OUT/$tag.err"
      echo "[matrix] $tag done (exit $?)"
    done
  done
done
echo "[matrix] ALL DONE for $K ${M}x${N}"
