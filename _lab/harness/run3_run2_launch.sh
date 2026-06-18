#!/bin/bash
# SINGLE AUTONOMOUS LAUNCHER for the Run-2 matrix.
# Each GPU process runs the WHOLE thing unattended: GLOBAL quick-pass first, then
# GLOBAL full-pass (--efforts quick,full), reps=1, checkpointed per shape/arm and
# per generation (autotune CSVs in _lab/logs/run3/pb/). Resumable: re-running skips
# completed cells. Uses 2 GPUs only (1 and 2) to avoid hogging.
#
#   bash run3_run2_launch.sh           # launches both GPU workers detached
#
# Monitor:  tail -f _lab/logs/run3/launch_logs/run2_gpu{1,2}.log
# Results:  _lab/logs/run3/run2_<kernel>_<MxN>.json   (per-shape, updated live)
set -u
WT=/home/calebkim/helion-new-heuristics/wt-reduction-2
PY=/home/calebkim/.conda/envs/helion/bin/python
LOGD=$WT/_lab/logs/run3/launch_logs
mkdir -p "$LOGD"

launch() {  # $1=gpu  $2=kernels  $3=tag
  CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$WT nohup "$PY" \
    "$WT/_lab/harness/run3_run2_matrix.py" \
    --kernels "$2" --efforts quick,full --reps 1 --gpu "$1" \
    > "$LOGD/run2_$3.log" 2>&1 &
  echo "launched GPU$1 ($3) pid=$! kernels=$2"
}

# Balanced split (heavy autotune kernels spread across both GPUs):
launch 2 cross_entropy,kl_div,jsd,sum,long_sum gpu2
launch 3 welford,softmax,layer_norm,rms_norm     gpu3
echo "Run-2 mass-run launched on GPU2+GPU3 (quick-pass then full-pass, autonomous)."
