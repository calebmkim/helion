#!/bin/bash
set -e
cd /home/calebkim/helion-new-heuristics/wt-reduction
export PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction
export CUDA_VISIBLE_DEVICES=1
PY=/home/calebkim/.conda/envs/helion/bin/python
for K in rms_norm layer_norm softmax sum; do
  echo "=========== $K (GPU1) ==========="
  $PY _lab/harness/measure_g_validation.py --kernel $K > logs/validation/${K}_gpu1.out 2>&1
  echo "DONE $K rc=$?"
done
echo "ALL GPU1 DONE"
