#!/bin/bash
set -e
cd /home/calebkim/helion-new-heuristics/wt-reduction
export PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction
export CUDA_VISIBLE_DEVICES=3
PY=/home/calebkim/.conda/envs/helion/bin/python
for K in long_sum cross_entropy kl_div jsd; do
  echo "=========== $K (GPU3) ==========="
  $PY _lab/harness/measure_g_validation.py --kernel $K > logs/validation/${K}_gpu3.out 2>&1
  echo "DONE $K rc=$?"
done
echo "ALL GPU3 DONE"
