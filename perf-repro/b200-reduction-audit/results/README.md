# B200 Reduction Audit Artifacts

The definitive run contains 114/114 planned cells and was recorded on physical
GPU 1 (`NVIDIA B200`, SM100) with `CUDA_VISIBLE_DEVICES=1`.

- [Full results](full/SUMMARY.md)
- [Smoke results](smoke/SUMMARY.md)
- `full/raw/`: one complete JSON record per measured cell
- `smoke/raw/`: one validation record per kernel, including Dynamo graph counts

## Charts

- [All cohorts](charts/all_cohorts_relative_performance.png)
- [Examples/general kernels](charts/examples_general_relative_performance.png)
- [General AOT](charts/general_aot_relative_performance.png)
- [Original kernels](charts/original_relative_performance.png)
- [vLLM](charts/vllm_relative_performance.png)

Ratios above 1 mean the reduction seed is faster.

| Cohort | vs default | vs `torch.compile` | vs SM100 AOT |
|---|---:|---:|---:|
| General AOT | 1.029 | 1.124 | 0.917 |
| Original kernels | 6.028 | 1.013 | n/a |
| vLLM | 1.918 | 1.388 | 0.918 |

All 16 smoke references compiled as one Dynamo graph with zero graph breaks.
In the full run, all 114 `torch.compile` arms and all 78 exact-key SM100 AOT
arms passed accuracy. The seed and unseeded default both failed the BF16
`grad_x` tolerance for `rms_norm_bwd(2048, 11008)` with the same error. That
finding reproduced in the initial full pass, a targeted repeat, and the
definitive pass, and is excluded from performance aggregates.

## Reproduction

From the Helion repository root:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  python perf-repro/b200-reduction-audit/run_all.py \
  --smoke --force --timeout-seconds 1800

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  python perf-repro/b200-reduction-audit/run_all.py \
  --force --timeout-seconds 1800

python perf-repro/b200-reduction-audit/aggregate.py
python perf-repro/b200-reduction-audit/plot_performance.py
```

The raw records identify Helion commit `2d753a5c6`, PyTorch
`2.12.0+cu130` (`7661cd9c6`), Triton `3.7.0`, CUDA runtime `13.0`, and NVIDIA
driver `595.71.05`.
