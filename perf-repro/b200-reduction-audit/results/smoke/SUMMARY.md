# B200 Reduction-Heuristic Audit Results

Recorded 16 of 114 planned cells. Ratios above 1 mean the reduction seed is faster.

## Environment

- Helion: `2d753a5c6e855638a8947a296484e187b4ea0bbf` (`b200-reduction-audit-run`)
- PyTorch: `2.12.0+cu130` (`7661cd9c6b841b62b7f411aa52ec51f05457263b`)
- Triton: `3.7.0`
- CUDA runtime / driver: `13.0` / `595.71.05`
- GPU: physical `1`, logical `0`, `NVIDIA B200` (SM100)
- `CUDA_VISIBLE_DEVICES=1`; metadata identical across cells: `True`

## Heuristic Coverage

- `triton_reduction_tile_sm100`: 10 cells
- `triton_reduction_user_tile_sm100`: 6 cells
- Reduction seed present: 16/16 cells

## Cohorts

Entries are geomean [min, max] (count).

| Cohort | vs default | vs torch.compile | vs SM100 AOT |
|---|---:|---:|---:|
| `general_aot` | 1.037 [0.863, 1.192] (n=4) | 0.978 [0.768, 1.342] (n=4) | 0.892 [0.698, 1.004] (n=4) |
| `original` | 4.813 [1.180, 19.566] (n=6) | 0.999 [0.825, 1.254] (n=6) | n/a (n=0) |
| `vllm` | 1.472 [0.918, 3.741] (n=6) | 1.239 [0.990, 2.057] (n=6) | 0.980 [0.895, 1.014] (n=6) |

## Kernels

| Kernel | Dtype | Cells | vs default | vs torch.compile | vs SM100 AOT |
|---|---|---:|---:|---:|---:|
| `rms_norm` | bf16 | 1/6 | 0.863 [0.863, 0.863] (n=1) | 0.768 [0.768, 0.768] (n=1) | 1.004 [1.004, 1.004] (n=1) |
| `layer_norm` | fp16 | 1/6 | 1.142 [1.142, 1.142] (n=1) | 0.943 [0.943, 0.943] (n=1) | 0.946 [0.946, 0.946] (n=1) |
| `softmax` | fp16 | 1/6 | 1.192 [1.192, 1.192] (n=1) | 0.940 [0.940, 0.940] (n=1) | 0.953 [0.953, 0.953] (n=1) |
| `cross_entropy` | bf16 | 1/6 | 0.986 [0.986, 0.986] (n=1) | 1.342 [1.342, 1.342] (n=1) | 0.698 [0.698, 0.698] (n=1) |
| `kl_div` | bf16 | 1/6 | 2.291 [2.291, 2.291] (n=1) | 0.825 [0.825, 0.825] (n=1) | n/a (n=0) |
| `jsd` | bf16 | 1/6 | 2.261 [2.261, 2.261] (n=1) | 0.876 [0.876, 0.876] (n=1) | n/a (n=0) |
| `fused_linear_jsd` | bf16 | 1/6 | 1.180 [1.180, 1.180] (n=1) | 0.914 [0.914, 0.914] (n=1) | n/a (n=0) |
| `grpo` | bf16 | 1/6 | 5.496 [5.496, 5.496] (n=1) | 1.121 [1.121, 1.121] (n=1) | n/a (n=0) |
| `rms_norm_bwd` | bf16 | 1/6 | 18.918 [18.918, 18.918] (n=1) | 1.068 [1.068, 1.068] (n=1) | n/a (n=0) |
| `layer_norm_bwd` | bf16 | 1/6 | 19.566 [19.566, 19.566] (n=1) | 1.254 [1.254, 1.254] (n=1) | n/a (n=0) |
| `dynamic_per_token_scaled_fp8_quant` | native | 1/9 | 1.697 [1.697, 1.697] (n=1) | 0.999 [0.999, 0.999] (n=1) | 1.014 [1.014, 1.014] (n=1) |
| `per_token_group_fp8_quant` | native | 1/9 | 0.918 [0.918, 0.918] (n=1) | 1.019 [1.019, 1.019] (n=1) | 0.895 [0.895, 0.895] (n=1) |
| `rms_norm_dynamic_per_token_quant` | native | 1/9 | 3.741 [3.741, 3.741] (n=1) | 0.990 [0.990, 0.990] (n=1) | 0.992 [0.992, 0.992] (n=1) |
| `rms_norm_per_block_quant` | native | 1/9 | 1.489 [1.489, 1.489] (n=1) | 1.486 [1.486, 1.486] (n=1) | 1.007 [1.007, 1.007] (n=1) |
| `silu_and_mul_per_block_quant` | native | 1/9 | 1.115 [1.115, 1.115] (n=1) | 1.177 [1.177, 1.177] (n=1) | 0.993 [0.993, 0.993] (n=1) |
| `fused_qk_norm_rope` | native | 1/9 | 1.050 [1.050, 1.050] (n=1) | 2.057 [2.057, 2.057] (n=1) | 0.982 [0.982, 0.982] (n=1) |

## Per Shape

Configs show `block_sizes`, `num_warps`, and `num_stages`; raw JSON contains each complete config.

| Kernel | Shape | seed config | default config | AOT config | seed us | default/seed | tc/seed | AOT/seed |
|---|---|---|---|---|---:|---:|---:|---:|
| `cross_entropy` | `[2048, 32000]` | `bs=[1];w=32;s=1` | `bs=[1];w=4;s=1` | `bs=[1];w=16;s=8` | 74.760 | 0.986 | 1.342 | 0.698 |
| `layer_norm` | `[4096, 1024]` | `bs=[2];w=2;s=1` | `bs=[1];w=4;s=1` | `bs=[2];w=4;s=4` | 10.736 | 1.142 | 0.943 | 0.946 |
| `rms_norm` | `[2048, 48]` | `bs=[1];w=2;s=1` | `bs=[32];w=4;s=1` | `bs=[2];w=4;s=6` | 8.208 | 0.863 | 0.768 | 1.004 |
| `softmax` | `[4096, 256]` | `bs=[2];w=2;s=1` | `bs=[16];w=4;s=1` | `bs=[4];w=4;s=7` | 8.656 | 1.192 | 0.940 | 0.953 |
| `fused_linear_jsd` | `[8192, 32000]` | `bs=[1];w=32;s=1` | `bs=[1];w=4;s=1` | `n/a` | 2328.344 | 1.180 | 0.914 | n/a |
| `grpo` | `[8, 1024, 32000]` | `bs=[1, 8, 2048];w=4;s=1` | `bs=[8, 16, 16];w=4;s=1` | `n/a` | 256.736 | 5.496 | 1.121 | n/a |
| `jsd` | `[8192, 32768]` | `bs=[2048, 1];w=4;s=1` | `bs=[32, 32];w=4;s=1` | `n/a` | 557.792 | 2.261 | 0.876 | n/a |
| `kl_div` | `[8192, 32768]` | `bs=[4096, 1];w=4;s=1` | `bs=[32, 32];w=4;s=1` | `n/a` | 484.840 | 2.291 | 0.825 | n/a |
| `layer_norm_bwd` | `[2048, 4096]` | `bs=[16, 2];w=8;s=1` | `bs=[32, 32];w=4;s=1` | `n/a` | 39.184 | 19.566 | 1.254 | n/a |
| `rms_norm_bwd` | `[2048, 4096]` | `bs=[16, 2];w=8;s=1` | `bs=[32, 32];w=4;s=1` | `n/a` | 30.744 | 18.918 | 1.068 | n/a |
| `dynamic_per_token_scaled_fp8_quant` | `[1, 2048]` | `bs=[2048, 2048];w=8;s=1` | `bs=[32, 32];w=4;s=1` | `bs=[16384, 4096];w=16;s=6` | 8.176 | 1.697 | 0.999 | 1.014 |
| `fused_qk_norm_rope` | `[1, 16, 8]` | `bs=[1];w=2;s=1` | `bs=[32];w=4;s=1` | `bs=[4];w=1;s=5` | 8.224 | 1.050 | 2.057 | 0.982 |
| `per_token_group_fp8_quant` | `[1, 2048, 128]` | `bs=[1];w=2;s=1` | `bs=[16];w=4;s=1` | `bs=[8];w=2;s=6` | 9.032 | 0.918 | 1.019 | 0.895 |
| `rms_norm_dynamic_per_token_quant` | `[1, 2048]` | `bs=[2048, 2048, 2048];w=8;s=1` | `bs=[16, 16, 16];w=4;s=1` | `bs=[2048, 2048, 2048];w=16;s=3` | 8.216 | 3.741 | 0.990 | 0.992 |
| `rms_norm_per_block_quant` | `[1, 2048, 128]` | `bs=[2048, 16];w=8;s=1` | `bs=[32, 16];w=4;s=1` | `bs=[2048, 16];w=16;s=8` | 8.232 | 1.489 | 1.486 | 1.007 |
| `silu_and_mul_per_block_quant` | `[1, 6144, 128]` | `bs=[2];w=2;s=1` | `bs=[32];w=4;s=1` | `bs=[4];w=2;s=3` | 8.248 | 1.115 | 1.177 | 0.993 |

## Timing Spread Above 5%

These cells already used the 15-round escalation. Very short kernels are especially sensitive to event-timing granularity.

| Kernel | Shape | Arm | Latency us | Spread | Rounds |
|---|---|---|---:|---:|---:|
| `dynamic_per_token_scaled_fp8_quant` | `[1, 2048]` | `seed` | 8.176 | 23.9% | 2 |
| `per_token_group_fp8_quant` | `[1, 2048, 128]` | `torch_compile` | 9.200 | 23.0% | 2 |
| `silu_and_mul_per_block_quant` | `[1, 6144, 128]` | `default` | 9.200 | 21.2% | 2 |
| `per_token_group_fp8_quant` | `[1, 2048, 128]` | `seed` | 9.032 | 19.0% | 2 |
| `softmax` | `[4096, 256]` | `seed` | 8.656 | 11.5% | 2 |
| `silu_and_mul_per_block_quant` | `[1, 6144, 128]` | `torch_compile` | 9.704 | 10.7% | 2 |
| `layer_norm` | `[4096, 1024]` | `seed` | 10.736 | 10.1% | 2 |
| `fused_qk_norm_rope` | `[1, 16, 8]` | `default` | 8.632 | 9.8% | 2 |
| `dynamic_per_token_scaled_fp8_quant` | `[1, 2048]` | `default` | 13.872 | 6.5% | 2 |
| `fused_qk_norm_rope` | `[1, 16, 8]` | `torch_compile` | 16.920 | 6.3% | 2 |

## Failures

None.
