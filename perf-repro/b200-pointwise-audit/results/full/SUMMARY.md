# B200 Pointwise-Heuristic Audit Results

Recorded 36 of 36 planned cells; 35 produced timings. The default arm is `1.00x`; relative performance above one is faster.

## Environment

- Helion: `61f4058f3610e1b2bbabc82df8267ac450f591df` (`b200-pointwise-audit-run`)
- PyTorch: `2.12.0+cu132` (`7661cd9c6b841b62b7f411aa52ec51f05457263b`)
- Triton: `3.7.0`
- CUDA runtime / driver: `13.2` / `595.71.05`
- GPU: physical `1`, `NVIDIA B200`; `CUDA_VISIBLE_DEVICES=1`
- Calibrated cold-L2 CUDA graphs, flush inside the timed graph, nine round medians with fifteen-round high-spread escalation.

## AOT Provenance

- RoPE and vLLM `silu_mul_fp8`: checked-in SM90 tables selected on B200.
- SGLang `silu_and_mul_interleaved`: checked-in SM100 table.
- SwiGLU and GEGLU have no AOT arm.

## Findings

- General pointwise: the seed is `19.351x` default and `torch.compile` is `18.508x`. The large gains come from replacing the tiny unseeded base configs. The RoPE-only SM90 AOT result is `22.145x`.
- vLLM `silu_mul_fp8`: seed `1.096x`, `torch.compile` `1.232x`, and SM90 AOT `1.130x` default.
- SGLang `silu_and_mul_interleaved`: seed `1.257x`, `torch.compile` `1.420x`, and SM100 AOT `1.036x` default.
- 35 cells completed; the remaining planned RoPE cell reached the 300-second timeout.

## Coverage

- `triton_pointwise`: 35 cells
- Pointwise seed present: 35/35 measured cells

## Cohorts

Entries are geomean [min, max] (count), relative to default.

| Cohort | Seed | torch.compile | AOT |
|---|---:|---:|---:|
| `general` | 19.351 [14.284, 40.623] (n=17) | 18.508 [14.221, 44.731] (n=17) | 22.145 [11.182, 39.213] (n=5) |
| `vllm` | 1.096 [0.958, 1.251] (n=9) | 1.232 [0.988, 1.487] (n=9) | 1.130 [0.847, 1.388] (n=9) |
| `sglang` | 1.257 [1.005, 1.669] (n=9) | 1.420 [1.215, 1.758] (n=9) | 1.036 [0.807, 1.187] (n=9) |

## Kernels

| Kernel | Dtype | Cells | Seed | torch.compile | AOT |
|---|---|---:|---:|---:|---:|
| `swiglu` | bf16 | 6/6 | 18.741 [18.185, 19.382] (n=6) | 16.562 [16.131, 16.987] (n=6) | n/a |
| `geglu` | bf16 | 6/6 | 14.711 [14.284, 15.175] (n=6) | 14.539 [14.221, 14.868] (n=6) | n/a |
| `rope` | bf16 | 6/6 | 27.943 [19.669, 40.623] (n=5) | 28.253 [19.113, 44.731] (n=5) | 22.145 [11.182, 39.213] (n=5) (sm90) |
| `silu_mul_fp8` | bf16_to_fp8 | 9/9 | 1.096 [0.958, 1.251] (n=9) | 1.232 [0.988, 1.487] (n=9) | 1.130 [0.847, 1.388] (n=9) (sm90) |
| `silu_and_mul_interleaved` | bf16 | 9/9 | 1.257 [1.005, 1.669] (n=9) | 1.420 [1.215, 1.758] (n=9) | 1.036 [0.807, 1.187] (n=9) (sm100) |

## Per Shape

Latency is calibrated cold-L2 device time. Configs show `block_sizes`, `num_warps`, and `num_stages`.

| Kernel | Shape | Default config | Seed config | AOT config | Default us | Seed x | torch.compile x | AOT x | Null delta |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `geglu` | `(16384,2880)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 800.787 | 14.324 | 14.250 | n/a | 0.00% |
| `geglu` | `(8192,6912)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 967.526 | 14.646 | 14.495 | n/a | 0.00% |
| `geglu` | `(8192,14336)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 1971.109 | 15.020 | 14.778 | n/a | 0.01% |
| `geglu` | `(4096,21504)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 1478.592 | 14.838 | 14.632 | n/a | 0.00% |
| `geglu` | `(4096,36864)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 2534.074 | 15.175 | 14.868 | n/a | 0.00% |
| `geglu` | `(2048,24576)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 845.625 | 14.284 | 14.221 | n/a | 0.00% |
| `rope` | `(1,32,2048,256)` | `bs=[1, 32];w=4;s=1` | `bs=[1, 1];w=2;s=None` | `bs=[1, 1];w=16;s=1` | 1182.468 | 40.623 | 44.731 | 39.213 | 0.37% |
| `rope` | `(1,32,8192,256)` | `bs=[1, 32];w=4;s=1` | `bs=[1, 1];w=1;s=None` | `bs=[1, 1];w=8;s=1` | 1831.434 | 19.669 | 19.113 | 17.849 | 0.03% |
| `rope` | `(1,32,4096,256)` | `bs=[1, 32];w=4;s=1` | `bs=[1, 1];w=1;s=None` | `bs=[1, 1];w=4;s=1` | 1449.292 | 28.475 | 29.246 | 30.875 | 0.04% |
| `rope` | `(2,32,2048,256)` | `n/a` | `n/a` | `n/a` | n/a | n/a | n/a | n/a | n/a |
| `rope` | `(1,32,4096,128)` | `bs=[1, 32];w=4;s=1` | `bs=[1, 1];w=1;s=None` | `bs=[1, 1];w=16;s=1` | 728.001 | 27.414 | 27.419 | 22.039 | 0.07% |
| `rope` | `(4,8,4096,128)` | `bs=[4, 32];w=4;s=1` | `bs=[1, 2];w=1;s=None` | `bs=[1, 1];w=16;s=1` | 722.844 | 27.313 | 26.259 | 11.182 | 0.15% |
| `swiglu` | `(32768,1536)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 860.093 | 18.185 | 16.209 | n/a | 0.01% |
| `swiglu` | `(16384,2880)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 806.404 | 18.232 | 16.131 | n/a | 0.01% |
| `swiglu` | `(8192,11008)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 1539.988 | 18.985 | 16.794 | n/a | 0.01% |
| `swiglu` | `(8192,14336)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 2005.000 | 19.373 | 16.987 | n/a | 0.01% |
| `swiglu` | `(4096,28672)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 2004.916 | 19.382 | 16.981 | n/a | 0.01% |
| `swiglu` | `(2048,24576)` | `bs=[32];w=4;s=1` | `bs=[2048];w=None;s=None` | `n/a` | 859.715 | 18.328 | 16.296 | n/a | 0.01% |
| `silu_and_mul_interleaved` | `(6,4096,False)` | `bs=[8, 32];w=4;s=1` | `bs=[1, 256];w=None;s=None` | `bs=[2, 512];w=1;s=2` | 2.309 | 1.005 | 1.215 | 0.807 | 0.09% |
| `silu_and_mul_interleaved` | `(16,16384,True)` | `bs=[16, 32];w=4;s=1` | `bs=[1, 256];w=None;s=None` | `bs=[1, 1024];w=1;s=5` | 2.618 | 1.021 | 1.314 | 0.951 | 1.11% |
| `silu_and_mul_interleaved` | `(48,512,False)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 256];w=None;s=None` | `bs=[2, 512];w=1;s=3` | 2.561 | 1.113 | 1.334 | 1.175 | 0.04% |
| `silu_and_mul_interleaved` | `(192,4096,True)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 512];w=8;s=None` | `bs=[1, 1024];w=1;s=5` | 3.517 | 1.193 | 1.525 | 1.098 | 0.26% |
| `silu_and_mul_interleaved` | `(768,3072,False)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 512];w=None;s=None` | `bs=[2, 512];w=1;s=3` | 5.632 | 1.313 | 1.758 | 1.187 | 0.66% |
| `silu_and_mul_interleaved` | `(1024,12288,False)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 512];w=None;s=None` | `bs=[2, 512];w=1;s=3` | 16.450 | 1.504 | 1.548 | 1.000 | 0.02% |
| `silu_and_mul_interleaved` | `(3072,2048,False)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 512];w=None;s=None` | `bs=[2, 512];w=1;s=2` | 9.795 | 1.425 | 1.709 | 1.021 | 0.08% |
| `silu_and_mul_interleaved` | `(12288,1536,True)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 512];w=None;s=None` | `bs=[1, 1024];w=1;s=5` | 23.593 | 1.225 | 1.237 | 1.084 | 0.11% |
| `silu_and_mul_interleaved` | `(98304,6144,False)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 512];w=None;s=None` | `bs=[2, 512];w=1;s=3` | 643.389 | 1.669 | 1.253 | 1.055 | 0.01% |
| `silu_mul_fp8` | `(1,2048)` | `bs=[1, 32];w=4;s=1` | `bs=[1, 256];w=None;s=None` | `bs=[1, 64];w=1;s=1` | 2.090 | 0.958 | 0.988 | 0.981 | 1.15% |
| `silu_mul_fp8` | `(2,8192)` | `bs=[2, 32];w=4;s=1` | `bs=[1, 256];w=None;s=None` | `bs=[2, 128];w=4;s=7` | 2.340 | 1.016 | 1.104 | 1.072 | 1.32% |
| `silu_mul_fp8` | `(8,4096)` | `bs=[8, 32];w=4;s=1` | `bs=[1, 256];w=None;s=None` | `bs=[8, 64];w=4;s=1` | 2.369 | 1.024 | 1.226 | 1.119 | 0.04% |
| `silu_mul_fp8` | `(16,11008)` | `bs=[16, 32];w=4;s=1` | `bs=[1, 256];w=None;s=None` | `bs=[2, 512];w=4;s=2` | 2.654 | 1.005 | 1.083 | 0.847 | 1.02% |
| `silu_mul_fp8` | `(64,2880)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 256];w=None;s=None` | `bs=[32, 64];w=16;s=1` | 3.326 | 1.240 | 1.366 | 1.366 | 0.06% |
| `silu_mul_fp8` | `(128,2048)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 256];w=None;s=None` | `bs=[32, 64];w=16;s=1` | 3.352 | 1.242 | 1.308 | 1.304 | 1.04% |
| `silu_mul_fp8` | `(256,7688)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 2048];w=None;s=None` | `bs=[128, 64];w=32;s=1` | 5.761 | 0.979 | 1.487 | 1.021 | 0.16% |
| `silu_mul_fp8` | `(384,8192)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 2048];w=None;s=None` | `bs=[128, 64];w=32;s=1` | 7.103 | 1.209 | 1.423 | 1.388 | 0.01% |
| `silu_mul_fp8` | `(512,14336)` | `bs=[32, 32];w=4;s=1` | `bs=[1, 2048];w=None;s=None` | `bs=[8, 256];w=8;s=4` | 12.802 | 1.251 | 1.191 | 1.191 | 0.49% |

## Measurement Flags

No arm spread above 5% and no default/null delta above 3%.

## Failures

| Kernel | Shape | Arm | Failure |
|---|---|---|---|
| `rope` | `[2, 32, 2048, 256]` | `cell` | timeout after 300s |
