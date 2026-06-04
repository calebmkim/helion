# Task 2 — Three-arm benchmark: my heuristic vs main's baseline heuristic vs torch.compile

**Setup (fair, config-only):** all three arms run on the **up-to-date main** checkout
(`origin/main` @ `07eb1c01`, which contains the just-landed baseline reduction heuristic
PR #2648 `ea35dfdd`). Only the **config** differs between the two helion arms — same compiler
substrate, same kernel sources. The one source patch applied (identically to all arms) is the
welford `Tn` masked-count fix (a correctness bug my branch fixes; see
`_lab/harness/patches/main_welford_Tn_fix.patch`).

**Arms:**
- **baseline** = main's `triton_reduction_tile` heuristic config *if it fires*, else `default_config()`. Source recorded per shape.
- **mine** = my heuristic's config, replayed from `task1_seed_configs.json` via `configs=[cfg]` (no autotune).
- **tc** = `torch.compile` default mode.

All fp32, correctness-gated rtol=1e-3/atol=1e-4 (not loosened), median-of-7 `do_bench`, single H100,
serial, GPU idle-checked. Ratios **<1 ⇒ the first arm is faster**. Test split only (66 shapes).

## Headline (per-kernel geomean over the test split)

| kernel | n | baseline config source | mine/base | mine/tc | base/tc |
|---|---|---|---|---|---|
| rms_norm | 8 | heuristic | **0.980** | 0.982 | 1.003 |
| layer_norm | 8 | heuristic | **0.955** | 0.978 | 1.024 |
| softmax | 8 | default | **0.257** | 0.896 | 3.494 |
| welford | 7 | default | **0.386** | 1.049 | 2.718 |
| sum | 7 | heuristic | **0.901** | 0.997 | 1.107 |
| long_sum | 7 | heuristic | **0.665** | 1.068 | 1.435 ⚠️1 baseline-incorrect (excluded) |
| cross_entropy | 7 | heuristic | **0.113** | 1.182 | 10.420 |
| kl_div | 7 | default | **0.173** | 0.922 | 5.344 |
| jsd | 7 | default | **0.258** | 0.963 | 3.737 |
| **OVERALL** | 66 | — | **0.403** | — | — |

- **mine/base = 0.403** overall ⇒ my heuristic is ~**2.5× faster** than the baseline arm on the test set (geomean of per-kernel geomeans).
- Split by baseline source:
  - baseline **heuristic** (36 shapes, cross_entropy, layer_norm, long_sum, rms_norm, sum): geomean mine/base = **0.591**
  - baseline **default** (29 shapes, jsd, kl_div, softmax, welford): geomean mine/base = **0.258**

## Which kernels the baseline heuristic actually covers

The baseline `triton_reduction_tile` is gated by `is_canonical_row_reduction` (1 non-reduction tile +
1 reduction loop + no matmul = the **T1** path). So:

- **Fires (T1):** rms_norm, layer_norm, sum, long_sum, cross_entropy → emits fixed
  `block_sizes=[1], reduction_loops=[None], eviction=['last']`.
- **Does NOT fire (T2) → falls back to `default_config()`:** softmax, welford, kl_div, jsd.

## Correctness findings

- **long_sum(8,2097152)**: baseline(heuristic) INCORRECT/failed. `CompilationError: at 6:16:
def _helion_longsum(x, out, _RDIM_SIZE_1: tl.constexpr):
    # src[long_sum.py:71]: for tile_m in hl.tile(m):
    pid_0 = tl.program_`

> **`long_sum(8, 2097152)` — baseline fails to compile.** The baseline heuristic forces
> `reduction_loops=[None]` (persistent) unconditionally; at N=2097152 the whole-row `tl.arange`
> exceeds Triton's max tensor numel (2²⁰=1048576) → `CompilationError`. My heuristic loops
> (`reduction_loops=[16384]`) and compiles+runs correctly. This is a **correctness** win, not just perf.

## Per-shape detail

### rms_norm

| shape | base src | base µs | mine µs | tc µs | mine/base | mine/tc | corr b/m/tc |
|---|---|---|---|---|---|---|---|
| (16384,896) | heuristic | 44.7 | 45.1 | 44.6 | 1.009 | 1.010 | 1/1/1 |
| (8192,1280) | heuristic | 33.3 | 32.8 | 33.7 | 0.985 | 0.973 | 1/1/1 |
| (16384,1536) | heuristic | 71.6 | 73.3 | 71.9 | 1.023 | 1.018 | 1/1/1 |
| (4096,2560) | heuristic | 33.5 | 32.1 | 33.6 | 0.958 | 0.954 | 1/1/1 |
| (2048,4096) | heuristic | 28.5 | 26.3 | 28.3 | 0.922 | 0.928 | 1/1/1 |
| (2048,6144) | heuristic | 40.2 | 38.4 | 39.6 | 0.955 | 0.970 | 1/1/1 |
| (2048,7168) | heuristic | 45.5 | 45.1 | 45.6 | 0.991 | 0.990 | 1/1/1 |
| (2048,10240) | heuristic | 63.0 | 62.9 | 61.9 | 0.998 | 1.017 | 1/1/1 |

### layer_norm

| shape | base src | base µs | mine µs | tc µs | mine/base | mine/tc | corr b/m/tc |
|---|---|---|---|---|---|---|---|
| (16384,896) | heuristic | 45.5 | 45.8 | 45.1 | 1.006 | 1.015 | 1/1/1 |
| (8192,1280) | heuristic | 35.0 | 33.2 | 34.7 | 0.950 | 0.958 | 1/1/1 |
| (4096,2048) | heuristic | 29.2 | 27.0 | 28.9 | 0.927 | 0.937 | 1/1/1 |
| (4096,3584) | heuristic | 45.9 | 45.6 | 45.1 | 0.994 | 1.011 | 1/1/1 |
| (2048,4096) | heuristic | 29.3 | 27.0 | 29.0 | 0.921 | 0.930 | 1/1/1 |
| (2048,6144) | heuristic | 44.1 | 39.7 | 40.3 | 0.901 | 0.987 | 1/1/1 |
| (2048,7168) | heuristic | 47.4 | 46.6 | 46.7 | 0.983 | 0.998 | 1/1/1 |
| (2048,10240) | heuristic | 66.7 | 64.1 | 64.8 | 0.961 | 0.989 | 1/1/1 |

### softmax

| shape | base src | base µs | mine µs | tc µs | mine/base | mine/tc | corr b/m/tc |
|---|---|---|---|---|---|---|---|
| (131072,128) | default | 72.1 | 51.0 | 54.8 | 0.708 | 0.932 | 1/1/1 |
| (8192,896) | default | 55.8 | 26.1 | 25.5 | 0.469 | 1.024 | 1/1/1 |
| (8192,1280) | default | 80.8 | 35.0 | 51.4 | 0.433 | 0.681 | 1/1/1 |
| (8192,3072) | default | 183.9 | 75.7 | 92.6 | 0.412 | 0.817 | 1/1/1 |
| (4096,7168) | default | 343.1 | 85.6 | 92.9 | 0.249 | 0.921 | 1/1/1 |
| (2048,16384) | default | 710.8 | 105.3 | 107.8 | 0.148 | 0.976 | 1/1/1 |
| (2048,40960) | default | 1742.4 | 317.4 | 323.8 | 0.182 | 0.980 | 1/1/1 |
| (512,98304) | default | 3949.4 | 186.1 | 208.7 | 0.047 | 0.892 | 1/1/1 |

### welford

| shape | base src | base µs | mine µs | tc µs | mine/base | mine/tc | corr b/m/tc |
|---|---|---|---|---|---|---|---|
| (16384,896) | default | 93.1 | 45.6 | 45.0 | 0.490 | 1.012 | 1/1/1 |
| (16384,1280) | default | 158.2 | 66.8 | 62.6 | 0.422 | 1.066 | 1/1/1 |
| (8192,3584) | default | 299.7 | 87.8 | 83.9 | 0.293 | 1.046 | 1/1/1 |
| (16384,5120) | default | 604.0 | 235.8 | 225.7 | 0.390 | 1.045 | 1/1/1 |
| (8192,14336) | default | 1182.9 | 352.2 | 326.6 | 0.298 | 1.079 | 1/1/1 |
| (32768,2560) | default | 510.9 | 238.9 | 226.9 | 0.468 | 1.053 | 1/1/1 |
| (16384,7168) | default | 843.3 | 326.6 | 313.0 | 0.387 | 1.043 | 1/1/1 |

### sum

| shape | base src | base µs | mine µs | tc µs | mine/base | mine/tc | corr b/m/tc |
|---|---|---|---|---|---|---|---|
| (16384,1280) | heuristic | 40.5 | 33.9 | 34.4 | 0.837 | 0.986 | 1/1/1 |
| (8192,2560) | heuristic | 40.5 | 33.9 | 33.9 | 0.836 | 0.999 | 1/1/1 |
| (8192,7168) | heuristic | 88.6 | 83.0 | 83.0 | 0.936 | 1.000 | 1/1/1 |
| (4096,10240) | heuristic | 68.3 | 61.4 | 61.6 | 0.899 | 0.996 | 1/1/1 |
| (4096,18432) | heuristic | 110.1 | 105.2 | 105.6 | 0.955 | 0.996 | 1/1/1 |
| (2048,24576) | heuristic | 79.3 | 72.7 | 73.3 | 0.916 | 0.992 | 1/1/1 |
| (16384,3072) | heuristic | 78.0 | 72.8 | 72.0 | 0.934 | 1.012 | 1/1/1 |

### long_sum

| shape | base src | base µs | mine µs | tc µs | mine/base | mine/tc | corr b/m/tc |
|---|---|---|---|---|---|---|---|
| (160,131072) | heuristic | 49.5 | 35.7 | 36.2 | 0.721 | 0.987 | 1/1/1 |
| (128,229376) | heuristic | 53.7 | 46.5 | 51.1 | 0.865 | 0.909 | 1/1/1 |
| (96,196608) | heuristic | 61.1 | 33.1 | 33.5 | 0.541 | 0.987 | 1/1/1 |
| (64,294912) | heuristic | 49.0 | 33.3 | 33.4 | 0.680 | 0.996 | 1/1/1 |
| (128,524288) | heuristic | 101.7 | 98.0 | 112.9 | 0.964 | 0.868 | 1/1/1 |
| (48,786432) | heuristic | 157.8 | 61.5 | 62.4 | 0.390 | 0.986 | 1/1/1 |
| (8,2097152) | heuristic | — | 68.6 | 32.7 | — | 2.099 | 0/1/1 |

### cross_entropy

| shape | base src | base µs | mine µs | tc µs | mine/base | mine/tc | corr b/m/tc |
|---|---|---|---|---|---|---|---|
| (8192,32768) | heuristic | 426.4 | 359.5 | 378.7 | 0.843 | 0.949 | 1/1/1 |
| (4096,49152) | heuristic | 4072.4 | 276.1 | 296.3 | 0.068 | 0.932 | 1/1/1 |
| (2048,50257) | heuristic | 2348.7 | 175.9 | 183.2 | 0.075 | 0.960 | 1/1/1 |
| (4096,114688) | heuristic | 11032.1 | 781.4 | 644.6 | 0.071 | 1.212 | 1/1/1 |
| (1024,128256) | heuristic | 3065.3 | 270.2 | 200.6 | 0.088 | 1.347 | 1/1/1 |
| (4096,151936) | heuristic | 14470.5 | 1243.7 | 846.8 | 0.086 | 1.469 | 1/1/1 |
| (1024,250000) | heuristic | 5769.6 | 606.2 | 383.5 | 0.105 | 1.581 | 1/1/1 |

### kl_div

| shape | base src | base µs | mine µs | tc µs | mine/base | mine/tc | corr b/m/tc |
|---|---|---|---|---|---|---|---|
| (8192,32768) | default | 1378.9 | 690.7 | 728.4 | 0.501 | 0.948 | 1/1/1 |
| (8192,49152) | default | 2889.3 | 1028.3 | 1066.9 | 0.356 | 0.964 | 1/1/1 |
| (2048,50257) | default | 1384.1 | 295.1 | 333.5 | 0.213 | 0.885 | 1/1/1 |
| (4096,114688) | default | 6466.6 | 1191.6 | 1235.6 | 0.184 | 0.964 | 1/1/1 |
| (1024,128256) | default | 7214.8 | 355.8 | 483.6 | 0.049 | 0.736 | 1/1/1 |
| (4096,151936) | default | 8493.4 | 1588.0 | 1611.8 | 0.187 | 0.985 | 1/1/1 |
| (1024,250000) | default | 9704.2 | 684.2 | 682.5 | 0.071 | 1.002 | 1/1/1 |

### jsd

| shape | base src | base µs | mine µs | tc µs | mine/base | mine/tc | corr b/m/tc |
|---|---|---|---|---|---|---|---|
| (8192,32768) | default | 1357.4 | 705.0 | 712.8 | 0.519 | 0.989 | 1/1/1 |
| (8192,49152) | default | 1968.9 | 1041.7 | 1056.5 | 0.529 | 0.986 | 1/1/1 |
| (2048,50257) | default | 2004.8 | 307.7 | 344.4 | 0.153 | 0.894 | 1/1/1 |
| (4096,114688) | default | 4223.3 | 1221.0 | 1229.3 | 0.289 | 0.993 | 1/1/1 |
| (2048,128256) | default | 4642.5 | 698.4 | 739.0 | 0.150 | 0.945 | 1/1/1 |
| (8192,151936) | default | 6019.9 | 3177.4 | 3174.0 | 0.528 | 1.001 | 1/1/1 |
| (1024,250000) | default | 9073.2 | 705.3 | 754.4 | 0.078 | 0.935 | 1/1/1 |
