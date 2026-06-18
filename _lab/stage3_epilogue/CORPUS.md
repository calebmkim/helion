# Stage-3 corpus: matmul + reduction-over-N-epilogue

Kernels + harness for the Stage-3 "matmul + reduction-epilogue" task. **No heuristic
changes** — corpus kernels, a forward-only benchmark harness, and a smoke test only.

- Kernels:  `matmul_epilogue_kernels.py`
- Harness:  `mm_epilogue_bench.py`
- Interpreter: `/home/calebkim/.conda/envs/helion/bin/python`
- Run from `cwd=/tmp` with `PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-3stage`
  and `CUDA_VISIBLE_DEVICES=0` (one GPU at a time).

## The shared skeleton

Every kernel shares ONE matmul skeleton (copied verbatim from
`_lab/matmul_rms_norm_template.py`); only the EPILOGUE over N changes:

```python
n = hl.specialize(y.size(1))          # N full-width, never tiled -> acc is [tile_m, n]
for tile_m in hl.tile(m):
    acc = hl.zeros([tile_m, n], dtype=torch.float32)
    for tile_k in hl.tile(k):
        acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
    <EPILOGUE: a reduction over the N axis on the register-resident acc>
    out[tile_m, :] = ...
```

`N = hl.specialize(y.size(1))` is the structural invariant — the [tile_m, n] fp32
accumulator stays register-resident (never tiled over N).

## Kernels authored (each from its own named torch/library reference)

| name                  | set      | epilogue over N (on fp32 `acc`)              | reference                         | output  |
|-----------------------|----------|----------------------------------------------|-----------------------------------|---------|
| `matmul_rms_norm`     | FIT      | `acc*rsqrt(mean(acc^2)+eps)*weight`          | template (`matmul_rms_norm_ref`)  | [M, N]  |
| `matmul_layernorm`    | FIT      | mean+var affine (2 reductions) +weight,+bias | `F.layer_norm` (in-tree carrier)  | [M, N]  |
| `matmul_softmax`      | FIT      | `exp(acc-max)/sum(exp(acc-max))`             | `torch.softmax(dim=-1)`           | [M, N]  |
| `matmul_l2_normalize` | FIT      | `acc*rsqrt(clamp(sum(acc^2),eps))`           | `F.normalize(p=2, dim=-1)`        | [M, N]  |
| `matmul_sum`          | FIT      | `acc.sum(-1)`  (scalar-output DOF)           | `(x@y).sum(-1, keepdim=True)`     | [M, 1]  |
| `matmul_logsumexp`    | HELD-OUT | `max + log(sum(exp(acc-max)))`               | `torch.logsumexp(dim=-1)`         | [M, 1]  |
| `matmul_max`          | HELD-OUT | `acc.amax(-1)` (max, not argmax)             | `(x@y).amax(-1, keepdim=True)`    | [M, 1]  |

Each `*_ref` computes the matmul in fp32, applies the epilogue in fp32, and casts back to
`promote_types(x.dtype, y.dtype)` — the accuracy oracle.

## Accuracy floors (measured 2026-06-18, H100, M=131072 K=256 N=256)

The matmul accumulation order in Helion differs from cuBLAS, so over a wide-N reduction the
per-element **absolute** error is large in absolute terms but tiny relative to the OUTPUT
SCALE. A naive `allclose(rtol=atol=2e-2)` therefore FALSE-fails `matmul_sum` (sum elements
near zero via cancellation → huge relative error → §4 #6b) and `matmul_softmax` (the peak
prob near 1.0 rounds in bf16). Verified these are accumulation-order / output-dtype rounding,
NOT kernel bugs: at fp32 two independent torch reduction orders already differ from each
other (`matmul_sum`: Helion-vs-torch rel-to-output-RMS = 0.31% vs torch-vs-torch 1.7e-6 —
both far below the kept threshold).

The gate (`MK.acc_ok`, shared by corpus + harness), all fp32-upcast first:
- **softmax** (bounded [0,1], RMS dominated by tiny non-peak probs): gate on **max_abs**
  — bf16 ≤ 0.09, fp16 ≤ 0.03, fp32 ≤ 0.02.
- **all others** (magnitude scales with N): gate on **max_abs / output-RMS** —
  bf16 ≤ 0.07, fp16 ≤ 0.03, fp32 ≤ 0.02.

Measured bf16 rel-to-RMS (and max_abs for softmax) at the smoke shape:
rms_norm 0.062, layernorm 0.044, softmax max_abs 0.0625, l2_normalize 0.031,
sum 0.015, logsumexp 0.011, max 0.0 — all under threshold.

## Harness (`mm_epilogue_bench.py`)

Cloned from `_lab/transfer/ab_three_arm_transfer.py`. Single-process, same inputs across
arms, **median-of-9 cold-L2 do_bench**, **accuracy-gate BEFORE timing**, geomean over
acc-passing rows. Honors hillclimb-method §4 footguns: forward-only (`helion.kernel(fn.fn,
config=cfg)` bare forward), dynamo-reset per shape, `empty_cache` between shapes, cold-L2
metric + effective-BW sanity check (< HBM peak), acc-fails excluded from geomean, sub-25us
rows flagged.

Arms per (kernel, shape, dtype):
- **helion_default** — `config_spec.default_config()` (heuristics off).
- **helion_seeded** — `compiler_seed_configs(...)[0]` if any else default (== default until
  the Stage-3 heuristic exists — expected; `seed_fired` records which).
- **helion_best** — best over a forced-config GRID (tile_m ∈ {16,32,64,128,256},
  tile_k ∈ {16,32,64}, num_warps ∈ {4,8}, num_stages=3) — the well-tuned Helion reference.
- **tc_default** / **tc_max** — `torch.compile(ref)` default + `mode="max-autotune"`, BOTH
  gated on the skinny predicate `M >= 8*max(K,N)`.

Reported per row: `seeded_vs_default`, `best_vs_default`, `best_vs_tc_default`,
`best_vs_tc_max`, plus `noisy_sub25us`, acc dict, `grid_valid_cfgs`, `best_bw_tbps`.

### CLI

```
python mm_epilogue_bench.py <kernel> --shapes "M,K,N;..." --dtypes bf16
python mm_epilogue_bench.py matmul_rms_norm                       # default = train split, bf16
python mm_epilogue_bench.py all --split val --dtypes bf16,fp16,fp32
python mm_epilogue_bench.py fit --split test
```

`<kernel>` is a corpus name or `all` / `fit` / `held_out`. Curriculum splits (M,K,N):
train (131072,256,256),(131072,256,512),(65536,512,512),(262144,128,256);
val (131072,256,1024),(98304,384,384);
test (196608,256,768),(131072,512,1024);
robustness (1024,256,256),(131072,256,2048 — expect no valid cfg, SMEM wall).

## Smoke results

Run 2026-06-18, H100, single GPU (`CUDA_VISIBLE_DEVICES=0`), median-of-9 cold-L2.

**Correctness** — all 7 corpus kernels compile and PASS accuracy at M=131072 K=256 N=256
bf16 (`matmul_epilogue_kernels.py main()` with `HELION_AUTOTUNE_EFFORT=none` for the
default config):

| kernel              | gate metric (bf16) | value  |
|---------------------|--------------------|--------|
| matmul_rms_norm     | rel_to_rms         | 0.062  |
| matmul_layernorm    | rel_to_rms         | 0.044  |
| matmul_softmax      | max_abs            | 0.0625 |
| matmul_l2_normalize | rel_to_rms         | 0.031  |
| matmul_sum          | rel_to_rms         | 0.015  |
| matmul_logsumexp    | rel_to_rms         | 0.011  |
| matmul_max          | rel_to_rms         | 0.0    |

`matmul_layernorm` compiles fine at N=256; the in-tree example's "n=64/128 throws" note
was NOT reproduced at the smoke shape (N=256). (N=64/128 are below the corpus curriculum's
smallest N=256, so not exercised here.)

**Benchmark** — `mm_epilogue_bench.py <kernel> --shapes "131072,256,256" --dtypes bf16`:

| kernel          | default µs | best µs | tc_def µs | tc_max µs | best_vs_default | best_vs_tc_default | best_vs_tc_max | best_cfg            | BW TB/s | noisy |
|-----------------|-----------:|--------:|----------:|----------:|----------------:|-------------------:|---------------:|---------------------|--------:|-------|
| matmul_rms_norm |     110.78 |   77.34 |    134.24 |    212.77 |          1.432  |             1.736  |         2.751  | tile[64,32] w4 s3   |   1.737 | no    |
| matmul_softmax  |     122.37 |   96.26 |    134.90 |    213.66 |          1.271  |             1.401  |         2.220  | tile[128,64] w8 s3  |   1.396 | no    |

- Both kernels: helion_best beats tc_default (1.40-1.74x) and tc_max (2.22-2.75x) — within
  the task's expected 1.4-2.6x small-N band; tc_max is slowest (max-autotune picks an
  UNFUSED cuBLAS matmul + separate epilogue).
- The grid found **30 valid configs** for each; all arms passed accuracy; effective BW
  (1.40-1.74 TB/s) is well under H100 HBM peak (~3.35 TB/s) — the cold-L2 metric is sane.
- `helion_seeded == helion_default` (`seed_fired=false`, ratio ≈ 1.00) — expected, since
  the Stage-3 heuristic does not exist yet. `best_vs_default` (1.27-1.43x) is the gap that
  heuristic would close.
- No noisy (sub-25µs) rows: at M=131072 the shapes are ~80-100µs, above the noise floor.
  (An earlier run mislabeled latencies because `do_bench` returns ms; fixed — `_med` now
  returns µs and the BW/noise checks use µs.)

### Reproduce

```
cd /tmp && CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-3stage \
  /home/calebkim/.conda/envs/helion/bin/python \
  /home/calebkim/helion-new-heuristics/helion-3stage/_lab/stage3_epilogue/mm_epilogue_bench.py \
  matmul_rms_norm --shapes "131072,256,256" --dtypes bf16

# corpus correctness (fast, default config):
cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
  PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-3stage \
  /home/calebkim/.conda/envs/helion/bin/python \
  /home/calebkim/helion-new-heuristics/helion-3stage/_lab/stage3_epilogue/matmul_epilogue_kernels.py
```
