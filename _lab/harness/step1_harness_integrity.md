# Step 1 — Harness Integrity Report

**Agent:** harness-integrity (Step 1). **Date:** 2026-05-28. **GPU:** H100 #2 (sm_90),
verified idle (4 MiB, 0%) before and re-checked between every trusted measurement.
**Interpreter:** `/home/calebkim/.conda/envs/helion/bin/python` (torch 2.12 dev cu128, triton 3.7).
**helion resolves to worktree:** asserted in every script (`helion.__file__` startswith the worktree).

---

## CERTIFICATION VERDICT

**Is the TritonBench harness systematically biased? NO.**

An independent, hand-rolled standalone harness — timing the SAME wrapper callables
that TritonBench times, with identical inputs/layout/dtype/strides/flags, the same
do_bench primitive (L2 flush on, median), in the same process — reconciles with
TritonBench's own do_bench numbers to **< 1%** on both sides at (4096, 8192) fp32:

| side | hand-rolled median | TritonBench (task 3) | delta |
|---|---|---|---|
| Helion-default | 0.16826 ms | 0.16739 ms | **+0.52%** |
| tc-default     | 0.12899 ms | 0.12906 ms | **−0.05%** |

Both deltas are far inside the ~10–15% agreement band and below run-to-run noise.
Kernel counts match on both sides (exactly **1 CUDA kernel/call**, profiler-confirmed
over 20 launches — no hidden host-side reduction / cross-block split on either path).
Correctness identical (max_abs = 9.5e-7 vs the fp32 PyTorch reference on both sides).
=> The harness measures both paths fairly; nothing downstream needs to distrust it.

---

## Task 1 — Env smoke for listed benchmarks

`benchmarks/run.py --kernel <k> --precision fp32` (HELION_AUTOTUNE_EFFORT=none):

| kernel | runs? | Helion-default vs eager | notes |
|---|---|---|---|
| rms_norm (2048,8192) | YES | 3.07× | accuracy=1; tc-max 3.81× |
| sum (default shapes) | YES | ~0.98× (aggregate) | accuracy=1 all variants; memory-bound → ties/inverts legit |
| layer_norm (M=2048) | YES | ~1.45× (aggregate) | helion accuracy=1; note `triton_fused_layer_norm` (3rd-party) reports accuracy=0 — not our path |

All three execute and produce latency+accuracy. No kernel failed to run.

---

## Task 2 — `torch_compile_rms_norm_default` baseline variant

Product A's baseline-to-beat is **torch.compile DEFAULT**. The operator previously
only shipped `torch_compile_rms` (`mode="max-autotune-no-cudagraphs"`). Added a new
`@register_benchmark()` variant `torch_compile_rms_norm_default` =
`torch.compile(module)` (no `mode=` → Inductor default codegen).

- **Live edit location (runs from here, NOT the worktree):**
  `/home/calebkim/helion-new-heuristics/local/helion/benchmarks/tritonbench/tritonbench/operators/rms_norm/operator.py`
  (per SETUP.md the tritonbench editable is a hardcoded MetaPathFinder into the
  original checkout; PYTHONPATH cannot shadow it; that dir is its own nested git
  repo, git-ignored by the parent — outside the worktree's git).
- **Saved patch (reproducible, in-worktree):**
  `_lab/harness/patches/torch_compile_rms_norm_default.patch` + `patches/README.md`
  (apply with `git apply` inside the nested tritonbench repo).
- **Sentinel-verified:** a one-line stderr sentinel in this operator fired from the
  ORIGINAL-checkout copy → the edited file is the one that runs. Sentinel removed.
- **Verified live:** new columns `torch_compile_rms_norm_default-{latency,accuracy,speedup}`
  appear in `--kernel rms_norm` output; accuracy=1; e.g. (2048,8192) → 0.0662 ms, 3.81×.
  Inductor meta confirms `'max_autotune': False` (genuinely default mode).

---

## Task 3 — rms_norm 5-way sanity (fp32, GPU2, do_bench median)

Driver: `_lab/harness/sanity_5way.py` (runs run.py at effort none/quick/full, parses
the stdout table; tc-* variants read from the effort=none run since they don't depend
on HELION effort). fp32 asserted (dtype=torch.float32 present in every run's input meta).

| shape | variant | latency_ms | speedup_vs_eager | accuracy |
|---|---|---|---|---|
| (4096,8192) | Helion-default | 0.16739 | 2.887 | 1.000 |
| (4096,8192) | Helion-quick | 0.12864 | 3.757 | 1.000 |
| (4096,8192) | Helion-max(full) | 0.12925 | 3.739 | 1.000 |
| (4096,8192) | tc-default | 0.12906 | 3.745 | 1.000 |
| (4096,8192) | tc-max | 0.12829 | 3.767 | 1.000 |
| (4096,8192) | eager(llama_rms) | 0.48326 | 1.000 | — |
| (8192,8192) | Helion-default | 0.33613 | 2.777 | 1.000 |
| (8192,8192) | Helion-quick | 0.25187 | 3.706 | 1.000 |
| (8192,8192) | Helion-max(full) | 0.25101 | 3.718 | 1.000 |
| (8192,8192) | tc-default | 0.25030 | 3.729 | 1.000 |
| (8192,8192) | tc-max | 0.24909 | 3.747 | 1.000 |
| (8192,8192) | eager(llama_rms) | 0.93331 | 1.000 | — |

**Ordering matches expectation:** Helion-default ≲ tc-default ≲ tc-max ≈ Helion-max
≈ Helion-quick. Helion-default loses ~23–25% to tc-default (acceptable for Product A;
this is the gap to hill-climb). quick/max/tc-default/tc-max cluster within ~1%.
**No HALT** — nothing "really wrong" (Helion-default does NOT beat tc-max).

Surprise logged (not a halt): at these shapes Helion-**default** picks a LOOPED
reduction (`reduction_loops=[4096]`, `block_sizes=[1]`) rather than persistent — i.e.
`default_config` is not the strongest persistent choice; quick/full close most of the
gap. This is exactly the kind of seed-quality gap the reduction heuristic should fix.

---

## Task 4 — Independent bias cross-check (the core)

Harness: `_lab/harness/crosscheck_bias.py`. Shape (4096,8192) fp32. Both sides time
the EXACT wrapper TritonBench times:
- Helion: `rms_norm(inp, weight, eps=1e-6)` → `RMSNormFunction.apply` (fwd-only path).
- tc-default: `torch.compile(LlamaRMSNorm)(input)` (default mode, same as new variant).

### Extracted generated code + shipped config

**Helion-default** (`default_config`, effort=none):
`block_sizes=[1], reduction_loops=[4096], num_warps=4, num_stages=1, pid_type='flat'`.
Generated Triton: single `@triton.jit` def `_helion_rms_norm_fwd`; **looped** reduction
(`for roffset_1 in tl.range(0, 8192, _REDUCTION_BLOCK_1)`, two passes — one for the
mean-of-squares accumulate, one for the normalize+store); grid `(4096,)` (one program
per row); `num_warps=4`.

**tc-default** (Inductor default): single fused kernel
`triton_red_fused_add_mean_mul_pow_rsqrt_0` (`@triton_heuristics.reduction`); **looped**
(`for r0_offset in tl.range(0, r0_numel, R0_BLOCK)`); inductor_meta `num_load=3,
num_store=1, num_reduction=1, max_autotune=False, add_persistent_rblock=True`. Full
output code archived at `_lab/harness/crosscheck_artifacts/tc_default_output_code.log`.

### Full set of launched kernels (profiler, 20 launches each)

| side | kernels/call | kernel name(s) |
|---|---|---|
| Helion-default | **1.0** | `_helion_rms_norm_fwd` (x20) |
| tc-default | **1.0** | `triton_red_fused_add_mean_mul_pow_rsqrt_0` (x20) |

No host-side reduction, no cross-block final-reduce, no multi-kernel split on EITHER
side. The "replicate the full set of launched kernels" caveat is satisfied: there is
only one kernel per side, profiler-verified (not assumed).

### Reconciliation (hand-rolled vs TritonBench task-3)

| side | hand-rolled median (min..max) | TritonBench | delta |
|---|---|---|---|
| Helion-default | 0.16826 ms (0.16797..0.16845) | 0.16739 ms | +0.52% |
| tc-default | 0.12899 ms (0.12864..0.12906) | 0.12906 ms | −0.05% |

Agreement < 1% both sides → **harness certified unbiased**. The hand-rolled number was
NOT treated as an oracle: it agrees with TritonBench AND the kernel counts match, so
both methods are measuring the same physical work. The ~23% Helion-vs-tc gap is REAL
(reproduced by both methods, identical kernel count/overhead), not a harness artifact —
it is a genuine seed-quality gap for the worker to close.

Note: `ncu` not used (kernel counts + profiler kernel-time + the two-method do_bench
agreement were sufficient to localize — there was nothing to localize, the numbers
agree; ncu would only matter on a disagreement).

---

## Task 5 — Footgun audit

| # | footgun | status | evidence |
|---|---|---|---|
| 1 | fp32 on both sides | **OK** | `--precision fp32` → `apply_precision` no-ops; operator gens torch.float32 tensors (input meta); cross-check asserts `x.dtype==float32` both sides; both correct to 9.5e-7 |
| 2 | identical tf32 settings | **OK** | tritonbench `set_allow_tf32(tb_args.allow_tf32)` is **process-global** and defaults to the original value → identical for every variant. rms_norm has no matmul anyway, so tf32 is math-irrelevant here. Cross-check explicitly sets matmul+cudnn allow_tf32=False on both sides |
| 3 | cudagraphs off both sides | **OK** | `use_cuda_graphs` defaults False; `--cudagraph` not passed → off for all variants identically |
| 4 | L2 flush on | **OK** | default `latency_measure_mode="triton_do_bench"` → `triton.runtime.driver...get_benchmarker()` = triton do_bench with L2 flush on; hand-rolled uses the same `triton.testing.do_bench` |
| 5 | fixed warmup/rep | **OK (consistent)** | TritonBench auto-resolves warmup/rep via `estimate_cuda_runtime_ms` (adaptive but deterministic for a given runtime); hand-rolled uses do_bench defaults (25ms/100ms). Different policy, but the <1% two-method agreement proves it introduces no bias |
| 6 | same-process comparison | **OK** | all 5 variants timed in one run.py process; cross-check times both sides in one python process |
| 7 | tc baseline not caching across configs | **OK** | in one process, tc-default and tc-max compile distinctly (`max_autotune` False AND True both observed); Inductor keys cache on mode. tc-default ≈ tc-max here is legitimate convergence (memory-bound op), not a cache collapse |
| 8 | seed actually used (Helion) | **OK** | cross-check prints the shipped `default_config` and the matching generated Triton (looped `reduction_loops=[4096]`, num_warps=4 in the launcher) |
| 9 | co-tenants on GPU | **OK** | GPU2 verified idle (0%/4MiB) before/between trusted timings |

---

## Reproduce

```
# 5-way sanity:
cd /home/calebkim/helion-new-heuristics/wt-reduction && \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction \
/home/calebkim/.conda/envs/helion/bin/python _lab/harness/sanity_5way.py

# bias cross-check:
cd /home/calebkim/helion-new-heuristics/wt-reduction && \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction \
HELION_AUTOTUNE_EFFORT=none TORCH_LOGS=output_code TORCH_LOGS_OUT=/tmp/tc_output_code.log \
/home/calebkim/.conda/envs/helion/bin/python _lab/harness/crosscheck_bias.py
```

## Artifacts
- `_lab/harness/sanity_5way.py` — 5-way driver
- `_lab/harness/crosscheck_bias.py` — bias cross-check harness
- `_lab/harness/crosscheck_artifacts/tc_default_output_code.log` — tc-default Inductor Triton
- `_lab/harness/crosscheck_artifacts/crosscheck_stdout.log` — cross-check raw output
- `_lab/harness/patches/torch_compile_rms_norm_default.patch` (+ README.md) — the new variant
