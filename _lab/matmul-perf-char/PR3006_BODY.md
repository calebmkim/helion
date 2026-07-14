# [autotuner] H100 matmul seed: a general budget-formula heuristic

## What

Adds `TritonH100MatmulHeuristic` (sm90) — a **budget/roofline FORMULA** that turns any static
`(M, N, K, operand-width)` into a strong `(block_m, block_n, block_k, num_warps, num_stages,
l2_grouping)` config with **no lookup table**. It fires on every `_batched_static_matmul_fact`
(dense `matmul`, `fp8_gemm`, `bmm`, `mamba2_chunk_state`'s fused inner dot, any static batched dot)
and pins batch/outer axes to 1, so a batched dot and a bare GEMM are the same case.

`promote_seed_to_default = True`: the budget formula **owns the H100 no-autotune compiler default**
(and also seeds the autotuner), so a real GEMM on sm90 never falls back to the catastrophic
`[16,16,16]`-style default. Gated `HARDWARE_TARGETS = (("cuda","sm90"),)` — **H100 only**; other
GPUs are unaffected (B200/sm100 is the separate stacked #3007).

Also included (needed by the seed path):
- fact plumbing for arbitrary static (possibly batched) matmuls (`_batched_static_matmul_fact`),
- a ranked multi-seed hook (`get_seed_configs`) so a heuristic can plant a primary + diverse
  alternates for the autotuner search,
- a spawn-worker fix (`precompile_future.py`): re-register kernel-origin modules from file so a
  by-path-loaded kernel's `import <origin>` resolves in the benchmark/precompile worker.

## The formula (per-lever justification)

1. **Register-budgeted wide-N tile** — the fp32 `[bm,bn]` accumulator dominates registers, so
   `bm*bn <= ACC_BUDGET (32768)`; base aspect `[128,256]` (N is the coalesced store axis).
2. **Shape-clamp + spill-outward** — never tile past a dim; spend leftover budget on the other axis.
3. **SMEM-budgeted `block_k` + `num_stages`** — `bk <= K/PIPE`, fits `[bm,bk]+[bk,bn]` in SMEM
   (width via itemsize: fp8 gets a deeper K than fp32); `num_stages` = deepest pipeline that fits,
   capped at 2 for an occupancy-saturated batched dot.
4. **Wave-quantization fill** — shrink the tile only while it improves `grid/(waves*num_sm)`.
5. **num_warps ramp** — 8 for a large tile, else 4.
6. **`l2_grouping`** — reorder PIDs to share an L2-resident B operand on a tall tile-grid.

## Perf — broad H100 characterization (48 shapes, cold-L2, no autotuning)

The previous numbers here covered only a handful of aligned "home" shapes. This is a **broad, honest
sweep**: 48 shapes across 4 kernels on an H100 SXM (sm90, torch 2.13.0.dev+cu130, triton 3.7.0), 3
arms — this **seed** (the promoted no-autotune default) vs. the old `[16,16,16]` **default** vs.
**`torch.compile(max-autotune-no-cudagraphs)`** (best-of-cuBLAS+Triton) — timed cold-L2 with CUDA-graph
device time. Every shape reported, nothing dropped; the hard shapes (non-pow2, prime, deep-K,
tall-skinny, decode) are first-class. Numbers are the **geomean of the per-shape ratios**:

| kernel | shapes | vs. `[16,16,16]` default | vs. `torch.compile` max-autotune |
| --- | --- | --- | --- |
| `matmul` (bf16) | 22 | 20.8× | 0.84 |
| `fp8_gemm` (e4m3) | 10 | 19.3× | 0.86 |
| `bmm` (bf16) | 8 | 10.9× | 0.68 |
| **GEMM (matmul+fp8+bmm)** | **40** | **18.0×** | **0.81** |
| `mamba2_chunk_state` (bf16) † | 8 | 7.5× | 3.42 |

`vs. default` = `default_time / seed_time`; `vs. tc` = `tc_time / seed_time` (>1 = seed faster).
`torch.compile` selected cuBLAS/cuBLASLt on 37/40 GEMM cells (3 small shapes picked a Triton
template — logged per shape).

† `mamba2_chunk_state` has no cuBLAS analog (fused batched GEMM + state decay), so its
`torch.compile` arm is a **Triton** kernel; the seed beats best-Triton by 3.4×. Kept in its own row,
never folded into the GEMM-vs-cuBLAS aggregate.

**Aligned vs. adversarial (GEMM):** 0.85 on aligned/pow2 shapes vs. 0.61 on the adversarial
non-pow2/prime/tile-tail set — the seed's known weak spot, surfaced not hidden. A bounded autotune
probe on the two worst cells shows the split is real: the thin-K `bmm` gap is mostly
config-recoverable (autotune finds TMA loads + `l2_grouping` for a 1.56× win over the seed), while
the 8191³ prime gap is a Triton static-shape tail-mask ceiling autotune can't beat (it converges
back to the seed's own tile).

**Full detail — per-shape tables, raw per-round data, methodology, and an adversarial self-audit:**
https://github.com/calebmkim/helion/tree/matmul-perf-char-h100/_lab/matmul-perf-char
(`summary.md` = the report, `h100_results.jsonl` = machine-readable raw, `README.md` = reproduce).

## Tests

`test_autotuner_heuristics.py` (formula unit tests + eligibility), `test_benchmark_worker.py`
(spawn-worker re-register), regenerated `test_autotuner.expected` golden.


---
**Stack (via stack-pr):**
- #3006 H100 matmul seed ⬅ (this PR, bottom)
- #3007 B200 matmul: subsume the table
