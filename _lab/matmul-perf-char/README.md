# H100 matmul-family perf characterization — harness + results

Broad, honest perf characterization of the H100 matmul-family seed
(`TritonH100MatmulHeuristic`, PR #3006) across **48 shapes / 4 kernels**, 3 arms, 2
cold-L2 timing methods, nothing cherry-picked. **Measurement + reporting only** — no
heuristic edits.

## What was measured

- **Seed under test:** `TritonH100MatmulHeuristic` — the H100 (sm90) budget-FORMULA of
  PR #3006. It fires on every `_batched_static_matmul_fact` (dense `matmul`, `fp8_gemm`,
  `bmm`, `mamba2_chunk_state`'s fused inner dot). **H100/sm90 ONLY** — gated by
  `HARDWARE_TARGETS = (("cuda","sm90"),)`; it declines on every other GPU. In PR #3006 it
  is promoted (`promote_seed_to_default=True`), so on H100 **the emitted config IS the
  no-autotune compiler default** — what `effort=none` returns — replacing the catastrophic
  `[16,16,16]` fallback. (The perf numbers were gathered on a byte-identical revision of
  the matmul config-generation code where the flag was still `False` and the config was
  read via `compiler_seed_configs[0]`; the config-gen logic —
  `_batched_static_matmul_fact` / `_h100_ranked_configs` / `_h100_matmul_tile` /
  `_h100_build_block_sizes` — is 0-diff, so the emitted configs and thus these numbers
  describe PR #3006's seed exactly. The promote flag changes only where the config is
  routed, not what it is.)
- **3 arms** (identical inputs, one process): `seed` / `helion_default` (the base
  `[16,16,16] w4 s1` the seed replaces) / `tc_max_autotune`
  (`torch.compile(op, mode="max-autotune-no-cudagraphs")` — best-of-cuBLAS+Triton).
- **48 cells:** matmul 22 (bf16) · fp8_gemm 10 (e4m3) · bmm 8 (bf16) ·
  mamba2_chunk_state 8 (bf16). The HARD shapes are first-class:
  non-pow2 / prime / deep-K / tall-skinny / decode.
- **2 timing methods, cold-L2, interleaved, raw per-round arrays retained.** M1 =
  CUDA-graph device time (flush-graph-diff) — **canonical**. M2 = `do_bench`.
- H100 SXM (sm90, 132 SM, L2 50 MiB → 256 MiB flush). torch 2.13.0.dev+cu130,
  triton 3.7.0, driver 595.71.05. TF32 off, bf16-reduced-precision-reduction off,
  `static_shapes=True`.

## Headline (M1 canonical; geomean of per-cell ratios)

| kernel | n | vs default (geo) | vs torch.compile max-autotune (geo) |
|---|---|---|---|
| matmul (bf16) | 22 | 20.8× | 0.84 |
| fp8_gemm (e4m3) | 10 | 19.3× | 0.86 |
| bmm (bf16) | 8 | 10.9× | 0.68 |
| **GEMM (mm+fp8+bmm)** | **40** | **18.0×** | **0.81** |
| mamba2_chunk_state (bf16) | 8 | 7.5× | 3.42 † |

† mamba has no cuBLAS analog — its `tc` arm is a Triton kernel, so `G_vs_tc` = seed vs
best-Triton (seed wins 3.4×), reported separately (never in the GEMM aggregate).

`vs default` = default_µs / seed_µs. `vs tc` = tc_µs / seed_µs (>1 = seed faster).
torch.compile picked cuBLAS/cuBLASLt on 37/40 GEMM cells (3 small shapes → a Triton
template won); mamba → Triton throughout. Every winner is recorded in the JSONL.

Aligned-friendly vs adversarial split (GEMM, M1): **0.85 aligned / 0.61 adversarial**
(non-pow2/prime/tile-tail/non-aligned) — the seed's known weak spot, surfaced not hidden.

All numbers are M1 (cudagraph device-time). M2 (do_bench) agrees to ~4% on long kernels;
it diverges on small/decode shapes only because `torch.compile`'s host-dispatch overhead
inflates the `tc` arm there — which would *flatter* the seed — so M1 is the fair headline.

## Config-recoverability probe (are the worst cells poor-heuristic or Triton-ceiling?)

A 600 s-bounded Differential-Evolution autotune (no LLM) on the two worst cells, then a
rigorous 4-arm cold-L2 re-bench (`autotune_one.py` + `compare_configs.py`):

| worst cell | seed vs cuBLAS | autotuned vs cuBLAS | autotune speedup over seed | verdict |
|---|---|---|---|---|
| bmm long_seq_attn (16,4096,128,4096) | 0.50 | 0.79 | **1.56×** | poor heuristic — autotune finds TMA (`tensor_descriptor`) loads + `l2_grouping` + a smaller N-tile the formula didn't reach for |
| matmul 8191³ (prime) | 0.26 | 0.28 | **1.06×** | **structural** — autotune converges back to the seed's own tile; the ~3.5× gap is a Triton static-shape tail-mask codegen limit, not a config miss |

So the thin-K bmm gap is mostly config-recoverable (a concrete heuristic-improvement
signal: reach for TMA + L2-grouping on thin-K batched); the prime gap is a Triton-vs-cuBLAS
ceiling autotune can't tile its way out of. (Characterization only — no heuristic change.)

## Files

| file | what |
|---|---|
| `summary.md` | **the full report** — per-shape tables, per-type/all-in/aligned-vs-adversarial geomeans, M1-vs-M2 analysis, default-failures, mamba own section, honesty caveats, harness caveat box |
| `h100_results.jsonl` | machine-readable raw — one record per (kernel,shape,method); every arm carries the full per-round `t_us[]` + status + tc `winner`; all summaries derived, not stored-in-place |
| `mmperf.py` | harness (`prepare` / `time` / `devinfo` / `backend` subcommands) |
| `mmperf_run.py` | orchestrator (shape loop, 120 s per-arm killpg timeout, JSONL checkpoint, `--resume`) |
| `summarize.py` | JSONL → `summary.md` (pure analysis, no GPU) |
| `collect_backends.py` | captures the torch.compile-selected backend per cell |
| `autotune_one.py` / `compare_configs.py` | the config-recoverability probe (DE autotune + rigorous N-arm re-bench) |
| `shapes_matmul_perf.json` | the shape curriculum |
| `AUDIT.md` | adversarial self-audit findings + dispositions |

## Reproduce

```bash
cd _lab/matmul-perf-char
export CUDA_VISIBLE_DEVICES=0            # pin one idle GPU; scripts auto-resolve the worktree
PY=<venv>/bin/python                     # a venv with torch+triton+helion (never pip install)

# Step-0 sanity (matmul 4096^3 bf16, all 3 arms, both methods)
$PY mmperf_run.py --shapes shapes_matmul_perf.json --out /tmp/step0.jsonl --step0

# full 48-cell sweep, one kernel at a time (foreground, GPU-serialized, resumable)
for k in matmul fp8_gemm bmm mamba2_chunk_state; do
  $PY mmperf_run.py --shapes shapes_matmul_perf.json --out h100_results.jsonl --only $k --resume
done
$PY collect_backends.py h100_results.jsonl     # add the tc winner per cell
$PY summarize.py h100_results.jsonl > summary.md
```

## Method notes

- **M1 (cudagraph graph-diff)** is canonical: `[flush+kernel]` graph minus `[flush]`
  graph, adaptive replay count, cold L2 every replay. Two-phase per cell (prepare-isolated
  then time) contains a potential `[16,16,16]` ptxas-hang/OOM (didn't fire on this H100 —
  the default compiled everywhere, just 5–40× slower).
- **R self-calibrates:** 7 rounds default, 15 when the seed's cold-L2 device time < 25 µs.
- Step-0 verified: accuracy passes, seed ≠ default, cold-L2 real (implied BW 0.54 ≪ 3.35
  TB/s HBM peak — not an L2-hot artifact), tc winner = cuBLAS `mm`, seed ≈ cuBLAS on 4096³.
- One known wrinkle: the 8191³ prime kernel has ~4% M1 replay jitter (its median sits ~10%
  above M2); reported as a range, not a false-precision point — see the harness-caveat box
  in `summary.md`.
