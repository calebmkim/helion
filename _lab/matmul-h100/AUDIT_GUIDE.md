# H100 matmul-seed climb — AUDIT GUIDE

This bundle is the hill-climb that produced **`TritonH100MatmulHeuristic`** — the H100/sm90
matmul autotuner-seed heuristic (shipped as pytorch/helion **#3006**). It has everything needed
to independently re-measure the perf claims: the heuristic code, the bench harness, the shape
curriculum, and the climb log.

## What's here
- `helion/_compiler/autotuner_heuristics/triton.py` — `TritonH100MatmulHeuristic` +
  `_h100_matmul_tile` (the budget/roofline formula) + `_batched_static_matmul_fact`. **This is the
  seed under audit.** It fires on sm90 for any static (possibly batched) `MatmulFact`.
- `_lab/matmul-h100/bench.py` — self-contained **cold-L2 + accuracy-gated** co-bench. Probes the
  live heuristic (`bind` → `config_spec`) to extract the emitted seed/default config, then times
  `{default | seed | literal-config}` back-to-back on identical inputs (median-of-N).
- `_lab/matmul-h100/sweep.py` — multi-shape driver (jobs JSON).
- `_lab/matmul-h100/autotune_one.py` — per-shape oracle (full autotune → the ceiling config).
- `_lab/matmul-h100/shapes.py` — the TRAIN/VAL/TEST shape curriculum for the 3 kernels
  (`matmul`, `fp8_gemm`, `mamba2_chunk_state`), with real-model citations.
- `_lab/matmul-h100/run_audit.py` — turnkey driver: runs `bench.py` per shape in an **isolated
  subprocess** over a split, prints per-shape ratios + geomean.
- `_lab/matmul-h100/NOTEBOOK.md`, `REPORT.md`, `ledger.jsonl` — the climb log + final report
  (i.e. the claims to audit).
- kernels: `examples/matmul.py`, `examples/fp8_gemm.py`, `examples/mamba2_chunk_state.py`,
  `examples/bmm.py`.

## Setup
- **Hardware: one NVIDIA H100 (sm90).** The seed only fires on sm90 (`matches_hardware`); on any
  other GPU the plain default fires and the heuristic is inert — you cannot audit it off-sm90.
- **Env:** a Helion dev env (PyTorch nightly + Triton). Pin ONE GPU and keep GPU work serial:
  `CUDA_VISIBLE_DEVICES=0`.
- **Path:** run from this worktree with it on `sys.path`: `PYTHONPATH=<worktree>`. `bench.py`
  hard-asserts that `import helion` resolves *under the worktree* (guards the silent-wrong-helion
  footgun — a stale system/editable install would audit the wrong code).

## Re-running
Probe (no GPU timing — shows the seed the heuristic emits + the `MatmulFact`):
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<wt> python _lab/matmul-h100/bench.py \
    --kernel matmul --shape 4096,4096,4096 --dtype bf16 --probe-only --out /tmp/p.json
```
Single co-bench vs cuBLAS (`torch.compile max-autotune-no-cudagraphs`):
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<wt> python _lab/matmul-h100/bench.py \
    --kernel matmul --shape 4096,4096,4096 --dtype bf16 \
    --configs '["default","seed"]' --tc --reps 5 --out /tmp/o.json
```
Full split (turnkey — one isolated subprocess per shape):
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<wt> python _lab/matmul-h100/run_audit.py \
    --kernel matmul --split train --tc --out /tmp/audit_matmul_train.json
```
Shape formats: `matmul`/`fp8_gemm` → `--shape M,K,N`; `mamba2_chunk_state` → `--shape b,seq,nh,chunk,hd,ds`.

## Metrics
- `G_vs_tc` = `tc_ms / seed_ms`. >1 ⇒ Helion faster than the baseline; this is the report's **G**.
- `x_over_default` = `default_ms / seed_ms`. Speedup over Helion's own (catastrophic) default.
- Only accuracy-passing configs get a `perf_ms`; a config that fails the fp32-upcast gate is excluded
  (perf `None`).

## Claims to check (from REPORT.md)
- matmul geomean **G**: bf16 0.959 · fp16 0.956 · fp32 1.465 · fp8 0.995.
- Every TRAIN shape beats the default **18–29×**.
- mamba: **4.5–7.7×** over default (no cuBLAS analog).

## Methodology caveats — READ before trusting any number
These are footguns this project actually hit. The harness handles most; an auditor must not undo them.
1. **Cold-L2, NO cudagraph.** `cold_l2_bench` uses plain `triton.testing.do_bench` (L2-flush between
   reps). Do **not** wrap in a cudagraph or reuse a hot L2 — for working sets ≤ ~50 MB that yields a
   fake 3–5 TB/s and fake speedups. This is the #1 footgun here.
2. **Accuracy-gate everything.** A wrong config trivially "wins" on speed. perf is `None` on gate
   failure; never report a win that failed accuracy.
3. **One kernel per process.** `run_audit.py` isolates each shape in its own subprocess; don't batch
   many kernels in one process (autotune/compile cache state contaminates timing).
4. **Baseline = `torch.compile max-autotune-no-cudagraphs`**, compiled + warmed **once** outside the
   timed loop (recompiling inside the timed fn measures dispatch, not the kernel).
5. **<25 µs noise floor.** Tiny shapes (decode M=1/8) sit near the timer floor; a G slightly >1 there
   is noise, not a real win.
6. **Interpretation gotchas — VERIFY, don't assume** (these change what a number *means*):
   - fp32 G≈1.47 is **Helion-TF32 vs tc-true-fp32** — a precision difference, not a same-precision
     win. At *equal* precision Helion is slower on a bare GEMM. Compare like-for-like before crediting it.
   - fp8 "tc" is `torch._scaled_mm` (a ~2.5× soft reference); the fp8 bar is *beat-default + approach
     the Helion ceiling*, not tc.
   - "18–29× over default" is vs Helion's own catastrophic `~[16,16,16]` default, **not** cuBLAS.
   - mamba has no cuBLAS analog — its only honest metric is `x_over_default`.
7. **ptxas hang footgun.** Some default / large-K-fp8 configs make ptxas spin for minutes (looks like
   a deadlock; the ptxas *child* burns CPU). Set `HELION_AUTOTUNE_COMPILE_TIMEOUT`; the seed avoids these.

## Splits
TRAIN was climbed on; VAL/TEST were held out behind an access firewall *during the climb* (see the
`shapes.py` docstring). For an independent re-audit that firewall no longer applies — run all splits
(`--split all`) and treat TRAIN vs VAL/TEST as an overfit check.

## Note
This is the lab **champion** branch (`matmul-h100-seed`); the shipped PR is pytorch/helion#3006 (same
formula + a few CI fixes). Unrelated to the perf numbers: the seed currently mis-promotes a WGMMA
config as the *no-autotune default* for nested-`hl.grid` matmuls on sm90, which hits an upstream
Triton miscompile — that's a correctness aside, not a perf-claim issue.
