# RUN 2 — Live Notebook (reasoning trace, decisions + empirical why, tried/rejected)

> The DURABLE source of truth for run 2. A fresh worker/agent reads THIS + the run-1 reference
> (`HANDOFF.md` §4 traps, `FINAL_REPORT.md`, `ledger.json`) to continue losslessly. Newest at bottom.
> All file:line refs are approximate — grep the symbol.

## Orchestration (this harness)
- Hub = main session (only spawner). No `SendMessage` → no persistent agents; "persistent worker" =
  fresh one-shot `Agent` calls briefed off this notebook + ledger, OR the hub drives directly.
- **Workflow** for deterministic parallel fan-outs + gates (G-sweeps, A/B batteries, oracle runs,
  Product-B, multi-skeptic adversarial verification). GPU-partitioned: timing concurrency ≤ #idle GPUs
  (1/2/3; GPU0 has a co-tenant). NEVER 2 timing runs on one GPU (corrupts do_bench). Read-only agents share.
- Acceptance independently gated: nothing enters champion unless results-referee reproduces it AND
  adversarial-auditor passes. Reject cheats/unreproducible/correctness-fails; never loosen; keep going.
- Never stop. "stuck/converged/broken infra" = a prompt for the next move, not an exit.

## Canonical invocation (run 2 — note the -2 and NO sys.path.insert footgun)
```
cd /tmp && CUDA_VISIBLE_DEVICES=<idle 1|2|3> \
  PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction-2 \
  /home/calebkim/.conda/envs/helion/bin/python <script>
```
Every run-2 script: `assert helion.__file__.startswith("/home/.../wt-reduction-2/")`. Do NOT reuse
run-1 harness scripts verbatim — they hardcode the OLD path and `sys.path.insert(0, OLD)`, and
`"wt-reduction-2".startswith("wt-reduction")==True`, so they'd SILENTLY run old code.

## v8 baseline (the floor; O_in_sample = 0.9785 over 9 forward kernels)
Heuristic `TritonReductionHeuristic` (helion/_compiler/autotuner_heuristics/triton.py). Branches (all on
ReductionFact, never kernel identity): persistent-vs-structural-looped (cap 2^20) → num_warps rnumel ramp
4/8/16/32 → multi-load persist byte-cap (num_load≥2 & >128KiB → looped) → Band-B R_BLOCK cap
(num_tiled_accumulators≥1 → 16KiB) → Band-C welford (is_structured_combine; combine=largest_pow2_div(N)
capped, apply=min(np2(N),8KiB) looped-at-wide, combine cap 8→16KiB when apply looped) → M-block at floor.
Seed emits only block_sizes/reduction_loops/num_warps/num_stages (codegen knobs NOT seeded = Goal 2 target).

## TRAPS carried from run 1 (HANDOFF §4 — apply to EVERY new A/B)
1. Matched-lever A/B: vary ONE lever, hold others equal, vs the best SIMPLE alt (persistent/w32), NOT the
   catastrophic default strawman.
2. Oracle is a BUNDLE: re-bench the FULL VERBATIM oracle config; never isolate-a-lever-and-re-pair-it.
   To split seedable-vs-oracle-only: oracle block_sizes at DEFAULT codegen knobs, then +candidate knob.
3. do_bench noise floor: sub-25µs / tiny-M shapes — lift M or exclude; any "win" there is undefendable.
4. Shared machine: re-check nvidia-smi before/after; re-run >5% spread; GPU0 co-tenant.
5. Persistent-seed round-trip fix in place (_encode_flat_value(None)→.high) — keep it.
6. fp32 fixed; assert dtype every call (softmax defaults fp16). Heuristic reads dtype/itemsize.

## Goal 1 — welford source fix + Band-C re-derivation  [STARTED 2026-05-30]
THE BUG: examples/welford.py L52 `Tn = chunk.size(-1)` = constexpr tile width, NOT masked valid count →
wrong mean/count/M2 on the last tile when block size doesn't divide N. Helion masks OOB loads with other=0
so sum_x/sum_x2 are already correct over valid cols; ONLY Tn is wrong. Fix: `Tn = (tile_n.index < n).sum()`
(grep `tile.index < bound` in helion/language/loops.py + examples/segment_reduction.py for the idiom).
CASCADE to undo: the buggy Tn forced combine tile to be a pow2 DIVISOR of N (→1 at prime N → cliff), which
spawned `is_structured_combine` + `apply_block_ids` + `largest_pow2_div` machinery. With the source fixed
the divisor constraint is GONE — combine tile can be a normal byte-capped min(N,cap). KEEP the legit
two-pass need: seed the APPLY pass wide (general T2 floors non-reduction blocks to 1 → catastrophic for
apply). Simplify/generalize Band-C; if is_structured_combine fits exactly one kernel that's a Goal-5 flag.
Plan: (1) fix+verify OOB zero-fill + correctness on well-factored/odd/PRIME(1543); (2) re-validate welford
oracle under corrected kernel (re-run fresh quick-autotune oracle on (262144,2048), compare to v8 cache; if
>3% beyond noise → oracle was confounded, re-run all welford oracles); (3) re-derive Band-C; (4) re-measure
welford in-sample G + pre-authorized welford TEST re-read; (5) commit (source fix ships w/ deliverable).
