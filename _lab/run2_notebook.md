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

## FIREWALL MAP (from TEST_readonce.py — guard Goal 4 + TEST re-reads)
- Authorized TEST re-reads (ONLY these two): **welford TEST** (invalid under corrected kernel) +
  **rms_norm TEST G ~0.828** (no raw log). No other kernel's TEST column is re-read.
- welford: in-sample {1024,1536,2048,4096}@262144; VALIDATION {2560,3072}@262144,(65536,16384);
  TEST {(262144,5120),(262144,7168),(262144,1280),(262144,1543),(131072,2048),(262144,768)}.
  → Goal-4 promotes 2560(val)+5120(TEST) to **in-sample-v2** (+M-var (512,4096),(8192,4096)). I have swept
  5120,2560,1024,2048,4096,(8192,4096) — OK as in-sample/in-sample-v2. KEEP CLEAN for welford TEST re-read
  (do NOT tune on): **7168, 1280, 1543(canary), 768, (131072,2048)** (+ can add 3072).
- rms_norm TEST: {(256,4096),(2048,2560),(2048,1025),(4096,10240),(8192,2048),(1,131072),(65536,512)} —
  re-read authorized (regen the ~0.828). Goal-4 tiny-M (256,4096) may go in-sample-v2 (rms TEST regenerated).
- DISCIPLINE for the 7 NON-re-read kernels (layer_norm/sum/softmax/cross_entropy/kl_div/jsd): in-sample-v2
  shapes MUST be disjoint from their sealed TEST. Brief collisions to AVOID/substitute: softmax (4096,3072)
  IS softmax TEST → use a different non-pow2 (e.g. (8192,3072)); layer_norm (256,4096) IS layer_norm TEST →
  use rms_norm for the tiny-M (256,4096) only, pick a distinct tiny-M for layer_norm. Loss-kernel real-vocab
  in-sample-v2: pick BT not already in VALIDATION/TEST (validation already has 2048/4096 × {32000,128256,...}).

## Goal 1 — DONE + GATED (2026-05-31). Commits ddf8fc34 (source), 43492809 (band-C).
RESULT: welford source bug fixed (`Tn=(tile_n.index<n).sum()`), Band-C re-derived to two INDEPENDENT
byte caps (combine=min(np2(N),32KiB/it) [persistent — looping it regresses via serial recurrence];
apply=persistent np2(N) if per-row-valid-bytes<=12KiB else looped 8KiB chunk). Deleted largest_pow2_div +
apply<->combine coupling (bug-artifacts). welford in-sample G(orig 1024/1536/2048/4096) 0.911->0.926;
1536 +6.7% (combine 512->2048), 2560(v2) +12.2% (->0.973), 5120(v2) ~tie 0.696; prime-1543 0.082(WRONG)
->0.958(CORRECT+FAST). 8 non-welford kernels BYTE-IDENTICAL (20/20). Both gates PASS (referee ACCEPT,
auditor PASS). Auditor proved the is_structured_combine GATE generalizes (2 synthetic structured-combine
kernels fire); FLAG: byte-cap VALUES are welford-curriculum-fit -> validate w/ real 2nd kernel in Goal 5.
ORACLE CONFOUND CONFIRMED: corrected welford oracle picks non-divisor combine=4096 at N=2560 (buggy
kernel's accuracy gate would reject) -> v8 welford oracle was artificially divisor-constrained.
GOAL-2 SEED FOR WELFORD: big codegen-knob residual — N=4096 seedable 0.76 vs corrected-oracle 0.961
(tensor_descriptor indexing + load_eviction last/first); N=2560 oracle 0.987 (eviction); N=8192 0.737.
DEFERRED: welford TEST re-read (clean shapes 7168,1280,768,(131072,2048); 1543 canary already 0.958) ->
consolidated TEST pass w/ rms_norm TEST G (Goal 6). warps residual: N=5120 ramp gives w16 (0.696) vs w8
optimum (0.721) — left simple (no regression); possible later refinement.
