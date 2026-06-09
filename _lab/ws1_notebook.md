# WS1 — Fortify reduction seed heuristic (sibling kernels + overfit hunt)

Fresh climb. Source of truth = this notebook + ledger key `ws1`. Trust the gated log over context (§6.1).
Branch: `reduction-pr-with-lab` (own worktree). DO-NOT-REGRESS baseline = the inherited reduction
heuristic serving the 9 forward kernels at fp32/bf16/fp16.

## The mission (from ws1-fortify-task.md)
Add sibling kernels that land in each overfit lever's regime; prove under the gates whether each
lever **generalizes** or is **overfit**; keep / faithfully-narrow / generalize accordingly; never
regress the 9. Bounded climb per sibling (geomean clears bar → move on). Primary deliverable =
gate-verified per-lever verdicts + initial transfer numbers, NOT per-sibling perf maximization.

## Done-bar (per-kernel GEOMEAN across 3 dtypes, RELAXED from strict per-shape — task scope)
A sibling is DONE / move-on when its geomean either:
- beats/matches tc (`seed ≤ tc`), OR
- within margin-of-error of oracle (`seed ≤ oracle × ~1.05`).
Measure INITIAL untuned transfer FIRST — if it clears the bar, do NO hill-climbing (hoped-for common
case). Only climb when initial transfer FAILS. Catastrophic single-shape outlier = overfit signal,
flag it even if geomean fine. No-regression backstop on the 9 still strict.

## The 3 named overfit suspects (start here, then hunt broader)
1. **REREAD_W8_MAX_BYTES + w8 branch** (`_num_warps` ~357-471). family-of-one: cross_entropy.
   sibling: **logsumexp** (+fp32 control). Gate keys on `row_reread AND not full_width_output`.
   Q: does w8 beat w32 on logsumexp bf16 vocabs (generalizes) or only CE (overfit)? + fp32 byte-cap boundary.
2. **STRUCTURED_COMBINE_CAP_BYTES + normalize-tile** persist_max_bytes/loop_chunk_bytes (Band C ~711-721, ~441-444).
   family-of-one: welford. sibling: **groupnorm** (welford bandmate, welford-idiomatic). HIGHEST RISK
   (code self-flags as M-unaware PROXY; loosening regressed welford(262144,5120) ~7.3x).
3. **persistent_interleaved + maxnreg=64** (T1 tail ~711-730). family-of-one: cross_entropy.
   sibling: **log_softmax** wide-N (untested FULL-WIDTH looped). Q: help full-width store, or CE-reread-specific?

Coverage probes (lower risk): **l2_norm** (ROW_PERSIST_MAX_BYTES byte-boundary), **argmax** (warp ramp + int64 acc).
Stretch: row_max/row_mean (only after core + verdicts banked).

## Facts the levers read (config_spec.py ReductionFact; built in device_ir.py)
- size_hint (rnumel), itemsize (fp32-promoted = 4 both dtypes for norm family), input_load_itemsize
  (HBM load width: 2 bf16/fp16, 4 fp32 — the dtype-faithful signal), num_load, num_carried_2d_tiles
  (Band-B: jsd=2 kl=1 welford=0), non_reduction_loop_block_ids (Band-C reduce-then-apply),
  row_reread (live across boundary), full_width_output (store over rdim [M,N] vs scalar [M]),
  grid_rows (M-grid product → occupancy = grid_rows//num_sm).

## Per-shape / per-lever status

### LEVER 2 — STRUCTURED_COMBINE_CAP_BYTES + normalize-tile (Band-C) — VERDICT: OVERFIT (combine cap), climbing
groupnorm initial transfer (seed-vs-tc CUDA-graph geomean) — FAILS the bar at bf16/fp16, marginal fp32:
| dtype | train geo | val geo | losers (wide-N small-M) |
|---|---|---|---|
| bf16 | 0.963 | **0.776** | (4096,8192)(1024,10240)(512,16384)(512,20480)(1024,24576)(512,32768); val min 0.425! |
| fp32 | 1.016 | 1.051 | (1024,10240)(512,20480)(1024,24576)(512,32768)(768,20480)(512,24576) |
| fp16 | 0.954 | 0.888 | (4096,8192)(1024,10240)(512,16384)(512,20480)(1024,24576)(512,32768) |
→ **OVERFIT SIGNAL CONFIRMED** — Band-C caps tuned on welford do NOT transfer to groupnorm. Losers
cluster at WIDE-N + SMALL-M (the code's self-flagged M-unaware proxy regime).
**Lever-2 A/B localizes the culprit = the COMBINE cap (STRUCTURED_COMBINE_CAP_BYTES):**
- COMBINE R_BLOCK cap (A): uncapping 8192→full HELPS groupnorm at every wide-N shape (-1% to **-17.4%**),
  i.e. the cap COSTS groupnorm 1-17%. groupnorm's lighter 2-moment combine affords a wider combine tile;
  the welford-tuned 32768-byte cap throttles it. Welford never sees this (welford curriculum N≤8192 →
  combine "already full", cap inert for welford).
- NORMALIZE tile cap (B): mostly HELPS groupnorm too (uncapping costs +6 to +62%) — so the normalize cap
  GENERALIZES (keep it). Exception fp32 (512,32768) where uncap helps -17.4% (edge).
- WELFORD control: normalize cap helps +100-110% at (262144,5120) [the 7.3× valley]; combine cap inert
  (already full at N≤8192). So narrowing the COMBINE cap is SAFE for welford (it never binds in-curriculum).
**FIX HYPOTHESIS — REVISED after decisive A/B:** the combine cap is NOT welford-vs-groupnorm overfit.
The wide-N small-M combine A/B shows the cap HURTS *welford too* (-9.8% bf16 / -11.5% fp32 at (512,16384)).
welford & groupnorm get IDENTICAL configs at the same (M,N) — same itemsize=4, same cap. The cap is simply
**MISTUNED: too tight at small M_BLOCK** because welford's TRAINING focused on huge-M (where M_BLOCK is
raised and the cap is correct). The real spill driver (per the code's own TODO) is **M_BLOCK × tile × itemsize**.
**THE FIX (M-aware, RAISE-ONLY combine cap):**
  combine_R = min(np2(N), max(STRUCTURED_COMBINE_CAP_BYTES//itemsize, COMBINE_PROG_BUDGET_BYTES // M_BLOCK // itemsize))
with COMBINE_PROG_BUDGET_BYTES = 262144 (a per-program resident-pressure ceiling, same 256 KiB scale as
NARROW_W1_OCC_BYTE_LIMIT — NOT a curriculum fence). M_BLOCK = the seed's floored block_sizes[0] (= raised
autotuner_min, computable at seed time via _block_floor). Raise-only: never below the validated 8192-elem floor.
**A/B validation (budget=262144):**
- WELFORD (no-regression control): every shape improves or flat. M_BLOCK=1 wide-N gains -1 to -11%; M_BLOCK≥4
  huge-M stays R=8192 (UNCHANGED → the 7.3× valley shape (262144,5120) is +0.0%, untouched). Worst +0.6% (noise).
- GROUPNORM (sibling): gains everywhere -1.1 to -16.5%; FIXES the (512,40960) partial-tile cliff that budget
  128k caused (+29.8%) — 256k lets it reach full R=65536.
Budget 262144 chosen over 524288 (which helped fp32 (131072,16384) M_BLOCK=8 but risks unvalidated bf16
huge-M); 262144 keeps M_BLOCK≥4 safely at 8192. Faithful (M-coupling), raise-only (welford-safe), Pareto-clean.
DONE: combine fix implemented + committed (9ad1c54e). Behavior oracle: only welford changed (45
configs), huge-M untouched, other 8 byte-identical. Welford before/after: combine-tile changes all
faster-or-flat (no regression). groupnorm AFTER combine fix: bf16 1.002/0.951, fp32 1.025/1.067,
fp16 0.995/0.936 (big lift from bf16 val 0.776).

### NORMALIZE-TILE sub-lever — verdict: faithful for bf16/fp16, NON-FAITHFUL for fp32 (dtype-gate)
Oracle on groupnorm losers: gap is the NORMALIZE tile (oracle 4096-8192 vs seed 2048). Flat M-aware
normalize widening (2048->4096) REJECTED — regressed welford fp32 mid-N M_BLOCK=1 (+5-10% at N=5120-8192).
Full INTERIOR width sweep (N=4096..32768, M_BLOCK=1, 3 dtypes) reveals the truth:
- **bf16/fp16**: norm 4096 helps MONOTONICALLY from N>=6144 (-2 to -20%), only N=4096 flat. CLEAN width gate.
- **fp32**: NON-MONOTONIC zigzag — HURTS N=4096-8192 (+2-9%), flat 10240-12288, helps 14336-16384,
  HURTS AGAIN N=20480 (+12%!), helps 24576+. NO clean width-separable rule for fp32.
→ The normalize tile DOES leave perf on the table for bf16/fp16 small-M wide-N, and a FAITHFUL gate
exists: key on input_load_itemsize<=2 (the HBM load width — bf16/fp16 rows, the SAME dtype-faithful
signal the w8/narrow levers use) AND looped + width. fp32 (itemsize 4 HBM) stays 2048 (its zigzag is
not faithfully keyable — correctly deferred to the autotuner; it's only a seed). TESTING this dtype-
faithful normalize gate next; if it's Pareto-clean on welford bf16/fp16 + lifts groupnorm over the bar, bank it.

### dtype-faithful normalize gate IMPLEMENTED (input_load_itemsize<=2 + width + M-aware, raise-only)
Gate: widen looped normalize chunk to 4096 only when input_load_itemsize<=2 (bf16/fp16 HBM row —
faithful, same signal as w8/narrow levers; fp32 itemsize-4 row stays 2048, dodges its zigzag) AND
np2(N)>raised (row spans >=2 chunks, so N=4096 stays floor) AND M_BLOCK small (footprint M_BLOCK*chunk;
huge-M keeps floor). Configs verified: bf16 widens at wide-N M_BLOCK=1, floor at N=4096 & huge-M & M_BLOCK=4;
fp32 normalize stays 2048 everywhere.
**MEASUREMENT ARTIFACT CAUGHT (footgun #7):** the batched before/after script (many kernels, one long-
lived process) reported phantom welford regressions (bf16 (65536,16384) +110%!, (4096,16384) +25%, fp32
(8192,12288) +20%, fp16 (8192,5120) +10%). CLEAN ISOLATED re-bench (fresh process, warmup+interleaved,
the proper method) shows ALL are FASTER-or-FLAT: (65536,16384) -1.7%, (4096,16384) -8.3%, (8192,12288 fp32)
-2.2%, (8192,5120 fp16) -0.5%, (8192,8192 fp32) +0.0% (config identical). Suspected the analysis not the
timer (method §5), re-benched full verbatim configs in isolation → artifacts confirmed. Running full
isolated no-regression sweep next.

### LEVER 2 — FINAL VERDICT: OVERFIT → FAITHFULLY GENERALIZED (both sub-levers). Pending Gate A + D.
Isolated no-regression sweep (24 changed welford shapes, fresh process each): ALL faster-or-flat,
**0 regressions** (bf16/fp16 wide-N -8 to -12.6%, fp32 -1.8 to -2.2%, unchanged +0.0%).
groupnorm after BOTH fixes CLEARS THE BAR all 3 dtypes:
| dtype | initial | combine-only | BOTH FIXES |
|---|---|---|---|
| bf16 | 0.96/0.78 | 1.00/0.95 | **1.12/1.10** |
| fp32 | 1.02/1.05 | 1.03/1.07 | **1.04/1.08** |
| fp16 | 0.95/0.89 | ~1.0 | **1.11/1.08** |
Both Band-C caps were M-unaware overfit (tuned at huge-M); fixed faithfully (M_BLOCK-aware footprint
+ input_load_itemsize for the dtype-dependent normalize optimum). The fix IMPROVES welford (the
family-of-one) too. Commits a11d28fc (combine) + 6d7cf8cc (normalize). NEXT: Gate A + Gate D.

### LEVER 1 — REREAD_W8 / w8 branch — VERDICT: GENERALIZES (pending Gate A + D)
logsumexp initial transfer (seed-vs-tc, CUDA-graph geomean, all CLEAR done-bar out of the gate):
| dtype | train geo | val geo | losers |
|---|---|---|---|
| bf16 | 1.213 | 1.293 | [2048,151936] (huge, not w8 window) |
| fp32 | 1.178 | 1.158 | [8192,24576] train only |
| fp16 | 1.234 | ~1.23 | none |
→ logsumexp DONE out of the gate (no climb). The w8 lever GENERALIZES — matched A/B (revert
num_warps 8→32 from seed): logsumexp bf16 w8 beats w32 by **+4.5–47.7%** at V∈{24576,32000,40960,49152,50257}.
CE control: w8 beats w32 +48.8/56.8%. fp32 byte-cap boundary EXACT: V=24576 fp32 (98304≤102400)→w8 +5.4%;
V=32000 fp32 (128000>cap)→w32 (correctly not fired); V=16384 fp32→w16 (ramp, rnumel not >16384).
NOT a CE-identity fence — fires on row_reread AND not full_width_output AND byte-cap; benefit transfers
to logsumexp (no target gather).
**GATES PASSED → VERDICT BANKED: GENERALIZES (no code change).**
- Gate A 3/3 skeptics PASS (no refute). All confirm w8 beats w32 +4.4-58.9% via faithful facts; w32 arm
  IS the ramp default with branch removed (monkeypatch REREAD_W8=0 → w32 verified); softmax/rms (full_width)
  and sum (not reread) correctly fenced out. Skeptic-1 framing caveat: say "SEED-vs-tc" (Product A
  configs=[seed]), not "out of the gate default" (EFFORT=none default_config=w4 loses tc — but seed is the
  correct arm per task). fp32 (8192,24576) per-shape loss tc/seed=0.912 at fp32 w8 boundary (within geomean).
- Gate D FAITHFUL: synthetic `amax_plus_sum` kernel (neither CE nor logsumexp) ALSO earns w8 → not identity.
  **BOUNDARY SWEEP resolves overfit-audit red-flag #1**: bf16 cap V=51200 — V≤cap w8 wins (+4-26%), V>cap
  w32 genuinely wins (+18-26%). The cap is a real hardware crossover, NOT a curriculum fence (it lands near
  the train/val vocab split only because both reflect the same ~50k real-model vocab regime).

## Tried-and-rejected (first-class data)
(none yet)

## Deferred hard-pile
(none yet)

## Overfit-hunt leads (from constant audit — Gate-E periodic raw material)
- **REREAD_W8_MAX_BYTES=102400** flagged: bf16 crossover at V=51200 sits near train(≤50k)/val(≥65k)
  vocab split. FRAMING CAVEAT: a heuristic boundary need not = curriculum split by luck. But MUST
  run a w8-vs-w32 BOUNDARY SWEEP (V=49152,51200,65536,98304) to confirm w32 genuinely wins past the
  cap (mechanism-driven), not that the cap was fit to fence train. → divergence test, part of lever-1.
- **ROW_PERSIST_MAX_BYTES=245760** flagged: persistent-vs-looped boundary bf16 N=122880 / fp32 61440.
  l2_norm sibling probes exactly this. Run a boundary sweep too.
- **STRUCTURED_COMBINE_CAP_BYTES=32768** + **persist_max_bytes=12288**: code self-flags as M-unaware
  proxy (welford 7.3x regression when loosened). groupnorm sibling = the divergence test. lever-2.
- Power-of-2 thresholds (LOOPED_CHUNK 16384, loop_chunk_bytes 8192, NARROW_W1 2048/262144,
  warps32_min_elems 16384) look hardware-faithful; lower priority but verify w/ held-out where cheap.

---
## LOG

### [2026-06-09] Step 0 PASS — banked
CE bf16 (8192,50257): seed 1.79× tc (CUDA-graph), acc exact, normalized cfg num_warps=8 (w8 branch
fired), hand-rolled crosscheck delta 0.01%. Env/seed-mechanism/harness all green. → entered loop.

### [2026-06-09] Task 2 — siblings authored + branch-verified (DONE)
Authored examples/{logsumexp,log_softmax,groupnorm,l2_norm,argmax}.py + wired all 5 into
bare_fwd_dtype.py (build-fns, KERNELS, baselines, argmax int64 acc path).
**CRITICAL idiom fix:** logsumexp/log_softmax first written with a pre-`.to(fp32)` on the row →
`itemsize=4` → fired w32 at V=32000 (NOT the w8 branch). Fixed to reduce the row in-dtype like
cross_entropy.py / softmax_two_pass (Triton accumulates fp32 internally, accuracy fine) → `itemsize=2`
→ V=32000 fires **w8**. The pure-sum family (l2_norm, sum) DOES need the explicit fp32 upcast.
Branch-firing confirmed (bf16):
  - logsumexp: row_reread=T full_width=F. V=8192→w16(ramp), V=32000→**w8** (64000≤102400), V=65536→w32. ✓ lever-1 probe
  - log_softmax: row_reread=T full_width=T. wide(131072)→**persistent_interleaved+maxnreg=64+sm_mult**. ✓ lever-3 probe (untested full-width)
  - groupnorm: Band-C nrl=(2,), num_load=4 (welford bandmate; 2-moment combine). block_sizes hit combine cap+normalize tile. ✓ lever-2 probe
  - l2_norm: num_load=1 row_reread=F streamed single-load; wide→looped(16384). ✓ ROW_PERSIST coverage
  - argmax: int64 out; exact-index acc path works (max_abs=0.0). ✓ warp-ramp coverage
Accuracy fp32 exact everywhere; bf16 max_rel is near-zero-output artifact → use max_abs (per task).
**Gate-D flag for later:** w8 cap keys on `itemsize` (=2 here because row reduced in-dtype). Question
the divergence test must answer: is `itemsize=2` a FAITHFUL footprint signal when the accumulator is
fp32 internally? (CE has the same property — so faithful-or-not, the sibling matches the family-of-one.)
</content>
</invoke>
