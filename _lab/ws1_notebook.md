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
