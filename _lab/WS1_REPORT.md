# WS1 — Fortify the reduction seed heuristic: report (LIVE — updated as levers close)

**Scope:** Stress-test the reduction seed heuristic's "family-of-one" levers by adding sibling
kernels that land in each lever's regime, then prove under the gates whether each lever
**generalizes** or is **overfit** (faithfully narrow/generalize accordingly). Never regress the
existing 9 kernels at any of fp32/bf16/fp16. Branch `reduction-pr-with-lab`; baseline = the inherited
dtype-climb champion. NOT a PR — a validated lab result.

## Siblings authored (examples/)
`logsumexp`, `log_softmax`, `groupnorm` (welford-idiomatic Band-C), `l2_norm`, `argmax`. All
fp32-accumulate, correct at 3 dtypes (argmax via int64 exact-index + gathered-value tie-break).
Each verified to fire its intended heuristic branch (facts dumped). Wired into
`_lab/bench/bare_fwd_dtype.py` (seed-vs-tc, CUDA-graph device time, accuracy-gated geomean).

## Initial untuned transfer perf (the overfit-generalization signal) — seed-vs-tc geomean
| sibling | bf16 (tr/val) | fp32 (tr/val) | fp16 (tr/val) | initial verdict |
|---|---|---|---|---|
| logsumexp | 1.21 / 1.29 | 1.18 / 1.16 | 1.23 / ~1.23 | CLEARS bar out of the gate (no climb) |
| groupnorm | 0.96 / **0.78** | 1.02 / 1.05 | 0.95 / 0.89 | **FAILS bar (overfit signal)** → climbed |
| log_softmax | (pending) | (pending) | (pending) | — |
| l2_norm | (pending) | (pending) | (pending) | — |
| argmax | (pending) | (pending) | (pending) | — |

## Per-lever verdicts (the primary deliverable)

### Lever 1 — REREAD_W8_MAX_BYTES + the w8 branch in `_num_warps` (family-of-one: cross_entropy)
**VERDICT: GENERALIZES** (kept as-is, no code change). Gate A (3/3 skeptics PASS) + Gate D (faithful).
- logsumexp sibling fires the w8 branch at the bf16 w8-window vocabs (V≤51200) via the same faithful
  facts as CE (`row_reread ∧ ¬full_width_output ∧ rnumel·itemsize≤cap`). w8 beats w32 by **+4.4–58.9%**
  on logsumexp (CE control +48.8–56.8%); held-out shapes win; logsumexp clears tc at all 3 dtypes.
- **Gate D divergence test:** a synthetic `amax_plus_sum` kernel (neither CE nor logsumexp) ALSO earns
  w8 → the rule keys on workload properties, not kernel identity. **Boundary sweep:** bf16 cap V=51200
  is MECHANISM-driven — V≤cap w8 wins, V>cap w32 genuinely wins (+18–26%). The cap is a real hardware
  crossover, not a curriculum fence (it sits near the train/val vocab split only because both reflect
  the same ~50k real-model vocab regime).

### Lever 2 — STRUCTURED_COMBINE_CAP_BYTES + normalize-tile cap (Band-C; family-of-one: welford)
**VERDICT: OVERFIT → FAITHFULLY GENERALIZED** (code changed; no welford/9-kernel regression).
- groupnorm initial transfer FAILED the bar (bf16 val 0.78). The lever-2 A/B localized the culprit:
  both Band-C caps were **M-unaware** — tuned at welford's huge-M training shapes (where a narrow tile
  is correct) and so **too tight at small M_BLOCK** for the *whole* Band-C family (welford included:
  uncapping helped welford -10% at small-M wide-N).
- **Fix (faithful):** the spill driver is the per-program footprint `M_BLOCK · tile · itemsize` (the
  code's own TODO). (a) **Combine cap** made M_BLOCK-aware, raise-only: `max(8192-floor,
  262144//M_BLOCK//itemsize)`. (b) **Normalize cap** made M-aware + width-gated, **gated on
  `input_load_itemsize≤2`** (bf16/fp16 HBM rows — fp32's normalize-width optimum is non-monotonic, so
  it's left to the autotuner). Both raise-only → huge-M (the welford(262144,5120) 7.3× valley) is
  byte-identical.
- **Result:** groupnorm now BEATS tc at all 3 dtypes (bf16 1.12/1.10, fp32 1.04/1.08, fp16 1.11/1.08).
  Behavior oracle: only welford configs change; the other 8 kernels + huge-M welford are byte-identical.
  Isolated no-regression sweep: every changed welford shape faster-or-flat (0 regressions) — the fix
  *improves* the family-of-one too.
- **Rejected en route (logged):** a flat normalize widen (2048→4096 for all) regressed welford fp32
  mid-N +5–10% — the dtype-faithful gate (`input_load_itemsize≤2`) was the fix.
- **GATES: Gate A 2/3 PASS (not majority-refuted), Gate D PASS (faithful).** The lone refuter ran a
  *batched-process* harness that produced phantom +15–51% welford "regressions"; the other two
  skeptics independently identified it as a contention artifact (their isolated re-bench: -1.8% to
  -13%), one validated the OLD-config reconstruction against the true parent commit, and my own
  quadruple-confirmed re-bench (isolated CUDA-graph both-orderings + plain do_bench) shows NEW
  faster-or-flat. Gate D: M_BLOCK is device-derived (n_cus=132), the 262144 budget pins per-program
  footprint at 256 KiB (not an N-fence), `input_load_itemsize` is a faithful HBM-load-width gate.

### Lever 3 — persistent_interleaved + maxnreg=64 (T1 tail; family-of-one: cross_entropy)
**VERDICT: GENERALIZES (not overfit) — kept as-is, no code change.** log_softmax initial transfer:
bf16 1.02/0.89, fp32 0.90/0.96, fp16 1.08/1.11 (wide-N losers).
- **Matched A/B:** reverting `pid→flat` makes log_softmax **+1.7–7.7% SLOWER** (the lever HELPS the
  full-width store); `maxnreg→None` is +0.6–5.9% (helps/noop). The lever is neutral-to-helpful on a
  full-width store — NOT the source of the wide-N losses. CE control: bundle helps +1.8–20.2%.
- **The wide-N losses are a SEPARATE, UN-keyable issue** (surfaced by the hunt): the T1 seed *persists*
  a wide full-width re-read row (e.g. log_softmax (2048,98304): 196 KB ≤ ROW_PERSIST cap → persistent),
  which spills 16×. But the persist-vs-loop crossover for full-width rows is **wildly non-monotonic**
  (bf16: N=16384 loop −44%, 24576–49152 persist +45–135%, 65536 loop −18%, 81920 persist +175%, 98304
  loop −79%) — **no faithful constant separates good/bad persist; any cap would be curriculum-fit** (the
  exact overfit being hunted). **Flagged, not chased:** the wide-full-width-looped regime is genuinely
  hard for a single static *seed*. **[CORRECTED — see Lever 4.]**
- **Gate A (3 skeptics) MAJORITY-REFUTED my first lever-3 verdict** on two counts, both correct: (1) my
  "the persist crossover is non-monotonic / unkeyable" claim was a **GPU-contention artifact in my own
  sweep** — on an idle GPU the crossover is **monotonic and element-keyed** (~80k elems); (2) the lever
  is shape-dependent even on CE (regresses CE bf16 V=131072). **Gate F PASSED** the lever's mechanism:
  persistent_interleaved helps via **grid-tail quantization** (rounds the grid to 132×N even waves vs
  flat's ragged tail), not register-occupancy (store is chunked, maxnreg harmless) — so the lever
  **generalizes** (kept). But the refutation surfaced the real fix → **Lever 4**.

### Lever 4 — full-width-output persist cap (NEW; surfaced by the lever-3 adversarial hunt)
**VERDICT: OVERFIT → FAITHFULLY FIXED** (`FULL_WIDTH_PERSIST_MAX_ELEMS=65536`, element-keyed, full_width-gated).
- The persist decision used `size_hint × itemsize ≤ ROW_PERSIST_MAX_BYTES` (an HBM-input-byte cap). For a
  **full-width-output** row the persistent pass holds the **fp32-promoted** row tile resident to feed the
  store, so it spills at a row **WIDTH (elements)**, not input bytes. The byte cap undercounts a
  half-precision full-width T1 row 2× → log_softmax bf16 N≈98304 persisted a 196 KB input row (fp32
  resident 384 KB), spilling ~16× (2.4× slower than the looped oracle).
- **Fix:** a full-width row also caps persist at 65536 elements (just below the measured ~80k crossover,
  monotonic across bf16+fp32). **Gated on `full_width_output`** → scalar re-read rows (cross_entropy)
  keep the byte cap and persist far past it (CE bf16 N≤98304 persist is ~40% faster than loop — kept
  byte-identical, verified). The existing full-width 9 (rms/ln/softmax/welford) reduce `x.to(fp32)`
  (itemsize 4) and top out at N=16384 or already loop → byte-identical (behavior oracle confirmed).
- **Result:** log_softmax bf16 val 0.89→**1.18**, fp16 1.11→**1.19** (clears tc). Surgical: only the
  half-precision full-width T1 sibling (log_softmax) is steered; the 9 + CE unchanged.

## Overfit hunt (beyond the named suspects)
- Gate-E periodic audit (during-climb): no curriculum fence found; new constants (262144, 16384) are
  hardware-aligned per-program footprint budgets; WS1 splits are interpolation-fair.
- (TEST-firewall read at freeze: pending — Gate E reads TEST once.)

## No-regression on the existing 9 (the backstop)
- Lever 1: no code change → 9 byte-identical.
- Lever 2: only welford configs change (M-aware caps); welford faster-or-flat everywhere; other 8
  byte-identical (behavior oracle, all 3 dtypes).
- (Full freeze no-regression measurement: pending.)
