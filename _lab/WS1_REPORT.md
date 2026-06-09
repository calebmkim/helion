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

### Lever 3 — persistent_interleaved + maxnreg=64 (T1 tail; family-of-one: cross_entropy)
(pending — log_softmax full-width looped probe)

## Overfit hunt (beyond the named suspects)
- Gate-E periodic audit (during-climb): no curriculum fence found; new constants (262144, 16384) are
  hardware-aligned per-program footprint budgets; WS1 splits are interpolation-fair.
- (TEST-firewall read at freeze: pending — Gate E reads TEST once.)

## No-regression on the existing 9 (the backstop)
- Lever 1: no code change → 9 byte-identical.
- Lever 2: only welford configs change (M-aware caps); welford faster-or-flat everywhere; other 8
  byte-identical (behavior oracle, all 3 dtypes).
- (Full freeze no-regression measurement: pending.)
