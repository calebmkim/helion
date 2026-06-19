# STAGE 2 REPORT — unify M-reduction into the standard/user-tiled track

**Base:** Stage-1 tip `26427709`. **Commits:** A `a90aa7b4`, B `ef8b390f`, C `faecdbfc`
(branch `reduction-3stage-stack`). **Box:** H100 sm90.

## Reframe (per the driver): a PORT, not a deletion
Current main has NO `m_reduction` heuristic — `MReductionFact` / `TritonMReductionHeuristic` /
`build_m_reduction_facts` are **grep-clean (0 references)**, so "delete m_reduction" is satisfied by
it never being introduced. Stage 2 makes the standard + user-tiled tracks serve every kernel a
bespoke M-reduction heuristic would, so none is needed.

## The three changes
- **A — materialized-feature recognizer (mechanical port).** `register_user_tiled_reductions` builds
  a STANDARD ReductionFact for an inner `ReductionLowering` axis the roller declined to roll (in
  NEITHER block_sizes NOR reduction_loops — rms/ln/instance backward). `_is_standard_reduction` keys
  on "rdim NOT a block_sizes entry" (covers rolled + materialized; byte-identical for the 9+8).
  Standard `get_seed_config` emits `reduction_loops=[]` for a materialized rdim.
- **B — M-collapse occupancy seed (hill-climb 1).** A user-tiled pure single-stream M-collapse
  (full_width_output False, num_carried_2d_tiles 0, no normalize loop, num_load==1) sizes the grid
  CTA for occupancy (next_pow2(grid_rows/num_sm), capped M_COLLAPSE_MAX_CTA=256) and reduces the CTA
  wave in one inner tile. `num_load==1` excludes dyt (3 loads + full grad_x [M,N]).
- **C — multi-materialized recognizer (hill-climb 2).** Materialized branch extended to ">=1, pick
  the dominant (largest-extent) feature reduction" — group_norm (2 materialized) fires standard T1.

## Result: every M-reduction kernel beats its default (DoD met)
| kernel | track | seed G (train) | default | note |
|---|---|---|---|---|
| rms_norm_bwd | standard T1 [1,1] | 0.80-0.90 | 0.14-0.22 | beats default ~5x |
| layer_norm_bwd | standard T1 [1,1] | 0.67 | 0.15 | beats default ~4x |
| instance_norm | standard T1 [1,1] | 0.48-0.68 (normal); 0.05-0.09 wide-S | 0.03 (broken/catastrophic) | beats default; wide-S kernel-authoring-bound |
| group_norm | standard T1 [1,1] | 1.16-1.89 (large-N); 0.035-0.27 wide-S | 0.038 / fails-to-compile wide-S | beats default; wide-S kernel-authoring-bound |
| bias_grad | user-tiled M-collapse | geo 0.94 (min 0.75) | 0.75-0.86 | beats default decisively (occupancy lever) |
| dyt | user-tiled (unchanged) | geo 0.66 | 0.21-0.33 | beats default; real seed, NOT a default-crutch |

vanilla-T1 perf is short of old m_reduction on rms/ln/instance — accepted as-is per the task
(recovering it is overtime, not a deletion requirement). Old m_reduction is reference only; the
anchor that matters this run is the DEFAULT, which every kernel now beats.

## Hard invariant — proven after EVERY edit
(a) 9 standard reduction kernels BYTE-IDENTICAL: config_recorder cfg_partB -> cfg_stage2A/B/C =
**0/739 changed** each time (config + the populator change is selection-only). (b) 8 Task-1 transfer
kernels UNCHANGED (config-identical: flj=[8192], grpo=[1,8,4096] excluded by num_load=7, rest [None]).

## Gate verdicts (fresh-context adversarial agents; ledger.json + gate_*.json)
- **Gate D: PASS** — recognizer doctrine-clean (no derived-fact walk; reuses _assemble_reduction_fact);
  M-collapse gate (full_width/carried_2d/normalize/num_load) + occupancy key are faithful structural
  properties, not a bias_grad fence; authored a lucky-proxy divergence kernel — the populator follows
  the real ReductionLowering, not bs.reduction.
- **Gate H: KEEP** — faithful occupancy key + structural M-collapse signature; num_load==1 a
  justified+tested fence (dyt genuinely regresses to 0.23 under the occupancy config); 256 a measured
  spill crossover, not a curriculum fence.
- **Gate F: mechanism-found** — the grid shrinks the M-way `gb_blocks.sum(0)` finalize from M
  partials (16384) to ~one wave (128≈num_sm); the 256 cap is the measured in-register spill boundary
  (512 rows 0.21x vs 256 1.00x); no inert field.
- **Gate R: accept** — both protected sets provably perf-invariant (byte-identical); only the 6
  out-of-scope M-reduction targets change, all lifted from default; grpo flip-risk excluded by num_load.
- **Gate E (freeze TEST read):** the structural recognizer generalizes — bias_grad geo ~0.91, dyt
  ~0.63 on TEST (both beat default), all fire; instance_norm wide-S kernel-authoring-bound but beats
  the broken/0.03 default.

## Deferred / logged
- rms/ln/instance vanilla-T1 perf < old m_reduction — OVERTIME (the M_CTA occupancy + inner byte-cap
  recovery), explicitly not required this run.
- Wide-S group_norm / instance_norm shapes are KERNEL-AUTHORING-BOUND (even old m_reduction couldn't
  beat tc there; the [32,32] default fails to compile) — characterized, not chased, per the task.
- Band-B unification (Stage 1) untouched here.

═══════════════════════════════════════════════════════════════════════════
## ADDENDUM (2026-06-19) — sub-problem D: recover the m_reduction perf (was overtime, now done)
═══════════════════════════════════════════════════════════════════════════
The original report logged "vanilla-T1 perf < old m_reduction (rms/ln/instance) — accepted as overtime."
Re-opened by the human. The four dual-axis backward norms (rms_norm_bwd, layer_norm_bwd, instance_norm,
group_norm) route to the standard MATERIALIZED branch but had their grid M block floored to 1, so the
per-feature grad accumulator finalized over M partials (the M-way collapse). The kernels ALREADY expose
the two-level tiling (`m_block = hl.register_block_size` + inner `hl.tile(mb_cta.begin, mb_cta.end)`),
so closing the gap is HEURISTIC-only — no kernel change.

### The change (one commit, `[stage2-unify]`, `triton.py` only)
In `TritonStandardReductionHeuristic.get_seed_config`, gated on
`is_materialized AND fact.per_feature_accumulator AND fact.full_width_output AND inner_tile_ids`:
- **grid M block → `next_pow2(grid_rows/num_sm)`** (`_m_collapse_grid_block`) — the dominant lever;
  shrinks the grad_w finalize from M partials to ~num_sm.
- **inner re-tile → byte-cap** (`_m_collapse_inner_tile`) against the TRUE product feature footprint
  (N for 2-D → inner 2-8; C*S for 3-D → inner=1, dodging the ws2 per-axis-MAX spill trap).
- **drop narrow-w1** for the collapse (unfaithful: it keys on the rdim extent, flooring instance's
  S=128 to 1 warp, but the resident tile is `[inner, feature]`-wide) → plain extent ramp.

`per_feature_accumulator` (the faithful grad-parameter-collapse provenance, fact field from 5f34e9d1)
is what makes the gate disjoint from everything else: it is False for all 9 standard + 8 transfer
kernels, so they are byte-identical.

### Perf — every dual-axis norm now matches/beats tc on the realistic regime (G = tc/seed, fp32)
| kernel | shape | BEFORE [1,1] | AFTER | old m_reduction |
|---|---|---|---|---|
| rms_norm_bwd | (4096,4096) | 0.90 | **1.49** | 1.48 |
| rms_norm_bwd | (8192,1024) | 0.72 | **1.81** | 1.85 |
| rms_norm_bwd | (8192,4096) | 0.64 | **1.07** | — |
| layer_norm_bwd | (4096,4096) | 0.68 | **1.49** | 1.52 |
| layer_norm_bwd | (8192,4096) | 0.49 | **1.13** | — |
| instance_norm | (512,64,128) | 0.67 | **1.65** | ~1.78 geo |
| instance_norm | (1024,32,256) | 0.43 | **1.13** | — |
| group_norm | (512,128,64,32) | 2.45 | **2.82** | 2.0-2.4 |
| group_norm | (1024,64,128,32) | 1.42 | **1.76** | — |

Wide-N (N=8192) is NEUTRAL, not regressed (rms 0.93→0.93, ln 0.74→0.75): the per-row feature
reduction dominates so the M-collapse buys little — old m_reduction was weak there too. bf16
spot-check all >1, accuracy-pass (rms 1.26/2.55, ln 1.44, instance 2.84, group 2.71).

### Invariant — held (a regression on either protected set would sink it)
- **9 standard reduction kernels byte-identical**: config_recorder final-code vs stashed-baseline =
  **739/739** unchanged (fp32/bf16/fp16 × train/val/robustness).
- **8 transfer kernels gate-disjoint**: compile-only probe → `per_feature_accumulator=False` (or
  non-materialized / user-tiled) for all 8 → GATE_FIRES=False.
- **bias_grad/dyt unchanged**: still `[128,128]`/`[128,8]` (user-tiled M-collapse, separate track).

### Deferred (smaller than before)
- Wide-N (N=8192) tail: m_reduction's HBM-bytes warp ramp (16/32) would close it; neutral as-is.
- A full curriculum sweep (all splits × dtypes) before landing — spot-shapes here are representative.
- Evidence: `RESULTS.md` (ADDENDUM), `probe_mcollapse.py`, ledger `perf_recovery_mcollapse`.
