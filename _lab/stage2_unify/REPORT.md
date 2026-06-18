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
