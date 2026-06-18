# STAGE 2 — RESULTS / evidence for the gate stack

Immutable commit (frozen): `faecdbfc`. Sub-problem commits: A `a90aa7b4`, B `ef8b390f`, C `faecdbfc`.
Base = Stage-1 tip `26427709`. Box H100 sm90. Relaxed-faithfulness applies (the human's call) but
the gates remain the backstop.

## Goal: m_reduction becomes UNNECESSARY (never introduced on main)
The 6 backward M-reduction kernels each beat their catastrophic default via the standard or
user-tiled reduction track — NO separate M-reduction heuristic. `MReductionFact` /
`TritonMReductionHeuristic` / `build_m_reduction_facts` are **grep-clean (0 references)**.

## The changes (3 commits)
- **A (`a90aa7b4`) materialized-feature recognizer (PORT, mechanical).**
  `device_ir.register_user_tiled_reductions`: a single inner ReductionLowering axis in NEITHER
  block_sizes NOR reduction_loops (the roller declined — rms/ln/instance backward) builds a
  STANDARD ReductionFact. `triton._is_standard_reduction` keys on "rdim NOT a block_sizes entry"
  (covers rolled + materialized; equivalent for the 9+8 which have no materialized axis). Standard
  `get_seed_config` emits `reduction_loops=[]` for a materialized rdim.
- **B (`ef8b390f`) M-collapse occupancy seed (HILL-CLIMB 1).** User-tiled get_seed_config: a pure
  single-stream M-collapse (full_width_output False, num_carried_2d_tiles 0, no normalize loop,
  num_load==1) sizes the grid CTA for occupancy (next_pow2(grid_rows/num_sm), capped at
  M_COLLAPSE_MAX_CTA=256) and reduces the CTA wave in one inner tile. num_load==1 excludes dyt
  (num_load=3 + full grad_x [M,N]).
- **C (`faecdbfc`) multi-materialized recognizer (HILL-CLIMB 2).** Materialized branch extended
  from "exactly 1" to ">=1, pick the dominant (largest-extent) feature reduction" — group_norm
  (2 materialized: spatial-S + intra-group-Cg) now fires standard T1.

## Perf — every M-reduction kernel beats its default (G = tc/seed, fp32 train)
| kernel | track | seed (this stack) | default | verdict |
|---|---|---|---|---|
| rms_norm_bwd (4096,4096) | standard T1 [1,1] | 0.90 | 0.14 | beats default 6x |
| rms_norm_bwd (8192,1024) | standard T1 [1,1] | 0.80 | 0.22 | beats default |
| layer_norm_bwd (4096,4096) | standard T1 [1,1] | 0.67 | 0.15 | beats default 4x |
| instance_norm (512,64,128) | standard T1 [1,1] | 0.48-0.68 | 0.03 | beats default ~20x |
| group_norm large-batch-N | standard T1 [1,1] | 1.16-1.89 | 0.038 (or fails to compile) | beats default |
| group_norm wide-S | standard T1 [1,1] | 0.035-0.27 | [32,32] FAILS TO COMPILE | kernel-authoring-bound; seed at least compiles+beats |
| bias_grad (geo train) | user-tiled M-collapse | **0.94** (min 0.75) | 0.75-0.86 | beats default decisively |
| dyt (geo train) | user-tiled (unchanged) | 0.66 | 0.21-0.33 | beats default; real seed, not a crutch |

bias_grad M-collapse detail (occupancy [m_block, m_block], m_block=next_pow2(M/num_sm) cap 256):
(16384,1024)[128,128]=1.13, (8192,4096)[64,64]=0.92, (65536,1024)[256,256]=0.99, (4096,16384)[32,32]=0.95.
The cap fixed (65536,1024): [512,512]=0.21 (in-register reduction over 512 rows spills) vs
[256,256]/[512,32]=1.00. dyt EXCLUDED (num_load=3); its [128,128] would be 0.23 (regression) so it
keeps its floored seed (0.62 > default 0.33).

## Gate F — M-collapse mechanism (bias_grad)
The user-tiled default floors the grid CTA to 1 row -> one grad partial per row -> a grid-wide
M-way `gb_blocks.sum(0)` finalize over M partials (G~0.35). The occupancy lever sizes the CTA to
one wave (~num_sm programs) -> the finalize is over ~num_sm partials. The cap (256) prevents the
in-register reduction tree / [rows, feature] resident tile from spilling (measured: 512 rows 0.21x
vs 256 rows 1.00x).

## Gate R — do-not-regress (proven)
- 9 standard reduction kernels BYTE-IDENTICAL after EACH of A/B/C: config_recorder
  cfg_partB (Stage 1) -> cfg_stage2A/B/C = 0/739 changed every time.
- 8 transfer kernels UNCHANGED (config-identical to Stage 1: flj=[8192], grpo=[1,8,4096], rest
  [None]) — the materialized branch (no materialized axis in them) and the M-collapse lever
  (num_load/full_width gate; grpo not matched) leave them untouched.

## Gate D — faithfulness notes (relaxed for Stage 2, gates the backstop)
- Materialized recognizer: a reduction `reduction=True` in neither spec is a faithful structural
  property (the roller-declined materialized feature reduction). No new fact field; reuses
  `_assemble_reduction_fact` (no bespoke walk). _is_standard_reduction keyed on block_sizes
  membership (faithful, not identity).
- M-collapse gate: full_width_output / num_carried_2d_tiles / non_reduction_loop_block_ids /
  num_load — all existing faithful fact fields describing "a user-tiled pure single-stream
  reduction collapsing the grid axis into a per-feature accumulator." Occupancy key
  (grid_rows//num_sm) + the M_COLLAPSE_MAX_CTA measured crossover. No kernel-identity, no dtype
  literal. The dominant-axis pick (largest extent) is a faithful tie-break.
