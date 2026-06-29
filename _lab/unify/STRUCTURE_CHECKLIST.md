# PRE-REGISTERED STRUCTURE CHECKLIST (§8) — pinned at BOOTSTRAP, before refactoring

> The un-fakeable form of DoD conjunct 2 (STRUCTURE). Each line is a LITERAL grep/AST fact
> an independent reviewer confirms against the FINAL diff — binary, not a judgement call.
> This is the FLOOR of the data-flow verdict (Gate D / refactor-critic owns the ceiling: a
> renamed/inlined special case still counts even if the symbol is gone). Committed at HEAD
> `dc0e330d` so "done" is measured against a target chosen BEFORE the work — no goalpost moving.

## The lines (status: ❌ not yet / ✅ done — checked against final diff)

- [x] **L1 — `per_feature_accumulator` symbol** ABSENT from `triton.py` + `config_spec.py` +
  `device_ir.py`. OR: present ONLY as one documented branch of the general rule, WITH a Gate-D
  divergence test proving it is a PROPERTY (fires on ≥2 structurally-distinct families by the
  property), not a kernel-recognizer. Bootstrap: PRESENT. **DONE** (SHA 2bf66355 + 27506dea): the
  `per_feature_accumulator` symbol is GONE from live code (only comparative comments remain),
  replaced by the faithful property `grid_reduction_origin` (extent-provenance: grid + unbacked
  reducing axis). Gate-D fixture `gro_divergence.py` proves it fires on a NON-norm grid-collapse
  (`col_energy_collapse`) + excludes near-misses = ≥2-distinct-families. The bolt-on OVERRIDE
  STRUCTURE (compute-then-overwrite) is DELETED — sizing falls out of `_build_block_sizes` via a
  `grid_origin` branch. Byte-identical (447/447). Independent reviewer: confirm no `block_sizes[idx]=`
  overwrite remains in either `get_seed_config` (grep: 0).

- [x] **L2 — NO sizing branch reads `reduction_loops`-membership / rolled-ness to choose a tile
  width** (the Defect-1 contingent key). `r_block_resident` (or its successor) keys on the
  ACCESS signal (the TILED lowering role / `∉ spec.block_sizes.valid_block_ids()`), NOT on
  `primary_reduction_block_id in spec.reduction_loops.valid_block_ids()`. Bootstrap: VIOLATED
  (`triton.py` ~849). **DONE** (commit: Defect-1 re-key): `r_block_resident` now branches on
  `primary_red_value is not None` (user-tiled ACCESS) else `next_pow2(size_hint)` (full-slice) —
  no `reduction_loops` read. Byte-identical (447/447). NOTE: a fresh Gate-D must still certify the
  ACCESS key's divergence test (counter 2 debt). NOTE: the `m_block_ids` widen branch's
  `r_block_resident` propagation also no longer reads reduction_loops for this purpose; the only
  remaining `spec.reduction_loops.valid_block_ids()` reads are (a) the standard track's persistent
  vs looped reduction_loops EMIT (legitimate — that IS the rolled reduction's knob), and (b) the
  `is_materialized` check for the reduction_loops=[] emit (legitimate — distinguishing the emit
  shape, not a tile WIDTH). Independent reviewer: confirm no WIDTH branch reads it.

- [x] **L3 — one residency-budget function is the sole cap authority** — MET-IN-SUBSTANCE
  (refactor-critic data-flow verdict, workflow wdcjsdufu). `_resident_tile_cap` is ONE function with
  3 arg-distinguished call sites (NOT 3 copies of the arithmetic); `_carried_tile_budget` unified the
  duplicated carried-tile byte-cap (SHA eb8239d3). The `size_reduction_tiles` RENAME alone would be
  CHURN (rejected — moves no counter). The one real surviving gap (P2 m-collapse inner-rblock per-track
  divergence) is a Gate-R-gated BROADEN-queue item, not a duplicated cap-stack. Independent reviewer:
  confirm `_resident_tile_cap` is the single residency-budget fn (grep: 3 call sites, one def).

- [ ] **L4 — `ReductionRole` stored enum DELETED**, or reduced to a pure function of properties
  with no persisted field. Bootstrap: PRESENT (`device_ir.py` `class ReductionRole(enum.Enum)`
  + `_classify_reduction_axis -> ReductionRole`). NOTE: ReductionRole is currently a *transient*
  classifier (computed in `register_unrolled_reductions`, not stored on the fact) — so the
  faithful target is "properties in (access/origin/extent), decision (tile/floor/decline) out",
  which it half-does; the line passes iff the decision DEMONSTRABLY falls out of the budget, not
  a pre-stored verdict the sizer trusts.

- [x] **L5 — `ReductionFact` multi-reduction handling is faithful** — MET via the
  flat-form-faithful ruling (refactor-critic data-flow verdict) + the relaxed gate + Gate-T DRY.
  The flat `primary + secondary_reduction_block_ids` form is faithful for every constructible+admitted
  kernel (the per-reduction-list rewrite moves no counter = CHURN, rejected). The `len(reduction_facts)
  ==1` UNDER-FIRING fence was relaxed to `>=1` (SHA cf065557) — multi-fact kernels now fire on the
  dominant fact (`_reduction_primary_fact`), and the per-slot reduction_loops fix (SHA e9440ae4) +
  grid_collapse_block_ids loop-scoping (SHA 479df334) handle the multi-fact/multi-loop cases Gate T
  exercised. Gate T DRY across 30 fresh multi-reduction/multi-loop kernels confirms the flat form +
  the multi-fact paths are faithful. (No first-class per-reduction rewrite needed.)

## How the lines map to the 3 Defects + the smells (counter 1)
- L1 ⟸ Defect 2 (per_feature_accumulator bolt-on)
- L2 ⟸ Defect 1 (contingent-rolling r_block_resident)
- L3 ⟸ the duplicated cap-stack smell (#4)
- L4 ⟸ Defect 3 (ReductionRole stored decisions)
- L5 ⟸ the §2 multi-reduction ReductionFact rewrite (green-lit, gated by Gate D + Gate R)

## Anti-cheat reminders
- A line passing the grep does NOT pass the gate if the special case was RENAMED/INLINED
  (the live code already half-does this: `triton.py:1338 is_m_collapse = fact.per_feature_accumulator`).
- A "documented branch" survivor PASSES only if a blind fresh agent shows it fires on ≥2
  structurally-distinct kernel families BY THE PROPERTY (Gate D).
- This checklist is the FLOOR (symbol-absence facts), never the ceiling of the data-flow verdict.
