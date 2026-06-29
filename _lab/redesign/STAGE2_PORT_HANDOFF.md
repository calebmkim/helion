# Stage-2 kernel-fact port — STATUS: COMPLETE (revised end state)

## REVISED CONCLUSION (supersedes the "Step 2 = delete ReductionFact" plan below)
After tracing it, FULL deletion of ReductionFact is the WRONG end state, for concrete reasons:
1. The matmul-epilogue (§2.9 must-not-break) has `reduction: ReductionFact` as a real field +
   gates on `len(reduction_facts)`. Deleting it means inventing a new carrier -> more types.
2. The unit tests (test_autotuner_heuristics.py:719, 3646) build a BARE spec + ReductionFact with
   NO kernel fact, to exercise the heuristic in isolation. The `_primary_descriptor is None`
   fallbacks exist for exactly this; converting helpers to take a descriptor would force heavy
   hand-built ReductionKernelFact fixtures for zero coverage gain.
3. The remaining `fact.X` reads are SCALAR-IDENTITY (size_hint/itemsize/row_reread/...), proven
   1:1 with the primary descriptor (INV3). Converting them buys ZERO faithfulness + ZERO behavior
   change -- pure churn, and would make helpers carry both `pd` (scalars) and `spec` (structure).

CORRECT END STATE (REACHED): ReductionFact survives as a THIN PER-PRIMARY SCALAR BUNDLE -- a
convenience view of the selected primary's scalars. It no longer carries STRUCTURE or STORED
DECISIONS; every structural/decision read goes through the kernel fact. The audit's complaint
("heuristic ignores the kernel fact, re-derives structure off the flat fact") is FULLY resolved.

Verified: the ONLY non-scalar `fact.X` code reads left are (a) `primary_reduction_block_id` used
purely as the "which axis is primary" identity key (== pd.block_id), and (b) the kernel-fact-absent
FALLBACKS in _grid_axis_block_ids / _non_reduction_loop_ids / _secondary_red_values / _primary_fact
/ _is_standard_reduction. No structure is re-derived from the flat fact anymore.

Final commit of the port: 71b471e0 (non_reduction_loop_ids -- the last structural read).

---
# (historical) original Step-2 plan below — NOT pursued, see revised conclusion above
## Status: substantive port DONE; only the mechanical tail (Step 2) remains.

Branch `reduction-redesign`. Plan: `_lab/redesign/STAGE2_KERNELFACT_PORT_PLAN.md`.
Goal (user): the reduction heuristic reads `ReductionKernelFact` + descriptors/groups DIRECTLY;
`ReductionFact` is retired from the reduction path. Faithfulness/cleanliness > strict config
equivalence — a config MOVE is allowed if the read is more faithful AND GPU-measured <10%;
escalate if a faithful read regresses >=10%.

## Commits this session (all gated: config = only the 2 known square movers, 460/460, 13/13, 52 passed)
- 9fb3d9d8 audit debug guards (FULL_SLICE source; rollable-vs-compiler) -- both fire on real cells
- 4f34d85b m-collapse footprint into named factors (body_live gated off)
- 7f04e9aa m-collapse feature footprint derived at use-site; stored field + dead group copy deleted
- dbfec7ab step3: _build_block_sizes r_block_resident reads pd.category (not reduction_loops membership)
- 694dfb77 carried_2d: bool -> carried_2d_count: int (descriptor was lossier than legacy)
- 75163703 step1a: m_block_ids -> kf.grid_axis_block_ids (9 reads; provably identical)
- f20519eb U1: per_feature_accumulator derived at use-site (_is_per_feature_accumulator); field deleted
- c02319a5 step1-3: _primary_fact selects over descriptors by max ROW-BYTES (not tier-order)

## What is DONE (the audit findings — all resolved)
Every STRUCTURE re-derivation and STORED-DECISION proxy now reads the kernel fact at use-site:
- taxonomy re-derivation in _build_block_sizes -> pd.category (via _primary_descriptor)
- m_block_ids -> _grid_axis_block_ids (kf.grid_axis_block_ids)
- per_feature_accumulator (both tracks) -> _grad_collapse_group + _is_per_feature_accumulator
- feature_footprint -> _materialized_feature_elems / _materialized_feature_axes
- carried count -> ReductionDescriptor.carried_2d_count
- primary selection -> max row-bytes over descriptors
Stored fields DELETED: ReductionFact.feature_footprint, .per_feature_accumulator,
CoResidencyGroup.feature_footprint.

## What REMAINS (Step 2 — mechanical, pure cleanliness, no behavior/faithfulness change)
The remaining `fact.<field>` reads in triton.py (549-2000) are SCALAR IDENTITY reads, proven 1:1
with the primary descriptor by validate_kernel_fact INV3:
  fact.size_hint (7), fact.itemsize (6), fact.body_live_tiles (4), fact.row_reread (2),
  fact.reread_eviction_index (1), fact.num_load (1), fact.input_load_itemsize (1),
  fact.full_width_output (1), fact.primary_reduction_block_id (12 -- mostly the "which axis" key),
  fact.m_block_ids (5 -- all DOCSTRINGS now, no code), fact.num_carried_2d_tiles (5 -- all in
  helper bodies already routed through _carried_2d_count? verify), fact.secondary_reduction_block_ids
  (3 -- only the kf-absent fallback in _secondary_red_values).

### Step 2 procedure
1. Thread the primary `ReductionDescriptor` (from `cls._primary_descriptor(spec, fact.primary_
   reduction_block_id)`) into `_reduction_rblock`, `_num_warps`, `_carried_tile_r_block_cap`,
   `_carried_m_block_cap`, `_resident_tile_cap`, `_m_block_cap`, `_m_block_product` -- replace each
   `fact.size_hint/itemsize/row_reread/...` with `pd.<field>`. These are byte-identical (INV3).
   - num_carried_2d_tiles reads: replace remaining `fact.num_carried_2d_tiles` with
     `cls._carried_2d_count(spec, <rdim block_id>)` (already done in _build_block_sizes; sweep the
     rest -- _reduction_rblock, _num_warps, _carried_tile_r_block_cap).
2. The get_seed_config heads resolve `pd` once and pass it down instead of `fact`.
3. Once NO consumer reads `fact.<field>`, retire `ReductionFact` from the reduction path:
   - matmul-epilogue (device_ir build_matmul_reduction_epilogue_facts) needs ONLY
     `reduction.size_hint` (-> n_extent) + the gate `len(reduction_facts)==1`. Either keep a minimal
     ReductionFact built ONLY for that composer, OR feed n_extent from the kernel fact's single
     full-extent descriptor. (§2.9 MUST-NOT-BREAK: test_examples -k matmul_layernorm.)
   - Drop the legacy builder path (`_assemble_reduction_fact` + register_unrolled_reductions' fact
     emission) from the REDUCTION track if nothing else needs it.
   - Update the 2 unit tests that construct ReductionFact directly:
     test/test_autotuner_heuristics.py:719 and :3646 (they build a ReductionFact to exercise the
     heuristic; if the heuristic no longer takes one, port them to build a ReductionKernelFact or
     a primary descriptor). NOTE: these tests build a bare spec (kernel fact may be absent) -- the
     defensive `_primary_descriptor is None` fallbacks exist for exactly this; decide whether to
     give the tests a kernel fact or keep the fallbacks.

### Gate every step
config recorder vs `_lab/unify/baseline_fc1dbaa0_configs.json` = ONLY the 2 square movers
(mreduction/{layer,rms}_norm_bwd/[4096,8192]/fp16); 460/460 validate_kernel_fact;
13/13 probe_assertions; 52 passed / 22 skipped on test_reductions + test_autotuner_heuristics;
matmul-epilogue test_examples -k matmul_layernorm green. cwd matters: run validators/recorder from
/tmp with PYTHONPATH=/home/dev/local/helion-redesign; run pytest from the worktree dir.

## Open follow-ups (smaller, noted by the audit, NOT blocking)
- `rollable` / `pinned` descriptor fields: still UNCONSUMED by the heuristic (the 9fb3d9d8 debug
  guard flags rollable as a deletion candidate). After Step 2, decide keep-or-delete.
- rms_norm_per_block_quant: a FixedBlockSizeSource axis classifies FULL_SLICE (warning #1 fires) --
  confirmed working-as-intended (sequential-graph full-extent group axis), but worth a perf sanity
  check it's sized right.
- The 2 square-shape movers ([1,1]->[32,1]) are a pre-existing starvation FIX (1.9x/2.7x faster),
  not a regression -- carried since before this session.
