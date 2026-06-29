# Make the reduction heuristic ReductionFact-FREE — self-contained task brief

## 0. ORIENTATION (read first — what this is)
This is the Helion reduction-seed-heuristic redesign. A two-stage design was built:
- **Stage 1** (`helion/_compiler/device_ir.py`): builds `ReductionKernelFact` -- a list of
  per-occurrence `ReductionDescriptor`s + `CoResidencyGroup`s (the faithful, positive Stage-1
  taxonomy). Structs in `helion/autotuner/config_spec.py`.
- **Stage 2** (`helion/_compiler/autotuner_heuristics/triton.py`): the seed heuristic
  (`_TritonReductionSeedBase` + `TritonStandardReductionHeuristic` +
  `TritonUserTiledReductionHeuristic`, ~lines 581-2047) that chooses the config.

A LEGACY flat fact `ReductionFact` predates the kernel fact. An audit found the heuristic was
still mostly reading `ReductionFact` and ignoring the kernel fact. A port has already moved every
STRUCTURAL and STORED-DECISION read onto the kernel fact (commits up through 2638ff36 on branch
`reduction-redesign`). What REMAINS: the heuristic still threads a `ReductionFact` (`fact`) through
its helpers for SCALAR reads (size_hint/itemsize/row_reread/...). 

YOUR TASK: make the reduction heuristic read the primary `ReductionDescriptor` (`pd`) for those
scalars too, so it NEVER reads a `ReductionFact`. Pure cleanliness -- behavior-identical (every
scalar is 1:1 with the descriptor, proven by the validator's INV3). `ReductionFact` stays BUILT
(the matmul-epilogue heuristic + the eligibility gate still use it); it just becomes unread by the
reduction tracks. FULL deletion of `ReductionFact` is a LATER, separate task -- NOT this one.

## 1. WHY IT'S SAFE / UNBLOCKED (the verdict that motivated this)
`build_reduction_kernel_fact` runs UNCONDITIONALLY right after `build_reduction_facts` on every
live compile (`device_ir.py` `lower_to_device_ir`, ~lines 3825 then 3829), ALWAYS assigning
`spec.reduction_kernel_fact`. So on the LIVE path the kernel fact is never None when reduction_facts
is populated -- every `kf is None` fallback in the heuristic is TEST-ONLY. The matmul-epilogue
(`TritonMatmulReductionEpilogueHeuristic`, a SEPARATE class @ ~line 2048, does NOT extend
`_TritonReductionSeedBase`) is the only legitimate `ReductionFact` consumer -- leave it untouched.

## 2. ENVIRONMENT (verbatim — the gotchas matter)
- Worktree: `/home/dev/local/helion-redesign`, branch `reduction-redesign`. Work from a CLEAN tree
  at HEAD (currently 2638ff36 or later). `git status --short` should be empty before you start.
- Interpreter: `/home/dev/helion/.venv/bin/python`. NEVER pip install / git push.
- COMMIT after each gated green step (immutable SHA per step). End commit messages with the
  Co-Authored-By trailer the repo uses.
- Scripts run from `cwd=/tmp` with `PYTHONPATH=/home/dev/local/helion-redesign`; they assert
  `helion.__file__` is under the worktree. BUT pytest must run FROM THE WORKTREE DIR (cwd matters:
  running pytest from /tmp reports "no tests ran").
- This is a GPU box but THIS task needs NO GPU (config recorder + validators run with
  `HELION_AUTOTUNE_EFFORT=none`, no autotune). If a faithful read ever moves a config (it should
  NOT here -- all scalar-identity), the rule is: GPU-measure it stays <10%, else escalate. You
  almost certainly will not need the GPU.

## 3. THE FIELD MAP (every fact.X the helpers read -> descriptor source; all verified equal)
  fact.size_hint            -> pd.size_hint
  fact.itemsize             -> pd.itemsize
  fact.input_load_itemsize  -> pd.input_load_itemsize
  fact.row_reread           -> pd.row_reread
  fact.reread_eviction_index-> pd.reread_eviction_index
  fact.num_load             -> pd.num_load
  fact.body_live_tiles      -> pd.body_live_tiles
  fact.full_width_output    -> pd.full_width_output
  fact.num_carried_2d_tiles -> pd.carried_2d_count
  fact.primary_reduction_block_id -> pd.block_id
  fact.m_block_ids          -> cls._grid_axis_block_ids(spec)   (= kf.grid_axis_block_ids)
  fact.non_reduction_loop_block_ids -> cls._non_reduction_loop_ids(spec)  (= kf field)
  fact.secondary_reduction_block_ids -> sized descriptors minus primary (already via kf in
                                        _secondary_red_values)
`ReductionDescriptor` fields (config_spec.py): category, block_id, graph_id, size_hint,
static_rnumel, itemsize, input_load_itemsize, rollable, pinned, carried_2d_count, row_reread,
reread_eviction_index, num_load, body_live_tiles, full_width_output. (Has every scalar above.)

## 4. THE SELECTOR (re-add — it was drafted then reverted from a clean-base decision)
Add to module scope (replacing `_primary_fact`):
```python
def _primary_descriptor_selected(env: CompileEnvironment) -> ReductionDescriptor | None:
    """Primary reduction descriptor: max ROW-BYTES (size_hint*input_load_itemsize) over BACKED
    sized descriptors (§6.2.1). NOT category tier-order (would flip rms_norm_per_block_quant).
    None if no sized reduction / no kernel fact. Zero-divergence from the legacy max-size_hint
    primary across the corpus."""
    from torch._inductor.utils import free_unbacked_symbols
    kf = env.config_spec.reduction_kernel_fact
    if kf is None:
        return None
    sized = [d for d in kf.reductions if d.category in SIZED_REDUCTION_CATEGORIES]
    if not sized:
        return None
    backed = [d for d in sized if not free_unbacked_symbols(env.block_sizes[d.block_id].size)]
    pool = backed or sized
    return max(pool, key=lambda d: (d.size_hint * max(1, d.input_load_itemsize), d.size_hint))
```
And `_is_standard_reduction(pd)` -> `return pd.category in FULL_EXTENT_CATEGORIES` (one line).

## 5. TRANSFORMATION (apply ONLY within ~lines 581-2047; leave the @2048 epilogue class ALONE)
1. The 16 base-helper signatures `fact: ReductionFact` -> `pd: ReductionDescriptor`. Helpers:
   _carried_tile_r_block_cap, _carried_m_block_cap, _num_warps, _resident_tile_cap,
   _pinned_inner_resident_elems, _m_block_cap, _m_axis_occupancy_cap, _m_block_product,
   _build_block_sizes, _reduction_rblock, _secondary_red_values, _m_collapse_grid_block,
   _m_collapse_inner_byte_cap, _m_collapse_resident_elems. (_carried_leading_dims takes only spec.)
2. Bodies: apply the field map. `cls._carried_2d_count(spec, rdim)` stays spec-based for the
   carried-M cap (it iterates accumulator dims, not the primary); for the primary's own count use
   `pd.carried_2d_count`.
3. The two `get_seed_config` heads: `fact = _primary_fact(env)` -> `pd = _primary_descriptor_selected(env)`;
   `if pd is None: return None`; thread `pd` everywhere `fact` went.
4. `is_eligible` heads: `fact = _primary_fact(env)` -> `pd = _primary_descriptor_selected(env)`;
   `return pd is not None and _is_standard_reduction(pd)` (standard) / `... and not _is_...` (user).
5. AFTER all readers are pd-native: drop the `fact` param + `kf is None` fallback from
   `_grid_axis_block_ids`, `_non_reduction_loop_ids` (-> take only `spec`, return the kf field).
   Drop the legacy `else` branch in `_secondary_red_values`. Remove `_primary_fact`.
   (Safe because get_seed_config only runs when _primary_descriptor_selected returned non-None,
   which requires kf present.)

## 6. THE TESTS TO PORT (`test/test_autotuner_heuristics.py`) — concrete
Two fixture patterns construct a `ReductionFact`. After the refactor:

(A) `_reduction_spec` (line ~702) + the `get_seed_config`/`is_eligible` tests (lines 748, 770, 796,
    819, 853-870). These call the REAL entry points, which will now read
    `_primary_descriptor_selected(env)` -> needs `spec.reduction_kernel_fact`. So `_reduction_spec`
    must ALSO append a kernel fact. Add (block_id=1 is a reduction_loops entry => FULL_SLICE):
```python
    from helion.autotuner.config_spec import (
        ReductionCategory, ReductionDescriptor, CoResidencyGroup, ReductionKernelFact)
    desc = ReductionDescriptor(
        category=ReductionCategory.FULL_SLICE, block_id=1, graph_id=0,
        size_hint=reduction_size_hint, static_rnumel=reduction_size_hint, itemsize=itemsize,
        input_load_itemsize=itemsize, row_reread=row_reread, num_load=num_load)
    spec.reduction_kernel_fact = ReductionKernelFact(
        reductions=(desc,), coresidency_groups=(CoResidencyGroup(graph_id=0, descriptor_indices=(0,)),),
        grid_axis_block_ids=(0,))
```
    Keep the `ReductionFact` append too (the eligibility gate `len(reduction_facts)` + matmul test
    in `test_not_eligible_without_single_reduction_tile` reads it). Adjust per-test fields
    (row_reread/num_load/size_hint) so the descriptor mirrors the fact -- they take the same kwargs.
    NOTE `test_not_eligible_without_single_reduction_tile` (line 811): the no-reduction spec has
    NO kernel fact -> `_primary_descriptor_selected` returns None -> not eligible. GOOD, keep.

(B) `test_dynamic_extent_normalize_tile_matches_reduction_tile` (line ~3621) calls
    `H._build_block_sizes(None, spec, fact(...), red_values, non_reduction_loop_ids=...)` DIRECTLY
    with env=None. If `_build_block_sizes` takes `pd`, build a `ReductionDescriptor` instead of the
    `fact(...)` helper. static_rnumel=None => DECLINED? NO -- this test wants a sized reduction with
    a dynamic extent. Use category USER_TILE (reduction_bid is a block_sizes entry), size_hint=4096,
    static_rnumel=None. Pass that pd. The `non_reduction_loop_ids={2}` arg is passed explicitly so
    it does NOT depend on the kernel fact -- fine. Verify the asserts still hold (bs[norm_idx]==777
    user-tiled, ==4096 standard). This test builds NO kernel fact and calls the helper directly, so
    the helper must work from `pd` alone (it does -- pd carries size_hint).

## 7. GATE after EACH logical group (verbatim commands)
```
# config recorder: must show ONLY the 2 known square movers
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
  PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
  /home/dev/local/helion-redesign/_lab/harness/unified_config_recorder.py --out /tmp/a.json
cd /tmp && PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
  /home/dev/local/helion-redesign/_lab/harness/unified_config_recorder.py \
  --diff /home/dev/local/helion-redesign/_lab/unify/baseline_fc1dbaa0_configs.json /tmp/a.json
# EXPECT: "CHANGED 2 field(s) across 2 cell(s)" = mreduction/{layer,rms}_norm_bwd/[4096,8192]/fp16.
# ANYTHING ELSE moving = a non-identity read slipped in; investigate before committing.

# fact faithfulness + Tier-1 probes
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=/home/dev/local/helion-redesign \
  /home/dev/helion/.venv/bin/python /home/dev/local/helion-redesign/_lab/redesign/validate_kernel_fact.py   # 460/460
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=/home/dev/local/helion-redesign \
  /home/dev/helion/.venv/bin/python /home/dev/local/helion-redesign/_lab/redesign/probe_assertions.py     # 13/13

# unit tests — RUN FROM THE WORKTREE DIR (not /tmp)
cd /home/dev/local/helion-redesign && PYTHONPATH=$PWD HELION_AUTOTUNE_EFFORT=none \
  /home/dev/helion/.venv/bin/python -m pytest test/test_reductions.py test/test_autotuner_heuristics.py \
  -q -p no:cacheprovider                                                                    # 52 passed, 22 skipped (may shift w/ fixtures)

# matmul-epilogue MUST-NOT-BREAK
cd /home/dev/local/helion-redesign && PYTHONPATH=$PWD HELION_AUTOTUNE_EFFORT=none \
  /home/dev/helion/.venv/bin/python -m pytest test/test_examples.py -k matmul_layernorm -q -p no:cacheprovider

# lint EVERY edited file
/home/dev/helion/.venv/bin/ruff format <file> && /home/dev/helion/.venv/bin/ruff check helion/
```

## 8. SUGGESTED SLICING (bottom-up = no broken intermediate; commit each green)
a) leaf scalar helpers (signature+body): _carried_tile_r_block_cap, _num_warps, _m_block_cap,
   _carried_m_block_cap, _resident_tile_cap, _pinned_inner_resident_elems, _m_block_product,
   _m_axis_occupancy_cap, _m_collapse_* . Their callers still pass `fact` -- TEMPORARILY pass
   `pd` by resolving it at the call site, OR convert callers in the same commit. (Easiest: convert
   a helper AND its callers together so each commit compiles + gates green.)
b) _reduction_rblock, _build_block_sizes, _secondary_red_values + the get_seed_config / is_eligible
   heads (add the selector, thread pd).
c) drop the kf-None fallbacks (_grid_axis_block_ids, _non_reduction_loop_ids, _secondary_red_values);
   remove _primary_fact.
d) port the 2 test fixtures (§6).
e) confirm NO `fact.` reads remain in 581-2047 except inside the matmul-epilogue class:
   `grep -n "fact\." helion/_compiler/autotuner_heuristics/triton.py` -- the only hits should be in
   the epilogue (@2048+) and `_primary_descriptor_selected` is gone of fact entirely.

## 9. DEFINITION OF DONE
- No `fact: ReductionFact` param on any `_TritonReductionSeedBase` helper; the reduction tracks
  read only `pd`/`spec`/`env`. `_primary_fact` removed.
- All gates green; config = ONLY the 2 known movers; matmul-epilogue test green.
- A short note appended to `_lab/redesign/WORKLOG.md` recording the change + final SHA.

## 10. OPEN FOLLOW-UPS (do NOT do here; just leave noted)
- `rollable` / `pinned` descriptor fields are UNCONSUMED -> deletion candidates (commit 9fb3d9d8).
- FULL ReductionFact deletion (retire from device_ir + port the matmul-epilogue off it) -- separate.
