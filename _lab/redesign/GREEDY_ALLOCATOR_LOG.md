# Greedy-allocator unification work log (#4 + #2/#3)

Task: unify `_reduction_rblock` + `_build_block_sizes` (+ the scattered M-axis cap loop and
ad-hoc carried caps) into ONE per-co-residency-group greedy-budget allocator
(`size_reduction_tiles`), per PROMPT §2.3/§2.4/§6.2.1. Fix findings #2 (`_carried_leading_dims`
leading-dim-only) and #3 (`_carried_m_block_cap` 2-D `[M,R]` assumption) DURING the rewrite by
computing the group_footprint from dim MEMBERSHIP, not POSITION.

Branch `reduction-redesign`. Interpreter `/home/dev/helion/.venv/bin/python`.
`HELION_AUTOTUNE_EFFORT=none` everywhere. No GPU expected. No pip / no push.

Zero-diff oracle: the config recorder vs `_lab/unify/baseline_fc1dbaa0_configs.json` must show
ONLY the 2 known movers (`mreduction/{layer,rms}_norm_bwd/[4096,8192]/fp16`, both `[1,1]→[32,1]`).

---

## Step 0 — start (HEAD 22c519cc)

`git status --short` empty at HEAD `22c519cc` (the docs commit on top of `be374be7`). Branch
`reduction-redesign`. helion.__file__ resolves under the worktree.

Read brief + CARRIED_AND_GREEDY_FINDINGS.md + PROMPT.md §2.3/§2.4/§6.2/§6.2.1 in full, plus the
current `_TritonReductionSeedBase` + both subclasses + config_spec structs.

BASELINE GATE GREEN (before any edit):
- config recorder vs baseline_fc1dbaa0_configs.json: CHANGED 2 cells = the 2 known movers
  (mreduction/{layer,rms}_norm_bwd/[4096,8192]/fp16, [1,1]->[32,1]). The frozen baseline JSON is
  the PRE-redesign fc1dbaa0 corpus; the 2 movers are the redesign's pre-existing starvation FIX,
  already present at HEAD. So "ONLY the 2 known movers" == zero NEW movers.
- validate_kernel_fact 460/460, probe_assertions 13/13, test_reductions+test_autotuner_heuristics
  52p/22s, test_examples -k matmul_layernorm 2p/2s.

IR ground truth (fact_dump.py, new instrument): confirmed the #2/#3 witnesses' real shapes —
- grpo: carried accums dim_block_ids=[0,1] = BOTH grid axes, NO rdim (rdim=bid2). Reduction
  carried_2d_count=0 (rdim 2 not in any accumulator's last dim). So the carried caps DON'T fire on
  grpo (gated on carried_2d_count>=1 in _build_block_sizes; in _carried_m_block_cap the accum's
  last dim is grid-axis 1 not the rdim — a position-based misread, see #3a/#3 below).
- kl_div/jsd: carried accum [grid_M, rdim] — leading=grid, last=rdim. carried_2d_count=1/2.
- group_norm_bwd: rank-4 accums [2,3,3,1]/[2,3,4,1] (rdim not last, multiple positions); BUT
  goes through _grad_collapse_group which post-writes block_sizes -> the carried caps are dark.
- instance_norm_bwd: accums [None,1,None] / [2,1,None] — rdim NOT at [-1]; grad-collapse-overridden.
- rms_norm_bwd/layer_norm_bwd: carried [grid_M, rdim]; grad-collapse-overridden.
- rms_norm_per_block_quant: rank-3 accum [0,None,None]; the two reductions are SEQUENTIAL (g0/g1).
CONCLUSION: every >=2D carried accumulator whose rdim is NOT at [-1] or has multiple grid dims is
EITHER grad-collapse-overridden (norm-bwd) OR has carried_2d_count=0 so the carried caps don't
fire (grpo). The corpus zero-diff for #2/#3 holds. (Matches the brief's prediction.)

## Step 1 — P1: unified `size_reduction_tiles` allocator (zero-diff) — COMMIT <pending>

Built `_TileAllocation` NamedTuple + `size_reduction_tiles(env, spec, device_ir, pd)` on
`_TritonReductionSeedBase`. It is the ONE allocator that drives every axis of every co-residency
group from one budget (PROMPT §2.3/§6.2.1), SUBSUMING the old
`_m_block_product -> _reduction_rblock -> _build_block_sizes -> grad-collapse-override` threading
that each `get_seed_config` did inline. Internally it reuses the cap primitives (`_reduction_rblock`
as the per-reduction budget+chunk; `_build_block_sizes` as the grid-remainder + loop sizer) but in
ONE coherent pass with NO value re-threaded between separately-called passes and NO rediscovery —
plus it folds in the grad-parameter M-collapse (both tracks) and the user-tiled m-collapse r_block
override, which previously lived as post-write overrides in `get_seed_config`.

Both `get_seed_config`s now call `cls.size_reduction_tiles(...)` and map the result onto knobs
(standard: reduction_loops emission from persistent/r_block + num_warps ramp + coresident cap +
warp_override + eviction; user-tiled: num_warps + eviction). The track discriminator inside the
allocator is `_is_standard_reduction(pd)` (the primary's category, a Stage-1 ACCESS property).

GATE (after ruff format): config recorder = ONLY the 2 known movers; validate_kernel_fact 460/460;
probe_assertions 13/13; test_reductions+test_autotuner_heuristics 52p/22s; matmul_layernorm 2p/2s;
ruff format + ruff check helion/ clean. Byte-identical reproduction confirmed.

