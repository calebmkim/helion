# Task brief: unify reduction sizing into ONE per-group greedy-budget allocator (#4), fixing carried-tile bugs (#2/#3) along the way

> This is the launch brief for a FRESH-CONTEXT agent. It is self-contained: orientation, the
> design intent, the exact files/paths, the gate commands (with cwd gotchas), the sequencing, and
> the definition of done. Read it fully before touching code. Companion docs (read these too):
> `_lab/redesign/CARRIED_AND_GREEDY_FINDINGS.md` (the verified findings + witnesses) and
> `/home/dev/local/prompts-lab/reduction-generality/PROMPT.md` (the LOCKED design — read §2.3/§2.4/
> §6.2/§6.2.1 especially; do NOT re-derive it).

## 0. WHAT THIS PROJECT IS
Helion's reduction SEED HEURISTIC (it picks a starting autotuner config for reduction kernels).
A two-stage redesign was built and audited:
- **Stage 1** `helion/_compiler/device_ir.py`: builds `ReductionKernelFact` — a list of
  per-occurrence `ReductionDescriptor`s + `CoResidencyGroup`s (the faithful Stage-1 taxonomy:
  FULL_SLICE / FULL_GRID / GRID_TILE / USER_TILE / DECLINED). Structs in
  `helion/autotuner/config_spec.py`.
- **Stage 2** `helion/_compiler/autotuner_heuristics/triton.py`: the seed heuristic
  (`_TritonReductionSeedBase` @ line ~544 + `TritonStandardReductionHeuristic` @ ~1631 +
  `TritonUserTiledReductionHeuristic` @ ~1834). It now reads the kernel fact's descriptors, not
  the legacy flat `ReductionFact` (that port is DONE). The `Cap`/`size_axis` primitive
  (triton.py ~506-541) is the agreed building block — KEEP it.
- A SEPARATE class `TritonMatmulReductionEpilogueHeuristic` (@ ~1984) handles fused
  matmul+reduction; it is NOT part of this task — DO NOT TOUCH IT. It legitimately still uses
  `ReductionFact`.

## 1. THE GOAL (#4 — the big one)
Stage 2 currently sizes tiles in SCATTERED, locally-greedy passes:
`_reduction_rblock` (sizes the reduction chunk r_block in isolation) THEN `_build_block_sizes`
(sizes the M/grid axes, reading r_block back to cap them, then non-reduction loops in a third
pass). Values are threaded between functions. This is NOT the design and the project owner objects
to it directly: "you really shouldn't have both the r_block and assign block sizes; they should be
unified into one function."

TARGET (PROMPT §2.3 "ONE function assigns ALL block sizes" + §6.2.1): a single allocator —
`size_reduction_tiles(kernel_fact, spec, env) -> {block_id: size}` (+ reduction_loops + num_warps
or whatever the seed needs) — that:
  1. Orders the CO-RESIDENCY GROUPS by priority (the group with the most/heaviest FULL-EXTENT
     reductions first; §2.3 cross-group ordering).
  2. For each group: form ONE budget = clamp(ceiling = byte/occupancy caps, floor =
     persistence/reuse), then GREEDILY assign in priority order — full-extent (FULL_SLICE+FULL_GRID)
     → user-tile → grid-tile — consuming the budget via the group_footprint rule (§2.3:
     `group_footprint = Σ over distinct resident tensors (∏ of that tensor's tiled dims)`; a shared
     dim DIVIDES, a parallel sibling tensor ADDS). Then the grid m_block takes the REMAINDER.
  3. Hold earlier groups' assignments FIXED when sizing later groups (shared tiles, e.g. a grid-M
     axis, are an input to later groups — §2.3 step 3).
  4. Size the non-reduction loops LAST, against the remaining headroom (own budget; §2.3 — they are
     a separate sequential pass, co-resident with nothing in the group).
This SUBSUMES `_reduction_rblock`, `_build_block_sizes`, `_m_block_product`, the M-axis cap loop,
and the carried-tile caps. The `Cap`/`size_axis` mechanism stays — what changes is that ONE
allocator drives every axis of a group from one budget, instead of r_block-then-m_block-then-loops
in separate functions.

## 2. #2 and #3 fall out of #4 (fix them DURING the rewrite, not before)
The carried-tile helpers are the same flaw twice: they assume the carried accumulator is a 2-D
`[leading_M, last_R]` tile. The corpus has rank-2/3/4 carried tiles, rdims NOT at `[-1]`, and tiles
with multiple grid dims. The group_footprint rule (∏ tiled dims, classify each dim by MEMBERSHIP —
is it an rdim? a grid axis? — not by POSITION) is the correct general statement and makes #2/#3
disappear. SPECIFICS + WITNESSES are in `CARRIED_AND_GREEDY_FINDINGS.md` §#2/§#3. Net:
  - #2 `_carried_leading_dims`: "leading dim only" → "any grid axis appearing anywhere in a carried
    >=2D accumulator's dims" (grpo witness: tile (0,1) has two grid dims).
  - #3 `_carried_m_block_cap`: drop `rdim=dim_block_ids[-1]` (use membership), drop the
    `dim_block_ids[0]==m_axis` gate (use `m_axis in dim_block_ids`), and the footprint is
    `M * Σ_buffers(∏ other-dims) * itemsize` — PRODUCT within a tile, SUM across separate buffers
    (witnesses: group_norm_bwd rank-4, rms_norm_per_block_quant rank-3, instance_norm_bwd rdim-not-last).
When the unified allocator computes group_footprint correctly, these caps are no longer separate
ad-hoc functions.

## 3. METHOD — zero-diff refactor FIRST, then generality
This is byte-identical-sensitive sizing code. The hard rule that has held for the whole redesign:
the config recorder must show ONLY the 2 known movers
(`mreduction/layer_norm_bwd/[4096,8192]/fp16` and `mreduction/rms_norm_bwd/[4096,8192]/fp16`,
both `[1,1]→[32,1]`, a pre-existing starvation FIX, NOT a regression).
STRATEGY:
  (a) Build the unified `size_reduction_tiles` to REPRODUCE the current corpus byte-for-byte first
      (it must emit what `_reduction_rblock`+`_build_block_sizes` emit today). Gate: only the 2
      known movers. Commit.
  (b) THEN let #2/#3's faithful group_footprint replace the ad-hoc carried caps. These are EXPECTED
      to stay zero-diff (the multi-dim carriers are grad-collapse-overridden or don't hit the cap on
      the corpus — verify). Commit.
  (c) Any NEW config mover beyond the 2 known = a faithful change. Per the owner's standing
      directive: it's ALLOWED if GPU-measured within 10% of the prior config; if a faithful read
      regresses >=10%, STOP and report (do NOT self-authorize). Most of this work should stay
      zero-diff; a mover is the exception, not the norm.
Work incrementally; EVERY commit must compile and pass the gate (never leave a half-built allocator
across a commit). The owner prefers small gated commits, but an atomic commit is acceptable when a
slice wouldn't compile (mutually-coupled signatures) — judgment call, documented.

## 4. ENVIRONMENT (verbatim — the gotchas matter)
- Worktree `/home/dev/local/helion-redesign`, branch `reduction-redesign`, HEAD `be374be7`
  (start CLEAN: `git status --short` empty). Base/runway commit to diff against conceptually is
  `de2c545f` (pristine `fc1dbaa0` helion/ + `_lab/` infra), but the byte-identical ORACLE is the
  frozen JSON below, not a git diff.
- Interpreter `/home/dev/helion/.venv/bin/python`. NEVER `pip install`, NEVER `git push`. Use
  `PYTHONPATH=/home/dev/local/helion-redesign` for scripts (they assert helion.__file__ is under
  the worktree). `HELION_AUTOTUNE_EFFORT=none` everywhere (no autotune).
- This is a GPU box but the zero-diff work needs NO GPU. GPU is ONLY for measuring a deliberate
  mover (§3c). If you need it, run FOREGROUND, one job at a time — NEVER background/detach a GPU job
  (it dies silently). You almost certainly won't need it.
- COMMIT after each gated-green step; end messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Keep a running work log at `_lab/redesign/GREEDY_ALLOCATOR_LOG.md` (append as you go: what you
  changed, gate result, commit SHA, decisions/surprises) so the work is resumable.

## 5. THE GATE (run after each logical step; cwd matters)
```
# (1) config recorder — MUST show ONLY the 2 known movers
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
  PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
  /home/dev/local/helion-redesign/_lab/harness/unified_config_recorder.py --out /tmp/a.json
cd /tmp && PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
  /home/dev/local/helion-redesign/_lab/harness/unified_config_recorder.py \
  --diff /home/dev/local/helion-redesign/_lab/unify/baseline_fc1dbaa0_configs.json /tmp/a.json
# EXPECT: "CHANGED 2 field(s) across 2 cell(s)" = mreduction/{layer,rms}_norm_bwd/[4096,8192]/fp16.
# Anything else = a non-identity change; investigate before committing.

# (2) Stage-1 fact faithfulness + (3) Tier-1 probes (fired-right-path)
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=/home/dev/local/helion-redesign \
  /home/dev/helion/.venv/bin/python /home/dev/local/helion-redesign/_lab/redesign/validate_kernel_fact.py   # 460/460
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=/home/dev/local/helion-redesign \
  /home/dev/helion/.venv/bin/python /home/dev/local/helion-redesign/_lab/redesign/probe_assertions.py       # 13/13

# (4) unit tests — RUN FROM THE WORKTREE DIR (from /tmp pytest reports "no tests ran")
cd /home/dev/local/helion-redesign && PYTHONPATH=$PWD HELION_AUTOTUNE_EFFORT=none \
  /home/dev/helion/.venv/bin/python -m pytest test/test_reductions.py test/test_autotuner_heuristics.py \
  -q -p no:cacheprovider                                                              # 52 passed, 22 skipped (may shift w/ fixtures)

# (5) matmul-epilogue MUST-NOT-BREAK
cd /home/dev/local/helion-redesign && PYTHONPATH=$PWD HELION_AUTOTUNE_EFFORT=none \
  /home/dev/helion/.venv/bin/python -m pytest test/test_examples.py -k matmul_layernorm -q -p no:cacheprovider  # 2 passed, 2 skipped

# (6) lint every edited file
/home/dev/helion/.venv/bin/ruff format <file> && /home/dev/helion/.venv/bin/ruff check helion/
```

## 6. THE KERNEL CORPUS (what the recorder/validators sweep — for tracing a mover)
- 9 reduction: `examples/{rms_norm,layer_norm,softmax,sum,cross_entropy,welford,kl_div,jsd,long_sum}.py`
- 8 transfer: `/home/dev/local/prompts-lab/transfer/transfer_kernels.py` (+ `examples/grpo_loss.py`,
  `examples/fused_linear_jsd.py`); shapes in `/home/dev/local/prompts-lab/transfer/shapes_transfer.py`
- 6 m-reduction (norm-bwd): `/home/dev/local/prompts-lab/vllm-bench/mreduction_styles_view_only.py`
  (bias_grad/dyt/group_norm/instance_norm) + rms_norm_bwd/layer_norm_bwd in examples/
- 5 vLLM: `/home/dev/local/prompts-lab/vllm-bench/kut/*.py`
- 13 stress probes: `/home/dev/local/prompts-lab/reduction-generality/kernels/<slug>/kernel.py`
- IR ground-truth dumper (to inspect a kernel's descriptors/groups/accumulators):
  `_lab/redesign/ir_introspect.py` (run with `--probes` or `--kernel <name>`).
- Frozen baseline (the byte-identical oracle): `_lab/unify/baseline_fc1dbaa0_configs.json` (447 cells).

## 7. CONTEXT YOU INHERIT (read for the WHY)
- `_lab/redesign/CARRIED_AND_GREEDY_FINDINGS.md` — #1(done)/#2/#3/#4 with witnesses + faithful rules.
- `_lab/redesign/WORKLOG.md` — the full phase-by-phase redesign narrative.
- `_lab/redesign/REDUCTIONFACT_REMOVAL_LOG.md` — the just-completed port (heuristic is now
  descriptor-native; the one subtlety: `_full_width_output` reconstructs a kernel-scalar over
  `{rdim} ∪ non_reduction_loops` because the per-descriptor flag is rdim-scoped).
- `/home/dev/local/prompts-lab/reduction-generality/PROMPT.md` — the LOCKED design (§2.3 budget +
  greedy allocation, §2.4 cap primitive, §6.2/§6.2.1 primary + num_warps owner). Build ON this.
- Recent commits `9fb3d9d8..be374be7` are this redesign's audit+port work.

## 8. DEFINITION OF DONE
- ONE allocator sizes all of a co-residency group's axes from one budget; `_reduction_rblock` +
  `_build_block_sizes` (+ the ad-hoc carried caps) are unified/subsumed, not separate threaded
  passes. #2/#3's position-based assumptions are gone (membership-based group_footprint).
- All gates green; config = ONLY the 2 known movers (or a documented, GPU-verified-<10% mover with
  the owner's sign-off).
- Work log current; a closing note appended to `_lab/redesign/WORKLOG.md` with the final SHA.

## 9. EXPLICIT NON-GOALS (do not do here)
- Do NOT touch `TritonMatmulReductionEpilogueHeuristic` or try to delete `ReductionFact` (separate
  task; it's legitimately used by the epilogue + eligibility gate).
- Do NOT remove the `rollable`/`pinned` descriptor fields (separate cleanup; they're unconsumed but
  out of scope here).
- Do NOT change Stage 1 (device_ir fact-building) unless the allocator genuinely needs a new
  descriptor/group field — if so, add it faithfully (a real workload property) and note it.
