# Stage 3 — FACT LANDSCAPE: composed matmul + reduction-epilogue

Compile-time investigation (bind/factdump, NO GPU timing). H100 box, branch
`reduction-3stage-stack`, `HELION_AUTOTUNE_EFFORT=none`. Helion module confirmed under the
worktree. Factdump script: `/tmp/factdump_stage3.py`.

---

## TASK 1 — FACTDUMP TABLE

All bf16. matmul_rms_norm: M=131072, K=256. matmul_layernorm: its `check()` shapes.
N is `hl.specialize`'d (compile-time const) so it is NOT a tiled axis: **`n_block_id=None`**.

| kernel              | M,K,N (req)     | n_matmul | n_reduction | static_m | static_n | static_k | m_block_id | n_block_id | k_block_id | n_block_sizes | default_config.block_sizes | seeds fired |
|---------------------|-----------------|----------|-------------|----------|----------|----------|------------|------------|------------|---------------|----------------------------|-------------|
| matmul_rms_norm     | 131072,256,256  | 1        | 0           | 131072   | 256      | 256      | 0          | None       | 1          | 2             | [32, 32]                   | NONE        |
| matmul_rms_norm     | 131072,256,512  | 1        | 0           | 131072   | 512      | 256      | 0          | None       | 1          | 2             | [32, 32]                   | NONE        |
| matmul_rms_norm     | 131072,256,1024 | 1        | 0           | 131072   | 1024     | 256      | 0          | None       | 1          | 2             | [32, 32]                   | NONE        |
| matmul_rms_norm     | 131072,256,2048 | 1        | 0           | 131072   | 2048     | 256      | 0          | None       | 1          | 2             | [32, 32]                   | NONE        |
| matmul_layernorm    | 32,64,200       | 1        | 0           | 32       | 256*     | 64       | 0          | None       | 1          | 2             | [32, 32]                   | NONE        |
| matmul_layernorm    | 128,256,400     | 1        | 0           | 128      | 512*     | 256      | 0          | None       | 1          | 2             | [32, 32]                   | NONE        |

- lhs_dtype = rhs_dtype = bfloat16; lhs_ndim = rhs_ndim = 2 in every case.
- `*` matmul_layernorm rounds N up to the next pow2 in `static_n` (200→256, 400→512); the
  reduction axis keeps the true extent (200 / 400) — see census below.
- **CONFIRMED**: 1 MatmulFact + 0 ReductionFact; N is specialized (`n_block_id=None`); the
  over-N reduction rides the carried `[tile_m, N]` fp32 accumulator (AccumulatorFact
  `dim_block_ids=(0, N_block_id), itemsize=4`).
- **No seed fires** — not even `triton_skinny_gemm` (it requires `n_block_id` non-None +
  3 block_sizes; here N is specialized so only 2 block_sizes and `n_block_id=None`).
  `compiler_seed_configs = []`, `autotuner_heuristics = []`. Empty niche → default only.

### Graph census (the crux: IS the over-N reduction a ReductionLowering?)

**YES.** The `.sum(-1)` over the accumulator lowers to a `ReductionLowering`:

| kernel           | ReductionLowering census (graph_id, op, block_index)            | env reduction rdims (block_id, size, reduction-flag) |
|------------------|-----------------------------------------------------------------|------------------------------------------------------|
| matmul_rms_norm  | `[(1, aten.sum.dim_IntList, 2)]`                                | block_id=2, size=N, reduction=True                   |
| matmul_layernorm | `[(1, aten.sum.dim_IntList, 3), (1, aten.sum.dim_IntList, 3)]`  | block_id=2 (size=N_true) AND block_id=3 (size=N_pow2, reduction=True) — the two LN sums |

- The N reduction is a **materialized full-width rdim** (block_id=2 for rms / 3 for ln):
  `reduction=True`, but it is in **NEITHER `block_sizes` NOR `reduction_loops`** (both empty
  for reduction_loops). This is exactly the "MATERIALIZED feature reduction" case the
  user-tiled walker describes — except here a matmul is present.
- The reduction is in **graph_id=1** (the epilogue subgraph), separate from **graph_id=0**
  (the matmul main graph). `grid_block_ids = [0]` (M is the grid).
- The two block_sizes are: block_id=0 = M (grid, size_hint=M, min=1) and block_id=1 = K
  (tile_k, size_hint=K, **min=16** from the matmul min_dot_size). The reduction axis is a
  THIRD block_id (2 or 3) that is NOT a block_sizes entry.

---

## TASK 2 — SUPPRESSION + MATMUL MACHINERY MAP

### Where the over-N reduction is declined (exact file:line)

There are TWO independent decline points — one per reduction track. BOTH must be relaxed.

**(A) Standard (rollable) track — `register_rollable_reductions`** in `device_ir.py`.
The N rdim is iterated as a reduction (it has `reduction=True`), but the roller refuses it:

- `helion/_compiler/device_ir.py:928-934` — the per-rdim roller probe:
  ```
  if (roller.has_matmul_with_rdim(graph_info.graph)
      or roller.has_stack_tensor_with_rdim(...)
      or roller.has_unrollable_reduction(...)):
      can_roll_graphs = False
      break
  ```
  → `rdim_results.append((rdim, False, set()))` (line 937) → no `reduction_loops` entry,
  nothing stashed in `_rollable_reduction_records`. So `build_reduction_facts`'s standard
  loop (`device_ir.py:1194`) has nothing to build.

- The actual trip is in `helion/_compiler/roll_reduction.py:362-397`
  (`ReductionRoller.has_matmul_with_rdim`). **MEASURED**: for matmul_rms_norm, rdim=2 returns
  `has_matmul_with_rdim=True` in **graph 0** (the matmul graph), `False` in graph 1.
  The trip is NOT the literal n_block_id (that is None); it is
  `roll_reduction.py:393-394`: the addmm rhs operand `y[tile_k, :]` has a dim whose
  `block_idx is None` (the full-width specialized N) → `return True`
  ("a dimension with no block_id ... cannot be sliced by the roller").

**(B) User-tiled / materialized track — `register_user_tiled_reductions`** in `device_ir.py`.
This is where the materialized full-width rdim WOULD be picked up (it owns the
"MATERIALIZED feature reduction ... in NEITHER block_sizes NOR reduction_loops" case), but
it bails first:

- `helion/_compiler/device_ir.py:1094-1095` (inside `register_user_tiled_reductions`):
  ```
  if spec.matmul_facts:
      return
  ```
  comment: *"A matmul kernel is out of scope (its carried 2D accumulators have a static int
  last-dim the reduction-axis walk does not expect). Decline before walking."*
  Reached from `build_reduction_facts` at `device_ir.py:1217` (guarded by
  `if not spec.reduction_loops:` at 1216 — true here, so it would otherwise run).

**(C) Heuristic eligibility gate** — `_triton_reduction_eligible` in
`helion/_compiler/autotuner_heuristics/triton.py:301-305`:
```
return len(spec.reduction_facts) == 1 and not spec.matmul_facts
```
Even if a ReductionFact were registered, this gate excludes any kernel with matmul_facts, so
neither `TritonStandardReductionHeuristic` (triton.py:658) nor
`TritonUserTiledReductionHeuristic` (triton.py:793) would seed it. Same `not matmul_facts`
guard also in `common.py:52` (`is_canonical_row_reduction`).

### To register the epilogue reduction as a fact — what must be relaxed

The reduction axis IS already a discoverable rdim: env.block_sizes has block_id=2 (rms) /
block_id=3 (ln) with `reduction=True`, and there IS a `ReductionLowering` with that
`block_index`. The cleanest path is the **user-tiled / materialized track**, NOT the roller
(rolling is impossible — the accumulator is register-resident full-width, there is no HBM row
to chunk):

1. **Relax `device_ir.py:1094`** so a fused matmul+reduction is NOT declined wholesale. The
   existing `materialized_inner` branch (`device_ir.py:1120-1129`) already does exactly the
   right thing for a `reduction=True` rdim in neither block_sizes nor reduction_loops: it
   picks `red_block_id` and sets `non_reduction_loop_block_ids=()`. The N rdim qualifies
   (block_id=2, not in bs_ids={0,1}, not in rl_ids={}).
2. The axis is identified by walking `ReductionLowering.block_index` and dropping grid axes
   (`device_ir.py:1099-1107`) — this already finds block_id=2 (rms) / 3 (ln). The
   `block_info.size` is a static int (256/512/...) so the `size=None` decline
   (`device_ir.py:1149-1150`) does NOT trip. So the existing walk handles it once the
   matmul-facts guard is bypassed.
3. The comment's stated worry ("carried 2D accumulators have a static int last-dim the
   reduction-axis walk does not expect") needs checking: the accumulator
   `dim_block_ids=(0, 2)` last-dim IS a resolved block_id (2), not a raw int, so the walk's
   block-id-based logic appears compatible. Verify `_assemble_reduction_fact` digests it (see
   below).
4. **Relax `_triton_reduction_eligible` (triton.py:305)** for the composed case, OR add a
   NEW heuristic that gates on (matmul_facts==1 AND reduction_facts==1) rather than
   (reduction_facts==1 AND not matmul_facts).

### Can `_assemble_reduction_fact` digest the epilogue reduction as-is?

Mostly yes — it is reduction-AGNOSTIC and reads off `memory_op_facts` / `accumulator_facts` /
the liveness slice, no bespoke graph walk. Two notes specific to the accumulator-carried case:
- **`input_load_itemsize`**: the reduction reduces the register-resident accumulator, NOT an
  HBM load, so it feeds no `reductions_fed` row load → the primary `fed_sizes` path
  (`device_ir.py:1337-1343`) is EMPTY. The Band-B structural fallback
  (`device_ir.py:1346-1364`, matching a rank>=2 row load whose inner subscript IS the
  reduction axis) was written for exactly this ("accumulator-carried reductions feed a loop
  `+=`, not a ReductionLowering, so the row load is invisible") — but here there is no HBM
  row at the N axis at all (N comes from the matmul, x/y are loaded on M/K), so it likely
  yields 0. Acceptable (input_load_itemsize=0 is a defined fallback), but the byte-cap lever
  should not key on it for this family.
- **`num_carried_2d_tiles`**: the AccumulatorFact `dim_block_ids=(0, N_block), itemsize=4`
  has `dim_block_ids[-1] == red_block_id` → counts as 1 carried 2D tile (the
  `[M_BLOCK, N]` fp32 accumulator). This is the correct Band-B signal: the resident
  footprint that the seed lever must respect.
- **`full_width_output`**: the store `out[tile_m, :]` writes back over N → full_width=True
  (matches layer_norm-style). Good.

So `_assemble_reduction_fact` can digest it; the over-N reduction needs NO bespoke assembler,
just the guard relaxations + a co-occurrence-aware heuristic. It does NOT reduce an HBM load,
so the footprint signal comes from `num_carried_2d_tiles` + `static_rnumel` (=N), not from a
reduction-fed load itemsize.

### The MatmulFact — where built, fields

- Built in `helion/language/matmul_ops.py:307-320` (`enforce_dot_requirements`, called from
  the dot/addmm lowering). Fields populated:
  - `m_block_id = env.get_block_id(m)` = 0 (the grid/M tile).
  - `n_block_id = env.get_block_id(n)` = **None** (N is `hl.specialize`'d → full-width, no
    block_id) — CONFIRMED in every factdump row.
  - `k_block_id = env.get_block_id(k)` = 1 (tile_k).
  - `static_m/n/k` via `static_problem_extent` (`matmul_ops.py:296-306`): the resolved int
    extents. `static_n` is rounded to next pow2 for matmul_layernorm (256/512).
  - `lhs_dtype=rhs_dtype=bfloat16`, `lhs_ndim=rhs_ndim=2`.
- `enforce_dot_requirements` ALSO sets the K min_size = 16 (`min_dot_size`,
  `matmul_ops.py:272-288`) → block_id=1 (K) has min_size=16 (CONFIRMED).

### Existing matmul heuristics — do any fire?

- `TritonSkinnyGemmHeuristic` (triton.py:34): **NOT eligible**. Needs `n_block_id != None`
  (triton.py:51-58) AND 3 clamped block_sizes. matmul_rms_norm has `n_block_id=None` and only
  2 block_sizes (M, K — N is specialized, not a block_size). So the aspect-ratio test
  (>=8, triton.py:60-63) is never reached. CONFIRMED NOT eligible on all shapes (even though
  M/N=131072/256 = 512:1 would pass the ratio).
- `TritonB200MatmulHeuristic` (triton.py:234) / `_single_2d_static_matmul_fact`
  (triton.py:107-118): requires `len(block_sizes)==3` and
  `(m_block_id,n_block_id,k_block_id)==(0,1,2)`. Here block_sizes=2 and n_block_id=None →
  returns None. (Also sm100-gated; this is H100/sm90.) NOT eligible.
- So **NO existing matmul heuristic fires** — the specialized-N shape (only M,K tiled) falls
  outside every current matmul seed. This is the empty niche.

### How many block_sizes does matmul_rms_norm have?

**2**: block_id=0 = `tile_m` (M, grid, size_hint=M, min_size=1) and block_id=1 = `tile_k`
(K, size_hint=K, min_size=16). N is specialized → NOT a block_size. (The reduction axis
block_id=2 is a separate `reduction=True` rdim, also not a block_size.)

### Cleanest place to compose the fact

(a) **Recognize co-occurrence**: in `register_user_tiled_reductions` (device_ir.py:1094),
    replace the blanket `if spec.matmul_facts: return` with a check that, when matmul_facts
    are present AND there is a materialized full-width `reduction=True` rdim (the existing
    `materialized_inner` branch), proceeds instead of returning. The co-occurrence test is:
    `spec.matmul_facts` (len 1) AND a ReductionLowering whose block_index is a `reduction=True`
    rdim not in block_sizes/reduction_loops.
(b) **Register a ReductionFact** for the epilogue reduction via the existing
    `_assemble_reduction_fact` (materialized_inner branch → `non_reduction_loop_block_ids=()`,
    `m_block_ids = grid_ids = {0}`). No bespoke assembler needed (see digest analysis).
(c) **Compose**: keep the MatmulFact as-is. Either (i) a NEW composed
    `MatmulWithReductionEpilogue` NamedTuple holding both, OR (ii) leave them as two facts
    (1 MatmulFact + 1 ReductionFact) and have a new heuristic gate on the conjunction. Option
    (ii) is less invasive — the facts already carry everything; only the heuristic gate
    (`_triton_reduction_eligible` style) needs to admit `matmul_facts==1 and
    reduction_facts==1` and emit the footprint-aware M/K block_sizes.

---

## TASK 3 — DEFAULT-vs-BEST GAP + the seed lever

### default_config

**`block_sizes=[32, 32]`, num_warps=4, num_stages=1** — IDENTICAL for ALL N (256/512/1024/
2048) and both kernels. The default is N-blind: it floors tile_m=32, tile_k=32 regardless of
how the `[tile_m, N]` accumulator + `[tile_k, N]` operand footprint scales with N. This is the
gap the seed targets.

### Tunable block_sizes

Exactly **2** levers: `tile_m` (block_id=0, M; min 1, max M, the dominant footprint lever) and
`tile_k` (block_id=1, K; min 16, max K). Plus `num_warps` and `num_stages` (standard Config
knobs). N is NOT tunable (specialized). The reduction axis is not a block_size.

### What a good config looks like (footprint reasoning, from the template comment)

Resident SMEM/registers scale with N: the fp32 accumulator is `[tile_m, N]` (4·tile_m·N bytes)
and the bf16 y-operand tile is `[tile_k, N]` (2·tile_k·N bytes). The win regime is small-N
because both terms scale with N → SMEM-bound. Per the template/NOTEBOOK:
- N<=512: tile_m can be moderately large; the [tile_m,N] acc + [tile_k,N] operand fit ~228KB.
- N=1024: only **tile_m <= 32** survives.
- N=2048: **nothing fits** (SMEM wall) — no valid config; Helion's win vanishes.

So the seed lever is a **footprint-aware M_BLOCK (tile_m) chooser keyed on resident bytes**
(≈ `4·tile_m·N + 2·tile_k·N` under the SMEM budget), eligibility = "does a productive tile
fit" (NOT aspect ratio, which is why TritonSkinnyGemm is the wrong template). The default
[32,32] is N-blind: too small at N=256 (leaves occupancy on the table → the 1.4–2.7× default→
best gap the NOTEBOOK cites) and possibly too large near the wall. The lever needs the
`ReductionFact.static_rnumel` (=N) + `num_carried_2d_tiles` (the [M_BLOCK,N] fp32 acc) — both
already populated by the composed fact — to size tile_m, then ramp num_warps with tile_m·N.

---

## SUMMARY (decline points + plan)

- **Factdump**: 1 MatmulFact + 0 ReductionFact on all shapes; `n_block_id=None` (N
  specialized), `m_block_id=0`, `k_block_id=1`; 2 block_sizes (tile_m, tile_k);
  default=[32,32]/warps4/stages1 for ALL N; **no seed fires** (skinny-gemm needs n_block_id +
  3 block_sizes). The `.sum(-1)` IS a ReductionLowering over a `reduction=True` materialized
  rdim (block_id 2/3) in graph 1.
- **Decline points**: standard track `device_ir.py:928-934` via
  `roll_reduction.py:362-397` (`has_matmul_with_rdim` → True, tripped by the addmm's
  full-width N operand with `block_idx is None`, line 393-394); user-tiled/materialized track
  `device_ir.py:1094-1095` (`if spec.matmul_facts: return`); heuristic gate
  `triton.py:301-305` (`not spec.matmul_facts`).
- **Register/compose**: relax `device_ir.py:1094` so the existing `materialized_inner`
  branch (1120-1129) registers a ReductionFact for the N rdim via `_assemble_reduction_fact`
  (digests it as-is: full_width_output=True, num_carried_2d_tiles=1, input_load_itemsize=0
  fallback); keep MatmulFact; add a NEW heuristic gating on `matmul_facts==1 and
  reduction_facts==1` (don't reuse `_triton_reduction_eligible`).
- **Seed lever**: footprint-aware `tile_m` (+ secondary `tile_k`, + num_warps), keyed on
  `static_rnumel`(=N) and the [M_BLOCK,N] fp32 accumulator bytes, replacing the N-blind
  [32,32] default. Eligibility = "a productive tile fits under SMEM" (N<=512 roomy, N=1024
  tile_m<=32, N=2048 infeasible).
