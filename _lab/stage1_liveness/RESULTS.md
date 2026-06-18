# Stage 1 — RESULTS / evidence for the gate stack

**Immutable commit (frozen champion):** `e99ade11` (Part B). Part A: `48ac703e`. Base: `0676dd32`.
Box: H100 sm90, L2=50MB. Interpreter conda `helion`. PYTHONPATH=worktree.

## The change (two commits)
- **Part A (`48ac703e`)** — refactor `_reduction_rblock` to return `(r_block, persistent)` from one
  budgeted decision; adds `footprint_factor` (default 1) + `live_budget` (default
  ROW_PERSIST_MAX_BYTES) params. **Behavior-preserving: config_recorder full active matrix
  (9 kernels × fp32/bf16/fp16 × train/val/robustness = 739 cells) → 0 changed.**
- **Part B (`e99ade11`)** — liveness signal:
  - Walker fact `body_live_tiles` = peak simultaneously-live rdim-shaped tiles per reduction axis,
    computed in the SINGLE collect pass (`_collect_memory_op_facts` → `_graph_peak_live_by_axis`,
    consumer-agnostic per-axis, max over graphs). Derived `ReductionFact.body_live_tiles` reads its
    axis's slice (default 1). NO derived-fact graph walk.
  - Standard track: `footprint_factor = fact.body_live_tiles`, `live_budget = LIVE_PERSIST_BUDGET =
    3 × ROW_PERSIST_MAX_BYTES = 737280`. `_reduction_rblock` persist test = base single-tile register
    caps (UNCHANGED) AND `body_live_tiles × m × sh × itemsize ≤ live_budget` (the multi-tile spill
    ceiling — only REMOVES persistence from a heavy body). Looped chunk shrinks by body_live_tiles.
  - User-tiled: defaults (footprint_factor=1, live_budget=ROW_PERSIST_MAX_BYTES → liveness term is a
    no-op) + the existing `_bandb_r_block_cap`. Byte-identical.

## body_live_tiles per kernel (graph-structure constant, shape-independent — probe_live_tiles.py)
flj=7, rms_norm=3, layer_norm=3, softmax=2, sum=2, long_sum=2, cross_entropy=2, welford=3, kl_div=6,
jsd=12; transfer: fused_add_{rms,layer}norm=4, gated_rmsnorm=4, scaled_masked_softmax=3,
cross_entropy_ls_zloss=3, dynamic_quant=3, grpo=2.

## Gate F — MECHANISM (measured n_spills, byte-identical result tile 201028 B)
flj (4096,50257) fp32, ONLY reduction_loops differs:
| reduction_loops | n_regs | n_spills | G vs grad-fair tc |
|---|---|---|---|
| [None] (base seed) | 64 | **480** | 0.499 |
| [8192] (Part-B seed) | 64 | **0** | 1.224 |
Persistent holds 7 live [1,V] fp32 tiles → register-file overflow (480 spills) → the disaster; the
liveness ceiling routes it to looped[8192] → 0 spills → win. Matches the task's reference (538→2).

## Gate F/A — flj headline (PRIMARY): base persistent [None] → Part-B looped [8192], grad-fair tc
(ab_flj.py, single-process median-of-9, tc baseline computes loss+grad = arm-fair)
| shape | dtype | [None] (BEFORE) | [8192] (AFTER seed) |
|---|---|---|---|
| (4096,32000) | bf16 | 0.577 | **1.324** |
| (4096,50257) | bf16 | 0.259 | **1.017** |
| (8192,32000) | bf16 | 0.564 | **1.341** |
| (4096,32000) | fp32 | 0.906 | **2.358** |
| (4096,50257) | fp32 | 0.499 | **1.224** |
| (2048,128256) | bf16 | 0.159 | 1.268 (was [16384] base→[8192], both ~1.26) |

## Gate R — config-recorder BEFORE/AFTER (the do-not-regress sweep)
- Part A→Part B over the FULL active matrix (739 cells): **0 changed** (config + `--triton` hash on a
  rms_norm/cross_entropy/welford/kl_div/softmax subset → every config-identical cell Triton-identical
  ⇒ selection-only, skip sound). The 9 standard reduction kernels are byte-identical.
- Transfer flips base→Part-B (the 8 off-curriculum kernels): **only fused_linear_jsd** (6 cells, all
  improvements above). cross_entropy_ls_zloss / fused_add_* / gated_rmsnorm / scaled_masked_softmax /
  dynamic_quant / grpo: UNCHANGED.
- Earlier WRONG design (footprint × scaled cap) flipped cross_entropy_ls_zloss V≈50k and REGRESSED it
  (persist 1.6× → loop 0.99×); the additive-ceiling design (this one) keeps it persistent. Verified by
  ab_transfer_flip.py: cross_entropy_ls_zloss persist G 1.36–1.64 (kept).

## Gate D — faithfulness notes
- `body_live_tiles` is a WALKER fact field (computed in the single collect pass, not a derived-fact
  walk). Consumer-agnostic per-axis (keyed by block_id, like `reductions_fed`); the derived
  ReductionFact reads its own axis's slice. Conservative over-count (counts all rdim-shaped live
  values, no rematerialization modeling) → errs toward looping (safe), declared conservative.
- The threshold `live × bytes ≤ 3 × ROW_PERSIST_MAX_BYTES` is a BYTE budget (itemsize is a FACTOR
  inside it, never a literal dtype comparison) vs a hardware fast-memory budget — faithful, not a
  dtype/identity fence. 3× = register (~256KB) + SMEM (~228KB) per-program fast memory; cross-checked:
  cross_entropy_ls_zloss's 3-tile/603KB body stays fast persistent, flj's 7-tile/0.9–1.4MB spills.
