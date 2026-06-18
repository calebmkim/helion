# STAGE 1 REPORT — liveness-aware reduction-chunk decision

**Base SHA:** `0676dd32` (reduction-seed-heuristic = merged PR #2762; origin/main lacked the merge,
so this is the driver-sanctioned base). **Commits:** Part A `48ac703e`, Part B `e99ade11`
(branch `reduction-3stage-stack`). **Box:** H100 sm90, L2 50MB, conda `helion`.

## The fix (two commits)

### Part A — `48ac703e` (refactor, behavior-preserving)
`_reduction_rblock` now decides the reduction chunk AND the persistent verdict in ONE budgeted
formula, returning `(r_block, persistent)` instead of an int the caller re-derives. Adds two params:
`footprint_factor` (resident live-tile count, default 1) and `live_budget` (default
`ROW_PERSIST_MAX_BYTES`). At the defaults the liveness term is implied by the base byte cap, so the
decision is byte-for-byte identical. **Proof:** config_recorder over the FULL active matrix (9 kernels
× fp32/bf16/fp16 × train/val/robustness = 739 cells) → **0 changed** (`cfg_base.json` → `cfg_partA.json`).

### Part B — `e99ade11` (the liveness signal)
- **New walker fact field `body_live_tiles`** (helion/autotuner/config_spec.py): peak number of
  simultaneously-live rdim-shaped (full-width) tensor tiles in the reduction body. Populated in the
  SINGLE collect pass — `_collect_memory_op_facts` calls `_graph_peak_live_by_axis` (an FX
  def→last-use liveness sweep, per-graph, max over graphs), returning a consumer-agnostic per-axis
  dict (keyed by block_id, like `reductions_fed`). The derived `ReductionFact.body_live_tiles` reads
  its own axis's slice; NO derived-fact graph walk. Conservative over-count (errs toward looping).
- **Standard track** passes `footprint_factor=body_live_tiles` and
  `live_budget=LIVE_PERSIST_BUDGET = 3 × ROW_PERSIST_MAX_BYTES = 737280`. The persist test keeps the
  unchanged single-tile register caps AND adds `body_live_tiles × m × size_hint × itemsize ≤
  live_budget` — the multi-tile spill ceiling, which only REMOVES persistence from a heavy body. The
  looped chunk shrinks by `body_live_tiles` (jsd → 8192).
- **User-tiled track**: defaults (footprint_factor=1, no-op liveness term) + the existing
  `_bandb_r_block_cap`. Byte-identical. (Unification of Band-B into footprint_factor was NOT done —
  it needs a different byte budget (16KB vs 240KB); the sanctioned standard-track-only fallback was
  taken and logged.)

`body_live_tiles` per kernel (graph-structure constant): flj=7, rms/layer_norm=3, softmax=2, sum=2,
long_sum=2, cross_entropy=2, welford=3, kl_div=6, jsd=12; transfer fused_add_*/gated=4, scaled_masked
=3, cross_entropy_ls_zloss=3, dynamic_quant=3, grpo=2.

## Perf — PRIMARY target fused_linear_jsd (standard track), grad-fair tc (loss+grad), med-of-9
BEFORE = base persistent `[None]`; AFTER = Part-B seed `[8192]`. Independent repro (gate_A_repro.py):
| shape | dtype | BEFORE G | AFTER G | after/before |
|---|---|---|---|---|
| (4096,32000) | bf16 | 0.54 | **1.23** | 2.27× |
| (8192,32000) | bf16 | 0.50 | **1.25** | 2.48× |
| (4096,50257) | bf16 | 0.24 | 0.80 | 3.36× |
| (4096,32000) | fp32 | 0.76 | **1.76** | 2.32× |
| (4096,50257) | fp32 | 0.47 | **1.17** | 2.49× |
Every narrow-V shape flips persistent→looped and improves **2.3–3.4×** over the base seed. Beats
grad-fair tc on 4/5 (G 1.17–1.76); flj bf16 V=50257 lands ~0.8–1.0 (run-dependent, near tc-parity,
above the 0.75 floor) — the secondary wide-V tail (bonus, not a loop target; a smaller chunk [2048]
reaches ~1.15 but is not what the liveness formula emits — logged as overtime).

## Mechanism (Gate F) — measured n_spills, byte-identical result tile (201028 B), only reduction_loops differs
flj (4096,50257) fp32: `[None]` → n_regs=64, **n_spills=480**; `[8192]` → n_regs=64, **n_spills=0**.
Verified in the lowered Triton: persistent materializes the whole [1,V] row + ~7 live full-width fp32
intermediates (softmax/log_softmax/KL/grad ×student,teacher) → register overflow; looped iterates an
8192 chunk carrying only scalar accumulators. (Matches the task's reference 538→2.) The flip also sets
`load_eviction_policies` (the looped re-read path) — a recorded coupling, spill-independent; the win
is fully attributable to `reduction_loops`.

## Do-not-regress (Gate R)
- **9 standard reduction kernels byte-identical**: config_recorder Part A→Part B = 0/739 changed
  (config + `--triton` hash on a rms_norm/cross_entropy/welford/kl_div/softmax subset → every
  config-identical cell Triton-identical ⇒ selection-only edit, the skip is sound).
- **8 transfer kernels**: only `fused_linear_jsd` flips (improves). cross_entropy_ls_zloss,
  fused_add_{rms,layer}norm, gated_rmsnorm, scaled_masked_softmax, dynamic_quant, grpo UNCHANGED. All
  8 PASS correctness bf16+fp32.
- A NaÏVE earlier design (footprint × *scaled* cap) wrongly flipped cross_entropy_ls_zloss V≈50k and
  regressed it (persist 1.6× → loop 0.99×). The shipped ADDITIVE-ceiling design keeps it persistent.

## Gate verdicts (fresh-context adversarial agents; verdicts in ledger.json + gate_*_verdict.json)
- **Gate D (Fact-gate): PASS** — doctrine clean (walk in the single collect pass, consumer-agnostic
  per-axis, derived fact does not walk); faithful byte-budget threshold (itemsize a factor, not a
  dtype fence); authored a compile-only divergence kernel (k_sequential live=2 / k_colive live=7 at
  near-equal node counts) proving `body_live_tiles` tracks REAL co-liveness, not a node-count proxy.
- **Gate H (Generality): KEEP** — faithful key, mechanism-matched breadth (applies to all standard
  reductions; budget 737280 is a hardware ceiling near the middle of the measured-safe window
  [458752, 896000), the additive design's upper bound being flj's own footprint — not a curriculum
  fence), measured-crossover form.
- **Gate F (Mechanism): PASS** — register-file overflow → n_spills 480→0, verified in lowered code;
  no inert fields (load_eviction_policies recorded as a coupling).
- **Gate A (Adversarial verify): refuted=false** — arm-fairness confirmed (tc computes loss+grad),
  seed==[8192] confirmed, curriculum byte-identical re-verified, no identity smuggling; independent
  repro run by the driver reproduces the flip + 2.3–3.4× improvement.
- **Gate R (Regression-referee): accept=true** — 0/739 curriculum changed, only flj flips (improving),
  no disaster, no collapse, net progress.

## Deferred / logged
- Band-B unification into footprint_factor (different byte budget) — fallback taken, logged.
- flj bf16 V=50257 residual gap to tc (chunk [2048] would close it) — secondary wide-V tail, overtime.
