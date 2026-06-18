# Stage 1 — liveness-aware chunk decision (NOTEBOOK / source of truth)

Base SHA: `0676dd32` (reduction-seed-heuristic = merged PR #2762). Branch `reduction-3stage-stack`,
worktree `/home/calebkim/helion-new-heuristics/helion-3stage`. GPU: 4×H100 (sm90), L2=50MB.

## Target (Task 1)
PRIMARY: `fused_linear_jsd` (examples/fused_linear_jsd.py, STANDARD track) narrow-V flips
persistent `[None]` → looped, AND the gate stack (D/R/H/A/F) passes. Standard track has NO liveness
term today → spills on heavy bodies.

## Established diagnosis (REPRODUCED on this box, grad-fair baseline, _lab/stage1_liveness/ab_flj.py)
flj persistent is a disaster; looping fixes it (G = fair_tc / seed, loss+grad baseline):
| shape | dtype | persist[None] | best looped (chunk) |
|---|---|---|---|
| (4096,32000) | bf16 | 0.577 | 1.324 (8192) |
| (4096,50257) | bf16 | 0.259 | 1.155 (2048) / 1.017 (8192) |
| (4096,32000) | fp32 | 0.906 | 2.358 (8192) |
| (4096,50257) | fp32 | 0.499 | 1.273 (4096) / 1.224 (8192) |
| (8192,32000) | bf16 | 0.564 | 1.341 (8192) |
chunk 16384 is weaker for wide-V (4096,50257 bf16 → 0.766); chunk 8192 → 1.017.

## The faithful signal: peak simultaneously-live rdim-shaped tiles (MAX_PEAK_LIVE)
`_lab/stage1_liveness/probe_live_tiles.py` — a real liveness sweep over the FX reduction graphs,
per reduction axis, max over graphs. **Shape-independent (graph structure)**, verified identical at
3 flj shapes. Counts: flj=7, rms_norm=3, layer_norm=3, softmax=2, sum=2, long_sum=2,
cross_entropy=2, welford=3, kl_div=6, jsd=12. Cleanly separates flj(7) from the standard-track light
kernels (≤3). Conservative over-count of "peak live" (counts all rdim-shaped values live at the
point; register rematerialization may reduce true pressure — declared conservative, errs to looped).

## The calibration crux (MEASURED, ab_persist_loop.py)
Naive `footprint_factor=peak_live` with the existing cap 245760 WRONGLY flips cross_entropy:
cross_entropy fp32 (8192,50257) persist G=1.092 (beats tc!), loop16384 G=0.802 — a real regression.
The existing cap was implicitly tuned for the curriculum's real liveness (~2). So the persist
budget must be recalibrated when liveness is made explicit.

## Budget window (budget_window.py)
persist iff `peak_live × m × sh × itemsize ≤ PERSIST_BUDGET`.
- keep ALL currently-persistent standard cells → PERSIST_BUDGET ≥ 458752 (binding: cross_entropy
  bf16 (4096,114688)).
- flip ALL flj narrow-V → PERSIST_BUDGET < 896000 (binding: flj(4096,32000)).
- no currently-LOOPED cell flips UP → PERSIST_BUDGET < 491522 (= 2×245760+).
=> safe window **[458752, 491520]**. Choose **PERSIST_BUDGET = 2 × ROW_PERSIST_MAX_BYTES = 491520**
   (top of window, principled = H100 per-program fast memory ≈ registers 256KB + SMEM 228KB ≈ 484KB;
   NOT a curriculum fence — it's a factor of 2). Curriculum stays byte-identical, flj flips.
The looped CHUNK keeps the register budget ROW_PERSIST_MAX_BYTES (hot-loop must stay register-
resident): chunk = min(LOOPED_CHUNK, prev_pow2(ROW_PERSIST_MAX_BYTES // (m×itemsize×peak_live))).
flj (peak=7, itemsize=4): 245760//28 = 8777 → 8192 (its best chunk). cross_entropy stays persistent.

## DESIGN (final)
- **Part A (refactor, zero-diff):** `_reduction_rblock(env, fact, m_block, footprint_factor=1,
  persist_scale=1) -> (r_block, persistent)`. Decide persist + chunk in ONE place; derive persistent
  from the result. At defaults (1,1) byte-identical (verified by config_recorder zero-diff).
- **Part B (liveness):**
  - Walker fact: peak-live per reduction axis, folded into the single collect pass
    (`_collect_memory_op_facts`) — consumer-agnostic per-axis (like `reductions_fed`).
  - Derived: `ReductionFact.body_live_tiles` reads its axis's slice (default 1).
  - Standard track: `footprint_factor = fact.body_live_tiles`, `persist_scale = LIVE_PERSIST_SCALE = 2`.
  - User-tiled track: KEEP ff=1 + `_bandb_r_block_cap` (sanctioned fallback; folding Band-B into
    footprint_factor needs a different byte budget 16KB vs 240KB — logged tradeoff). → byte-identical.
- New constant `LIVE_PERSIST_SCALE = 2` (register+SMEM story). Gate D: byte budget (itemsize a FACTOR,
  not a literal). Gate H: measured-crossover form, faithful key (peak_live × bytes ≤ HW budget).

## STATUS: design done; implementing Part A next.
