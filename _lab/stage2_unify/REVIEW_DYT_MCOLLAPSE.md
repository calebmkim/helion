# Stage-2 review follow-up — faithful M-collapse detection + dyt fix

Review-driven follow-up to the Stage-2 M-reduction work (Sub-problems A/B/C). Two related
changes to the reduction-fact layer + the user-tiled seed. **No new heuristic class, no
MReductionFact** — it stays on the existing standard/user-tiled tracks; it only makes the
M-collapse recognition faithful and gives `dyt` a real seed.

Box: H100 sm90, conda `helion`. Measured fp32-train, G = tc/seed (G > 1 beats torch.compile).

## Problem 1 — the `is_m_collapse` detector was a brittle proxy conjunction

The Stage-2 detector was four AND-ed symptom fields:

```python
is_m_collapse = (not full_width_output and num_carried_2d_tiles == 0
                 and not non_reduction_loop_block_ids and num_load == 1)
```

`num_load == 1` was a curriculum fence: it *excluded* dyt (num_load=3, a genuine grad-param
collapse) purely to separate it from bias_grad, and it would have over-excluded the whole
`grad_weight = Σ_M(grad_out·x̂)` family (num_load≥2). dyt fell back to the floored `[1,1]`
user-tiled seed → **G≈0.66** (loses to tc), vs the ws2 m_reduction prototype's **1.37**.

### Fix — a single faithful structural fact

A grad-parameter collapse is *defined* by accumulating a per-feature gradient buffer across
the rows. Its fingerprint, read from accumulator provenance (the walker's `accumulator_facts`):

> **`per_feature_accumulator`** — a loop-carried accumulator whose dims are ALL the
> *materialized feature axis* (the `[N]` grad buffer, e.g. `grad_bias[N]`/`grad_weight[N]`).

`is_m_collapse = fact.per_feature_accumulator` — one fact, replacing the 4 proxies.

**Materialized feature axis** = a reduction-dim block (`bs.reduction` provenance) left
full-width: in NEITHER `block_sizes` (not user-tiled) NOR `reduction_loops` (not rolled) NOR
grid, static extent. For a grad-param collapse this is the `[N]` output width.

Verified the signal cleanly classifies every user-tiled kernel (compile-only factdump):

| kernel | accumulator dims | per_feature_accumulator | is_m_collapse |
|---|---|---|---|
| bias_grad | `(FEAT)` | True | fires |
| dyt | `(FEAT)`,`(FEAT)`,`(None,FEAT)` | True | fires |
| grpo | `(GRID,GRID)` | False | excluded |
| softmax_two_pass | `(GRID)` | False | excluded |
| kl_div / jsd | `(GRID, bs)` | False | excluded |
| welford | `(GRID)`,`(GRID,None)` | False | excluded |

(rms/ln/instance/group_norm bwd are `per_feature_accumulator=True` too, but ride the **standard**
track — `is_m_collapse` is never evaluated there, seed stays `[1,1]` unchanged.)

## Problem 2 — the M-collapse seed coupled two independent knobs

The seed set `r_block = m_collapse_block`, i.e. both block-size entries = the occupancy block.
That spilled dyt's grad_x-laden body: `[128,128]` → **0.23x**.

### Fix — decouple grid (occupancy) from inner (footprint), branch on `body_live_tiles`

- **grid CTA** = occupancy (`next_pow2(grid_rows // num_sm)`, cap 256) — unchanged.
- **inner reduction tile**:
  - `body_live_tiles <= 1` (pure collapse, bias_grad: read+sum, one resident tile) → reduce
    the whole CTA wave (`r_block = m_collapse_block`). A big inner tile is cheap here.
  - `body_live_tiles > 1` (per-row work, dyt: grad_x store + tanh intermediates) → byte-cap the
    resident `[inner, N]` tile to `M_COLLAPSE_TILE_BYTES = 32768` (~2–8 rows) for occupancy.
    `inner = next_pow2(32768 // (feature_extent * itemsize))` — mirrors the ws2 m_reduction byte cap.

New fact `feature_extent` (materialized feature width N) feeds the byte cap.

## Results (fp32 train, G = tc/seed)

Head-to-head (cold-L2 do_bench, accuracy-gated; only block_sizes varied):

| dyt config | (16384,1024) | note |
|---|---|---|
| `[1,1]` (old seed) | 0.62 | floored — the laggard |
| `[128,128]` (old occupancy) | 0.23 | grad_x spills |
| `[128,8]` (**new seed**) | **1.44** | byte-capped inner |
| `[128,2]` | 0.97 | over-tight |

`dyt` actual new seed across shapes: `[128,8]`=1.44, `[64,2]`=1.55, `[256,8]`=1.50, `[32,1]`=0.98
→ **geo ≈ 1.35** (was 0.66; beats tc and the old m_reduction 1.37). `bias_grad` **byte-identical**
to before (body_live=1 → occupancy `[128,128]`, geo ~0.94).

## Invariant (verified at the seed level; full config_recorder gate still pending)

- 9 standard: the user-tiled members (softmax/kl_div/jsd/welford) are `per_feature_accumulator=False`
  → bypass the M-collapse branch → seeds unchanged; the rolled members never evaluate it. The new
  fact fields are inert for all of them (the standard seed doesn't read them).
- 8 transfer: grpo `per_feature_accumulator=False` → unchanged; the rest are standard-track.
- The only changed emitted configs: **dyt** (`[1,1]`→`[128,8]`/`[64,2]`), an M-reduction target
  outside both protected sets. bias_grad/rms/ln/instance/group_norm unchanged.

**TODO before banking:** run `config_recorder` over the full 739-cell matrix (× dtypes × robustness)
to gate "9 standard byte-identical, 8 transfer unchanged" on HEAD.

## Files
- `helion/autotuner/config_spec.py` — `ReductionFact.feature_extent`, `.per_feature_accumulator`.
- `helion/_compiler/device_ir.py` — derive both in `_assemble_reduction_fact`; materialized
  apply-loop widening + warn in `register_user_tiled_reductions` (was forced `()`).
- `helion/_compiler/autotuner_heuristics/triton.py` — `is_m_collapse = fact.per_feature_accumulator`;
  `M_COLLAPSE_TILE_BYTES`; decoupled grid/inner seed.
