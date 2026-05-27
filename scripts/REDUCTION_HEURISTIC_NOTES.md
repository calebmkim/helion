# Triton reduction heuristic — work log

Persistent record of the hill-climbing work on the H100/sm90 INNER reduction
heuristic. Pick up here if you're continuing the iteration.

## Where things live

- Heuristic: `helion/_compiler/autotuner_heuristics/triton_reduction.py`
- Fact extraction: `helion/_compiler/device_ir.py` (square-shape `static_xnumel`
  fix in `_collect_reduction_facts`).
- Bench driver: `scripts/bench_reduction_heuristic.py` — supports all v1
  kernels via `--kernel {sum,rms_norm,layer_norm,softmax,softmax_decomposed,cross_entropy,longsum,all}`
  and `--validation` / `--with-torch-compile`.
- Tests: `test/test_reduction_heuristic.py` — pinned recipe values; update
  these in lockstep with the heuristic.

## Recipe v7 — current state

The heuristic classifies each `ReductionFact` into one of three classes and
applies a per-class persistent-vs-looped boundary plus per-class warp/xblock
formulas. Per `CLAUDE.md` §7.5, NO per-shape special-casing.

### Kernel classes (`_kernel_class`)

| Class | Predicate | Examples |
|---|---|---|
| `sum` | `num_reduction <= 1` | sum (1L+1R), rms_norm (2L+1R), longsum |
| `norm` | `num_reduction >= 2 AND num_load <= 1` | softmax, softmax_decomposed (1L+2R) |
| `multi-load` | `num_reduction >= 2 AND num_load >= 2` | layer_norm (3L+2R), cross_entropy (3L+2R) |

### Persistent boundary (`_persistent_r_max`)

| Class | persistent up to R = |
|---|---|
| sum | 4096 |
| norm | 16384 |
| multi-load | 32768 if X<=16384, else 8192 |

### Persistent recipe — xblock (`_xblock_for_persistent`)

xblock>1 only at very small R. The cap formula `tile_target/R`:

| Class | R<=256 | R<=512 | R>512 |
|---|---|---|---|
| sum | `min(16, 2048//R)` | 1 | 1 |
| norm | `min(16, 4096//R)` | 4 | 1 |
| multi-load | 4 | 4 | 1 |

### Persistent recipe — num_warps (`_num_warps_persistent`)

| Class | Formula |
|---|---|
| sum (R<=256) | nw=1 |
| sum (tile<=2048) | nw=4 |
| sum (tile>2048) | nw=8 |
| norm/multi | `next_pow2_clamped(max(4, tile//1024), 4, min(nw_max, 16))` |

### Looped recipe (`_recipe_inner_looped`)

Used when `R > _persistent_r_max(fact)`. Same for all classes:

- `r0_block = min(prev_pow2(R), MAX_R0_BLOCK=4096)`, with backoff (`//= 2`) if
  it would equal `r_size_hint` (otherwise the autotuner re-decodes as
  persistent — see `block_id_sequence.py:222-232`).
- `xblock = 2 if R<=2048 else 1`.
- `num_warps = next_pow2_clamped(r0_block // 128, 4, nw_max)`.

`MAX_R0_BLOCK` drops to 2048 and `nw_max` halves under register pressure
(`x>=1024 AND num_load+num_reduction >= 10`), per Inductor's
`triton_heuristics.py:3834-3856`.

## Latest scoreboard (v7, training shapes)

| Kernel | geomean(default) | geomean(tc) | worst(default) | worst(tc) |
|---|---|---|---|---|
| sum | 1.006 | 1.239 | 0.984 | 0.933 |
| rms_norm | 1.060 | 1.515 | 0.943 | 0.748 |
| layer_norm | 1.071 | 1.764 | 0.977 | 0.981 |
| softmax | 1.174 | 1.653 | 0.988 | 1.128 |
| softmax_decomposed | 1.177 | 1.590 | 0.999 | 1.051 |
| cross_entropy | 1.509 | 1.377 | 1.005 | 0.952 |
| longsum | 3.220 | 5.114 | 1.712 | 3.588 |

Validation (4-5 held-out shapes per kernel) tracks the same picture; worst
default speedup 0.957x (layer_norm 2048×1023, where heuristic and default emit
identical configs — pure bench-harness noise).

Bench artifacts: `/tmp/bench_v3_full.txt` (pre-hill-climb baseline),
`/tmp/bench_v7_full.txt` (current), `/tmp/bench_v7_val.txt` (validation).

## What was tried, in order

1. **v3 baseline** — three classes, sum-class persistent up to R=2048, num_warps
   ramped `tile/512 ∈ [4, 16]`. Worked except: layer_norm(8192,5120) at 0.911
   speedup vs default (over-warped to nw=16), softmax(4096,4096) at 0.972
   (over-warped to nw=8), rms_norm(2048,4096) on the persistent/looped seam.
2. **v4** — bumped sum-class persistent boundary 2048→4096 (rms_norm wants
   persistent at R=4096), switched norm/multi-load to `tile/1024 ∈ [4, 16]`.
   Fixed the layer_norm regression but rms_norm(32768,256) regressed because
   I had restricted xb>1 in sum-class to `num_load<=1`.
3. **v5** — removed the `num_load<=1` restriction; xb>1 in sum-class is a
   small-R scheduling win regardless of load count (verified head-to-head:
   xb=8,nw=4 wins by ~10% over xb=1,nw=4 at R=256).
4. **v6→v7** — refined the sum-class warp threshold (`tile<=1024` then
   `tile<=2048` for nw=4 floor) based on h2h sweeps showing nw=4 ties or wins
   over nw=8 at tile=1536-2048.

## Sweep scripts

In `/tmp/`:
- `sweep_focused.py` — sum/norm/multi-load nw sweeps at the gap shapes.
- `sweep_more.py` — softmax and ce persistent vs looped boundary.
- `sweep_small_ml.py` — layer_norm at small R.
- `sweep_smalltile.py` — small-tile xblock packing.
- `sweep_rmsnorm_small.py` — rms_norm at R=256.
- `sweep_sumlarge.py`, `sweep_sum.py`, `sweep_ce*.py` — class-specific.
- `h2h_*.py` — multi-trial head-to-head benches; trust these over single
  do_bench runs (which have ~5-10% jitter on small kernels — see auto-memory
  `reduction_heuristic_bench_noise.md`).
- `probe_*.py` — print what the heuristic emits per (kernel, shape).

## Where to push next

Remaining gaps that aren't pure noise:

1. **rms_norm worst tc_ratio = 0.748** (and similar at small shapes). The
   heuristic and `default_config()` emit identical configs there, so any gain
   would have to come from a fundamentally different recipe shape (e.g.
   different indexing mode, different num_stages). Not addressable through
   the current recipe knobs.
2. **cross_entropy(4096,4096) tc_ratio = 0.952**. Sweep showed best
   persistent at nw=4 (51us) vs torch.compile at 46us. The 5us gap is
   structural — Inductor fuses softmax+CE in a way Helion doesn't yet. Could
   be worth re-examining whether multi-load at low R should go looped.
3. **sum tc_ratio worst = 0.933** at large-N small-X shapes. torch.compile
   uses a different reduction tree shape; Helion's split is bandwidth-
   competitive but slightly higher launch overhead. A `sum`-only specialized
   recipe wouldn't help much since default already converges to similar
   configs.

## Discipline reminders

- Single-process head-to-head only for noise <2% (see auto-memory).
- Do NOT special-case shapes (CLAUDE.md §7.5) — adjust class predicates or
  formulas instead.
- When updating the heuristic, update `test/test_reduction_heuristic.py` in
  the same commit. Tests pin recipe values, not behavioral invariants, so
  they will fail for any rule change.
- `device_ir.py` fact extraction has a workaround for square shapes (M==N)
  where `block_id_for_dim` would otherwise pick the reduction block. Don't
  remove that without understanding why every M==N reduction returned
  `static_xnumel=None` before.
