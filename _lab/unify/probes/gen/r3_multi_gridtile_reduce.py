"""CELL r3_multi_gridtile_reduce — ADVERSARIAL re-probe of the grid-tile-reduction fix
(bfa012cc): TWO reductions each OVER a DIFFERENT tunable grid axis in a 2-D grid where BOTH
axes are reduced over.

Background (the fix this attacks)
---------------------------------
``_user_tiled_reduction_block_ids`` now gates secondary-reduction collection on
``_classify_reduction_axis(...) is RESIDENT``, so a reduction whose axis is ALSO a tunable
grid (program) axis is classified ``PARTIAL_GRID`` (``cdiv > 1``) and EXCLUDED from the sized
``red_values`` map -- it floors as a grid ROW instead of being sized full-extent (which would
COLLAPSE the grid). The reference RED that motivated the fix (``adv_gro_falsify``) had a SINGLE
1-D grid axis reduced via ``x[cta, :].sum(0)``.

This cell goes one level harder: a 2-D grid ``for ti, tj in hl.tile([I, J])`` where BOTH ``ti``
and ``tj`` are tunable grid axes AND BOTH are reduced over (the body collapses the [I,J] CTA
tile over BOTH grid dims into a per-feature [N] partial, finalized cross-CTA). Two PARTIAL_GRID
axes at once. The adversarial question:

  * Does EACH grid axis correctly classify PARTIAL_GRID and floor (stay a grid row), or does the
    seed widen ONE of them toward its full extent -> collapse that grid dimension?
  * Does the SECONDARY grid axis (the non-dominant grid dim) leak into ``red_values`` and get
    sized full-extent (the exact hole the fix closed, but now via the *secondary* of a 2-D grid)?

The ``adv_gro_falsify`` finding flagged that on the BACKED-extent grid-collapse the axis takes
the FOOTPRINT-WIDEN branch (``min(own, fits, occ, _m_block_cap)``) and a degenerate ``itemsize``
on a cross-CTA-finalize-over-intermediate reduction can disable the byte cap, letting ``own``
(= ``next_pow2(extent)``) win -> grid collapse. With TWO grid axes the occupancy guard
(``_m_axis_occupancy_cap``) sees a 2-D grid product; this probes whether the per-axis widen on
EITHER axis can still over-collapse.

Property point (§1)
-------------------
  ACCESS         = user-tiled (manual 2-D grid tile, both dims reduced)
  ORIGIN         = grid (both reductions are OVER grid/program axes -> PARTIAL_GRID x2)
  EXTENT         = static (I, J, N all real sizes; BACKED grid tiles)
  CARRIED-RESIDENT = per-feature [N] accumulator finalized cross-CTA (no [M_BLOCK,R_BLOCK] carry)
  CO-RESIDENCY   = TWO reductions in the SAME loop body, over TWO different grid axes
  DIMS           = 3 (input [I, J, N]); 2-D grid (I, J)
  PINNED-GRID    = neither pinned; both I and J are TUNABLE grid axes

OUT-OF-SCOPE guards deliberately avoided
----------------------------------------
  * JAGGED -- I, J, N are all real static sizes; no None / data-dependent extent.
  * STRIDED-DIM0 -- ``x`` is a contiguous [I, J, N] tensor; the reduced elements ``x[ti, tj, :]``
    are loaded as full contiguous N-rows (innermost stride == itemsize). The reduction is over
    the OUTER grid dims (I, J), but the per-element access within each summed row is contiguous
    (the ordinary 2-D batch-collapse access, same family as a [B,M,N] mean). This is NOT the
    strided-dim0 predicate (a 1-D reduction axis whose memory stride over the reduced elements
    != itemsize) -- the reduced rows are dense contiguous slabs, not scattered gathers.

Run
---
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-unify /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-unify/_lab/unify/probes/gen/r3_multi_gridtile_reduce.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/dev/local/helion-unify/_lab/unify/probes")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import check_kernel  # noqa: E402

_WT = "/home/dev/local/helion-unify"
assert os.path.abspath(helion.__file__).startswith(_WT + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT})"
)

torch.manual_seed(0)
DEV = "cuda"
F32 = torch.float32


# ARM A: 2-D grid, BOTH axes reduced over into a per-feature [N] partial, finalized cross-CTA.
# ``for ti, tj in hl.tile([I, J])`` -> ti and tj are BOTH tunable grid (program) axes. The body
# sums the static [ti, tj, N] CTA tile over BOTH grid dims (0, 1) into a [N] partial stored at
# the flat CTA id; the host finalizes across CTAs with a final sum(0). Both grid axes are
# reductions -> each must classify PARTIAL_GRID and FLOOR (stay a grid row), NOT be sized
# full-extent (which would collapse that grid dimension).
@helion.kernel(static_shapes=False)
def multi_gridtile_reduce(x: torch.Tensor) -> torch.Tensor:
    I, J, N = x.shape
    bi = hl.register_block_size(I)
    bj = hl.register_block_size(J)
    nbi = (I + bi - 1) // bi
    nbj = (J + bj - 1) // bj
    blocks = x.new_zeros([nbi, nbj, N], dtype=torch.float32)
    for ti, tj in hl.tile([I, J], block_size=[bi, bj]):
        # Reduce the [bi, bj, N] CTA tile over BOTH grid dims into the per-feature [N] partial,
        # via TWO sequential single-dim reductions (Helion lowers each as its own
        # ReductionLowering over a distinct grid block_id). First collapse the J grid dim
        # (dim 1), then the I grid dim (dim 0) -> both ti and tj are reduced-over grid axes.
        slab = x[ti, tj, :].to(torch.float32)
        over_j = torch.sum(slab, dim=1)  # reduce the tj grid axis (dim 1) -> [bi, N]
        over_ij = torch.sum(over_j, dim=0)  # reduce the ti grid axis (dim 0) -> [N]
        blocks[ti.id, tj.id, :] = over_ij
    return torch.sum(torch.sum(blocks, dim=1), dim=0)


# ARM B: a grid-tile reduction COMBINED with a normal inner (non-grid) reduction. The grid axis
# ``ti`` is reduced over (PARTIAL_GRID) AND there is a separate inner full-slice reduction over
# the feature axis N that is NOT a grid axis (RESIDENT). Output is a per-(grid-tile, scalar)
# collapse: for each CTA tile we both (a) sum the grid-rows into a per-feature partial AND
# (b) emit a scalar = mean over N of that partial. Mixes a PARTIAL_GRID grid-reduction with a
# RESIDENT inner reduction in the same kernel -> tests that the inner RESIDENT reduction is
# sized as a reduction while the grid axis still floors.
@helion.kernel(static_shapes=False)
def gridtile_plus_inner_reduce(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    bm = hl.register_block_size(M)
    nbm = (M + bm - 1) // bm
    out = x.new_zeros([nbm], dtype=torch.float32)
    for tm in hl.tile(M, block_size=bm):
        # Inner full-slice reduction over N (RESIDENT, not a grid axis) AFTER collapsing the
        # grid-row dim 0 (PARTIAL_GRID). Per-feature partial p[N] = sum over grid rows; then a
        # second reduction (mean over N) -> scalar per CTA tile.
        per_feat = torch.sum(x[tm, :].to(torch.float32), dim=0)  # PARTIAL_GRID over tm (dim 0)
        out[tm.id] = torch.mean(per_feat)  # RESIDENT inner reduction over N
    return out


def _dump(tag: str, v: dict) -> None:
    obs = v["observed"]
    ns = obs.get("normalized_cfg", {}) or {}
    red = v["red"] or "green"
    print(f"\n=== [{red}] {tag} ===")
    print(f"  fired                   = {obs.get('fired')}")
    print(f"  n_reduction_facts       = {obs.get('n_reduction_facts')}")
    print(f"  n_matmul_facts          = {obs.get('n_matmul_facts')}")
    print(f"  lowering_reduction_axes = {obs.get('lowering_reduction_axes')}")
    print(f"  grid_block_ids          = {obs.get('grid_block_ids')}")
    print(f"  block_sizes_valid_ids   = {obs.get('block_sizes_valid_ids')}")
    print(f"  reduction_loops_valid   = {obs.get('reduction_loops_valid_ids')}")
    print(f"  fact                    = {obs.get('fact')}")
    print(f"  raw_seed                = {obs.get('raw_seed')}")
    print(f"  normalized block_sizes  = {ns.get('block_sizes')}")
    print(f"  normalized reduction_loops = {ns.get('reduction_loops')}")
    print(f"  reasons                 = {v['reasons']}")


def main() -> dict:
    print(f"helion={helion.__file__}\n")
    # ARM A: 2-D grid, both axes reduced. I, J both grid axes; N the per-feature width.
    xa = torch.randn(4096, 256, 1024, device=DEV, dtype=F32)
    intended_a = {
        "cell": "r3_multi_gridtile_reduce",
        "arm": "A: 2-D grid, BOTH axes reduced (two PARTIAL_GRID)",
        "access": "user-tiled",
        "origin": "grid (both axes)",
        "extent": "static (backed)",
        "co_residency": "two grid reductions, same loop",
        "dims": 3,
        "expect": "BOTH grid axes floor as PARTIAL_GRID; neither sized full-extent",
        "hole_if": "either grid axis seeds to full extent -> grid dim collapse",
    }
    va = check_kernel("r3_multi_gridtile_reduce__armA_2d_grid",
                      multi_gridtile_reduce, (xa.clone(),), intended_a)
    _dump("ARM A: 2-D grid both-axes reduced", va)

    # ARM B: grid-tile reduction COMBINED with a normal inner reduction.
    xb = torch.randn(4096, 2048, device=DEV, dtype=F32)
    intended_b = {
        "cell": "r3_multi_gridtile_reduce",
        "arm": "B: grid-tile reduction + inner RESIDENT reduction",
        "access": "user-tiled",
        "origin": "grid (dim0) + inner (N)",
        "extent": "static",
        "expect": "grid axis floors PARTIAL_GRID; inner N reduction sized RESIDENT",
        "hole_if": "grid axis sized full-extent, or inner reduction floored to 1",
    }
    vb = check_kernel("r3_multi_gridtile_reduce__armB_grid_plus_inner",
                      gridtile_plus_inner_reduce, (xb.clone(),), intended_b)
    _dump("ARM B: grid reduction + inner reduction", vb)

    return {"armA": va, "armB": vb}


if __name__ == "__main__":
    main()
