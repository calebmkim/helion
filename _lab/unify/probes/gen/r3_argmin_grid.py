"""CELL: r3_argmin_grid — an INDEX reduction (argmin/argmax) OVER A GRID AXIS.

Adversarial re-probe of the grid-tile-reduction exclusion (the recent fix that
classifies a reduction OVER a tunable grid axis as ``ReductionAxisKind.PARTIAL_GRID``
and FLOORS it -- keeps it a grid ROW -- so sizing it full-extent does NOT collapse the
grid). That fix is keyed on the AXIS ROLE (``_classify_reduction_axis``: grid axis with
``cdiv > 1`` => PARTIAL_GRID), agnostic to the reduction OP. This cell asks: does the
exclusion still hold when the reduction is an INDEX reduction (``torch.argmin`` /
``torch.argmax``) rather than a value reduction (sum / amax)?

WHY THIS MIGHT MIS-ROUTE (the hole hypothesis): an index reduction is structurally
distinct from a value reduction -- the reduction tree carries TWO loop-resident values
per lane (the running extremum VALUE + its running INDEX), the combine is the 2-ary
argmin-combine, and the materialized output is int64 (8 bytes), NOT the fp32 accumulator.
The PARTIAL_GRID classifier reads only the axis's ``size`` + grid membership + cdiv, so it
SHOULD route argmin-over-grid identically to amax-over-grid (the jsd ``amax(dim=0)`` idiom).
The adversarial question is whether (a) the index reduction still lands on a
``ReductionLowering`` block_index the classifier sees (so it IS classified PARTIAL_GRID and
floored), or (b) the int64/2-ary lowering routes it onto a DIFFERENT axis / origin so the
grid axis gets sized full-extent (grid collapse) or the kernel NO_FIREs.

STRUCTURE (mirrors jsd's per-feature ``amax(dim=0)`` over the ``tile_bt`` grid axis, but
with ``argmin``): grid over the batch/row axis M (tunable block_size => PARTIAL grid,
cdiv > 1), inner tile over the feature axis N; reduce dim=0 (the GRID tile axis) with
``torch.argmin`` into a per-feature ``[N]`` index, combined across CTAs (cross-CTA atomic
min on the value, index follows). The reduction's ORIGIN is a grid axis (M); its whole-axis
extent is a per-grid count, NOT a per-program reduction size.

Property point (§1):
  ACCESS         = grid-tile reduction (reduce over the grid program axis M, dim=0)
  ORIGIN         = grid (PARTIAL_GRID role: M is a tunable grid axis, cdiv > 1)
  EXTENT         = static (M = 4096), but a PER-GRID count, not a per-program size
  CARRIED-RESIDENT = per-feature [N] running value+index (the argmin 2-ary carry)
  CO-RESIDENCY   = single index reduction
  DIMS           = 2D
  PINNED-GRID    = none (M is the WIDE tunable grid axis being reduced)
  NOT-modeled axis varied = reduction-OP identity (argmin: index, int64, combine-arity-2)
                            COMPOSED WITH origin=grid (the exclusion's target role).

In-scope (NOT one of the two CLOSED out-of-scope predicates):
  - NOT JAGGED: M is statically 4096 (size is a real int, not None).
  - NOT STRIDED-DIM0: the reduced grid axis M is dim-0, but its memory stride over the
    reduced elements is N*itemsize and we load a CONTIGUOUS [M_tile, N_tile] sub-block;
    the per-feature partials are written contiguously. We assert this is the jsd-style
    contiguous grid reduction, not a strided dim-0 gather. (See note at bottom.)
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROBES = os.path.abspath(os.path.join(_THIS, ".."))
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import check_kernel  # noqa: E402

torch.manual_seed(0)
DEV = "cuda"
F32 = torch.float32


# argmin OVER the GRID axis M: grid over the batch/row axis (tunable block_size => PARTIAL
# grid, cdiv > 1), inner tile over the feature axis N, reduce dim=0 (the GRID tile axis) with
# torch.argmin into a per-feature [N] index. Mirrors jsd's amax(dim=0) over tile_bt, but the
# OP is an INDEX reduction (int64 out, 2-ary combine). This is the grid-tile-reduction the
# PARTIAL_GRID exclusion must FLOOR (keep as a grid row) -- sizing M full-extent collapses
# the grid.
@helion.kernel(static_shapes=False)
def col_argmin_grid(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    # Per-feature partials, one row per grid CTA over M. Grid the batch axis M (tunable
    # block_size => PARTIAL grid, cdiv > 1) AND reduce dim=0 (the GRID tile axis) with
    # torch.argmin within each CTA. The block-local index is offset to a global index so the
    # cross-CTA finalize (argmin over the per-block partials) yields the true global argmin.
    # This mirrors jsd's amax(dim=0) over the tile_bt grid axis, but the OP is an INDEX
    # reduction (int64 out, 2-ary combine): the grid-tile-reduction the PARTIAL_GRID exclusion
    # must FLOOR (keep M a grid row) -- sizing M full-extent collapses the grid.
    m_block = hl.register_block_size(M)
    nb = (M + m_block - 1) // m_block
    blk_idx = torch.empty([nb, N], dtype=torch.int64, device=x.device)
    blk_val = x.new_empty([nb, N], dtype=torch.float32)
    for tile_m, tile_n in hl.tile([M, N], block_size=[m_block, None]):
        sub = x[tile_m, tile_n].to(torch.float32)
        # Reduce the GRID tile axis M (dim=0) -> per-feature block-local index + value.
        local = torch.argmin(sub, dim=0)
        blk_idx[tile_m.id, tile_n] = local + tile_m.begin
        blk_val[tile_m.id, tile_n] = torch.amin(sub, dim=0)
    # Cross-CTA finalize: pick, per feature, the global index whose block produced the min.
    winner = torch.argmin(blk_val, dim=0)
    return torch.gather(blk_idx, 0, winner.unsqueeze(0)).squeeze(0)


def main() -> None:
    print(f"helion={helion.__file__}\n")
    M = 4096  # WIDE tunable grid/batch axis being reduced (PARTIAL grid: cdiv > 1)
    N = 4096  # feature axis (kept in the output)
    x = torch.randn(M, N, device=DEV, dtype=F32)
    intended = {
        "cell": "r3_argmin_grid",
        "access": "grid-tile reduction (reduce over grid program axis M, dim=0)",
        "origin": "grid (PARTIAL_GRID: M tunable grid axis, cdiv>1)",
        "extent": "static-4096 (per-grid count, NOT per-program size)",
        "carried_resident": "per-feature value+index (argmin 2-ary carry)",
        "co_residency": "single index reduction",
        "dims": "2D",
        "pinned_grid": "none (M is the reduced grid axis)",
        "not_modeled_axis_varied": "reduction-OP identity = argmin (index, int64, "
        "combine-arity-2) COMPOSED WITH origin=grid (the exclusion's target role)",
        "expect": "PARTIAL_GRID role => M floored (grid row), NOT sized full-extent; "
        "FIRE with a JUSTIFIED config (no grid collapse).",
    }
    v = check_kernel("r3_argmin_grid__col_argmin_grid", col_argmin_grid, (x,), intended)
    import json

    print(json.dumps({
        "red": v["red"],
        "reasons": v["reasons"],
        "observed": v["observed"],
    }, indent=2, default=repr))


if __name__ == "__main__":
    main()
