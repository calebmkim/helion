"""CELL adv_gro_falsify: ADVERSARIAL-against-GRO (the just-changed Defect-2 re-key).

GRO (grid_reduction_origin) is recorded as::

    grid_reduction_origin = bool(m_block_ids) AND any(
        free_unbacked_symbols(env.block_sizes[b].size) for b in reducing_axes
    )

i.e. it claims "the reduction's ORIGIN is a GRID axis -> reduced ACROSS programs into a
per-feature accumulator, finalized cross-CTA" but it READS THIS OFF EXTENT-PROVENANCE: a grid
(m_block_ids non-empty) AND some reducing axis whose block-size carries an UNBACKED symbol. The
unbacked symbol is taken as the in-graph signature of the two-level grid decomposition
``for cta in hl.tile(M, m_block): for mb in hl.tile(cta.begin, cta.end): acc += reduce(...)``.

This cell attacks branch (b) of the falsification: a REAL grid-collapse (reduce ACROSS the grid
programs into a per-feature [N] accumulator, finalized cross-CTA via ``blocks.sum(0)``) whose
reducing axis is BACKED -- i.e. the per-CTA partial is computed in ONE statically-sized
reduction over the grid M tile rather than via an unbacked inner re-tile. Provenance:

    for cta in hl.tile(M, block_size=BLK):        # GRID axis, BACKED block_size=BLK
        blocks[cta.id, :] = torch.sum(x[cta, :], dim=0)   # per-feature [N] partial
    return blocks.sum(0)                          # cross-CTA finalize -> [N]

This is STRUCTURALLY identical to gro_divergence.col_energy_collapse (a per-feature sum-of /
energy collapse across the batch/grid axis into an [N] accumulator finalized cross-CTA) EXCEPT
the inner reduction is a single backed ``torch.sum(x[cta, :], dim=0)`` over the static grid tile
instead of an unbacked ``for mb in hl.tile(cta.begin, cta.end)`` re-tile. Because every reducing
axis is now BACKED, ``free_unbacked_symbols`` is empty for all of them, so::

    grid_reduction_origin = bool(m_block_ids) AND any(... unbacked ...) = True AND False = False

GRO does NOT fire even though this IS a genuine grid-origin M-collapse. The falsification
question: does GRO-False here produce a BAD config -- specifically, does the grid M axis take the
FOOTPRINT-widen path (``min(own, fits, occ, _m_block_cap)``) instead of the occupancy-collapse
sizing (``_m_collapse_grid_block`` = ``next_pow2(grid_rows // num_sm)``), and is that footprint
widen WRONG for a cross-CTA-finalized collapse?

OUT-OF-SCOPE guards deliberately avoided:
  * JAGGED -- every extent is a real static size (M, N, BLK), none is None/data-dependent.
  * STRIDED-DIM0 -- ``x`` is a contiguous [M, N] tensor; the reduction over the dim-0 grid tile
    walks rows whose stride over the reduced elements is N*itemsize for the row stride, but the
    per-element access within the summed slab is contiguous in the standard row-major layout (the
    reduced ELEMENTS x[cta, :] are loaded as full contiguous rows). This is the ordinary
    batch-collapse access pattern (same as col_energy_collapse), NOT the strided-dim0 predicate
    (a 1-D reduction axis whose memory stride over the reduced elements != itemsize).
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROBES = os.path.abspath(os.path.join(_THIS, ".."))
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import check_kernel  # noqa: E402

_WT = os.path.abspath(os.path.join(_THIS, "..", "..", "..", ".."))
assert os.path.abspath(helion.__file__).startswith(_WT + os.sep), helion.__file__

DEV = "cuda"


# REAL grid-origin collapse with a BACKED reducing axis: per-feature SUM over the batch/grid (M)
# axis into an [N] accumulator, finalized cross-CTA -- but the per-CTA partial is ONE statically-
# sized reduction over the grid M tile (block_size=BLK, BACKED) rather than an unbacked inner
# re-tile. Structurally an ORIGIN=grid M-collapse; GRO should fire on the property but its
# EXTENT-PROVENANCE key (unbacked reducing axis) does NOT -> the falsification.
@helion.kernel(static_shapes=False)
def backed_col_sum_collapse(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    m_block = hl.register_block_size(M)
    nb = (M + m_block - 1) // m_block
    blocks = x.new_zeros([nb, N], dtype=torch.float32)
    for cta in hl.tile(M, block_size=m_block):
        # BACKED reduction: sum the whole static grid tile [m_block, N] over dim-0 into the
        # per-feature [N] partial in one shot (no unbacked inner re-tile).
        blocks[cta.id, :] = torch.sum(x[cta, :].to(torch.float32), dim=0)
    return torch.sum(blocks, dim=0)


def main() -> None:
    print(f"helion={helion.__file__}\n")
    x = torch.randn(4096, 1024, device=DEV, dtype=torch.float32)
    intended = {
        "cell": "adv_gro_falsify",
        "attack": "GRO scenario (b): real grid-collapse, BACKED reducing axis",
        "access": "user-tiled",
        "origin": "grid",
        "extent": "backed (static grid M tile)",
        "family": "column-sum collapse (non-norm)",
        "expect_property": "grid_reduction_origin TRUE (it IS a grid-origin collapse)",
        "expect_key": "grid_reduction_origin FALSE (no unbacked reducing axis)",
        "question": "does GRO-False here mis-size the grid M axis (footprint-widen vs "
                    "occupancy-collapse)?",
    }
    v = check_kernel("adv_gro_falsify__backed_col_sum_collapse",
                     backed_col_sum_collapse, (x.clone(),), intended)
    obs = v["observed"]
    print(f"red                     = {v['red']}")
    print(f"reasons                 = {v['reasons']}")
    print(f"fired                   = {obs.get('fired')}")
    print(f"n_reduction_facts       = {obs.get('n_reduction_facts')}")
    print(f"n_matmul_facts          = {obs.get('n_matmul_facts')}")
    print(f"lowering_reduction_axes = {obs.get('lowering_reduction_axes')}")
    print(f"grid_block_ids          = {obs.get('grid_block_ids')}")
    print(f"block_sizes_valid_ids   = {obs.get('block_sizes_valid_ids')}")
    print(f"reduction_loops_valid   = {obs.get('reduction_loops_valid_ids')}")
    print(f"fact                    = {obs.get('fact')}")
    print(f"normalized_cfg          = {obs.get('normalized_cfg')}")


if __name__ == "__main__":
    main()
