"""TARGET CELL: r3_gridtile_plus_sibling.

ADVERSARIAL re-probe of the grid-tile-reduction exclusion fix (a reduction OVER a tunable grid
axis is now classified PARTIAL_GRID and FLOORED, kept in m_block_ids rather than sized full-extent
as a secondary, so it doesn't collapse the grid).

The adversarial twist: the reduced grid axis (PARTIAL_GRID, floored) co-exists with a WIDE FAN-OUT
SIBLING grid axis -- a second tunable grid tile that is NOT reduced (an output "row" axis). After
flooring the reduced grid axis, does the sibling axis size SANELY (widen via the footprint cap), or
does flooring BOTH starve the kernel (or, conversely, does the sibling widen to FULL EXTENT and
collapse the grid -- an UNJUSTIFIED grid-collapse the property+caps cannot explain)?

Structure: x[BT, G, F].
  - tile_bt : grid dim-0, REDUCED via amax(dim over the bt grid tile) -> PARTIAL_GRID -> floored,
              kept in m_block_ids (a grid row).  NOTE: stride over the reduced elems == G*F*itemsize
              != itemsize, so this is the GRID dim-0 axis but reduced over a CONTIGUOUS-within-tile
              gather; we keep it a partial grid tile, not pinned block_size=1, so it is a tunable
              m_block axis (the in-scope adversarial shape, not the STRIDED-DIM0 out-of-scope one --
              this reduction's stride is over the tile, and the axis IS a tunable block_sizes entry).
  - tile_g  : grid dim-1, the WIDE FAN-OUT SIBLING -- NOT reduced, a tunable output-row axis.
  - F       : inner RESIDENT reduction (small extent) -> the dominant/primary reduction that makes
              the kernel FIRE (tiled_reduction_axes non-empty). Small so the footprint cap leaves
              register room: the question is whether the sibling G then widens sanely or runs away.

We REDUCE the bt grid axis by accumulating across the bt tile into a [tile_g] per-(g) accumulator
(amax over the bt tile rows), AND reduce the inner F axis. Output is [G] (the bt grid axis is
collapsed; G is the fan-out sibling). To keep BOTH bt and g as tunable grid tiles we tile them
together and write a per-(g) partial, finalized host-side -- exactly the PARTIAL_GRID grid-collapse
shape, but now with a wide independent sibling fan-out axis.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/dev/local/helion-unify/_lab/unify/probes")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import check_kernel  # noqa: E402

DEV = "cuda"


@helion.kernel(static_shapes=False)
def gridtile_plus_sibling(x: torch.Tensor) -> torch.Tensor:
    BT, G, F = x.shape
    # Per-(bt-tile, g) partials, reduced cross-CTA host-side. nb_bt rows of [G] partials.
    bt_block = hl.register_block_size(BT)
    nb_bt = (BT + bt_block - 1) // bt_block
    partials = x.new_empty([nb_bt, G], dtype=torch.float32)
    # Two grid axes: bt (reduced over, PARTIAL_GRID-floored) + g (the WIDE FAN-OUT SIBLING, unreduced).
    for tile_bt, tile_g in hl.tile([BT, G], block_size=[bt_block, None]):
        acc = hl.zeros([tile_g], dtype=torch.float32)
        # reduce the INNER F axis (RESIDENT primary) AND the bt grid tile rows (PARTIAL_GRID):
        # amax over (bt-rows, F) into a per-(g) scalar.
        blk = x[tile_bt, tile_g, :].to(torch.float32).abs()  # [bt, g, F]
        per_g = torch.amax(blk, dim=2)  # reduce F (RESIDENT inner) -> [bt, g]
        acc = torch.amax(per_g, dim=0)  # reduce bt grid tile (PARTIAL_GRID) -> [g]
        partials[tile_bt.id, tile_g] = acc
    return partials.amax(0)  # cross-CTA finalize over the bt grid blocks -> [G]


def main() -> None:
    print(f"helion={helion.__file__}")
    assert os.path.abspath(helion.__file__).startswith(
        "/home/dev/local/helion-unify" + os.sep
    ), helion.__file__
    # Wide fan-out sibling G; tall reduced grid axis BT; small inner reduction F.
    x = torch.randn(8192, 4096, 64, device=DEV, dtype=torch.float32)
    v = check_kernel(
        "r3_gridtile_plus_sibling",
        gridtile_plus_sibling,
        (x,),
        {
            "access": "grid-tile-reduction + wide fan-out sibling",
            "origin": "grid (bt) reduced PARTIAL_GRID-floored",
            "sibling": "tunable wide grid axis G (unreduced fan-out)",
            "inner": "RESIDENT F (primary)",
        },
    )
    import json

    print(json.dumps(v, indent=2, default=repr))


if __name__ == "__main__":
    main()
