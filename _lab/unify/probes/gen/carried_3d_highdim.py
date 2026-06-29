"""GEN CELL: carried_3d_highdim

A 3-D carried-resident reduction whose accumulator's LAST dim IS the reduction-tile dim,
carried across a high-dim (rank-3) grid loop nest. Stresses the carried-2d-tile residency
cap at rank > 2.

Distinction from the existing corpus:
  - C3 carried_2d_usertiled : rank-2, accumulator [M_BLOCK] (reduction dim collapsed each pass).
  - C5 four_d_inner_reduce  : rank-4 but accumulator [a,b,c] (reduction dim collapsed each pass).
  - HERE                    : rank-3, accumulator [M_BLOCK, P_BLOCK, R_BLOCK] — the R_BLOCK
                              (reduction-tile) dim is CARRIED RESIDENT across the user-tiled N
                              loop, then folded with a final .sum(-1). The carried tile is
                              3-D, so the carried-2d cap must hold at rank>2.

ACCESS=user-tiled  ORIGIN=inner-loop (N tiled inside a [M,P] grid)
EXTENT=static  CARRIED-RESIDENT=yes (last dim == reduction-tile dim)  DIMS=3
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
BF16 = torch.bfloat16


@helion.kernel(static_shapes=False)
def carried_3d_highdim(x: torch.Tensor) -> torch.Tensor:
    """x: [M, P, R, K]. Reduce over BOTH R and K. Grid over [M, P]; user-tile over K (inner loop).
    Carry a 3-D resident accumulator [M_BLOCK, P_BLOCK, R] whose LAST dim is the (resident)
    reduction dim R, accumulating per-r-lane partials across the user-tiled K loop, then fold
    the carried R dim with a final .sum(-1). The carried tile is rank-3 -> stresses the
    carried-2d residency cap at rank > 2."""
    M, P, R, K = x.shape
    R = hl.specialize(R)
    out = torch.empty([M, P], dtype=torch.float32, device=x.device)
    for tile_m, tile_p in hl.tile([M, P]):
        # 3-D carried-resident partial [M_BLOCK, P_BLOCK, R]: last dim R is a reduction dim,
        # carried across the user-tiled K loop (each iter sums its K-tile into the r-lanes).
        acc = hl.zeros([tile_m, tile_p, R], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = acc + x[tile_m, tile_p, :, tile_k].to(torch.float32).sum(-1)
        out[tile_m, tile_p] = acc.sum(-1)  # fold the carried resident reduction dim R
    return out


def main() -> None:
    print(f"helion={helion.__file__}\n")
    # [M, P, R, K]: grid [M,P], resident reduction over R (last acc dim), user-tiled over K.
    x = torch.randn(512, 48, 32, 2048, device=DEV, dtype=BF16)
    v = check_kernel(
        "carried_3d_highdim",
        carried_3d_highdim,
        (x.clone(),),
        {"dims": 3, "access": "user-tiled", "origin": "inner",
         "carried_resident": True, "carried_rank": 3},
    )
    import json
    print(json.dumps(v, indent=2, default=repr))


if __name__ == "__main__":
    main()
