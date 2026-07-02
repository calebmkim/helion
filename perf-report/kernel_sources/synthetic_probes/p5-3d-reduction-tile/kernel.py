"""STRESS-TEST KERNEL — P5: a >=3-D reduction tile.

TAXONOMY POINT: a reduction whose resident tile is 3-D — reduce over TWO inner axes of a
[m, a, b] tile within one program. Stresses the "don't assume rank <= 2" claim: the
group-footprint formula (PROMPT.md §2.3, `prod of tiled dims`) and the cap arithmetic must
carry a 3-D resident tile, not a hard-coded [m_block, r_block].

WHY IT MATTERS: nearly every corpus kernel has a 2-D [m, r] resident tile. If the budget /
cap code implicitly assumes rank 2, a 3-D tile silently mis-sizes (or crashes). The design
says dims are 1..4 — this is the >=3 witness.

IMPLEMENTER'S ASSERTIONS (add when Stage 1/2 exist):
  Tier 1: the reduced axes classify (USER_TILE or FULL_SLICE as written); the resident tile
          is recognized as rank-3 (the footprint includes a_block * b_block).
  Tier 2: block_sizes size the 3-D tile against the byte budget (the product, not a 2-D
          assumption); no crash, no floor-to-1 of a real reduction.

CORRECTNESS NOTE (fixed 2026-06-29): the original draft tiled the reduced axes
(`hl.tile([M, A, B])`), making A,B GRID tiles → each program reduced only its sub-tile with NO
cross-CTA combine → WRONG result. A reduction axis must NOT be grid-tiled. The faithful version
GRIDS over M and holds the full [A, B] inner block resident, so `sum` over the two inner axes
is a genuine within-program reduction over a rank-3 resident tile.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def reduce_two_inner_axes(x: torch.Tensor) -> torch.Tensor:
    """out[m] = sum over (a, b) of x[m, a, b]. Grid over M; the full [A, B] inner block is
    resident per program and reduced over both inner axes (a rank-3 [m_block, A, B] tile)."""
    M, A, B = x.shape
    out = torch.empty([M], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M):
        blk = x[tile_m, :, :].to(torch.float32)  # [m_block, A, B] resident
        out[tile_m] = blk.sum(-1).sum(-1)  # reduce B then A -> [m_block]
    return out


def make_args(M: int = 4096, A: int = 64, B: int = 64, dtype=torch.float32, device="cuda"):
    return (torch.randn(M, A, B, device=device, dtype=dtype),)


def main() -> None:
    reduce_two_inner_axes(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
