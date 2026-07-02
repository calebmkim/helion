"""STRESS-TEST KERNEL — P1: outer-product co-residency.

TAXONOMY POINT: two FULL_SLICE reductions over DIFFERENT axes of one resident input tile,
co-resident (same graph_id), combined into an outer product. The shared input tile [M, N]
is resident across both reductions.

WHY IT MATTERS: this is the ONLY kernel that exercises the `m·n` MULTIPLY footprint case in
the group-footprint formula (PROMPT.md §2.3) — every corpus kernel is either a single
reduction or a `r_A + r_B` ADD. The budget must size the resident [M, N] tile so it fits,
with the two reduction RESULTS ([M] and [N]) plus the [M, N] output materialized.

CORRECTNESS NOTE (fixed 2026-06-29): the original draft tiled BOTH reduced axes
(`hl.tile([M, N])`), which makes them GRID tiles — so each program reduced only its sub-tile
with NO cross-CTA combine → WRONG result (the redesign correctly classified those as GRID_TILE,
not FULL_SLICE, and declined). A reduction axis must NOT be grid-tiled. The faithful version
GRIDS over a non-reduced BATCH axis and holds the full [M, N] resident per program, so both
`sum(M)` and `sum(N)` are genuine FULL_SLICE reductions within one program.

IMPLEMENTER'S ASSERTIONS:
  Tier 1: both reductions classify FULL_SLICE; coresidency_groups groups them together
          (same graph_id); neither is rollable (two distinct rdims in one graph, §3).
  Tier 2: the resident [M, N] tile fits the byte budget — NOT floored, NOT spilling.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def outer_product_coresident(x: torch.Tensor) -> torch.Tensor:
    """out[b, m, n] = (sum over m of X[b])[n]  *  (sum over n of X[b])[m]. Per batch row, the
    full [M, N] tile is resident and reduced over BOTH inner axes (two co-resident FULL_SLICE
    reductions over different axes), combined into an outer product."""
    B, M, N = x.shape
    out = torch.empty([B, M, N], dtype=x.dtype, device=x.device)
    for tile_b in hl.tile(B, block_size=1):
        blk = x[tile_b, :, :].to(torch.float32)  # [1, M, N] resident
        col = blk.sum(1)   # reduce axis M (full slice) -> [1, N]
        row = blk.sum(2)   # reduce axis N (full slice) -> [1, M]
        out[tile_b, :, :] = (row[:, :, None] * col[:, None, :]).to(x.dtype)
    return out


def make_args(B: int = 2048, M: int = 64, N: int = 128, dtype=torch.float32,
              device="cuda"):
    return (torch.randn(B, M, N, device=device, dtype=dtype),)


def main() -> None:
    args = make_args()
    outer_product_coresident(*args)
    print("compiled + ran")


if __name__ == "__main__":
    main()
