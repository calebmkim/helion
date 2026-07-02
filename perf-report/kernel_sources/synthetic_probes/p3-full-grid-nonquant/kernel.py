"""STRESS-TEST KERNEL — P3: FULL_GRID reduction that is NOT a quant kernel.

TAXONOMY POINT: a reduction over a specialized full-extent axis placed on the grid
(cdiv==1, FixedBlockSizeSource) — the FULL_GRID category — but in a plain pooling/stats
kernel, not the per_token_group fp8-quant shape.

WHY IT MATTERS: the only corpus FULL_GRID example is per_token_group. If the FULL_GRID
category is secretly fitted to that one kernel's surrounding structure (the fp8 scale/quant
ops, the 3-axis tile), a second FULL_GRID kernel with different surroundings would expose it.
This proves FULL_GRID is a real category, not a per_token_group recognizer.

IMPLEMENTER'S ASSERTIONS (add when Stage 1/2 exist):
  Tier 1: the group-axis reduction classifies FULL_GRID (cdiv==1, pinned), NOT GRID_TILE or
          USER_TILE; the per-row grid sibling is a `grid_axes` entry (no reduction over it).
  Tier 2: the reduction claims ~nothing (grid-resident) -> the grid sibling inherits the
          remainder and WIDENS, occupancy-capped (the per_token_group widen mechanism, on a
          different kernel).
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def grouped_channel_max(x: torch.Tensor) -> torch.Tensor:
    """Per-(token, group) max over a specialized group_size axis — a grouped pooling stat.
    Same FULL_GRID idiom as per_token_group (3-D input + specialize + block_size=group_size)
    but the body is a plain max-pool, no quant. x: [num_tokens, num_groups, group_size]."""
    num_tokens, num_groups, group_size = x.shape
    hl.specialize(group_size)
    hl.specialize(num_groups)
    out = torch.empty([num_tokens, num_groups], dtype=torch.float32, device=x.device)
    for tile_m, tile_g, tile_c in hl.tile(
        [num_tokens, num_groups, group_size], block_size=[1, None, group_size]
    ):
        blk = x[tile_m, tile_g, tile_c].to(torch.float32)
        out[tile_m, tile_g] = torch.amax(blk, dim=-1)  # reduce the FULL_GRID group_size axis
    return out


def make_args(num_tokens: int = 8192, num_groups: int = 32, group_size: int = 128,
              dtype=torch.bfloat16, device="cuda"):
    x = torch.randn(num_tokens, num_groups, group_size, device=device, dtype=dtype)
    return (x,)


def main() -> None:
    grouped_channel_max(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
