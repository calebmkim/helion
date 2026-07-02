"""STRESS-TEST KERNEL — P10: a user-tiled reduction and a (partial) GRID_TILE reduction.

REQUESTED as "a user-tiled reduction CO-RESIDENT with a grid reduction." PROBED RESULT: the
claim is FALSIFIED — the compiler puts them in SEPARATE graphs (SEQUENTIAL), NOT co-resident.
This is a genuine DESIGN FINDING (PROMPT.md §2.7): grid-axis reductions and inner user-tiled
reductions do NOT fuse into one graph; co-residency is confined to same-axis / feature+gradaccum
shapes. The kernel is kept as a SEQUENTIAL exemplar + the falsification record.

VERIFIED @fc1dbaa0 (_lab/explore_5_kernels.py): n_graphs=3, reds/graph={0:[0], 1:[2], 2:[2]},
non_reduction_loops=[1], n_facts=1 -> SEQUENTIAL (the requested co-residency does NOT occur).

IMPLEMENTER'S ASSERTIONS (add when Stage 1/2 exist):
  Tier 1: the grid-tile reduction and the user-tiled reduction land in DIFFERENT coresidency
          groups (different graph_id) — confirm the "grid+user-tile don't fuse" finding holds.
  Tier 2: sized as sequential passes, each its own budget.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def usertile_and_gridtile(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial grid tile over M; inside it a user-tile over N does a column reduction (over the
    grid tile_m) AND a per-row full-slice reduction over N. Requested co-resident; lands
    sequential (separate graphs)."""
    M, N = x.shape
    o1 = torch.empty([M, N], dtype=torch.float32, device=x.device)
    o2 = torch.empty([M], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=128):                  # partial grid tile (cdiv>1)
        for tile_n in hl.tile(N):                              # user tile
            blk = x[tile_m, tile_n].to(torch.float32)
            o1[tile_m, tile_n] = torch.amax(blk, dim=0)[None, :]  # reduce over the grid tile_m
        o2[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)       # user-tiled full-slice over N
    return o1, o2


def make_args(M: int = 2048, N: int = 4096, dtype=torch.float32, device="cuda"):
    return (torch.randn(M, N, device=device, dtype=dtype),)


def main() -> None:
    usertile_and_gridtile(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
