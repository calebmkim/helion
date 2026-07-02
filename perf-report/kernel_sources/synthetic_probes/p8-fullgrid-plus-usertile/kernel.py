"""STRESS-TEST KERNEL — P8: a FULL_GRID reduction alongside a USER_TILE/full-slice reduction
in the same outer grid body.

REQUESTED as "FULL_GRID co-resident with a user-tiled reduction." PROBED RESULT: see the
docstring's VERIFIED line after running — the compiler's graph partition decides co-residency
(PROMPT.md §2.2/§2.7), and grid-axis vs inner-tile reductions tend to land in SEPARATE graphs
(jsd/p10 lesson). This kernel exists to pin down what this specific shape lands on.

TAXONOMY POINT: a per-(token,group) FULL_GRID max (specialized group_size on the grid) and a
per-token full-slice sum over a SECOND tensor, in one outer `hl.tile` body.

IMPLEMENTER'S ASSERTIONS (add when Stage 1/2 exist):
  Tier 1: the group reduction classifies FULL_GRID; the K reduction classifies (USER_TILE or
          FULL_SLICE); record their coresidency_groups (same or different graph_id — whatever
          the probe shows; do NOT assume co-resident).
  Tier 2: each sized correctly for its actual (co-resident vs sequential) group.

VERIFIED @fc1dbaa0 (this file): n_graphs=2, grid=[[0,1,2]], reds/graph={0:[2,3], 1:[3]}, n_facts=1
-> MIXED. The FULL_GRID group reduction (block 2) IS co-resident (graph 0) with part of the K
reduction (block 3), AND block 3 also has a sequential copy (graph 1). So "FULL_GRID + user-tile"
is PARTIALLY co-resident here — distinct from p10's clean falsification. A genuine mixed case.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def fullgrid_plus_usertile(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x:[T,G,GS] full-grid group max; y:[T,K] per-token full-slice sum; same outer body."""
    T, G, GS = x.shape
    K = y.shape[1]
    hl.specialize(GS)
    hl.specialize(G)
    o1 = torch.empty([T, G], dtype=torch.float32, device=x.device)
    o2 = torch.empty([T], dtype=torch.float32, device=x.device)
    for tile_t, tile_g, tile_c in hl.tile([T, G, GS], block_size=[1, None, GS]):
        o1[tile_t, tile_g] = torch.amax(x[tile_t, tile_g, tile_c].to(torch.float32), -1)  # FULL_GRID
        o2[tile_t] = y[tile_t, :].to(torch.float32).sum(-1)                                # full-slice over K
    return o1, o2


def make_args(T: int = 8192, G: int = 32, GS: int = 128, K: int = 2048,
              dtype=torch.bfloat16, device="cuda"):
    return (
        torch.randn(T, G, GS, device=device, dtype=dtype),
        torch.randn(T, K, device=device, dtype=dtype),
    )


def main() -> None:
    fullgrid_plus_usertile(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
