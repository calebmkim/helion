"""STRESS-TEST KERNEL — P7: a GRID-axis reduction THEN a user-tiled reduction (separate loops).

TAXONOMY POINT: loop 1 reduces over the M GRID axis (a per-column amax), loop 2 is a user-tiled
full-slice reduction over N. Separate loops -> separate graphs -> SEQUENTIAL.

REQUESTED as "a grid-tile reduction followed by a user-tiled reduction (separate loops)."
VERIFIED @fc1dbaa0 (_lab/explore_5_kernels.py): n_graphs=4, reds in graphs {0,1,2,3},
rolled=[1,3], n_facts=2 -> SEQUENTIAL, two first-class facts. (Note: loop 1's reduction over the
M axis presents as a column reduction; the point is the SEQUENTIAL grid-then-user ordering and
that it builds TWO facts.)

IMPLEMENTER'S ASSERTIONS (add when Stage 1/2 exist):
  Tier 1: two reductions, DIFFERENT coresidency_groups (different graph_id); kernel FIRES with
          n_facts==2 (the relaxed >=1 gate).
  Tier 2: each sized against its own extent + own budget (sequential, not shared).
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def gridtile_then_usertile(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    col = torch.empty([N], dtype=torch.float32, device=x.device)
    row = torch.empty([M], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(N):                                   # loop 1: reduce over M (grid axis)
        col[tile_n] = torch.amax(x[:, tile_n].to(torch.float32), dim=0)
    for tile_m in hl.tile(M):                                   # loop 2: user-tiled full-slice over N
        row[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
    return col, row


def make_args(M: int = 2048, N: int = 4096, dtype=torch.float32, device="cuda"):
    return (torch.randn(M, N, device=device, dtype=dtype),)


def main() -> None:
    gridtile_then_usertile(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
