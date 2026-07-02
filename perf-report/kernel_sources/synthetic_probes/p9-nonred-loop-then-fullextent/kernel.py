"""STRESS-TEST KERNEL — P9: a NON-reduction loop THEN a full-extent reduction.

TAXONOMY POINT: loop 1 is an elementwise apply (NO reduction) over N; loop 2 is a full-slice
reduction over N. Tests that a non-reduction loop PRECEDING a reduction is recorded as
`non_reduction_loop_block_ids` and sized as a separate pass — and that ORDER (apply-then-reduce)
doesn't confuse the categorization.

REQUESTED as "a non reduction loop, followed by a full-extent reduction."
VERIFIED @fc1dbaa0 (_lab/explore_5_kernels.py): n_graphs=3, non_reduction_loops=[1], the
reduction in a separate graph, n_facts=1, rolled=[2] -> the apply loop is correctly a
non-reduction loop; the reduction is full-extent (rolled).

IMPLEMENTER'S ASSERTIONS (add when Stage 1/2 exist):
  Tier 1: the apply loop -> `non_reduction_loops` (NOT mistaken for a reduction); the sum ->
          FULL_SLICE. ORDER (apply first) does not change the classification.
  Tier 2: the apply loop sized as a SEPARATE pass (own budget); the reduction full-extent.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def nonred_loop_then_reduce(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    tmp = torch.empty_like(x)
    out = torch.empty([M], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M):
        for tile_n in hl.tile(N):                              # loop 1: NON-reduction apply
            tmp[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
        out[tile_m] = tmp[tile_m, :].to(torch.float32).sum(-1)  # loop 2: full-slice reduction
    return out


def make_args(M: int = 2048, N: int = 4096, dtype=torch.float32, device="cuda"):
    return (torch.randn(M, N, device=device, dtype=dtype),)


def main() -> None:
    nonred_loop_then_reduce(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
