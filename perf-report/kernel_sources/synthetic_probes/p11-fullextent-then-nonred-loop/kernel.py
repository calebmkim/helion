"""STRESS-TEST KERNEL — P11: a full-extent reduction THEN a non-reduction loop block.

TAXONOMY POINT: loop 1 (full-slice reduction over N) followed by loop 2 (an elementwise
normalize/apply over N — NO reduction). The canonical reduce-then-apply shape (rms_norm-like).
Tests the non_reduction_loop sizing AFTER a reduction (the common ordering).

REQUESTED as "a full extent reduction followed by a non-reduction loop block."
VERIFIED @fc1dbaa0 (_lab/explore_5_kernels.py): n_graphs=3, non_reduction_loops=[2], reduction
in its own graph, n_facts=1, rolled=[1] -> the apply loop is a non-reduction loop, the reduction
is full-extent. (This is the rms_norm fwd shape; included for the explicit reduce-THEN-apply
ordering as a clean taxonomy anchor.)

IMPLEMENTER'S ASSERTIONS (add when Stage 1/2 exist):
  Tier 1: the sum -> FULL_SLICE; the normalize loop -> `non_reduction_loops`.
  Tier 2: the apply tile sized to match the reduction tile (the §2.3 normalize-loop rule);
          byte-identical to rms_norm's existing behavior (this IS that shape).
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def fullextent_then_nonred(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(M):
        s = (x[tile_m, :].to(torch.float32) ** 2).sum(-1)       # loop 1: full-slice reduction
        rms = torch.rsqrt(s / N + 1e-5)
        for tile_n in hl.tile(N):                               # loop 2: NON-reduction apply
            out[tile_m, tile_n] = (
                x[tile_m, tile_n].to(torch.float32) * rms[:, None] * w[tile_n]
            ).to(x.dtype)
    return out


def make_args(M: int = 2048, N: int = 4096, dtype=torch.float32, device="cuda"):
    return (torch.randn(M, N, device=device, dtype=dtype), torch.randn(N, device=device, dtype=dtype))


def main() -> None:
    fullextent_then_nonred(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
