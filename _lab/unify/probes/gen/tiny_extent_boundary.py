"""CELL: tiny_extent_boundary — boundary extents on the reduction axis.

Stresses next_pow2 rounding + the persist/loop boundary at the edges by reducing
over a TINY non-pow2 axis (extent 7) AND an EXTREME axis (>1M) in the SAME kernel,
each via its own user-tiled reduction loop (so both are tunable block_sizes entries
the FLOOR1_TILED check applies to).

  - tiny axis  K=7   : non-pow2, next_pow2 -> 8; does the sizer floor the tile to 1?
  - extreme axis N=2,097,152 (2^21): >1M; does the persist/loop boundary mis-size or
    NO_FIRE at the extreme edge?

Both reductions live in DIFFERENT loops (sequential passes), grid-pinned over M
(block_size=1) so the reduction axes are the only tunable reduction tiles. Property
point: ACCESS=user-tiled x ORIGIN=inner x EXTENT=static{tiny-nonpow2, extreme} x
CO-RESIDENCY=different-loop x PINNED-GRID(M).
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
F32 = torch.float32


@helion.kernel(static_shapes=False)
def tiny_and_extreme_reduce(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x: [M, K=7] reduced over the TINY non-pow2 axis (user-tiled).
    y: [M, N=2^21] reduced over the EXTREME axis (user-tiled, separate loop)."""
    M, K = x.shape
    _, N = y.shape
    out = torch.empty([M, 2], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        # TINY non-pow2 reduction over K=7
        acc_k = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc_k = acc_k + x[tile_m, tile_k].to(torch.float32).sum(-1)
        # EXTREME reduction over N=2^21, separate loop
        acc_n = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(N):
            acc_n = acc_n + y[tile_m, tile_n].to(torch.float32).sum(-1)
        out[tile_m, 0] = acc_k
        out[tile_m, 1] = acc_n
    return out


def main():
    print(f"helion={helion.__file__}\n")
    M = 4096
    K = 7                # tiny, non-pow2
    N = 2 * 1024 * 1024  # 2,097,152 = 2^21, extreme (>1M)
    x = torch.randn(M, K, device=DEV, dtype=F32)
    y = torch.randn(M, N, device=DEV, dtype=F32)
    intended = {
        "cell": "tiny_extent_boundary",
        "access": "user-tiled",
        "origin": "inner",
        "extent": "static{tiny-nonpow2=7, extreme=2^21}",
        "co_residency": "different-loop",
        "pinned_grid": "M(block_size=1)",
    }
    v = check_kernel("tiny_and_extreme_reduce", tiny_and_extreme_reduce, (x, y), intended)
    import json
    print(json.dumps({
        "red": v["red"],
        "reasons": v["reasons"],
        "observed": v["observed"],
    }, indent=2, default=repr))


if __name__ == "__main__":
    main()
