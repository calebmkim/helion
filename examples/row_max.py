"""
Row Max Example
===============

This example implements a row-wise maximum reduction (``max(x, dim=-1)`` values, no index)
using Helion. It is a compute-TRIVIAL reduction — a single ``amax`` with no fp32-accumulate
arithmetic and a per-row scalar output ([M]) — the clean second probe (alongside argmax) for
whether the ``_num_warps`` element ramp / occupancy levers, tuned on compute-HEAVY reductions
(softmax exp/sum, welford), generalize to compute-light ones.
"""

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel()
def row_max(x: torch.Tensor) -> torch.Tensor:
    """
    Row-wise maximum of a 2D tensor along the last dimension.

    Args:
        x: Input tensor of shape [M, N]

    Returns:
        Output tensor of shape [M] holding ``amax(x, dim=-1)``.
    """
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        # A max reduction is order- and precision-stable in the input dtype (no summation
        # to accumulate error), so no fp32 upcast is needed or wanted — the reduced-tile
        # width stays the input dtype.
        out[tile_m] = torch.amax(x[tile_m, :], dim=-1)

    return out


# %%
def check(m: int, n: int) -> None:
    """Verify row_max against ``torch.amax``."""
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    run_example(row_max, lambda x: torch.amax(x, dim=-1), (x,))


# %%
def main() -> None:
    check(8192, 4096)
    check(65536, 512)


if __name__ == "__main__":
    main()
