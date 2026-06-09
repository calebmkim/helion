"""
Argmax Example
==============

This example implements a row-wise argmax (index of the maximum element along the last
dimension) using Helion. It is an index-carrying reduction with an integer ([M] int64)
output — the coverage probe for the ``_num_warps`` element ramp on an op-variety kernel
that takes no override branch, and for the integer-output accuracy path (greedy decode /
MoE routing / class prediction).
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
def argmax(x: torch.Tensor) -> torch.Tensor:
    """
    Row-wise argmax of a 2D tensor along the last dimension.

    Args:
        x: Input tensor of shape [M, N]

    Returns:
        Output tensor of shape [M] (int64) holding ``argmax(x, dim=-1)``.
    """
    m, n = x.shape
    out = torch.empty([m], dtype=torch.int64, device=x.device)

    for tile_m in hl.tile(m):
        # argmax compares values directly (no accumulation), so no fp32 upcast is needed
        # or wanted — the reduced-tile width stays the input dtype.
        out[tile_m] = torch.argmax(x[tile_m, :], dim=-1)

    return out


# %%
def check(m: int, n: int) -> None:
    """Verify argmax against ``torch.argmax``."""
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    run_example(argmax, lambda x: torch.argmax(x, dim=-1), (x,))


# %%
def main() -> None:
    check(8192, 4096)
    check(4096, 50257)


if __name__ == "__main__":
    main()
