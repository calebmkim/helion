"""
L2 Norm Example
===============

This example implements a row-wise L2 norm (``sqrt(sum(x*x))``) using Helion. It is a
streamed *single-load* reduction (one HBM read of the row, ``num_load == 1``) with a
per-row scalar output ([M]) — the clean coverage probe for the ``ROW_PERSIST_MAX_BYTES``
persistent-vs-looped byte boundary on a single-pass reduction (currently only ``long_sum``
exercises that regime).
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
def l2_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Row-wise L2 norm of a 2D tensor along the last dimension.

    Args:
        x: Input tensor of shape [M, N]

    Returns:
        Output tensor of shape [M] holding ``sqrt(sum(x*x, dim=-1))``.
    """
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        # fp32-accumulate the sum-of-squares (HBM load stays x.dtype). A half-precision
        # square+sum overflows / loses precision at realistic widths.
        row = x[tile_m, :].to(torch.float32)
        sq = torch.sum(row * row, dim=-1)
        out[tile_m] = torch.sqrt(sq).to(out.dtype)

    return out


# %%
def check(m: int, n: int) -> None:
    """Verify l2_norm against ``torch.linalg.vector_norm``."""
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    run_example(l2_norm, lambda x: torch.linalg.vector_norm(x, dim=-1), (x,))


# %%
def main() -> None:
    check(8192, 4096)
    check(256, 262144)


if __name__ == "__main__":
    main()
