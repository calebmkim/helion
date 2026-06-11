"""
Long Dimension Sum Example
==========================

This example demonstrates how to implement efficient sum reduction along a long dimension using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Baseline Implementation
# -----------------------


# %%
def baseline_sum(x: torch.Tensor) -> torch.Tensor:
    """
    PyTorch baseline implementation of sum reduction along the last dimension.

    Args:
        x: Input tensor

    Returns:
        Tensor with sum of elements along the last dimension
    """
    return x.sum(-1)


# %%
# Naive Reduction Kernel
# ----------------------


# %%
@helion.kernel(
    config=helion.Config(
        block_sizes=[1],
        reduction_loops=[None],
        num_warps=32,
        num_stages=4,
        indexing="block_ptr",
    )
)
def longsum(x: torch.Tensor) -> torch.Tensor:
    """
    Naive reduction kernel that sums elements along the last dimension.

    Loads the entire reduction dimension at once and reduces in registers.

    Args:
        x: Input tensor of shape [m, n]

    Returns:
        Output tensor of shape [m] containing the sum of each row
    """
    m, _ = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        # fp32-accumulate (HBM load stays x.dtype): a bf16/fp16 accumulator loses small
        # addends over a long row and can overflow fp16. No-op at fp32. See sum.py.
        out[tile_m] = x[tile_m, :].to(torch.float32).sum(-1).to(out.dtype)
    return out


# %%
# Looped Reduction Kernel
# -----------------------


# %%
@helion.kernel(
    config=helion.Config(
        block_sizes=[1],
        reduction_loops=[
            32768
        ],  # [None] for naive reduction, [tile_size] for looped reduction
        num_warps=16,
        num_stages=5,
        indexing="pointer",
    )
)
def longsum_w_red_loop(x: torch.Tensor) -> torch.Tensor:
    """
    Looped reduction kernel that sums elements along the last dimension.

    Uses a reduction loop with a specified tile size to handle large dimensions efficiently.

    Args:
        x: Input tensor of shape [m, n]

    Returns:
        Output tensor of shape [m] containing the sum of each row
    """
    m, _ = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        # fp32-accumulate (HBM load stays x.dtype): a bf16/fp16 accumulator loses small
        # addends over a long row and can overflow fp16. No-op at fp32. See sum.py.
        out[tile_m] = x[tile_m, :].to(torch.float32).sum(-1).to(out.dtype)
    return out


# %%
# Manual Looped Reduction Kernel
# ------------------------------


# %%
@helion.kernel(
    config=helion.Config(
        block_sizes=[32768, 1], num_warps=16, num_stages=5, indexing="pointer"
    )
)
def longsum_manual(x: torch.Tensor) -> torch.Tensor:
    """
    Manual implementation of looped reduction for summing elements along the last dimension.

    Manually implements the reduction loop with explicit accumulation and final reduction.

    Args:
        x: Input tensor of shape [m, n]

    Returns:
        Output tensor of shape [m] containing the sum of each row
    """
    m, n = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    # Call register_block_size to know block_size_n outside of the reduction loop.
    block_size_n = hl.register_block_size(n)

    for tile_m in hl.tile(m):
        # fp32-accumulate the carried partial sums (see sum.py): a bf16/fp16 acc loses
        # precision over a long row. fp32 acc + bf16 addend stays fp32; no-op at fp32.
        acc = hl.zeros([tile_m, block_size_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):  # Reduction loop
            acc += x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1).to(out.dtype)
    return out


# %%
# Verification Function
# ---------------------


# %%
def check(m: int, n: int) -> None:
    """
    Verify the sum kernel implementations against PyTorch's native sum function.

    Tests all three kernel variants (naive, looped, manual) against the baseline.

    Args:
        m: First dimension of the test tensor
        n: Second dimension of the test tensor (reduction dimension)
    """
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)

    # Test all three kernel variants against the baseline
    kernels = {
        "helion naive": longsum,
        "helion loop": longsum_w_red_loop,
        "helion manual": longsum_manual,
    }

    run_example(kernels, baseline_sum, (x,))


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the sum kernel verification with a large tensor.

    Tests with a tensor of shape [4, 130000] to demonstrate handling of long reduction dimensions.
    """
    check(4, 130000)  # seq_len = 128k


if __name__ == "__main__":
    main()
