"""
Sum Reduction Example
=====================

This example demonstrates how to implement a sum reduction operation along the last dimension using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Sum Kernel
# ----------


# %%
@helion.kernel()
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    """
    Sums a 2D tensor along the last dimension.

    Args:
        x: Input tensor of shape [M, N]

    Returns:
        Output tensor of shape [M] containing the sum of each row
    """
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        # Reduce in fp32 even for bf16/fp16 inputs. An input-width (bf16/fp16)
        # accumulator loses small addends as the running sum grows (and can overflow
        # fp16), giving a result far less accurate than torch.sum / torch.compile,
        # which upcast the accumulator to fp32. The HBM load stays x.dtype, so this
        # costs no extra bandwidth; it is a true no-op at fp32 input.
        out[tile_m] = x[tile_m, :].to(torch.float32).sum(-1).to(out.dtype)

    return out


# %%
# Benchmark Wrapper
# -----------------


# %%
def sum_tritonbench(tb_op: object, x: torch.Tensor) -> Callable[[], torch.Tensor]:
    """
    Wrapper for tritonbench that handles 1D input.

    Args:
        tb_op: TritonBench operator instance
        x: Input tensor (1D or 2D)

    Returns:
        Callable that returns sum of the tensor along the last dimension
    """

    def compute_sum() -> torch.Tensor:
        if x.ndim == 1:
            # For 1D tensors, reshape to 2D for sum_kernel
            x_2d = x.unsqueeze(0)
            result = sum_kernel(x_2d)
            return result.squeeze()
        return sum_kernel(x)

    return compute_sum


# %%
# Verification Function
# ---------------------


# %%
def check(m: int, n: int) -> None:
    """
    Verify the sum kernel implementation against PyTorch's native sum function.

    Args:
        m: First dimension of the test tensor
        n: Second dimension of the test tensor
    """
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    kernels = {"helion": sum_kernel}
    run_example(kernels, lambda x: x.sum(-1), (x,))


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the sum kernel verification with different tensor sizes.
    """
    check(5120, 2560)
    check(10240, 10240)


if __name__ == "__main__":
    main()
