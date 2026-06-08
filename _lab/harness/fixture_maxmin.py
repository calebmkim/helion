"""TEST FIXTURES (NOT curriculum): generic row-wise max / min reductions over the
last dim, modeled EXACTLY on examples/sum.py::sum_kernel. A single @helion.kernel
doing torch.amax(x, dim=-1) / torch.amin(x, dim=-1). Minimal + compiler-managed
(T1 rollable, like sum) so the reduction axis is the compiler's to roll.

These exist ONLY to test whether the FROZEN v8 heuristic seeds a NEW reduction OP
(max/min — a genuinely different accumulator than the sum/mean/amax-in-softmax of
the 9-kernel curriculum) out of the box. They do NOT touch helion/ and are NOT in
examples/ (they are not curriculum kernels).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel()
def max_kernel(x: torch.Tensor) -> torch.Tensor:
    """Row-wise max of a 2D tensor along the last dim. Shape [M, N] -> [M].

    Modeled on sum_kernel: a single compiler-rollable reduction over the last dim.
    """
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        out[tile_m] = torch.amax(x[tile_m, :], dim=-1)

    return out


@helion.kernel()
def min_kernel(x: torch.Tensor) -> torch.Tensor:
    """Row-wise min of a 2D tensor along the last dim. Shape [M, N] -> [M].

    Modeled on sum_kernel: a single compiler-rollable reduction over the last dim.
    """
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        out[tile_m] = torch.amin(x[tile_m, :], dim=-1)

    return out
