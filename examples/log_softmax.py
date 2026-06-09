"""
LogSoftmax Example
==================

This example implements a numerically-stable row-wise log-softmax using Helion:
``log_softmax(x)[i, j] = (x[i, j] - max_k x[i, k]) - log(sum_k exp(x[i, k] - max_k x[i, k]))``.

It re-reads the input row across the ``amax``/``sum`` reductions AND writes the result
back FULL-WIDTH ([M, N], unlike cross_entropy's scalar loss). It is used to fortify the
``persistent_interleaved`` + ``maxnreg`` lever (the untested full-width looped-reread case;
that lever was tuned only on cross_entropy's scalar output).
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
def log_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically-stable row-wise log-softmax of a 2D tensor along the last dimension.

    Args:
        x: Input tensor of shape [M, N]

    Returns:
        Output tensor of shape [M, N] holding ``log_softmax(x, dim=-1)``.
    """
    m, n = x.shape
    out = torch.empty_like(x)

    for tile_m in hl.tile(m):
        # Reduce the row in its own dtype (re-read across amax + sum + the apply pass),
        # like softmax_two_pass / cross_entropy: the max-shift conditions the exp and
        # Triton accumulates the sum in fp32 internally, so half precision stays accurate
        # without a pre-upcast that would change the reduced-tile width.
        row = x[tile_m, :]
        max_logits = torch.amax(row, dim=-1, keepdim=True)
        shifted = row - max_logits
        sum_exp = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
        log_sum_exp = torch.log(sum_exp)
        out[tile_m, :] = (shifted - log_sum_exp).to(out.dtype)

    return out


# %%
def check(m: int, n: int) -> None:
    """Verify log_softmax against ``torch.nn.functional.log_softmax``."""
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    run_example(
        log_softmax, lambda x: torch.nn.functional.log_softmax(x, dim=-1), (x,)
    )


# %%
def main() -> None:
    check(4096, 16384)
    check(2048, 131072)


if __name__ == "__main__":
    main()
