"""
LogSumExp Example
=================

This example implements a numerically-stable row-wise log-sum-exp reduction using
Helion. It is the cross-entropy loss primitive minus the target gather:
``lse(x)[i] = max_j x[i, j] + log(sum_j exp(x[i, j] - max_j x[i, j]))``.

Like cross_entropy, it re-reads the input row across two reductions (an ``amax`` then a
``sum`` over the shifted row), but its output is a per-row scalar ``[M]`` (no full-width
store) — so it lands in the same reduction-tree-bound regime as cross_entropy and is used
to fortify the ``REREAD_W8_MAX_BYTES`` num_warps lever (a "family of one" otherwise).
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
def logsumexp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically-stable row-wise log-sum-exp of a 2D tensor along the last dimension.

    Args:
        x: Input tensor of shape [M, N]

    Returns:
        Output tensor of shape [M] holding ``logsumexp(x, dim=-1)``.
    """
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        # Reduce the input row in its own dtype (re-read across amax + sum), exactly like
        # cross_entropy.py — logsumexp IS cross_entropy minus the target gather. The
        # max-subtraction makes the exp well-conditioned (values in [0, 1]) and Triton
        # accumulates the reduction in fp32 internally, so bf16/fp16 stay accurate without
        # a pre-upcast (an explicit ``.to(fp32)`` would change the reduced-tile width and
        # is unnecessary here, unlike the pure sum/l2 family).
        row = x[tile_m, :]  # [tile_m, N]
        max_logits = torch.amax(row, dim=-1, keepdim=True)
        shifted = row - max_logits
        sum_exp = torch.sum(torch.exp(shifted), dim=-1)
        lse = max_logits.squeeze(-1) + torch.log(sum_exp)
        out[tile_m] = lse.to(out.dtype)

    return out


# %%
def check(m: int, n: int) -> None:
    """Verify logsumexp against ``torch.logsumexp``."""
    x = torch.randn([m, n], device=DEVICE, dtype=torch.float32)
    run_example(logsumexp, lambda x: torch.logsumexp(x, dim=-1), (x,))


# %%
def main() -> None:
    check(8192, 32000)
    check(4096, 50257)


if __name__ == "__main__":
    main()
