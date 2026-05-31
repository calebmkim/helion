"""
Standardize Example
===================

This example implements LayerNorm via a PLAIN two-moment reduce-then-apply
combine (sum + sum-of-squares), as opposed to ``welford.py``'s online
Welford recurrence.

It is a GENERALITY PROBE for the Band-C ``is_structured_combine`` autotuner
seed: it has the SAME reduce-then-apply STRUCTURE as welford (a combine pass
then an apply pass over the same axis) but a STRUCTURALLY DISTINCT combine —
two scalar moment accumulators (sum_x, sum_xsq; num_reduction_ops=2) computed
in a SINGLE non-recurrent pass, versus welford's 3-statistic
(count/mean/M2) online recurrence (num_reduction_ops=3). This exercises a
DIFFERENT ReductionFact profile under the same Band-C recipe.
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
# Standardize Kernel Implementation
# ---------------------------------


# %%
@helion.kernel()
def standardize(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, eps: float = 1e-05
) -> torch.Tensor:
    """
    Applies LayerNorm using a plain two-moment (sum + sum-of-squares) combine.

    Structurally a reduce-then-apply STRUCTURED COMBINE (like welford) but with a
    plain two-moment combine instead of Welford's online recurrence:
      Pass 1 (combine): accumulate sum_x and sum_xsq over the row, then
        mean = sum_x / n, var = sum_xsq / n - mean*mean, rstd = rsqrt(var + eps).
      Pass 2 (apply): y = (x - mean) * rstd * weight + bias.

    Args:
        weight: weight tensor of shape [N]
        bias: bias tensor of shape [N]
        x: input tensor of shape [M, N]
    Returns:
        Output tensor of shape [M, N]
    """
    m, n = x.size()

    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        acc_sum = torch.zeros_like(x[tile_m, 0], dtype=torch.float32)
        acc_sumsq = torch.zeros_like(acc_sum)
        acc_cnt = torch.zeros_like(acc_sum)

        # Pass 1 (combine): plain two-moment accumulation over the row.
        for tile_n in hl.tile(n):
            chunk = x[tile_m, tile_n]
            # Count of VALID columns in this tile via the masked-index idiom.
            # Helion masks out-of-bounds loads with other=0, so sum_x/sum_xsq are
            # already correct over the valid columns; the divisor must be the TRUE
            # valid count, NOT the constexpr tile width chunk.size(-1) (which
            # over-counts the padding on the last tile when block_size does not
            # divide n -> wrong mean/var at non-divisor N). `tile_n.index` is the
            # canonical Helion masked-index idiom (tile_interface.index).
            Tn = (tile_n.index < n).sum()
            acc_sum = acc_sum + torch.sum(chunk, dim=-1)
            acc_sumsq = acc_sumsq + torch.sum(chunk * chunk, dim=-1)
            acc_cnt = acc_cnt + Tn

        mean = acc_sum / acc_cnt
        var = acc_sumsq / acc_cnt - mean * mean
        rstd = torch.rsqrt(var + eps)
        mean_col = mean[:, None]
        rstd_col = rstd[:, None]

        # Pass 2 (apply): normalize, scale, shift.
        for tile_n in hl.tile(n):
            xi_chunk = x[tile_m, tile_n]
            w_chunk = weight[tile_n][None, :]
            b_chunk = bias[tile_n][None, :]

            y = (xi_chunk - mean_col) * rstd_col
            y = y * w_chunk + b_chunk

            out[tile_m, tile_n] = y.to(x.dtype)
    return out


# %%
# Baseline Function
# -----------------


# %%
def eager_standardize(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, eps: float = 1e-05
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=eps
    )


# %%
# Verification Function
# ---------------------


# %%
def check(s: int, d: int) -> None:
    """
    Verify the standardize kernel implementation against PyTorch's native
    layer_norm function.

    Args:
        s: First dimension of the test tensor
        d: Second dimension of the test tensor
    """

    weight = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    bias = torch.rand((d,), device=DEVICE, dtype=torch.float32)
    x = torch.rand((s, d), device=DEVICE, dtype=torch.float32)

    kernels = {"helion": standardize}
    run_example(kernels, eager_standardize, (weight, bias, x))


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the standardize kernel verification with different
    tensor sizes.
    """
    check(262144, 1024)
    check(262144, 1536)
    check(262144, 2048)


if __name__ == "__main__":
    main()
