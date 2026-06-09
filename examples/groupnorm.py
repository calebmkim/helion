"""
GroupNorm Example
=================

This example implements a GroupNorm-style normalization using Helion in the welford-
bandmate idiom: a per-group reduction over the within-group extent, then a full-width
normalize/apply pass over the SAME extent (a reduce-then-apply, "Band C" structure).

Layout convention (matches the WS1 curriculum): the input is flattened to ``[M, N]`` where
``M = batch * num_groups`` (one row per group) and ``N = (channels / num_groups) * spatial``
(the within-group extent reduced over). Each group is standardized by its own mean/variance
and an element-wise affine ``weight``/``bias`` is applied over ``N``.

Unlike ``welford`` (a 3-statistic count/mean/M2 streaming recurrence), GroupNorm here uses
the lighter 2-moment combine (sum and sum-of-squares), so its combine carries less live
state per row. It is the clean welford *bandmate* used to fortify the Band-C combine /
normalize-tile sizing levers (``STRUCTURED_COMBINE_CAP_BYTES`` etc.) — proving whether they
encode a faithful reduce-then-apply property or are welford-spill-specific.
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
def groupnorm(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, eps: float = 1e-05
) -> torch.Tensor:
    """
    GroupNorm-style normalization via a 2-moment (sum, sum-of-squares) combine + apply.

    Args:
        weight: affine weight of shape [N]
        bias: affine bias of shape [N]
        x: input tensor of shape [M, N] (M = batch*groups, N = within-group extent)
        eps: numerical-stability epsilon

    Returns:
        Output tensor of shape [M, N].
    """
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m in hl.tile(m):
        acc_sum = torch.zeros_like(x[tile_m, 0], dtype=torch.float32)
        acc_sq = torch.zeros_like(acc_sum)
        acc_cnt = torch.zeros_like(acc_sum)

        for tile_n in hl.tile(n):
            # Reduce in fp32 (HBM load stays x.dtype; no-op at fp32). The 2-moment
            # combine (sum + sum-of-squares) is the GroupNorm-distinct arithmetic vs
            # welford's 3-stat recurrence.
            chunk = x[tile_m, tile_n].to(torch.float32)
            # True valid column count (OOB loads masked to 0); int32 accumulator so a
            # non-divisor N gives the correct divisor, not the padded tile width.
            cnt = (tile_n.index < n).to(torch.int32).sum()
            acc_sum = acc_sum + torch.sum(chunk, dim=-1)
            acc_sq = acc_sq + torch.sum(chunk * chunk, dim=-1)
            acc_cnt = acc_cnt + cnt

        mean = acc_sum / acc_cnt
        var = acc_sq / acc_cnt - mean * mean
        rstd_col = torch.rsqrt(var + eps)[:, None]
        mean_col = mean[:, None]

        for tile_n in hl.tile(n):
            xi = x[tile_m, tile_n]
            w = weight[tile_n][None, :]
            b = bias[tile_n][None, :]
            y = (xi - mean_col) * rstd_col
            y = y * w + b
            out[tile_m, tile_n] = y.to(x.dtype)
    return out


# %%
def eager_groupnorm(
    weight: torch.Tensor, bias: torch.Tensor, x: torch.Tensor, eps: float = 1e-05
) -> torch.Tensor:
    """Reference: per-row standardization over N + element-wise affine (matches the
    kernel's [M, N] layout, where each row is one group)."""
    return torch.nn.functional.layer_norm(
        x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=eps
    )


# %%
def check(m: int, n: int) -> None:
    """Verify groupnorm against the eager per-row standardize + affine reference."""
    weight = torch.randn((n,), device=DEVICE, dtype=torch.float32)
    bias = torch.randn((n,), device=DEVICE, dtype=torch.float32)
    x = torch.randn((m, n), device=DEVICE, dtype=torch.float32)
    run_example(groupnorm, eager_groupnorm, (weight, bias, x))


# %%
def main() -> None:
    check(2048, 4096)
    check(512, 32768)


if __name__ == "__main__":
    main()
