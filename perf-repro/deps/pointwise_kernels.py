"""Vendored pointwise kernels for the PR #2866 reproduction.

Two groups:
  A. Authored flat-family kernels (from _lab/pointwise/ptw_kernels.py) — relu_squared,
     bias_gelu, dyt. Reduction-free forward; fp32-internal compute; tanh-approx GELU.
  B. The PR's two lever kernels (from the pointwise-2866-lab branch shards) —
     transposed_out_add (contig/coalescing lever), heavy_transcendental_1d (SFU warp ramp).

swiglu / geglu / residual_add come from examples/ directly (see pointwise_builders.py);
they are NOT re-authored here.

All kernels are authored on PRE-PROJECTED tensors (no matmul), so the PR #2866
PointwiseElementwiseFact fires (no reduction/matmul/accumulator fact present).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

_SQRT_2_OVER_PI = 0.7978845608028654


def _gelu_tanh(v: torch.Tensor) -> torch.Tensor:
    return 0.5 * v * (1.0 + torch.tanh(_SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))


# --------------------------------------------------------------------------- #
#  A. Authored flat-family kernels (verbatim from _lab/pointwise/ptw_kernels.py)
# --------------------------------------------------------------------------- #
@helion.kernel()
def relu_squared(x: torch.Tensor) -> torch.Tensor:
    """max(x,0)^2, single input — the ungated traffic-2 contrast to the GLU traffic-3 ops."""
    out = torch.empty_like(x)
    total = x.numel()
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    for tile in hl.tile(total):
        v = x_flat[tile].to(torch.float32)
        r = torch.where(v > 0, v, 0.0)
        out_flat[tile] = (r * r).to(x.dtype)
    return out


@helion.kernel()
def bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """GELU_tanh(x + bias[N]) on a pre-projected x[M,N]; bias[N] broadcasts over rows."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn].to(torch.float32) + bias[tn].to(torch.float32)
        out[tm, tn] = _gelu_tanh(v).to(x.dtype)
    return out


@helion.kernel()
def dyt(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, alpha: float) -> torch.Tensor:
    """Dynamic-Tanh (forward only): tanh(alpha*x) * gamma[N] + beta[N]. gamma/beta broadcast."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = torch.tanh(alpha * x[tm, tn].to(torch.float32))
        out[tm, tn] = (v * gamma[tn].to(torch.float32) + beta[tn].to(torch.float32)).to(x.dtype)
    return out


def ref_relu_squared(x):
    r = torch.clamp(x.float(), min=0.0)
    return (r * r).to(x.dtype)


def ref_bias_gelu(x, bias):
    v = x.float() + bias.float()
    return _gelu_tanh(v).to(x.dtype)


def ref_dyt(x, gamma, beta, alpha):
    return (torch.tanh(alpha * x.float()) * gamma.float() + beta.float()).to(x.dtype)


# --------------------------------------------------------------------------- #
#  B. The PR's two lever kernels (verbatim from the pointwise-2866-lab shards)
# --------------------------------------------------------------------------- #
@helion.kernel()
def transposed_out_add(x, y):
    """PR lever: coalescing / contig. Output is a transposed VIEW (store stride-1 on dim0),
    input reads are contiguous (stride-1 on dim1) -> a load-vs-store coalescing CONFLICT that
    the contiguity distributor must resolve with a balanced tile."""
    M, N = x.size()
    out = torch.empty(N, M, device=x.device, dtype=x.dtype).t()
    for tm, tn in hl.tile([M, N]):
        out[tm, tn] = (x[tm, tn].to(torch.float32) + y[tm, tn].to(torch.float32)).to(out.dtype)
    return out


@helion.kernel()
def heavy_transcendental_1d(x):
    """PR lever: SFU num_warps ramp. A 1-D tile with many transcendental (SFU) ops -> the
    arithmetic-intensity signal ramps num_warps to w16."""
    out = torch.empty_like(x)
    for t in hl.tile(x.size(0)):
        v = x[t].to(torch.float32)
        v = torch.sin(v) + torch.cos(v * 1.3)
        v = torch.tanh(v) - torch.tanh(v * 0.7)
        v = torch.exp(-v * v)
        v = torch.log(v + 2.0) / (torch.exp(v) + 1.0)
        v = torch.erf(v) + torch.sin(v * 2.1)
        v = torch.tanh(v) * torch.cos(v)
        v = torch.exp(v) / (torch.exp(-v) + 3.0)
        v = torch.log1p(torch.abs(v)) + torch.sin(v)
        v = v / (torch.cos(v) * torch.cos(v) + 1.5)
        v = torch.tanh(torch.exp(-torch.abs(v)))
        v = torch.erf(v * 0.9) - torch.sin(v)
        v = torch.exp(v) / (torch.log(torch.abs(v) + 2.0) + 1.0)
        out[t] = v.to(out.dtype)
    return out


def _ref_heavy_transcendental_1d(x):
    v = x.float()
    v = torch.sin(v) + torch.cos(v * 1.3)
    v = torch.tanh(v) - torch.tanh(v * 0.7)
    v = torch.exp(-v * v)
    v = torch.log(v + 2.0) / (torch.exp(v) + 1.0)
    v = torch.erf(v) + torch.sin(v * 2.1)
    v = torch.tanh(v) * torch.cos(v)
    v = torch.exp(v) / (torch.exp(-v) + 3.0)
    v = torch.log1p(torch.abs(v)) + torch.sin(v)
    v = v / (torch.cos(v) * torch.cos(v) + 1.5)
    v = torch.tanh(torch.exp(-torch.abs(v)))
    v = torch.erf(v * 0.9) - torch.sin(v)
    v = torch.exp(v) / (torch.log(torch.abs(v) + 2.0) + 1.0)
    return v.to(x.dtype)


def ref_transposed_out_add(x, y):
    return (x.float() + y.float()).to(x.dtype)


def ref_heavy_transcendental_1d(x):
    return _ref_heavy_transcendental_1d(x)
