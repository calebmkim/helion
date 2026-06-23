"""Authored standalone elementwise corpus kernels (Phase 2 + held-out), for the pointwise
seed bench. Kept in _lab (the FACT + HEURISTIC deliverable does not depend on them; they
only exercise the heuristic). Each is reduction-FREE in the forward (the disjointness rule).

  relu_squared : max(x,0)^2          1 input, 1-D flatten, traffic-2  (Liger relu_squared)
  bias_gelu    : gelu_tanh(x+bias[N]) broadcast bias, N-D,    traffic-2  (GPT-2/BERT FFN)
  dyt          : tanh(alpha*x)*gamma[N]+beta[N]  broadcast, N-D, traffic-2  (Dynamic-Tanh, fwd only)

All authored on PRE-PROJECTED [M,N] tensors (no matmul in the kernel).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

_SQRT_2_OVER_PI = 0.7978845608028654


def _gelu_tanh(v: torch.Tensor) -> torch.Tensor:
    return 0.5 * v * (1.0 + torch.tanh(_SQRT_2_OVER_PI * (v + 0.044715 * v * v * v)))


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


# ---- standalone references (match the kernel's fp32-internal compute) ----
def ref_relu_squared(x):
    r = torch.clamp(x.float(), min=0.0)
    return (r * r).to(x.dtype)


def ref_bias_gelu(x, bias):
    v = x.float() + bias.float()
    return _gelu_tanh(v).to(x.dtype)


def ref_dyt(x, gamma, beta, alpha):
    return (torch.tanh(alpha * x.float()) * gamma.float() + beta.float()).to(x.dtype)
