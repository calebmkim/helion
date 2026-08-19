"""Pinned copies of the two example pointwise kernel bodies.

Copied from current Helion commit 61f4058f3610e1b2bbabc82df8267ac450f591df:

- examples/swiglu.py:_swiglu_fwd
- examples/geglu.py:_geglu

Keeping the bodies here avoids importing ``helion._testing`` and its unrelated
pytest dependency from the executable example modules.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel()
def swiglu_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, (
        f"Input tensors must have same shape, got {a.shape} != {b.shape}"
    )
    out = torch.empty_like(a, dtype=torch.promote_types(a.dtype, b.dtype))
    total_elements = a.numel()
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    out_flat = out.view(-1)
    for tile_idx in hl.tile(total_elements):
        a_vals = a_flat[tile_idx].to(torch.float32)
        b_vals = b_flat[tile_idx]
        sigmoid_a = torch.sigmoid(a_vals)
        silu_a = a_vals * sigmoid_a
        result = silu_a.to(b_vals.dtype) * b_vals
        out_flat[tile_idx] = result
    return out


@helion.kernel()
def geglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, (
        f"Input tensors must have same shape, got {a.shape} != {b.shape}"
    )
    out = torch.empty_like(a, dtype=torch.promote_types(a.dtype, b.dtype))
    total_elements = a.numel()
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    out_flat = out.view(-1)
    for tile_idx in hl.tile(total_elements):
        a_vals = a_flat[tile_idx].to(torch.float32)
        b_vals = b_flat[tile_idx]
        sqrt_2_over_pi = 0.7978845608028654
        a_cubed = a_vals * a_vals * a_vals
        tanh_arg = sqrt_2_over_pi * (a_vals + 0.044715 * a_cubed)
        tanh_result = torch.tanh(tanh_arg)
        gelu_a = 0.5 * a_vals * (1.0 + tanh_result)
        result = gelu_a.to(b_vals.dtype) * b_vals
        out_flat[tile_idx] = result
    return out
