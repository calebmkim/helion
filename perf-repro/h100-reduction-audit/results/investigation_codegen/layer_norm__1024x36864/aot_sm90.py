from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

_BLOCK_SIZE_0 = tl.constexpr(1)

@triton.jit
def _helion_layer_norm(bias, x, weight, out, bias_size_0, bias_stride_0, out_stride_0, out_stride_1, weight_stride_0, x_stride_0, x_stride_1, _REDUCTION_BLOCK_1: tl.constexpr):
    # src[layer_norm.py:18]: for tile_m in hl.tile(m):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    # src[layer_norm.py:20]: mean_val = torch.sum(acc, dim=-1) / n
    sum_1_acc = tl.full([_BLOCK_SIZE_0, _REDUCTION_BLOCK_1], 0, tl.float32)
    # src[layer_norm.py:19]: acc = x[tile_m, :].to(torch.float32)
    for roffset_1 in tl.range(0, bias_size_0, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < bias_size_0
        load = tl.load(x + (indices_0[:, None] * x_stride_0 + (0 + rindex_1)[None, :] * x_stride_1), mask_1[None, :], other=0, eviction_policy='evict_last')
        v_0 = tl.cast(load, tl.float32)
        # src[layer_norm.py:20]: mean_val = torch.sum(acc, dim=-1) / n
        v_1 = sum_1_acc + v_0
        sum_1_acc = v_1
    sum_1 = tl.cast(tl.sum(sum_1_acc, 1), tl.float32)
    v_2 = tl.cast(bias_size_0, tl.float32)
    v_3 = sum_1 / v_2
    # src[layer_norm.py:21]: centered = acc - mean_val[:, None]
    subscript = v_3[:, None]
    # src[layer_norm.py:22]: var_val = torch.sum(centered * centered, dim=-1) / n
    sum_2_acc = tl.full([_BLOCK_SIZE_0, _REDUCTION_BLOCK_1], 0, tl.float32)
    # src[layer_norm.py:19]: acc = x[tile_m, :].to(torch.float32)
    for roffset_1 in tl.range(0, bias_size_0, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < bias_size_0
        subscript_copy = subscript
        load_1 = tl.load(x + (indices_0[:, None] * x_stride_0 + (0 + rindex_1)[None, :] * x_stride_1), mask_1[None, :], other=0)
        v_4 = tl.cast(load_1, tl.float32)
        # src[layer_norm.py:21]: centered = acc - mean_val[:, None]
        v_5 = v_4 - subscript_copy
        # src[layer_norm.py:22]: var_val = torch.sum(centered * centered, dim=-1) / n
        v_6 = v_5 * v_5
        _mask_to_1 = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _REDUCTION_BLOCK_1]), v_6, tl.full([], 0, tl.float32))
        v_7 = sum_2_acc + _mask_to_1
        sum_2_acc = v_7
    sum_2 = tl.cast(tl.sum(sum_2_acc, 1), tl.float32)
    v_8 = tl.cast(bias_size_0, tl.float32)
    v_9 = sum_2 / v_8
    # src[layer_norm.py:23]: rstd_val = torch.rsqrt(var_val + 1e-5)
    v_10 = tl.full([], 1e-05, tl.float32)
    v_11 = v_9 + v_10
    v_12 = libdevice.rsqrt(v_11)
    # src[layer_norm.py:25]: centered * rstd_val[:, None] * weight[:].to(torch.float32)
    subscript_1 = v_12[:, None]
    # src[layer_norm.py:19]: acc = x[tile_m, :].to(torch.float32)
    for roffset_1 in tl.range(0, bias_size_0, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < bias_size_0
        v_3_copy = v_3
        subscript_1_copy = subscript_1
        load_2 = tl.load(x + (indices_0[:, None] * x_stride_0 + (0 + rindex_1)[None, :] * x_stride_1), mask_1[None, :], other=0, eviction_policy='evict_first')
        v_13 = tl.cast(load_2, tl.float32)
        # src[layer_norm.py:21]: centered = acc - mean_val[:, None]
        subscript_2 = v_3_copy[:, None]
        v_14 = v_13 - subscript_2
        # src[layer_norm.py:25]: centered * rstd_val[:, None] * weight[:].to(torch.float32)
        v_15 = v_14 * subscript_1_copy
        load_3 = tl.load(weight + (0 + rindex_1) * weight_stride_0, mask_1, other=0)
        v_16 = tl.cast(load_3, tl.float32)
        v_17 = v_16[None, :]
        v_18 = v_15 * v_17
        # src[layer_norm.py:26]: + bias[:].to(torch.float32)
        load_4 = tl.load(bias + (0 + rindex_1) * bias_stride_0, mask_1, other=0)
        v_19 = tl.cast(load_4, tl.float32)
        # src[layer_norm.py:25]: centered * rstd_val[:, None] * weight[:].to(torch.float32)
        # src[layer_norm.py:26]: + bias[:].to(torch.float32)
        v_20 = v_19[None, :]
        v_21 = v_18 + v_20
        # src[layer_norm.py:24]: out[tile_m, :] = (
        # src[layer_norm.py:25]:     centered * rstd_val[:, None] * weight[:].to(torch.float32)
        # src[layer_norm.py:26]:     + bias[:].to(torch.float32)
        # src[layer_norm.py:24-27]: ...
        v_22 = tl.cast(v_21, tl.float16)
        tl.store(out + (indices_0[:, None] * out_stride_0 + (0 + rindex_1)[None, :] * out_stride_1), v_22, mask_1[None, :])

def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, _launcher=_default_launcher):
    # src[layer_norm.py:16]: m, n = x.size()
    m, n = x.size()
    # src[layer_norm.py:17]: out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    # src[layer_norm.py:19]: acc = x[tile_m, :].to(torch.float32)
    _REDUCTION_BLOCK_1 = 8192
    # src[layer_norm.py:18]: for tile_m in hl.tile(m):
    # src[layer_norm.py:19]:     acc = x[tile_m, :].to(torch.float32)
    # src[layer_norm.py:20]:     mean_val = torch.sum(acc, dim=-1) / n
    # src[layer_norm.py:18-27]: ...
    _launcher(_helion_layer_norm, (m,), bias, x, weight, out, bias.size(0), bias.stride(0), out.stride(0), out.stride(1), weight.stride(0), x.stride(0), x.stride(1), _REDUCTION_BLOCK_1, num_warps=32, num_stages=2)
    # src[layer_norm.py:28]: return out
    return out