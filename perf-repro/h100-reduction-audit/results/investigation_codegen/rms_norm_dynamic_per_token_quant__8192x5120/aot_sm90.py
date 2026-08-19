from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher
import _h100_audit_rms_norm_dynamic_per_token_quant as _source_module

_BLOCK_SIZE_0 = tl.constexpr(1)
_BLOCK_SIZE_1 = tl.constexpr(8192)
_BLOCK_SIZE_2 = tl.constexpr(8192)
_BLOCK_SIZE_3 = tl.constexpr(2048)

@triton.jit
def _helion_rms_norm_dynamic_per_token_quant(input_1, weight, scale, result, input_1_stride_0, input_1_stride_1, result_stride_0, result_stride_1, scale_stride_0, scale_stride_1, weight_stride_0, epsilon, qtype_traits_max, min_scaling_factor, qtype_traits_min):
    # src[rms_norm_dynamic_per_token_quant.py:87]: for tile_m in hl.tile(num_tokens, block_size=1):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    # src[rms_norm_dynamic_per_token_quant.py:88]: rms = hl.zeros([tile_m], dtype=torch.float32)
    rms = tl.full([_BLOCK_SIZE_0], 0.0, tl.float32)
    # src[rms_norm_dynamic_per_token_quant.py:89]: for tile_n in hl.tile(hidden_size):
    # src[rms_norm_dynamic_per_token_quant.py:90]:     x_blk = input[tile_m, tile_n].to(torch.float32)
    # src[rms_norm_dynamic_per_token_quant.py:91]:     if residual is not None:
    # src[rms_norm_dynamic_per_token_quant.py:89-93]: ...
    for offset_1 in tl.range(0, 5120, _BLOCK_SIZE_1, loop_unroll_factor=4, disallow_acc_multi_buffer=True, flatten=True):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < 5120
        rms_copy = rms
        rms_copy_0 = rms_copy
        # src[rms_norm_dynamic_per_token_quant.py:90]: x_blk = input[tile_m, tile_n].to(torch.float32)
        load = tl.load(input_1 + (indices_0[:, None] * input_1_stride_0 + indices_1[None, :] * input_1_stride_1), mask_1[None, :], other=0)
        v_0 = tl.cast(load, tl.float32)
        # src[rms_norm_dynamic_per_token_quant.py:93]: rms = rms + x_blk.pow(2).sum(dim=-1)
        v_1 = v_0 * v_0
        sum_1 = tl.cast(tl.sum(v_1, 1), tl.float32)
        rms = rms_copy_0 + sum_1
    # src[rms_norm_dynamic_per_token_quant.py:95]: rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)
    v_3 = tl.full([], 0.0001953125, tl.float32)
    v_4 = rms * v_3
    v_5 = v_4 + epsilon
    v_6 = libdevice.rsqrt(v_5)
    # src[rms_norm_dynamic_per_token_quant.py:96]: s_blk = hl.zeros([tile_m], dtype=torch.float32)
    s_blk = tl.full([_BLOCK_SIZE_0], 0.0, tl.float32)
    # src[rms_norm_dynamic_per_token_quant.py:98]: for tile_n in hl.tile(hidden_size):
    # src[rms_norm_dynamic_per_token_quant.py:99]:     x_blk = input[tile_m, tile_n].to(torch.float32)
    # src[rms_norm_dynamic_per_token_quant.py:100]:     if residual is not None:
    # src[rms_norm_dynamic_per_token_quant.py:98-104]: ...
    tl.debug_barrier()
    for offset_2 in tl.range(0, 5120, _BLOCK_SIZE_2, loop_unroll_factor=3, flatten=True):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < 5120
        v_6_copy = v_6
        s_blk_copy = s_blk
        v_6_copy_0 = v_6_copy
        s_blk_copy_0 = s_blk_copy
        # src[rms_norm_dynamic_per_token_quant.py:99]: x_blk = input[tile_m, tile_n].to(torch.float32)
        load_1 = tl.load(input_1 + (indices_0[:, None] * input_1_stride_0 + indices_2[None, :] * input_1_stride_1), mask_2[None, :], other=0)
        v_7 = tl.cast(load_1, tl.float32)
        # src[rms_norm_dynamic_per_token_quant.py:102]: x_blk = (x_blk * rms[:, None]).to(input.dtype) * weight[None, tile_n]
        subscript = v_6_copy_0[:, None]
        v_8 = v_7 * subscript
        v_9 = tl.cast(v_8, tl.bfloat16)
        load_2 = tl.load(weight + indices_2[None, :] * weight_stride_0, mask_2[None, :], other=0)
        v_10 = v_9 * load_2
        # src[rms_norm_dynamic_per_token_quant.py:103]: tmp_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)
        v_11 = tl_math.abs(v_10)
        _mask_to = tl.where(tl.broadcast_to(mask_2[None, :], [_BLOCK_SIZE_0, _BLOCK_SIZE_2]), v_11, tl.full([], float('-inf'), tl.bfloat16))
        amax = tl.cast(tl.max(_mask_to, 1), tl.bfloat16)
        v_12 = tl.cast(amax, tl.float32)
        # src[rms_norm_dynamic_per_token_quant.py:104]: s_blk = torch.maximum(s_blk, tmp_blk)
        s_blk = triton_helpers.maximum(s_blk_copy_0, v_12)
    # src[rms_norm_dynamic_per_token_quant.py:109]: s_blk = s_blk * (1.0 / qtype_max)
    truediv = 1.0 / qtype_traits_max
    v_14 = s_blk * truediv
    # src[rms_norm_dynamic_per_token_quant.py:110]: s_blk = s_blk.clamp(min=min_scaling_factor)
    v_15 = triton_helpers.maximum(v_14, min_scaling_factor)
    # src[rms_norm_dynamic_per_token_quant.py:111]: scale[tile_m, 0] = s_blk
    tl.store(scale + (indices_0 * scale_stride_0 + 0 * scale_stride_1), v_15, None)
    # src[rms_norm_dynamic_per_token_quant.py:113]: for tile_n in hl.tile(hidden_size):
    # src[rms_norm_dynamic_per_token_quant.py:114]:     x_blk = input[tile_m, tile_n].to(torch.float32)
    # src[rms_norm_dynamic_per_token_quant.py:115]:     if residual is not None:
    # src[rms_norm_dynamic_per_token_quant.py:113-128]: ...
    tl.debug_barrier()
    for offset_3 in tl.range(0, 5120, _BLOCK_SIZE_3, loop_unroll_factor=2, disallow_acc_multi_buffer=True):
        indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
        mask_3 = indices_3 < 5120
        v_6_copy_1 = v_6
        v_15_copy = v_15
        v_6_copy_1_0 = v_6_copy_1
        v_15_copy_0 = v_15_copy
        # src[rms_norm_dynamic_per_token_quant.py:114]: x_blk = input[tile_m, tile_n].to(torch.float32)
        load_3 = tl.load(input_1 + (indices_0[:, None] * input_1_stride_0 + indices_3[None, :] * input_1_stride_1), mask_3[None, :], other=0)
        v_16 = tl.cast(load_3, tl.float32)
        # src[rms_norm_dynamic_per_token_quant.py:118]: x_blk = (x_blk * rms[:, None]).to(input.dtype) * weight[None, tile_n]
        subscript_1 = v_6_copy_1_0[:, None]
        v_17 = v_16 * subscript_1
        v_18 = tl.cast(v_17, tl.bfloat16)
        load_4 = tl.load(weight + indices_3[None, :] * weight_stride_0, mask_3[None, :], other=0, eviction_policy='evict_last')
        v_19 = v_18 * load_4
        # src[rms_norm_dynamic_per_token_quant.py:124]: y_blk = x_blk / s_blk[:, None]
        subscript_2 = v_15_copy_0[:, None]
        v_20 = tl.cast(v_19, tl.float32)
        v_21 = v_20 / subscript_2
        # src[rms_norm_dynamic_per_token_quant.py:126]: result[tile_m, tile_n] = y_blk.clamp(qtype_traits_min, qtype_traits_max).to(
        v_22 = triton_helpers.maximum(v_21, qtype_traits_min)
        v_23 = triton_helpers.minimum(v_22, qtype_traits_max)
        # src[rms_norm_dynamic_per_token_quant.py:126]: result[tile_m, tile_n] = y_blk.clamp(qtype_traits_min, qtype_traits_max).to(
        # src[rms_norm_dynamic_per_token_quant.py:127]:     result.dtype
        # src[rms_norm_dynamic_per_token_quant.py:128]: )
        v_24 = tl.cast(v_23, tl.float8e4nv)
        tl.store(result + (indices_0[:, None] * result_stride_0 + indices_3[None, :] * result_stride_1), v_24, mask_3[None, :])

def rms_norm_dynamic_per_token_quant(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, epsilon: float, scale_ub: torch.Tensor | None=None, residual: torch.Tensor | None=None, *, _launcher=_default_launcher):
    # src[rms_norm_dynamic_per_token_quant.py:53]: assert input.ndim == 2
    assert input.ndim == 2
    # src[rms_norm_dynamic_per_token_quant.py:54]: num_tokens, hidden_size = input.shape
    num_tokens, hidden_size = input.shape
    # src[rms_norm_dynamic_per_token_quant.py:57]: fp8_dtype = _FP8_DTYPE
    fp8_dtype = _source_module._FP8_DTYPE
    # src[rms_norm_dynamic_per_token_quant.py:58]: assert result.dtype in [fp8_dtype, torch.int8]
    assert result.dtype in [_source_module._FP8_DTYPE, torch.int8]
    # src[rms_norm_dynamic_per_token_quant.py:59]: assert result.is_contiguous() and input.is_contiguous()
    assert result.is_contiguous() and input.is_contiguous()
    # src[rms_norm_dynamic_per_token_quant.py:61]: if scale_ub is not None:
    # src[rms_norm_dynamic_per_token_quant.py:62]:     assert result.dtype == fp8_dtype
    # src[rms_norm_dynamic_per_token_quant.py:63]:     assert scale_ub.dtype == torch.float32
    if scale_ub is not None:
        # src[rms_norm_dynamic_per_token_quant.py:62]: assert result.dtype == fp8_dtype
        assert result.dtype == fp8_dtype
        # src[rms_norm_dynamic_per_token_quant.py:63]: assert scale_ub.dtype == torch.float32
        assert scale_ub.dtype == torch.float32
    # src[rms_norm_dynamic_per_token_quant.py:65]: assert input.dtype == weight.dtype
    assert input.dtype == weight.dtype
    # src[rms_norm_dynamic_per_token_quant.py:66]: assert scale.shape[0] == num_tokens
    assert scale.shape[0] == num_tokens
    # src[rms_norm_dynamic_per_token_quant.py:67]: assert scale.dtype == torch.float32
    assert scale.dtype == torch.float32
    # src[rms_norm_dynamic_per_token_quant.py:69]: if residual is not None:
    # src[rms_norm_dynamic_per_token_quant.py:70]:     assert residual.dtype == input.dtype
    if residual is not None:
        # src[rms_norm_dynamic_per_token_quant.py:70]: assert residual.dtype == input.dtype
        assert residual.dtype == input.dtype
    # src[rms_norm_dynamic_per_token_quant.py:72]: quant_dtype = result.dtype
    quant_dtype = result.dtype
    # src[rms_norm_dynamic_per_token_quant.py:73]: qtype_traits_min: int | float
    qtype_traits_min: int | float
    # src[rms_norm_dynamic_per_token_quant.py:74]: qtype_traits_max: int | float
    qtype_traits_max: int | float
    # src[rms_norm_dynamic_per_token_quant.py:75]: if quant_dtype == torch.int8:
    # src[rms_norm_dynamic_per_token_quant.py:76]:     qtype_traits_min, qtype_traits_max = _INT8_MIN, _INT8_MAX
    # src[rms_norm_dynamic_per_token_quant.py:77]:     min_scaling_factor = _INT8_MIN_SCALING_FACTOR
    # src[rms_norm_dynamic_per_token_quant.py:75-80]: ...
    if quant_dtype == torch.int8:
        # src[rms_norm_dynamic_per_token_quant.py:76]: qtype_traits_min, qtype_traits_max = _INT8_MIN, _INT8_MAX
        qtype_traits_min, qtype_traits_max = (_INT8_MIN, _INT8_MAX)
        # src[rms_norm_dynamic_per_token_quant.py:77]: min_scaling_factor = _INT8_MIN_SCALING_FACTOR
        min_scaling_factor = _INT8_MIN_SCALING_FACTOR
    else:
        # src[rms_norm_dynamic_per_token_quant.py:79]: qtype_traits_min, qtype_traits_max = _FP8_MIN, _FP8_MAX
        qtype_traits_min, qtype_traits_max = (_source_module._FP8_MIN, _source_module._FP8_MAX)
        # src[rms_norm_dynamic_per_token_quant.py:80]: min_scaling_factor = _MIN_SCALING_FACTOR
        min_scaling_factor = _source_module._MIN_SCALING_FACTOR
    # src[rms_norm_dynamic_per_token_quant.py:85]: qtype_max = qtype_traits_max
    qtype_max = _source_module._FP8_MAX
    # src[rms_norm_dynamic_per_token_quant.py:87]: for tile_m in hl.tile(num_tokens, block_size=1):
    # src[rms_norm_dynamic_per_token_quant.py:88]:     rms = hl.zeros([tile_m], dtype=torch.float32)
    # src[rms_norm_dynamic_per_token_quant.py:89]:     for tile_n in hl.tile(hidden_size):
    # src[rms_norm_dynamic_per_token_quant.py:87-128]: ...
    _launcher(_helion_rms_norm_dynamic_per_token_quant, (num_tokens,), input, weight, scale, result, input.stride(0), input.stride(1), result.stride(0), result.stride(1), scale.stride(0), scale.stride(1), weight.stride(0), epsilon, qtype_traits_max, min_scaling_factor, qtype_traits_min, num_warps=8, num_stages=1)