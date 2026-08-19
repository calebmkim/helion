from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

_BLOCK_SIZE_0 = tl.constexpr(1)
_BLOCK_SIZE_1 = tl.constexpr(8192)
_BLOCK_SIZE_3 = tl.constexpr(128)
_BLOCK_SIZE_2 = tl.constexpr(64)

@triton.jit
def _helion_rms_norm_per_block_quant(input_1, residual, weight, scale_ub, scale, result, input_1_stride_0, input_1_stride_1, residual_stride_0, residual_stride_1, result_stride_0, result_stride_1, scale_stride_0, scale_stride_1, weight_stride_0, epsilon):
    # src[rms_norm_per_block_quant.py:96]: for tile_m in hl.tile(num_tokens, block_size=1):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    # src[rms_norm_per_block_quant.py:97]: rms = hl.zeros([tile_m], dtype=torch.float32)
    rms = tl.full([_BLOCK_SIZE_0], 0.0, tl.float32)
    # src[rms_norm_per_block_quant.py:98]: for tile_n in hl.tile(hidden_size):
    # src[rms_norm_per_block_quant.py:99]:     x_blk = input[tile_m, tile_n].to(torch.float32)
    # src[rms_norm_per_block_quant.py:100]:     if residual is not None:
    # src[rms_norm_per_block_quant.py:98-102]: ...
    for offset_1 in tl.range(0, 5120, _BLOCK_SIZE_1):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < 5120
        rms_copy = rms
        rms_copy_0 = rms_copy
        # src[rms_norm_per_block_quant.py:99]: x_blk = input[tile_m, tile_n].to(torch.float32)
        load = tl.load(input_1 + (indices_0[:, None] * input_1_stride_0 + indices_1[None, :] * input_1_stride_1), mask_1[None, :], other=0)
        v_0 = tl.cast(load, tl.float32)
        # src[rms_norm_per_block_quant.py:101]: x_blk = x_blk + residual[tile_m, tile_n]
        load_1 = tl.load(residual + (indices_0[:, None] * residual_stride_0 + indices_1[None, :] * residual_stride_1), mask_1[None, :], other=0)
        v_1 = tl.cast(load_1, tl.float32)
        v_2 = v_0 + v_1
        # src[rms_norm_per_block_quant.py:102]: rms = rms + x_blk.pow(2).sum(dim=-1)
        v_3 = v_2 * v_2
        sum_1 = tl.cast(tl.sum(v_3, 1), tl.float32)
        rms = rms_copy_0 + sum_1
    # src[rms_norm_per_block_quant.py:104]: rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)
    v_5 = tl.full([], 0.0001953125, tl.float32)
    v_6 = rms * v_5
    v_7 = v_6 + epsilon
    v_8 = libdevice.rsqrt(v_7)
    # src[rms_norm_per_block_quant.py:106]: m_idx = tile_m.begin + hl.arange(tile_m.block_size)
    iota = tl.arange(0, _BLOCK_SIZE_0)
    v_9 = tl.cast(offset_0, tl.int32)
    v_10 = iota + v_9
    # src[rms_norm_per_block_quant.py:107]: m_blk = m_idx[:, None, None]
    m_blk = v_10[:, None, None]
    # src[rms_norm_per_block_quant.py:108]: for tile_gn, tile_n in hl.tile(
    # src[rms_norm_per_block_quant.py:109]:     [groups_per_row, group_size], block_size=[None, group_size]
    # src[rms_norm_per_block_quant.py:110]: ):
    # src[rms_norm_per_block_quant.py:108-148]: ...
    tl.debug_barrier()
    for offset_2 in tl.range(0, 40, _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < 40
        for offset_3 in tl.range(0, 128, _BLOCK_SIZE_3):
            indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
            m_blk_copy = m_blk
            v_8_copy = v_8
            m_blk_copy_0 = m_blk_copy
            v_8_copy_0 = v_8_copy
            # src[rms_norm_per_block_quant.py:113]: n_idx = gn_idx[:, None] * group_size + n_offset[None, :]
            subscript = indices_2[:, None]
            v_11 = tl.full([], 128, tl.int32)
            v_12 = tl.cast(subscript * v_11, tl.int32)
            subscript_1 = indices_3[None, :]
            v_13 = v_12 + subscript_1
            # src[rms_norm_per_block_quant.py:114]: n_blk = n_idx[None, :, :]
            n_blk = v_13[None, :, :]
            # src[rms_norm_per_block_quant.py:115]: mask = (gn_idx < groups_per_row)[None, :, None]
            v_14 = tl.full([], 40, tl.int32)
            v_15 = indices_2 < v_14
            mask = v_15[None, :, None]
            # src[rms_norm_per_block_quant.py:117]: x_blk = hl.load(input, [m_blk, n_blk], extra_mask=mask).to(
            load_2 = tl.load(input_1 + (m_blk_copy_0 * input_1_stride_0 + n_blk * input_1_stride_1), mask, other=0)
            # src[rms_norm_per_block_quant.py:117]: x_blk = hl.load(input, [m_blk, n_blk], extra_mask=mask).to(
            # src[rms_norm_per_block_quant.py:118]:     dtype=torch.float32
            # src[rms_norm_per_block_quant.py:119]: )
            v_16 = tl.cast(load_2, tl.float32)
            # src[rms_norm_per_block_quant.py:121]: r_blk = hl.load(residual, [m_blk, n_blk], extra_mask=mask)
            r_blk = tl.load(residual + (m_blk_copy_0 * residual_stride_0 + n_blk * residual_stride_1), mask, other=0)
            # src[rms_norm_per_block_quant.py:122]: x_blk = x_blk + r_blk
            v_17 = tl.cast(r_blk, tl.float32)
            v_18 = v_16 + v_17
            # src[rms_norm_per_block_quant.py:124]: w_blk = hl.load(weight, [n_blk], extra_mask=mask)
            w_blk = tl.load(weight + n_blk * weight_stride_0, mask_2[None, :, None] & mask, other=0)
            # src[rms_norm_per_block_quant.py:125]: x_norm_blk = (x_blk * rms[:, None, None]).to(input.dtype) * w_blk
            subscript_4 = v_8_copy_0[:, None, None]
            v_19 = v_18 * subscript_4
            v_20 = tl.cast(v_19, tl.bfloat16)
            v_21 = v_20 * w_blk
            # src[rms_norm_per_block_quant.py:126]: s_blk = torch.amax(torch.abs(x_norm_blk), dim=-1).to(torch.float32)
            v_22 = tl_math.abs(v_21)
            _mask_to = tl.where(tl.broadcast_to(mask_2[None, :, None], [_BLOCK_SIZE_0, _BLOCK_SIZE_2, _BLOCK_SIZE_3]), v_22, tl.full([], float('-inf'), tl.bfloat16))
            amax = tl.cast(tl.max(_mask_to, 2), tl.bfloat16)
            v_23 = tl.cast(amax, tl.float32)
            # src[rms_norm_per_block_quant.py:129]: scale_ub_s = hl.load(scale_ub, [])
            scale_ub_s = tl.load(scale_ub + tl.zeros([], tl.int32), None)
            # src[rms_norm_per_block_quant.py:130]: s_blk = s_blk.clamp(max=scale_ub_s)
            v_24 = triton_helpers.minimum(v_23, scale_ub_s)
            # src[rms_norm_per_block_quant.py:132]: s_blk = s_blk * (1.0 / qtype_max)
            v_25 = tl.full([], 0.002232142857142857, tl.float32)
            v_26 = v_24 * v_25
            # src[rms_norm_per_block_quant.py:133]: s_blk = s_blk.clamp(min=min_scaling_factor)
            v_27 = tl.full([], 4.359654017857143e-06, tl.float32)
            v_28 = triton_helpers.maximum(v_26, v_27)
            # src[rms_norm_per_block_quant.py:135]: scale[tile_m, tile_gn] = s_blk
            tl.store(scale + (indices_0[:, None] * scale_stride_0 + indices_2[None, :] * scale_stride_1), v_28, mask_2[None, :])
            # src[rms_norm_per_block_quant.py:140]: y_blk = x_norm_blk / s_blk[:, :, None]
            subscript_5 = v_28[:, :, None]
            v_29 = tl.cast(v_21, tl.float32)
            v_30 = v_29 / subscript_5
            # src[rms_norm_per_block_quant.py:142]: y_blk = y_blk.clamp(qtype_traits_min, qtype_traits_max).to(result.dtype)
            v_31 = tl.full([], -448.0, tl.float32)
            v_32 = triton_helpers.maximum(v_30, v_31)
            v_33 = tl.full([], 448.0, tl.float32)
            v_34 = triton_helpers.minimum(v_32, v_33)
            v_35 = tl.cast(v_34, tl.float8e4nv)
            # src[rms_norm_per_block_quant.py:143]: hl.store(result, [m_blk, n_blk], y_blk, extra_mask=mask)
            tl.store(result + (m_blk_copy_0 * result_stride_0 + n_blk * result_stride_1), v_35, mask)
            # src[rms_norm_per_block_quant.py:147]: residual, [m_blk, n_blk], x_blk.to(residual.dtype), extra_mask=mask
            v_36 = tl.cast(v_18, tl.bfloat16)
            # src[rms_norm_per_block_quant.py:146]: hl.store(
            # src[rms_norm_per_block_quant.py:147]:     residual, [m_blk, n_blk], x_blk.to(residual.dtype), extra_mask=mask
            # src[rms_norm_per_block_quant.py:148]: )
            tl.store(residual + (m_blk_copy_0 * residual_stride_0 + n_blk * residual_stride_1), v_36, mask)

def rms_norm_per_block_quant(result: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, epsilon: float, scale_ub: torch.Tensor | None, residual: torch.Tensor | None, group_size: int, is_scale_transposed: bool, *, _launcher=_default_launcher):
    # src[rms_norm_per_block_quant.py:54]: assert input.ndim == 2
    assert input.ndim == 2
    # src[rms_norm_per_block_quant.py:55]: num_tokens, hidden_size = input.shape
    num_tokens, hidden_size = input.shape
    # src[rms_norm_per_block_quant.py:59]: groups_per_row = scale.shape[1]
    groups_per_row = scale.shape[1]
    # src[rms_norm_per_block_quant.py:61]: assert hidden_size % group_size == 0 and hidden_size // group_size == groups_per_row
    assert hidden_size % group_size == 0 and hidden_size // group_size == groups_per_row
    # src[rms_norm_per_block_quant.py:62]: assert scale.shape[0] == num_tokens
    assert scale.shape[0] == num_tokens
    # src[rms_norm_per_block_quant.py:63]: assert scale.dtype == torch.float32
    assert scale.dtype == torch.float32
    # src[rms_norm_per_block_quant.py:64]: if scale.stride(1) > 1:
    # src[rms_norm_per_block_quant.py:65]:     assert is_scale_transposed
    if scale.stride(1) > 1:
        # src[rms_norm_per_block_quant.py:65]: assert is_scale_transposed
        assert is_scale_transposed
    # src[rms_norm_per_block_quant.py:67]: fp8_dtype = torch.float8_e4m3fn
    fp8_dtype = torch.float8_e4m3fn
    # src[rms_norm_per_block_quant.py:68]: assert result.dtype in [fp8_dtype, torch.int8]
    assert result.dtype in [torch.float8_e4m3fn, torch.int8]
    # src[rms_norm_per_block_quant.py:69]: assert result.is_contiguous() and input.is_contiguous()
    assert result.is_contiguous() and input.is_contiguous()
    # src[rms_norm_per_block_quant.py:71]: if scale_ub is not None:
    # src[rms_norm_per_block_quant.py:72]:     assert result.dtype == fp8_dtype
    # src[rms_norm_per_block_quant.py:73]:     assert scale_ub.dtype == torch.float32
    if scale_ub is not None:
        # src[rms_norm_per_block_quant.py:72]: assert result.dtype == fp8_dtype
        assert result.dtype == torch.float8_e4m3fn
        # src[rms_norm_per_block_quant.py:73]: assert scale_ub.dtype == torch.float32
        assert scale_ub.dtype == torch.float32
    # src[rms_norm_per_block_quant.py:75]: assert input.dtype == weight.dtype
    assert input.dtype == weight.dtype
    # src[rms_norm_per_block_quant.py:77]: if residual is not None:
    # src[rms_norm_per_block_quant.py:78]:     assert residual.dtype == input.dtype
    if residual is not None:
        # src[rms_norm_per_block_quant.py:78]: assert residual.dtype == input.dtype
        assert residual.dtype == input.dtype
    # src[rms_norm_per_block_quant.py:80]: assert group_size in [64, 128]
    assert group_size in [64, 128]
    # src[rms_norm_per_block_quant.py:82]: quant_dtype = result.dtype
    quant_dtype = result.dtype
    # src[rms_norm_per_block_quant.py:83]: qtype_traits_min: int | float
    qtype_traits_min: int | float
    # src[rms_norm_per_block_quant.py:84]: qtype_traits_max: int | float
    qtype_traits_max: int | float
    # src[rms_norm_per_block_quant.py:85]: if quant_dtype == torch.int8:
    # src[rms_norm_per_block_quant.py:86]:     # torch.iinfo(torch.int8) min/max, torch.finfo(torch.float32).eps.
    # src[rms_norm_per_block_quant.py:87]:     qtype_traits_min, qtype_traits_max = -128, 127
    # src[rms_norm_per_block_quant.py:85-92]: ...
    if quant_dtype == torch.int8:
        # src[rms_norm_per_block_quant.py:87]: qtype_traits_min, qtype_traits_max = -128, 127
        qtype_traits_min, qtype_traits_max = (-128, 127)
        # src[rms_norm_per_block_quant.py:88]: min_scaling_factor = 1.1920928955078125e-07
        min_scaling_factor = 1.1920928955078125e-07
    else:
        # src[rms_norm_per_block_quant.py:91]: qtype_traits_min, qtype_traits_max = -448.0, 448.0
        qtype_traits_min, qtype_traits_max = (-448.0, 448.0)
        # src[rms_norm_per_block_quant.py:92]: min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)
    # src[rms_norm_per_block_quant.py:96]: for tile_m in hl.tile(num_tokens, block_size=1):
    # src[rms_norm_per_block_quant.py:97]:     rms = hl.zeros([tile_m], dtype=torch.float32)
    # src[rms_norm_per_block_quant.py:98]:     for tile_n in hl.tile(hidden_size):
    # src[rms_norm_per_block_quant.py:96-148]: ...
    _launcher(_helion_rms_norm_per_block_quant, (num_tokens,), input, residual, weight, scale_ub, scale, result, input.stride(0), input.stride(1), residual.stride(0), residual.stride(1), result.stride(0), result.stride(1), scale.stride(0), scale.stride(1), weight.stride(0), epsilon, num_warps=16, num_stages=1)