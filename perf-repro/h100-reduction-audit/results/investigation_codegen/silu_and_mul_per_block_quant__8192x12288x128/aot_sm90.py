from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
from helion.runtime import default_launcher as _default_launcher

_BLOCK_SIZE_2 = tl.constexpr(128)
_BLOCK_SIZE_1 = tl.constexpr(8)

@triton.jit
def _helion_silu_and_mul_per_block_quant(input_1, scales, out, input_1_stride_0, input_1_stride_1, input_1_stride_2, out_stride_0, out_stride_1, out_stride_2, scales_stride_0, scales_stride_1):
    # src[silu_and_mul_per_block_quant.py:94]: for tile_m, tile_gn, tile_n in hl.tile(
    # src[silu_and_mul_per_block_quant.py:95]:     [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    # src[silu_and_mul_per_block_quant.py:96]: ):
    num_blocks_0 = tl.cdiv(128, _BLOCK_SIZE_2)
    num_blocks_1 = tl.cdiv(96, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0 % num_blocks_1
    pid_2 = tl.program_id(0) // (num_blocks_0 * num_blocks_1)
    offset_2 = pid_0 * _BLOCK_SIZE_2
    indices_2 = (offset_2 + tl.arange(0, _BLOCK_SIZE_2)).to(tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    offset_0 = pid_2
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    # src[silu_and_mul_per_block_quant.py:97]: x_a_blk = input[tile_m, tile_gn, tile_n].to(torch.float32)
    load = tl.load(input_1 + (indices_0[:, None, None] * input_1_stride_0 + indices_1[None, :, None] * input_1_stride_1 + indices_2[None, None, :] * input_1_stride_2), None, eviction_policy='evict_first')
    v_0 = tl.cast(load, tl.float32)
    # src[silu_and_mul_per_block_quant.py:100]: [tile_m, tile_gn.index + groups_per_row, tile_n],
    v_1 = tl.full([], 96, tl.int32)
    v_2 = indices_1 + v_1
    # src[silu_and_mul_per_block_quant.py:101]: extra_mask=(tile_gn.index + groups_per_row < 2 * groups_per_row)[
    v_3 = tl.full([], 96, tl.int32)
    v_4 = indices_1 + v_3
    v_5 = tl.full([], 192, tl.int32)
    v_6 = v_4 < v_5
    # src[silu_and_mul_per_block_quant.py:101]: extra_mask=(tile_gn.index + groups_per_row < 2 * groups_per_row)[
    # src[silu_and_mul_per_block_quant.py:102]:     None, :, None
    # src[silu_and_mul_per_block_quant.py:103]: ],
    subscript = v_6[None, :, None]
    # src[silu_and_mul_per_block_quant.py:98]: x_b_blk = hl.load(
    # src[silu_and_mul_per_block_quant.py:99]:     input,
    # src[silu_and_mul_per_block_quant.py:100]:     [tile_m, tile_gn.index + groups_per_row, tile_n],
    # src[silu_and_mul_per_block_quant.py:98-104]: ...
    load_1 = tl.load(input_1 + (indices_0[:, None, None] * input_1_stride_0 + (indices_1 + 96)[None, :, None] * input_1_stride_1 + indices_2[None, None, :] * input_1_stride_2), subscript, other=0)
    v_7 = tl.cast(load_1, tl.float32)
    # src[silu_and_mul_per_block_quant.py:105]: x_blk = x_a_blk * torch.sigmoid(x_a_blk) * x_b_blk
    v_8 = tl.cast(v_0, tl.float32)
    v_9 = tl.sigmoid(v_8)
    v_10 = v_0 * v_9
    v_11 = v_10 * v_7
    # src[silu_and_mul_per_block_quant.py:106]: s_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)
    v_12 = tl_math.abs(v_11)
    s_blk = tl.cast(tl.max(v_12, 2), tl.float32)
    # src[silu_and_mul_per_block_quant.py:111]: s_blk = s_blk * (1.0 / qtype_max)
    v_13 = tl.full([], 0.002232142857142857, tl.float32)
    v_14 = s_blk * v_13
    # src[silu_and_mul_per_block_quant.py:112]: s_blk = s_blk.clamp(min=min_scaling_factor)
    v_15 = tl.full([], 4.359654017857143e-06, tl.float32)
    v_16 = triton_helpers.maximum(v_14, v_15)
    # src[silu_and_mul_per_block_quant.py:114]: scales[tile_m, tile_gn] = s_blk
    tl.store(scales + (indices_0[:, None] * scales_stride_0 + indices_1[None, :] * scales_stride_1), v_16, None)
    # src[silu_and_mul_per_block_quant.py:118]: y_blk = x_blk / s_blk[:, :, None]
    subscript_1 = v_16[:, :, None]
    v_17 = v_11 / subscript_1
    # src[silu_and_mul_per_block_quant.py:120]: out[tile_m, tile_gn, tile_n] = y_blk.clamp(
    # src[silu_and_mul_per_block_quant.py:121]:     qtype_traits_min, qtype_traits_max
    # src[silu_and_mul_per_block_quant.py:122]: ).to(out.dtype)
    v_18 = tl.full([], -448.0, tl.float32)
    v_19 = triton_helpers.maximum(v_17, v_18)
    v_20 = tl.full([], 448.0, tl.float32)
    v_21 = triton_helpers.minimum(v_19, v_20)
    v_22 = tl.cast(v_21, tl.float8e4nv)
    tl.store(out + (indices_0[:, None, None] * out_stride_0 + indices_1[None, :, None] * out_stride_1 + indices_2[None, None, :] * out_stride_2), v_22, None)

def silu_and_mul_per_block_quant(out: torch.Tensor, input: torch.Tensor, scales: torch.Tensor, group_size: int, scale_ub: torch.Tensor | None=None, is_scale_transposed: bool=False, *, _launcher=_default_launcher):
    # src[silu_and_mul_per_block_quant.py:43]: assert input.ndim == 2
    assert input.ndim == 2
    # src[silu_and_mul_per_block_quant.py:44]: num_tokens, two_intermediate_size = input.shape
    num_tokens, two_intermediate_size = input.shape
    # src[silu_and_mul_per_block_quant.py:47]: assert two_intermediate_size % 2 == 0
    assert two_intermediate_size % 2 == 0
    # src[silu_and_mul_per_block_quant.py:48]: intermediate_size = two_intermediate_size // 2
    intermediate_size = two_intermediate_size // 2
    # src[silu_and_mul_per_block_quant.py:50]: assert out.shape[0] == num_tokens
    assert out.shape[0] == num_tokens
    # src[silu_and_mul_per_block_quant.py:51]: assert out.shape[1] == intermediate_size
    assert out.shape[1] == intermediate_size
    # src[silu_and_mul_per_block_quant.py:52]: fp8_dtype = torch.float8_e4m3fn
    fp8_dtype = torch.float8_e4m3fn
    # src[silu_and_mul_per_block_quant.py:53]: assert out.dtype in [fp8_dtype, torch.int8]
    assert out.dtype in [torch.float8_e4m3fn, torch.int8]
    # src[silu_and_mul_per_block_quant.py:55]: if scale_ub is not None:
    # src[silu_and_mul_per_block_quant.py:56]:     assert out.dtype == fp8_dtype
    # src[silu_and_mul_per_block_quant.py:57]:     assert scale_ub.dtype == torch.float32
    if scale_ub is not None:
        # src[silu_and_mul_per_block_quant.py:56]: assert out.dtype == fp8_dtype
        assert out.dtype == fp8_dtype
        # src[silu_and_mul_per_block_quant.py:57]: assert scale_ub.dtype == torch.float32
        assert scale_ub.dtype == torch.float32
    # src[silu_and_mul_per_block_quant.py:59]: assert scales.ndim == 2 and scales.dtype == torch.float32
    assert scales.ndim == 2 and scales.dtype == torch.float32
    # src[silu_and_mul_per_block_quant.py:61]: assert scales.shape[0] == num_tokens
    assert scales.shape[0] == num_tokens
    # src[silu_and_mul_per_block_quant.py:62]: groups_per_row = scales.shape[1]
    groups_per_row = scales.shape[1]
    # src[silu_and_mul_per_block_quant.py:64]: assert (
    # src[silu_and_mul_per_block_quant.py:65]:     intermediate_size % group_size == 0
    # src[silu_and_mul_per_block_quant.py:66]:     and intermediate_size // group_size == groups_per_row
    # src[silu_and_mul_per_block_quant.py:64-67]: ...
    assert intermediate_size % group_size == 0 and intermediate_size // group_size == groups_per_row
    # src[silu_and_mul_per_block_quant.py:69]: assert group_size in [64, 128]
    assert group_size in [64, 128]
    # src[silu_and_mul_per_block_quant.py:72]: assert input.stride()[-1] == 1
    assert input.stride()[-1] == 1
    # src[silu_and_mul_per_block_quant.py:73]: assert out.stride()[-1] == 1
    assert out.stride()[-1] == 1
    # src[silu_and_mul_per_block_quant.py:75]: quant_dtype = out.dtype
    quant_dtype = out.dtype
    # src[silu_and_mul_per_block_quant.py:76]: qtype_traits_min: int | float
    qtype_traits_min: int | float
    # src[silu_and_mul_per_block_quant.py:77]: qtype_traits_max: int | float
    qtype_traits_max: int | float
    # src[silu_and_mul_per_block_quant.py:82]: if quant_dtype == torch.int8:
    # src[silu_and_mul_per_block_quant.py:83]:     qtype_traits_min, qtype_traits_max = -128, 127
    # src[silu_and_mul_per_block_quant.py:84]:     min_scaling_factor = 1.1920928955078125e-07
    # src[silu_and_mul_per_block_quant.py:82-87]: ...
    if quant_dtype == torch.int8:
        # src[silu_and_mul_per_block_quant.py:83]: qtype_traits_min, qtype_traits_max = -128, 127
        qtype_traits_min, qtype_traits_max = (-128, 127)
        # src[silu_and_mul_per_block_quant.py:84]: min_scaling_factor = 1.1920928955078125e-07
        min_scaling_factor = 1.1920928955078125e-07
    else:
        # src[silu_and_mul_per_block_quant.py:86]: qtype_traits_min, qtype_traits_max = -448.0, 448.0
        qtype_traits_min, qtype_traits_max = (-448.0, 448.0)
        # src[silu_and_mul_per_block_quant.py:87]: min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)
    # src[silu_and_mul_per_block_quant.py:89]: qtype_max = float(qtype_traits_max)
    qtype_max = float(qtype_traits_max)
    # src[silu_and_mul_per_block_quant.py:91]: input = input.view(num_tokens, -1, group_size)  # noqa: A001
    input = input.view(num_tokens, -1, group_size)
    # src[silu_and_mul_per_block_quant.py:92]: out = out.view(num_tokens, -1, group_size)
    out = out.view(num_tokens, -1, group_size)
    # src[silu_and_mul_per_block_quant.py:94]: for tile_m, tile_gn, tile_n in hl.tile(
    # src[silu_and_mul_per_block_quant.py:95]:     [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    # src[silu_and_mul_per_block_quant.py:96]: ):
    _BLOCK_SIZE_2 = 128
    _BLOCK_SIZE_1 = 8
    # src[silu_and_mul_per_block_quant.py:94]: for tile_m, tile_gn, tile_n in hl.tile(
    # src[silu_and_mul_per_block_quant.py:95]:     [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    # src[silu_and_mul_per_block_quant.py:96]: ):
    # src[silu_and_mul_per_block_quant.py:94-122]: ...
    _launcher(_helion_silu_and_mul_per_block_quant, ((128 + _BLOCK_SIZE_2 - 1) // _BLOCK_SIZE_2 * ((96 + _BLOCK_SIZE_1 - 1) // _BLOCK_SIZE_1) * num_tokens,), input, scales, out, input.stride(0), input.stride(1), input.stride(2), out.stride(0), out.stride(1), out.stride(2), scales.stride(0), scales.stride(1), num_warps=4, num_stages=5)