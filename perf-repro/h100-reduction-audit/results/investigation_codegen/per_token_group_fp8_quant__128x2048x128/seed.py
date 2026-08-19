from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

_BLOCK_SIZE_2 = tl.constexpr(128)

@triton.jit
def _helion_per_token_group_fp8_quant(input_1, output_s, output_q, input_1_stride_0, input_1_stride_1, input_1_stride_2, output_q_stride_0, output_q_stride_1, output_q_stride_2, output_s_stride_0, output_s_stride_1, num_tokens, eps, fp8_max, scale_ue8m0, fp8_min):
    # src[per_token_group_fp8_quant.py:59]: for tile_m, tile_gn, tile_n in hl.tile(
    # src[per_token_group_fp8_quant.py:60]:     [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    # src[per_token_group_fp8_quant.py:61]: ):
    num_blocks_0 = num_tokens
    num_blocks_1 = 16
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0 % num_blocks_1
    pid_2 = tl.program_id(0) // (num_blocks_0 * num_blocks_1)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    offset_1 = pid_1
    indices_1 = offset_1 + tl.zeros([1], tl.int32)
    offset_2 = pid_2 * _BLOCK_SIZE_2
    indices_2 = (offset_2 + tl.arange(0, _BLOCK_SIZE_2)).to(tl.int32)
    # src[per_token_group_fp8_quant.py:62]: x_blk = input[tile_m, tile_gn, tile_n]
    x_blk = tl.load(input_1 + (indices_0[:, None, None] * input_1_stride_0 + indices_1[None, :, None] * input_1_stride_1 + indices_2[None, None, :] * input_1_stride_2), None, eviction_policy='evict_first')
    # src[per_token_group_fp8_quant.py:63]: y_s_blk = torch.clamp(torch.amax(torch.abs(x_blk), dim=-1), min=eps)
    v_0 = tl_math.abs(x_blk)
    amax = tl.cast(tl.max(v_0, 2), tl.bfloat16)
    v_1 = tl.cast(amax, tl.float32)
    v_2 = triton_helpers.maximum(v_1, eps)
    v_3 = tl.cast(v_2, tl.bfloat16)
    # src[per_token_group_fp8_quant.py:64]: y_s_blk = y_s_blk / fp8_max
    v_4 = tl.cast(fp8_max, tl.bfloat16)
    v_5 = v_3 / v_4
    # src[per_token_group_fp8_quant.py:66]: if scale_ue8m0:
    # src[per_token_group_fp8_quant.py:67]:     y_s_blk = torch.exp2(torch.ceil(torch.log2(y_s_blk)))
    if scale_ue8m0:
        v_5_copy = v_5
        v_5_copy_0 = v_5_copy
        # src[per_token_group_fp8_quant.py:67]: y_s_blk = torch.exp2(torch.ceil(torch.log2(y_s_blk)))
        v_6 = libdevice.log2(v_5_copy_0)
        v_7 = libdevice.ceil(v_6)
        v_5 = libdevice.exp2(v_7)
    else:
        pass
    # src[per_token_group_fp8_quant.py:69]: y_q_blk = torch.clamp(x_blk / y_s_blk[:, :, None], fp8_min, fp8_max).to(
    subscript = v_5[:, :, None]
    v_9 = x_blk / subscript
    v_10 = tl.cast(v_9, tl.float32)
    v_11 = triton_helpers.maximum(v_10, fp8_min)
    v_12 = triton_helpers.minimum(v_11, fp8_max)
    v_13 = tl.cast(v_12, tl.bfloat16)
    # src[per_token_group_fp8_quant.py:69]: y_q_blk = torch.clamp(x_blk / y_s_blk[:, :, None], fp8_min, fp8_max).to(
    # src[per_token_group_fp8_quant.py:70]:     output_q.dtype
    # src[per_token_group_fp8_quant.py:71]: )
    v_14 = tl.cast(v_13, tl.float8e4nv)
    # src[per_token_group_fp8_quant.py:73]: output_s[tile_m, tile_gn] = y_s_blk
    v_15 = tl.cast(v_5, tl.float32)
    tl.store(output_s + (indices_0[:, None] * output_s_stride_0 + indices_1[None, :] * output_s_stride_1), v_15, None)
    # src[per_token_group_fp8_quant.py:74]: output_q[tile_m, tile_gn, tile_n] = y_q_blk
    tl.store(output_q + (indices_0[:, None, None] * output_q_stride_0 + indices_1[None, :, None] * output_q_stride_1 + indices_2[None, None, :] * output_q_stride_2), v_14, None)

def per_token_group_fp8_quant(input: torch.Tensor, output_q: torch.Tensor, output_s: torch.Tensor, group_size: int, eps: float, fp8_min: float, fp8_max: float, scale_ue8m0: bool, dummy_is_scale_transposed: bool=False, dummy_is_tma_aligned: bool=False, *, _launcher=_default_launcher):
    # src[per_token_group_fp8_quant.py:47]: assert input.ndim == 2
    assert input.ndim == 2
    # src[per_token_group_fp8_quant.py:48]: num_tokens, hidden_size = input.shape
    num_tokens, hidden_size = input.shape
    # src[per_token_group_fp8_quant.py:52]: groups_per_row = output_s.shape[1]
    groups_per_row = output_s.shape[1]
    # src[per_token_group_fp8_quant.py:54]: assert hidden_size % group_size == 0 and hidden_size // group_size == groups_per_row
    assert hidden_size % group_size == 0 and hidden_size // group_size == groups_per_row
    # src[per_token_group_fp8_quant.py:55]: assert output_s.ndim == 2 and output_s.dtype == torch.float32
    assert output_s.ndim == 2 and output_s.dtype == torch.float32
    # src[per_token_group_fp8_quant.py:57]: input = input.view(num_tokens, -1, group_size)  # noqa: A001
    input = input.view(num_tokens, -1, group_size)
    # src[per_token_group_fp8_quant.py:58]: output_q = output_q.view(num_tokens, -1, group_size)
    output_q = output_q.view(num_tokens, -1, group_size)
    # src[per_token_group_fp8_quant.py:59]: for tile_m, tile_gn, tile_n in hl.tile(
    # src[per_token_group_fp8_quant.py:60]:     [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    # src[per_token_group_fp8_quant.py:61]: ):
    _BLOCK_SIZE_2 = 128
    # src[per_token_group_fp8_quant.py:59]: for tile_m, tile_gn, tile_n in hl.tile(
    # src[per_token_group_fp8_quant.py:60]:     [num_tokens, groups_per_row, group_size], block_size=[1, None, group_size]
    # src[per_token_group_fp8_quant.py:61]: ):
    # src[per_token_group_fp8_quant.py:59-74]: ...
    _launcher(_helion_per_token_group_fp8_quant, (num_tokens * 16 * ((128 + _BLOCK_SIZE_2 - 1) // _BLOCK_SIZE_2),), input, output_s, output_q, input.stride(0), input.stride(1), input.stride(2), output_q.stride(0), output_q.stride(1), output_q.stride(2), output_s.stride(0), output_s.stride(1), num_tokens, eps, fp8_max, scale_ue8m0, fp8_min, num_warps=4, num_stages=1)