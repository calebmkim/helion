from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
from helion.runtime import default_launcher as _default_launcher

_BLOCK_SIZE_0 = tl.constexpr(1)
_BLOCK_SIZE_1 = tl.constexpr(8192)
_BLOCK_SIZE_2 = tl.constexpr(1024)

@triton.jit
def _helion_dynamic_per_token_scaled_fp8_quant(input_1, scale, result, input_1_stride_0, input_1_stride_1, result_stride_0, result_stride_1, scale_stride_0, scale_stride_1):
    # src[dynamic_per_token_scaled_fp8_quant.py:54]: for tile_m in hl.tile(num_tokens, block_size=1):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    # src[dynamic_per_token_scaled_fp8_quant.py:55]: s_blk = hl.zeros([tile_m], dtype=torch.float32)
    s_blk = tl.full([_BLOCK_SIZE_0], 0.0, tl.float32)
    # src[dynamic_per_token_scaled_fp8_quant.py:56]: for tile_n in hl.tile(hidden_size):
    # src[dynamic_per_token_scaled_fp8_quant.py:57]:     x_blk = input[tile_m, tile_n].to(dtype=torch.float32)
    # src[dynamic_per_token_scaled_fp8_quant.py:58]:     tmp_blk = torch.amax(torch.abs(x_blk), dim=-1)
    # src[dynamic_per_token_scaled_fp8_quant.py:56-59]: ...
    for offset_1 in tl.range(0, 5120, _BLOCK_SIZE_1, disallow_acc_multi_buffer=True, flatten=False):
        indices_1 = offset_1 + tl.arange(0, _BLOCK_SIZE_1).to(tl.int32)
        mask_1 = indices_1 < 5120
        s_blk_copy = s_blk
        s_blk_copy_0 = s_blk_copy
        # src[dynamic_per_token_scaled_fp8_quant.py:57]: x_blk = input[tile_m, tile_n].to(dtype=torch.float32)
        load = tl.load(input_1 + (indices_0[:, None] * input_1_stride_0 + indices_1[None, :] * input_1_stride_1), mask_1[None, :], other=0)
        v_0 = tl.cast(load, tl.float32)
        # src[dynamic_per_token_scaled_fp8_quant.py:58]: tmp_blk = torch.amax(torch.abs(x_blk), dim=-1)
        v_1 = tl_math.abs(v_0)
        _mask_to = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _BLOCK_SIZE_1]), v_1, tl.full([], float('-inf'), tl.float32))
        tmp_blk = tl.cast(tl.max(_mask_to, 1), tl.float32)
        # src[dynamic_per_token_scaled_fp8_quant.py:59]: s_blk = torch.maximum(s_blk, tmp_blk)
        s_blk = triton_helpers.maximum(s_blk_copy_0, tmp_blk)
    # src[dynamic_per_token_scaled_fp8_quant.py:64]: s_blk = s_blk * (1.0 / fp8_max)
    v_3 = tl.full([], 0.002232142857142857, tl.float32)
    v_4 = s_blk * v_3
    # src[dynamic_per_token_scaled_fp8_quant.py:65]: s_blk = s_blk.clamp(min=min_scaling_factor)
    v_5 = tl.full([], 4.359654017857143e-06, tl.float32)
    v_6 = triton_helpers.maximum(v_4, v_5)
    # src[dynamic_per_token_scaled_fp8_quant.py:66]: scale[tile_m, 0] = s_blk
    tl.store(scale + (indices_0 * scale_stride_0 + 0 * scale_stride_1), v_6, None)
    # src[dynamic_per_token_scaled_fp8_quant.py:68]: for tile_n in hl.tile(hidden_size):
    # src[dynamic_per_token_scaled_fp8_quant.py:69]:     x_blk = input[tile_m, tile_n].to(torch.float32)
    # src[dynamic_per_token_scaled_fp8_quant.py:70]:     y_blk = x_blk * (1.0 / s_blk[:, None])
    # src[dynamic_per_token_scaled_fp8_quant.py:68-72]: ...
    tl.debug_barrier()
    for offset_2 in tl.range(0, 5120, _BLOCK_SIZE_2, disallow_acc_multi_buffer=False, flatten=False):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        v_6_copy = v_6
        v_6_copy_0 = v_6_copy
        # src[dynamic_per_token_scaled_fp8_quant.py:69]: x_blk = input[tile_m, tile_n].to(torch.float32)
        load_1 = tl.load(input_1 + (indices_0[:, None] * input_1_stride_0 + indices_2[None, :] * input_1_stride_1), None)
        v_7 = tl.cast(load_1, tl.float32)
        # src[dynamic_per_token_scaled_fp8_quant.py:70]: y_blk = x_blk * (1.0 / s_blk[:, None])
        subscript = v_6_copy_0[:, None]
        v_8 = tl.full([], 1.0, tl.float32)
        v_9 = v_8 / subscript
        v_10 = tl.full([], 1.0, tl.float32)
        v_11 = v_9 * v_10
        v_12 = v_7 * v_11
        # src[dynamic_per_token_scaled_fp8_quant.py:72]: result[tile_m, tile_n] = y_blk.clamp(fp8_min, fp8_max).to(result.dtype)
        v_13 = tl.full([], -448.0, tl.float32)
        v_14 = triton_helpers.maximum(v_12, v_13)
        v_15 = tl.full([], 448.0, tl.float32)
        v_16 = triton_helpers.minimum(v_14, v_15)
        v_17 = tl.cast(v_16, tl.float8e4nv)
        tl.store(result + (indices_0[:, None] * result_stride_0 + indices_2[None, :] * result_stride_1), v_17, None)

def dynamic_per_token_scaled_fp8_quant(result: torch.Tensor, input: torch.Tensor, scale: torch.Tensor, scale_ub: torch.Tensor | None=None, *, _launcher=_default_launcher):
    # src[dynamic_per_token_scaled_fp8_quant.py:40]: assert input.ndim == 2
    assert input.ndim == 2
    # src[dynamic_per_token_scaled_fp8_quant.py:41]: num_tokens, hidden_size = input.shape
    num_tokens, hidden_size = input.shape
    # src[dynamic_per_token_scaled_fp8_quant.py:44]: assert result.shape == input.shape
    assert result.shape == input.shape
    # src[dynamic_per_token_scaled_fp8_quant.py:45]: assert scale.shape[0] == num_tokens
    assert scale.shape[0] == num_tokens
    # src[dynamic_per_token_scaled_fp8_quant.py:46]: assert scale.dtype == torch.float32
    assert scale.dtype == torch.float32
    # src[dynamic_per_token_scaled_fp8_quant.py:47]: assert input.stride()[-1] == 1
    assert input.stride()[-1] == 1
    # src[dynamic_per_token_scaled_fp8_quant.py:48]: assert result.stride()[-1] == 1
    assert result.stride()[-1] == 1
    # src[dynamic_per_token_scaled_fp8_quant.py:51]: fp8_min, fp8_max = -448.0, 448.0
    fp8_min, fp8_max = (-448.0, 448.0)
    # src[dynamic_per_token_scaled_fp8_quant.py:54]: for tile_m in hl.tile(num_tokens, block_size=1):
    # src[dynamic_per_token_scaled_fp8_quant.py:55]:     s_blk = hl.zeros([tile_m], dtype=torch.float32)
    # src[dynamic_per_token_scaled_fp8_quant.py:56]:     for tile_n in hl.tile(hidden_size):
    # src[dynamic_per_token_scaled_fp8_quant.py:54-72]: ...
    _launcher(_helion_dynamic_per_token_scaled_fp8_quant, (num_tokens,), input, scale, result, input.stride(0), input.stride(1), result.stride(0), result.stride(1), scale.stride(0), scale.stride(1), num_warps=8, num_stages=6)