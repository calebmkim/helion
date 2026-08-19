from __future__ import annotations

import torch
import helion
import triton
import triton.language as tl
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

_BLOCK_SIZE_2 = tl.constexpr(128)
_BLOCK_SIZE_1 = tl.constexpr(2)

@triton.jit
def _helion_fused_qk_norm_rope(qkv, q_weight, k_weight, position_ids, cos_sin_cache, cos_sin_cache_stride_0, cos_sin_cache_stride_1, k_weight_stride_0, position_ids_stride_0, q_weight_stride_0, qkv_stride_0, qkv_stride_1, qkv_stride_2, num_tokens, eps, is_neox, _NUM_SM: tl.constexpr):
    # src[fused_qk_norm_rope.py:86]: for tile_m, tile_gn, tile_n in hl.tile(
    # src[fused_qk_norm_rope.py:87]:     [num_tokens, qk_heads, head_dim], block_size=[1, None, head_dim]
    # src[fused_qk_norm_rope.py:88]: ):
    # src[fused_qk_norm_rope.py:86-121]: ...
    total_pids = tl.cdiv(128, _BLOCK_SIZE_2) * tl.cdiv(72, _BLOCK_SIZE_1) * num_tokens
    block_size = tl.cdiv(total_pids, _NUM_SM * 128)
    start_pid = tl.program_id(0) * block_size
    end_pid = start_pid + block_size
    if end_pid > total_pids:
        end_pid = total_pids
    for virtual_pid in tl.range(start_pid, end_pid, disallow_acc_multi_buffer=True):
        # src[fused_qk_norm_rope.py:86]: for tile_m, tile_gn, tile_n in hl.tile(
        # src[fused_qk_norm_rope.py:87]:     [num_tokens, qk_heads, head_dim], block_size=[1, None, head_dim]
        # src[fused_qk_norm_rope.py:88]: ):
        num_blocks_0 = tl.cdiv(128, _BLOCK_SIZE_2)
        num_blocks_1 = tl.cdiv(72, _BLOCK_SIZE_1)
        num_pid_m = tl.cdiv(128, _BLOCK_SIZE_2)
        num_pid_n = tl.cdiv(72, _BLOCK_SIZE_1)
        inner_2d_size = num_pid_m * num_pid_n
        inner_2d_pid = virtual_pid % inner_2d_size
        num_pid_in_group = 2 * num_pid_n
        group_id = inner_2d_pid // num_pid_in_group
        first_pid_m = group_id * 2
        group_size_m = min(num_pid_m - first_pid_m, 2)
        pid_0 = first_pid_m + inner_2d_pid % num_pid_in_group % group_size_m
        pid_1 = inner_2d_pid % num_pid_in_group // group_size_m
        pid_2 = virtual_pid // (num_blocks_0 * num_blocks_1)
        offset_2 = pid_0 * _BLOCK_SIZE_2
        indices_2 = (offset_2 + tl.arange(0, _BLOCK_SIZE_2)).to(tl.int32)
        offset_1 = pid_1 * _BLOCK_SIZE_1
        indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
        offset_0 = pid_2
        indices_0 = offset_0 + tl.zeros([1], tl.int32)
        # src[fused_qk_norm_rope.py:89]: x_blk = qkv[tile_m, tile_gn, tile_n].to(dtype=torch.float32)
        load = tl.load(qkv + (indices_0[:, None, None] * qkv_stride_0 + indices_1[None, :, None] * qkv_stride_1 + indices_2[None, None, :] * qkv_stride_2), None)
        v_0 = tl.cast(load, tl.float32)
        # src[fused_qk_norm_rope.py:91]: rms = x_blk.pow(2).sum(dim=-1)
        v_1 = v_0 * v_0
        rms = tl.cast(tl.sum(v_1, 2), tl.float32)
        # src[fused_qk_norm_rope.py:92]: rms = torch.rsqrt(rms * (1.0 / head_dim) + eps)
        v_2 = tl.full([], 0.0078125, tl.float32)
        v_3 = rms * v_2
        v_4 = v_3 + eps
        v_5 = libdevice.rsqrt(v_4)
        # src[fused_qk_norm_rope.py:94]: use_q_weight = (tile_gn.index < num_heads_q)[None, :, None]
        v_6 = tl.full([], 64, tl.int32)
        v_7 = indices_1 < v_6
        use_q_weight = v_7[None, :, None]
        # src[fused_qk_norm_rope.py:96]: use_q_weight, q_weight[None, None, tile_n], k_weight[None, None, tile_n]
        load_1 = tl.load(q_weight + indices_2[None, None, :] * q_weight_stride_0, None, eviction_policy='evict_first')
        load_2 = tl.load(k_weight + indices_2[None, None, :] * k_weight_stride_0, None, eviction_policy='evict_last')
        # src[fused_qk_norm_rope.py:95]: w_blk = torch.where(
        # src[fused_qk_norm_rope.py:96]:     use_q_weight, q_weight[None, None, tile_n], k_weight[None, None, tile_n]
        # src[fused_qk_norm_rope.py:97]: )
        w_blk = tl.where(use_q_weight, load_1, load_2)
        # src[fused_qk_norm_rope.py:99]: x_blk = (x_blk * rms[:, :, None]).to(qkv.dtype) * w_blk
        subscript_1 = v_5[:, :, None]
        v_8 = v_0 * subscript_1
        v_9 = tl.cast(v_8, tl.bfloat16)
        v_10 = v_9 * w_blk
        # src[fused_qk_norm_rope.py:101]: qkv[tile_m, tile_gn, tile_n] = x_blk
        tl.store(qkv + (indices_0[:, None, None] * qkv_stride_0 + indices_1[None, :, None] * qkv_stride_1 + indices_2[None, None, :] * qkv_stride_2), v_10, None)
        # src[fused_qk_norm_rope.py:103]: pos_id = position_ids[tile_m]
        pos_id = tl.load(position_ids + indices_0 * position_ids_stride_0, None)
        # src[fused_qk_norm_rope.py:104]: cos_blk = cos_sin_cache[pos_id, hl.arange(embed_dim)]
        iota = tl.arange(0, 64)
        cos_blk = tl.load(cos_sin_cache + (pos_id[:, None] * cos_sin_cache_stride_0 + iota[None, :] * cos_sin_cache_stride_1), None)
        # src[fused_qk_norm_rope.py:105]: sin_blk = cos_sin_cache[pos_id, hl.arange(embed_dim) + embed_dim]
        iota_1 = tl.arange(0, 64)
        v_11 = tl.full([], 64, tl.int32)
        v_12 = iota_1 + v_11
        sin_blk = tl.load(cos_sin_cache + (pos_id[:, None] * cos_sin_cache_stride_0 + v_12[None, :] * cos_sin_cache_stride_1), None)
        # src[fused_qk_norm_rope.py:107]: if is_neox:
        # src[fused_qk_norm_rope.py:108]:     x1_offset = hl.arange(embed_dim)
        # src[fused_qk_norm_rope.py:109]:     x2_offset = x1_offset + embed_dim
        # src[fused_qk_norm_rope.py:107-112]: ...
        if is_neox:
            # src[fused_qk_norm_rope.py:108]: x1_offset = hl.arange(embed_dim)
            x1_offset = tl.arange(0, 64)
            # src[fused_qk_norm_rope.py:109]: x2_offset = x1_offset + embed_dim
            v_13 = tl.full([], 64, tl.int32)
            v_14 = x1_offset + v_13
        else:
            # src[fused_qk_norm_rope.py:111]: x1_offset = hl.arange(embed_dim) * 2
            iota_2 = tl.arange(0, 64)
            v_15 = tl.full([], 2, tl.int32)
            x1_offset = tl.cast(iota_2 * v_15, tl.int32)
            # src[fused_qk_norm_rope.py:112]: x2_offset = x1_offset + 1
            v_17 = tl.full([], 1, tl.int32)
            v_14 = x1_offset + v_17
        # src[fused_qk_norm_rope.py:114]: x1_blk = qkv[tile_m, tile_gn, x1_offset]
        tl.debug_barrier()
        x1_blk = tl.load(qkv + (indices_0[:, None, None] * qkv_stride_0 + indices_1[None, :, None] * qkv_stride_1 + x1_offset[None, None, :] * qkv_stride_2), None)
        # src[fused_qk_norm_rope.py:115]: x2_blk = qkv[tile_m, tile_gn, x2_offset]
        x2_blk = tl.load(qkv + (indices_0[:, None, None] * qkv_stride_0 + indices_1[None, :, None] * qkv_stride_1 + v_14[None, None, :] * qkv_stride_2), None, eviction_policy='evict_first')
        # src[fused_qk_norm_rope.py:117]: o1_blk = x1_blk * cos_blk[:, None, :] - x2_blk * sin_blk[:, None, :]
        subscript_2 = cos_blk[:, None, :]
        v_19 = x1_blk * subscript_2
        subscript_3 = sin_blk[:, None, :]
        v_20 = x2_blk * subscript_3
        v_21 = v_19 - v_20
        # src[fused_qk_norm_rope.py:118]: o2_blk = x2_blk * cos_blk[:, None, :] + x1_blk * sin_blk[:, None, :]
        subscript_4 = cos_blk[:, None, :]
        v_22 = x2_blk * subscript_4
        subscript_5 = sin_blk[:, None, :]
        v_23 = x1_blk * subscript_5
        v_24 = v_22 + v_23
        # src[fused_qk_norm_rope.py:120]: qkv[tile_m, tile_gn, x1_offset] = o1_blk
        tl.store(qkv + (indices_0[:, None, None] * qkv_stride_0 + indices_1[None, :, None] * qkv_stride_1 + x1_offset[None, None, :] * qkv_stride_2), v_21, None)
        # src[fused_qk_norm_rope.py:121]: qkv[tile_m, tile_gn, x2_offset] = o2_blk
        tl.store(qkv + (indices_0[:, None, None] * qkv_stride_0 + indices_1[None, :, None] * qkv_stride_1 + v_14[None, None, :] * qkv_stride_2), v_24, None)

def fused_qk_norm_rope(qkv: torch.Tensor, num_heads_q: int, num_heads_k: int, num_heads_v: int, head_dim: int, eps: float, q_weight: torch.Tensor, k_weight: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool, position_ids: torch.Tensor, forced_token_heads_per_warp: int=-1, *, _launcher=_default_launcher):
    # src[fused_qk_norm_rope.py:46]: assert qkv.ndim == 2
    assert qkv.ndim == 2
    # src[fused_qk_norm_rope.py:47]: num_tokens = qkv.shape[0]
    num_tokens = qkv.shape[0]
    # src[fused_qk_norm_rope.py:48]: total_heads = num_heads_q + num_heads_k + num_heads_v
    total_heads = num_heads_q + num_heads_k + num_heads_v
    # src[fused_qk_norm_rope.py:49]: assert qkv.shape[1] == total_heads * head_dim
    assert qkv.shape[1] == total_heads * head_dim
    # src[fused_qk_norm_rope.py:52]: assert cos_sin_cache.ndim == 2
    assert cos_sin_cache.ndim == 2
    # src[fused_qk_norm_rope.py:53]: max_position, rotary_dim = cos_sin_cache.shape
    max_position, rotary_dim = cos_sin_cache.shape
    # src[fused_qk_norm_rope.py:56]: assert rotary_dim % 2 == 0
    assert rotary_dim % 2 == 0
    # src[fused_qk_norm_rope.py:57]: assert rotary_dim <= head_dim
    assert rotary_dim <= head_dim
    # src[fused_qk_norm_rope.py:65]: assert position_ids.ndim == 1 and position_ids.shape[0] == num_tokens
    assert position_ids.ndim == 1 and position_ids.shape[0] == num_tokens
    # src[fused_qk_norm_rope.py:68]: assert q_weight.ndim == 1 and q_weight.shape[0] == head_dim
    assert q_weight.ndim == 1 and q_weight.shape[0] == head_dim
    # src[fused_qk_norm_rope.py:70]: assert k_weight.ndim == 1 and k_weight.shape[0] == head_dim
    assert k_weight.ndim == 1 and k_weight.shape[0] == head_dim
    # src[fused_qk_norm_rope.py:73]: assert qkv.dtype == q_weight.dtype and q_weight.dtype == k_weight.dtype
    assert qkv.dtype == q_weight.dtype and q_weight.dtype == k_weight.dtype
    # src[fused_qk_norm_rope.py:74]: assert position_ids.dtype == torch.int64
    assert position_ids.dtype == torch.int64
    # src[fused_qk_norm_rope.py:76]: assert qkv.is_contiguous()
    assert qkv.is_contiguous()
    # src[fused_qk_norm_rope.py:77]: assert position_ids.is_contiguous()
    assert position_ids.is_contiguous()
    # src[fused_qk_norm_rope.py:78]: assert q_weight.is_contiguous()
    assert q_weight.is_contiguous()
    # src[fused_qk_norm_rope.py:79]: assert k_weight.is_contiguous()
    assert k_weight.is_contiguous()
    # src[fused_qk_norm_rope.py:80]: assert cos_sin_cache.is_contiguous()
    assert cos_sin_cache.is_contiguous()
    # src[fused_qk_norm_rope.py:84]: qkv = qkv.view(num_tokens, -1, head_dim)
    qkv = qkv.view(num_tokens, -1, head_dim)
    # src[fused_qk_norm_rope.py:86]: for tile_m, tile_gn, tile_n in hl.tile(
    # src[fused_qk_norm_rope.py:87]:     [num_tokens, qk_heads, head_dim], block_size=[1, None, head_dim]
    # src[fused_qk_norm_rope.py:88]: ):
    _NUM_SM = helion.runtime.get_num_sm(qkv.device)
    # src[fused_qk_norm_rope.py:86]: for tile_m, tile_gn, tile_n in hl.tile(
    # src[fused_qk_norm_rope.py:87]:     [num_tokens, qk_heads, head_dim], block_size=[1, None, head_dim]
    # src[fused_qk_norm_rope.py:88]: ):
    # src[fused_qk_norm_rope.py:86-121]: ...
    _launcher(_helion_fused_qk_norm_rope, (_NUM_SM * 128,), qkv, q_weight, k_weight, position_ids, cos_sin_cache, cos_sin_cache.stride(0), cos_sin_cache.stride(1), k_weight.stride(0), position_ids.stride(0), q_weight.stride(0), qkv.stride(0), qkv.stride(1), qkv.stride(2), num_tokens, eps, is_neox, _NUM_SM, num_warps=1, num_stages=2, maxnreg=64)