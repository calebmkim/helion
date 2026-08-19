from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

_BLOCK_SIZE_0 = tl.constexpr(1)

@triton.jit
def _helion_cross_entropy(labels, logits_flat, logits, losses, _REDUCTION_BLOCK_1: tl.constexpr):
    # src[cross_entropy.py:24]: for tile_n in hl.tile(n):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    # src[cross_entropy.py:25]: labels_tile = labels[tile_n]
    labels_tile = tl.load(labels + indices_0 * 1, None)
    # src[cross_entropy.py:26]: base_indices_tile = tile_n.index * v
    v_0 = tl.full([], 256000, tl.int32)
    v_1 = tl.cast(indices_0 * v_0, tl.int32)
    # src[cross_entropy.py:27]: flat_indices = base_indices_tile + labels_tile
    v_2 = tl.cast(v_1, tl.int64)
    v_3 = v_2 + labels_tile
    # src[cross_entropy.py:28]: logits_at_target = hl.load(logits_flat, [flat_indices])
    logits_at_target = tl.load(logits_flat + v_3 * 1, None)
    # src[cross_entropy.py:31]: max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
    max_logits_acc = tl.full([_BLOCK_SIZE_0, _REDUCTION_BLOCK_1], float('-inf'), tl.float32)
    # src[cross_entropy.py:30]: logits_rows = logits[tile_n, :]
    for roffset_1 in tl.range(0, 256000, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < 256000
        logits_rows = tl.load(logits + (indices_0[:, None] * 256000 + (0 + rindex_1)[None, :] * 1), mask_1[None, :], other=0)
        # src[cross_entropy.py:31]: max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
        _mask_to = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _REDUCTION_BLOCK_1]), logits_rows, tl.full([], float('-inf'), tl.bfloat16))
        v_4 = triton_helpers.maximum(max_logits_acc, _mask_to)
        max_logits_acc = v_4
    max_logits = tl.cast(tl.reshape(tl.max(max_logits_acc, 1), [_BLOCK_SIZE_0, 1]), tl.bfloat16)
    # src[cross_entropy.py:35]: log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
    squeeze = tl.reshape(max_logits, [_BLOCK_SIZE_0])
    # src[cross_entropy.py:34]: sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
    sum_exp_acc = tl.full([_BLOCK_SIZE_0, _REDUCTION_BLOCK_1], 0, tl.float32)
    # src[cross_entropy.py:25]: labels_tile = labels[tile_n]
    for roffset_1 in tl.range(0, 256000, _REDUCTION_BLOCK_1):
        rindex_1 = roffset_1 + tl.arange(0, _REDUCTION_BLOCK_1).to(tl.int32)
        mask_1 = rindex_1 < 256000
        max_logits_copy = max_logits
        # src[cross_entropy.py:30]: logits_rows = logits[tile_n, :]
        logits_rows_1 = tl.load(logits + (indices_0[:, None] * 256000 + (0 + rindex_1)[None, :] * 1), mask_1[None, :], other=0)
        # src[cross_entropy.py:32]: shifted = logits_rows - max_logits
        v_5 = logits_rows_1 - max_logits_copy
        # src[cross_entropy.py:33]: exp_shifted = torch.exp(shifted)
        v_6 = tl.cast(v_5, tl.float32)
        v_7 = libdevice.exp(v_6)
        v_8 = tl.cast(v_7, tl.bfloat16)
        # src[cross_entropy.py:34]: sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
        _mask_to_1 = tl.where(tl.broadcast_to(mask_1[None, :], [_BLOCK_SIZE_0, _REDUCTION_BLOCK_1]), v_8, tl.full([], 0, tl.bfloat16))
        v_9 = sum_exp_acc + _mask_to_1
        sum_exp_acc = v_9
    sum_exp = tl.cast(tl.reshape(tl.sum(sum_exp_acc, 1), [_BLOCK_SIZE_0, 1]), tl.bfloat16)
    # src[cross_entropy.py:35]: log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
    squeeze_1 = tl.reshape(sum_exp, [_BLOCK_SIZE_0])
    v_10 = tl.cast(squeeze_1, tl.float32)
    v_11 = tl_math.log(v_10)
    v_12 = tl.cast(v_11, tl.bfloat16)
    v_13 = squeeze + v_12
    # src[cross_entropy.py:37]: losses[tile_n] = log_sum_exp - logits_at_target
    v_14 = v_13 - logits_at_target
    tl.store(losses + indices_0 * 1, v_14, None)

def cross_entropy(logits: torch.Tensor, labels: torch.Tensor, *, _launcher=_default_launcher):
    # src[cross_entropy.py:20]: n, v = logits.shape
    n, v = logits.shape
    # src[cross_entropy.py:21]: losses = torch.empty([n], dtype=logits.dtype, device=logits.device)
    losses = torch.empty([n], dtype=logits.dtype, device=logits.device)
    # src[cross_entropy.py:22]: logits_flat = logits.view(-1)
    logits_flat = logits.view(-1)
    # src[cross_entropy.py:30]: logits_rows = logits[tile_n, :]
    _REDUCTION_BLOCK_1 = 16384
    # src[cross_entropy.py:24]: for tile_n in hl.tile(n):
    # src[cross_entropy.py:25]:     labels_tile = labels[tile_n]
    # src[cross_entropy.py:26]:     base_indices_tile = tile_n.index * v
    # src[cross_entropy.py:24-37]: ...
    _launcher(_helion_cross_entropy, (1024,), labels, logits_flat, logits, losses, _REDUCTION_BLOCK_1, num_warps=32, num_stages=1)
    # src[cross_entropy.py:39]: return losses.mean()
    return losses.mean()