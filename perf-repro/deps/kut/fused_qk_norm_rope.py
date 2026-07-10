# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# VENDORED from vllm/kernels/helion/ops/fused_qk_norm_rope.py (upstream main).
# The @helion.kernel BODY is byte-identical to upstream. Only the vLLM-internal imports used
# solely by the upstream baseline()/generate_inputs() (vllm.ir, RotaryEmbedding) are dropped —
# our harness supplies its own torch reference (refs.ref_fused_qk_norm_rope) and input builder.
# pick_config + _compute_cos_sin_cache are kept verbatim (needed for the vs-vLLM-tuned arm + builder).

from __future__ import annotations

from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.register import register_kernel

import helion
import helion.language as hl


def _compute_cos_sin_cache(max_position_embeddings, rotary_dim, device="cuda", dtype=torch.float):
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim)
    )
    t = torch.arange(max_position_embeddings, device=device, dtype=dtype)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1)


_pick_cache: dict[tuple[int, int, int], "CaseKey | None"] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> "CaseKey | None":
    """Nearest-shape lookup mirroring upstream: match (q_heads, kv_heads), then smallest
    tuned num_tokens >= request (else largest)."""
    qkv, num_heads_q, num_heads_k = args[0], args[1], args[2]
    head_dim = args[4]
    num_tokens = qkv.shape[0]
    cache_key = (num_heads_q, num_heads_k, num_tokens)
    if cache_key in _pick_cache:
        return _pick_cache[cache_key]
    # group tuned keys by (q_heads, kv_heads) -> sorted num_tokens
    by_heads: dict[tuple[int, int], list[int]] = {}
    for key in config_keys:
        hk = (key["q_heads"], key["kv_heads"])
        by_heads.setdefault(hk, []).append(key["num_tokens"])
    if (num_heads_q, num_heads_k) not in by_heads:
        _pick_cache[cache_key] = None
        return None
    toks = sorted(by_heads[(num_heads_q, num_heads_k)])
    chosen_tok = next((n for n in toks if n >= num_tokens), toks[-1])
    chosen = CaseKey({"q_heads": num_heads_q, "kv_heads": num_heads_k, "num_tokens": chosen_tok})
    _pick_cache[cache_key] = chosen
    return chosen


@register_kernel(
    mutates_args=["qkv"],
    config_picker=pick_config,
)
def fused_qk_norm_rope(
    qkv: torch.Tensor,  # [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_position, rotary_dim]
    is_neox: bool,
    position_ids: torch.Tensor,  # [num_tokens],
    forced_token_heads_per_warp: int = -1,  # dummy
) -> None:
    assert qkv.ndim == 2
    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v
    assert qkv.shape[1] == total_heads * head_dim
    hl.specialize(qkv.shape[1])

    assert cos_sin_cache.ndim == 2
    max_position, rotary_dim = cos_sin_cache.shape
    hl.specialize(max_position)
    hl.specialize(rotary_dim)
    assert rotary_dim % 2 == 0
    assert rotary_dim <= head_dim
    embed_dim = rotary_dim // 2

    hl.specialize(num_heads_q)
    hl.specialize(num_heads_k)
    hl.specialize(num_heads_v)
    hl.specialize(head_dim)

    assert position_ids.ndim == 1 and position_ids.shape[0] == num_tokens
    hl.specialize(position_ids.shape[0])

    assert q_weight.ndim == 1 and q_weight.shape[0] == head_dim
    hl.specialize(q_weight.shape[0])
    assert k_weight.ndim == 1 and k_weight.shape[0] == head_dim
    hl.specialize(k_weight.shape[0])

    assert qkv.dtype == q_weight.dtype and q_weight.dtype == k_weight.dtype
    assert position_ids.dtype == torch.int64

    assert qkv.is_contiguous()
    assert position_ids.is_contiguous()
    assert q_weight.is_contiguous()
    assert k_weight.is_contiguous()
    assert cos_sin_cache.is_contiguous()

    qk_heads = num_heads_q + num_heads_k

    qkv = qkv.view(num_tokens, -1, head_dim)

    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, qk_heads, head_dim], block_size=[1, None, head_dim]
    ):
        x_blk = qkv[tile_m, tile_gn, tile_n].to(dtype=torch.float32)

        rms = x_blk.pow(2).sum(dim=-1)
        rms = torch.rsqrt(rms * (1.0 / head_dim) + eps)

        use_q_weight = (tile_gn.index < num_heads_q)[None, :, None]
        w_blk = torch.where(
            use_q_weight, q_weight[None, None, tile_n], k_weight[None, None, tile_n]
        )

        x_blk = (x_blk * rms[:, :, None]).to(qkv.dtype) * w_blk

        qkv[tile_m, tile_gn, tile_n] = x_blk

        pos_id = position_ids[tile_m]
        cos_blk = cos_sin_cache[pos_id, hl.arange(embed_dim)]
        sin_blk = cos_sin_cache[pos_id, hl.arange(embed_dim) + embed_dim]

        if is_neox:
            x1_offset = hl.arange(embed_dim)
            x2_offset = x1_offset + embed_dim
        else:
            x1_offset = hl.arange(embed_dim) * 2
            x2_offset = x1_offset + 1

        x1_blk = qkv[tile_m, tile_gn, x1_offset]
        x2_blk = qkv[tile_m, tile_gn, x2_offset]

        o1_blk = x1_blk * cos_blk[:, None, :] - x2_blk * sin_blk[:, None, :]
        o2_blk = x2_blk * cos_blk[:, None, :] + x1_blk * sin_blk[:, None, :]

        qkv[tile_m, tile_gn, x1_offset] = o1_blk
        qkv[tile_m, tile_gn, x2_offset] = o2_blk
