"""SGLang Inkling's Helion SiLU-and-mul kernel and selected SM100 AOT table.

The kernel body and configs come from local SGLang commit
5f79cf35110d6a0be828f266160b75d83a2a6276. The SM100 table key is
``(hidden, (itemsize,), (has_topk_weights,))`` and does not include rows.
"""

from __future__ import annotations

from typing import Any

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=False, backend="triton")
def silu_and_mul_interleaved(
    gateup_output: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    out_dtype: hl.constexpr | None = None,
) -> torch.Tensor:
    batch_size, hidden_size = gateup_output.shape
    hidden_size = hl.specialize(hidden_size)
    assert hidden_size % 2 == 0, f"{hidden_size=}"
    half_hidden_size = hidden_size // 2
    down_input = gateup_output.new_empty(
        batch_size,
        half_hidden_size,
        dtype=out_dtype or gateup_output.dtype,
    )
    for batch_tile, hidden_tile in hl.tile([batch_size, half_hidden_size]):
        gate_output = gateup_output[batch_tile, 2 * hidden_tile.index].to(torch.float32)
        up_output = gateup_output[batch_tile, 2 * hidden_tile.index + 1].to(
            torch.float32
        )
        silu_mul_output = gate_output * torch.sigmoid(gate_output) * up_output
        if topk_weights is not None:
            weight_scale = topk_weights[batch_tile, None].to(torch.float32)
            silu_mul_output = silu_mul_output * weight_scale
        down_input[batch_tile, hidden_tile] = silu_mul_output
    return down_input


def silu_and_mul_interleaved_torch(
    gateup_output: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    gate = gateup_output[:, 0::2].float()
    up = gateup_output[:, 1::2].float()
    output = gate * torch.sigmoid(gate) * up
    if topk_weights is not None:
        output = output * topk_weights[:, None].float()
    return output.to(out_dtype or gateup_output.dtype)


_USEFUL_CONFIGS: dict[int, dict[str, Any]] = {
    0: {
        "block_sizes": [2, 512],
        "indexing": [
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
            "tensor_descriptor",
        ],
        "l2_groupings": [1],
        "load_eviction_policies": ["last", "", ""],
        "loop_orders": [[1, 0]],
        "num_stages": 3,
        "num_warps": 1,
        "pid_type": "flat",
        "range_flattens": [None],
        "range_multi_buffers": [None],
        "range_num_stages": [0],
        "range_unroll_factors": [0],
        "range_warp_specializes": [None],
    },
    1: {
        "block_sizes": [1, 1024],
        "indexing": [
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "pointer",
        ],
        "l2_groupings": [1],
        "load_eviction_policies": ["last", "first", "last"],
        "loop_orders": [[1, 0]],
        "num_stages": 5,
        "num_warps": 1,
        "pid_type": "flat",
        "range_flattens": [None],
        "range_multi_buffers": [None],
        "range_num_stages": [0],
        "range_unroll_factors": [0],
        "range_warp_specializes": [None],
    },
    4: {
        "block_sizes": [2, 512],
        "indexing": ["pointer", "tensor_descriptor", "pointer"],
        "l2_groupings": [2],
        "load_eviction_policies": ["last", ""],
        "loop_orders": [[0, 1]],
        "num_stages": 2,
        "num_warps": 1,
        "pid_type": "flat",
        "range_flattens": [None],
        "range_multi_buffers": [None],
        "range_num_stages": [0],
        "range_unroll_factors": [0],
        "range_warp_specializes": [None],
    },
}

_INDEX_BY_KEY = {
    (512, (2,), (False,)): 0,
    (512, (2,), (True,)): 1,
    (1024, (2,), (False,)): 0,
    (1024, (2,), (True,)): 0,
    (1536, (2,), (False,)): 1,
    (1536, (2,), (True,)): 1,
    (2048, (2,), (False,)): 4,
    (2048, (2,), (True,)): 1,
    (3072, (2,), (False,)): 0,
    (3072, (2,), (True,)): 0,
    (4096, (2,), (False,)): 4,
    (4096, (2,), (True,)): 1,
    (4608, (2,), (False,)): 0,
    (4608, (2,), (True,)): 0,
    (6144, (2,), (False,)): 0,
    (6144, (2,), (True,)): 1,
    (7680, (2,), (False,)): 1,
    (7680, (2,), (True,)): 1,
    (8192, (2,), (False,)): 4,
    (8192, (2,), (True,)): 1,
    (9216, (2,), (False,)): 0,
    (9216, (2,), (True,)): 0,
    (10240, (2,), (False,)): 0,
    (10240, (2,), (True,)): 1,
    (12288, (2,), (False,)): 0,
    (12288, (2,), (True,)): 1,
    (14336, (2,), (False,)): 0,
    (14336, (2,), (True,)): 1,
    (16384, (2,), (False,)): 4,
    (16384, (2,), (True,)): 1,
}


def select_sm100_config(
    hidden: int,
    has_topk_weights: bool,
) -> tuple[helion.Config, int, tuple[int, tuple[int], tuple[bool]]]:
    key = (hidden, (2,), (has_topk_weights,))
    index = _INDEX_BY_KEY[key]
    return helion.Config(**_USEFUL_CONFIGS[index]), index, key
