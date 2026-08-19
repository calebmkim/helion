"""Fixed kernel, dtype, and shape matrix for the B200 reduction audit."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelSpec:
    cohort: str
    kernel: str
    dtype: str
    shapes: tuple[tuple[int, ...], ...]
    has_aot: bool


SPECS = (
    KernelSpec(
        "general_aot",
        "rms_norm",
        "bf16",
        (
            (2048, 48),
            (2048, 1023),
            (2048, 4096),
            (4096, 7168),
            (16384, 8192),
            (589824, 256),
        ),
        True,
    ),
    KernelSpec(
        "general_aot",
        "layer_norm",
        "fp16",
        (
            (4096, 1024),
            (4096, 3072),
            (8192, 5120),
            (4096, 12288),
            (4096, 16384),
            (1024, 36864),
        ),
        True,
    ),
    KernelSpec(
        "general_aot",
        "softmax",
        "fp16",
        (
            (4096, 256),
            (4096, 384),
            (4096, 768),
            (4096, 4096),
            (4096, 16384),
            (2048, 32768),
        ),
        True,
    ),
    KernelSpec(
        "general_aot",
        "cross_entropy",
        "bf16",
        (
            (2048, 32000),
            (1024, 256000),
            (2048, 128256),
            (8192, 128000),
            (4096, 152064),
            (2048, 256000),
        ),
        True,
    ),
    KernelSpec(
        "original",
        "kl_div",
        "bf16",
        (
            (8192, 32768),
            (2048, 50257),
            (4096, 114688),
            (1024, 128256),
            (4096, 151936),
            (1024, 250000),
        ),
        False,
    ),
    KernelSpec(
        "original",
        "jsd",
        "bf16",
        (
            (8192, 32768),
            (2048, 50257),
            (4096, 114688),
            (2048, 128256),
            (8192, 151936),
            (1024, 250000),
        ),
        False,
    ),
    KernelSpec(
        "original",
        "fused_linear_jsd",
        "bf16",
        (
            (8192, 32000),
            (4096, 50257),
            (8192, 128256),
            (2048, 151936),
            (2048, 256000),
            (16384, 32000),
        ),
        False,
    ),
    KernelSpec(
        "original",
        "grpo",
        "bf16",
        (
            (8, 1024, 32000),
            (8, 2048, 64000),
            (4, 2048, 128256),
            (8, 4096, 128256),
            (16, 1024, 50257),
            (4, 1024, 256000),
        ),
        False,
    ),
    KernelSpec(
        "original",
        "rms_norm_bwd",
        "bf16",
        (
            (2048, 4096),
            (8192, 4096),
            (4096, 8192),
            (16384, 4096),
            (8192, 2048),
            (2048, 11008),
        ),
        False,
    ),
    KernelSpec(
        "original",
        "layer_norm_bwd",
        "bf16",
        (
            (2048, 4096),
            (8192, 4096),
            (4096, 8192),
            (16384, 4096),
            (8192, 2048),
            (2048, 11008),
        ),
        False,
    ),
)


def _vllm_specs() -> tuple[KernelSpec, ...]:
    tokens = (1, 128, 8192)
    hidden = (2048, 4096, 5120)
    intermediate = (6144, 12288, 25600)
    q_heads = (16, 32, 64)
    return (
        KernelSpec(
            "vllm",
            "dynamic_per_token_scaled_fp8_quant",
            "native",
            tuple((t, h) for h in hidden for t in tokens),
            True,
        ),
        KernelSpec(
            "vllm",
            "per_token_group_fp8_quant",
            "native",
            tuple((t, h, 128) for h in hidden for t in tokens),
            True,
        ),
        KernelSpec(
            "vllm",
            "rms_norm_dynamic_per_token_quant",
            "native",
            tuple((t, h) for h in hidden for t in tokens),
            True,
        ),
        KernelSpec(
            "vllm",
            "rms_norm_per_block_quant",
            "native",
            tuple((t, h, 128) for h in hidden for t in tokens),
            True,
        ),
        KernelSpec(
            "vllm",
            "silu_and_mul_per_block_quant",
            "native",
            tuple((t, i, 128) for i in intermediate for t in tokens),
            True,
        ),
        KernelSpec(
            "vllm",
            "fused_qk_norm_rope",
            "native",
            tuple((t, qh, 8) for qh in q_heads for t in tokens),
            True,
        ),
    )


SPECS += _vllm_specs()
SPEC_BY_NAME = {spec.kernel: spec for spec in SPECS}


def iter_cells(
    kernels: set[str] | None = None,
) -> list[tuple[KernelSpec, int, tuple[int, ...]]]:
    cells = []
    for spec in SPECS:
        if kernels is not None and spec.kernel not in kernels:
            continue
        cells.extend((spec, index, shape) for index, shape in enumerate(spec.shapes))
    return cells
