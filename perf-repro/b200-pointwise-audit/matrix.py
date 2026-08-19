"""Fixed kernel, dtype, and shape matrix for the B200 pointwise audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KernelSpec:
    cohort: str
    kernel: str
    dtype: str
    shapes: tuple[tuple[Any, ...], ...]
    has_aot: bool
    aot_arch: str | None = None


SPECS = (
    KernelSpec(
        "general",
        "swiglu",
        "bf16",
        (
            (32768, 1536),
            (16384, 2880),
            (8192, 11008),
            (8192, 14336),
            (4096, 28672),
            (2048, 24576),
        ),
        False,
    ),
    KernelSpec(
        "general",
        "geglu",
        "bf16",
        (
            (16384, 2880),
            (8192, 6912),
            (8192, 14336),
            (4096, 21504),
            (4096, 36864),
            (2048, 24576),
        ),
        False,
    ),
    KernelSpec(
        "general",
        "rope",
        "bf16",
        (
            (1, 32, 2048, 256),
            (1, 32, 8192, 256),
            (1, 32, 4096, 256),
            (2, 32, 2048, 256),
            (1, 32, 4096, 128),
            (4, 8, 4096, 128),
        ),
        True,
        "sm90",
    ),
    KernelSpec(
        "vllm",
        "silu_mul_fp8",
        "bf16_to_fp8",
        (
            (1, 2048),
            (2, 8192),
            (8, 4096),
            (16, 11008),
            (64, 2880),
            (128, 2048),
            (256, 7688),
            (384, 8192),
            (512, 14336),
        ),
        True,
        "sm90",
    ),
    KernelSpec(
        "sglang",
        "silu_and_mul_interleaved",
        "bf16",
        (
            (6, 4096, False),
            (16, 16384, True),
            (48, 512, False),
            (192, 4096, True),
            (768, 3072, False),
            (1024, 12288, False),
            (3072, 2048, False),
            (12288, 1536, True),
            (98304, 6144, False),
        ),
        True,
        "sm100",
    ),
)

SPEC_BY_NAME = {spec.kernel: spec for spec in SPECS}


def iter_cells(
    kernels: set[str] | None = None,
) -> list[tuple[KernelSpec, int, tuple[Any, ...]]]:
    cells = []
    for spec in SPECS:
        if kernels is not None and spec.kernel not in kernels:
            continue
        cells.extend((spec, index, shape) for index, shape in enumerate(spec.shapes))
    return cells
