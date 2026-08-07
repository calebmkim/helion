"""Lowering-level row residency across scalar, classic-vector, and TV loads."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _two_read_rows_2d(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        total = row.sum(-1, keepdim=True)
        out[tile_m, :] = (x[tile_m, :].to(torch.float32) - total).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _two_read_rows_3d(x: torch.Tensor) -> torch.Tensor:
    m, k, _n = x.size()
    out = torch.empty_like(x)
    for tile_m, tile_k in hl.tile([m, k]):
        row = x[tile_m, tile_k, :].to(torch.float32)
        total = row.sum(-1, keepdim=True)
        out[tile_m, tile_k, :] = (x[tile_m, tile_k, :].to(torch.float32) - total).to(
            out.dtype
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _alias_store_between_reads(x: torch.Tensor, alias: torch.Tensor) -> torch.Tensor:
    m, _n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        first = x[tile_m, :].to(torch.float32)
        total = first.sum(-1, keepdim=True)
        alias[tile_m, :] = alias[tile_m, :] + 3
        out[tile_m, :] = (x[tile_m, :].to(torch.float32) - total).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _conditional_first_read(x: torch.Tensor, take_branch: int) -> torch.Tensor:
    m, _n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        if take_branch > 0:
            first = x[tile_m, :].to(torch.float32)
            total = first.sum(-1, keepdim=True)
        else:
            total = x[tile_m, 0].to(torch.float32)[:, None] * 0
        out[tile_m, :] = (x[tile_m, :].to(torch.float32) - total).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_then_column(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        total = row.sum(-1, keepdim=True)
        out[:, tile_m] = (x[:, tile_m].to(torch.float32) - total.T).to(out.dtype)
    return out


def _poisoned_rows(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    columns = torch.arange(shape[-1], device=DEVICE)
    row = torch.where(columns % 2 == 0, 1.0, -1.0).to(dtype)
    result = row.expand(shape).clone()
    for column, value in zip(
        (0, 127, 128, 511, 512, 1023),
        (8, 16, 24, 32, 40, 48),
        strict=True,
    ):
        result[..., column] = value
    return result


def _config(
    *,
    rank: int,
    vec: int,
    residency: str | None,
    reduction_loop: int | None,
) -> helion.Config:
    values: dict[str, object] = {
        "block_sizes": [1] * (rank - 1),
        "num_threads": [1] * (rank - 1),
        "cute_threads_per_row": [128],
        "reduction_loops": [reduction_loop],
        "cute_vector_widths": [vec],
    }
    if residency is not None:
        values["cute_row_residency"] = [residency]
    return helion.Config(
        **values,
    )


def _compile_and_check(
    kernel: helion.Kernel[torch.Tensor],
    x: torch.Tensor,
    config: helion.Config,
) -> str:
    bound = kernel.bind((x,))
    source = bound.to_triton_code(config)
    output = bound.compile_config(config)(x)
    torch.cuda.synchronize()
    expected = (x.float() - x.float().sum(-1, keepdim=True)).to(x.dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    return source


@onlyBackends(["cute"])
class TestCuteRowResidency(TestCase):
    def test_scalar_looped_and_persistent(self) -> None:
        x = _poisoned_rows((3, 1024), torch.float32)
        for reduction_loop in (None, 512, 1024):
            for residency in ("registers", "smem"):
                with self.subTest(reduction_loop=reduction_loop, residency=residency):
                    source = _compile_and_check(
                        _two_read_rows_2d,
                        x,
                        _config(
                            rank=2,
                            vec=1,
                            residency=residency,
                            reduction_loop=reduction_loop,
                        ),
                    )
                    self.assertEqual(source.count(").load()"), 1)
                    self.assertIn(f"'row residency: {residency}'", source)
                    cache = (
                        "_row_rmem_cache_"
                        if residency == "registers"
                        else "_row_smem_cache_"
                    )
                    self.assertIn(cache, source)

    def test_classic_vector_looped_and_persistent(self) -> None:
        # Rank 3 keeps the classic Uint16 vector hoist: the reduction owns a
        # vector loop, while the TV copy intentionally declines this load site.
        x = _poisoned_rows((3, 2, 1024), torch.bfloat16)
        for reduction_loop in (None, 512, 1024):
            for residency in ("registers", "smem"):
                with self.subTest(reduction_loop=reduction_loop, residency=residency):
                    source = _compile_and_check(
                        _two_read_rows_3d,
                        x,
                        _config(
                            rank=3,
                            vec=4,
                            residency=residency,
                            reduction_loop=reduction_loop,
                        ),
                    )
                    self.assertEqual(source.count("cute.arch.load("), 1)
                    self.assertNotIn("cute.copy(", source)
                    self.assertIn(f"row residency: {residency}", source)

    def test_persistent_tv_registers(self) -> None:
        x = _poisoned_rows((3, 1024), torch.bfloat16)
        source = _compile_and_check(
            _two_read_rows_2d,
            x,
            _config(
                rank=2,
                vec=4,
                residency="registers",
                reduction_loop=None,
            ),
        )
        self.assertIn("_tv_rmem_cache_0 = cute.make_rmem_tensor", source)
        self.assertEqual(source.count("cute.copy(_tv_atom, _tv_part_0"), 1)
        self.assertIn("_tv_frag_0[reduction_vec_lane", source)
        self.assertIn("= _tv_rmem_cache_0[", source)
        self.assertIn("'row residency: registers'", source)

        gmem_source = _compile_and_check(
            _two_read_rows_2d,
            x,
            _config(
                rank=2,
                vec=4,
                residency="gmem",
                reduction_loop=None,
            ),
        )
        self.assertNotIn("_tv_rmem_cache_", gmem_source)
        self.assertEqual(gmem_source.count("cute.copy(_tv_atom, _tv_part_0"), 2)
        self.assertIn("'row residency: gmem'", gmem_source)

    def test_gmem_preserves_repeated_non_tv_loads(self) -> None:
        cases = (
            (
                _two_read_rows_2d,
                _poisoned_rows((3, 1024), torch.float32),
                2,
                1,
                ").load()",
            ),
            (
                _two_read_rows_3d,
                _poisoned_rows((3, 2, 1024), torch.bfloat16),
                3,
                4,
                "cute.arch.load(",
            ),
        )
        for kernel, x, rank, vec, load_token in cases:
            for reduction_loop in (None, 512, 1024):
                with self.subTest(rank=rank, reduction_loop=reduction_loop):
                    source = _compile_and_check(
                        kernel,
                        x,
                        _config(
                            rank=rank,
                            vec=vec,
                            residency="gmem",
                            reduction_loop=reduction_loop,
                        ),
                    )
                    self.assertGreaterEqual(source.count(load_token), 2)
                    self.assertNotIn("_row_rmem_cache_", source)
                    self.assertNotIn("_row_smem_cache_", source)
                    self.assertIn("row residency: gmem", source)

    def test_default_ladder_applies_to_scalar_lowering(self) -> None:
        for n, expected in ((1024, "registers"), (4096, "smem")):
            with self.subTest(n=n, expected=expected):
                source = _compile_and_check(
                    _two_read_rows_2d,
                    _poisoned_rows((3, n), torch.float32),
                    _config(
                        rank=2,
                        vec=1,
                        residency=None,
                        reduction_loop=512,
                    ),
                )
                cache = (
                    "_row_rmem_cache_"
                    if expected == "registers"
                    else "_row_smem_cache_"
                )
                self.assertIn(cache, source)
                self.assertIn(f"'row residency: {expected}'", source)

    def test_fp8_scalar_cache_uses_storage_dtype(self) -> None:
        x = _poisoned_rows((3, 1024), torch.float8_e4m3fn)
        for residency in ("registers", "smem"):
            with self.subTest(residency=residency):
                source = _compile_and_check(
                    _two_read_rows_2d,
                    x,
                    _config(
                        rank=2,
                        vec=1,
                        residency=residency,
                        reduction_loop=None,
                    ),
                )
                self.assertIn("cutlass.Uint8", source)
                self.assertIn(f"'row residency: {residency}'", source)

    def test_argument_alias_store_declines_residency(self) -> None:
        x = _poisoned_rows((3, 1024), torch.float32)
        original = x.clone()
        alias = x.view_as(x)
        config = _config(
            rank=2,
            vec=1,
            residency=None,
            reduction_loop=None,
        )
        bound = _alias_store_between_reads.bind((x, alias))
        source = bound.to_triton_code(config)
        output = bound.compile_config(config)(x, alias)
        torch.cuda.synchronize()

        expected = original + 3 - original.sum(-1, keepdim=True)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertNotIn("_row_rmem_cache_", source)
        self.assertNotIn("_row_smem_cache_", source)
        self.assertIn("'row residency: gmem", source)

    def test_conditional_first_read_does_not_publish_residency(self) -> None:
        x = _poisoned_rows((3, 1024), torch.float32)
        config = _config(
            rank=2,
            vec=1,
            residency=None,
            reduction_loop=None,
        )
        bound = _conditional_first_read.bind((x, 0))
        source = bound.to_triton_code(config)
        compiled = bound.compile_config(config)
        without_first_read = compiled(x, 0)
        with_first_read = compiled(x, 1)
        torch.cuda.synchronize()

        torch.testing.assert_close(without_first_read, x, rtol=0, atol=0)
        torch.testing.assert_close(
            with_first_read,
            x - x.sum(-1, keepdim=True),
            rtol=0,
            atol=0,
        )
        self.assertNotIn("_row_rmem_cache_", source)
        self.assertNotIn("_row_smem_cache_", source)
        self.assertIn("'row residency: gmem", source)

    def test_reduction_axis_is_part_of_row_identity(self) -> None:
        x = _poisoned_rows((1024, 1024), torch.float32)
        config = _config(
            rank=2,
            vec=1,
            residency=None,
            reduction_loop=None,
        )
        bound = _row_then_column.bind((x,))
        source = bound.to_triton_code(config)
        output = bound.compile_config(config)(x)
        torch.cuda.synchronize()

        expected = x.float() - x.float().sum(-1)[None, :]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertGreaterEqual(source.count(").load()"), 2)
        self.assertIn("'row residency: gmem", source)
