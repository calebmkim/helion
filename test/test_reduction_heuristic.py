from __future__ import annotations

import unittest

import torch

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.autotuner_heuristics.triton_reduction import (
    TritonReductionHeuristic,
)
from helion._compiler.autotuner_heuristics.triton_reduction import _select_recipe
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel()
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


class TestTritonReductionHeuristic(TestCase):
    @onlyBackends(["triton"])
    @skipIfRefEager("Heuristic seeds are skipped in ref eager mode")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0))
    def test_eligible_inner_persistent_mid_r(self) -> None:
        # sum_kernel R=1024: sum-class persistent (R <= persistent_r_max=4096).
        # xblock cap=1 (R>256), tile=1024, nw=4 (tile<=2048).
        x = torch.empty((2048, 1024), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        fact = bound.config_spec.reduction_facts[0]
        self.assertTrue(
            TritonReductionHeuristic.is_eligible(
                bound.env, bound.host_function.device_ir
            )
        )
        recipe = _select_recipe(fact)
        self.assertEqual(recipe.xblock, 1)
        self.assertEqual(recipe.reduction_loops, [None])
        self.assertEqual(recipe.num_warps, 4)

    @onlyBackends(["triton"])
    @skipIfRefEager("Heuristic seeds are skipped in ref eager mode")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0))
    def test_persistent_small_r_packs_xblock(self) -> None:
        # sum_kernel R=256: small-R fast path packs xblock to keep tile near
        # 2048. xblock cap = min(16, 2048//256) = 8, tile = 8*256 = 2048,
        # num_warps=1 (R<=256).
        x = torch.empty((2048, 256), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        fact = bound.config_spec.reduction_facts[0]
        recipe = _select_recipe(fact)
        self.assertEqual(recipe.xblock, 8)
        self.assertEqual(recipe.reduction_loops, [None])
        self.assertEqual(recipe.num_warps, 1)

    @onlyBackends(["triton"])
    @skipIfRefEager("Heuristic seeds are skipped in ref eager mode")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0))
    def test_persistent_at_boundary(self) -> None:
        # sum_kernel R=4096 = persistent_r_max for sum-class -> still
        # persistent. xblock=1 (R>256), tile=4096, num_warps=8 (tile>2048).
        x = torch.empty((2048, 4096), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        fact = bound.config_spec.reduction_facts[0]
        recipe = _select_recipe(fact)
        self.assertEqual(recipe.xblock, 1)
        self.assertEqual(recipe.reduction_loops, [None])
        self.assertEqual(recipe.num_warps, 8)

    @onlyBackends(["triton"])
    @skipIfRefEager("Heuristic seeds are skipped in ref eager mode")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0))
    def test_looped_above_persistent_boundary(self) -> None:
        # sum_kernel R=8192 -> sum-class looped (R > persistent_r_max=4096).
        # pow2_cap = min(prev_pow2(8192)=8192, MAX_R0_BLOCK=4096) = 4096,
        # 4096 < r_size_hint(8192) so no backoff -> r0_block=4096.
        # xblock=1 (R>2048), num_warps=next_pow2(4096//128=32) clamp [4,16]=16.
        x = torch.empty((2048, 8192), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        fact = bound.config_spec.reduction_facts[0]
        recipe = _select_recipe(fact)
        self.assertEqual(recipe.xblock, 1)
        self.assertEqual(recipe.reduction_loops, [4096])
        self.assertEqual(recipe.num_warps, 16)

    @onlyBackends(["triton"])
    @skipIfRefEager("Heuristic seeds are skipped in ref eager mode")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0))
    def test_seed_emitted_in_compiler_seed_configs(self) -> None:
        # End-to-end: no env flag, the heuristic wins eligibility and emits
        # exactly one seed for sum_kernel R=8192 (looped path).
        x = torch.empty((2048, 8192), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        names = bound.config_spec.autotuner_heuristics
        self.assertIn("triton_reduction", names)
        self.assertEqual(len(seeds), 1)
        seed = seeds[0]
        self.assertEqual(seed.reduction_loops, [4096])
        self.assertEqual(seed.num_warps, 16)
        self.assertEqual(seed.num_stages, 1)


if __name__ == "__main__":
    unittest.main()
