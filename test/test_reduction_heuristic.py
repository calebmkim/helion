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
    def test_eligible_inner_persistent(self) -> None:
        # R=1024 INNER, sum_kernel: Recipe A small-persistent branch.
        x = torch.empty((2048, 1024), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        fact = bound.config_spec.reduction_facts[0]
        self.assertTrue(
            TritonReductionHeuristic.is_eligible(
                bound.env, bound.host_function.device_ir
            )
        )
        recipe = _select_recipe(fact)
        # 1024//1024 = 1, capped at 8 -> xblock=1, persistent (None), num_warps=1.
        self.assertEqual(recipe.xblock, 1)
        self.assertEqual(recipe.reduction_loops, [None])
        self.assertEqual(recipe.num_warps, 1)

    @onlyBackends(["triton"])
    @skipIfRefEager("Heuristic seeds are skipped in ref eager mode")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0))
    def test_recipe_a_general_path(self) -> None:
        # R=256: still <=1024 (Recipe A), but small-persistent gate (256<=R) holds,
        # so xblock=min(1024//256, 8)=4, num_warps=1.
        x = torch.empty((2048, 256), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        fact = bound.config_spec.reduction_facts[0]
        recipe = _select_recipe(fact)
        self.assertEqual(recipe.xblock, 4)
        self.assertEqual(recipe.reduction_loops, [None])
        self.assertEqual(recipe.num_warps, 1)

    @onlyBackends(["triton"])
    @skipIfRefEager("Heuristic seeds are skipped in ref eager mode")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0))
    def test_recipe_b_looped(self) -> None:
        # R=4096 -> Recipe B. pow2_cap=min(prev_pow2(4096),2048)=2048, !>= r_size_hint(4096),
        # so r0_block=2048. R>2048 -> xblock=1. num_warps = next_pow2(2048//128, 2, 16) = 16.
        x = torch.empty((2048, 4096), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        fact = bound.config_spec.reduction_facts[0]
        recipe = _select_recipe(fact)
        self.assertEqual(recipe.xblock, 1)
        self.assertEqual(recipe.reduction_loops, [2048])
        self.assertEqual(recipe.num_warps, 16)

    @onlyBackends(["triton"])
    @skipIfRefEager("Heuristic seeds are skipped in ref eager mode")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0))
    def test_recipe_b_r_equals_pow2_cap_backoff(self) -> None:
        # R=2048 -> Recipe B. pow2_cap=min(2048, MAX_R0_BLOCK=2048)=2048, ==r_size_hint
        # triggers the backoff: r0_block=1024 (must be < r_size_hint to stay looped).
        x = torch.empty((2048, 2048), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        fact = bound.config_spec.reduction_facts[0]
        recipe = _select_recipe(fact)
        self.assertEqual(recipe.reduction_loops, [1024])
        # R<=2048 -> xblock=2.
        self.assertEqual(recipe.xblock, 2)

    @onlyBackends(["triton"])
    @skipIfRefEager("Heuristic seeds are skipped in ref eager mode")
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0))
    def test_seed_emitted_in_compiler_seed_configs(self) -> None:
        # End-to-end: with no env flag, the heuristic still wins eligibility
        # and produces exactly one seed. The recipe-emitted shape matches Recipe B.
        x = torch.empty((2048, 4096), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        names = bound.config_spec.autotuner_heuristics
        self.assertIn("triton_reduction", names)
        self.assertEqual(len(seeds), 1)
        seed = seeds[0]
        self.assertEqual(seed.reduction_loops, [2048])
        self.assertEqual(seed.num_warps, 16)
        self.assertEqual(seed.num_stages, 1)


if __name__ == "__main__":
    unittest.main()
