from __future__ import annotations

import unittest

import torch

import helion
from helion._compiler.reduction_hint import ReductionHint
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel()
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    n, v = logits.shape
    losses = torch.zeros([n], dtype=logits.dtype, device=logits.device)
    logits_flat = logits.view(-1)
    for tile_n in hl.tile(n):
        labels_tile = labels[tile_n]
        base_indices_tile = tile_n.index * v
        flat_indices = base_indices_tile + labels_tile
        logits_at_target = hl.load(logits_flat, [flat_indices])
        logits_rows = logits[tile_n, :]
        max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
        shifted = logits_rows - max_logits
        exp_shifted = torch.exp(shifted)
        sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
        log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
        losses[tile_n] = log_sum_exp - logits_at_target
    return losses.mean()


class TestReductionFacts(TestCase):
    @onlyBackends(["triton"])
    @skipIfRefEager("Reduction facts are not collected in ref eager mode")
    def test_sum_kernel_facts(self) -> None:
        x = torch.empty((2048, 1024), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        facts = bound.config_spec.reduction_facts
        self.assertEqual(len(facts), 1)
        fact = facts[0]
        self.assertEqual(fact.static_rnumel, 1024)
        self.assertEqual(fact.static_xnumel, 2048)
        self.assertEqual(fact.r_size_hint, 1024)
        self.assertEqual(fact.hint, ReductionHint.INNER)
        self.assertTrue(fact.is_rollable)
        self.assertEqual(fact.num_load, 1)
        self.assertEqual(fact.num_reduction, 1)
        self.assertEqual(fact.num_store, 1)
        self.assertEqual(fact.accum_dtype, torch.float32)

    @onlyBackends(["triton"])
    @skipIfRefEager("Reduction facts are not collected in ref eager mode")
    def test_sum_kernel_non_pow2_rdim(self) -> None:
        # Non-power-of-two reduction extent: r_size_hint mirrors the live
        # ReductionLoopSpec.size_hint, which carries the raw size (not pow2-
        # rounded). Recipes use it as the persistent-decode boundary, so any
        # emitted r0_block must be strictly less than this value.
        x = torch.empty((2048, 1023), device=DEVICE, dtype=torch.float32)
        bound = sum_kernel.bind((x,))
        facts = bound.config_spec.reduction_facts
        self.assertEqual(len(facts), 1)
        fact = facts[0]
        self.assertEqual(fact.static_rnumel, 1023)
        self.assertEqual(fact.r_size_hint, 1023)
        self.assertEqual(fact.hint, ReductionHint.INNER)

    @onlyBackends(["triton"])
    @skipIfRefEager("Reduction facts are not collected in ref eager mode")
    def test_cross_entropy_two_reductions(self) -> None:
        # cross_entropy has amax + sum per row -> num_reduction == 2.
        # Use distinct n != v so x_block_id is unambiguous.
        logits = torch.empty((2048, 4096), device=DEVICE, dtype=torch.float32)
        labels = torch.empty((2048,), device=DEVICE, dtype=torch.int64)
        bound = cross_entropy.bind((logits, labels))
        facts = bound.config_spec.reduction_facts
        self.assertEqual(len(facts), 1)
        fact = facts[0]
        self.assertEqual(fact.num_reduction, 2)
        self.assertEqual(fact.static_rnumel, 4096)
        self.assertEqual(fact.static_xnumel, 2048)
        self.assertEqual(fact.hint, ReductionHint.INNER)
        self.assertEqual(fact.accum_dtype, torch.float32)
        self.assertTrue(fact.is_rollable)


if __name__ == "__main__":
    unittest.main()
