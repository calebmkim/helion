"""Tests for Phase-D's constants sweep: the CuTe layout knob and its derivations.

Three kinds of decision were audited in ``cute/tv_layout.py``,
``reduction_strategy.py`` and the two ``memory_ops.py`` halves, and this file pins the
two kinds that CHANGED.

⛔ THE KNOB THIS FILE USED TO PIN IS GONE.  ``cute_stage_smem_kb`` (a whole-kernel SMEM
budget for ``smem`` staging tiles) was deleted in run 2, together with
``cute_tv_sweep_cache``: both were *performance* ceilings that could overrule the residency
a config NAMES, which is how 13 of 40 frozen cells came to record a residency their kernel
never used.  Its four test classes went with it; the device-capacity half it was clamped
against survives as ``TileStrategy._cute_stage_smem_capacity_bytes`` and is pinned by
``_notes/tests/test_staging_lane_extent.py::test_smem_capacity_refusal_still_fires``.
⇒ what remains here is THE DERIVATIONS, which are independent of that knob.

⭐ THE DERIVATIONS.  Four numbers were replaced by the computation that produced them:

* ``_THREADS_PER_ROW_MAX = 256`` -> :data:`_DEFAULT_MAX_THREADS_PER_ROW`, which is
  ``max(_NUM_THREADS_SMALL, _NUM_THREADS_LARGE)``.  The 256 was never independent: a CTA
  must own at least one row, so the ladder's ``threads_per_row`` cannot exceed its own
  ``num_threads``.  Writing it down instead of deriving it is what produced the
  seed/search asymmetry below.
* ``THREADS_PER_ROW_CHOICES`` -> powers of two up to
  ``thread_budget.MAX_THREADS_PER_BLOCK``.  ⚠ THE ASYMMETRY THIS FIXES: run 2 widened
  the search menu to 1024 while leaving the ladder capped at a literal 256, so the SEED
  could not reach values the SEARCH could -- invisible unless both lines are read
  together.  The two bounds have genuinely different principles (``num_threads`` vs the
  hardware CTA limit), which is why they still differ; only their derivation is shared.
* ``CLUSTER_N_CHOICES`` -> powers of two up to the arch cluster maximum.
* ``TV_COPY_MAX_VEC_BITS = 128`` -> ``MAX_COPY_BITS``, and ``assumed_align_bytes=16`` /
  ``alignment=16`` (five sites) -> ``ASSUMED_ALIGN_BYTES``, which names helion's blanket
  pointer-alignment promise (``runtime/__init__.py:3981``) once.

A derivation must be VALUE-IDENTICAL, not merely principled: the acceptance test for the
whole sweep is byte-identical emitted CuTe on all 40 frozen cells, and these tests are
the host-side half of that -- they check the numbers themselves, so a future edit to a
bound is caught here rather than as a moved hash.

Lives in:
- ``helion/_compiler/cute/tv_layout.py`` (the derivations, the domain, the ladder)
- ``helion/_compiler/cute/thread_budget.py`` (``MAX_THREADS_PER_BLOCK``)
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.thread_budget import MAX_THREADS_PER_BLOCK
from helion._compiler.cute.tv_layout import _DEFAULT_MAX_THREADS_PER_ROW
from helion._compiler.cute.tv_layout import ASSUMED_ALIGN_BYTES
from helion._compiler.cute.tv_layout import BYTES_PER_KIB
from helion._compiler.cute.tv_layout import CLUSTER_N_CHOICES
from helion._compiler.cute.tv_layout import MAX_COPY_BITS
from helion._compiler.cute.tv_layout import THREADS_PER_ROW_CHOICES
from helion._compiler.cute.tv_layout import TV_COPY_MAX_VEC_BITS
from helion._compiler.cute.tv_layout import max_cluster_n_for_arch
from helion._compiler.cute.tv_layout import quack_num_threads_for
from helion._compiler.cute.tv_layout import quack_rows_per_cta_for
from helion._compiler.cute.tv_layout import threads_per_row_for
from helion._testing import TestCase
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

# Every N the ladders are asked about anywhere in this run, plus the awkward ones the
# frozen 40-cell table does NOT cover (tiny, odd, prime, far past the last rung).
_LADDER_NS: tuple[int, ...] = (
    1,
    7,
    8,
    32,
    63,
    64,
    128,
    1023,
    1024,
    3072,
    4096,
    4097,
    6144,
    8191,
    8192,
    12000,
    16384,
    32768,
    65536,
    100000,
    131072,
    262144,
    524288,
    1048576,
)


@helion.kernel(backend="cute")
def _staged_norm(x: torch.Tensor) -> torch.Tensor:
    """A row read by TWO sweeps of one rolled reduction, i.e. the shape whose second
    read ``reload_from="smem"`` serves out of a staged tile rather than out of DRAM.

    ⚠ This docstring must not name the emitted staging variable: helion copies a
    kernel's docstring verbatim into the emitted host wrapper, so a marker string the
    tests grep for would match the docstring and report staging as present on a kernel
    where the budget had correctly declined it (the trap ``test_cute_pass_knobs.py``
    records for its own kernel).
    """
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        mean = torch.mean(row, dim=-1)
        centered = row - mean[:, None]
        var = torch.mean(centered * centered, dim=-1)
        out[tile_m, :] = (centered * torch.rsqrt(var[:, None] + 1e-5)).to(x.dtype)
    return out


def _norm_ref(x: torch.Tensor) -> torch.Tensor:
    xf = x.to(torch.float32)
    centered = xf - xf.mean(dim=-1, keepdim=True)
    var = (centered * centered).mean(dim=-1, keepdim=True)
    return (centered * torch.rsqrt(var + 1e-5)).to(x.dtype)


class TestCuteTvLayoutDerivations(TestCase):
    """The four numbers that became computations.  Pure host arithmetic, no GPU."""

    def test_ladder_cap_is_the_cta_size_not_a_literal(self) -> None:
        """⭐ THE SEED/SEARCH ASYMMETRY, as an invariant rather than a coincidence.

        ``threads_per_row <= num_threads`` is forced: ``rows_per_cta`` is
        ``num_threads // threads_per_row`` and a CTA must own at least one row.  So the
        ladder's saturation value is not a tuning choice, and pinning it to a literal
        256 while the search menu reached 1024 meant the seed could not express what the
        search could.
        """
        cap = _DEFAULT_MAX_THREADS_PER_ROW
        for n in _LADDER_NS:
            with self.subTest(n=n):
                tpr = threads_per_row_for(n)
                self.assertLessEqual(tpr, cap)
                # The derivation's whole content: the cap IS the CTA size, so a CTA
                # always owns a whole number of rows, >= 1.
                self.assertLessEqual(tpr, quack_num_threads_for(n))
                self.assertGreaterEqual(quack_rows_per_cta_for(n), 1)
                self.assertEqual(
                    quack_rows_per_cta_for(n) * tpr <= MAX_THREADS_PER_BLOCK, True
                )

    def test_ladder_saturates_at_its_derived_cap(self) -> None:
        """Far past the last rung the ladder must actually REACH the derived cap --
        otherwise the derivation would be vacuous (any cap >= 128 would pass the
        inequality above)."""
        self.assertEqual(threads_per_row_for(1 << 30), _DEFAULT_MAX_THREADS_PER_ROW)
        self.assertEqual(_DEFAULT_MAX_THREADS_PER_ROW, 256)

    def test_search_menu_is_powers_of_two_to_the_cta_limit(self) -> None:
        """The menu's upper bound is the HARDWARE CTA limit, not the ladder's cap: the
        reduction thread axis may be the whole block, and ``LoopedReductionStrategy``
        only ever LOWERS ``thread_count`` toward the requested value.

        Powers of two is the load-bearing property, not the magnitude -- ``tpr`` is a
        butterfly shuffle's lane group and LEDGER E031 measured relerr 0.15-2.6 for a
        non-power-of-two group.
        """
        self.assertEqual(max(THREADS_PER_ROW_CHOICES), MAX_THREADS_PER_BLOCK)
        for choice in THREADS_PER_ROW_CHOICES:
            with self.subTest(choice=choice):
                self.assertEqual(choice & (choice - 1), 0)
        self.assertEqual(list(THREADS_PER_ROW_CHOICES), sorted(THREADS_PER_ROW_CHOICES))
        # The seed is REACHABLE by the search at every N -- the property the literal
        # cap silently broke in the other direction.
        for n in _LADDER_NS:
            with self.subTest(n=n):
                self.assertIn(threads_per_row_for(n), THREADS_PER_ROW_CHOICES)

    def test_cluster_menu_is_powers_of_two_to_the_arch_cap(self) -> None:
        """No choice may exceed what any arch can launch, or the fragment would offer a
        value ``max_cluster_n_for_arch`` filters out on every device."""
        widest = max(max_cluster_n_for_arch(arch) for arch in (None, 8, 9, 10, 12, 13))
        self.assertEqual(max(CLUSTER_N_CHOICES), widest)
        self.assertEqual(min(CLUSTER_N_CHOICES), 1)
        for choice in CLUSTER_N_CHOICES:
            with self.subTest(choice=choice):
                self.assertEqual(choice & (choice - 1), 0)

    def test_copy_width_ceiling_is_the_atom_width(self) -> None:
        """``TV_COPY_MAX_VEC_BITS`` is the SAME hardware fact as ``MAX_COPY_BITS``.

        ``cute.copy`` through the atom reaches the atom's own limit while
        ``cute.arch.load`` with an explicit ``VectorType`` falls short of it; only the
        SHORTFALL is a separate number (the ``vec_width > 4`` gate in
        ``cute/memory_ops.py``).  Two literals could drift, and the drift would be
        silent -- ``ChunkTVPlan.__post_init__`` checks a plan built at ``MAX_COPY_BITS``
        against ``TV_COPY_MAX_VEC_BITS``.
        """
        self.assertEqual(TV_COPY_MAX_VEC_BITS, MAX_COPY_BITS)

    def test_alignment_promise_matches_the_widest_copy(self) -> None:
        """helion promises a blanket 16-byte ``assumed_align``
        (``runtime/__init__.py:3981``), and that is exactly the bytes one
        ``MAX_COPY_BITS``-wide copy moves -- which is why a narrower alignment would
        fail the DSL's proof.  Named once so the gmem leg, the SMEM allocation and
        ``legal_vec``'s clamp cannot disagree.
        """
        self.assertEqual(ASSUMED_ALIGN_BYTES, MAX_COPY_BITS // 8)

    def test_kib_conversion(self) -> None:
        self.assertEqual(BYTES_PER_KIB, 1024)


if __name__ == "__main__":
    import unittest

    unittest.main()
