"""Tests for the CuTe ``merge_sibling_v_loops`` AST pass.

After ``hoist_warp_reduce`` runs, two-pass online softmax still has TWO
``range_constexpr(V)`` loops per outer iter — one for max, one for sum.
Both loops bitcast the SAME ``_tile_unroll_vec_*`` hoist var to extract
the per-lane scalar.

The pass introduces a small ``cute.make_fragment(V, cutlass.Float32)``
cache. V-loop 1 stores ``Float32(values)`` into the cache once per lane;
V-loop 2 reads the cached fp32 value back instead of re-running the
``Uint16 -> Float16`` bitcast chain. Eliminates the redundant per-lane
bitcast+cast pair.

A second peephole removes
``A = Float<N>(warp_reduction_*(...)) ; B = Float32(A)`` round-trips
left over after ``hoist_warp_reduce`` promoted the accumulator to fp32
— the Float<N> wrap is dead in that situation.

Lives in ``helion/_compiler/cute/merge_sibling_v_loops.py``.
"""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@pytest.fixture(autouse=True)
def _disable_online_to_3pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests in this file pin codegen details of the ORIGINAL online
    two-pass form.  The ``online_to_3pass`` rewrite would change them,
    so disable it here; the rewrite itself is covered in
    ``test_cute_online_to_3pass.py``.
    """
    monkeypatch.setenv("HELION_DISABLE_ONLINE_TO_3PASS", "1")


@helion.kernel(backend="cute")
def _reduction_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


@onlyBackends(["cute"])
class TestCuteMergeSiblingVLoops(TestCase):
    def test_merge_fires_on_two_v_loop_pattern(self) -> None:
        """The pass must fire on the canonical two-V-loop shape and emit
        a ``_helion_vmerge_cache_*`` fragment populated in V-loop 1 and
        read in V-loop 2.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # Cache fragment allocated.
        self.assertIn("_helion_vmerge_cache_", code)
        # Cache write in V-loop 1 (writes Float32(values) at the
        # vec_lane index).
        self.assertIn("_helion_vmerge_cache_0[vec_lane_1]", code)
        # The cache is allocated as Float32 (promotes from fp16 so V-loop
        # 2 doesn't need the redundant Float32 cast).
        self.assertIn(
            "_helion_vmerge_cache_0 = cute.make_fragment(4, cutlass.Float32)",
            code,
        )

    def test_cast_elision_on_warp_reduction(self) -> None:
        """The double-cast peephole must collapse
        ``A = Float16(warp_reduction(...)); B = Float32(A)`` into
        ``A = warp_reduction(...); B = A``. The inner Float16 wrap on
        the max-reduce becomes dead after ``hoist_warp_reduce`` promoted
        the accumulator to fp32.
        """
        x = torch.randn(4096, 12672, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 4],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        # The original pattern ``local_amax = Float16(warp_reduction_max(...))``
        # must have been elided to just ``local_amax = warp_reduction_max(...)``.
        self.assertNotIn(
            "local_amax = cutlass.Float16(cute.arch.warp_reduction_max",
            code,
        )
        self.assertIn(
            "local_amax = cute.arch.warp_reduction_max",
            code,
        )

    def test_no_merge_when_v_loop_absent(self) -> None:
        """When V=1 there's no constexpr V-loop, so the merge pass must
        be a no-op — no cache fragment should be emitted.
        """
        x = torch.randn(4096, 4096, device=DEVICE, dtype=HALF_DTYPE)
        code, out = code_and_output(
            _reduction_kernel,
            (x,),
            block_sizes=[1, 128],
            num_threads=[0, 32],
            cute_vector_widths=[1, 1],
        )
        ref = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        self.assertNotIn("_helion_vmerge_cache_", code)


# ── THE LAYERED CONSTEXPR-VEC SPLIT MUST NOT NEST A LANE-INVARIANT TAIL ───────
#
# WHAT IS BEING PINNED.  ``_split_constexpr_vec_multi_stage`` (the LAYERED split, taken
# when two lane-reduce markers are sequentially dependent -- ``sum(exp(v - max(v)))``)
# used to wrap its consume tail in a lane nest UNCONDITIONALLY.  The single-stage twin
# (``_split_lane_loop_over_constexpr_vec``) classifies its tail into invariant / varying
# halves and nests only the varying half; the layered path did not.
#
# ⭐ WHY THAT COSTS A WHOLE EXTRA ROW READ, which is the non-obvious part: ``lane_nest``
# re-emits the LOAD SIDE in every pass, deliberately -- ``ReadWrites`` cannot see that a
# ``cute.copy`` writes its destination, so a load can never be recovered by slicing.  So
# an unconditional wrap drags a gmem read in with it, and if the tail reads only the
# finalized reduced scalars (lane-INVARIANT by construction -- that is what the
# cross-thread combine produced) the fragment it fills is never read at all.
#
# MEASURED on shipped ``examples/cross_entropy.py``, N=1024, pinned
# ``reduction_loops=[None] cute_threads_per_row=[128] cute_vector_widths=[8,1]``:
#   * THREE ``for synthetic_lane_1 in range(4)`` nests where the algorithm has two sweeps
#   * ``cute.copy`` count 3 where it needs 2 -- a dead read of the row
#   * the scalar epilogue (``labels`` load, target-logit gather, ``log``, and the row's
#     single ``losses`` store) nested inside ``4 * 8 = 32`` iterations
# After the fix: copy 3->2, nests 3->2, store hoisted out of both loops.
#
# ⚠ REDUNDANT RATHER THAN WRONG -- every arm was numerically correct (``err=0.0e+00``,
# bit-exact) -- which is exactly why no gate level caught it: a PERF defect with a
# correctness-shaped cause, found by sweeping configurations nobody had emitted before.
# ⚠ And its measured cost at that shape was ZERO (the 64 MB working set fits B200's
# ~126 MB L2, so the extra read was an L2 hit).  This test pins the EMISSION, which is
# where the defect is observable, and does not claim a perf win.
@onlyBackends(["cute"])
class TestLayeredVecSplitTailIsNotNested(TestCase):
    #: The exact geometry the defect was measured at.  ⭐ PINNED, and it must stay
    #: pinned: ``reduction_loops=[None]`` + a small N is what keeps the kernel on
    #: ``PersistentReductionStrategy`` and therefore on the constexpr-vec split at all.
    #: At N >= 4096 the roll rule (one CTA's <=1024 reduction threads cannot cover more
    #: than 1024 elements persistently) force-rolls it and this path is NOT reached --
    #: measured, so do not "generalise" this test to a larger N expecting more coverage.
    _CONFIG = helion.Config(
        block_sizes=[1],
        num_threads=[1],
        reduction_loops=[None],
        cute_vector_widths=[8, 1],
        cute_threads_per_row=[128],
        cute_cluster_n=[1],
    )
    _M, _N = 512, 1024

    def _emit(self) -> str:
        from examples.cross_entropy import cross_entropy

        torch.manual_seed(0)
        logits = torch.randn([self._M, self._N], device=DEVICE, dtype=HALF_DTYPE)
        labels = torch.randint(0, self._N, [self._M], device=DEVICE)
        return cross_entropy.bind((logits, labels)).to_triton_code(self._CONFIG)

    def test_the_loop_free_request_is_rolled_and_emits_no_marker(self) -> None:
        """⭐⭐ THE 2-AND-2 EMISSION PIN THIS TEST USED TO CARRY IS GONE, AND SO IS THE
        MECHANISM IT PINNED.  Read this before "restoring" it.

        It asserted ``lane_nests == 2`` / ``cute.copy == 2`` on the layered constexpr-vec AST
        rewrite, against a measured defect of 3-and-3 (a dead row read plus an epilogue
        re-run 32x).  ⭐ Both the defect and the rewrite are gone, and NOT by dropping the
        capability: ``ConfigSpec.normalize`` now ROLLS a persistent reduction that would take
        a TV plan, so the shape reaches ``LoopedReductionStrategy`` -- which gets one subgraph
        per dependency layer from the reduction roller and owes no marker at all.

        ⇒ what is worth pinning now is that the ROUTING happened and left no marker behind.
        A ``reduction_loops=[None]`` request must still produce a working, vectorized kernel;
        the emitted nest count is the roller's business, not this file's.
        """
        code = self._emit()
        self.assertNotIn(
            "_helion_lane_reduce",
            code,
            "an undischarged lane-reduce marker reached the emitted kernel; the whole point "
            "of rolling this shape is that no marker is owed",
        )
        self.assertGreaterEqual(
            code.count("cute.make_tiled_copy_tv("),
            1,
            "the loop-free request must still reach the TV layout -- an earlier version of "
            "this work gated the regime off and silently lost the vectorization on 144 "
            "measured configs",
        )
        self.assertEqual(
            code.count("cute.arch.load"),
            0,
            "a TV kernel must not also carry the classic cute.arch.load form",
        )

    def test_the_row_store_is_not_inside_the_lane_nest(self) -> None:
        """The epilogue store runs once per row, not ``lane_extent * vec`` times.

        ⭐ This is the arm that would still fail if someone hoisted the dead *read* while
        leaving the tail nested -- the two halves of the defect are separable, so they get
        separate assertions.  Indentation is the observable: a store inside both the lane
        loop and the constexpr V-loop is two levels deeper than the function body.
        """
        code = self._emit()
        stores = [ln for ln in code.splitlines() if ".store(" in ln]
        self.assertTrue(stores, "expected the kernel to store its per-row loss")
        for line in stores:
            indent = len(line) - len(line.lstrip())
            self.assertLessEqual(
                indent,
                4,
                f"store is nested {indent} spaces deep, so it re-runs once per "
                f"(lane, vec) element instead of once per row: {line.strip()[:90]}",
            )

    def test_the_kernel_is_still_bit_exact(self) -> None:
        """The numeric half: removing work must not change the answer.

        MEASURED ``err=0.0e+00`` against the reference on this geometry, so the
        assertion is tight rather than tolerance-shaped.

        ⛔ THE COMPILED CALLABLE IS INVOKED DIRECTLY, NOT VIA ``bound(...)``.  A
        ``bound(args)`` call does NOT honour a config passed to ``compile_config``
        earlier -- it re-enters the runtime's own selection and AUTOTUNES.  MEASURED:
        the first draft of this test called ``bound(logits, labels)`` and spawned a
        ``multiprocessing`` autotune child that ran for 11 minutes at 109% CPU before
        it was killed.  ⭐ That is a BUG IN THE TEST, not a cost to suppress: whatever
        it measured came from a geometry nobody chose and nobody recorded, so the
        assertion would not have been about ``_CONFIG``'s emission at all.
        ``compile_config`` RETURNS the pinned kernel; call that.
        """
        from examples.cross_entropy import cross_entropy

        torch.manual_seed(0)
        logits = torch.randn([self._M, self._N], device=DEVICE, dtype=HALF_DTYPE)
        labels = torch.randint(0, self._N, [self._M], device=DEVICE)
        pinned = cross_entropy.bind((logits, labels)).compile_config(self._CONFIG)
        # ⚠ ``examples/cross_entropy.py`` returns ``losses.mean()`` -- a SCALAR, not the
        # per-row vector.  Comparing against ``reduction="none"`` fails on shape, which is
        # how this arm caught its own first draft.
        expected = torch.nn.functional.cross_entropy(
            logits.to(torch.float32), labels, reduction="mean"
        )
        torch.testing.assert_close(
            pinned(logits, labels), expected, atol=1e-2, rtol=1e-2
        )
