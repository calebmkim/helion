"""The lane-loop split's unsound fallback now RAISES instead of dropping data.

WHAT IS BEING PINNED.  ``_split_one_lane_loop`` lowers a reduction whose axis is
distributed across a synthetic lane loop.  When two markers are *sequentially
dependent* (``mx = amax(v)``; ``s = sum(exp(v - mx))``) it emits one
accumulate/finalize pass per dependency layer (``_split_lane_loop_multi_stage``).

If that layered split declines, the historical behaviour was to call
``_restore_per_lane_markers``, which replaces each marker with its raw per-lane
input.  That is a **silent wrong answer** on this shape, not a safe degradation:
the lane loop IS distributing the reduction axis, so dropping the cross-lane
accumulator drops data.  ``mx = v`` makes every summand ``exp(v - v) == 1.0``, and
the emitted kernel contains zero cross-lane reductions -- bug class 8 P1, which
returned exactly 1.0 for 256/256 rows at ``default_config()``.

The fallback at that one site is now ``exc.BackendUnsupported``.

⭐⭐ AND SINCE G1, THIS SHAPE DOES NOT REACH THAT PASS AT ALL.  Lowering now emits a
lane-distributed reduction's serial fold AND its cross-thread combine INLINE
(``ReductionStrategy._emit_inline_lane_reduce``), so the P1 shape emits **no marker**
and ``split_lane_loop_reductions`` has nothing to rewrite.  ⇒ the marker + AST path
is still REACHABLE (at every condition the inline emitter declines on: a prebuilt TV
nest, a nested sink, an unduplicatable producer, no registered lane loop) and on it
the historical restore is still a silent wrong answer -- so everything below is still
load-bearing, but the tests that pin the AST path must now say so explicitly via
``_withhold_inline_lane_reduce()``.  Two of them started failing when G1 landed, for a
reason that was the GOAL rather than a defect; that is why the switch exists.

WHY THIS TEST MONKEYPATCHES.  The decline is **unreachable today** (measured 0
reaches across the P1 repro, a 3-deep dependency chain, rms/layer_norm fwd,
softmax, attention and matmul_layernorm), because the register-stash lowering runs
first and absorbs the one shape that would reach it.  A test that only ran real
kernels would pass vacuously and prove nothing about the raise.  So the layered
split is forced to decline, exactly as the original attribution measurement did.

The tests are a set, and each is load-bearing:

  1. ``test_forced_decline_raises`` -- with the decline forced, compilation raises.
  2. ``test_old_fallback_really_was_wrong`` -- the FAIL-CAPABILITY control: with the
     decline forced and the old fallback re-installed, the answer is the P1
     signature.  Without this, (1) could pass because something unrelated failed.
  3. ``test_unforced_kernel_is_correct`` -- the no-regression half: unpatched, the
     same kernel is correct and emits real accumulators, i.e. the raise is not on
     the path any working kernel takes.
  4. ``test_inline_lowering_emits_no_marker_and_two_lane_nests`` -- G1's own claim:
     the obligation is discharged AT LOWERING, so no marker is ever created.
     ⚠ Its structural counts are a no-regression check only; the marker-CALL count is
     the sole fail-capable assertion (measured: the AST path emits the same 2 nests
     and 2 outside combines on this shape, so the counts alone were VACUOUS).
  5. ``test_withholding_the_inline_emission_brings_the_marker_back`` -- the
     fail-capability arm for (4): a zero is only evidence if withholding the feature
     brings the markers back, in the same process.
  6. ``test_three_dependency_layers_get_three_lane_nests`` -- the ``#nests == #layers``
     identity at N=3, since every other shape here has exactly two and a mechanism
     that silently capped at two would pass all of them.

⚠ EACH TEST BUILDS ITS OWN KERNEL OBJECT.  ``_restore_per_lane_markers`` mutates
the loop body in place (``loop.body = body``), and a ``helion.kernel`` caches its
compiled result -- so a module-level kernel shared with the control arm serves the
POISONED lowering to the tests that run after it.  MEASURED: sharing one kernel
makes (3) fail with 100% of rows wrong while it passes in isolation, which reads
exactly like a real defect.  Do not hoist the kernel back to module scope.
"""

from __future__ import annotations

import ast
import os
import re
from typing import Callable
import unittest
import unittest.mock

import torch

import helion
from helion import Config
from helion._compiler import tile_strategy
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion.exc import BackendUnsupported
import helion.language as hl
from helion.language import _tracing_ops

# ``block_sizes=[32]`` + an ``hl.arange``-indexed reduction axis is the class-8 P1
# configuration: reachable from ``default_config()``, and it synthesises the lane
# loop FOR this reduction -- which is what makes the per-lane restore a wrong
# answer rather than a conservative one.
_P1_CONFIG = Config(block_sizes=[32], cute_vector_widths=[1])

_M, _N = 32, 64


def _make_kernel() -> Callable[[torch.Tensor], torch.Tensor]:
    """A FRESH ``sum(exp(v - amax(v)))`` kernel: two sequentially dependent markers.

    The sum's summand needs the max as a finalized scalar, so this is the shape
    that requires the layered split.  Written with ``hl.arange`` rather than a
    nested ``hl.tile`` because an arange axis is deliberately never registered as a
    ``ReductionLoopSpec``, which is what routes it through the lane loop.

    Built per test rather than once at module scope -- see the file docstring.
    """

    @helion.kernel(config=_P1_CONFIG, static_shapes=True)
    def dependent_reductions(x: torch.Tensor) -> torch.Tensor:
        m, n = x.size()
        out = torch.empty([m], dtype=torch.float32, device=x.device)
        for tile_m in hl.tile(m):
            idx = hl.arange(0, n)
            v = x[tile_m, idx].to(torch.float32)
            mx = torch.amax(v, dim=-1)
            out[tile_m] = torch.sum(torch.exp(v - mx[:, None]), dim=-1)
        return out

    return dependent_reductions


def _reference(x: torch.Tensor) -> torch.Tensor:
    xf = x.to(torch.float32)
    return torch.sum(torch.exp(xf - xf.amax(dim=-1, keepdim=True)), dim=-1)


def _force_decline() -> unittest.mock._patch:
    """Make the layered split decline, so the code under test is reached.

    The raise is guarded by ``_markers_feed_cross_lane_carry`` being False, and
    this kernel has no cross-lane carry, so forcing the decline is sufficient.
    ⚠ It is NOT sufficient for a kernel that does carry (attention): there the
    per-lane restore is correct and must keep compiling -- see
    ``test_a_cross_lane_carry_still_falls_back``.
    """
    return unittest.mock.patch.object(
        tile_strategy, "_split_lane_loop_multi_stage", return_value=None
    )


def _force_old_fallback() -> unittest.mock._patch:
    """Reproduce the pre-fix behaviour exactly.

    At the merge base the code was
    ``if dependent: return [_restore_per_lane_markers(loop, markers)]``.  Making
    the layered split *return* that list is equivalent and reaches the fallback
    without the raise, which is what the control arm needs.
    """
    return unittest.mock.patch.object(
        tile_strategy,
        "_split_lane_loop_multi_stage",
        lambda loop, lane_var, markers: [
            tile_strategy._restore_per_lane_markers(loop, markers)
        ],
    )


def _withhold_inline_lane_reduce() -> unittest.mock._patch:
    """Route this shape back to the MARKER + AST-pass path.

    ⭐⭐ WHY THE TESTS BELOW NEED THIS NOW.  G1 made lowering emit a lane-distributed
    reduction's serial fold and its cross-thread combine INLINE
    (``ReductionStrategy._emit_inline_lane_reduce``), so the P1 shape no longer emits a
    marker at all and never reaches ``_split_one_lane_loop``.  Patching
    ``_split_lane_loop_multi_stage`` to decline therefore had **no effect** -- the pass it
    patches is not on this shape's path any more -- and the two tests below started
    failing for a reason that is the *goal*, not a defect.

    ⛔ THE WRONG FIX WOULD HAVE BEEN TO DELETE THEM.  What they pin is still live: the
    marker path remains reachable at every decline condition the inline emitter has (a
    prebuilt TV nest, a nested sink, an unduplicatable producer, no registered lane loop),
    and on that path the historical restore is still a silent wrong answer.  So the tests
    keep their subject and gain an explicit "put me on that path" arm.

    ``HELION_INLINE_LANE_REDUCE=0`` is the withholding switch the inline emitter carries
    precisely so this arm has an instrument -- see the ⭐⭐ comment at its top.
    """
    return unittest.mock.patch.dict(os.environ, {"HELION_INLINE_LANE_REDUCE": "0"})


@onlyBackends(["cute"])
class TestLaneSplitDeclineRaises(TestCase):
    def test_forced_decline_raises(self) -> None:
        """The load-bearing assertion: a decline is an ERROR, not wrong numbers."""
        kernel = _make_kernel()
        x = torch.randn(_M, _N, device=DEVICE, dtype=torch.float32)
        with (
            _withhold_inline_lane_reduce(),
            _force_decline(),
            self.assertRaises(BackendUnsupported) as ctx,
        ):
            kernel.bind((x,)).to_triton_code(_P1_CONFIG)
        # The message must name the shape, so a future reader knows which lowering
        # to extend rather than which fallback to restore.
        self.assertIn("lane-distributed", str(ctx.exception))

    def test_old_fallback_really_was_wrong(self) -> None:
        """FAIL-CAPABILITY CONTROL for the test above.

        A raise is only evidence if the path it guards would otherwise return the
        P1 signature.  With the old fallback re-installed the kernel compiles and
        runs, and the answer collapses: ``mx = v`` makes the summand
        ``exp(v - v) == 1.0`` and the cross-lane ``sum`` is dropped too, so the
        result is nowhere near a sum of ``_N`` positive terms.
        """
        kernel = _make_kernel()
        x = torch.randn(_M, _N, device=DEVICE, dtype=torch.float32)
        ref = _reference(x)
        # The reference must be far from the collapsed value, or "got ~1.0" would
        # not discriminate.  Every term is > 0 and one is exactly 1, so the true
        # sum exceeds 1 by construction; assert the margin is large.
        self.assertGreater(float(ref.min()), 2.0)

        with _withhold_inline_lane_reduce(), _force_old_fallback():
            code = kernel.bind((x,)).to_triton_code(_P1_CONFIG)
            got = kernel(x)

        # Structural: the accumulators are GONE, which is the mechanism.
        self.assertNotIn("_lane_acc", code)
        # Numeric: and therefore the answer is wrong, not merely imprecise.
        self.assertGreater(
            float((got - ref).abs().max()),
            1.0,
            f"the control arm did NOT reproduce the P1 signature: got "
            f"{got[:4].tolist()} against ref {ref[:4].tolist()}.  If this fails, "
            f"the raise above is not guarding what this file claims it guards.",
        )

    def test_unforced_kernel_is_correct(self) -> None:
        """NO-REGRESSION: on the real path this shape is correct.

        ⚠ Unpatched, the reduction is now lowered INLINE (G1) rather than by the AST
        split, so the ``_lane_acc`` assertion below is checking the inline emitter's
        accumulator.  Both paths spell it ``*_lane_acc``, deliberately: the inline emitter
        reuses ``_finalize_lane_reduce_marker``, so the two cannot disagree about the
        cross-thread combine for the same layout.
        """
        kernel = _make_kernel()
        x = torch.randn(_M, _N, device=DEVICE, dtype=torch.float32)
        code = kernel.bind((x,)).to_triton_code(_P1_CONFIG)
        got = kernel(x)
        torch.testing.assert_close(got, _reference(x), rtol=1e-4, atol=1e-4)
        # Structural half: the accumulators must actually exist.  A numeric check
        # alone passed on main for shapes where the reduction had silently
        # collapsed, which is how class 8 P1 survived.
        self.assertIn("_lane_acc", code)

    def test_inline_lowering_emits_no_marker_and_two_lane_nests(self) -> None:
        """⭐⭐ G1's OWN ASSERTION: no marker is CREATED, and the structure is right.

        The P1 shape has TWO dependency layers (``amax``, then ``sum`` of a term needing
        the finalized max).  Lowering must emit exactly two lane loops with the
        cross-thread combine BETWEEN them, and must not create a marker at all.

        ⛔⛔ THE MARKER-CALL COUNT IS THE ONLY DISCRIMINATING ASSERTION HERE, AND FINDING
        THAT OUT IS WHY THIS TEST WAS REVERT-VERIFIED.  MEASURED: with the inline emitter
        forced off, the AST split emits **the same 2 lane nests and the same 2 outside
        combines** on this shape -- so every structural count in the first draft of this
        test passed with the feature reverted, i.e. the test was VACUOUS.  ⇒ the structure
        assertions below are a no-regression check (they pin that the inline path does not
        emit MORE than the AST path did -- the old pass's own defect was 3 nests where the
        algorithm has 2, ~50% more traffic that numerics cannot see), and the ``0 markers``
        assertion is the one that fails when the feature is removed.

        ⇒ if you weaken this test, keep the marker-call count.
        """
        kernel = _make_kernel()
        x = torch.randn(_M, _N, device=DEVICE, dtype=torch.float32)

        marker_calls: list[str] = []
        real = tile_strategy._lane_reduce_marker_expr

        def spy(*args: object, **kwargs: object) -> str:
            marker_calls.append(str(args[1]) if len(args) > 1 else "?")
            return real(*args, **kwargs)  # pyrefly: ignore

        with unittest.mock.patch.object(tile_strategy, "_lane_reduce_marker_expr", spy):
            code = kernel.bind((x,)).to_triton_code(_P1_CONFIG)

        # ⭐ THE DISCRIMINATING ASSERTION: the obligation was discharged at lowering, so
        # no marker was ever created -- not merely stripped afterwards.
        self.assertEqual(
            marker_calls,
            [],
            f"lowering still deferred {len(marker_calls)} reduction(s) to a marker: "
            f"{marker_calls}",
        )
        self.assertNotIn("_helion_lane_reduce", code)
        lane_nests = [
            ln
            for ln in code.splitlines()
            if re.match(r"^\s*for\s+synthetic_lane\w*\s+in\s+range\(", ln)
        ]
        self.assertEqual(
            len(lane_nests),
            2,
            f"expected one lane nest per dependency layer (2), got "
            f"{len(lane_nests)}:\n" + "\n".join(lane_nests),
        )
        # One cross-thread combine per reduction, and BOTH outside a lane loop.
        combines = [ln for ln in code.splitlines() if "warp_reduction" in ln]
        self.assertEqual(len(combines), 2, "\n".join(combines))
        for ln in combines:
            indent = len(ln) - len(ln.lstrip())
            self.assertLessEqual(
                indent,
                4,
                f"a cross-thread combine is INSIDE a lane loop -- it would fold an "
                f"unfinished accumulator: {ln!r}",
            )

    def test_withholding_the_inline_emission_brings_the_marker_back(self) -> None:
        """⭐ THE FAIL-CAPABILITY ARM for the test above.

        ⛔ "0 markers" is also what a dead hook, a kernel that stopped compiling, and a
        shape that never emitted all report -- and an unfalsifiable zero has fooled four
        investigations in this tree.  So the zero is only evidence if withholding the new
        emission brings the markers BACK.  This is that arm, in-process.
        """
        with _withhold_inline_lane_reduce():
            kernel = _make_kernel()
            x = torch.randn(_M, _N, device=DEVICE, dtype=torch.float32)
            marker_calls: list[str] = []
            real = tile_strategy._lane_reduce_marker_expr

            def spy(*args: object, **kwargs: object) -> str:
                marker_calls.append(str(args[1]) if len(args) > 1 else "?")
                return real(*args, **kwargs)  # pyrefly: ignore

            with unittest.mock.patch.object(
                tile_strategy, "_lane_reduce_marker_expr", spy
            ):
                code = kernel.bind((x,)).to_triton_code(_P1_CONFIG)
        self.assertEqual(
            len(marker_calls),
            2,
            f"withheld, this shape must emit its 2 markers; got {marker_calls}",
        )
        # And the AST pass still discharges them correctly, so the withheld arm is a
        # working fallback rather than merely a different code path.
        self.assertNotIn("_helion_lane_reduce", code)
        self.assertIn("_lane_acc", code)
        with _withhold_inline_lane_reduce():
            got = _make_kernel()(x)
        torch.testing.assert_close(got, _reference(x), rtol=1e-4, atol=1e-4)

    def test_three_dependency_layers_get_three_lane_nests(self) -> None:
        """⭐⭐ THE ``#nests == #layers`` IDENTITY, CHECKED AT N=3 RATHER THAN ASSUMED.

        Every other shape in this file has exactly TWO dependency layers (``amax`` then
        ``sum``, or ``argmax``'s value then index).  ⛔ A mechanism that happens to work for
        two and silently caps there would pass every one of them: the failure mode is not a
        crash but a *missing* nest, i.e. a reduction folded over the wrong extent.

        So this is the first shape with THREE: ``amax`` -> ``sum(exp(v-mx))`` ->
        ``sum(p*p)`` where ``p = exp(v-mx)/s``.  Each layer needs its predecessor
        *finalized*, so the emission owes three lane loops with a cross-thread combine
        between each pair.

        ⚠ The reference is far from any collapsed value: a dropped fold on the last layer
        gives sum-of-squares over one element instead of the row, so this is discriminating
        rather than a tolerance check.
        """
        cfg = Config(block_sizes=[32], cute_vector_widths=[1])

        @helion.kernel(config=cfg, static_shapes=True)
        def three_deep(x: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                idx = hl.arange(0, n)
                v = x[tile_m, idx].to(torch.float32)
                mx = torch.amax(v, dim=-1)  # layer 1
                e = torch.exp(v - mx[:, None])
                s = torch.sum(e, dim=-1)  # layer 2 -- needs mx final
                p = e / s[:, None]
                out[tile_m] = torch.sum(p * p, dim=-1)  # layer 3 -- needs s final
            return out

        x = torch.randn(256, 1024, device=DEVICE, dtype=torch.bfloat16)
        marker_calls: list[str] = []
        real = tile_strategy._lane_reduce_marker_expr

        def spy(*args: object, **kwargs: object) -> str:
            marker_calls.append(str(args[1]) if len(args) > 1 else "?")
            return real(*args, **kwargs)  # pyrefly: ignore

        with unittest.mock.patch.object(tile_strategy, "_lane_reduce_marker_expr", spy):
            code = three_deep.bind((x,)).to_triton_code(cfg)
        got = three_deep(x)

        xf = x.float()
        e = torch.exp(xf - xf.amax(-1, keepdim=True))
        p = e / e.sum(-1, keepdim=True)
        ref = (p * p).sum(-1)

        self.assertEqual(marker_calls, [], "a layer was deferred to a marker")
        nests = [
            ln
            for ln in code.splitlines()
            if re.match(r"^\s*for\s+synthetic_lane\w*\s+in\s+range\(", ln)
        ]
        self.assertEqual(
            len(nests),
            3,
            f"expected 3 lane nests, got {len(nests)}:\n" + "\n".join(nests),
        )
        combines = [ln for ln in code.splitlines() if "warp_reduction" in ln]
        self.assertEqual(len(combines), 3, "\n".join(combines))
        for ln in combines:
            self.assertLessEqual(
                len(ln) - len(ln.lstrip()),
                4,
                f"a cross-thread combine is INSIDE a lane loop: {ln!r}",
            )
        torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)

    def test_a_second_lane_axis_seals_the_right_loop(self) -> None:
        """⭐⭐ TWO LANE AXES: each seal must segment the axis it BELONGS to.

        ⛔ THE DEFECT THIS PINS, found by adversarially attacking the mechanism.
        ``_wrap_segmented_body`` originally segmented ``lane_loops[-1]`` on the premise that a
        reduction's axis is the innermost registered one.  ``lane_loops`` is ONE FLAT LIST fed
        by two unrelated producers -- a reduction's synthetic lane, and ``generate_ast``'s
        free-``hl.arange`` lane -- so ``[-1]`` is *the last registered*, not *the reduction's*.

        The trigger is a lane-distributed dependent-reduction pair followed by a free
        ``hl.arange`` that registers its own lane loop AFTERWARDS.  With the wrong axis
        segmented, both accumulator seeds and both cross-thread combines land INSIDE the
        reduction's own lane loop: the accumulator is re-initialised and the combine runs on an
        unfinished partial, once per lane.

        ⚠ AND THE FIX COULD NOT BE A COUNT-BASED DECLINE.  MEASURED registration order is
        ``add_lane_loop(synthetic)`` → seal → seal → ``add_lane_loop(arange)``, so at seal time
        there is always exactly ONE lane loop and any ``len(lane_loops) != 1`` guard there is
        dead.  The axis therefore has to be carried as DATA on each seal.

        ⭐ Note the reference: the MARKER path is *wrong* on this shape (measured maxerr ~2e2,
        a pre-existing silent wrong answer), so this test asserts the inline path is correct
        rather than that the two paths agree.
        """
        cfg = Config(block_sizes=[1], cute_vector_widths=[1])

        @helion.kernel(config=cfg, static_shapes=True)
        def two_axes(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            for tm in hl.tile(m):
                i = hl.arange(0, n)
                v = x[tm, i].to(torch.float32)
                mx = torch.amax(v, dim=-1)
                s = torch.sum(torch.exp(v - mx[:, None]), dim=-1)
                j = hl.arange(0, 2048)
                out[tm, j] = s[:, None] + j.to(torch.float32)
            return out

        # ⚠⚠ N=4096 IS LOAD-BEARING AND N=1024 MAKES THIS TEST VACUOUS.  MEASURED: at
        # N=1024 the reduction takes the SHARED-MEMORY path instead (3
        # ``_cute_grouped_reduce``, 0 lane loops, 0 seals), so the segment mechanism never
        # runs and the test passes with the axis bug fully reverted.  At N=4096 it emits 2
        # seals and 3 lane nests, i.e. the code under test actually executes.  Caught by
        # revert-verify; do not "simplify" the size.
        m, n = 2, 4096
        x = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        out = torch.zeros(m, 2048, device=DEVICE, dtype=torch.float32)
        got = two_axes(x, out)

        xf = x.float()
        s_ref = torch.exp(xf - xf.amax(-1, keepdim=True)).sum(-1)
        jj = torch.arange(2048, device=DEVICE, dtype=torch.float32)
        ref = s_ref[:, None] + jj[None, :]
        # ⚠ NOT a tolerance question: with the wrong axis segmented the answer is off by ~2e2
        # against values of ~2e2, i.e. wrong by order-one factors.
        torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)

    def test_a_cross_lane_carry_still_falls_back(self) -> None:
        """⭐ THE TEST THAT CATCHES AN OVER-BROAD RAISE.

        ``examples/attention.py::attention_output`` REACHES this decline -- the
        layered split returns ``None`` for it -- and the per-lane restore is
        CORRECT there, because the online-softmax ``mi``/``di`` recurrence folds
        the lane axis on every lane iteration.  So it must keep compiling.

        MEASURED: raising unconditionally at this site broke all eight attention
        examples in ``test_examples.py``.  The first draft of this file asserted
        only that the raise fires and therefore did NOT catch that; this test is
        the other half, and it is the reason the raise is gated on
        ``_markers_feed_cross_lane_carry``.
        """
        import examples.attention as attention_mod

        b, heads, seq, head_dim = 2, 4, 256, 64
        args = tuple(
            torch.randn(b * heads, seq, head_dim, device=DEVICE, dtype=torch.float32)
            for _ in range(3)
        )
        cfg = Config(block_sizes=[1, 32, 64], loop_orders=[[0, 1]])
        # Compiles rather than raising: that is the whole assertion.
        code = attention_mod.attention_output.bind(args).to_triton_code(cfg)
        self.assertIn("def _helion_attention_output", code)


# ── STEP 2.5: A MATMUL-FOLDED LANE AXIS LICENSES THE RESTORE ──────────────────
#
# WHAT IS BEING PINNED, and why it needed a criterion rather than a loosening.
# ``_decline_structural_lane_split`` step 3 refuses when nothing can express the
# split.  That refusal was OVER-BROAD: it fired on an upstream kernel that is
# genuinely correct at the merge base (``test_indexing.py::
# test_full_slice_in_reduction_loop``, maxabs 1.1e-05 on 4 of its 5 block sizes),
# turning a missed optimisation into a support regression.
#
# ⛔ AND THE OBVIOUS FIX -- restore instead of raising -- IS MEASURED UNSOUND.
# ``_broadcast_store_kernel`` below reaches the same step 3 and the per-lane
# restore returns rel 1.04-1.05 there (every lane writing its own partial sum).
# So the two situations must be TOLD APART, which is
# ``_matmul_already_folded_lane_axis``: the ``baddbmm`` kernel's marker is
# redundant because ``_emit_cute_matmul_n_collapse`` already folded that axis, and
# the ``addmm`` kernel's is not because no such fold was emitted.
#
# The four tests are a set:
#   1. the regressed kernel COMPILES AND IS CORRECT on every block size;
#   2. FAIL-CAP: with the recording suppressed, (1) goes back to raising -- so (1)
#      passes because of the criterion and not for some unrelated reason;
#   3. the counterexample STILL RAISES (the criterion did not just loosen);
#   4. FAIL-CAP for (3): forcing the restore there produces a wrong answer, which
#      is what makes keeping the raise load-bearing rather than cautious.
_FOLD_BLOCK_SIZES = ([16, 16], [16, 8], [8, 16], [16, 4])


def _full_slice_kernel(block_sizes: list[int]) -> Callable[..., torch.Tensor]:
    """``test_indexing.py::test_full_slice_in_reduction_loop``, config PINNED.

    A full slice between two tiled dims plus a matmul in the reduction loop.  The
    ``baddbmm``'s M and N axes share a block id (N == C == D == 16 with
    ``static_shapes``), so the matmul lowering folds that axis itself and the
    trailing ``.sum(-1)`` is a no-op.

    Built per test: ``_restore_per_lane_markers`` mutates the loop body in place and
    a ``helion.kernel`` caches its compile -- see the file docstring.
    """

    @helion.kernel(static_shapes=True, config=Config(block_sizes=block_sizes))
    def full_slice_in_reduction_loop(q: torch.Tensor) -> torch.Tensor:
        n, c, d = q.size(0), q.size(1), q.size(2)
        out = torch.empty([n, c], dtype=q.dtype, device=q.device)
        for (tile_n,) in hl.tile([n]):
            attn = hl.zeros([tile_n, c, c], dtype=torch.float32)
            for tile_d in hl.tile(d):
                qt = q[tile_n, :, tile_d]
                attn = torch.baddbmm(attn, qt, qt.transpose(-2, -1))
            out[tile_n, :] = attn.sum(-1).to(out.dtype)
        return out

    return full_slice_in_reduction_loop


def _broadcast_store_kernel() -> Callable[..., torch.Tensor]:
    """⛔ THE COUNTEREXAMPLE. Config PINNED.

    ``addmm`` in a K loop, then ``out[tile_m, :] = acc.sum(-1, keepdim=True)``.
    Here the lane axis IS the reduced axis, so the marker's fold is REQUIRED and
    deleting it is a wrong answer.

    ⭐ It is deliberately indistinguishable from ``_full_slice_kernel`` by every
    cheaper signal, and all three were tried and refuted: the store signature
    (both store a marker-only value to a lane-varying address), the lane block id
    (both ``block_index=2``), and "does a fold exist" (both emit a
    ``_cute_grouped_reduce``).  That is what this arm defends.
    """

    @helion.kernel(static_shapes=True, config=Config(block_sizes=[32, 32]))
    def broadcast_store(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        n = hl.specialize(y.size(1))
        out = torch.empty([m, n], dtype=torch.float32, device=x.device)
        for tile_m in hl.tile(m):
            acc = hl.zeros([tile_m, n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
            out[tile_m, :] = acc.sum(dim=-1, keepdim=True)
        return out

    return broadcast_store


def _suppress_fold_recording() -> unittest.mock._patch:
    """FAIL-CAPABILITY: make the criterion answer False even where a fold happened.

    Patching the *consumer* (rather than deleting the producer's record) isolates
    exactly the branch under test, and leaves the emitted matmul unchanged so the
    arm still exercises the same kernel.
    """
    return unittest.mock.patch.object(
        tile_strategy, "_matmul_already_folded_lane_axis", return_value=False
    )


def _force_restore_at_step3() -> unittest.mock._patch:
    """FAIL-CAPABILITY: license the restore unconditionally at step 2.5.

    This is precisely the "just revert the raise" fix, and the arm that measures
    why it is not available.
    """
    return unittest.mock.patch.object(
        tile_strategy, "_matmul_already_folded_lane_axis", return_value=True
    )


@onlyBackends(["cute"])
class TestMatmulFoldedLaneAxisLicensesRestore(TestCase):
    def _q(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(16, 16, 16, device=DEVICE)

    @staticmethod
    def _full_slice_reference(q: torch.Tensor) -> torch.Tensor:
        return torch.baddbmm(
            torch.zeros(16, 16, 16, device=DEVICE, dtype=q.dtype),
            q,
            q.transpose(-2, -1),
        ).sum(-1)

    def test_matmul_folded_axis_compiles_and_is_correct(self) -> None:
        """(1) The regressed kernel works on every block size the merge base did.

        MEASURED at the merge base and here: maxabs 1.14e-05 on all four.  The
        assertion is tight (1e-04) rather than the upstream test's ``atol=0.2``,
        which is loose enough that "passes" would prove nothing.
        """
        q = self._q()
        expected = self._full_slice_reference(q)
        for block_sizes in _FOLD_BLOCK_SIZES:
            with self.subTest(block_sizes=block_sizes):
                result = _full_slice_kernel(block_sizes)(q)
                torch.testing.assert_close(result, expected, atol=1e-04, rtol=1e-04)

    def test_without_the_criterion_it_raises(self) -> None:
        """(2) FAIL-CAP for (1): suppress the criterion and the refusal returns.

        Without this arm, (1) could be passing because the kernel stopped reaching
        the decline at all rather than because step 2.5 licensed the restore.
        """
        q = self._q()
        with _suppress_fold_recording():
            for block_sizes in _FOLD_BLOCK_SIZES:
                with (
                    self.subTest(block_sizes=block_sizes),
                    self.assertRaises(BackendUnsupported),
                ):
                    _full_slice_kernel(block_sizes)(q)

    def test_broadcast_store_still_raises(self) -> None:
        """(3) The criterion did not merely loosen the refusal.

        No ``_emit_cute_matmul_n_collapse`` fold is recorded for this kernel (its M
        and N are distinct blocks), so step 2.5 declines and step 3 raises.
        """
        torch.manual_seed(0)
        for n in (256, 512):
            with self.subTest(n=n):
                x = torch.randn([64, 64], device=DEVICE, dtype=torch.float32)
                y = torch.randn([64, n], device=DEVICE, dtype=torch.float32)
                with self.assertRaises(BackendUnsupported):
                    _broadcast_store_kernel()(x, y)

    def test_restoring_the_broadcast_store_is_wrong(self) -> None:
        """(4) FAIL-CAP for (3): the raise there is load-bearing, not cautious.

        Forcing the restore is exactly "revert step 3's raise".  MEASURED rel
        1.04-1.05: every lane writes its own partial sum instead of the folded
        total.  This arm is why that fallback was rejected.
        """
        torch.manual_seed(0)
        n = 256
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.float32)
        y = torch.randn([64, n], device=DEVICE, dtype=torch.float32)
        expected = (x @ y).sum(dim=-1, keepdim=True).expand(64, n)
        with _force_restore_at_step3():
            result = _broadcast_store_kernel()(x, y)
        relative = (
            (result - expected).abs().max() / expected.abs().max().clamp_min(1e-12)
        ).item()
        # Not "assert it differs" -- assert it is WRONG BY A LOT, so a future change
        # that merely perturbs numerics cannot make this arm pass vacuously.
        self.assertGreater(
            relative,
            0.5,
            f"the per-lane restore was expected to be grossly wrong here, got "
            f"rel={relative:.3e}; if this is now correct the raise may be removable",
        )

    @unittest.expectedFailure
    def test_consumed_collapse_fold_is_double_reduced(self) -> None:
        """⛔ A PRE-EXISTING UPSTREAM WRONG ANSWER, pinned here so it is not lost.

        This is ``_full_slice_kernel`` with ONE change -- the store consumes the
        reduced value (``s * 2.0``) instead of storing it directly:

            s = attn.sum(-1)
            out[tile_n, :] = s * 2.0        # ← the only difference

        MEASURED WRONG ON ALL THREE TREES: rel 6.34 at the merge base
        (``a1e9642e5``), 7.09 at committed ``HEAD`` (``81b602b04``), 7.75 on the
        working tree.  So it is **not** attributable to this branch and **not** to
        step 2.5 -- the trace shows it never reaches the decline at all (the
        collapse fold is emitted, then no step-3 decision is made).

        ⭐ The signature is diagnostic: every output column holds the SAME value
        where the reference has distinct ones, i.e. the lane axis was collapsed a
        SECOND time on top of the matmul's already-complete fold, then broadcast.
        So the marker's fold is redundant here too, but nothing deletes it and
        nothing skips it -- a DOUBLE reduction.

        ``expectedFailure`` rather than a fix: root-causing this needs the
        consumed-marker path in the collapse lowering, which is a different piece
        of work from the decline ladder.  ⚠ **If this test starts XPASSing, that is
        the bug being fixed** -- delete the marker and keep the assertion.
        """
        torch.manual_seed(0)

        @helion.kernel(static_shapes=True, config=Config(block_sizes=[16, 16]))
        def consumed_collapse(q: torch.Tensor) -> torch.Tensor:
            n, c, d = q.size(0), q.size(1), q.size(2)
            out = torch.empty([n, c], dtype=torch.float32, device=q.device)
            for (tile_n,) in hl.tile([n]):
                attn = hl.zeros([tile_n, c, c], dtype=torch.float32)
                for tile_d in hl.tile(d):
                    qt = q[tile_n, :, tile_d]
                    attn = torch.baddbmm(attn, qt, qt.transpose(-2, -1))
                s = attn.sum(-1)
                out[tile_n, :] = s * 2.0
            return out

        q = self._q()
        expected = self._full_slice_reference(q) * 2.0
        torch.testing.assert_close(
            consumed_collapse(q), expected, atol=1e-04, rtol=1e-04
        )


@onlyBackends(["cute"])
class TestLaneReduceMarkerCarriesItsObligation(TestCase):
    """⭐ TASK 2 STEP 1: the marker says it owes a fold, and the SAFETY NET refuses.

    The class above pins the raise at ONE decline site (``_split_one_lane_loop``'s layered
    split).  This class pins the other, later one -- ``restore_unprocessed_lane_reduce_markers``
    -- and it matters because that function catches the shapes the split pass **cannot
    see**: a marker inside a ``range_constexpr`` vec loop (the TV path) or inside an ``if``
    is not a direct child of a lane loop, so the split pass never looks at it and it used to
    arrive here to be silently reverted to ``R = IN``.

    ⛔ WHY ``R = IN`` IS A WRONG ANSWER AND NOT A DEGRADATION.  Read as IR,
    ``MARKER(v, 'max')`` is a valid ONE-ELEMENT reduction, so ``mx = v`` is a *faithful*
    lowering of it -- which is exactly why the bug was plausible instead of loud.  The
    marker owes TWO combines (a serial lane fold and a cross-thread ``warp_reduction_*``)
    and the revert emits neither: measured as ``exp(v - v) == 1.0``, softmax rows summing
    to 128.0 instead of 1.0, and relerr 7.685 on ``matmul_layernorm`` at N=512.

    ⚠ THESE TESTS DRIVE THE PASS DIRECTLY, ON A HAND-BUILT MARKER, and that is deliberate
    for the same reason the class above monkeypatches: no real kernel reaches this site
    today, so a test that only ran kernels would pass vacuously and prove nothing.  Driving
    the function is what makes the raise's liveness checkable.

    The five tests are a set:
      1. ``test_partial_marker_raises``              -- the raise fires.
      2. ``test_complete_marker_still_reverts``      -- FAIL-CAPABILITY for (1): the raise is
         conditioned on the PAYLOAD, not on merely reaching this function.  Without it, (1)
         would also pass if the raise had been made unconditional -- the regression that
         broke all 8 attention examples, twice.
      3. ``test_field_survives_the_round_trip``      -- the emitter's field is what the parser
         reads back, so (1) and (2) are testing the channel and not a default.
      4. ``test_a_nine_arg_marker_is_still_seen``    -- the arity guard.  The recogniser tests
         ``len(args)`` for EQUALITY and answers "not a marker" on a mismatch, so an
         un-updated emission site would make the marker INVISIBLE and leak an undefined
         ``_helion_lane_reduce`` call into the kernel.  Old spellings stay recognised.
      5. ``test_old_behaviour_is_the_wrong_answer``  -- REVERT-VERIFY: reinstate the old
         "ignore the obligation" behaviour and confirm the wrong shape returns.
    """

    def _marker_stmt(self, partial: bool) -> ast.stmt:
        expr = tile_strategy._lane_reduce_marker_expr(
            "v_0",
            "sum",
            "cutlass.Float32(0)",
            32,
            acc_dtype_str="cutlass.Float32",
            partial_fold=partial,
        )
        return ast.parse(f"acc_res = {expr}").body[0]

    def test_partial_marker_raises(self) -> None:
        with self.assertRaises(BackendUnsupported) as caught:
            tile_strategy.restore_unprocessed_lane_reduce_markers(
                [self._marker_stmt(True)]
            )
        message = str(caught.exception)
        # The diagnostic must name the reduction and the obligation, not just fail.
        self.assertIn("'sum'", message)
        self.assertIn("cross-thread combine", message)

    def test_complete_marker_still_reverts(self) -> None:
        """FAIL-CAPABILITY for the test above: a marker that owes nothing still reverts.

        Nothing in the tree emits ``partial_fold=False`` today, which is the point -- this
        arm exists to prove the raise reads the field rather than firing on arrival.
        """
        out = tile_strategy.restore_unprocessed_lane_reduce_markers(
            [self._marker_stmt(False)]
        )
        src = ast.unparse(out[0])
        self.assertNotIn("_helion_lane_reduce", src)
        self.assertEqual(src, "acc_res = v_0")

    def test_field_survives_the_round_trip(self) -> None:
        for partial in (True, False):
            with self.subTest(partial_fold=partial):
                parsed = tile_strategy._is_lane_reduce_marker_assign(
                    self._marker_stmt(partial)
                )
                self.assertIsNotNone(parsed)
                assert parsed is not None
                self.assertIs(parsed.partial_fold, partial)

    def test_a_nine_arg_marker_is_still_seen(self) -> None:
        """⚠ THE ARITY GUARD.  ``len(args) != N -> None`` means "not a marker", so a stale
        emission site would make its marker INVISIBLE to both the split pass and this
        safety net, and leak an undefined ``_helion_lane_reduce`` call into the emitted
        kernel (a DSL NameError, not a helion diagnostic).  The pre-task-2 9-arg spelling
        must therefore stay recognised, and must default to the SAFE polarity.
        """
        old = ast.parse(
            "acc_res = _helion_lane_reduce(v_0, 'sum', cutlass.Float32(0), 32, "
            "1, 0, '', 1, 'cutlass.Float32')"
        ).body[0]
        parsed = tile_strategy._is_lane_reduce_marker_assign(old)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertIs(parsed.partial_fold, True)

    def test_old_behaviour_is_the_wrong_answer(self) -> None:
        """REVERT-VERIFY: reinstate "ignore the obligation" and watch the bug return.

        Patches the recogniser to clear the field, which is precisely the pre-task-2 state
        (no obligation was recorded at all), then asserts the emitted statement is the raw
        per-lane input with BOTH combines gone -- the class-8 P1 signature.  Finally proves
        the patch was undone, so a later test cannot inherit it.
        """
        real = tile_strategy._is_lane_reduce_marker_assign

        def blind(stmt: ast.AST) -> object:
            parsed = real(stmt)
            if parsed is not None:
                parsed.partial_fold = False
            return parsed

        with unittest.mock.patch.object(
            tile_strategy, "_is_lane_reduce_marker_assign", blind
        ):
            out = tile_strategy.restore_unprocessed_lane_reduce_markers(
                [self._marker_stmt(True)]
            )
            src = ast.unparse(out[0])
            self.assertEqual(src, "acc_res = v_0")
            self.assertNotIn("warp_reduction", src)
            self.assertNotIn("lane_acc", src)
        # ⭐ Prove the revert landed, or this test leaves the raise disabled for the rest
        # of the session and every later assertion about it is vacuous.
        with self.assertRaises(BackendUnsupported):
            tile_strategy.restore_unprocessed_lane_reduce_markers(
                [self._marker_stmt(True)]
            )


@onlyBackends(["cute"])
class TestLaneCarriedPhiRecord(TestCase):
    """⭐ TASK 2 STEP 2's PRODUCER: a loop-carried value announces itself at the ``_phi``.

    ``_phi`` is how helion spells "these two names are one loop-carried value", and its
    codegen handler already calls ``merge_variable_names`` -- the ``X_copy = X`` text that
    ``_markers_feed_cross_lane_carry`` scans for is that call's OUTPUT.  So the fact was
    available one phase earlier, and
    ``language/_tracing_ops.py::_record_cute_lane_carried_phi`` now records it on
    ``CuteDeviceFunctionState.lane_carried_fx_nodes``.

    ⛔⛔ AND THE CONSUMER DELIBERATELY DOES NOT USE IT YET -- that is a MEASURED FINDING,
    pinned here so it cannot be "tidied up" by someone deleting the scan.  Swapping the
    consumer over changes the answer on ``matmul_layernorm`` (the only kernel in the example
    suite that reaches the predicate) from False to True, which FAILS
    ``test_matmul_layernorm_static_shapes`` and turns
    ``test_consumed_collapse_fold_is_double_reduced`` from xfail into a hard failure.
    Reason: ``DeviceGridState.lane_loops`` stays non-empty for the whole device body once a
    lane loop is registered, so a phi whose real carrier is an inner SERIAL loop (the
    matmul's K accumulator) is attributed to the lane loop.  ⇒ the producer must learn which
    loop it is being emitted INTO before the scan can go.

    These tests pin the producer's OBSERVABLE behaviour so the follow-up has a baseline:
    it must fire on a real cross-lane recurrence and stay silent otherwise.
    """

    def test_producer_records_a_real_cross_lane_recurrence(self) -> None:
        """Attention's online-softmax ``m_i``/``l_i`` carry must be recorded.

        This is the shape whose carry is genuinely cross-lane, and the one the historical
        ``_markers_feed_cross_lane_carry`` docstring names as the reason the per-lane
        restore must stay legal there.
        """
        from examples.attention import attention

        shape = (2, 4, 256, 64)
        q = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
        bound = attention.bind((q, k, v))
        seen: dict[str, int] = {}
        real = _tracing_ops._record_cute_lane_carried_phi

        def spy(state: object) -> None:
            real(state)
            cute_state = getattr(state.device_function, "_cute_state", None)  # pyrefly: ignore [missing-attribute]
            if cute_state is not None:
                seen.update(
                    {k2: len(v2) for k2, v2 in cute_state.lane_carried_fx_nodes.items()}
                )

        with unittest.mock.patch.object(
            _tracing_ops, "_record_cute_lane_carried_phi", spy
        ):
            bound.to_triton_code(bound.env.config_spec.default_config())
        self.assertTrue(
            seen, "the phi producer recorded NOTHING on an online-softmax recurrence"
        )
        self.assertTrue(
            all(count > 0 for count in seen.values()),
            f"a lane var was registered with an empty carry set: {seen}",
        )

    def test_producer_is_silent_without_a_cross_lane_carry(self) -> None:
        """⛔ FAIL-CAPABILITY: a recorder that fired on everything would prove nothing.

        The ``sum(exp(v - amax(v)))`` arange shape has two dependent reductions and NO
        cross-lane carried accumulator, so the record must stay empty.  Together with the
        test above this is what makes the producer's output evidence rather than noise.
        """
        kernel = _make_kernel()
        x = torch.randn(_M, _N, device=DEVICE, dtype=torch.float32)
        seen: dict[str, int] = {}
        real = _tracing_ops._record_cute_lane_carried_phi

        def spy(state: object) -> None:
            real(state)
            cute_state = getattr(state.device_function, "_cute_state", None)  # pyrefly: ignore [missing-attribute]
            if cute_state is not None:
                seen.update(
                    {k2: len(v2) for k2, v2 in cute_state.lane_carried_fx_nodes.items()}
                )

        with unittest.mock.patch.object(
            _tracing_ops, "_record_cute_lane_carried_phi", spy
        ):
            kernel.bind((x,)).to_triton_code(_P1_CONFIG)
        self.assertEqual(seen, {}, f"recorded a carry where there is none: {seen}")

    def test_the_record_reader_is_backend_scoped(self) -> None:
        """``_cute_lane_carried_records`` must return None rather than raise off-cute.

        It is a read of OPTIONAL evidence: every caller falls back to the scan when it
        answers None, so failing to obtain the record must never propagate an exception out
        of an AST pass.  Called with no active DeviceFunction, which is the degenerate case.
        """
        self.assertIsNone(tile_strategy._cute_lane_carried_records())


def _tiled_rowsub_kernel() -> object:
    """A 2-D ``hl.tile`` whose inner axis is reduced, then broadcast-subtracted.

    The shape that reaches BOTH defects below: ``CuteNDTileStrategy.codegen_grid``
    pre-populates ONE lane loop per tile block whose ``block_size > num_threads``, so at
    ``block_sizes=[32,32]`` with small ``num_threads`` the grid carries TWO lane loops
    before the body lowers, and the reduced axis sits ABOVE a sibling thread axis on the
    linear thread index (so the reduce group is STRIDED).
    """

    @helion.kernel(static_shapes=True)
    def kernel(x: torch.Tensor) -> torch.Tensor:
        m, n = x.shape
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tm, tn in hl.tile([m, n]):
            v = x[tm, tn]
            out[tm, tn] = v - torch.sum(v, dim=-1, keepdim=True)
        return out

    return kernel


_ROWSUB_BLOCK = 32


def _rowsub_pm1_input(m: int, n: int) -> torch.Tensor:
    """``{-1,+1}`` data, so a CORRECT kernel must be BIT-EXACT.

    A tolerance-based check is the wrong instrument for these defects: both fold the
    wrong *lanes* together, which on random floats is a plausible-looking small error.
    With every element ``±1`` and the tile sums integral, any mis-grouping moves the
    result by a whole integer and ``maxdiff == 0`` becomes a yes/no question.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    return (
        torch.randint(0, 2, (m, n), device=DEVICE, generator=gen, dtype=torch.int32) * 2
        - 1
    ).to(torch.float32)


def _rowsub_reference(x: torch.Tensor, block: int) -> torch.Tensor:
    """Reduce PER TILE, not per row.

    ⚠ The obvious reference (``x - x.sum(-1, keepdim=True)``) is WRONG here and cost a
    reviewer real time: ``torch.sum`` inside ``hl.tile([m, n])`` reduces the *tile's* n
    extent, so with ``n > block`` the full-row sum differs from the kernel's contract and
    manufactures failures at every config.
    """
    m, n = x.shape
    out = x.clone()
    for i0 in range(0, m, block):
        for j0 in range(0, n, block):
            blk = x[i0 : i0 + block, j0 : j0 + block]
            out[i0 : i0 + block, j0 : j0 + block] = blk - blk.sum(-1, keepdim=True)
    return out


def _rowsub_config(num_threads: list[int]) -> Config:
    return Config(
        block_sizes=[_ROWSUB_BLOCK, _ROWSUB_BLOCK],
        num_threads=num_threads,
        cute_vector_widths=[1, 1],
        loop_orders=[[0, 1]],
    )


@onlyBackends(["cute"])
class TestTwoLaneLoopTileReductionIsCorrect(TestCase):
    """Two defects that stacked on one path: a decline that leaked, and a wrong fold.

    Both were reachable from ``_tiled_rowsub_kernel`` with no monkeypatching, and neither
    was covered by any gate level -- the frozen basis is 40 rolled/ND-tile reduction cells,
    none of which registers two lane loops.

    1. **The decline leaked.**  ``_emit_inline_lane_reduce`` tested
       "no synthetic axis and != 1 registered lane loop" at the SEAL, i.e. *after* it had
       already appended ``acc = identity`` to the segment prefix and emitted
       ``acc = combine(acc, v)``.  Returning there left a half-built reduction with no
       seed and no cross-thread combine, and the downstream
       ``_has_extra_cross_lane_carry`` raised ``BackendUnsupported`` on the orphan -- so a
       kernel that compiles on ``origin/main`` did not compile here.  The predicate now
       runs before any emission.
    2. **The fallback it declines to was itself wrong.**
       ``_lane_loop_cross_warp_group_params`` returned ``None`` for
       ``group_span <= 32``, which hands the caller a *consecutive-lane* warp reduce even
       though ``pre > 1`` means the group is strided -- folding different rows together.
       The single-warp strided form (``_cute_grouped_reduce_warp``) already existed and is
       already dispatched by ``_finalize_lane_reduce_marker``; only the producer refused
       to ask for it.

    ⚠ THE TWO ARE SEQUENCED, WHICH IS WHY THEY ARE PINNED TOGETHER: fixing (1) alone
    routes these configs onto (2) and converts a loud raise into a SILENT WRONG ANSWER.
    ``test_products_at_or_below_one_warp_are_bit_exact`` is the arm that would catch that
    regression, so do not split this class.
    """

    def test_compiles_and_is_bit_exact_across_num_threads(self) -> None:
        """The kernel must compile AND be bit-exact at every legal ``num_threads``.

        ``[8,8]`` and above exercise defect 1 only (they were already correct once
        compiling); ``[4,8]`` / ``[8,4]`` exercise both.
        """
        kernel = _tiled_rowsub_kernel()
        x = _rowsub_pm1_input(64, 64)
        ref = _rowsub_reference(x, _ROWSUB_BLOCK)
        for num_threads in ([4, 8], [8, 4], [8, 8], [16, 8], [8, 16], [16, 16]):
            with self.subTest(num_threads=num_threads):
                got = helion.kernel(
                    kernel.fn, config=_rowsub_config(num_threads), static_shapes=True
                )(x)
                self.assertEqual(
                    float((got - ref).abs().max()),
                    0.0,
                    f"num_threads={num_threads} is not bit-exact; a mis-grouped "
                    f"cross-thread fold moves the result by a whole integer",
                )

    def test_products_at_or_below_one_warp_are_bit_exact(self) -> None:
        """⭐ Defect 2 in isolation -- the arm that must not be dropped.

        ``num_threads`` products <= 32 are the regime
        ``_lane_loop_cross_warp_group_params`` used to decline.  They are separated from
        the sweep above because they are the ones that fail *silently* rather than loudly:
        if this passes while the emission below regresses, the kernel is returning
        confident wrong numbers.
        """
        kernel = _tiled_rowsub_kernel()
        x = _rowsub_pm1_input(64, 128)
        ref = _rowsub_reference(x, _ROWSUB_BLOCK)
        for num_threads in ([4, 8], [8, 4], [16, 2], [2, 16]):
            with self.subTest(num_threads=num_threads):
                got = helion.kernel(
                    kernel.fn, config=_rowsub_config(num_threads), static_shapes=True
                )(x)
                self.assertEqual(float((got - ref).abs().max()), 0.0)

    def test_a_strided_single_warp_group_emits_the_grouped_warp_reduce(self) -> None:
        """STRUCTURAL half, so the numeric arms above cannot pass vacuously.

        A numeric-only test would still pass if the emitter silently switched to some
        other correct-but-unintended form.  Pin the actual dispatch:
        ``group_span == 32`` (``pre=8`` x ``reduce_extent=4``) must take the single-warp
        strided helper, and a plain consecutive-lane ``warp_reduction_*`` must NOT appear
        -- that is precisely the form that folded different rows together.
        """
        kernel = _tiled_rowsub_kernel()
        x = _rowsub_pm1_input(64, 64)
        code = kernel.bind((x,)).to_triton_code(_rowsub_config([4, 8]))
        self.assertIn("_cute_grouped_reduce_warp(", code)
        self.assertNotIn("warp_reduction_", code)

    def test_a_cross_warp_group_still_takes_the_two_stage_helper(self) -> None:
        """NO-REGRESSION: widening the producer must not steal the ``> 32`` case.

        ``group_span`` a multiple of 32 greater than 32 is genuinely cross-warp and cannot
        be folded by one shuffle, so it must keep the two-stage shared helper.  This is the
        arm that fails if the ``group_span`` predicate is widened too far.
        """
        kernel = _tiled_rowsub_kernel()
        x = _rowsub_pm1_input(64, 64)
        code = kernel.bind((x,)).to_triton_code(_rowsub_config([8, 8]))
        self.assertIn("_cute_grouped_reduce_shared_two_stage(", code)

    def test_the_decline_is_taken_before_any_emission(self) -> None:
        """MECHANISM: the inline path must DECLINE this shape, not emit into it.

        Pins the fix as a *decline*, not as an accident of the emission: with two
        registered lane loops and no synthetic axis, ``_emit_inline_lane_reduce`` returns
        ``None`` and the marker path handles the kernel.  If a future change makes the
        inline path accept two lane loops it should seal per-axis rather than leak, and
        this assertion is the tripwire that forces the question to be asked.
        """
        from helion._compiler.reduction_strategy import ReductionStrategy

        kernel = _tiled_rowsub_kernel()
        x = _rowsub_pm1_input(64, 64)
        original = ReductionStrategy._emit_inline_lane_reduce
        seen: list[tuple[int, bool]] = []

        def spy(self, state, *args, **kwargs):  # type: ignore[no-untyped-def]
            grid = state.codegen.current_grid_state
            n_loops = len(getattr(grid, "lane_loops", []) or [])
            result = original(self, state, *args, **kwargs)
            seen.append((n_loops, result is None))
            return result

        with unittest.mock.patch.object(
            ReductionStrategy, "_emit_inline_lane_reduce", spy
        ):
            kernel.bind((x,)).to_triton_code(_rowsub_config([4, 8]))

        self.assertTrue(seen, "the inline lane-reduce path was never consulted")
        # Every call on this shape sees 2 lane loops and must decline.
        self.assertEqual(
            [(n, declined) for n, declined in seen if n != 1],
            [(n, True) for n, _ in seen if n != 1],
            f"a call with != 1 lane loop did NOT decline: {seen}",
        )
        self.assertIn(2, [n for n, _ in seen], f"expected 2 lane loops, got {seen}")


if __name__ == "__main__":
    unittest.main()
