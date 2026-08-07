"""Tests for the two CuTe AST-pass CONFIG KNOBS.

Two passes added by the CuTe reduction rework were originally reachable only
through an environment variable:

* ``defer_online_merge`` (``HELION_DISABLE_DEFER_ONLINE_MERGE=1``) -- where an
  online ``(max, sum-of-exp)`` recurrence combines across lanes;
* ``fuse_tv_copy_sweeps`` (``HELION_TV_SWEEP_FUSE=disabled`` plus the
  ``_DEFAULT_MAX_CACHE_SLOTS = 128`` cap and its
  ``HELION_TV_SWEEP_FUSE_SLOTS`` override) -- how many per-thread register
  slots may be spent caching a row across repeated sweeps.

An env var is invisible to the autotuner and is not recorded in a winning
``Config``, so a config could not say which of those decisions produced it.  Both
are now real, searchable ``Config`` keys:

* ``cute_online_defer``  -- ``CuteOnlineDeferSpec``,   ``EnumFragment((True, False))``
⛔ ``cute_tv_sweep_cache`` WAS THE SECOND KNOB HERE AND IS DELETED (run 2): it was a
per-thread register BUDGET whose decline could turn a config's named ``cute_row_residency``
into a different emitted one.  Its tests went with it; what a row's footprint now costs is
paid in spills under the residency that was asked for, not swapped for another kernel.

WHAT THESE TESTS ARE FOR.  Promoting a hardcoded decision to a knob is a
REACHABILITY change, so the properties worth pinning are:

1. the default reproduces today's kernel EXACTLY (``test_*_default_*``);
2. both values are actually reachable and reach DIFFERENT kernels -- a knob that
   does not change anything is documentation, not a tunable
   (``test_*_off_*`` / ``test_*_budget_*``);
3. an illegal value fails LOUDLY rather than emitting a wrong kernel
   (``test_*_rejects_*``);
4. an omitted key means "use the ladder", not "transform off" -- the specs' own
   ``_fill_missing`` (``test_*_omitted_*``);
5. the keys are cute-only and are REJECTED elsewhere (``test_*_triton_*``);
6. the values survive the flat round trip the autotuner uses
   (``test_*_flat_round_trip``).

Lives in:
- ``helion/autotuner/config_spec.py`` (the two spec classes + 5 registration sites)
- ``helion/_compiler/device_ir.py`` (``_register_cute_ast_pass_specs``)
- ``helion/_compiler/device_function.py`` (``_cute_pass_knob_by_offset_var``)
- ``helion/_compiler/cute/tv_layout.py`` (the domains and the ladders)
- ``helion/_compiler/cute/defer_online_merge.py`` (``disabled_offsets``)
- ``helion/_compiler/cute/fuse_tv_copy_sweeps.py`` (``slot_budgets``)
"""

from __future__ import annotations

from typing import ClassVar
import unittest.mock

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.exc import InvalidConfig
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute")
def _online_recurrence(x: torch.Tensor) -> torch.Tensor:
    """``cross_entropy_online``'s shape: a loop-carried ``(max, sum-of-exp)`` pair.

    This is the ONLY shape ``defer_online_merge`` matches, and the reduction axis is
    an inner ``hl.tile`` rather than a reduction block -- which is exactly why
    ``cute_online_defer`` is registered per DEVICE LOOP and not per reduction block.
    """
    m, n = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n].to(torch.float32)
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        out[tile_m] = mi + torch.log(di)
    return out


@helion.kernel(backend="cute")
def _two_moment_norm(x: torch.Tensor) -> torch.Tensor:
    """layer_norm's shape: a row read by SEVERAL sweeps of one rolled reduction.

    mean must exist before centered, so the row is walked more than once and every
    walk re-reads it from gmem -- the gap the sweep-cache pass closes, and therefore
    the shape the sweep-cache knob budgets.

    ⚠ THIS DOCSTRING DELIBERATELY NAMES NEITHER THE PASS NOR THE KNOB.  Helion copies
    a kernel's docstring verbatim into the emitted host wrapper, so any marker string
    a test greps for must not appear here -- an earlier draft asserted
    ``assertNotIn("_tv_sweep_cache", code)`` and it matched the docstring's own
    mention of the key, reporting the fusion as present on a kernel where it had
    correctly declined.
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


def _reduce_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(x.to(torch.float32), dim=1)


def _norm_ref(x: torch.Tensor) -> torch.Tensor:
    xf = x.to(torch.float32)
    centered = xf - xf.mean(dim=-1, keepdim=True)
    var = (centered * centered).mean(dim=-1, keepdim=True)
    return (centered * torch.rsqrt(var + 1e-5)).to(x.dtype)


@onlyBackends(["cute"])
class TestCuteOnlineDeferKnob(TestCase):
    """``cute_online_defer``: WHERE the online recurrence combines across lanes."""

    _CONFIG: ClassVar[dict[str, object]] = {
        "block_sizes": [4, 512],
        "num_threads": [0, 32],
        "cute_vector_widths": [1, 4],
    }

    # The in-loop cross-lane combine the deferral removes.  The name is the
    # dispatcher's, not this pass's, so matching it is matching the emitted
    # reduction rather than a variable this test invented.
    _IN_LOOP_REDUCE = "_cute_grouped_reduce_warp(_helion_vfold_acc_0"
    # The post-loop merge the deferral inserts.
    _DEFERRED_MERGE = "_defer_merged_"

    def _code(self, x: torch.Tensor, **overrides: object) -> str:
        code, out = code_and_output(
            _online_recurrence, (x,), **{**self._CONFIG, **overrides}
        )  # pyrefly: ignore [bad-argument-type]
        torch.testing.assert_close(out, _reduce_ref(x), atol=1e-2, rtol=1e-2)
        return code

    def test_default_defers_the_merge(self) -> None:
        """DEFAULT = deferred.  This is the whole reason the fragment's choices are
        ``(True, False)`` in that order rather than a ``BooleanFragment`` (whose
        ``default()`` is a hardcoded ``False``): promoting the pass to a knob must
        not change what the compiler emits when nobody asks for anything.
        """
        x = torch.randn(512, 2048, device=DEVICE, dtype=HALF_DTYPE)
        code = self._code(x)
        self.assertIn(self._DEFERRED_MERGE, code)
        self.assertNotIn(self._IN_LOOP_REDUCE, code)

    def test_omitted_key_uses_the_ladder(self) -> None:
        """An OMITTED key means "use the ladder", not "transform off".

        The distinction matters because ``_fill_missing`` applies its answer
        SILENTLY.  Here the ladder value is what every pre-existing config already
        compiles to (the pass ran unconditionally before the knob existed), so
        filling ``False`` would have quietly slowed every frozen config down.
        Contrast ``CuteClusterNSpec._fill_missing``, which deliberately fills the
        conservative ``1`` because a cluster CHANGES the launch geometry.
        """
        x = torch.randn(512, 2048, device=DEVICE, dtype=HALF_DTYPE)
        omitted = self._code(x)
        explicit = self._code(x, cute_online_defer=[True])
        self.assertEqual(omitted, explicit)

    def test_false_restores_the_in_loop_merge(self) -> None:
        """``False`` must reach a DIFFERENT kernel -- otherwise the knob is inert.

        The exchange is exact and visible: the in-loop cross-lane reduce comes back
        and the post-loop merge disappears.
        """
        x = torch.randn(512, 2048, device=DEVICE, dtype=HALF_DTYPE)
        code = self._code(x, cute_online_defer=[False])
        self.assertIn(self._IN_LOOP_REDUCE, code)
        self.assertNotIn(self._DEFERRED_MERGE, code)

    def test_both_values_are_numerically_correct(self) -> None:
        """Both values must COMPILE and be correct.  A knob that widens the search
        space can expose a kernel nothing has compiled before, so correctness is
        asserted at both values, not just at the default.  (``_code`` asserts it;
        this test exists to say so explicitly and to cover a second shape.)
        """
        x = torch.randn(256, 1024, device=DEVICE, dtype=HALF_DTYPE) * 8.0
        for value in (True, False):
            with self.subTest(cute_online_defer=value):
                self._code(x, cute_online_defer=[value])

    def test_env_override_still_forces_the_pass_off(self) -> None:
        """``HELION_DISABLE_DEFER_ONLINE_MERGE`` survives as a DEBUGGING OVERRIDE.

        Env vars remain useful for A/B attribution -- one variable turns the pass
        off on every cell at once, with no config surgery.  What changed is that it
        is no longer the ONLY way to reach the in-loop form.  It must WIN over the
        knob, so that "off everywhere" really is off everywhere.
        """
        x = torch.randn(512, 2048, device=DEVICE, dtype=HALF_DTYPE)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("HELION_DISABLE_DEFER_ONLINE_MERGE", "1")
            code = self._code(x, cute_online_defer=[True])
        self.assertIn(self._IN_LOOP_REDUCE, code)
        self.assertNotIn(self._DEFERRED_MERGE, code)

    def test_the_plan_channel_is_the_licence(self) -> None:
        """⭐ TASK 3's FAIL-CAPABILITY ARM: with the plan withheld, the pass must DECLINE.

        The deferral's LICENCE now comes from a plan its emitter registered
        (``DeviceFunction.register_online_defer`` -> ``cute_online_defer_plans``), keyed by
        the loop's offset variable, instead of being inferred from **sibling position**.

        ⛔ WHY THAT MATTERS: the emitted nest is
        ``for tile_offset in range(...): for lane in range(...)`` and the INNER lane loop
        satisfies every other structural predicate the outer one does, so the old inference
        could pick it -- merging mid-stream, at measured relerr **5.18** (a ~nt-fold
        overcount).  The lane loop is not a device loop and is never registered, so with the
        plan channel the wrong loop is *unrepresentable* rather than merely rejected.

        ⚠ THIS ARM IS WHAT STOPS THE POSITIVE TESTS PASSING VACUOUSLY.  If the pass silently
        stopped firing everywhere -- **this pass's exact historical failure mode: its first
        version fired on nothing at all** -- every "the deferral happened" assertion would go
        green by accident.  So: empty plans must remove the merge AND bring the in-loop
        reduce back, and the normal path must do the opposite.
        """
        from helion._compiler.cute import defer_online_merge as dom

        x = torch.randn(512, 2048, device=DEVICE, dtype=HALF_DTYPE)

        # ARM 1 -- the normal path: the plan is registered, so the deferral fires.
        granted = self._code(x, cute_online_defer=[True])
        self.assertIn(self._DEFERRED_MERGE, granted)
        self.assertNotIn(self._IN_LOOP_REDUCE, granted)

        # ARM 2 -- the plan WITHHELD.  Forced by handing the walker an empty map, which is
        # what a compile with no registered device loop would produce.  ⚠ Patched at
        # ``_walk`` rather than at the entry point because the entry point's ``plans=None``
        # default deliberately means "no channel, use the old behaviour" (the pass stays
        # callable on a hand-written body); an empty map is the real "channel present, this
        # loop not registered" case.
        real_walk = dom._walk

        def no_plans(
            body: object, groups: object, disabled: object, plans: object = None
        ) -> object:
            return real_walk(body, groups, disabled, {})

        with unittest.mock.patch.object(dom, "_walk", no_plans):
            withheld = self._code(x, cute_online_defer=[True])
        self.assertNotIn(self._DEFERRED_MERGE, withheld)
        # ⭐ And the in-loop reduces must be BACK -- "no merge" alone would also be true of a
        # kernel that lost its reduction entirely.
        self.assertIn(self._IN_LOOP_REDUCE, withheld)

    def test_the_producer_registers_the_outer_loop_not_the_lane_loop(self) -> None:
        """⭐ The plan map must contain the DEVICE loop's offset var, and nothing else.

        This is the structural half of the argument above: it is not enough that the pass
        consults a map, the map must not contain the loop whose deferral is wrong.  The lane
        loop's variable (``lane_*``) is emitted by the lane-nest machinery and is not a
        device loop, so it must be absent.
        """
        from helion._compiler.device_function import DeviceFunction

        x = torch.randn(512, 2048, device=DEVICE, dtype=HALF_DTYPE)
        seen: dict[str, object] = {}
        real_init = DeviceFunction.register_online_defer

        def spy(self_fn: object, offset_var: str) -> None:
            real_init(self_fn, offset_var)  # pyrefly: ignore [bad-argument-type]
            seen.update(dict.fromkeys(self_fn.cute_online_defer_plans))  # pyrefly: ignore [missing-attribute]

        with unittest.mock.patch.object(DeviceFunction, "register_online_defer", spy):
            self._code(x, cute_online_defer=[True])

        self.assertTrue(seen, "the producer registered NO device loop at all")
        lane_keys = [k for k in seen if k.startswith("lane")]
        self.assertEqual(
            lane_keys,
            [],
            f"a LANE loop was registered as deferrable: {lane_keys}. Deferring at the "
            "lane loop is the measured relerr-5.18 bug.",
        )


@onlyBackends(["cute"])
class TestCutePassKnobSpecs(TestCase):
    """Spec-level properties: domain, defaults, normalisation, round trip."""

    def _spec(self) -> object:
        x = torch.randn(1024, 1024, device=DEVICE, dtype=HALF_DTYPE)
        return _two_moment_norm.bind((x,)).config_spec

    def test_slots_registered_for_the_device_loop(self) -> None:
        spec = self._spec()
        self.assertEqual(len(spec.cute_online_defer), 1)  # pyrefly: ignore [missing-attribute]

    def test_online_defer_slot_exists_without_a_reduction_block(self) -> None:
        """⭐ THE REGISTRATION-DOMAIN REGRESSION TEST.

        ``_online_recurrence`` expresses its reduction as an inner ``hl.tile``
        (the recurrence is loop-carried), so it owns NO reduction block --
        ``reduction_loops`` is empty.  Registering these knobs per reduction block,
        as the three TV-layout knobs are, would therefore create zero slots on the
        one kernel ``cute_online_defer`` controls: a knob that looks registered and
        controls nothing.
        """
        x = torch.randn(512, 2048, device=DEVICE, dtype=HALF_DTYPE)
        spec = _online_recurrence.bind((x,)).config_spec
        self.assertEqual(len(spec.reduction_loops), 0)
        self.assertGreater(len(spec.cute_online_defer), 0)

    def test_default_config_reports_the_ladder(self) -> None:
        config = self._spec().default_config().config  # pyrefly: ignore [missing-attribute]
        self.assertEqual(config["cute_online_defer"], [True])

    def test_fragments_are_searchable(self) -> None:
        """The fragments must offer more than their default -- that is the
        difference between a tunable and a merely settable value.  ``search_choices``
        must be ``None`` (i.e. the whole domain), unlike the TV-layout knobs that
        codegen does not read yet and which are deliberately pinned.
        """
        spec = self._spec()
        defer_fragment = spec.cute_online_defer[0]._fragment(spec)  # pyrefly: ignore [missing-attribute]
        self.assertEqual(defer_fragment.default(), True)
        self.assertEqual(set(defer_fragment.choices), {True, False})
        self.assertIsNone(defer_fragment.search_choices)

    def test_normalize_fills_the_ladder_when_omitted(self) -> None:
        spec = self._spec()
        config = helion.Config(block_sizes=[1], num_threads=[1], reduction_loops=[512])
        spec.normalize(config)  # pyrefly: ignore [missing-attribute]
        self.assertEqual(config.config["cute_online_defer"], [True])

    def test_normalize_rejects_illegal_values(self) -> None:
        """A hand-written config with an illegal value must fail LOUDLY rather than
        emit a wrong kernel.  ``[1]`` for a boolean is included on purpose: ``bool``
        is a subclass of ``int``, so a lazy ``isinstance`` check would coerce it.
        """
        spec = self._spec()
        for key, value in (
            ("cute_online_defer", [1]),
            ("cute_online_defer", ["yes"]),
            ("cute_online_defer", [None]),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(InvalidConfig):
                spec.normalize(  # pyrefly: ignore [missing-attribute]
                    helion.Config(
                        block_sizes=[1],
                        num_threads=[1],
                        reduction_loops=[512],
                        **{key: value},
                    )
                )

    def test_flat_round_trip(self) -> None:
        """``cute_online_defer`` must survive ``flatten`` -> ``unflatten``, or the autotuner
        cannot carry it and a winning config would not reproduce.

        ⚠ THE ROUND TRIP IS ASSERTED AS A **FIXED POINT** of ``normalize()``, not as
        "the input survives verbatim", and that is the property the autotuner actually needs
        -- it flattens configs that have already been normalized.

        ⭐ WHY THAT DISTINCTION IS LOAD-BEARING, kept because the lesson outlives the knob:
        ``flatten -> unflatten`` writes EVERY key explicitly, so a config that named only one
        key comes back naming all of them, each filled from its ladder.  An earlier draft of
        the residency work *rejected* a key pair it considered contradictory, and this test is
        what caught it -- the raise fired inside ``unflatten``, i.e. on a config the SEARCH
        can draw.  ⇒ a knob combination the search can produce must never raise at normalize
        time.  (The pair in question was ``registers`` + a zero register budget; that budget
        knob is now deleted, so the pair is unrepresentable and the reconciliation it needed
        is gone with it.  The fixed-point property is what remains worth pinning.)
        """
        from helion.autotuner.config_generation import ConfigGeneration

        spec = self._spec()
        generation = ConfigGeneration(spec)  # pyrefly: ignore [bad-argument-type]
        for defer in (True, False):
            with self.subTest(defer=defer):
                config = helion.Config(
                    block_sizes=[1],
                    num_threads=[1],
                    reduction_loops=[512],
                    cute_online_defer=[defer],
                )
                spec.normalize(config)  # pyrefly: ignore [missing-attribute]
                # The post-normalize values are what the autotuner carries.
                restored = generation.unflatten(generation.flatten(config))
                self.assertEqual(restored.config["cute_online_defer"], [defer])
                # ⭐ FIXED POINT, which is the property that actually matters: normalizing
                # the restored config must not move it again.  Without this the assertion
                # above could be satisfied by any self-consistent rewrite, including one
                # that keeps drifting on each pass.
                again = dict(restored.config)
                spec.normalize(again)  # pyrefly: ignore [missing-attribute]


@onlyBackends(["triton"])
class TestCutePassKnobsAreCuteOnly(TestCase):
    """Both keys are cute-only, so another backend must REJECT them.

    ``config_spec.py`` is backend-wide, so a key that leaked past
    ``BACKEND_SPECIFIC_KEYS`` would be silently accepted (and silently ignored)
    everywhere.  Membership in that set is what turns it into an error.
    """

    def test_triton_rejects_the_pass_knobs(self) -> None:
        @helion.kernel(backend="triton")
        def double(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2
            return out

        spec = double.bind((torch.randn(64, device=DEVICE),)).config_spec
        for key, value in (
            ("cute_online_defer", [True]),
            # ``cute_row_residency`` -- the three-way row-residency axis.  Included
            # here because its three values name CuTe EMISSION ARMS (the register
            # sweep cache, the SMEM staging tile, the plain gmem re-read), none of
            # which the Triton backend has, so silently accepting it would accept a
            # key that cannot be honoured.  This is the shared-file guard: the spec
            # class lives in the backend-wide ``config_spec.py``.
            ("cute_row_residency", ["smem"]),
        ):
            with self.subTest(key=key):
                self.assertFalse(spec.supports_config_key(key))
                with self.assertRaises(InvalidConfig):
                    spec.normalize(helion.Config(block_sizes=[32], **{key: value}))

    def test_triton_default_config_has_no_pass_knobs(self) -> None:
        @helion.kernel(backend="triton")
        def double(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] * 2
            return out

        config = double.bind((torch.randn(64, device=DEVICE),)).config_spec
        defaults = config.default_config().config
        self.assertNotIn("cute_online_defer", defaults)
        self.assertNotIn("cute_row_residency", defaults)


@onlyBackends(["cute"])
class TestTvTileIdChannel(TestCase):
    """⭐ TASK 4: the tile's IDENTITY is recorded by the plan that emits the copy.

    ``fuse_tv_copy_sweeps`` is load CSE: to delete a later sweep's ``cute.copy`` it must
    prove two copies address the SAME tile.  ``ChunkTVPlan``'s ``emit_local_tile`` /
    ``emit_partition_source`` / ``emit_copy`` generate **both** sweeps' texts from **one
    plan object**, so the identity is known at emission -- and it was thrown away into a
    variable name (each sweep mints its own ``_tv_tile_N`` / ``_tv_part_N``), leaving the
    pass to unparse, inline single-assignment temporaries, and **string-compare**.

    ⚠ THIS IS LANDED AS A **WIDENING**, NOT A REPLACEMENT, AND THE MEASUREMENT IS WHY:
    on every cell where the pass fires, the text comparison already succeeds, so the id
    path matches **zero** additional producers today.  Replacing the matcher on that
    evidence would be a pure-risk edit -- it can only change behaviour where the two
    disagree, and no measured cell disagrees.  What the channel buys is task 6's
    de-risking: a per-site ``registers`` lowering branch needs exactly this "is this the
    2nd read of that tile?" fact.

    ⇒ these tests pin the CHANNEL (it must carry the right tuple, and must not lie), not a
    deletion that has not happened.
    """

    _CONFIG: ClassVar[dict[str, object]] = {
        "block_sizes": [1],
        "num_threads": [1],
        "reduction_loops": [512],
        "cute_vector_widths": [8, 1],
    }

    def _emit(self, x: torch.Tensor) -> dict[str, tuple[str, int, str, str]]:
        """The ids recorded during one compile, captured at the PRODUCER.

        ⚠ CAPTURED WHERE THEY ARE WRITTEN, NOT WHERE THEY ARE READ.  An earlier version of
        this helper hooked ``fuse._tile_ids`` (the reader) and came back EMPTY -- because
        the reader only runs on a text-match MISS, and the text matcher currently succeeds
        on every cell.  That is the measured reason task 4 is a widening rather than a
        replacement, and it is exactly the shape of vacuous test this file warns about: the
        hook proved the reader was not called, not that the channel was empty.
        """
        from helion._compiler.cute import memory_ops as cute_memory_ops

        captured: dict[str, tuple[str, int, str, str]] = {}
        real = cute_memory_ops._cute_tv_partition_hoist

        def spy(*args: object, **kwargs: object) -> object:
            out = real(*args, **kwargs)  # pyrefly: ignore [bad-argument-type]
            state = args[0]
            cute_state = getattr(state.device_function, "_cute_state", None)  # pyrefly: ignore [missing-attribute]
            if cute_state is not None:
                captured.update(cute_state.tv_tile_ids)
            return out

        with unittest.mock.patch.object(
            cute_memory_ops, "_cute_tv_partition_hoist", spy
        ):
            code_and_output(_two_moment_norm, (x,), **self._CONFIG)  # pyrefly: ignore [bad-argument-type]
        return captured

    def test_the_channel_carries_the_tile_identity(self) -> None:
        """Every recorded id must be a full ``(tensor, chunk, row, col)`` tuple.

        ``chunk`` is the plan's own field, so a wrong or missing chunk would mean the id
        describes a tile of a different width than the copy actually reads -- which for a
        CSE proof is the difference between sound and silently wrong.
        """
        x = torch.randn(1024, 1024, device=DEVICE, dtype=HALF_DTYPE)
        ids = self._emit(x)
        self.assertTrue(ids, "no tile ids were recorded at all")
        for frag, tile_id in ids.items():
            with self.subTest(frag=frag):
                self.assertEqual(len(tile_id), 4)
                tensor_name, chunk, row, col = tile_id
                self.assertIsInstance(tensor_name, str)
                self.assertGreater(chunk, 0, "chunk must be the plan's real width")
                self.assertTrue(row, "row coordinate must not be empty")
                self.assertTrue(col, "column coordinate must not be empty")

    def test_sweeps_of_one_row_share_an_id_and_distinct_tensors_do_not(self) -> None:
        """⛔ THE FAIL-CAPABILITY ARM: the id must DISCRIMINATE, not merely exist.

        An id scheme that gave every fragment the same tuple would "prove" any two copies
        read the same tile and license deleting a load that reads something else.  So both
        directions are asserted: the repeated sweeps of the reduced row must AGREE, and at
        least two different ids must exist in the kernel (the row vs the output tile) --
        otherwise the channel is a constant wearing an identity's name.
        """
        x = torch.randn(1024, 1024, device=DEVICE, dtype=HALF_DTYPE)
        ids = self._emit(x)
        distinct = set(ids.values())
        self.assertGreater(
            len(distinct),
            1,
            f"every fragment got the SAME tile id ({distinct}) -- the channel does not "
            "discriminate, so it cannot support a CSE proof",
        )
        # The reduced row is read by several sweeps, so at least one id must be shared by
        # more than one fragment var -- that sharing IS the fact the pass consumes.
        counts: dict[tuple[str, int, str, str], int] = {}
        for tile_id in ids.values():
            counts[tile_id] = counts.get(tile_id, 0) + 1
        self.assertTrue(
            any(n > 1 for n in counts.values()),
            f"no tile id is shared by two fragments, so no re-read was identified: {counts}",
        )

    # ⛔⛔ TWO TESTS REMOVED HERE: ``test_an_unrecorded_fragment_never_matches`` and
    # ``test_the_pass_stays_config_free_without_a_device_function``.
    #
    # Both poked internals of ``fuse_tv_copy_sweeps`` (``_match_by_tile_id``, ``_tile_ids``),
    # and that pass has been DELETED -- the ``registers`` residency is now decided and emitted
    # at lowering (``memory_ops._cute_tv_partition_hoist``'s rmem branch), which consults this
    # very channel.  ⇒ they tested the consumer, not the channel, so they went with it.
    #
    # ⭐ THE CHANNEL'S OWN TESTS ABOVE ARE KEPT AND ARE NOW MORE LOAD-BEARING THAN WHEN THEY
    # WERE WRITTEN.  This class's docstring predicted exactly that: "what the channel buys is
    # task 6's de-risking: a per-site ``registers`` lowering branch needs exactly this 'is this
    # the 2nd read of that tile?' fact."  It is now the live matcher rather than a widening --
    # and the id being keyed on the chunk index EXPRESSION (not the per-sweep variable name) is
    # what makes two sweeps of one row compare equal.  MEASURED when the tile path was missing
    # that expression: 2 first-reads and 0 later-reads, i.e. the row published twice and served
    # never.  ⇒ do not weaken ``test_sweeps_of_one_row_share_an_id_and_distinct_tensors_do_not``.


if __name__ == "__main__":
    import unittest

    unittest.main()
