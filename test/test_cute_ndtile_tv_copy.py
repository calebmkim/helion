"""``CuteNDTileStrategy`` emits a vectorized ``cute.copy`` through a ``ChunkTVPlan``.

This is the TV-layout capability on a **non-reduction looping strategy**.  Before
it, the ``make_tiled_copy_tv`` / ``cute.copy`` emission was reachable only from
``LoopedReductionStrategy``; a tiled reduction axis (``x[tile_m, tile_n]``) took
the hand-built ``cute.arch.load`` route instead, even though NDTile already owned
the lane body and the constexpr-V loop the protocol needs.

Two things had to learn the tiled-axis shape TOGETHER, and the tests below pin
both, because a plan built at a width no emission site honours is exactly the
"index and access width disagree" bug class:

* the PLAN side -- ``CuteNDTileStrategy._cute_tv_participants``, which selects
  participants by ``block_id`` identity rather than by the subscript's syntactic
  form (a rolled reduction's axis is a bare ``slice(None)``; a tiled axis is a
  ``SymInt``, and in the device IR an ``fx.Node`` wrapping one);
* the SITE side -- ``memory_ops._cute_tv_site_eligible``, which asks the strategy
  which axis its plan addresses (``cute_tv_lane_block_id``) instead of requiring a
  literal slice.

⚠ THE PATH IS OPT-IN (``HELION_CUTE_NDTILE_TV=1``) and these tests set it
explicitly.  Turning it on by default is a REPLACEMENT of an established
emission, not a widening: MEASURED, it displaces the ``cute.arch.load`` form
pinned by three tests in ``test_cute_tile_loop_vec_hoist.py`` and changes 8 of
the 40 frozen perf cells (all ``cross_entropy_online/*``, attributed by an A/B
toggling only this gate).  Which instruction mix is faster is a timing question;
these tests pin only that the capability is CORRECT and, with the gate unset,
INERT.
"""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING
from typing import ClassVar
import unittest.mock

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@contextlib.contextmanager
def _tv_enabled() -> Iterator[None]:
    """Switch the NDTile TV path on for the duration of one test.

    ⚠ SCOPED, and that is not hygiene -- it is the only way the default-off test in
    this file can be trusted.  A leaked ``HELION_CUTE_NDTILE_TV`` makes
    ``test_tv_copy_is_off_by_default`` assert against an emission produced with the
    gate ON, which is exactly the failure this file saw before the env handling was
    scoped: the inert half reported the positive half's artifact.
    """
    with unittest.mock.patch.dict("os.environ", {"HELION_CUTE_NDTILE_TV": "1"}):
        yield


@helion.kernel(backend="cute", static_shapes=True)
def _ndtile_rowsum(x: torch.Tensor) -> torch.Tensor:
    """A row sum whose reduction axis is an INNER ``hl.tile``, not a slice.

    That is what puts the axis on ``CuteNDTileStrategy`` rather than on
    ``LoopedReductionStrategy``: ``x[tile_m, tile_n]`` indexes the reduction axis
    with a ``SymInt`` naming a tile block, so the rolled reduction path -- which
    recognises its axis as a bare ``slice(None)`` -- never sees it.
    """
    m, n = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc += x[tile_m, tile_n].to(torch.float32).sum(dim=-1)
        out[tile_m] = acc
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _ndtile_two_sweep(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """TWO sweeps over the same row, both through an inner ``hl.tile`` -- an RMS norm.

    ⭐ THE SHAPE THE SINGLE-SWEEP KERNEL ABOVE CANNOT EXPRESS, and the one that exposed
    a dangling-symbol bug.  ``_ndtile_rowsum`` lowers its row ONCE, so a per-block cache
    of ``(tensor, direction) -> fragment`` can never be consulted twice; this kernel
    lowers ``x`` in the reduce sweep AND again in the consume sweep.

    ⚠⚠ ``hl.register_block_size(n)`` IS LOAD-BEARING, NOT STYLE.  MEASURED: with two
    independent ``hl.tile(n)`` loops the sweeps get **two block ids and two
    ``CuteNDTileStrategy`` instances**, so nothing is shared between them and the bug is
    unreachable.  Naming ONE registered block size gives **one block id and ONE
    instance whose ``codegen_device_loop`` runs twice** -- which is exactly the shape
    ``LoopedReductionStrategy`` presents for a two-sweep norm, and the only shape under
    which a stale cache entry can be served across a sweep boundary.
    """
    m, n = x.size()
    nb = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=nb):
            xt = x[tile_m, tile_n].to(torch.float32)
            acc += (xt * xt).sum(dim=-1)
        inv = torch.rsqrt(acc / n + eps)
        for tile_n in hl.tile(n, block_size=nb):
            xt = x[tile_m, tile_n].to(torch.float32)
            wt = weight[tile_n].to(torch.float32)
            out[tile_m, tile_n] = (xt * inv[:, None] * wt[None, :]).to(out.dtype)
    return out


# ⚠ SLOT ORDER IS [n, m], NOT [m, n], AND IT IS MEASURED.  ``hl.register_block_size(n)``
# runs BEFORE the ``hl.tile(m)`` loop, so the REDUCTION axis is block_id 0 and the grid row
# axis is block_id 1.  Writing ``block_sizes=[1, 512]`` here silently gives a reduction
# chunk of 1 and an m tile of 512 -- which still computes the RIGHT ANSWER (measured relerr
# 1.66e-03) while emitting no lane loop and no vector load at all, so every structural
# assertion would pass VACUOUSLY.  That is the same failure mode the ``lane_extent > 1``
# guard exists to prevent, reached through the config instead of through the shape.
_TWO_SWEEP_CFG: dict[str, list[int]] = {
    "block_sizes": [512, 1],
    "num_threads": [32, 0],
    "cute_vector_widths": [8, 1],
}


@onlyBackends(["cute"])
class TestCuteNDTileTvCopy(TestCase):
    def test_ndtile_emits_vectorized_tv_copy(self) -> None:
        """⭐ THE POSITIVE HALF: a ``cute.copy`` through a ``make_tiled_copy_tv``,
        at the full 128-bit atom, with NO scalar load left on the vectorized axis
        -- and a numerically correct answer.

        Every assertion is on the EMITTED ARTIFACT rather than on a config value or
        an internal flag, because those are not evidence: this tree has already had
        a cell whose ``Config`` recorded a mechanism the kernel never used.
        """
        x = torch.randn(64, 512, device=DEVICE, dtype=torch.bfloat16)
        with _tv_enabled():
            code, out = code_and_output(
                _ndtile_rowsum,
                (x,),
                block_sizes=[1, 512],
                num_threads=[0, 32],
                cute_vector_widths=[1, 8],
            )
        torch.testing.assert_close(
            out, x.to(torch.float32).sum(dim=-1), atol=1e-2, rtol=1e-2
        )
        # (1) The TV layout itself, and it must be THE one layout: a second
        # ``get_slice`` would let the load and store legs address different
        # elements (LEDGER E001).
        self.assertEqual(code.count("cute.make_tiled_copy_tv("), 1)
        self.assertIn("cute.make_ordered_layout((1, 32), order=(1, 0))", code)
        # ``val_layout = (1, vec)`` -- a thread's 8 elements contiguous along N.
        self.assertIn("cute.make_layout((1, 8))", code)
        # (2) The copy, at vec * dtype_bits == 8 * 16 == 128 bits.
        self.assertGreaterEqual(code.count("cute.copy("), 1)
        self.assertIn("num_bits_per_copy=128", code)
        # (3) ZERO scalar loads for the vectorized axis: the point of the path is
        # that the ``cute.arch.load`` route is REPLACED, not supplemented.  Both
        # legacy forms are named so neither can creep back in.
        self.assertNotIn("cute.arch.load(", code)
        self.assertNotIn("_tile_unroll_vec_", code)
        # ...and the per-thread POINTER load too.  ``cute.arch.load`` is the vector
        # form; ``(ptr + ...).load()`` is the scalar one, and they are different
        # emitters, so excluding only the first would let the row be read twice.
        # MEASURED at this cell: zero, on BOTH gate arms -- so this assert is not
        # what distinguishes them (that is ``cute.copy``), it is what stops a future
        # change from re-adding a scalar read alongside the copy.
        self.assertNotIn(").load()", code)
        # (3b) ⭐ ``lane_extent > 1``, READ OFF THE EMITTED LOOP.  At lane_extent == 1
        # the plan's bijection admits only vec == 1, so a "TV copy" there is
        # indistinguishable from a scalar load and every assertion above would hold
        # VACUOUSLY.  EPT = 512/32 = 16 and vec = 8, so the outer lane loop must run
        # 16 // 8 == 2 times; that number is also ``plan.lane_extent``, which is the
        # identity the whole path rests on.
        lane_extents = {
            var: int(bound_)
            for var, bound_ in re.findall(r"for (lane_\d+) in range\((\d+)\)", code)
        }
        self.assertEqual(lane_extents, {"lane_1": 2}, code)
        # (4) The constexpr V-loop SURVIVES, and that is load-bearing rather than
        # incidental: it iterates the fragment's own extent, so the elements the
        # copy's width assumes are also visited by loop structure.  Eliding it
        # would make coverage depend on the copy alone.
        self.assertIn("cutlass.range_constexpr(8)", code)
        # (5) The fragment is read per element, through the plan's own partition
        # chain: ``local_tile`` -> ``partition_S`` -> ``make_rmem_tensor_like``.
        self.assertIn("cute.local_tile(x, (1, 512),", code)
        self.assertIn(".partition_S(", code)
        self.assertIn("cute.make_rmem_tensor_like(", code)

    def test_chunk_coordinate_advances_across_chunks(self) -> None:
        """The chunk's ``local_tile`` coordinate must be a VARIABLE, not ``0``.

        ⚠ THIS IS THE ASSERTION THAT SEPARATES A CORRECT MULTI-CHUNK KERNEL FROM A
        FASTER WRONG ONE.  With ``block_size < N`` the serial loop runs several
        chunks, and a constant column coordinate makes every one of them alias the
        FIRST chunk -- which reads plausible and is measurably quicker.  The same
        hazard on the rolled path was MEASURED at relerr 261.6.
        """
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        with _tv_enabled():
            code, out = code_and_output(
                _ndtile_rowsum,
                (x,),
                block_sizes=[1, 512],
                num_threads=[0, 32],
                cute_vector_widths=[1, 8],
            )
        # Correctness first: 8 chunks per row, so an aliasing coordinate would be
        # wrong by a factor of ~8 on a sum.
        torch.testing.assert_close(
            out, x.to(torch.float32).sum(dim=-1), atol=1e-2, rtol=1e-2
        )
        self.assertIn("_tv_chunk_1 = tile_offset_1 // _BLOCK_SIZE_1", code)
        self.assertIn("_tv_chunk_1", code.split("cute.local_tile(x")[1][:120])

    def test_tv_copy_is_off_by_default(self) -> None:
        """The INERT half: with the gate unset the emission is the legacy one.

        Pinned on the artifact, not as "no crash".  The capability's whole claim is
        that it changes nothing until asked for, and the only way to check that is
        to look at what is emitted.
        """
        x = torch.randn(64, 512, device=DEVICE, dtype=torch.bfloat16)
        code, out = code_and_output(
            _ndtile_rowsum,
            (x,),
            block_sizes=[1, 512],
            num_threads=[0, 32],
            cute_vector_widths=[1, 8],
        )
        torch.testing.assert_close(
            out, x.to(torch.float32).sum(dim=-1), atol=1e-2, rtol=1e-2
        )
        self.assertNotIn("make_tiled_copy_tv", code)
        self.assertNotIn("cute.copy(", code)
        # The legacy V=8 bf16 form: two 4-wide ``cute.arch.load``s.
        self.assertIn("_tile_unroll_vec_", code)
        # ⭐ THE FAIL-CAPABILITY CONTROL, and it is the reason this test is not just
        # a mirror of the positive one.  The two arms differ ONLY in the env var, so
        # running the positive test's own mechanism predicates against THIS arm's
        # source must FAIL -- which proves those predicates discriminate rather than
        # being satisfied by anything the compiler emits.  Without this, a bug that
        # made ``_tv_signature`` always report a copy would leave both arms green.
        #
        # MEASURED at this cell: gate-off gives copies=0 tv=0 bits=[] arch=2, gate-on
        # gives copies=1 tv=1 bits=[128] arch=0.
        with self.assertRaises(AssertionError):
            self._assert_tv_mechanism(code, vec=8, bits=128)
        # ...and the SAME predicates must pass on the gate-ON source, so the control
        # cannot be green merely because the helper is broken in both directions.
        with _tv_enabled():
            on_code, _ = code_and_output(
                _ndtile_rowsum,
                (x,),
                block_sizes=[1, 512],
                num_threads=[0, 32],
                cute_vector_widths=[1, 8],
            )
        self._assert_tv_mechanism(on_code, vec=8, bits=128)

    def _assert_tv_mechanism(self, code: str, *, vec: int, bits: int) -> None:
        """The mechanism predicates, in ONE place so both arms use the same ones.

        Shared deliberately: the fail-capability control above asserts that these
        FAIL on the gate-off source, and a control that re-implemented them could
        drift from what the positive test actually checks -- at which point it would
        stop being a control at all.
        """
        assert code.count("make_tiled_copy_tv") >= 1, "no make_tiled_copy_tv"
        assert code.count("cute.copy(") >= 1, "no cute.copy"
        assert f"num_bits_per_copy={bits}" in code, f"no {bits}-bit atom"
        assert "cute.arch.load(" not in code, "legacy vector load survived"
        assert ").load()" not in code, "scalar pointer load survived"
        assert f"cutlass.range_constexpr({vec})" in code, "no constexpr-V loop"
        extents = [
            int(b) for _v, b in re.findall(r"for (lane_\d+) in range\((\d+)\)", code)
        ]
        assert extents and max(extents) > 1, f"lane_extent not > 1: {extents}"

    def test_two_sweeps_each_lower_their_own_partition(self) -> None:
        """⛔ REGRESSION: a second sweep must NOT reuse the first sweep's fragment.

        ``_cute_tv_partition_hoist`` caches ``(tensor, direction) -> fragment`` and
        early-returns the cached entry, but the fragment is DECLARED inside the serial
        ``for`` loop of the sweep that minted it -- so it is out of scope in any later
        sweep.  The cache is keyed by BLOCK, and one registered block size means both
        sweeps share a key, so before the fix the second sweep's ``x`` load hit sweep 1's
        entry and the emitted kernel referenced a dead symbol.

        MEASURED before the fix -- and note it is a hard FAILURE, not a slow kernel::

            DSLRuntimeError: name '_tv_frag_0' is not defined
            💡 Using variables defined in dynamic control flow is not supported.

        ⚠ THE STRUCTURAL ASSERTION IS THE POINT.  ``code_and_output`` running at all is
        already most of the proof (the pre-fix tree raises), but "it ran" would also be
        satisfied by a future change that declined the TV path here entirely -- which is
        why the per-sweep ``local_tile`` count and the two distinct chunk vars are pinned
        too.
        """
        x = torch.randn(64, 2048, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(2048, device=DEVICE, dtype=torch.bfloat16)
        with _tv_enabled():
            code, out = code_and_output(
                _ndtile_two_sweep, (x, w, 1e-5), **_TWO_SWEEP_CFG
            )
        torch.testing.assert_close(
            out.float(),
            torch.nn.functional.rms_norm(x.float(), (2048,), w.float(), eps=1e-5),
            atol=2e-2,
            rtol=2e-2,
        )
        # ⭐ NON-VACUITY, BOTH GUARDS.  Without these the assertions below hold for a
        # kernel that took the legacy scalar path and emitted no TV copy at all.
        self._assert_tv_mechanism(code, vec=8, bits=128)
        # (1) TWO distinct chunk coordinate vars -- one per sweep.  A single var would
        # mean only one sweep ever built a chunk prefix.
        chunk_vars = sorted(set(re.findall(r"_tv_chunk_\d+", code)))
        self.assertEqual(
            len(chunk_vars), 2, f"expected one chunk var per sweep, got {chunk_vars}"
        )
        # (2) ⭐ THE FIX ITSELF: ``x`` is tiled ONCE PER SWEEP.  Before the reset, sweep 2
        # reused sweep 1's partition and emitted only ONE ``local_tile(x, ...)``.
        x_tiles = re.findall(r"cute\.local_tile\(x, ", code)
        self.assertEqual(
            len(x_tiles),
            2,
            "x must be re-tiled in each sweep; a single local_tile means the second "
            "sweep reused the first sweep's out-of-scope fragment",
        )
        # ...and each sweep's tile carries its OWN chunk coordinate, so the two are not
        # two tiles at one coordinate (which would be the aliasing shape instead).
        tiled_chunks = re.findall(
            r"cute\.local_tile\(x, \(1, \d+\), \(.*?, (\w+)\)\)", code
        )
        self.assertEqual(sorted(tiled_chunks), chunk_vars)

    def test_declines_when_vec_would_narrow(self) -> None:
        """A width the plan cannot honour is a DECLINE, never a narrower copy.

        ⭐ WHY DECLINING IS THE ONLY SOUND ANSWER.  The outer lane loop's trip count
        is fixed at ``EPT // V`` before any address is emitted, so a copy narrower
        than ``V`` would visit fewer elements than the loop assumes -- the "index
        and access width disagree" bug class exactly.  Here a sliced input whose row
        stride is ODD (4097) forces ``legal_vec``'s alignment clamp down, so the
        plan must not be built and the legacy per-element enumeration -- which is
        already complete -- must handle it.

        The same input is also the regression guard for the row stride: emitting a
        128-bit copy against an odd row stride is an IR verification failure at
        compile time, so a plan built here would not merely be slow.
        """
        base = torch.randn(64, 4097, device=DEVICE, dtype=torch.bfloat16)
        x = base[:, :4096]
        self.assertEqual(x.stride(0), 4097)
        with _tv_enabled():
            code, out = code_and_output(
                _ndtile_rowsum,
                (x,),
                block_sizes=[1, 512],
                num_threads=[0, 32],
                cute_vector_widths=[1, 8],
            )
        torch.testing.assert_close(
            out, x.to(torch.float32).sum(dim=-1), atol=1e-2, rtol=1e-2
        )
        self.assertNotIn("make_tiled_copy_tv", code)

    def test_atom_width_and_loop_bound_are_one_number(self) -> None:
        """``num_bits_per_copy``, ``val_layout`` and the V-loop bound agree at every V.

        All three are read off ``plan.vec``, so any two of them disagreeing is the
        failure this sweep would catch -- and it is the failure that matters, since
        a trip count assuming a width the copy does not use silently under-reads.
        """
        for vec, bits in ((2, 32), (4, 64), (8, 128)):
            x = torch.randn(64, 512, device=DEVICE, dtype=torch.bfloat16)
            with _tv_enabled():
                code, out = code_and_output(
                    _ndtile_rowsum,
                    (x,),
                    block_sizes=[1, 512],
                    num_threads=[0, 32],
                    cute_vector_widths=[1, vec],
                )
            torch.testing.assert_close(
                out, x.to(torch.float32).sum(dim=-1), atol=1e-2, rtol=1e-2
            )
            self.assertIn(f"num_bits_per_copy={bits}", code)
            self.assertIn(f"cute.make_layout((1, {vec}))", code)
            self.assertIn(f"cutlass.range_constexpr({vec})", code)
            self.assertNotIn("cute.arch.load(", code)


# ── THE ENV GATE, PROMOTED TO A CONFIG KNOB ───────────────────────────────────
#
# WHAT IS BEING PINNED.  ``HELION_CUTE_NDTILE_TV`` was an env var, which is the right
# instrument for asking "is this trade a win?" once, and the wrong one for the answer,
# because the answer is *sometimes*.  MEASURED on all 8 ``cross_entropy_online`` cells
# the gate moves -- one gate arm per process, position-balanced, judged on the mean
# because ``cuda.Event.elapsed_time`` is quantized to ~2.04us on this box:
#
#     lane extent 2      -> the TV path LOSES ~2%
#     lane extent 4      -> WINS ~8%
#     lane extent 8-16   -> WINS +81% to +110%
#
# so three of the eight cells prefer OFF at their current geometry and ON once
# re-tuned.  A global default cannot express that; ``cute_ndtile_tv`` (one slot per
# device loop) can.
#
# ⭐ THE PROMOTION MUST BE A PURE REACHABILITY CHANGE, and that is what these tests
# police.  Every config in the tree omits the new key and every one of them was
# measured with the env gate OFF, so "omitted" and "False" must both be byte-identical
# to today.  The knob is only useful if ``True`` also genuinely changes the emission --
# hence the fail-capable pair rather than a single positive.
@onlyBackends(["cute"])
class TestCuteNDTileTvConfigKnob(TestCase):
    def _emit(self, **extra: object) -> str:
        x = torch.randn([64, 512], device=DEVICE, dtype=HALF_DTYPE)
        w = torch.randn([512], device=DEVICE, dtype=HALF_DTYPE)
        cfg = helion.Config(**{**_TWO_SWEEP_CFG, **extra})
        return _ndtile_two_sweep.bind((x, w, 1e-5)).to_triton_code(cfg)

    @staticmethod
    def _counts(src: str) -> tuple[int, int]:
        """``(vectorized cute.copy, legacy cute.arch.load)`` -- the two emissions."""
        return src.count("cute.copy("), src.count("cute.arch.load")

    def test_the_knob_has_a_slot_on_the_tile_path(self) -> None:
        """⭐ A knob with no slot is a knob that controls nothing.

        The domain is shared with ``cute_online_defer`` deliberately: this strategy's
        loop is a plain TILE block, so a per-reduction-block registration would have
        created ZERO slots on the very kernels the knob governs.
        """
        x = torch.randn([64, 512], device=DEVICE, dtype=HALF_DTYPE)
        w = torch.randn([512], device=DEVICE, dtype=HALF_DTYPE)
        spec = _ndtile_two_sweep.bind((x, w, 1e-5)).config_spec
        self.assertGreater(
            len(spec.cute_ndtile_tv),
            0,
            "cute_ndtile_tv has no slot on an hl.tile reduction loop, so no config "
            "could ever request it",
        )

    def test_omitting_the_key_is_byte_identical_to_false(self) -> None:
        """⭐ THE INERTNESS ARM — this is what makes the promotion safe to land.

        Every frozen config, every ``.expected`` golden and every hand-written config
        omits this key, and all of them were measured with the env gate OFF.  If
        ``_fill_missing`` returned the other value they would all silently switch to
        the TV emission: 8 frozen perf cells move and 3 tests that pin the
        ``cute.arch.load`` form go red.
        """
        omitted = self._emit()
        explicit_false = self._emit(cute_ndtile_tv=[False])
        self.assertEqual(
            omitted,
            explicit_false,
            "omitting cute_ndtile_tv must emit exactly what cute_ndtile_tv=[False] "
            "emits, or the promotion is a behaviour change rather than a "
            "reachability change",
        )
        copies, legacy = self._counts(omitted)
        self.assertEqual(copies, 0, "the default must NOT emit a vectorized TV copy")
        self.assertGreater(legacy, 0, "the default must keep the cute.arch.load form")

    def test_true_actually_emits_the_tv_copy(self) -> None:
        """⭐ THE FAIL-CAPABLE HALF: without it the inertness arm passes vacuously.

        A knob that is inert at ``False`` **and** inert at ``True`` would satisfy the
        test above while controlling nothing at all -- which is precisely the
        "registered but dead" failure this codebase has shipped before.  Asserted on
        the COUNTS of the two rival emissions, not on presence: the legacy form must
        disappear as the TV form appears, because this is a REPLACEMENT.
        """
        copies, legacy = self._counts(self._emit(cute_ndtile_tv=[True]))
        self.assertGreater(copies, 0, "cute_ndtile_tv=[True] emitted no cute.copy")
        self.assertEqual(
            legacy,
            0,
            "cute_ndtile_tv=[True] must REPLACE the cute.arch.load form, not coexist "
            "with it -- a plan alive beside declined copies is bug class 1",
        )

    def test_the_env_var_still_overrides(self) -> None:
        """The env var is kept as a debugging override, and only ON.

        ⚠ It is how a one-variable A/B is run against a config that does not name the
        key -- which is exactly how this knob's ladder was measured.  ⭐ It can only
        turn the path ON: an explicit ``False`` losing to ``HELION_CUTE_NDTILE_TV=1``
        is a contradiction, and the env var (the debugging tool) wins LOUDLY rather
        than the config silently losing.
        """
        with _tv_enabled():
            forced = self._emit()
            forced_over_false = self._emit(cute_ndtile_tv=[False])
        for src, label in ((forced, "omitted"), (forced_over_false, "explicit False")):
            copies, legacy = self._counts(src)
            self.assertGreater(copies, 0, f"env override did not fire ({label})")
            self.assertEqual(legacy, 0, f"env override left the legacy form ({label})")

    def test_a_non_boolean_is_rejected_not_coerced(self) -> None:
        """``[1]`` must raise rather than be silently read as ``[True]``.

        ``bool`` subclasses ``int``, so a coercing check would accept ``1`` and a
        config could request the TV path without saying so.
        """
        x = torch.randn([64, 512], device=DEVICE, dtype=HALF_DTYPE)
        w = torch.randn([512], device=DEVICE, dtype=HALF_DTYPE)
        bound = _ndtile_two_sweep.bind((x, w, 1e-5))
        cfg = dict(helion.Config(**_TWO_SWEEP_CFG, cute_ndtile_tv=[1]).config)
        with self.assertRaises(InvalidConfig):
            bound.config_spec.normalize(cfg)

    def test_the_knob_and_the_env_var_agree_on_the_emission(self) -> None:
        """⭐ ONE EMISSION, TWO SPELLINGS — no second code path.

        The knob must reach the *same* kernel the env var reaches, or the promotion
        has forked the TV path in two and every measurement taken through the env var
        (including the ladder in ``CuteNDTileTvSpec``) stops applying to it.
        """
        via_knob = self._emit(cute_ndtile_tv=[True])
        with _tv_enabled():
            via_env = self._emit()
        self.assertEqual(
            via_knob,
            via_env,
            "cute_ndtile_tv=[True] and HELION_CUTE_NDTILE_TV=1 emitted different "
            "kernels; the knob must not be a second path",
        )


# ── the ONE plan constructor, and the four families that must all reach it ─────────
#
# ⭐ THE PROPERTY UNDER TEST IS "ONE OWNER", WHICH NO EMISSION ASSERTION CAN SEE.
# Four strategy families can want a TV copy, and each used to be able to acquire a width
# from a place of its own.  Emitted-code checks cannot distinguish "all four went through
# one builder" from "all four happen to agree today", so these tests SPY ON THE BUILDER
# ITSELF and assert it is the one that ran.
#
# ⚠ WHY A SPY AND NOT A GREP.  A ``grep`` for a second ``chunk_plan(`` call site would
# pass the moment someone spelled the second constructor differently (a local ``vec``
# clamp, a ``with_vec`` on a plan built elsewhere).  Counting CALLS to the one entry point,
# per family, is the property: a family that acquired a width without calling it registers
# ZERO calls and fails here.


@helion.kernel(backend="cute", static_shapes=True)
def _rolled_rowsum(x: torch.Tensor) -> torch.Tensor:
    """A row sum over a BARE SLICE — the rolled/looped and loop-free reduction shape.

    ``x[tile_m, :]`` is the spelling ``LoopedReductionStrategy`` and
    ``PersistentReductionStrategy`` own; which of the two runs is decided by
    ``reduction_loops`` (an int rolls it, ``None`` makes it loop-free).
    """
    m, n = x.size()
    out = torch.zeros([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].to(torch.float32).sum(dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _pointwise_row(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``out[m, :] = a[m, :] + b[m, :]`` — a PURE POINTWISE row, no reduction op at all.

    ⭐ AND IT IS NOT A "POINTWISE STRATEGY".  MEASURED: this dispatches to
    ``LoopedReductionStrategy`` (alongside ``CuteFlattenedTileStrategy`` for the grid),
    because a bare trailing ``slice(None)`` registers a reduction-SHAPED block whether or
    not a reduction op consumes it.  So the family that needs proving here is "a kernel
    with no reduction still reaches the one builder", not "a fourth strategy class does".
    """
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        out[tile_m, :] = x[tile_m, :] + y[tile_m, :]
    return out


@onlyBackends(["cute"])
class TestOneTvPlanConstructor(TestCase):
    """Every family that gets a TV copy gets it from ``tv_layout.build_tv_plan``."""

    # A reduction-path config in the shape the frozen cells use: the row axis carries
    # ``num_threads=1`` and the reduction axis's thread count comes from
    # ``cute_threads_per_row``.  ⚠ ``num_threads=[32]`` instead raises
    # ``block size must be divisible by num_threads`` at ``block_sizes=[1]``, which reads
    # as "the TV path declined" and would make every assertion below vacuous.
    _RED_CFG: ClassVar[dict[str, object]] = {
        "block_sizes": [1],
        "num_threads": [1],
        "cute_threads_per_row": [32],
        "cute_vector_widths": [8, 1],
    }

    @staticmethod
    @contextlib.contextmanager
    def _spy() -> Iterator[list[object]]:
        """Record every ``build_tv_plan`` call and its returned ``vec``.

        ⚠ PATCHED AT **BOTH** THE DEFINITION AND THE IMPORT SITES.  Both callers do
        ``from .cute.tv_layout import build_tv_plan``, which binds the function object
        into their own module namespace, so patching only ``tv_layout.build_tv_plan``
        would record ZERO calls while the real one ran -- an unfalsifiable pass.
        """
        from helion._compiler import reduction_strategy as _rs
        from helion._compiler.cute import tv_layout as _tvl

        calls: list[object] = []
        real = _tvl.build_tv_plan

        def spy(**kwargs: object) -> object:
            plan = real(**kwargs)  # pyrefly: ignore [bad-argument-type]
            calls.append(
                {
                    "chunk": kwargs.get("chunk"),
                    "threads_per_row": kwargs.get("threads_per_row"),
                    "vec_cap": kwargs.get("vec_cap"),
                    "exact": kwargs.get("require_exact_vec_cap"),
                    "tail": kwargs.get("tail_predicated"),
                    "vec": None if plan is None else plan.vec,
                }
            )
            return plan

        with (
            unittest.mock.patch.object(_tvl, "build_tv_plan", spy),
            unittest.mock.patch.object(_rs, "build_tv_plan", spy),
        ):
            yield calls

    def _emit_with_spy(
        self, kernel: object, args: tuple[object, ...], **cfg: object
    ) -> tuple[str, list[object]]:
        with self._spy() as calls:
            src = kernel.bind(args).to_triton_code(helion.Config(**cfg))  # pyrefly: ignore [missing-attribute]
        return src, calls

    @staticmethod
    def _strategy_classes(
        kernel: object, args: tuple[object, ...], **cfg: object
    ) -> list[str]:
        """Which reduction strategy classes get CONSTRUCTED for this config.

        ⭐ The robust form of "did the force-roll happen".  The emitted text can be made to
        look either way by a rename or by a roll to the full extent; which class the
        dispatcher built cannot.
        """
        from helion._compiler import reduction_strategy as _rs

        seen: list[str] = []
        originals = {}
        for name in ("PersistentReductionStrategy", "LoopedReductionStrategy"):
            cls = getattr(_rs, name)
            originals[name] = cls.__init__

        def make(label: str, real: object) -> object:
            def spy(self: object, *a: object, **k: object) -> None:
                real(self, *a, **k)  # pyrefly: ignore [not-callable]
                seen.append(label)

            return spy

        try:
            for name, real in originals.items():
                getattr(_rs, name).__init__ = make(name, real)  # pyrefly: ignore [bad-assignment]
            kernel.bind(args).to_triton_code(helion.Config(**cfg))  # pyrefly: ignore [missing-attribute]
        finally:
            for name, real in originals.items():
                getattr(_rs, name).__init__ = real  # pyrefly: ignore [bad-assignment]
        return seen

    @staticmethod
    def _tv_counts(src: str) -> dict[str, int]:
        """Count EVERY load/copy form, because missing one form draws a wrong conclusion.

        ``make_tiled_copy_tv`` is the TV layout's own fingerprint; ``cute.arch.load`` and
        the ``.load()`` form are the two NON-TV emissions.  A test that counted only
        ``cute.copy`` could not tell a TV copy from a staging copy.
        """
        return {
            "tiled_copy_tv": src.count("make_tiled_copy_tv"),
            "cute_copy": src.count("cute.copy("),
            "arch_load": src.count("cute.arch.load"),
            "scalar_load": src.count(").load()"),
        }

    def _x(self) -> torch.Tensor:
        return torch.randn([64, 1024], device=DEVICE, dtype=HALF_DTYPE)

    def test_looped_reduction_reaches_the_one_builder(self) -> None:
        src, calls = self._emit_with_spy(
            _rolled_rowsum, (self._x(),), reduction_loops=[256], **self._RED_CFG
        )
        self.assertEqual(len(calls), 1, f"expected ONE plan build, got {calls}")
        self.assertEqual(calls[0]["vec"], 8)
        self.assertEqual(calls[0]["chunk"], 256, "chunk is the rolled loop's block")
        self.assertGreaterEqual(self._tv_counts(src)["tiled_copy_tv"], 1)

    def test_loop_free_request_reaches_the_one_builder_via_the_ROLLED_path(
        self,
    ) -> None:
        """The fourth family: a ``reduction_loops=[None]`` REQUEST -- served PERSISTENTLY.

        ⭐⭐ THIS TEST USED TO ASSERT THE OPPOSITE, and the rewrite is the point.  It was
        ``..._via_the_ROLLED_path`` and pinned ``chunk == 512``: a loop-free request that
        wanted a TV plan owed a lane-reduce marker, so ``ConfigSpec.normalize`` answered it
        with ``reduction_loops[i] = size_hint // 2`` and routed it to
        ``LoopedReductionStrategy`` instead.  That force-roll is DELETED -- a ``[None]``
        request now yields a persistent reduction or a loud error, never a silent
        substitution -- so the old assertion described behaviour that no longer exists and
        is rewritten to the new truth rather than adjusted until it passes.

        ⚠ The invariant this test actually exists for is UNCHANGED: the family reaches
        ``build_tv_plan`` **exactly once**, whichever strategy asks.  What moved is only
        which strategy asks and with what ``chunk``.

        MEASURED at this config (``N = 1024``, ``cute_threads_per_row=[32]``,
        ``cute_vector_widths=[8, 1]``), and the CONTRAST is what makes it falsifiable:

            reduction_loops=[None]  -> chunk 1024  vec 8  roffset ABSENT   persistent
            reduction_loops=[1024]  -> chunk 1024  vec 8  roffset ABSENT   persistent
            reduction_loops=[512]   -> chunk  512  vec 8  roffset PRESENT  looped
            reduction_loops=[256]   -> chunk  256  vec 8  roffset PRESENT  looped

        i.e. ``[None]`` and an explicit full-extent chunk are now the SAME kernel, and both
        differ from a genuine chunking request.  Asserting the absence of ``roffset``
        alongside ``chunk == numel`` is what distinguishes "persistent" from "rolled to a
        chunk that happens to equal the extent".
        """
        x = self._x()
        numel = x.size(1)
        src, calls = self._emit_with_spy(
            _rolled_rowsum, (x,), reduction_loops=[None], **self._RED_CFG
        )
        self.assertEqual(len(calls), 1, f"expected ONE plan build, got {calls}")
        self.assertEqual(calls[0]["vec"], 8)
        self.assertEqual(
            calls[0]["chunk"],
            numel,
            "a loop-free request must be served at the FULL extent -- a smaller chunk "
            "means the deleted force-roll came back",
        )
        self.assertIn(calls[0]["exact"], (None, False))
        self.assertGreaterEqual(self._tv_counts(src)["tiled_copy_tv"], 1)
        self.assertNotIn(
            "roffset",
            src,
            "a persistent reduction has no outer chunk loop, so no roffset induction "
            "variable; its presence means the request was rolled",
        )
        self.assertNotIn(
            "_helion_lane_reduce",
            src,
            "the marker is discharged at lowering (one lane nest per dependency layer), "
            "not left for an AST pass",
        )

    def test_loop_free_request_and_full_extent_chunk_agree(self) -> None:
        """⛔ THE CONTRAST ARM for the test above, and it is not decoration.

        ``chunk == numel`` on its own cannot distinguish "honoured as persistent" from
        "rolled to a chunk that coincidentally equals the extent".  This pins the stronger
        property: ``[None]`` and ``[numel]`` produce the SAME emitted kernel, while a
        genuine chunking request produces a different one.

        ⚠ The second half matters as much as the first.  Without it, a tree on which the
        TV path had collapsed to one shape would pass the equality vacuously.
        """
        x = self._x()
        numel = x.size(1)
        src_none, _ = self._emit_with_spy(
            _rolled_rowsum, (x,), reduction_loops=[None], **self._RED_CFG
        )
        src_full, _ = self._emit_with_spy(
            _rolled_rowsum, (x,), reduction_loops=[numel], **self._RED_CFG
        )
        src_half, _ = self._emit_with_spy(
            _rolled_rowsum, (x,), reduction_loops=[numel // 2], **self._RED_CFG
        )
        self.assertEqual(
            src_none,
            src_full,
            "reduction_loops=[None] and [numel] are the same request and must emit the "
            "same kernel",
        )
        self.assertNotEqual(
            src_none,
            src_half,
            "a real chunking request must emit a DIFFERENT kernel -- otherwise the "
            "equality above is vacuous",
        )
        self.assertIn(
            "roffset", src_half, "the chunked arm is the one with a chunk loop"
        )
        # ⛔⛔ AND THE STRATEGY CLASS, because everything above is weaker than it looks.
        #
        # ⚠ THE EQUALITY IS CLOSE TO A TAUTOLOGY.  ``ReductionLoopSpec._normalize`` maps any
        # value ``>= size_hint`` back to ``None``, so ``[None]`` and ``[numel]`` are THE SAME
        # CONFIG before codegen ever runs.  MEASURED: both normalize to ``[None]`` and both
        # construct ``PersistentReductionStrategy``.  The equality therefore cannot fail, and
        # on its own it would also pass on a tree that force-rolled BOTH arms to the full
        # extent -- which is precisely the mode this test's docstring claims to catch.
        #
        # ⚠ AND ``assertNotIn("roffset", ...)`` above is a STRING GREP: a pure rename of the
        # induction variable would satisfy it while a real chunk loop was emitted.
        #
        # ⇒ assert the STRATEGY CLASS, which is the fact the test is actually about and which
        # no rewrite of the emitted text can fake.  The spy already records it.
        strategies_none = self._strategy_classes(
            _rolled_rowsum, (x,), reduction_loops=[None], **self._RED_CFG
        )
        strategies_half = self._strategy_classes(
            _rolled_rowsum, (x,), reduction_loops=[numel // 2], **self._RED_CFG
        )
        self.assertIn(
            "PersistentReductionStrategy",
            strategies_none,
            f"a loop-free request must construct PersistentReductionStrategy; got "
            f"{strategies_none}. A LoopedReductionStrategy here means the force-roll is "
            f"back, whatever the emitted text says",
        )
        self.assertNotIn(
            "PersistentReductionStrategy",
            strategies_half,
            f"an explicit half-extent chunk must NOT be persistent; got "
            f"{strategies_half}. If it is, the two arms are the same kernel and the "
            f"inequality above is passing for the wrong reason",
        )

    def test_ndtile_reaches_the_one_builder(self) -> None:
        """``CuteNDTileStrategy`` — a reduction axis that is an inner ``hl.tile``.

        ⭐ UPDATED BY A7b, and the change is the POINT of A7b.  This used to assert that the
        tile path is "the ONLY family that passes ``require_exact_vec_cap``", because its outer
        lane loop's trip count was fixed at ``EPT // vec_cap`` in ``__init__`` before the plan
        was asked for -- so a narrowed plan would have left the loop visiting fewer elements
        than it assumed (bug class 1).

        A7b removed that asymmetry at the source: ``_build_cute_vec_lane_loop`` now asks for the
        plan FIRST and derives the trip count from ``plan.vec``, which is what the reduction
        path always did.  ⇒ **no caller passes the flag any more**, and this family now behaves
        like every other: a layout-imposed narrowing becomes a narrower VECTORISED copy instead
        of a decline.  Asserting the flag is ABSENT is the same claim as before -- one policy for
        one builder -- read the other way round.
        """
        with _tv_enabled():
            src, calls = self._emit_with_spy(
                _ndtile_rowsum,
                (self._x(),),
                block_sizes=[1, 512],
                num_threads=[0, 32],
                cute_vector_widths=[1, 8],
            )
        self.assertEqual(len(calls), 1, f"expected ONE plan build, got {calls}")
        self.assertEqual(calls[0]["vec"], 8)
        self.assertIn(
            calls[0]["exact"],
            (None, False),
            "after A7b no caller may pass require_exact_vec_cap: the trip count is derived "
            "from plan.vec, so exactness is structural rather than requested",
        )
        self.assertGreaterEqual(self._tv_counts(src)["tiled_copy_tv"], 1)

    def test_pointwise_no_reduction_reaches_the_one_builder(self) -> None:
        """A kernel with NO reduction op still gets its width from the one builder."""
        x = self._x()
        src, calls = self._emit_with_spy(
            _pointwise_row, (x, x.clone()), reduction_loops=[256], **self._RED_CFG
        )
        self.assertEqual(len(calls), 1, f"expected ONE plan build, got {calls}")
        self.assertEqual(calls[0]["vec"], 8)
        counts = self._tv_counts(src)
        self.assertGreaterEqual(counts["tiled_copy_tv"], 1)
        self.assertEqual(
            counts["arch_load"],
            0,
            f"a pointwise row on the TV path must not fall back to cute.arch.load: "
            f"{counts}",
        )

    def test_the_reduction_path_must_not_require_an_exact_cap(self) -> None:
        """⭐ THE ASYMMETRY IS DELIBERATE, so it is pinned rather than left to drift.

        The reduction path reads ``lane_extent`` back OFF the returned plan, so a
        narrowing re-derives the trip count together with the width and cannot skew.
        The tile path fixed its trip count first and must refuse.  Making them agree
        either way is a bug: ``True`` here loses legitimate narrowings, and ``False`` on
        the tile path is bug class 1.
        """
        _src, calls = self._emit_with_spy(
            _rolled_rowsum, (self._x(),), reduction_loops=[256], **self._RED_CFG
        )
        self.assertEqual(len(calls), 1)
        self.assertIn(calls[0]["exact"], (None, False))

    def test_the_loop_free_regime_is_SERVED_not_gated(self) -> None:
        """⛔ THE REGRESSION ARM.  An earlier version of this work gated the loop-free TV
        regime OFF to retire the AST rewrite, which retired the CAPABILITY with it: 144
        measured configs lost their ``cute.copy`` and fell back to a non-vectorized
        emission.  This asserts that trade stays undone.

        ⚠ NO ENV VAR IS SET, deliberately -- the property is about a shipped tree.  A
        version of this test that had to enable something would be testing a switch, not
        the default behaviour a user gets.
        """
        src, calls = self._emit_with_spy(
            _rolled_rowsum, (self._x(),), reduction_loops=[None], **self._RED_CFG
        )
        self.assertEqual(
            len(calls),
            1,
            "a loop-free request must still build a TV plan at the default environment; "
            "zero builds means the regime was gated off again",
        )
        self.assertGreaterEqual(
            self._tv_counts(src)["tiled_copy_tv"],
            1,
            "a plan was built but no TV layout was emitted",
        )
        self.assertEqual(
            self._tv_counts(src)["arch_load"],
            0,
            "the loop-free request fell back to the classic cute.arch.load form",
        )

    def test_no_second_plan_constructor_exists(self) -> None:
        """⛔ THE FAIL-CAPABILITY ARM: with the one builder neutered, NOTHING emits TV.

        Every test above would also pass if a family quietly kept a second width source
        AND happened to agree with the builder.  Forcing ``build_tv_plan`` to decline for
        everyone must therefore remove the TV emission from EVERY family -- a family that
        still emits ``make_tiled_copy_tv`` here has a width source of its own.

        ⚠ This is the arm that makes the counts above meaningful.  "One call recorded"
        proves the builder RAN; only this proves nothing else could have.
        """
        from helion._compiler import reduction_strategy as _rs
        from helion._compiler.cute import tv_layout as _tvl

        x = self._x()
        families: list[tuple[str, object, tuple[object, ...], dict[str, object]]] = [
            (
                "looped",
                _rolled_rowsum,
                (x,),
                {"reduction_loops": [256], **self._RED_CFG},
            ),
            (
                "loop-free",
                _rolled_rowsum,
                (x,),
                {"reduction_loops": [None], **self._RED_CFG},
            ),
            (
                "pointwise",
                _pointwise_row,
                (x, x.clone()),
                {"reduction_loops": [256], **self._RED_CFG},
            ),
        ]
        with (
            unittest.mock.patch.dict("os.environ", {"HELION_CUTE_LOOP_FREE_TV": "1"}),
            unittest.mock.patch.object(_tvl, "build_tv_plan", lambda **_kw: None),
            unittest.mock.patch.object(_rs, "build_tv_plan", lambda **_kw: None),
        ):
            for name, kernel, args, cfg in families:
                with self.subTest(family=name):
                    src = kernel.bind(args).to_triton_code(helion.Config(**cfg))  # pyrefly: ignore [missing-attribute]
                    self.assertEqual(
                        self._tv_counts(src)["tiled_copy_tv"],
                        0,
                        f"{name} still emitted a TV layout with build_tv_plan "
                        f"declining -- it has a second width source",
                    )
        with (
            _tv_enabled(),
            unittest.mock.patch.object(_tvl, "build_tv_plan", lambda **_kw: None),
        ):
            src = _ndtile_rowsum.bind((x,)).to_triton_code(
                helion.Config(
                    block_sizes=[1, 512],
                    num_threads=[0, 32],
                    cute_vector_widths=[1, 8],
                )
            )
            self.assertEqual(
                self._tv_counts(src)["tiled_copy_tv"],
                0,
                "ndtile still emitted a TV layout with build_tv_plan declining",
            )


@onlyBackends(["cute"])
class TestResidencyAndLaneSplitCannotCoOccur(TestCase):
    """⭐⭐ THE INVARIANT THAT MAKES rmem/smem RESIDENCY DECIDABLE AT LOWERING.

    A residency (``registers`` = an rmem cache, ``smem`` = a staged tile) is emitted by
    ``_cute_tv_partition_hoist`` while loads lower -- i.e. BEFORE the AST passes run. That is
    only sound if no later pass can restructure the lane loop those buffers are scoped to.
    ``split_lane_loop_reductions`` does exactly that kind of restructuring, so the two must
    never apply to the same reduction.

    ⭐ THEY CANNOT, AND THE REASON IS CLASS-LEVEL rather than a coincidence:

    * ``BlockReductionStrategy`` emits markers but **never builds a TV plan**, so its splits
      have no residency to disturb;
    * ``LoopedReductionStrategy`` builds plans but **never emits a marker** -- its
      ``for roffset`` loop carries its own accumulator and fold, which is what
      ``_lane_reduce_marker_unsupported`` declines on;
    * ``PersistentReductionStrategy`` **does both halves' worth of work and still emits no
      marker.**  ⚠ THIS BULLET USED TO READ "could do both -- and that is the shape
      ``ConfigSpec.normalize`` now ROLLS away, so it keeps no plan."  That is no longer
      true, and the change is deliberate: honouring ``reduction_loops=[None]`` means a
      persistent reduction now DOES keep a TV plan.  What keeps the invariant is not the
      absence of the plan but the absence of the marker -- lowering opens one lane nest per
      dependency layer (``prebuilt_lane_nest_factory`` x ``_wrap_segmented_body``), so the
      fold and the cross-thread combine are placed by loop structure and the marker is
      never emitted for an AST pass to consume.  MEASURED: 0 markers across the 9-point
      chunk x residency grid, and 0 on a dependent ``amax`` -> ``sum(exp(v - amax))`` pair.

    ⚠ THIS IS A NEGATIVE INVARIANT, so it needs a POSITIVE control or it passes vacuously on
    a tree where nothing builds a plan at all. ``test_a_plan_is_actually_built`` is that
    control -- and the test below additionally asserts its own observation set is non-empty
    AND that some config really did keep a plan, because the previous formulation was
    measured to pass by observing nothing at all.
    """

    _KERNEL_CFG: ClassVar[dict[str, object]] = {
        "block_sizes": [1],
        "num_threads": [1],
        "cute_threads_per_row": [128],
        "cute_vector_widths": [8, 1],
    }

    @staticmethod
    @contextlib.contextmanager
    def _watch() -> Iterator[list[tuple[str, bool]]]:
        """Record, per ``PersistentReductionStrategy`` init, whether it holds a TV plan."""
        from helion._compiler import reduction_strategy as _rs

        seen: list[tuple[str, bool]] = []
        real = _rs.PersistentReductionStrategy.__init__

        def spy(self: object, fn: object, block_index: int) -> None:
            real(self, fn, block_index)  # pyrefly: ignore [bad-argument-type]
            seen.append(("persistent", self._cute_tv_plan is not None))  # pyrefly: ignore [missing-attribute]

        _rs.PersistentReductionStrategy.__init__ = spy  # pyrefly: ignore [bad-assignment]
        try:
            yield seen
        finally:
            _rs.PersistentReductionStrategy.__init__ = real  # pyrefly: ignore [bad-assignment]

    def test_no_residency_ever_coexists_with_a_lane_reduce_marker(self) -> None:
        """A residency and a lane-reduce marker never co-occur on the same reduction.

        ⭐⭐ THIS TEST USED TO ASSERT SOMETHING STRICTLY STRONGER AND NOW FALSE.  It was
        ``test_no_persistent_reduction_ever_keeps_a_tv_plan`` and asserted
        ``with_plan == []`` -- that a ``PersistentReductionStrategy`` NEVER holds a TV plan.
        That held only because ``ConfigSpec.normalize`` force-rolled exactly that shape
        away.  With ``reduction_loops=[None]`` now honoured, **a persistent reduction DOES
        hold a plan** -- that is the point of the change, not a violation of it.

        ⚠ So the old assertion is not merely stale, it is inverted: rewriting it to pass
        would mean asserting that the new capability does not exist.  What is rewritten to
        is the property the class actually exists to protect, stated directly instead of
        via a proxy.

        THE REAL HAZARD, unchanged.  A residency (``registers`` = an rmem cache, ``smem`` =
        a staged tile) is emitted while loads lower -- BEFORE the AST passes run.  That is
        sound only if no later pass restructures the lane loop those buffers are scoped to.
        ``split_lane_loop_reductions`` does exactly that kind of restructuring, and it is
        driven by a ``_helion_lane_reduce`` MARKER.  ⇒ the thing that must never co-occur is
        **a residency and a marker**, and the plan was only ever a proxy for the marker.

        MEASURED over the same 9-point grid (3 chunks x 3 residencies) on
        ``N = 1024``, ``cute_threads_per_row=[128]``, ``cute_vector_widths=[8, 1]``:

            reduction_loops=[None]  persistent, KEEPS a plan, markers=0   (x3 residencies)
            reduction_loops=[512]   looped,     no plan kept,  markers=0
            reduction_loops=[256]   looped,     no plan kept,  markers=0

            configs where persistent kept a plan:              3 of 9
            configs where a plan AND a marker co-occur:        0 of 9   <- THE INVARIANT

        and separately, on a DEPENDENT pair (``amax`` -> ``sum(exp(v - amax))``) at
        ``[None]`` -- the shape the force-roll existed to dodge -- persistent keeps a plan,
        reaches a TV copy, and emits **zero** markers.

        ⚠ STILL A NEGATIVE INVARIANT, so it still needs the positive control below
        (``test_a_plan_is_actually_built``) or it passes vacuously on a tree where nothing
        builds a plan at all.  The control is now doubly load-bearing, because the old
        formulation could ALSO pass vacuously by observing no persistent strategy at all --
        which the review measured it doing.  This version asserts a non-empty observation
        set explicitly.
        """
        import itertools

        x = torch.randn([256, 1024], device=DEVICE, dtype=HALF_DTYPE)
        observed: list[tuple[list[int | None], str, bool, int]] = []
        for rl, res in itertools.product(
            ([None], [512], [256]), ("gmem", "registers", "smem")
        ):
            with self._watch() as seen:
                try:
                    src = _rolled_rowsum.bind((x,)).to_triton_code(
                        helion.Config(
                            **self._KERNEL_CFG,
                            reduction_loops=rl,
                            cute_row_residency=[res],
                        )
                    )
                except (exc.BackendUnsupported, exc.CuteRowResidencyUnavailable):
                    # A loud decline is a legal outcome of the honour-or-error contract
                    # and emits no kernel, so there is nothing to check.
                    continue
            kept_plan = any(held for _kind, held in seen)
            observed.append((rl, res, kept_plan, src.count("_helion_lane_reduce")))

        self.assertTrue(
            observed,
            "every config in the grid declined -- the invariant below is vacuous",
        )
        both = [(rl, res) for rl, res, kept, mk in observed if kept and mk]
        self.assertEqual(
            both,
            [],
            "a reduction kept a TV plan (so it can carry a residency) AND emitted a "
            f"lane-reduce marker (so an AST split can restructure the lane loop those "
            f"rmem/smem buffers are scoped to): {both}",
        )
        # ⛔ ANTI-VACUITY, and it is the half the old version lacked: prove the grid
        # actually reached the interesting arm.  Without this, "no config has both" would
        # pass on a tree where no config keeps a plan -- which is what the PRE-A1 tree did,
        # and the review measured the old test passing for exactly that reason.
        self.assertTrue(
            [1 for _rl, _res, kept, _mk in observed if kept],
            "no config kept a TV plan at all -- persistent+TV is meant to be reachable "
            "now, so this grid is not exercising the hazard it screens for",
        )

    def test_a_plan_is_actually_built(self) -> None:
        """⛔ THE POSITIVE CONTROL for the negative invariant above.

        Without this, ``no persistent strategy holds a plan`` would also pass on a tree where
        the TV path is broken and NOTHING holds a plan -- the unfalsifiable-zero failure this
        repo has been fooled by repeatedly.
        """
        x = torch.randn([256, 1024], device=DEVICE, dtype=HALF_DTYPE)
        with self._spy_plans() as calls:
            src = _rolled_rowsum.bind((x,)).to_triton_code(
                helion.Config(**self._KERNEL_CFG, reduction_loops=[None])
            )
        self.assertTrue(
            calls, "no TV plan was built at all -- the invariant is vacuous"
        )
        self.assertGreaterEqual(src.count("cute.make_tiled_copy_tv("), 1)

    @staticmethod
    @contextlib.contextmanager
    def _spy_plans() -> Iterator[list[object]]:
        from helion._compiler import reduction_strategy as _rs
        from helion._compiler.cute import tv_layout as _tvl

        calls: list[object] = []
        real = _tvl.build_tv_plan

        def spy(**kw: object) -> object:
            plan = real(**kw)  # pyrefly: ignore [bad-argument-type]
            calls.append(plan)
            return plan

        with (
            unittest.mock.patch.object(_tvl, "build_tv_plan", spy),
            unittest.mock.patch.object(_rs, "build_tv_plan", spy),
        ):
            yield calls


@onlyBackends(["cute"])
class TestFrozenBasisTakesTheTvPathNotClassicVec(TestCase):
    """⭐⭐ THE TASK-6 ACCEPTANCE PROPERTY: every frozen perf cell goes through the TV
    layout, and the CLASSIC VEC path is reached by none of them.

    ⛔ WHY THIS NEEDS A TEST AND NOT A ONE-OFF MEASUREMENT.  The branch's OOB-guard layer
    (``CuteVecLoadDesc`` and friends, ~260 lines) was deleted on exactly this basis, and the
    argument the task demands is *"a positive argument that the shapes they guarded no longer
    reach a classic vec load (not merely 'no test failed')"*.  That argument is a **reach
    count**, and a reach count decays silently: a config edit, a new decline, or a widened
    gate could put a cell back on ``cute.arch.load`` with every existing test still green
    (the emitted code would change, but only the frozen HASHES would notice, and a hash
    change is re-recorded by hand).

    ⚠ Counted at the DEFAULT environment, deliberately.  Two of the 40 cells carry
    ``cute_ndtile_tv=[True]`` in their own config, so no env var is needed -- and asserting
    this with ``HELION_CUTE_NDTILE_TV=1`` set would be a weaker claim about a tree nobody
    ships.
    """

    @staticmethod
    def _frozen_cells() -> list[tuple[str, str, dict[str, object]]]:
        import json
        import pathlib as _pathlib

        import helion as _helion

        root = _pathlib.Path(_helion.__file__).parent.parent
        path = root / "_redfix2" / "frozen_configs.json"
        if not path.is_file():
            return []
        out = []
        for kernel, cells in sorted(json.loads(path.read_text()).items()):
            for cell, entry in sorted(cells.items()):
                cfg = entry.get("config") if isinstance(entry, dict) else entry
                if cfg:
                    out.append((kernel, cell, cfg))
        return out

    def test_no_frozen_cell_reaches_a_classic_vec_emitter(self) -> None:
        """0 calls into the three ``cute.arch.load`` hoists; >=1 TV hoist call per cell."""
        import sys

        cells = self._frozen_cells()
        if not cells:
            self.skipTest(
                "_redfix2/frozen_configs.json is not present in this checkout"
            )
        sys.path.insert(
            0, str(__import__("pathlib").Path(helion.__file__).parent.parent)
        )
        from _redfix2.bench_reductions import KERNELS

        import helion._compiler.cute.memory_ops as cmo
        import helion.language.memory_ops as lmo

        counts: dict[str, int] = {}

        def wrap(mod: object, name: str, also: object = None) -> None:
            real = getattr(mod, name)

            def spy(*a: object, **k: object) -> object:
                counts[name] = counts.get(name, 0) + 1
                return real(*a, **k)

            setattr(mod, name, spy)
            # ⚠ The consumer imported these BY NAME, so patching only the defining module
            # would count zero while the real one ran -- an unfalsifiable pass.
            if also is not None:
                setattr(also, name, spy)

        CLASSIC = (
            "_cute_register_tile_unroll_vec_hoist",
            "_cute_register_tile_unroll_vec_hoist_split2",
        )
        # ⚠ PATCHED BY HAND AND RESTORED IN A ``finally``, not via ``patch.object``: the
        # spies must persist across the whole 40-cell loop AND be installed in BOTH the
        # defining module and the consumer that imported them by name, which is four
        # separate bindings for two functions.
        originals = {
            (lmo, CLASSIC[0]): lmo._cute_register_tile_unroll_vec_hoist,
            (lmo, CLASSIC[1]): lmo._cute_register_tile_unroll_vec_hoist_split2,
            (cmo, CLASSIC[0]): cmo._cute_register_tile_unroll_vec_hoist,
            (cmo, CLASSIC[1]): cmo._cute_register_tile_unroll_vec_hoist_split2,
            (
                cmo,
                "_cute_register_unroll_vec_hoist",
            ): cmo._cute_register_unroll_vec_hoist,
            (cmo, "_cute_tv_partition_hoist"): cmo._cute_tv_partition_hoist,
        }
        try:
            wrap(lmo, CLASSIC[0], cmo)
            wrap(lmo, CLASSIC[1], cmo)
            wrap(cmo, "_cute_register_unroll_vec_hoist")
            wrap(cmo, "_cute_tv_partition_hoist")
            per_cell_tv: list[tuple[str, int]] = []
            for kernel, cell, cfg in cells:
                counts.clear()
                m, n = (int(v) for v in cell.split("x"))
                KERNELS[kernel](m, n)[0].to_triton_code(helion.Config(**cfg))
                classic = sum(counts.get(name, 0) for name in CLASSIC) + counts.get(
                    "_cute_register_unroll_vec_hoist", 0
                )
                self.assertEqual(
                    classic,
                    0,
                    f"{kernel}/{cell} reached a CLASSIC VEC emitter {classic} times; the "
                    f"branch's OOB-guard layer was deleted on the basis that no frozen "
                    f"cell does",
                )
                per_cell_tv.append(
                    (f"{kernel}/{cell}", counts.get("_cute_tv_partition_hoist", 0))
                )
        finally:
            for (mod, name), fn in originals.items():
                setattr(mod, name, fn)
        without_tv = [name for name, tv in per_cell_tv if tv == 0]
        self.assertEqual(
            without_tv,
            [],
            f"{len(without_tv)} of {len(per_cell_tv)} frozen cells emitted no TV copy: "
            f"{without_tv}",
        )


@onlyBackends(["cute"])
class TestTvWidthDecisionIsStructurallySingular(TestCase):
    """⭐ THE PROPERTY ``TestOneTvPlanConstructor`` CANNOT REACH: not "everyone calls the
    one builder today", but "there is nowhere else a TV width could come from".

    The call-counting tests above are dynamic -- they prove the four families reached
    ``build_tv_plan`` on the shapes exercised. A fifth caller, or a shape those tests do
    not cover, would not register. These are STATIC and cover the whole tree.

    ⚠ SOURCE-LEVEL ASSERTIONS ARE USUALLY A SMELL, and they are the right tool exactly
    here: the claim is about the SET of construction sites, which is not observable from
    any single emission. The alternative -- making ``ChunkTVPlan``'s constructor private
    and routing through a factory -- buys the same guarantee with more machinery, and this
    repo has a documented preference for a checkable statement over a mechanism.
    """

    @staticmethod
    def _source(module: object) -> str:
        import inspect

        return inspect.getsource(module)

    def test_chunk_tv_plan_has_exactly_one_construction_site(self) -> None:
        """``ChunkTVPlan(...)`` is constructed in ONE place, inside ``chunk_plan``.

        Every field of a plan -- the atom width, the value layout, the lane extent -- is
        derived from its ``vec``, so a second construction site is a second answer to
        "how wide is one copy", which is bug class 1 by definition.
        """
        from helion._compiler.cute import tv_layout

        src = self._source(tv_layout)
        sites = [
            ln
            for ln in src.splitlines()
            if "ChunkTVPlan(" in ln
            and "def " not in ln
            and "-> " not in ln
            and ": ChunkTVPlan" not in ln
            and '"ChunkTVPlan"' not in ln
        ]
        self.assertEqual(
            len(sites),
            1,
            f"expected ONE ChunkTVPlan construction site, found {len(sites)}: {sites}",
        )

    def test_no_module_outside_tv_layout_constructs_a_plan_or_calls_chunk_plan(
        self,
    ) -> None:
        """``chunk_plan`` and ``ChunkTVPlan`` are not reachable from anywhere else.

        ⚠ Checked over the whole ``helion/`` tree rather than the modules this test
        happens to import, because the failure being prevented is a NEW caller appearing
        in a file nobody thought to look at -- which is how the two plan constructors this
        unification removed came to exist in the first place.
        """
        import pathlib

        import helion

        root = pathlib.Path(helion.__file__).parent
        owner = root / "_compiler" / "cute" / "tv_layout.py"
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            if path == owner:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                code = line.split("#", 1)[0]
                if (
                    "chunk_plan(" in code
                    and "build_tv_plan" not in code
                    or "ChunkTVPlan(" in code
                ):
                    offenders.append(
                        f"{path.relative_to(root)}:{lineno}: {line.strip()}"
                    )
        self.assertEqual(
            offenders,
            [],
            "a TV width is being decided outside cute/tv_layout.py:\n  "
            + "\n  ".join(offenders),
        )

    def test_no_module_hand_writes_a_tv_lane_column_index(self) -> None:
        """⛔ The lane-base COLUMN index is emitted only by ``ChunkTVPlan``.

        This pins a fixed WRONG-ANSWER bug, not a style rule.  The formula
        ``column(lane, tid) = offset + tid*vec + lane*(tpr*vec)`` used to be
        hand-written at three call sites.  Two agreed; the third
        (``CuteNDTileStrategy.codegen_device_loop``) had the strides TRANSPOSED --
        ``offset + tid*EPT + lane*vec`` -- which addresses a *blocked* layout while
        the ``make_tiled_copy_tv`` it feeds addresses an *interleaved* one.  On a
        divisible extent that only permutes the columns, so a reduction cannot see
        it; on a RAGGED extent it walks off the end of the row, and two frozen
        ``cross_entropy_online`` cells returned ``nan`` and read out of bounds.

        ⇒ the width was single-sourced (the two tests above) but the INDEX was not,
        and the index is what broke.  Anything emitting ``* <tpr*vec>`` next to a
        lane variable outside ``tv_layout.py`` is re-deriving this and must instead
        call :meth:`ChunkTVPlan.emit_lane_base`.

        ⚠ Deliberately a SOURCE scan over all of ``helion/``, matching the sibling
        tests: the failure being prevented is a *new* consumer appearing in a file
        nobody thought to look at, which is exactly how the transposed copy came to
        exist.  A behavioural test would need that consumer to already exist.
        """
        import pathlib
        import re

        import helion

        root = pathlib.Path(helion.__file__).parent
        owner = root / "_compiler" / "cute" / "tv_layout.py"
        # ``lane`` (or ``lane_var``) multiplied by a *derived stride expression* --
        # i.e. a lane term whose multiplier is computed from a thread count and a
        # vector width, which is the plan's job.  A literal ``* 1`` or a bare name
        # is not matched: the defect is re-deriving the STRIDE.
        pattern = re.compile(
            r"\{?[A-Za-z_][A-Za-z_0-9]*lane[A-Za-z_0-9]*\}?\s*\)?\s*\*\s*"
            r"\{(?:[^{}]*(?:thread_count|threads_per_row|_tpr|num_threads)"
            r"[^{}]*\*[^{}]*|[^{}]*\*[^{}]*(?:vec|vector_width)[^{}]*)\}"
        )
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            if path == owner:
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                code = line.split("#", 1)[0]
                if pattern.search(code):
                    offenders.append(
                        f"{path.relative_to(root)}:{lineno}: {line.strip()}"
                    )
        self.assertEqual(
            offenders,
            [],
            "a TV lane-column index is being derived outside cute/tv_layout.py "
            "-- call ChunkTVPlan.emit_lane_base() instead:\n  "
            + "\n  ".join(offenders),
        )


@onlyBackends(["cute"])
class TestA7bOneWidthPolicy(TestCase):
    """⭐ A7b: ``build_tv_plan``'s two callers now pass the SAME policy.

    They used to differ in exactly one way -- WHEN the caller committed to its lane-loop trip
    count:

        reduction path   build the plan first, then read ``lane_extent`` back off it, so a
                         narrowing re-derives the trip count for free.
        ND tile path     fix the trip count at ``EPT // vec_cap`` in ``__init__``, THEN ask --
                         so a narrowed plan was unusable and had to be rejected outright via
                         ``require_exact_vec_cap=True``.

    ``_build_cute_vec_lane_loop`` now asks FIRST and derives the emitted geometry from
    ``plan.vec``, so ``require_exact_vec_cap`` has no caller left and the ND path GAINS the
    coverage it used to refuse: a layout-imposed narrowing becomes a narrower VECTORISED copy
    instead of a fallback to the per-element enumeration.

    The narrowing is provoked with a SLICED input.  A ``(512, 512+pad)`` base sliced to
    ``[:, :512]`` has row stride ``512+pad``; a V-wide copy moves ``V * dtype_bytes`` from
    ``base + row*stride``, so the layout must clamp V to ``gcd(stride, V)``.  At ``pad=4`` and
    V=8 that is 4 -- the exact case the old code declined.

    MEASURED, A7a-only (``18fe74f0c``) vs A7b, bf16 512x512 sliced, ``bs=[32,512] vw=[1,8]``:

        pad=4, 12, 20   ->   tv=0 bits=0 arch.load=4      (declined, per-element fallback)
                        ->   tv=1 bits=64 arch.load=0     (narrowed TV copy), err 0.0

    ⚠ ``pad=0`` is the control: an unsliced row stride of 512 admits the full width, so it
    stays at 128 bits on BOTH arms.  Without that arm this test could pass because the width
    collapsed everywhere.
    """

    @staticmethod
    def _sliced_pointwise(
        pad: int, block_sizes: list[int], vec_width: int
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def pw2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile(x.shape):
                out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
            return out

        torch.manual_seed(0)
        base_x = torch.randn(512, 512 + pad, device=DEVICE, dtype=HALF_DTYPE)
        base_y = torch.randn(512, 512 + pad, device=DEVICE, dtype=HALF_DTYPE)
        x = base_x[:, :512]
        y = base_y[:, :512]
        # ⚠ TWO slots, and this helper's only callers are SKIPPED.  This is a ROOT-GRID
        # kernel, and the root-grid axes no longer register a ``cute_ndtile_tv`` slot (the
        # root-grid vectorisation was deleted -- see ``CuteNDTileStrategy.codegen_grid``), so
        # ``valid_block_ids()`` is now ``[]`` here and ANY value raises ``InvalidConfig``.
        # Left as-is deliberately: it records the config the arms need, and un-skipping them
        # means restoring the grid slots too.  Do not "fix" it to ``[True]`` -- that would be
        # wrong for a 0-slot kernel and would hide the coupling.
        code, out = code_and_output(
            pw2d,
            (x, y),
            block_sizes=block_sizes,
            cute_vector_widths=[1, vec_width],
            cute_ndtile_tv=[True, True],
        )
        return code, out, x + y

    @pytest.mark.skip(
        reason=(
            "These arms demonstrate A7b's narrowing GAIN through the 2-D root-grid path, "
            "which is disabled (see CuteNDTileStrategy.codegen_grid's grid_vec_enabled). "
            "⚠ A7b ITSELF IS LIVE -- it changed _build_cute_vec_lane_loop, which "
            "codegen_device_loop uses; only this demonstration vehicle is off. "
            "test_require_exact_vec_cap_has_no_callers_left still runs and pins A7b's "
            "structural claim. Un-skip together with the 2-D arms above."
        )
    )
    def test_a_narrowed_width_still_gets_a_TV_copy(self) -> None:
        """The gain: a clamped width vectorises instead of declining."""
        for pad in (4, 12, 20):
            with self.subTest(pad=pad):
                code, out, ref = self._sliced_pointwise(pad, [32, 512], 8)
                torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
                self.assertIn("make_tiled_copy_tv", code)
                # Narrowed to 64 bits by the row stride, NOT dropped to the scalar or
                # per-element path.
                self.assertIn("num_bits_per_copy=64", code)
                self.assertNotIn("num_bits_per_copy=128", code)
                self.assertNotIn("cute.arch.load(", code)

    @pytest.mark.skip(
        reason=(
            "These arms demonstrate A7b's narrowing GAIN through the 2-D root-grid path, "
            "which is disabled (see CuteNDTileStrategy.codegen_grid's grid_vec_enabled). "
            "⚠ A7b ITSELF IS LIVE -- it changed _build_cute_vec_lane_loop, which "
            "codegen_device_loop uses; only this demonstration vehicle is off. "
            "test_require_exact_vec_cap_has_no_callers_left still runs and pins A7b's "
            "structural claim. Un-skip together with the 2-D arms above."
        )
    )
    def test_an_UNSLICED_row_keeps_the_FULL_width(self) -> None:
        """⛔ THE CONTROL.  Without it, "everything narrowed" would also pass the test above."""
        code, out, ref = self._sliced_pointwise(0, [32, 512], 8)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        self.assertIn("num_bits_per_copy=128", code)

    def test_require_exact_vec_cap_has_no_callers_left(self) -> None:
        """A7b's structural claim: ONE policy, so the flag is dead.

        Pinned as a source assertion because it is the difference between "the two callers
        agree" and "the two callers happen to agree on the shapes I measured".  ``build_tv_plan``
        still DEFINES the parameter (it documents the hazard for any future caller that commits
        a trip count before asking); what must not exist is a caller passing it.
        """
        import inspect

        import helion._compiler.reduction_strategy as reduction_strategy
        import helion._compiler.tile_strategy as tile_strategy

        # ⚠ Strip comments first.  Both modules still DISCUSS the flag in the notes that
        # record why the asymmetry existed and how it was removed -- that prose is the point,
        # not a violation.  What must not exist is a live keyword argument.
        for module in (tile_strategy, reduction_strategy):
            with self.subTest(module=module.__name__):
                code_lines = [
                    line.split("#", 1)[0]
                    for line in inspect.getsource(module).splitlines()
                ]
                self.assertNotIn("require_exact_vec_cap=True", "\n".join(code_lines))
