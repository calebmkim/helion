"""Capability ③ -- SMEM staging of the reduction row -- on the **TILE** regime.

Staging was reachable only from a ROLLED reduction (``x[tile_m, :]`` ->
``LoopedReductionStrategy``).  A reduction axis written as an inner ``hl.tile``
(``x[tile_m, tile_n]`` -> ``CuteNDTileStrategy``) had capability ① (a vectorized TV
``cute.copy``) but not ③, so its second sweep re-read the row from global.

⭐ WHY THE TWO REGIMES ARE THE SAME PROBLEM, which is what makes the port small.  Both
iterate the reduction axis in chunks; the only difference is who wrote the loop.  MEASURED,
and it is the fact the whole port rests on: ``cute_stage_restages_cloned_sweeps()`` is
**False** on this path (``offset_var`` is a real ``tile_offset_N``, not the literal ``"0"``),
so the tile regime is a **two-lowering-site** path exactly like the looped one --
``_cute_tv_stage_slice`` is called once per sweep and ``first_read`` flips True -> False by
itself.  The loop-free (persistent) machinery -- ``_tv_restage_cloned_loads`` /
``_tv_stage_read_by_frag`` -- is bypassed entirely: measured **0 calls** on this path.

⚠ THE REQUEST IS INJECTED, NOT CONFIGURED, AND THAT IS THE DELIBERATE SCOPE LINE.
``cute_row_residency`` / ``cute_stage_smem_kb`` are registered
per REDUCTION BLOCK (``DeviceIR._register_cute_tv_layout_slots``), and a tiled axis owns no
reduction block -- MEASURED: ``InvalidConfig: Too many values for
config['cute_row_residency'], expected 0, got 1``.  Widening that domain changes the FLAT
CONFIG WIDTH and runs a ``_fill_missing`` ladder that could turn a residency ON where none
was requested, on ``cross_entropy_online`` (10 of the 40 frozen perf cells).  So the
MECHANISM lands here and the KNOB is a separate, measured change; these tests drive the
request through ``_cute_row_residency_config``, the one method that reads it.
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import TYPE_CHECKING
import unittest.mock

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@contextlib.contextmanager
def _residency(want: str) -> Iterator[None]:
    """Drive the row-residency REQUEST, and switch the NDTile TV path on.

    ⚠ BOTH, TOGETHER, AND SCOPED.  Staging rides on the TV emission, so a residency request
    without ``HELION_CUTE_NDTILE_TV=1`` is inert -- a test that set only the residency would
    assert against the legacy scalar path and pass vacuously.

    The request is injected at ``_cute_row_residency_config`` -- the ONE method that reads
    the knobs -- rather than by monkeypatching the resolved state.  That keeps every
    downstream decision (the plan check, the row axis, the SMEM budget, the multi-read walk,
    the ``first_read`` discriminator) in real code, so what these tests exercise is the
    shipped mechanism and only the config surface is stubbed.
    """
    from helion._compiler.tile_strategy import CuteNDTileStrategy

    with (
        unittest.mock.patch.dict("os.environ", {"HELION_CUTE_NDTILE_TV": "1"}),
        unittest.mock.patch.object(
            CuteNDTileStrategy,
            "_cute_row_residency_config",
            lambda self, fn, block_index: want,
        ),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True)
def _tile_two_sweep_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """An RMS norm whose reduction axis is an inner ``hl.tile``: TWO sweeps over one row.

    ⛔ ``cross_entropy_online`` CANNOT STAND IN FOR THIS, and the reason is structural rather
    than a matter of taste.  It is the benchmark kernel on this path and owns 10 of the 40
    frozen cells, but it is **single-sweep by construction** -- its whole point is fusing two
    passes into one via an online ``(max, sum-of-exp)`` recurrence.  A single-sweep kernel
    reads the row ONCE, so there is no second read for staging to serve: staging it is a
    no-op and a test built on it passes VACUOUSLY.  ③ needs a genuinely multi-pass kernel.

    ⚠ ``hl.register_block_size(n)`` IS LOAD-BEARING.  MEASURED: two independent
    ``hl.tile(n)`` loops get **two block ids and two strategy instances**, so the per-instance
    staging state (above all the ``first_read`` discriminator) resets between sweeps and the
    buffer would be written twice and read never.  ONE registered block size gives ONE
    instance whose ``codegen_device_loop`` runs twice -- the shape the looped path presents.
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


# ⚠ SLOT ORDER IS [n, m].  ``hl.register_block_size(n)`` runs before the ``hl.tile(m)`` loop,
# so the REDUCTION axis is block 0 and the grid row axis is block 1.  ``block_sizes=[1, 512]``
# would silently give a reduction chunk of 1 and no lane loop at all -- while still computing
# the right answer, so every structural assertion would pass vacuously.
#
# ⭐ WITHHOLDING THE REGISTER FUSER IS LOAD-BEARING, NOT TIDINESS.  When the register cache
# is live, ``fuse_tv_copy_sweeps`` serves sweep 2 from registers, so the staged SMEM read
# stops being what feeds the consume sweep -- and corrupting the staged tile would leave the
# output BIT-IDENTICAL, i.e. every assertion in this file would pass vacuously.  Disabling it
# makes SMEM the only mechanism in play.
#
# ⛔⛔ THIS USED TO BE ``cute_tv_sweep_cache=[0]`` AND THAT KNOB NO LONGER EXISTS.  Task 1
# deleted it along with the slot budget it capped, so the config below raised
# ``InvalidConfig: Invalid config keys ['cute_tv_sweep_cache']`` in ``config_spec.normalize``
# -- BEFORE reaching any kernel.  ⇒ four tests in this file were red for a dead config key,
# which reads exactly like a broken mechanism and is not one.  The surviving equivalent is the
# env switch ``HELION_TV_SWEEP_FUSE=disabled`` (``fuse_tv_copy_sweeps.py:481``), which turns
# the whole pass off; unlike the old numeric override it cannot silently UN-VETO an explicit
# ``gmem`` request, which is why the budget was replaced by a set in the first place.
#
# ⚠ ARITY IS MEASURED RATHER THAN GUESSED.  ``cute_online_defer``'s domain is DEVICE LOOPS
# (``_register_cute_ast_pass_specs``: ``reduction_loops`` union ``block_sizes`` minus the grid
# axes), and the row axis here is a GRID axis -- so this kernel has ONE slot, on block 0,
# while the per-axis keys have two.  MEASURED:
#     block_sizes 2 [0,1] | num_threads 2 [0,1] | cute_vector_widths 2 [0,1]
#     cute_online_defer 1 [0] | cute_row_residency 0 []
# (measured before task 1 deleted ``cute_reduction_reload`` / ``cute_tv_sweep_cache`` /
# ``cute_stage_smem_kb``, all of which were 0 [] or 1 [0] as well)
# ``cute_row_residency`` being 0 [] is the item-6 gap: a TILED reduction axis owns no
# reduction block, so it owns no residency slot -- which is why these tests INJECT the
# request instead of configuring it.
_CFG: dict[str, list[int]] = {
    "block_sizes": [512, 1],
    "num_threads": [32, 0],
    "cute_vector_widths": [8, 1],
}


def _no_register_fuser() -> unittest.mock._patch:
    """Withhold the register-cache pass, so SMEM staging is the only mechanism in play.

    Replaces the deleted ``cute_tv_sweep_cache=[0]`` config entry -- see the ⛔⛔ note above.
    """
    return unittest.mock.patch.dict(os.environ, {"HELION_TV_SWEEP_FUSE": "disabled"})


_M, _N = 64, 2048


def _gmem_loads_per_tensor(code: str) -> dict[str, int]:
    """``tensor -> number of GMEM ``cute.copy`` LOADS``.

    ⭐ THE COUNT, NOT THE PRESENCE.  Staging does not remove a tensor's gmem reads: the FIRST
    sweep must still read global, because that read is what FILLS the tile.  What staging
    removes is the SECOND.  "Is x still loaded from gmem?" therefore answers yes, once,
    correctly -- and asserting absence would pin the opposite of the property.

    Loads are told from store flushes by operand position: a load has the partition in the
    SOURCE slot, a flush in the DESTINATION slot.  Both share the ``cute.copy`` spelling.
    """
    tile_owner = dict(re.findall(r"(_tv_tile_\d+) = cute\.local_tile\((\w+), ", code))
    part_owner = {
        part: tile_owner[tile]
        for part, tile in re.findall(
            r"(_tv_part_\d+) = \w+\.partition_[DS]\((_tv_tile_\d+)\)", code
        )
        if tile in tile_owner
    }
    counts: dict[str, int] = {}
    for part in re.findall(
        r"cute\.copy\(\w+, (_tv_part_\d+)\[[^]]*\], _tv_frag_\d+\)", code
    ):
        owner = part_owner.get(part)
        if owner is not None:
            counts[owner] = counts.get(owner, 0) + 1
    return counts


def _staged_reads(code: str) -> list[str]:
    """Fragments served from SMEM: ``autovec_copy(_tv_spart_N[...], frag)``."""
    return re.findall(
        r"cute\.autovec_copy\(_tv_spart_\d+\[[^]]*\], (_tv_frag_\d+)\)", code
    )


def _staged_publishes(code: str) -> list[str]:
    """Fragments published TO SMEM: ``autovec_copy(frag, _tv_spart_N[...])``."""
    return re.findall(
        r"cute\.autovec_copy\((_tv_frag_\d+), _tv_spart_\d+\[[^]]*\]\)", code
    )


def _partition_dirs(code: str) -> dict[str, str]:
    return dict(re.findall(r"(_tv_spart_\d+) = \w+\.partition_([DS])\(", code))


@onlyBackends(["cute"])
class TestCuteNDTileStageSmem(TestCase):
    def _reference(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            x.float(), (x.size(1),), w.float(), eps=1e-5
        )

    def _inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        return (
            torch.randn(_M, _N, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(_N, device=DEVICE, dtype=torch.bfloat16),
        )

    def _assert_not_vacuous(self, code: str) -> None:
        """The TV path really engaged, at a width where the copy is not a scalar load.

        ⭐ WITHOUT THIS EVERY ASSERTION IN THE FILE CAN HOLD TRIVIALLY.  At
        ``lane_extent == 1`` the plan admits only ``vec == 1``, so a "TV copy" is
        indistinguishable from a scalar load; and if the plan declined outright there is no
        emission to assert about at all.  Both are read off the EMITTED loop, not off the
        config, because the config is not evidence -- a cell can carry a knob the kernel
        never used.
        """
        assert code.count("cute.make_tiled_copy_tv(") == 1, "no TV layout"
        assert "num_bits_per_copy=128" in code, "not the full 128-bit atom"
        assert "cute.arch.load(" not in code, "legacy vector load survived"
        extents = [
            int(b) for _v, b in re.findall(r"for (lane_\d+) in range\((\d+)\)", code)
        ]
        assert extents and max(extents) > 1, f"lane_extent not > 1: {extents}"
        # TWO chunk coordinate vars == two sweeps really lowered.  One would mean only one
        # sweep built a chunk prefix, and the staged read would have nothing to pair with.
        chunks = sorted(set(re.findall(r"_tv_chunk_\d+", code)))
        assert len(chunks) == 2, f"expected one chunk var per sweep, got {chunks}"

    def test_staging_replaces_the_second_gmem_read(self) -> None:
        """⭐ THE POSITIVE ARM: ``smem`` allocates, publishes, AND is READ BACK.

        The last conjunct is the one that matters.  A previous staging implementation
        allocated and published a buffer **nobody read**: every answer was correct and
        nothing failed, and only an emission-side check caught it.  So the assertion is not
        "a buffer exists" but "a gmem read was REPLACED by a read of that buffer".
        """
        x, w = self._inputs()
        with _residency("smem"), _no_register_fuser():
            code, out = code_and_output(_tile_two_sweep_norm, (x, w, 1e-5), **_CFG)
        torch.testing.assert_close(
            out.float(), self._reference(x, w), atol=2e-2, rtol=2e-2
        )
        self._assert_not_vacuous(code)

        # (1) ONE buffer, sized ``rows_per_cta * N``.  Here the row axis is a GRID axis
        # (``num_threads=[32, 0]``), so ``rows_per_cta == 1`` and the tile is 1 x N.
        self.assertEqual(code.count("alloc_smem"), 1)
        self.assertIn(
            f"cute.arch.alloc_smem(cutlass.BFloat16, {_N}, alignment=16)", code
        )

        # (2) A writer AND a reader, off the SAME shared slice -- ``partition_D`` for the
        # publish and ``partition_S`` for the read-back.  Both directions must be present:
        # a ``D`` with no ``S`` is exactly the write-nobody-reads bug.
        self.assertEqual(sorted(_partition_dirs(code).values()), ["D", "S"])
        self.assertEqual(len(_staged_publishes(code)), 1)
        self.assertEqual(len(_staged_reads(code)), 1)

        # (3) ⭐ AND THE GMEM READ IS GONE FROM SWEEP 2.  ``x`` is read from global exactly
        # ONCE (sweep 1, which fills the tile); its second read comes from SMEM.  This is
        # the count-not-presence assertion -- see ``_gmem_loads_per_tensor``.
        self.assertEqual(
            _gmem_loads_per_tensor(code).get("x"),
            1,
            f"x must read gmem once and SMEM once: {_gmem_loads_per_tensor(code)}",
        )

        # (4) The staged tile is CHUNK-indexed, not constant.  A constant column makes every
        # chunk alias the first -- which is fast and wrong (MEASURED relerr 261.6 on the
        # rolled path), so it is pinned structurally as well as numerically.
        staged = re.findall(
            r"cute\.local_tile\(_tv_smem, \(1, \d+\), \((.*?), (\w+)\)\)", code
        )
        self.assertEqual(
            len(staged), 2, f"expected writer + reader tiles, got {staged}"
        )
        for _row, chunk in staged:
            self.assertRegex(chunk, r"^_tv_chunk_\d+$")
        # The row coordinate is the CTA-LOCAL thread index, never the clamped global row:
        # the buffer only has ``rows_per_cta`` rows.
        for row, _chunk in staged:
            self.assertIn("thread_idx()[1]", row)

    def test_residency_axis_is_three_distinct_kernels(self) -> None:
        """⭐ ``smem`` / ``registers`` / ``gmem`` must not collapse into each other.

        The axis is only meaningful if its three values emit three kernels.  The
        discriminator that makes ``gmem`` distinct from ``registers`` is
        ``cute_row_residency_forbids_sweep_cache`` -- without it both spell "no staging" and
        ``fuse_tv_copy_sweeps`` fires in each, giving byte-identical output.  MEASURED before
        that veto reached this path: ``gmem`` emitted 10 ``_tv_sweep_cache`` statements.

        The load COUNT is what separates them, and it reads exactly as the axis is defined:
        one gmem read of the row when a mechanism is in effect, two when none is.

        ⚠⚠ THE ``registers`` ARM MUST LET THE FUSER RUN, and the other two must not -- getting
        that wrong makes the test assert the opposite of its own claim.

        ⛔⛔ THIS ARM USED TO SET ``cute_tv_sweep_cache=[128]`` AND THAT KNOB IS DELETED.  The
        old encoding was a per-thread SLOT BUDGET, and the arms differed by budget: ``[0]`` for
        ``smem``/``gmem`` (``0`` was also the pass's own spelling for "decline this loop") and
        ``[128]`` for ``registers`` -- 128 because, MEASURED on this kernel, budgets
        ``{0, 8, 32}`` all emitted ONE identical kernel and ``{128, 256}`` another, so an
        insufficient positive value was indistinguishable from zero.

        Task 1 deleted the budget entirely: the rule is now *honour the request and let it
        spill*, only capacity and geometry may refuse.  ⇒ **the numeric ladder is gone and with
        it the "insufficient budget" hazard** -- a ``registers`` request now simply fires.  What
        replaces the ``[0]`` arms is the withholding switch ``HELION_TV_SWEEP_FUSE=disabled``,
        which turns the pass off outright and, unlike the old override, cannot silently
        UN-VETO an explicit ``gmem``.

        ⚠ Also measured, and still true: under a ``smem`` request the budget never mattered --
        every value from 0 to 256 gave ONE kernel, because staging consumes the second read
        before the fuser can find two sibling sweeps to fuse.  So withholding the fuser on the
        ``smem`` arm is belt-and-braces, while on the ``gmem`` arm it is the discriminator.
        """
        x, w = self._inputs()
        seen: dict[str, tuple[int, int, int]] = {}
        for want in ("smem", "registers", "gmem"):
            # ``registers`` is the arm under test, so the fuser must be free to fire; the
            # other two arms withhold it, which is what keeps them distinct from it.
            fuser = (
                contextlib.nullcontext()
                if want == "registers"
                else _no_register_fuser()
            )
            with _residency(want), fuser:
                code, out = code_and_output(_tile_two_sweep_norm, (x, w, 1e-5), **_CFG)
            torch.testing.assert_close(
                out.float(), self._reference(x, w), atol=2e-2, rtol=2e-2
            )
            self._assert_not_vacuous(code)
            self.assertIn(f"'row residency: {want}'", code)
            seen[want] = (
                code.count("alloc_smem"),
                _gmem_loads_per_tensor(code).get("x", 0),
                code.count("_tv_sweep_cache"),
            )
        # (alloc_smem, x's gmem loads, register-cache statements)
        self.assertEqual(seen["smem"][0], 1, "smem did not allocate")
        self.assertEqual(seen["registers"][0], 0, "registers allocated SMEM")
        self.assertEqual(seen["gmem"][0], 0, "gmem allocated SMEM")
        self.assertEqual(seen["smem"][1], 1, "smem re-read the row from global")
        self.assertEqual(
            seen["registers"][1], 1, "registers re-read the row from global"
        )
        self.assertEqual(seen["gmem"][1], 2, "gmem did NOT re-read the row from global")
        self.assertEqual(seen["gmem"][2], 0, "gmem let the register cache fire")
        self.assertNotEqual(
            seen["registers"],
            seen["gmem"],
            "registers and gmem collapsed to one kernel",
        )

    def test_fail_cap_a_dead_buffer_is_detected(self) -> None:
        """⛔ THE FAIL-CAP: the positive arm's predicates must be able to go RED.

        The failure mode being guarded is a buffer that is allocated and published but never
        read -- correct answers, nothing failing.  Feed the predicates that exact shape
        (allocation + a ``partition_D`` publish, no ``partition_S`` read-back) and require
        that each one rejects it.  Without this, a drift in the emitted spelling would make
        the positive assertions match nothing and still show green.
        """
        dead = (
            f"_tv_smem_ptr = cute.arch.alloc_smem(cutlass.BFloat16, {_N}, alignment=16)\n"
            "_tv_stile_0 = cute.local_tile(_tv_smem, (1, 512), "
            "(cutlass.Int32(cute.arch.thread_idx()[1]), _tv_chunk_0))\n"
            "_tv_spart_0 = _tv_thr.partition_D(_tv_stile_0)\n"
            "_tv_tile_0 = cute.local_tile(x, (1, 512), (indices_1, _tv_chunk_0))\n"
            "_tv_part_0 = _tv_thr.partition_S(_tv_tile_0)\n"
            "cute.copy(_tv_atom, _tv_part_0[None, 0, lane_1], _tv_frag_0)\n"
            "cute.autovec_copy(_tv_frag_0, _tv_spart_0[None, 0, lane_1])\n"
            "_tv_tile_1 = cute.local_tile(x, (1, 512), (indices_1, _tv_chunk_1))\n"
            "_tv_part_1 = _tv_thr.partition_S(_tv_tile_1)\n"
            "cute.copy(_tv_atom, _tv_part_1[None, 0, lane_1], _tv_frag_1)\n"
        )
        # The publish IS seen -- so a test that only counted publishes would pass here.
        self.assertEqual(_staged_publishes(dead), ["_tv_frag_0"])
        # ...and each real predicate rejects it:
        self.assertEqual(
            _staged_reads(dead), [], "a read was found where there is none"
        )
        self.assertEqual(sorted(_partition_dirs(dead).values()), ["D"])
        self.assertEqual(
            _gmem_loads_per_tensor(dead).get("x"),
            2,
            "the dead-buffer shape must show TWO gmem reads of x -- that is the tell",
        )

    def test_declines_when_the_smem_capacity_refuses(self) -> None:
        """A device that cannot hold the tile is a DECLINE, and it says so.

        ⚠ A DECLINE MUST STAY A DECLINE.  Making one a raise broke all 8 attention examples,
        so the contract is that an unaffordable request falls back to a working kernel and
        records the reason on the artifact -- the ``Config`` is not evidence, since a cell can
        carry ``['smem']`` while the kernel stages nothing.

        ⛔⛔ THIS TEST WAS RED THREE INDEPENDENT WAYS AND EACH ONE WAS STALE, NOT BROKEN.
        Task 1 turned the refusal from a *tunable budget* into a *hardware capacity* check
        (the rule is now "honour the request and let it spill; only capacity and geometry may
        refuse"), and the test still spoke the old vocabulary:

        1. it passed the deleted config key ``cute_tv_sweep_cache`` (via ``_CFG``) ->
           ``InvalidConfig`` before any kernel was built;
        2. it patched ``_cute_stage_smem_budget_bytes``, which no longer exists ->
           ``AttributeError: <TileStrategy> does not have the attribute ...``.  ⚠ Note
           ``patch.object`` on a MISSING attribute raises, which is the good outcome; a
           ``create=True`` patch would have silently stubbed a method nobody calls and the
           test would have passed while exercising NOTHING;
        3. it asserted the reason string ``"SMEM budget"``, which the decline no longer
           emits -- it now names the device: ``"exceeds the device's shared memory"``.

        ⇒ renamed from ``..._budget_refuses`` and re-pointed at
        ``_cute_stage_smem_capacity_bytes``.  The DECLINE PATH itself -- the arithmetic, the
        per-kernel charge, the reason string -- stays in real code; only the capacity number
        is stubbed.
        """
        from helion._compiler.tile_strategy import TileStrategy

        x, w = self._inputs()
        with (
            _residency("smem"),
            _no_register_fuser(),
            unittest.mock.patch.object(
                TileStrategy, "_cute_stage_smem_capacity_bytes", lambda self: 0
            ),
        ):
            code, out = code_and_output(_tile_two_sweep_norm, (x, w, 1e-5), **_CFG)
        torch.testing.assert_close(
            out.float(), self._reference(x, w), atol=2e-2, rtol=2e-2
        )
        self.assertEqual(
            code.count("alloc_smem"), 0, "staging engaged on a zero capacity"
        )
        # The request is recorded, and the reason names DEVICE CAPACITY specifically -- two
        # different declines must read differently or one hardcoded string is passing for a
        # diagnosis.
        self.assertIn("requested smem", code)
        self.assertIn("exceeds the device's shared memory", code)

    def test_inert_with_the_tv_gate_off(self) -> None:
        """With ``HELION_CUTE_NDTILE_TV`` unset, a residency request changes NOTHING.

        Staging rides on the TV emission, so the gate-off arm must be byte-identical to the
        arm with no request at all.  Pinned as a SOURCE comparison rather than "no crash":
        the capability's claim is that it is invisible until asked for, and only the emitted
        text can check that.
        """
        from helion._compiler.tile_strategy import CuteNDTileStrategy

        x, w = self._inputs()
        base, out_base = code_and_output(_tile_two_sweep_norm, (x, w, 1e-5), **_CFG)
        with unittest.mock.patch.object(
            CuteNDTileStrategy,
            "_cute_row_residency_config",
            lambda self, fn, block_index: "smem",
        ):
            asked, out_asked = code_and_output(
                _tile_two_sweep_norm, (x, w, 1e-5), **_CFG
            )
        self.assertEqual(base, asked, "a residency request leaked through the gate")
        self.assertNotIn("alloc_smem", base)
        self.assertNotIn("make_tiled_copy_tv", base)
        torch.testing.assert_close(
            out_base.float(), self._reference(x, w), atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(out_asked.float(), out_base.float())


if __name__ == "__main__":
    import unittest

    unittest.main()
