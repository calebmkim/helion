"""The SMEM staging tile (``cute_row_residency="smem"``, capability ③) admits ONE tensor.

⭐ THIS PINS A FIXED SOUNDNESS BUG, NOT A POLICY.  The staged tile is minted once per
reduction (``memory_ops._cute_tv_stage_slice``) and sized ``rows_per_cta * N`` by
``ChunkTVPlan.stage_smem_elems`` -- it has a ROW mode and a CHUNK mode and **no TENSOR
mode**.  ``_cute_tv_staged_tensors`` is a set of NAMES, though, so before the gate this
file pins, an arbitrary number of tensors was admitted into that one buffer and each
partitioned ``local_tile(_tv_smem, (1, chunk), (row, chunk_idx))`` at the SAME
coordinates.  The second tensor's publish overwrote the first's, and both staged reads
returned the second.

MEASURED on the LOOPED path in an unmodified tree -- so this was a live wrong-answer bug
on the path the frozen perf cells use, not a hazard introduced by widening ③:

    _tv_smem_ptr = cute.arch.alloc_smem(cutlass.BFloat16, 2048, alignment=16)
    _tv_stile_0 = cute.local_tile(_tv_smem, (1, 512), (thread_idx()[1], _tv_chunk_1))
    _tv_spart_0 = _tv_thr.partition_D(_tv_stile_0)                   # x's writer
    _tv_stile_1 = cute.local_tile(_tv_smem, (1, 512), (thread_idx()[1], _tv_chunk_1))
    _tv_spart_1 = _tv_thr.partition_D(_tv_stile_1)                   # y's writer, SAME SLOT
    cute.autovec_copy(_tv_frag_0, _tv_spart_0[None, 0, reduction_lane_1])
    cute.autovec_copy(_tv_frag_1, _tv_spart_1[None, 0, reduction_lane_1])   # clobbers x

⚠ WHY NO EXISTING TEST SEES IT, and why this file had to be new rather than a case added
to an existing one.  Every cell that stages today stages exactly ONE 2-D tensor:
``rms_norm`` / ``layer_norm`` / ``cross_entropy`` stage the input and nothing else,
because a rank-1 ``weight[:]`` is declined separately as an M-broadcast.  MEASURED across
all 40 frozen cells: **0 of them stage more than one tensor**, so the bug is unreachable
from the perf table by construction.  It takes TWO multi-read 2-D tensors on ONE reduction
axis -- which is what ``_two_tensor_norm`` below is -- to make it reachable at all.

⚠ AND THE ASSERTIONS ARE ON THE EMITTED SOURCE, not only on the numbers.  A previous
staging implementation allocated and published a buffer nobody read: every answer was
correct and nothing failed.  The structural half is what distinguishes "staging engaged"
from "staging looked like it engaged", which is the same standard the chunk-coordinate
guard applies.
"""

from __future__ import annotations

import os
import re
import unittest.mock

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(backend="cute", static_shapes=True)
def _two_tensor_norm(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """TWO multi-read 2-D row tensors on ONE rolled reduction axis.

    ``x[tile_m, :]`` is a bare ``slice(None)``, which is what puts the axis on
    ``LoopedReductionStrategy`` -- the path 21 of the frozen cells use and the path the
    aliasing bug was measured on.  Both ``x`` and ``y`` are read in the reduce sweep AND
    again in the consume sweep, so ``_cute_tv_multi_read_tensors`` reports BOTH and both
    reach the staging gate.  That is the whole point of the shape: one multi-read tensor
    can never expose the alias.
    """
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        xt = x[tile_m, :].to(torch.float32)
        yt = y[tile_m, :].to(torch.float32)
        inv = torch.rsqrt((xt * xt).sum(dim=-1) / n + (yt * yt).sum(dim=-1) / n + eps)
        out[tile_m, :] = ((xt + yt) * inv[:, None]).to(out.dtype)
    return out


# ⚠ EVERY CONFIG HERE IS PINNED, so the autotuner never runs: an autotuned config would
# be a different kernel per invocation and the structural assertions below would be
# pinning whatever the tuner happened to pick.
def _cfg(**extra: object) -> helion.Config:
    return helion.Config(
        block_sizes=[1],
        num_threads=[1],
        reduction_loops=[512],
        cute_threads_per_row=[32],
        cute_vector_widths=[8, 1],
        # Task 1: was ``cute_reduction_reload=["smem"]``; the residency axis is now the
        # one key that requests SMEM staging.
        cute_row_residency=["smem"],
        **extra,  # pyrefly: ignore [bad-argument-type]
    )


def _no_register_fuser() -> unittest.mock._patch:
    """Withhold the register-cache pass, so SMEM staging is the only mechanism in play.

    ⭐ THIS IS LOAD-BEARING, NOT TIDINESS.  When the register cache is live,
    ``fuse_tv_copy_sweeps`` caches the row's fragment lanes and serves the second sweep from
    THERE -- so the staged read stops being the thing that feeds the consume sweep, and a test
    that then corrupted the staged tile would still see a BIT-IDENTICAL answer.  Withholding it
    is what makes the numeric assertions below evidence rather than decoration.

    ⛔⛔ THIS USED TO BE ``cute_tv_sweep_cache=[0]`` IN ``_cfg`` AND THAT KNOB IS DELETED.
    Task 1 removed it with the slot budget it capped, so every config built here raised
    ``InvalidConfig: Invalid config keys ['cute_tv_sweep_cache']`` in ``config_spec.normalize``
    -- before any kernel existed.  ⇒ two tests in this file were red for a dead config key,
    which reads like a broken mechanism and is not one.  ``HELION_TV_SWEEP_FUSE=disabled``
    (``fuse_tv_copy_sweeps.py:481``) is the surviving equivalent and is strictly better: unlike
    the numeric override it cannot silently UN-VETO an explicit ``gmem`` request.
    """
    return unittest.mock.patch.dict(os.environ, {"HELION_TV_SWEEP_FUSE": "disabled"})


def _staged_tile_coords(code: str) -> list[tuple[str, str]]:
    """``(row_coord, chunk_coord)`` for every ``local_tile`` of the staging buffer."""
    return [
        (row, chunk)
        for row, chunk in re.findall(
            r"cute\.local_tile\(_tv_smem, \(1, \d+\), \((.*?), (\w+)\)\)", code
        )
    ]


def _partition_dirs(code: str) -> dict[str, str]:
    """``_tv_spart_N -> "D" | "S"`` for each staging partition declaration."""
    return dict(re.findall(r"(_tv_spart_\d+) = \w+\.partition_([DS])\(", code))


def _staged_read_frags(code: str) -> set[str]:
    """Fragments whose read is served from SMEM (``autovec_copy(_tv_spart..., frag)``)."""
    return set(
        re.findall(
            r"cute\.autovec_copy\(_tv_spart_\d+\[[^]]*\], (_tv_frag_\d+)\)", code
        )
    )


def _gmem_loads_per_tensor(code: str) -> dict[str, int]:
    """``tensor name -> how many GMEM ``cute.copy`` LOADS it issues``.

    ⭐ THE COUNT IS THE POINT, NOT THE PRESENCE.  Staging does NOT remove a tensor's
    gmem reads -- the FIRST sweep must still read global, because that read is what
    FILLS the tile.  What staging removes is the SECOND read.  So "is x still loaded
    from gmem" is the wrong question (the answer is yes, once, correctly) and "how many
    times" is the right one.  Asserting presence/absence here is how a check that looks
    strict ends up pinning the opposite of the property.

    Loads are told apart from STORES by operand position: a load has the partition in
    the SOURCE slot (``copy(atom, part[...], frag)``), a store flush has it in the
    DESTINATION slot.  Both share the ``cute.copy`` spelling.
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


@onlyBackends(["cute"])
class TestCuteStagedTileAlias(TestCase):
    def test_second_tensor_declines_instead_of_aliasing(self) -> None:
        """⭐ THE POSITIVE HALF: exactly ONE tensor owns the staged tile.

        Asserted on the emitted source, because the numbers alone cannot tell "y was
        refused and re-read from gmem" (correct) from "y aliased x and happened to
        cancel" (wrong).  The four assertions are the four ways the alias was visible.
        """
        m, n = 256, 2048
        torch.manual_seed(0)
        x = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
        cfg = _cfg()
        bound = _two_tensor_norm.bind((x, y, 1e-5))
        with _no_register_fuser():
            code = bound.to_triton_code(cfg)

        # (0) NON-VACUITY.  If staging never engaged at all, every assertion below holds
        # trivially -- which is exactly how this test would rot into a no-op.
        self.assertEqual(
            code.count("alloc_smem"), 1, "staging did not engage; the test is vacuous"
        )
        self.assertIn("'row residency: smem'", code)

        # (1) ONE buffer, and exactly ONE writer + ONE reader off it.  Before the fix
        # there were TWO ``partition_D`` writers (x's and y's) into one buffer.
        dirs = _partition_dirs(code)
        self.assertEqual(
            sorted(dirs.values()),
            ["D", "S"],
            f"expected exactly one staged writer and one staged reader, got {dirs}",
        )

        # (2) NO TWO STAGED TILES SHARE A COORDINATE.  This is the alias itself: the
        # buffer has no tensor mode, so two tiles at the same ``(row, chunk)`` ARE the
        # same memory.  MEASURED before the fix: ``_tv_stile_0`` and ``_tv_stile_1`` both
        # at ``(thread_idx()[1], _tv_chunk_1)``.
        coords = _staged_tile_coords(code)
        self.assertEqual(
            len(coords),
            len(set(coords)),
            f"two staged local_tiles share a (row, chunk) coordinate -- they alias the "
            f"same SMEM: {coords}",
        )

        # (3) ⭐ THE REFUSED TENSOR'S SECOND READ STILL COMES FROM GLOBAL, and the
        # OWNER'S DOES NOT.  This is the assertion that says "y was declined" rather
        # than "y silently shared x's slot": exactly one tensor is served from SMEM and
        # the other is still issuing a gmem ``cute.copy`` in the consume sweep.
        #
        # ⚠ NOT ASSERTED ON THE ``row residency:`` MARKER, and that is deliberate rather
        # than a weaker check.  ``_cute_tv_record_residency`` records PER-TENSOR facts at
        # ``log.debug`` only; the ONE canonical artifact line is written by
        # ``cute_emit_row_residency_marker`` from the FINAL body and describes the
        # kernel's reduced row, not each tensor's fate.  memory_ops.py:812-822 records
        # that per-tensor marker lines were REMOVED precisely because "x -> smem" plus
        # "weight -> gmem" on one kernel does not answer one question.  So the decline is
        # asserted where it is observable: in the instructions.
        staged_frags = _staged_read_frags(code)
        self.assertEqual(
            len(staged_frags),
            1,
            f"expected exactly one fragment served from SMEM, got {staged_frags}",
        )
        loads = _gmem_loads_per_tensor(code)
        # ⭐ THE ASYMMETRY IS THE EVIDENCE.  This kernel sweeps the row THREE times
        # (reduce x, reduce y, consume both).  ``x`` owns the tile, so one of its reads
        # is replaced by the staged read; ``y`` was declined, so it keeps all of its.
        # If ``y`` had aliased ``x``'s slot instead of declining, its count would have
        # DROPPED too -- which is the wrong answer that looked fast.
        self.assertGreater(
            loads.get("y", 0),
            loads.get("x", 0),
            f"y should issue MORE gmem loads than the staged x (y was declined and "
            f"re-reads global; x's later read comes from SMEM): {loads}",
        )

        # (4) ...and the answer is right.
        ref = (x.float() + y.float()) * torch.rsqrt(
            (x.float() ** 2).mean(-1, keepdim=True)
            + (y.float() ** 2).mean(-1, keepdim=True)
            + 1e-5
        )
        got = bound.compile_config(cfg)(x, y, 1e-5)
        torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)

    def test_fail_cap_a_shared_coordinate_is_detected(self) -> None:
        """⛔ THE FAIL-CAP: the coordinate check must be able to go RED.

        Assertion (2) above is the load-bearing one -- it is what "the tile has no tensor
        mode" reduces to -- and a regex-based structural check is exactly the kind that
        passes because it matched nothing.  So feed it the PRE-FIX emission (verbatim,
        from the measurement in this file's docstring) and require that it fails.

        Without this, deleting the gate under test and leaving assertion (2) in place
        could still show green if the emitted spelling ever drifted.
        """
        pre_fix = (
            "_tv_smem_ptr = cute.arch.alloc_smem(cutlass.BFloat16, 2048, alignment=16)\n"
            "_tv_stile_0 = cute.local_tile(_tv_smem, (1, 512), "
            "(cutlass.Int32(cute.arch.thread_idx()[1]), _tv_chunk_1))\n"
            "_tv_spart_0 = _tv_thr.partition_D(_tv_stile_0)\n"
            "_tv_stile_1 = cute.local_tile(_tv_smem, (1, 512), "
            "(cutlass.Int32(cute.arch.thread_idx()[1]), _tv_chunk_1))\n"
            "_tv_spart_1 = _tv_thr.partition_D(_tv_stile_1)\n"
        )
        coords = _staged_tile_coords(pre_fix)
        # The parser must actually SEE both tiles: a regex that matched zero lines would
        # make the duplicate test pass vacuously, which is the failure this guards.
        self.assertEqual(len(coords), 2, f"parser saw {coords}, expected 2 tiles")
        self.assertNotEqual(
            len(coords),
            len(set(coords)),
            "the coordinate check cannot detect the pre-fix alias, so it proves nothing",
        )
        # And the direction check must see the two writers the fix removed.
        self.assertEqual(sorted(_partition_dirs(pre_fix).values()), ["D", "D"])
        # The staged-read parser must also be able to distinguish a read from a publish:
        # both are ``autovec_copy``, told apart only by which operand is subscripted.
        # If this could not tell them apart, assertion (3) would count publishes as
        # reads and a buffer nobody read would look like a working one.
        publish_only = (
            "cute.autovec_copy(_tv_frag_0, _tv_spart_0[None, 0, reduction_lane_1])\n"
        )
        self.assertEqual(_staged_read_frags(publish_only), set())
        read_only = (
            "cute.autovec_copy(_tv_spart_1[None, 0, reduction_lane_1], _tv_frag_3)\n"
        )
        self.assertEqual(_staged_read_frags(read_only), {"_tv_frag_3"})

    def test_single_staged_tensor_is_unaffected(self) -> None:
        """The ordinary one-tensor case still stages -- the gate is a decline, not a ban.

        ⚠ THE POINT IS THAT THE FIX IS NOT A REGRESSION.  "Admit at most one tensor" and
        "admit none" are indistinguishable from the two-tensor kernel above, so the
        inertness of the fix on every shape that works today needs its own arm.
        """
        m, n = 256, 2048
        torch.manual_seed(0)
        x = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
        from examples.rms_norm import rms_norm_fwd

        cfg = _cfg()
        bound = rms_norm_fwd.bind((x, w, 1e-5))
        with _no_register_fuser():
            code = bound.to_triton_code(cfg)
        # Staging engaged, one writer + one reader, and no aliasing coordinate.
        self.assertEqual(code.count("alloc_smem"), 1)
        self.assertEqual(sorted(_partition_dirs(code).values()), ["D", "S"])
        coords = _staged_tile_coords(code)
        self.assertEqual(len(coords), len(set(coords)))
        # ⭐ STAGING REPLACES A GMEM READ.  ``rms_norm`` sweeps twice, so ``x`` must
        # issue exactly ONE gmem load (sweep 1, which FILLS the tile) and get sweep 2
        # from SMEM.  A buffer that were allocated and published but never read would
        # leave ``x`` at TWO -- and would still compute the right answer, which is why
        # this is asserted structurally.
        self.assertEqual(len(_staged_read_frags(code)), 1)
        self.assertEqual(
            _gmem_loads_per_tensor(code).get("x"),
            1,
            f"x should read gmem exactly once (sweep 1 fills the tile; sweep 2 reads "
            f"SMEM): {_gmem_loads_per_tensor(code)}",
        )
        out, _ = bound.compile_config(cfg)(x, w, 1e-5)
        torch.testing.assert_close(
            out.float(),
            torch.nn.functional.rms_norm(x.float(), (n,), w.float(), eps=1e-5),
            atol=2e-2,
            rtol=2e-2,
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
