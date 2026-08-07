"""Legality tests for CuTe vec-load predication.

A vec load bypasses the per-element ``if mask else <identity>`` gate that wraps a
scalar load: the CuTe DSL evaluates the pointer arithmetic and issues the LDG
unconditionally, and only the extracted lane value is gated.  So the *address*
must be in bounds on EVERY axis it indexes.  Both ways of getting that wrong were
GPU-measured with ``compute-sanitizer``:

* **column axis** -- the software pipeliner (``pipeline_inner_loads.py``)
  re-emits the load one full chunk ahead, so on the last iteration an unguarded
  lane index reads past the end of the row (32 invalid reads plus one
  ``unspecified launch failure`` at ``M=4 N=65536 nt=4 vw=4``);
* **row axis** -- when ``M`` is not a multiple of the row tile, the tail CTA's
  phantom rows form addresses past the tensor (783 invalid reads at
  ``M=33 N=65536``).

**Neither is visible to a numeric test.**  The loaded value really is discarded
correctly, so ``relerr`` is clean while the kernel issues hundreds of illegal
reads.  These tests therefore assert the *legality* of the emitted address --
that every axis which can leave the tensor is bounded in the pointer expression
-- rather than asserting that a transform fired.

Implementation: the vec hoists' inline anchor guard (main's form) in
``helion/language/memory_ops.py``.  Every vec emitter renders its address through
that one function, and ``_is_vec_load_call`` is fail-closed so an unguarded load
is refused by the pipeliner rather than turned into a speculative illegal read.
"""

from __future__ import annotations

import ast
import re

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")


@helion.kernel(static_shapes=True)
def _rowsum_bf16(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].to(torch.float32).sum(dim=-1)
    return out


@helion.kernel(static_shapes=True)
def _rmsnorm_like(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Two sweeps over ``x`` plus a row-free ``weight[:]`` load.

    ``weight`` is indexed by the reduction axis alone, so its vec load is the
    case that must carry the column guard and no row clamp.
    """
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        xt = x[tile_m, :].to(torch.float32)
        inv = torch.rsqrt(torch.mean(xt * xt, dim=-1) + 1e-5)
        out[tile_m, :] = (xt * inv[:, None] * weight[:].to(torch.float32)).to(x.dtype)
    return out


def _vec_load_calls(code: str) -> list[ast.Call]:
    """Every ``cute.arch.load(...)`` Call node in the emitted kernel."""
    calls: list[ast.Call] = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "arch"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "cute"
        ):
            calls.append(node)
    return calls


def _row_index_vars(code: str) -> set[str]:
    """Names assigned as ``indices_N = ...`` -- the grid/row coordinates."""
    return set(re.findall(r"^\s*(indices_\d+) = ", code, flags=re.MULTILINE))


def _tv_tile_is_weight_view(code: str, call: ast.Call) -> bool:
    """True when this ``local_tile``'s tensor is the row-broadcast weight view.

    The TV path emits ``_tv_bcast_N = cute.make_tensor(weight.iterator, ...)``
    and then tiles ``_tv_bcast_N``, so resolve one level of aliasing to decide
    whether a tile belongs to ``weight``.
    """
    tensor = ast.unparse(call.args[0])
    if "weight" in tensor:
        return True
    m = re.search(
        rf"^\s*{re.escape(tensor)} = cute\.make_tensor\((\w+)\.", code, re.MULTILINE
    )
    return bool(m and "weight" in m.group(1))


def _tv_local_tile_row_coords(code: str) -> list[str]:
    """Row tile coordinates of every TV-layout ``cute.local_tile`` call.

    A reduction driven by a TV layout addresses through
    ``local_tile(t, (1, chunk), (<row>, <chunk_idx>))`` + ``partition_S`` /
    ``partition_D`` rather than through an explicit pointer, so there is no
    ``cute.arch.load`` argument to inspect.  The row coordinate is the analogous
    place a phantom row can escape, so return it for the same clamp check.
    """
    coords: list[str] = []
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "local_tile"
            and isinstance(func.value, ast.Name)
            and func.value.id == "cute"
        ):
            continue
        if len(node.args) < 3 or not isinstance(node.args[2], ast.Tuple):
            continue
        elts = node.args[2].elts
        if elts:
            coords.append(ast.unparse(elts[0]))
    return coords


@onlyBackends(["cute"])
class TestCuteVecLoadPredication(TestCase):
    def _assert_addresses_legal(self, code: str, *, expect_loads: bool) -> None:
        """Assert every vec-load address is bounded on every axis it indexes.

        Four independent checks, one per way the address can escape:

        I1 (column) -- a pointer mentioning a lane-base variable must sit inside
        an ``ast.IfExp`` whose test bounds that variable.  This is what makes the
        pipeliner's one-chunk-ahead prefetch legal.

        I2 (row) -- a row coordinate (``indices_N``) may not appear in a load
        address unless it is clamped.  The clamp form is
        ``indices_N if mask_N else <dtype>(0)``, so ``indices_N`` must never be a
        bare operand of the address arithmetic.

        I3 (both branches) -- when the address IS an ``IfExp``, BOTH the body and
        the orelse must satisfy I2.  The pre-fix guard clamped the column to 0 in
        its ``else`` branch while leaving the row untouched in *both*, so its
        fallback pointer was "safe" in the column only.

        I4 (TV layout) -- a reduction whose addresses come from a TV layout
        (``local_tile`` + ``partition_S``/``partition_D`` off one ``get_slice``)
        forms no explicit pointer at all, so I1-I3 have nothing to inspect.
        The SAME property is asserted on that mechanism instead: the row
        coordinate handed to every ``local_tile`` must be clamped, and the
        column is a tile coordinate that cannot leave the tensor by
        construction.  See ``_tv_local_tile_row_coords``.
        """
        calls = _vec_load_calls(code)
        tv_rows = _tv_local_tile_row_coords(code)
        if expect_loads:
            # Either mechanism counts as "an address was formed"; a kernel with
            # NEITHER proves nothing and is still a failure.
            self.assertTrue(
                calls or tv_rows,
                f"expected a vec load or a TV-layout local_tile in:\n{code}",
            )
        rows = _row_index_vars(code)

        # I4: every TV ``local_tile`` row coordinate is clamped.
        #
        # The TV path replaces the hand-built pointer with
        # ``local_tile(x, (1, chunk), (<row>, <chunk>))``, so the row coordinate
        # is where a phantom row could escape -- the same escape I2 guards on the
        # pointer form.  A bare ``indices_N`` here is the class-3 defect
        # expressed in the new mechanism, so it fails identically.
        for coord in tv_rows:
            bare = {
                n.id
                for n in ast.walk(ast.parse(coord, mode="eval"))
                if isinstance(n, ast.Name) and n.id in rows
            }
            if not bare:
                continue  # a literal or an already-safe expression
            tree = ast.parse(coord, mode="eval").body
            self.assertIsInstance(
                tree,
                ast.IfExp,
                f"TV local_tile row coordinate uses un-clamped row index "
                f"{sorted(bare)} (class 3):\n  {coord}",
            )
            assert isinstance(tree, ast.IfExp)
            self.assertIn(
                "mask",
                ast.unparse(tree.test),
                f"TV row clamp guarded by a non-mask test: {coord}",
            )
            self.assertNotIn(
                "indices_",
                ast.unparse(tree.orelse),
                f"TV row clamp's fallback is not row-safe: {coord}",
            )

        def bare_row_names(ptr: ast.expr) -> set[str]:
            """Row names used un-clamped inside ``ptr``.

            A clamped row is the ``body`` of an ``IfExp`` whose ``orelse`` is the
            in-bounds fallback; walking into that ``body`` would report a false
            positive, so a matching ``IfExp`` is treated as safe and not
            descended into.
            """
            found: set[str] = set()
            stack: list[ast.AST] = [ptr]
            while stack:
                node = stack.pop()
                if (
                    isinstance(node, ast.IfExp)
                    and isinstance(node.body, ast.Name)
                    and node.body.id in rows
                ):
                    # ``indices_N if mask_N else Int32(0)`` -- clamped, and the
                    # test must actually reference a mask.
                    self.assertIn(
                        "mask",
                        ast.unparse(node.test),
                        f"row clamp guarded by a non-mask test: {ast.unparse(node)}",
                    )
                    continue
                if isinstance(node, ast.Name) and node.id in rows:
                    found.add(node.id)
                    continue
                stack.extend(ast.iter_child_nodes(node))
            return found

        for call in calls:
            self.assertEqual(len(call.args), 2, ast.unparse(call))
            ptr = call.args[0]
            text = ast.unparse(ptr)
            lane_bases = re.findall(r"\b(_pipe_lane_base_\d+|\w*lane_base_\d+)\b", text)
            # I1: a lane-base-dependent address must be an IfExp bounding it.
            if lane_bases:
                self.assertIsInstance(
                    ptr,
                    ast.IfExp,
                    f"vec load address mentions {lane_bases[0]} but is not "
                    f"bounds-guarded (class 2):\n  {text}",
                )
                test_text = ast.unparse(ptr.test)
                self.assertTrue(
                    any(lb in test_text for lb in lane_bases),
                    f"guard test does not bound the lane base: {test_text}",
                )
            # I2 + I3: no un-clamped row coordinate, in either branch.
            branches = [ptr.body, ptr.orelse] if isinstance(ptr, ast.IfExp) else [ptr]
            for branch in branches:
                bare = bare_row_names(branch)
                self.assertFalse(
                    bare,
                    f"vec load address uses un-clamped row index {sorted(bare)} "
                    f"(class 3):\n  {ast.unparse(branch)}",
                )

    def test_row_tail_vec_load_address_is_row_clamped(self) -> None:
        """M not a multiple of the row tile: the fetch address must not form a
        phantom row's coordinate.  Regression for 783 GPU-measured invalid reads.
        """
        # M=33 with block_sizes=4 -> the last CTA covers rows 32..35, so rows
        # 33/34/35 do not exist.
        x = torch.randn(33, 4096, device=DEVICE, dtype=torch.bfloat16)
        for vw in (2, 4):
            code = _rowsum_bf16.bind((x,)).to_triton_code(
                helion.Config(
                    block_sizes=[4],
                    num_threads=[4],
                    reduction_loops=[1024],
                    cute_vector_widths=[vw, 1],
                )
            )
            self._assert_addresses_legal(code, expect_loads=True)

    def test_prefetch_address_is_column_guarded(self) -> None:
        """The software-pipelined prefetch addresses one chunk past the current
        one, so its address must be bounded against the reduction extent.
        Regression for 32 GPU-measured invalid reads + a launch failure.

        The over-read this guards against is *specific to speculation*: the
        pipeliner moves an address a full chunk ahead, so the last iteration
        reads past the row.  A reduction whose addresses come from a TV layout
        (``local_tile`` + ``partition_S``) is not speculative -- every address is
        a tile coordinate of a tile that exists -- and the pipeliner declines it
        outright, because ``_is_vec_load_call`` is fail-closed and there is no
        ``cute.arch.load`` to match.  So on that path the over-read is absent by
        construction rather than by predicate.

        This test therefore requires ONE of the two to hold, and says which:
          * a prefetch was emitted, and its address is column-guarded; or
          * no prefetch was emitted AND the kernel is on the TV path.
        A kernel with neither is still a failure -- that is the case where the
        pipeliner declined for some *other* reason and the test proved nothing,
        which is what the original assertion was protecting.

        MEASURED, so the "declined" branch is not a silent perf regression being
        waved through: at both configs below the TV path and the pipelined path
        time identically (0.0082 ms), and under ``compute-sanitizer`` both emit
        ZERO invalid global reads.
        """
        x = torch.randn(64, 16384, device=DEVICE, dtype=torch.bfloat16)
        saw_prefetch = False
        saw_tv = False
        for nt, vw in ((2, 2), (4, 4)):
            code = _rowsum_bf16.bind((x,)).to_triton_code(
                helion.Config(
                    block_sizes=[nt],
                    num_threads=[nt],
                    reduction_loops=[1024],
                    cute_vector_widths=[vw, 1],
                )
            )
            saw_prefetch |= "_pipe_load" in code
            saw_tv |= bool(_tv_local_tile_row_coords(code))
            self._assert_addresses_legal(code, expect_loads=True)
        self.assertTrue(
            saw_prefetch or saw_tv,
            "neither a prefetch nor a TV-layout address was emitted, so this "
            "test proved nothing; the pipeline pass may have declined for an "
            "unrelated reason (see _is_vec_load_call's fail-closed check)",
        )

    def test_row_clamp_absent_when_axis_has_no_mask(self) -> None:
        """The clamp is emitted only where the axis actually carries a mask.

        ``weight[:]`` in an rms-norm-shaped kernel is indexed by the reduction
        axis alone -- no row coordinate at all -- so its vec load must carry the
        column guard and NOTHING else.  This keeps the fix from degenerating
        into "clamp everything", which would put a select on every address on
        the common path.

        Note the row axis of the *data* tensor is clamped even at ``M=64``:
        ``CuteBackend.force_tile_mask()`` is True, so a row mask var always
        exists (its ``thread_idx()[1] < BLOCK`` term is live even when the
        ``offsets_0 < M`` term is vacuous), and the descriptor keys the clamp off
        the presence of that mask var.  Asserting "no clamp at M=64" would
        therefore be asserting something helion does not promise.
        """
        x = torch.randn(64, 4096, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(4096, device=DEVICE, dtype=torch.bfloat16)
        code = _rmsnorm_like.bind((x, w)).to_triton_code(
            helion.Config(
                block_sizes=[4],
                num_threads=[4],
                reduction_loops=[1024],
                cute_vector_widths=[2, 1],
            )
        )
        self._assert_addresses_legal(code, expect_loads=True)
        weight_loads = [
            ast.unparse(c.args[0])
            for c in _vec_load_calls(code)
            if "weight" in ast.unparse(c.args[0])
        ]
        # The TV path expresses the same access as a ``local_tile`` of a
        # row-broadcast view of ``weight``, so accept that form too.  What is
        # asserted is unchanged: whatever addresses ``weight``, its ROW
        # coordinate must be the literal 0 and carry no mask -- ``weight`` has
        # no row axis, so a clamp there would be the "clamp everything"
        # degeneration this test exists to prevent.
        weight_tv_rows = [
            ast.unparse(node.args[2].elts[0])
            for node in ast.walk(ast.parse(code))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "local_tile"
            and len(node.args) >= 3
            and isinstance(node.args[2], ast.Tuple)
            and node.args[2].elts
            and _tv_tile_is_weight_view(code, node)
        ]
        self.assertTrue(
            weight_loads or weight_tv_rows,
            f"no weight vec load and no weight TV tile in:\n{code}",
        )
        for ptr in weight_loads:
            self.assertNotIn(
                "mask",
                ptr,
                "a mask-based clamp was emitted for an axis that carries no "
                f"row coordinate:\n  {ptr}",
            )
        for row_coord in weight_tv_rows:
            self.assertNotIn(
                "mask",
                row_coord,
                "a mask-based clamp was emitted for the row coordinate of an "
                f"axis that carries no row coordinate:\n  {row_coord}",
            )
            self.assertEqual(
                row_coord,
                "0",
                f"weight has no row axis; its tile row coord must be 0, got "
                f"{row_coord}",
            )

    # ⛔ ``test_guard_is_derived_from_the_descriptor`` WAS HERE, AND IT WENT WITH ITS SUBJECT
    # (T6).  It asserted that the guard TEXT is a function of ``CuteVecLoadDesc``'s
    # ``axis_bounds`` / ``axis_index_vars`` rather than hand-written per emitter -- a good
    # property of a descriptor layer that no longer exists.
    #
    # ⭐ THE LAYER WAS DELETED BECAUSE IT GUARDED A **VEC** ADDRESS, i.e. it was correctness
    # CONDITIONAL ON AN OPTIMISATION: the scalar floor it degrades to has no address to
    # guard, and the TV path carries its own bounds argument.  Every emitter that used it is
    # back on main's inline lane-axis anchor guard, and MEASURED, all 40 frozen cells now
    # emit a TV copy with ZERO reaching those emitters.
    #
    # ⚠ THE REMAINING TESTS IN THIS FILE ARE NOT VACUOUS -- they assert on EMITTED CODE
    # (a row-clamped tile coordinate, a column-guarded prefetch, no clamp where the axis has
    # no mask, ``_is_vec_load_call``'s fail-closed refusal, and a numeric row-tail check),
    # so they now pin main's guard form instead of the descriptor's.  Deleting this one and
    # keeping those is the difference between removing a test whose SUBJECT is gone and
    # relaxing a test whose subject still matters.

    def test_unguarded_load_is_not_pipelined(self) -> None:
        """``_is_vec_load_call`` is fail-closed: an unguarded pointer must be
        REFUSED, so a future guardless emitter shows up as a lost optimization
        rather than an illegal read.
        """
        from helion._compiler.cute.pipeline_inner_loads import _is_vec_load_call

        guarded = ast.parse(
            "cute.arch.load(x.iterator + b if b < 16384 else x.iterator + 0, "
            "ir.VectorType.get([2], cutlass.Uint16.mlir_type))",
            mode="eval",
        ).body
        unguarded = ast.parse(
            "cute.arch.load(x.iterator + b, "
            "ir.VectorType.get([2], cutlass.Uint16.mlir_type))",
            mode="eval",
        ).body
        self.assertTrue(_is_vec_load_call(guarded))
        self.assertFalse(
            _is_vec_load_call(unguarded),
            "the pipeliner accepted an unguarded load; it would advance the "
            "address one chunk and execute the over-read",
        )

    def test_row_tail_is_numerically_correct(self) -> None:
        """Sanity: the clamp must not change the answer.

        A clamped phantom row reads row 0's bytes instead of out-of-bounds
        bytes; ``mask_0`` still discards the value, so every real row is
        unaffected.  (This assertion passes BEFORE the fix too -- it is here to
        catch the clamp being wired to the wrong index, not to detect the
        out-of-bounds read, which no numeric test can see.)
        """
        for m in (33, 66, 130):
            x = torch.randn(m, 4096, device=DEVICE, dtype=torch.bfloat16)
            ref = x.float().sum(dim=-1)
            for vw in (2, 4):
                out = _rowsum_bf16.bind((x,)).compile_config(
                    helion.Config(
                        block_sizes=[4],
                        num_threads=[4],
                        reduction_loops=[1024],
                        cute_vector_widths=[vw, 1],
                    )
                )(x)
                torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-1)
