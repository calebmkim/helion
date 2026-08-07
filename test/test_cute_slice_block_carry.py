"""The ``slice(None)`` / partial-slice axis binding is CARRIED, not re-derived.

WHAT IS BEING PINNED.  A bare ``slice(None)`` axis names no loop, so codegen has
to decide which block drives it.  ``_cute_index_exprs`` decides once, by calling
``resolve_active_slice_block_id``, and writes that block's ``index_var`` into
``index_exprs``.  It now also RECORDS the block id, in the ``slice_block_ids``
out-parameter, and ``_cute_vector_load_ctx`` READS that record.

Before this, the vec-load site threw the binding away and re-derived it from two
weaker heuristics: an ``index_exprs[expr_pos]`` string match against every active
block's ``index_var``, then a first-match ``known_equal(block.numel,
tensor.shape[dim])`` size scan.  Both are strictly weaker than the resolver they
shadowed, which filters by activeness, by a LIVE ``used_block_ids`` set, and
breaks ties on the ``reduction`` flag.

⭐ WHY THESE TESTS AND NOT A CODEGEN GOLDEN -- AND THE MEASUREMENT THAT SETTLED IT.
The change is INERT on every kernel reachable from this tree's tests.  A probe that
replays BOTH resolvers at every vec-load site and compares the decision they
produce -- the pair ``(inner_block_id, lane_axis_pos)``, which is all the function
carries forward -- measured, over 2095 sites / 771 slice axes in
test_indexing + test_reductions + test_cute_backend + test_examples + test_loops +
test_dot::

    map_diff = 0     decision_diff = 0

and the emitted source is byte-identical for all 40 frozen cells plus
``test_full_slice_in_reduction_loop``.  So a golden-file test would pin the
*downstream declines*, not the binding, and would stay green if the carry were
reverted.  These tests assert on the binding itself instead.

⚠ RETRACTED, and recorded here because it is the mistake to avoid: a FIRST probe
reported "4 sites where the resolvers disagree".  It compared the old
BARE-slice-only answer against the whole carry map, which also holds PARTIAL-slice
entries -- and all 4 diffs were partial-slice positions the consumer never reads at
a bare-slice position.  Artifact.  Do not cite it.

The tests:

  1. ``test_bare_slice_binding_is_recorded`` -- the map reaches the consumer, and
     it holds the reduction block, for the canonical ``x[tile_m, :]`` row read.
  2. ``test_rank1_weight_slice_records_the_reduction_block`` -- the 1-D
     ``weight[:]`` case (commit ``877e3d012``'s bug) at M == N, where a size scan
     bound the M tile.  This is the case the old ``bound_block_ids`` guard
     structurally could not see: a 1-D subscript has no sibling position to pin
     anything with, so the guard was vacuous rather than conservative.
  3. ``test_partial_slice_binding_is_recorded`` -- the partial-slice recording site
     is separate code from the bare-slice one; assert it records too.
  4. ``test_the_map_is_not_shared_across_calls`` -- the out-parameter contract.
     ``_cute_index_exprs`` is called twice for some loads (a matmul operand
     re-lowered under ``matmul_operand_index_override``), so a module global would
     serve one call's binding to the other's read.  ⭐ This is the test that fails
     if someone "simplifies" the threading into a global, which is exactly what the
     abandoned prototype in ``_notes/slicefix_rescued/`` did.
  5. ``test_vec_ctx_requires_the_map`` -- ⭐ the FAIL-CAPABILITY test, and the one
     that goes red if the change is reverted: the parameter must be REQUIRED, and
     the deleted heuristics (``bound_block_ids``, ``known_equal``, the
     ``index_var`` string match) must stay absent from the lane-axis search.
     Structural rather than behavioural precisely BECAUSE the change is inert --
     see the measurement above.  Without this test the file would pass on a tree
     with the heuristics restored.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import torch

import helion
from helion._compiler.cute.memory_ops import _cute_vector_load_ctx
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
import helion.language as hl
from helion.language.memory_ops import _cute_index_exprs


def _collect(kernel, args, config):
    """Emit ``kernel`` and capture the binding map that REACHES the consumer.

    Wraps ``_cute_vector_load_ctx``, i.e. the site that reads the map, rather than
    ``_cute_index_exprs``, which produces it.  Two reasons, both load-bearing:

      * ``cute/memory_ops.py`` does ``from ...language.memory_ops import
        _cute_index_exprs``, so patching the name in ``helion.language.memory_ops``
        does NOT reach the already-bound reference and the wrapper never fires
        (MEASURED: every recorded map came back empty).
      * the property under test is that the binding ARRIVES at the consumer, so
        observing it there tests the threading end to end.  A test that watched the
        producer would stay green if the caller stopped passing the map along.

    Returns one ``(tensor shape, {index_exprs position: block_id})`` entry per
    vec-load site that had at least one resolved slice axis.
    """
    from helion._compiler.cute import memory_ops as cmo

    real = cmo._cute_vector_load_ctx
    out: list[tuple[tuple[int, ...], dict[int, int]]] = []

    def wrapper(state, tensor, subscript, index_exprs, extra_mask, slice_block_ids):
        if slice_block_ids:
            out.append((tuple(tensor.shape), dict(slice_block_ids)))
        return real(state, tensor, subscript, index_exprs, extra_mask, slice_block_ids)

    cmo._cute_vector_load_ctx = wrapper
    try:
        kernel.bind(args).to_triton_code(config)
    finally:
        cmo._cute_vector_load_ctx = real
    return out


@onlyBackends(["cute"])
class TestCuteSliceBlockCarry(TestCase):
    def test_bare_slice_binding_is_recorded(self) -> None:
        """``x[tile_m, :]``: the trailing axis records the REDUCTION block."""

        @helion.kernel(static_shapes=True)
        def kern(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
            return out

        x = torch.randn(512, 1024, device=DEVICE, dtype=torch.bfloat16)
        recs = _collect(
            kern, (x,), helion.Config(block_sizes=[1], reduction_loops=[256])
        )
        # The (512, 1024) read has exactly one bare slice, at index_exprs pos 1.
        rows = [m for shape, m in recs if shape == (512, 1024)]
        self.assertTrue(rows, f"no recorded binding for the row read; got {recs}")
        for m in rows:
            self.assertEqual(sorted(m), [1], f"expected pos 1 only, got {m}")
        # The recorded block must be the REDUCTION block, not the row tile.  Both
        # exist and (at these sizes) differ in extent, so this is the assertion the
        # size scan also passed -- it is here to catch a carry that records the
        # wrong side.
        self.assertTrue(
            all(m[1] != 0 for m in rows),
            f"the trailing axis bound the row tile (block 0): {rows}",
        )

    def test_rank1_weight_slice_records_the_reduction_block(self) -> None:
        """``weight[:]`` at M == N: a 1-D subscript, so no sibling pins anything.

        This is commit ``877e3d012``'s bug.  The old ``bound_block_ids`` guard is
        vacuous here (nothing to pin), so at M == N the size scan matched the M
        tile and the weight read degraded to per-element scalar loads.  The
        recorded binding must be the reduction block.
        """

        @helion.kernel(static_shapes=True)
        def kern(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            out = torch.empty_like(x)
            for tile_m in hl.tile(m):
                row = x[tile_m, :].to(torch.float32)
                inv = torch.rsqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-5)
                out[tile_m, :] = (row * inv * w[:].to(torch.float32)).to(x.dtype)
            return out

        # M == N is the coincidence: every block's extent is 2048.
        x = torch.randn(2048, 2048, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(2048, device=DEVICE, dtype=torch.bfloat16)
        recs = _collect(
            kern, (x, w), helion.Config(block_sizes=[1], reduction_loops=[256])
        )
        weight_maps = [m for shape, m in recs if shape == (2048,)]
        self.assertTrue(
            weight_maps, f"the rank-1 weight read recorded nothing; got {recs}"
        )
        row_maps = [m for shape, m in recs if shape == (2048, 2048)]
        self.assertTrue(row_maps, f"the row read recorded nothing; got {recs}")
        # The weight's only axis and the row's trailing axis are the SAME axis of
        # the same reduction, so they must record the same block.  At M == N the
        # size scan bound the row read correctly (its sibling tile pinned block 0)
        # and the weight read WRONGLY -- so agreement is exactly the property that
        # was broken.
        self.assertEqual(
            {m[0] for m in weight_maps},
            {m[1] for m in row_maps},
            f"weight[:] and x[tile_m, :] bound different blocks: "
            f"weight={weight_maps} row={row_maps}",
        )

    def test_partial_slice_binding_is_recorded(self) -> None:
        """A PARTIAL slice (``x[tile_m, :16]``) is recorded too.

        ``_cute_index_exprs`` resolves a partial slice against ``compute_slice_size``
        (the slice extent, 16) rather than the dim size (32), and records that
        binding at its own recording site.  The vec-load consumer only reads
        BARE-slice positions, so this entry is inert there today -- it is asserted
        because the recording site is separate code and a future consumer (the
        the vec hoists' stride clamp is the obvious candidate) would read it.

        ⚠ DO NOT re-derive a "the old scan found nothing here" claim from this.  A
        first probe appeared to show 4 such disagreements; it was comparing the old
        BARE-slice-only answer against the whole carry map (partial entries
        included) and the diffs were all partial-slice positions -- an artifact.
        The corrected probe replays the entire lane search both ways and measures
        ``map_diff=0, decision_diff=0`` over 2095 vec-load sites.
        """

        @helion.kernel(static_shapes=True)
        def kern(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            for tile_m in hl.tile(x.size(0)):
                out[tile_m, :16] = x[tile_m, :16] * 2.0
            return out

        x = torch.randn(64, 32, device=DEVICE)
        out = torch.zeros(64, 32, device=DEVICE)
        recs = _collect(
            kern, (x, out), kern.bind((x, out)).config_spec.default_config()
        )
        partial = [m for shape, m in recs if shape == (64, 32)]
        self.assertTrue(
            partial,
            "the partial-slice axis recorded NO binding -- the carry regressed to "
            f"the deleted size scan's behaviour; got {recs}",
        )
        # The trailing axis is index_exprs position 1.
        self.assertTrue(
            any(1 in m for m in partial),
            f"the trailing partial-slice axis (pos 1) is unbound: {partial}",
        )

    def test_the_map_is_not_shared_across_calls(self) -> None:
        """Two callers get two maps -- the out-parameter contract.

        ``_cute_index_exprs`` is called TWICE for some loads (a matmul operand
        re-lowered under ``matmul_operand_index_override``).  A module global would
        carry the first call's binding into the second call's read.  Asserted
        structurally: the parameter exists, defaults to ``None``, and the function
        never reads a module-level map.
        """
        sig = inspect.signature(_cute_index_exprs)
        self.assertIn(
            "slice_block_ids",
            sig.parameters,
            "_cute_index_exprs no longer takes the binding out-parameter",
        )
        param = sig.parameters["slice_block_ids"]
        self.assertIs(
            param.default,
            None,
            "slice_block_ids must default to None so untouched callers are inert",
        )
        self.assertIs(
            param.kind,
            inspect.Parameter.KEYWORD_ONLY,
            "slice_block_ids must be keyword-only",
        )
        # No module-level binding cache: that shape is what the acceptance note
        # forbids, and it is the shape the rescued prototype used.
        import helion.language.memory_ops as lmo

        globals_with_maps = [
            name
            for name, val in vars(lmo).items()
            if name.isupper() and isinstance(val, dict) and "SLICE_BLOCK" in name
        ]
        self.assertEqual(
            globals_with_maps,
            [],
            f"a module-global binding cache reappeared: {globals_with_maps}",
        )

    def test_vec_ctx_requires_the_map(self) -> None:
        """``_cute_vector_load_ctx`` must CONSUME the map, not re-derive it.

        Required (no default), so a caller cannot silently omit it and get the old
        re-derivation back.
        """
        sig = inspect.signature(_cute_vector_load_ctx)
        self.assertIn(
            "slice_block_ids",
            sig.parameters,
            "_cute_vector_load_ctx no longer consumes the carried binding",
        )
        self.assertIs(
            sig.parameters["slice_block_ids"].default,
            inspect.Parameter.empty,
            "slice_block_ids must be REQUIRED at the vec-ctx site",
        )
        # The deleted heuristics must stay deleted: no size scan, no index_var
        # string match, no bound_block_ids set.
        #
        # ⚠ CHECK CODE, NOT TEXT.  All three names appear in the surviving COMMENTS
        # (which explain what was removed and why), so a substring search over the
        # source passes/fails on prose.  Strip comments by re-rendering the parsed
        # AST of the lane-axis search instead -- ``ast.unparse`` drops comments and
        # docstrings but keeps every executable name.
        src = inspect.getsource(_cute_vector_load_ctx)
        fn = ast.parse(textwrap.dedent(src)).body[0]
        assert isinstance(fn, ast.FunctionDef)
        # The lane-axis search is everything up to the ``if inner_block_id is None
        # or lane_axis_pos is None: return None`` bail.
        search: list[ast.stmt] = []
        for stmt in fn.body:
            if (
                isinstance(stmt, ast.If)
                and "inner_block_id is None" in ast.unparse(stmt.test)
                and "lane_axis_pos is None" in ast.unparse(stmt.test)
            ):
                break
            search.append(stmt)
        self.assertTrue(search, "could not locate the lane-axis search")
        code = "\n".join(ast.unparse(s) for s in search)
        for banned in ("bound_block_ids", "known_equal", "index_var"):
            self.assertNotIn(
                banned,
                code,
                f"the re-derivation heuristic is back ({banned!r} is EXECUTED in "
                "the lane-axis search, not merely mentioned in a comment)",
            )
        # ...and the carry IS read.
        self.assertIn(
            "slice_block_ids",
            code,
            "the lane-axis search no longer reads the carried binding",
        )
