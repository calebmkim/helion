"""Split a reduction loop into a MASK-FREE BULK and a masked ragged TAIL.

WHAT THIS FIXES.  ``NDTileStrategy.load_mask_var`` elides a tile's per-element bounds
mask from *reads* when it can prove the index in range for every launched thread.  That
proof (``_read_index_always_in_range``) is a conjunction:

    (1)  numel % B == 0        -- a property of the loop BOUNDS
    (2)  SPAN == B             -- a property of the LAUNCH

and it is all-or-nothing over the whole loop.  When (2) holds but (1) does not -- i.e. the
extent is simply not a multiple of the tile -- the mask is elided *nowhere*, even though
it is vacuous in every iteration except the last.  The kernel then pays, on every element
of every iteration, exactly the tax whose removal is what made the divisible shapes fast.

MEASURED, on ``cross_entropy_online`` at ``32768 x 12000`` (bf16, B200, vs quack):
``N = 12000 = 2^5 * 375`` has no power-of-two divisor at or above 32, so **no config can
make (1) true** -- the cell was stuck at 0.83 while every neighbouring power-of-two shape
was at 0.96-1.37.  Peeling takes it to **0.9997** at the frozen geometry and **1.106** at
``bi=1024 / nt=64``, with ``relerr == 0`` and a null arm at exactly 0.0000
(``_redfix3/repro/e1_peel_probe.py``).

THE REWRITE.  With ``E = floor(numel / B) * B``::

    for t in range(0, numel, B):   <masked body>          # before

    for t in range(0, E, B):       <body, selects folded>  # after: numel//B iters
    for t in range(E, numel, B):   <masked body>           # after: exactly 1 iter

⭐ THE INVARIANT, STATED FOR EVERY CALLER
=========================================
Let ``B`` be the tile, ``numel`` the axis extent, and ``E = floor(numel/B) * B``.  This
pass rewrites a loop only when :meth:`TileStrategy.ragged_peel_plan` returns a plan, which
requires ``SPAN == B``, ``numel % B != 0``, ``E > 0``, and a static ``numel``.  Then:

  **(P1) COVERAGE -- and it is not a permutation.**  ``list(range(0, E, B)) +
  list(range(E, numel, B)) == list(range(0, numel, B))``, elementwise and in order,
  because ``B | E`` and ``0 < numel - E < B``.  The two loops' bodies are the same
  statements over the same induction variable, so **every element is visited exactly once
  by exactly the same thread as before, in the same order.**  There is no re-partition and
  no reordering to certify -- only the loop bounds are split.  :func:`_assert_peel_sound`
  re-derives this at every application and raises rather than emit an unproved split.

  **(P2) THE PREDICATE IS IDENTICALLY TRUE IN THE BULK.**  For any ``t`` the bulk loop
  visits, ``t + B <= E <= numel``.  ``SPAN == B`` bounds a thread's displacement ``d`` by
  ``B - 1``.  So ``index = t + d <= t + B - 1 < numel``, i.e. ``mask == True`` for every
  thread, every element, every bulk iteration.  This is the *same* arithmetic
  ``_read_index_always_in_range`` uses; the split supplies ``max(t) == E - B`` where
  divisibility used to.

  **(P3) SCOPE -- ``IfExp`` ONLY, NEVER ``If``.**  Given (P2), folding *any* consumer of
  the mask would be sound.  This pass nevertheless folds only ``X if mask else Y``
  (:class:`ast.IfExp`) and leaves ``if mask: ...`` (:class:`ast.If`) alone, so a store's
  statement-level predication is preserved **by construction** rather than by an argument
  that has to be right.  That is deliberate and it is free: MEASURED, folding the mask's
  *definition* instead (which de-predicates stores too, and is the ``734eea2d9`` /
  E070-E071 failure mode) buys **nothing extra** -- 0.9997 vs 0.9998 at ``bi=512``,
  1.1057 vs 1.1059 at ``bi=1024``, both inside a 0.0000 noise floor.  So the pass takes
  the free safety.

  **(P4) LOOP-CARRIED STATE IS PRESERVED.**  The bulk runs before the tail and they share
  the enclosing scope, so a loop-carried accumulator (an online ``(m, s)`` recurrence, a
  running sum) flows from the last bulk iteration into the tail unchanged.  Splitting a
  loop at an iteration boundary is sound for *any* carried dependence, because the
  iteration order is untouched (P1).

  **(P5) A SOFTWARE-PIPELINED PREFETCH STILL LANDS.**  This pass runs *after*
  ``pipeline_inner_loads``, whose in-body prefetch is expressed *relative* to the
  induction variable (``t + STEP``), so the last bulk iteration prefetches exactly the
  tail's tile and the tail consumes it.  The prefetch's own address guard is a comparison
  against ``lane_base``, not against the peeled mask var, so it is **not** folded -- which
  matters, because that guard is the one keeping the last bulk iteration's prefetch of the
  *ragged* tile in bounds.

DECLINES (fail closed).  No plan, no rewrite:
  * a data-dependent or jagged extent -- ``_static_read_numel`` screens ``.size`` before
    touching ``BlockSizeInfo.numel``, which *asserts* on an ``AutoSize``;
  * ``numel % B == 0`` -- ``load_mask_var`` already elides, so a split would be dead code
    and would perturb the emitted source of every already-fast shape;
  * ``B > numel`` (``E == 0``) -- the only iteration *is* the ragged one;
  * ``SPAN != B`` -- surplus threads form indices past their own tile, which no rewrite of
    the loop bounds can fix.  This is the conjunct ``734eea2d9`` did not have;
  * the loop is not found, or folding removed no select -- then the split would be pure
    code growth and the pass leaves the loop alone.
"""

from __future__ import annotations

import ast
import dataclasses
import os
from typing import TYPE_CHECKING
from typing import cast

from ..ast_extension import ExtendedAST
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ._ast_pass_utils import ext_deepcopy

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclasses.dataclass(frozen=True)
class RaggedPeelPlan:
    """One peelable reduction loop, as proved by the tile strategy.

    ``offset_var`` names the induction variable so the pass never has to guess which
    ``for`` belongs to which axis; the strategy that emitted the loop supplies it.
    Frozen, because a plan is a *proof result*: nothing downstream may adjust the split
    point without re-running ``ragged_peel_plan``'s arithmetic.
    """

    offset_var: str
    mask_var: str
    bulk_end: int
    block_size: int
    numel: int


def _assert_peel_sound(plan: RaggedPeelPlan) -> None:
    """(P1), re-derived at every application rather than assumed."""
    numel, bs, end = plan.numel, plan.block_size, plan.bulk_end
    assert bs > 0 and numel > 0, (bs, numel)
    assert end % bs == 0, (end, bs)  # the bulk loop tiles exactly
    assert 0 < end < numel, (end, numel)  # non-empty bulk, non-empty tail
    assert numel - end < bs, (numel, end, bs)  # the tail is one iteration
    assert list(range(0, end, bs)) + list(range(end, numel, bs)) == list(
        range(0, numel, bs)
    )


class _FoldTrueSelects(ast.NodeTransformer):
    """Fold ``X if <mask> else Y`` to ``X`` where ``<mask>`` is known ``True``.

    Handles the two test shapes the emitters produce: a bare ``ast.Name`` and an
    ``and``-chain that includes it (``_cute_combined_mask`` conjoins one term per axis).
    In the chain case only *this* axis's term is removed; the others survive, because only
    this axis's bound was proved.

    ⚠ ``visit_If`` is deliberately NOT defined.  A statement-level ``if mask:`` is a
    store's predication and this pass must not touch it -- see (P3).
    """

    def __init__(self, mask_var: str) -> None:
        self.mask_var = mask_var
        self.folded = 0

    def _is_mask(self, node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id == self.mask_var

    def visit_IfExp(self, node: ast.IfExp) -> ast.expr:
        node = self.generic_visit(node)  # type: ignore[assignment]
        test = node.test
        if self._is_mask(test):
            self.folded += 1
            return node.body
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            kept = [v for v in test.values if not self._is_mask(v)]
            if len(kept) == len(test.values):
                return node
            self.folded += 1
            if not kept:
                return node.body
            node.test = (
                kept[0]
                if len(kept) == 1
                else create(ast.BoolOp, op=ast.And(), values=kept)
            )
        return node


def _range_call(begin: str, end: str, step_src: ast.expr) -> ast.expr:
    """``range(cutlass.Int32(begin), cutlass.Int32(end), <step>)``.

    The step is copied from the loop this pass is splitting rather than re-derived, so a
    block-size constexpr name (``_BLOCK_SIZE_1``) or a cast wrapper is carried over
    verbatim.  Only the bounds change; that is the whole rewrite.
    """
    return expr_from_string(
        f"range(cutlass.Int32({begin}), cutlass.Int32({end}), {{step}})",
        step=cast("ast.expr", ext_deepcopy(step_src)),
    )


def _rebuild_for(
    template: ast.For, iter_node: ast.expr, body: list[ast.stmt]
) -> ast.For:
    """A ``for`` with ``template``'s target but new bounds and body.

    Rebuilt through ``ExtendedAST.copy`` when the template has the mixin, so the emitted
    loop keeps the source location and ``_loop_type`` tag the original carried -- the
    printer reads both, and a bare ``create(ast.For, ...)`` would drop them.
    """
    target = cast("ast.expr", ext_deepcopy(template.target))
    if isinstance(template, ExtendedAST):
        return cast(
            "ast.For",
            template.copy(target=target, iter=iter_node, body=body, orelse=[]),
        )
    return create(
        ast.For,
        target=target,
        iter=iter_node,
        body=body,
        orelse=[],
        type_comment=None,
    )


def _loop_step_src(node: ast.For) -> ast.expr | None:
    """The third argument of the loop's ``range(...)`` call, if it has one."""
    call = node.iter
    if not isinstance(call, ast.Call):
        return None
    if not (isinstance(call.func, ast.Name) and call.func.id == "range"):
        return None
    if len(call.args) != 3:
        return None
    return call.args[2]


class _PeelLoops(ast.NodeTransformer):
    def __init__(self, plans: dict[str, RaggedPeelPlan]) -> None:
        self.plans = plans
        self.peeled: list[str] = []

    def visit_For(self, node: ast.For) -> ast.AST | list[ast.AST]:
        node = self.generic_visit(node)  # type: ignore[assignment]
        target = node.target
        if not isinstance(target, ast.Name):
            return node
        plan = self.plans.get(target.id)
        if plan is None:
            return node
        step_src = _loop_step_src(node)
        if step_src is None:
            return node
        _assert_peel_sound(plan)

        # THE BULK.  An ``ExtendedAST``-safe deep copy, so the tail keeps its (masked)
        # body untouched and both copies keep their source locations / loop-type tags.
        bulk_body = cast("list[ast.stmt]", ext_deepcopy(node.body))
        folder = _FoldTrueSelects(plan.mask_var)
        bulk_body = [cast("ast.stmt", folder.visit(stmt)) for stmt in bulk_body]
        if not folder.folded:
            # Nothing to save: emitting two loops would be pure code growth.
            return node
        bulk = _rebuild_for(
            node, _range_call("0", str(plan.bulk_end), step_src), bulk_body
        )
        # THE TAIL.  Original body, original mask, one iteration.
        tail = _rebuild_for(
            node,
            _range_call(str(plan.bulk_end), str(plan.numel), step_src),
            node.body,
        )
        self.peeled.append(target.id)
        # The two loops need no explanatory comment in the emitted source: their literal
        # bounds ``(0, bulk_end)`` and ``(bulk_end, numel)`` state the split exactly, and
        # this AST has no comment statement -- ``statement_from_string("# ...")`` parses to
        # zero statements and raises.
        return [bulk, tail]


def peel_ragged_tiles(
    body: list[ast.stmt], plans: Sequence[RaggedPeelPlan]
) -> list[ast.stmt]:
    """Apply :class:`RaggedPeelPlan` to ``body``.  A no-op when ``plans`` is empty.

    ``HELION_NO_PEEL_RAGGED_TILE=1`` disables the pass.  That switch exists for A/B
    *attribution* only -- it proves involvement, never absence (``_redfix3/01_HOWTO.md``
    Part 3) -- and is not a tuning knob: the transform is strictly less work per element
    on ``numel//B`` of ``ceil(numel/B)`` iterations, so there is no shape whose winner
    flips and therefore nothing for the autotuner to choose.
    """
    if not plans or os.environ.get("HELION_NO_PEEL_RAGGED_TILE") == "1":
        return body
    by_offset = {p.offset_var: p for p in plans}
    transformer = _PeelLoops(by_offset)
    out = [transformer.visit(stmt) for stmt in body]
    flat: list[ast.stmt] = []
    for item in out:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat
