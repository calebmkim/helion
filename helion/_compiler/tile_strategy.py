from __future__ import annotations

import ast
import collections
import dataclasses
import functools
import itertools
import logging
import math
import operator
import os
import re
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Mapping
from typing import NamedTuple
from typing import TypeVar
from typing import cast
import weakref

import sympy
import torch

from .. import exc
from .._compat import shape_env_size_hint
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from .compile_environment import CompileEnvironment
from .compile_environment import _has_unbacked
from .compile_environment import _to_sympy

# The capability-③ staging block below reads these four.  ⚠ Top-level rather than
# per-method, and it is NOT a new cycle: ``cute/tv_layout.py`` imports nothing from this
# module, and ``reduction_strategy.py`` -- which the staging block moved OUT of -- already
# imported all four at module level, so this is the same edge that already existed, now
# spelled from the module that owns the code.
from .cute.ragged_tail import assert_vec_divides_extent
from .cute.ragged_tail import ragged_tile_admissible
from .cute.ragged_tail import rounded_extent
from .cute.tv_layout import ROW_RESIDENCY_GMEM
from .cute.tv_layout import ROW_RESIDENCY_SMEM
from .cute.tv_layout import TVParticipants
from .device_function import DeviceFunction
from .host_function import HostFunction
from .program_id import FlatProgramIDs
from .program_id import ForEachProgramID
from .program_id import L2GroupingProgramIDs
from .program_id import PersistentBlockedProgramIDs
from .program_id import PersistentInterleavedProgramIDs
from .program_id import PIDInfo
from .program_id import ProgramIDs
from .program_id import Tcgen05PersistentProgramIDs
from .program_id import XYZProgramIDs
from .source_location import SyntheticLocation

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .cute.tv_layout import ChunkTVPlan
    from .inductor_lowering import CodegenState

    _T = TypeVar("_T")
    SymIntLike = torch.SymInt | int
    ShapeLike = Sequence[SymIntLike]


# ⚠ THE MOVED CAPABILITY-③ BLOCK BELOW LOGS EVERY DECLINE AT DEBUG, and it did so on
# ``reduction_strategy``'s logger before the move.  Declaring one here keeps those
# lines alive; without it the moved code raises ``NameError`` on the first decline --
# i.e. exactly on the diagnostic path, which is the worst place to lose a symbol.
# ⚠ The logger NAME changes with the module, so a reader who filters
# ``HELION_LOGS=+helion._compiler.reduction_strategy`` will no longer see the staging
# declines; they are now under ``helion._compiler.tile_strategy``.
log = logging.getLogger(__name__)


class ThreadAxisTracker:
    """Tracks thread axis assignments for block dimensions during codegen."""

    __slots__ = ("sizes", "block_axes")

    def __init__(self) -> None:
        self.sizes: dict[int, int] = {}
        self.block_axes: dict[int, int] = {}

    def record(self, block_idx: int, axis: int, size: int) -> None:
        """Record a thread axis mapping for a single block dimension."""
        self.sizes[axis] = max(self.sizes.get(axis, 1), size)
        self.block_axes[block_idx] = axis

    def record_all(self, block_ids: list[int], axis: int, size: int) -> None:
        """Record the same thread axis mapping for all block dimensions."""
        self.sizes[axis] = size
        for block_id in block_ids:
            self.block_axes[block_id] = axis


def _lane_loop_iter(extent: int) -> ast.AST:
    # CuTe lane loops carry per-thread scalar state. Emitting them via
    # cutlass.range(_constexpr) miscompiles scalar matmul paths, so keep them
    # as ordinary Python loops.
    return expr_from_string(f"range({extent})")


def cute_ndtile_tv_enabled() -> bool:
    """Is the ``CuteNDTileStrategy`` TV-copy path switched on?

    ⭐ ONE FUNCTION, ONE ENV VAR, READ NOWHERE ELSE.  The gate exists because
    enabling the TV path is a REPLACEMENT of an established emission, not a
    widening: MEASURED, it displaces the ``cute.arch.load`` form pinned by 3 tests
    in ``test_cute_tile_loop_vec_hoist.py`` and changes 8 of the 40 frozen perf
    cells (all ``cross_entropy_online/*``, attributed by an A/B toggling only this
    gate).  Whether that trade is a WIN is a timing question, and this makes it a
    one-variable experiment instead of a default nobody chose.

    ⚠ Read at PLAN-CONSTRUCTION time only, so "off" means no plan exists and every
    downstream capability query answers False through the ordinary route.  A gate
    consulted at each emission site instead could leave a plan alive while its
    copies were declined -- a trip count derived from a width nothing uses, i.e.
    bug class 1 reintroduced by the off-switch itself.
    """
    return os.environ.get("HELION_CUTE_NDTILE_TV", "") == "1"


def _tiled_axis_block_id(idx: object) -> int | None:
    """The block id a subscript entry names, or ``None`` if it names no tile axis.

    ⭐ ONE FUNCTION FOR **BOTH** REPRESENTATIONS OF A SUBSCRIPT, and that is the
    whole reason it exists rather than being inlined at its two call sites.  The
    same logical subscript appears as:

    * ``torch.fx.Node`` entries in the DEVICE IR, whose ``meta["val"]`` holds the
      ``SymInt`` (what a plan-building walk over ``device_ir.graphs`` sees), and
    * plain ``SymInt`` entries at CODEGEN time (what a per-site eligibility gate
      sees).

    ⚠ MEASURED, AND IT IS EXACTLY THE FAILURE THIS FUNCTION PREVENTS: a device-IR
    walk written with a bare ``isinstance(idx, torch.SymInt)`` test matches nothing
    at all -- the entries print as ``[Node:sym_size_int, Node:block_size_1]`` -- so
    it returns an empty participant list for EVERY kernel.  From the caller that is
    indistinguishable from an honest "this shape is not eligible", i.e. a silently
    vacuous gate, which is the same class of defect as a fail-capability-less
    check.  Sharing the unwrapping means the plan-side and site-side gates cannot
    disagree about which axis a subscript entry names, whichever form they see it
    in -- and they MUST agree, or a plan is built at a width no access honours.
    """
    if isinstance(idx, torch.fx.Node):
        idx = idx.meta.get("val")
    if not isinstance(idx, torch.SymInt):
        return None
    return CompileEnvironment.current().get_block_id(idx)


def _create_lane_loop(lane_var: str, extent: int, body: list[ast.AST]) -> ast.For:
    loop = create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=_lane_loop_iter(extent),
        body=body,
        orelse=[],
        type_comment=None,
    )
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, lane_var)
    return loop


# Marker call emitted by reduction strategies when a reduction over a
# lane-distributed block is generated inside a single-pass lane loop.  The
# ``split_lane_loop_reductions`` post-pass recognizes these markers and
# rewrites the enclosing lane loop into a two-pass structure:
#
#   (phase 1) accumulate the per-lane reduction inputs across the lane loop,
#             then combine across the live thread axis (``threads_in_group``)
#             into the final scalar;
#   (finalize) define the reduced scalar between the two passes;
#   (phase 2) re-iterate the lanes to apply any lane-varying consumers (e.g.
#             the broadcast normalize / store) using the finalized scalar.
#
# The marker never reaches the emitted kernel — the post-pass strips every
# marker it processes.
_HELION_LANE_REDUCE_MARKER = "_helion_lane_reduce"


def _lane_reduce_marker_expr(
    input_name: str,
    reduction_type: str,
    identity_expr: str,
    threads_in_group: int,
    *,
    acc_dtype_str: str,
    group_pre: int = 1,
    group_span: int = 0,
    group_lane_expr: str = "",
    group_count: int = 1,
    partial_fold: bool = True,
) -> str:
    # ``acc_dtype_str`` is the CuTe dtype the reduction ACCUMULATES in (e.g.
    # ``cutlass.Float32``, ``cutlass.Int64``), carried EXPLICITLY rather than
    # re-derived from ``identity_expr``. See the ABI note in
    # ``cute/reduce_helpers.py``: inferring it is what let an int32 sum wrap.
    # It is passed as a string literal so the post-pass can read it back without
    # re-parsing the identity expression.
    #
    # ``group_*`` (optional) carry the parameters of a strided grouped
    # reduction. They are required when the reduction's live thread axis is
    # interleaved with an unrelated sibling thread axis. ``group_lane_expr`` is
    # base64-free but may contain commas/parens, so it is passed as a string
    # literal that the post-pass re-parses.
    #
    # When ``group_span <= 32`` the de-interleaving fits in a single warp and
    # the finalize uses ``_cute_grouped_reduce_warp``. When ``group_span > 32``
    # (and a multiple of 32) the reduction group is spread across warps, so the
    # finalize uses the cross-warp ``_cute_grouped_reduce_shared_two_stage``;
    # ``group_count`` (the number of independent groups in the CTA) is needed
    # only by that two-stage helper.
    #
    # ⭐⭐ ``partial_fold`` IS THE OBLIGATION THE MARKER OWES, CARRIED AS DATA (task 2
    # step 1).  ⛔ THE PROBLEM IT SOLVES: read as IR, ``MARKER(v, 'max')`` is a valid
    # ONE-ELEMENT reduction, so ``mx = v`` is a *faithful* lowering of it -- which is why
    # the safety net :func:`restore_unprocessed_lane_reduce_markers` could discharge the
    # obligation BY DELETING THE FOLD and produce a compilable, plausible, WRONG kernel
    # (bug class 8 P1: ``exp(v - v) == 1.0``, softmax rows summing to 128.0 instead of
    # 1.0; and relerr 7.685 on ``matmul_layernorm`` at N=512).
    #
    # ⇒ every marker this function emits owes TWO combines -- a serial LANE FOLD across
    # the ``for lane in range(extent)`` iterations, and a CROSS-THREAD combine
    # (``warp_reduction_*``) -- so a marker that reaches the emitted kernel with only its
    # per-lane input substituted is UNFINISHED, not merely unoptimised.  The field says so
    # explicitly, so the safety net can RAISE instead of silently reverting: the fact stops
    # being reconstructible-in-principle and becomes stated.
    #
    # ⚠ IT DEFAULTS TO ``True`` ON PURPOSE.  All four emission sites
    # (``reduction_strategy.py`` :1287, :1321, :3158, :4903 -- ⚠ the four line numbers this
    # comment carried before were ALL STALE by ~7 lines; re-verified 2026-08-01, and prefer
    # ``grep -n '_lane_reduce_marker_expr('`` to trusting them) emit a partial fold -- there
    # is currently no site that emits a complete one -- so the default is the truth for
    # every caller, and a NEW site would have to opt OUT deliberately rather than inherit
    # the silent-fallback hazard by forgetting an argument.  That polarity is the whole
    # point: the dangerous value is the one you must ask for.
    return (
        f"{_HELION_LANE_REDUCE_MARKER}({input_name}, {reduction_type!r}, "
        f"{identity_expr}, {threads_in_group}, {group_pre}, {group_span}, "
        f"{group_lane_expr!r}, {group_count}, {acc_dtype_str!r}, {partial_fold!r})"
    )


@dataclasses.dataclass
class _LaneReduceMarker:
    result_var: str
    input_name: str
    reduction_type: str
    identity_expr: str
    threads_in_group: int
    # The original RHS expression with the marker call replaced by the string
    # ``{finalized}``; ``finalize_expr(x)`` substitutes ``x`` to re-apply any
    # surrounding dtype cast / reshape to the finalized reduced scalar.
    wrap_template: str
    # Optional strided grouped reduction (the reduction's live thread axis
    # shares a warp / CTA with an unrelated sibling axis). When ``group_span``
    # > 0 the finalize uses a grouped reduction keyed on ``group_lane_expr``
    # instead of a plain consecutive-lane ``cute.arch.warp_reduction_*``:
    # ``_cute_grouped_reduce_warp`` when ``group_span <= 32`` (single warp),
    # ``_cute_grouped_reduce_shared_two_stage`` when ``group_span`` is a
    # multiple of 32 greater than 32 (cross-warp, ``group_count`` groups).
    group_pre: int = 1
    group_span: int = 0
    group_lane_expr: str = ""
    group_count: int = 1
    # The CuTe dtype constructor the reduction accumulates in (e.g.
    # ``cutlass.Int64``).  Explicit, never inferred from ``identity_expr``.
    acc_dtype_str: str = ""
    # ⭐ THE OBLIGATION.  True == this marker still owes BOTH a serial lane fold and a
    # cross-thread combine, so substituting its raw per-lane input is a WRONG ANSWER and
    # not a conservative fallback.  See :func:`_lane_reduce_marker_expr` for the
    # measurement, and :func:`restore_unprocessed_lane_reduce_markers` for what reads it.
    partial_fold: bool = True

    def finalize_expr(self, reduced: str) -> str:
        return self.wrap_template.replace("__HELION_FINALIZED__", f"({reduced})")


def _find_lane_reduce_call(node: ast.AST) -> ast.Call | None:
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == _HELION_LANE_REDUCE_MARKER
        ):
            return sub
    return None


class _ReplaceLaneReduceCall(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == _HELION_LANE_REDUCE_MARKER
        ):
            return ast.copy_location(
                create(ast.Name, id="__HELION_FINALIZED__", ctx=ast.Load()), node
            )
        return node


def _is_lane_reduce_marker_assign(stmt: ast.AST) -> _LaneReduceMarker | None:
    """If ``stmt`` assigns an expression containing a single
    ``_helion_lane_reduce(IN, TYPE, ID, T)`` marker call, return a
    :class:`_LaneReduceMarker`; otherwise return ``None``.

    The marker may be nested inside a surrounding cast/reshape (e.g.
    ``R = cutlass.Float32(_helion_lane_reduce(...))``); the wrapping is
    captured so it can be re-applied to the finalized reduced scalar.
    """
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        return None
    call = _find_lane_reduce_call(stmt.value)
    # ⚠⚠ THE ARITY CHECK IS AN EQUALITY, AND THAT MAKES IT A SILENT-FAILURE SITE.
    # ``len(call.args) != N -> return None`` means "this is not a marker", so a marker
    # emitted with the WRONG NUMBER OF ARGUMENTS is not rejected -- it is not SEEN.  It
    # then flows past both the split pass and the safety net (which uses this same
    # recogniser) straight into the emitted kernel as a call to an undefined
    # ``_helion_lane_reduce``, i.e. a NameError at DSL compile time rather than a helion
    # diagnostic.  ⇒ when task 2 added ``partial_fold`` the arity moved 9 -> 10, and the
    # OLD spelling must stay recognised: a 9-arg marker is a marker from a site that has
    # not been updated, and reading it as "not a marker" is the worst of the three
    # available behaviours.  It defaults to ``partial_fold=True`` (the safe polarity --
    # see :func:`_lane_reduce_marker_expr`), so an un-updated site fails LOUD rather than
    # silently reverting.
    if call is None or len(call.args) not in (9, 10):
        return None
    (
        input_node,
        type_node,
        identity_node,
        threads_node,
        group_pre_node,
        group_span_node,
        group_lane_node,
        group_count_node,
        acc_dtype_node,
    ) = call.args[:9]
    partial_fold = (
        bool(ast.literal_eval(call.args[9])) if len(call.args) == 10 else True
    )
    input_name = ast.unparse(input_node)
    reduction_type = ast.literal_eval(type_node)
    identity_expr = ast.unparse(identity_node)
    threads_in_group = int(ast.literal_eval(threads_node))
    group_pre = int(ast.literal_eval(group_pre_node))
    group_span = int(ast.literal_eval(group_span_node))
    group_lane_expr = ast.literal_eval(group_lane_node)
    group_count = int(ast.literal_eval(group_count_node))
    acc_dtype_str = ast.literal_eval(acc_dtype_node)
    # Build the wrap template by replacing the marker call with a sentinel.
    wrapped = _ReplaceLaneReduceCall().visit(ast.parse(ast.unparse(stmt.value)).body[0])
    assert isinstance(wrapped, ast.Expr)
    wrap_template = ast.unparse(wrapped.value)
    return _LaneReduceMarker(
        result_var=target.id,
        input_name=input_name,
        reduction_type=reduction_type,
        identity_expr=identity_expr,
        threads_in_group=threads_in_group,
        wrap_template=wrap_template,
        group_pre=group_pre,
        group_span=group_span,
        group_lane_expr=group_lane_expr,
        group_count=group_count,
        acc_dtype_str=acc_dtype_str,
        partial_fold=partial_fold,
    )


def _combine_expr(reduction_type: str, acc: str, val: str) -> str:
    if reduction_type == "sum":
        return f"({acc}) + ({val})"
    if reduction_type == "prod":
        return f"({acc}) * ({val})"
    if reduction_type == "max":
        return f"({acc}) if ({acc}) > ({val}) else ({val})"
    if reduction_type == "min":
        return f"({acc}) if ({acc}) < ({val}) else ({val})"
    raise NotImplementedError(f"lane reduce combine {reduction_type!r}")


def _dtype_ctor_from_identity(identity_expr: str) -> str | None:
    """Extract the dtype constructor (e.g. ``cutlass.Float32``) from an identity
    expression like ``cutlass.Float32(0)``.

    Only a fallback: prefer :attr:`_LaneReduceMarker.acc_dtype_str`, which the
    emitter sets explicitly.  See :func:`_marker_acc_ctor`.
    """
    try:
        node = ast.parse(identity_expr, mode="eval").body
    except SyntaxError:
        return None
    if isinstance(node, ast.Call):
        return ast.unparse(node.func)
    return None


def _marker_acc_ctor(m: _LaneReduceMarker) -> str | None:
    """The CuTe dtype constructor a marker's accumulator uses.

    Uses the marker's EXPLICIT ``acc_dtype_str`` when present, falling back to
    parsing the identity only for markers built by paths that predate the
    explicit field.
    """
    return m.acc_dtype_str or _dtype_ctor_from_identity(m.identity_expr)


def _grouped_warp_reduce_expr(
    reduction_type: str,
    acc: str,
    identity_expr: str,
    lane_expr: str,
    *,
    acc_dtype_str: str,
    pre: int,
    group_span: int,
) -> str:
    """Strided grouped warp reduction over a single warp.

    Reduces ``acc`` across the ``group_span`` lanes that share the same
    ``lane % pre`` within each ``group_span``-lane block, so an interleaved
    sibling thread axis (occupying the low ``pre`` strides) stays distinct.
    """
    return (
        "_cute_grouped_reduce_warp("
        f"{acc}, {reduction_type!r}, {identity_expr}, {lane_expr}, "
        f"acc_dtype={acc_dtype_str}, pre={pre}, group_span={group_span})"
    )


def _grouped_two_stage_reduce_stmts(
    result_acc: str,
    reduction_type: str,
    acc: str,
    identity_expr: str,
    lane_expr: str,
    *,
    acc_dtype_str: str,
    pre: int,
    group_span: int,
    group_count: int,
) -> list[ast.AST]:
    """Cross-warp grouped reduction over ``group_span`` (> 32) lanes.

    Mirrors ``BlockReductionStrategy._strided_thread_reduction_expr``'s
    ``group_span > 32`` branch: the reduce group is spread across warps, so a
    single ``cute.arch.warp_reduction_*`` cannot fold it. The two-stage shared
    helper reduces each warp, stages the per-warp partials in shared memory, and
    combines them, keeping the ``pre`` interleaved sibling lanes distinct.

    Unlike that reference emitter this path has no shared-memory-budget fallback,
    but it does not need one: it only fires for a block-resident reduced tile
    whose live thread count is bounded by ``MAX_THREADS_PER_BLOCK`` (<= 1024, so
    <= 32 staged per-warp partials), which cannot overflow the reduction SMEM
    budget.

    Returns the (lane-index setup + reduce) statements that define
    ``result_acc`` (used in place of the single-shuffle ``reduced`` scalar).
    """
    lane_var = f"{result_acc}_lane"
    lane_in_group_var = f"{result_acc}_lane_in_group"
    lane_mod_pre_var = f"{result_acc}_lane_mod_pre"
    return [
        statement_from_string(f"{lane_var} = {lane_expr}"),
        statement_from_string(f"{lane_in_group_var} = ({lane_var}) % {group_span}"),
        statement_from_string(f"{lane_mod_pre_var} = ({lane_in_group_var}) % {pre}"),
        statement_from_string(
            f"{result_acc} = _cute_grouped_reduce_shared_two_stage("
            f"{acc}, {reduction_type!r}, {identity_expr}, "
            f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
            f"acc_dtype={acc_dtype_str}, "
            f"pre={pre}, group_span={group_span}, group_count={group_count})"
        ),
    ]


def _finalize_lane_reduce_marker(m: _LaneReduceMarker, acc_var: str) -> list[ast.AST]:
    """Combine a marker's per-lane accumulator ``acc_var`` across the live
    thread axis and assign the finalized scalar to ``m.result_var``.

    Picks the cross-thread combine that matches the marker's thread layout:

    * a cross-warp two-stage shared reduction when the reduce group spans more
      than one warp (``group_span`` a multiple of 32 > 32);
    * a single-warp strided grouped reduction when the reduce axis shares a warp
      with an unrelated sibling axis (``1 < group_span <= 32`` with ``pre`` > 1);
    * a plain consecutive-lane warp reduction otherwise;
    * the accumulator unchanged when there is no live thread axis to combine.
    """
    acc_ctor = _marker_acc_ctor(m) or "cutlass.Float32"
    if m.group_span > 32 and m.group_span % 32 == 0 and m.group_lane_expr:
        # Cross-warp: the reduce group is spread across warps, so fold the
        # per-lane accumulator with the two-stage shared-memory reduction.
        stmts = _grouped_two_stage_reduce_stmts(
            f"{acc_var}_reduced",
            m.reduction_type,
            acc_var,
            m.identity_expr,
            m.group_lane_expr,
            acc_dtype_str=acc_ctor,
            pre=m.group_pre,
            group_span=m.group_span,
            group_count=m.group_count,
        )
        stmts.append(
            statement_from_string(
                f"{m.result_var} = {m.finalize_expr(f'{acc_var}_reduced')}"
            )
        )
        return stmts
    if m.group_span > 1 and m.group_pre > 1 and m.group_lane_expr:
        # Strided grouped reduction: the reduce axis shares a warp with an
        # unrelated sibling axis, so combine only the lanes that share the
        # current lane's sibling coordinate.
        reduced = _grouped_warp_reduce_expr(
            m.reduction_type,
            acc_var,
            m.identity_expr,
            m.group_lane_expr,
            acc_dtype_str=acc_ctor,
            pre=m.group_pre,
            group_span=m.group_span,
        )
    elif m.threads_in_group > 1:
        reduced = _warp_reduce_expr(m.reduction_type, acc_var, m.threads_in_group)
    else:
        reduced = acc_var
    return [statement_from_string(f"{m.result_var} = {m.finalize_expr(reduced)}")]


def _warp_reduce_expr(reduction_type: str, acc: str, threads_in_group: int) -> str:
    tg = f", threads_in_group={threads_in_group}"
    if reduction_type == "sum":
        return f"cute.arch.warp_reduction_sum({acc}{tg})"
    if reduction_type == "max":
        return f"cute.arch.warp_reduction_max({acc}{tg})"
    if reduction_type == "min":
        return f"cute.arch.warp_reduction(({acc}), lambda a, b: a if a < b else b{tg})"
    if reduction_type == "prod":
        return f"cute.arch.warp_reduction(({acc}), lambda a, b: (a * b){tg})"
    raise NotImplementedError(f"lane warp reduce {reduction_type!r}")


def _backward_slice(body: list[ast.AST], roots: set[str]) -> tuple[list[int], set[str]]:
    """Return the indices of the statements in ``body`` that (transitively)
    produce any name in ``roots``, plus the set of all names those statements
    write.  Statements are scanned in reverse so a producer is included once a
    later consumer (already selected) reads its output.
    """
    from .ast_read_writes import ReadWrites

    needed = set(roots)
    selected: list[int] = []
    written: set[str] = set()
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        rw = ReadWrites.from_ast(stmt)
        writes = set(rw.writes)
        if writes & needed:
            selected.append(idx)
            written |= writes
            needed |= set(rw.reads)
    selected.reverse()
    return selected, written


def split_lane_loop_reductions(body: list[ast.AST]) -> list[ast.AST]:
    """Rewrite single-pass lane loops that contain ``_helion_lane_reduce``
    markers into the two-pass accumulate / finalize / consume structure.

    Operates bottom-up so nested lane loops are handled before their parents.
    Lane loops without markers are returned unchanged (their inner statements
    are still recursed into so nested markers are processed).

    ⚠ "NESTED MARKERS ARE PROCESSED" MEANS *NESTED LANE LOOPS*, NOT NESTED
    STATEMENTS -- AND THE DIFFERENCE IS A SILENT WRONG ANSWER.  A marker is only
    ever recognised as a DIRECT CHILD of a lane loop: :func:`_split_one_lane_loop`
    enumerates ``loop.body`` and calls :func:`_is_lane_reduce_marker_assign` on
    each element.  The recursion above descends into every statement-bearing
    field, so a marker inside a lane loop that is itself inside another lane loop
    is handled -- but a marker inside a *serial* ``for`` (or an ``if``) that is
    inside a lane loop is NOT: that inner ``For`` carries no
    ``HELION_LANE_LOOP_VAR_ATTR``, so it is returned untouched and the enclosing
    lane loop sees no markers among its own children.

    ⭐ THIS IS WHY THE TV PROTOCOL CANNOT YET BE GIVEN TO A REDUCTION THAT NEEDS
    THE MARKER.  The TV path emits its reduction inside
    ``for vi in cutlass.range_constexpr(vec)`` -- deliberately, so the fragment's
    every element is visited by loop structure and not only by the copy's width
    (``PORT_SPEC_layout.md`` §8b Rule 1).  A marker emitted there survives this
    pass and is then rewritten by
    :func:`restore_unprocessed_lane_reduce_markers` into its RAW PER-LANE INPUT,
    which DROPS the cross-lane accumulator -- the ``got[0]=1.0`` vs
    ``ref[0]=59.4`` failure mode the comment in :func:`_split_one_lane_loop` calls
    bug class 8 P1.  MEASURED, both directions, by
    ``_notes/tests/test_capability_matrix.py::test_nested_marker_is_invisible_to_the_lane_split``:
    the direct-child shape gets ``acc_res_lane_acc`` + ``warp_reduction_sum``, and
    the ``range_constexpr``-nested shape gets ``acc_res = cutlass.Float32(v_0)``
    with no fold at all.

    ``LoopedReductionStrategy`` is unaffected because it does not use the marker:
    its ``for roffset`` loop carries its own accumulator and its own vec-fold
    (which is exactly what :meth:`ReductionStrategy._lane_reduce_marker_unsupported`
    detects and declines on).  Any *loop-free* strategy given a TV plan would need
    this pass taught the nested shape FIRST.
    """
    new_body: list[ast.AST] = []
    for stmt in body:
        new_body.extend(_split_stmt_lane_reductions(stmt))
    return new_body


def _restore_stmt_lane_reduce_markers(stmt: ast.AST) -> ast.AST:
    """Recurse into statement-list fields and replace any surviving
    ``R = ..._helion_lane_reduce(IN, TYPE, ID, T)...`` assignment with
    ``R = ...IN...`` (the raw per-lane input).

    ⛔ A marker that declared ``partial_fold`` RAISES here instead: see
    :func:`restore_unprocessed_lane_reduce_markers`.
    """
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(stmt, field, [_restore_stmt_lane_reduce_markers(s) for s in old])
    if isinstance(stmt, ast.Assign):
        m = _is_lane_reduce_marker_assign(stmt)
        if m is not None:
            if m.partial_fold:
                # ⛔⛔ THE SILENT ARM, CLOSED (task 2 step 1).  Substituting the raw
                # per-lane input here DISCHARGES AN OBLIGATION BY DELETING IT: the marker
                # owes a serial lane fold AND a cross-thread combine, and ``R = IN`` has
                # neither.  It compiles, it looks plausible, and it is wrong -- MEASURED
                # as ``exp(v - v) == 1.0`` (softmax rows summing to 128.0 instead of 1.0),
                # ``got[0]=1.0`` vs ``ref[0]=59.4``, and relerr 7.685 on
                # ``matmul_layernorm`` at N=512.
                #
                # ⭐ WHY THIS IS A RAISE AND NOT A BETTER FALLBACK: there is no correct
                # fallback available HERE.  The two combines have to be emitted as loop
                # structure, and by this point the loop is built and the pass that could
                # have restructured it has already declined.  The honest options are "emit
                # a wrong kernel" or "refuse", so it refuses -- and it refuses with the
                # payload's own testimony rather than with an inference about the shape.
                #
                # ⚠ IT CLOSES SHAPES THE SPLIT PASS CANNOT SEE, which is the point of
                # putting it here.  ``split_lane_loop_reductions`` only recognises a
                # marker that is a DIRECT CHILD of a lane loop, so a marker inside a
                # ``range_constexpr`` vec loop (the TV path) or inside an ``if`` is
                # invisible to it and used to arrive here to be silently reverted.
                raise exc.BackendUnsupported(
                    "cute",
                    f"a lane reduction ({m.reduction_type!r} over {m.input_name!r}) "
                    f"reached the end of codegen still owing its lane fold and its "
                    f"cross-thread combine, in a context the lane-split pass could not "
                    f"restructure. Reverting it to the per-lane input would drop both "
                    f"combines and silently compute the wrong answer, so this refuses "
                    f"instead. (Marker declared partial_fold=True.)",
                )
            return statement_from_string(
                f"{m.result_var} = {m.finalize_expr(m.input_name)}"
            )
    return stmt


def restore_unprocessed_lane_reduce_markers(
    body: list[ast.AST],
) -> list[ast.AST]:
    """Replace any surviving ``R = ..._helion_lane_reduce(IN, TYPE, ID, T)...``
    assignment with ``R = ...IN...`` (the raw per-lane input).

    A safety net: ``split_lane_loop_reductions`` only rewrites markers it can
    place in a two-pass lane structure. A marker emitted in a context neither
    that pass nor ``interchange_lane_outside_serial_reductions`` handles would
    otherwise leak the ``_helion_lane_reduce`` call into the emitted kernel.

    ⛔⛔ THE ``partial_fold`` MARKER RAISES RATHER THAN REVERTING (task 2 step 1), AND
    THE OLD DOCSTRING'S CLAIM WAS THE BUG.  It used to read:

        "Reverting to the per-lane input keeps the kernel compilable (it falls back to
         the original single-pass per-lane reduction behavior)."

    ⚠ "Compilable" was true and "falls back to the original behavior" was FALSE.  There is
    no single-pass behaviour to fall back to: the marker exists precisely because the
    reduction owes a serial lane fold AND a cross-thread combine, and ``R = IN`` emits
    neither.  What the revert produced was a kernel that compiled and computed the wrong
    answer -- measured as softmax rows of 128.0 instead of 1.0, and relerr 7.685 on
    ``matmul_layernorm``.  The sentence made a wrong-answer path read as a conservative
    one, which is why it is quoted here rather than deleted.

    ⇒ a marker that DECLARED itself partial now raises ``BackendUnsupported``.  A marker
    that declares itself complete (nothing emits one today -- see
    :func:`_lane_reduce_marker_expr`) still reverts, because for such a marker the revert
    genuinely is the identity.

    ⚠ THE RAISE IS CONDITIONED ON THE PAYLOAD, NOT ON THE SHAPE, and that is what keeps it
    from repeating a known regression: an UNCONDITIONAL raise at a lane-split decline site
    broke all 8 attention examples, twice.  Here the marker itself testifies, so a context
    that legitimately reaches this function with a complete fold is unaffected.

    Recurses only into statement-bearing fields (``body``/``orelse``/
    ``finalbody``) instead of using ``ast.NodeTransformer``; markers are always
    statement-level assignments, so this avoids the transformer's in-place
    mutation of expression list fields, which fails when an AST node carries a
    ``torch.fx`` ``immutable_list`` (e.g. multi-output ``inline_asm_elementwise``).
    """
    return [_restore_stmt_lane_reduce_markers(stmt) for stmt in body]


def _split_stmt_lane_reductions(stmt: ast.AST) -> list[ast.AST]:
    # Recurse into any statement-list-bearing fields first so nested lane
    # loops are rewritten before the enclosing one.
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(stmt, field, split_lane_loop_reductions(old))
    lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
    if (
        lane_var is None
        or not isinstance(stmt, ast.For)
        or not isinstance(stmt.target, ast.Name)
        or stmt.target.id != lane_var
    ):
        return [stmt]
    return _split_one_lane_loop(stmt, lane_var)


def _constexpr_vec_loop(stmt: ast.AST) -> ast.For | None:
    """``stmt`` if it is a ``for vi in cutlass.range_constexpr(V):`` loop, else None.

    Recognised by its ITERATOR, not by its position or its variable name: the TV
    protocol holds this node by reference and both TV legs insert siblings around
    it, so anything positional would drift.  ``range_constexpr`` is what makes the
    loop a compile-time unroll over one fragment's elements
    (``PORT_SPEC_layout.md`` §8b Rule 2), which is exactly the property that makes
    the rewrite in :func:`_split_lane_loop_over_constexpr_vec` sound.
    """
    if not isinstance(stmt, ast.For) or not isinstance(stmt.iter, ast.Call):
        return None
    if getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None) is not None:
        return None
    func = stmt.iter.func
    if not isinstance(func, ast.Attribute) or func.attr != "range_constexpr":
        return None
    if len(stmt.iter.args) != 1:
        return None
    return stmt


def _tv_copy_dest_name(stmt: ast.AST) -> str | None:
    """The name a ``cute.copy`` / ``cute.autovec_copy`` statement WRITES, else None.

    ⚠ Matches only the UNGUARDED statement form; callers that must also see a copy
    nested under an ``if`` guard use :func:`_tv_copy_dest_names`.

    ⭐ THIS EXISTS BECAUSE ``ReadWrites`` CANNOT SEE A COPY'S DESTINATION, and that
    blind spot is a wrong answer rather than a missed optimisation.  MEASURED on the
    emitted TV form: ``ReadWrites.from_ast`` on
    ``cute.copy(_tv_atom, _tv_part_0[None, 0, lane], _tv_frag_0)`` reports reads
    ``{cute, _tv_atom, _tv_part_0, lane, _tv_frag_0}`` and **writes {}** -- the
    destination is an ARGUMENT, so every dataflow question asked through ``ReadWrites``
    alone treats a copy as a pure read.

    Two consequences, both observed on ``rms`` before this function existed:

    * a backward slice from a reduction's input cannot tell that the load copy is what
      FILLS the fragment, so slicing drops it and the reduction reads uninitialised
      registers;
    * a STORE flush is indistinguishable from a load, so it gets kept in the
      accumulate pass -- where its fragment has not been written yet -- and copies
      garbage to global memory.  Benign only while the output aliases nothing the
      kernel also reads; with an in-place output it clobbers the input before the
      second pass reads it.

    The spelling is fixed by ``ChunkTVPlan.emit_copy`` (``cute.copy(atom, src, dst)``)
    and ``emit_stage_copy`` (``cute.autovec_copy(src, dst)``), so the destination is
    the LAST argument in both -- which is what this reads, rather than guessing by
    argument name.
    """
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
    elif isinstance(stmt, ast.Call):
        # ``ast.walk`` yields the Call itself, which is what
        # ``_tv_copy_dest_names`` iterates over.
        call = stmt
    else:
        return None
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in (
        "copy",
        "autovec_copy",
    ):
        return None
    if not call.args:
        return None
    dest = call.args[-1]
    while isinstance(dest, ast.Subscript):
        dest = dest.value
    return dest.id if isinstance(dest, ast.Name) else None


def _tv_copy_dest_names(stmt: ast.AST) -> set[str]:
    """Every name a ``cute.copy`` / ``cute.autovec_copy`` inside ``stmt`` WRITES.

    ⚠ IT LOOKS **THROUGH GUARDS**, and that is the whole reason it returns a set
    rather than one optional name.  A copy is not always a bare expression
    statement: ``_cute_tv_partition_hoist`` wraps the store leg in
    ``if <row mask> and <tail predicate>:``, so the statement is an ``ast.If``.

    MEASURED, when this only matched a bare ``ast.Expr``: at ``block_sizes=[2]`` the
    row mask exists, so ``rms_norm``'s flush was emitted as
    ``if mask_0: cute.copy(_tv_atom, _tv_frag_2, _tv_part_2[...])`` -- which this
    function did not recognise as a store, so it was classified load-side and ran in
    the ACCUMULATE pass, publishing an unwritten fragment to the output.  The gate's
    ``class8_persistent_path_correct`` caught it at relerr 120.27 (``bs`` 2 and 4,
    ``vw`` 2 and 4; ``bs=1`` passed precisely because no mask means no ``ast.If``).
    """
    found: set[str] = set()
    for sub in ast.walk(stmt):
        if (name := _tv_copy_dest_name(sub)) is not None:
            found.add(name)
    return found


def _tv_backward_slice(body: list[ast.AST], roots: set[str]) -> list[int]:
    """:func:`_backward_slice`, but a ``cute.copy`` counts as writing its destination.

    Used where the producers of a reduction's input are COPIES rather than assignments --
    the lane level of a TV nest, and the shared-producer re-materialisation between sibling
    lane segments (:meth:`DeviceGridState._emit_lane_segment`).  See
    :func:`_tv_copy_dest_name` for why the plain ``ReadWrites`` view cannot answer this and
    what breaks when it is used: the destination is an ARGUMENT, so a fragment-filling copy
    is invisible as a producer and the reduction ends up folding uninitialised registers.

    ⚠ USES :func:`_tv_copy_dest_names` (PLURAL), WHICH LOOKS THROUGH GUARDS.  A copy is not
    always a bare expression statement -- ``_cute_tv_partition_hoist`` wraps a leg in
    ``if <row mask> and <tail predicate>:`` -- and the singular ``_tv_copy_dest_name`` only
    matches the unguarded form.  MEASURED elsewhere in this file: reading only the unguarded
    form classified a masked flush as a non-store and published an unwritten fragment
    (relerr 120.27).  ⇒ the plural view is the correct one for a dependence question.

    ⭐ THIS FUNCTION HAD **ZERO CALLERS** until the lane-segment work needed exactly it.
    Rather than adding a second near-identical walk (the duplicate was written first, then
    removed), the existing one was widened to the guard-looking-through view and reused --
    two walks answering the same dependence question is how they drift.
    """
    from .ast_read_writes import ReadWrites

    needed = set(roots)
    selected: list[int] = []
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        rw = ReadWrites.from_ast(stmt)
        dests = _tv_copy_dest_names(stmt)
        writes = set(rw.writes) | dests
        if writes & needed:
            selected.append(idx)
            # A copy READS its destination name too (``ReadWrites`` counts the
            # argument), so drop it from the demand set: keeping it would make the
            # copy appear to depend on itself and pull in every earlier writer of the
            # same fragment.
            needed |= set(rw.reads) - dests
    selected.reverse()
    return selected


def _split_one_lane_loop(loop: ast.For, lane_var: str) -> list[ast.AST]:
    body: list[ast.AST] = list(loop.body)
    markers: list[tuple[int, _LaneReduceMarker]] = []
    for idx, stmt in enumerate(body):
        parsed = _is_lane_reduce_marker_assign(stmt)
        if parsed is not None:
            markers.append((idx, parsed))
    if not markers:
        # ⛔ THE CONSTEXPR-V ARM WAS HERE, AND IT IS GONE (T9 option (b)).
        #
        # A loop-free reduction carrying a TV plan emits its marker one level deeper than
        # this function looks -- inside ``for vi in cutlass.range_constexpr(vec)`` -- so
        # this arm called ``_split_lane_loop_over_constexpr_vec`` to recognise and rewrite
        # that shape.  ⭐ THAT WAS THE **ONLY** NEW CORRECTNESS RELIANCE ON THIS AST PASS
        # the CuTe reduction rework added, and the functions it needed (recogniser,
        # single-stage rewrite, layered rewrite, loop rebuilder, clone-patching set) were
        # the bulk of the pass's growth: 615 -> 987 statements.
        #
        # ⭐⭐ IT IS RETIRED BY MOVING THE DECISION UP, NOT BY DROPPING THE CAPABILITY.
        # ``ConfigSpec.normalize`` now ROLLS a persistent reduction that would take a TV
        # plan (see its "A PERSISTENT REDUCTION THAT WOULD TAKE A TV PLAN IS ROLLED
        # INSTEAD" block), so the shape reaches ``LoopedReductionStrategy`` instead --
        # which gets one subgraph per dependency layer from the reduction roller and needs
        # no marker at all.  ⇒ the marker this arm existed to recognise is never emitted,
        # because the structure it encoded is now expressed by the GRAPH.
        #
        # MEASURED over 216 loop-free configs across three shipped examples: ZERO raises,
        # ZERO markers, 198 carrying a TV layout.  ⛔ An earlier version of this work
        # instead gated the loop-free TV plan off, which retired the capability along with
        # the rewrite (144 configs lost their TV copy).  That trade is undone.
        #
        # ⚠ IF A MARKER EVER REACHES THIS DEPTH AGAIN it now flows to
        # :func:`restore_unprocessed_lane_reduce_markers`, which RAISES
        # ``BackendUnsupported`` (its ``partial_fold`` arm) rather than reverting to the raw
        # per-lane input.  That is the good failure mode -- loud, at compile time, naming
        # the reduction -- and not the silent ``R = IN`` revert that dropped the fold.
        return [loop]

    marker_indices = {i for i, _ in markers}

    # A matmul whose *output* is reduced over a lane-distributed axis (e.g.
    # matmul_layernorm's ``acc.sum(-1)`` over the synthetic-lane N output) cannot
    # be handled by the per-lane / two-pass paths below: each lane owns a
    # distinct output column, so the reduction must combine DIFFERENT lanes, and
    # the matmul (an unduplicatable cross-thread shared-memory reduction) cannot
    # be re-run in a second lane pass.  The register-stash lowering runs the
    # matmul once, stashes each lane's output in a per-thread fragment, and
    # re-derives every downstream reduction / consumer from the stash.
    #
    # This is gated narrowly so it does NOT disturb kernels the existing paths
    # already handle correctly:
    #   * an unduplicatable op must feed the reduction, and
    #   * the marker results must NOT be consumed by a cross-lane loop-carried
    #     accumulator.  Online-softmax attention carries ``mi``/``di`` across the
    #     lane loop (``di = di * alpha + sum``); there the per-lane restore is
    #     correct, so the stash path must stay out of the way.
    if any(
        _contains_unduplicatable_op(stmt) for stmt in body
    ) and not _markers_feed_cross_lane_carry(body, lane_var, markers):
        stashed = _split_lane_loop_with_register_stash(loop, lane_var, markers)
        if stashed is not None:
            return stashed

    # Safety: the two-pass split is only valid when the reduction marker is the
    # ONLY cross-lane carried value in this lane loop. If the body has another
    # loop-carried accumulator across the lanes (e.g. a matmul ``dot_acc`` or a
    # plain ``extra += per_lane`` sum that already accumulates over the lanes),
    # splitting would drop or double-count it, so this split is not available.
    #
    # ⛔ THAT IS A REASON NOT TO SPLIT.  IT IS NOT A REASON THE RESTORE IS CORRECT.
    # This site used to ``return [_restore_per_lane_markers(...)]`` directly, and
    # that was a LIVE SILENT WRONG ANSWER -- see ``_decline_structural_lane_split``
    # for the measurement and why ``_has_extra_cross_lane_carry`` cannot answer the
    # question this decline actually has to ask.
    if _has_extra_cross_lane_carry(body, lane_var, marker_indices):
        return _decline_structural_lane_split(
            loop,
            lane_var,
            markers,
            reason=(
                "a reduction over a lane-distributed axis in a lane loop that also "
                "carries an accumulator independent of that reduction, so the "
                "two-pass split would drop or double-count the accumulator"
            ),
        )

    input_roots = {m.input_name for _, m in markers}

    # Phase 1: the backward slice that produces all reduction inputs.
    phase1_indices, _phase1_written = _backward_slice(body, input_roots)

    # Sequentially-dependent reductions (one marker's input depends on another
    # marker's result, e.g. an arange softmax denominator's sum-of-exp needing
    # the row max first) cannot be expressed by the single
    # phase-1/finalize/phase-2 structure below: the second marker's input is not
    # computable until the first marker's reduced scalar exists.  Emit one
    # accumulate/finalize pass per dependency LAYER instead (see
    # ``_split_lane_loop_multi_stage``).
    #
    # ⛔ IF THE LAYERED SPLIT DECLINES, THE PER-LANE RESTORE IS ONLY LEGAL WHEN
    #    SOMETHING ELSE FOLDS THE LANE AXIS -- see
    #    ``_decline_structural_lane_split``, which all three decline sites share.
    #
    # MEASURED at this site: ``attention_output`` at ``block_sizes=[1,32,64]``
    # reaches the decline with a genuine cross-lane carry (its ``mi``/``di``
    # recurrence re-folds the lane axis itself), so it must keep compiling.  All
    # eight attention examples reach here, and raising UNCONDITIONALLY broke every
    # one of them -- do not "simplify" the helper's carry check away.
    if set(phase1_indices) & marker_indices:
        staged = _split_lane_loop_multi_stage(loop, lane_var, markers)
        if staged is not None:
            return staged
        return _decline_structural_lane_split(
            loop,
            lane_var,
            markers,
            reason=(
                "sequentially-dependent reductions over a lane-distributed axis "
                "whose producers contain a matmul or cross-thread collective that "
                "would have to be re-emitted in more than one pass"
            ),
        )

    # The phase-1 (accumulate) and phase-2 (consume) passes both re-run the
    # reduction-input producers. That is only safe for side-effect-free
    # producers. A matmul / collective in the slice (cross-thread shared-memory
    # reductions, ``cute.gemm``, ``dot``) cannot be duplicated without racing on
    # shared memory, so the two-pass split is not available here either.
    #
    # ⛔ Again: "cannot split" is not "the restore is correct".  The comment this
    # replaced asserted the stash "handles the cases where the per-lane restore
    # would be numerically wrong" -- but the stash runs only when it can find
    # stashable names and the extent is affordable, and when it declines control
    # arrives HERE.  Route through the same shared decline as the other two sites.
    if any(_contains_unduplicatable_op(body[i]) for i in phase1_indices):
        return _decline_structural_lane_split(
            loop,
            lane_var,
            markers,
            reason=(
                "a reduction over a lane-distributed axis whose input is produced by "
                "a matmul or cross-thread collective that cannot be re-run in the "
                "second pass of a two-pass split"
            ),
        )

    extent = _lane_loop_extent(loop)

    prefix: list[ast.AST] = []  # acc init statements (outside the lane loops)
    accumulate_body: list[ast.AST] = [body[i] for i in phase1_indices]
    finalize: list[ast.AST] = []
    for _, m in markers:
        acc_var = f"{m.result_var}_lane_acc"
        prefix.append(statement_from_string(f"{acc_var} = {m.identity_expr}"))
        # Cast the per-lane input to the accumulator dtype before combining so
        # the CUTLASS DSL's strict ternary type check (max/min emit a Python
        # ``a if a > b else b``) does not see mixed fp32/bf16 operands.  This is
        # also what makes an int32 sum accumulate in int64 rather than wrapping.
        ctor = _marker_acc_ctor(m)
        combine_val = f"{ctor}({m.input_name})" if ctor is not None else m.input_name
        accumulate_body.append(
            statement_from_string(
                f"{acc_var} = {_combine_expr(m.reduction_type, acc_var, combine_val)}"
            )
        )
        finalize.extend(_finalize_lane_reduce_marker(m, acc_var))

    invariant_indices, varying_indices = _lane_reduce_consume_tail(
        body, lane_var, marker_indices
    )

    result: list[ast.AST] = []
    result.extend(prefix)
    result.append(_create_lane_loop(lane_var, extent, accumulate_body))
    result.extend(finalize)
    result.extend(body[i] for i in invariant_indices)
    if varying_indices:
        result.append(
            _create_lane_loop(lane_var, extent, [body[i] for i in varying_indices])
        )
    return result


def _lane_reduce_consume_tail(
    body: list[ast.AST], lane_var: str, marker_indices: set[int]
) -> tuple[list[int], list[int]]:
    """Partition the post-reduction consume statements of a lane-loop body.

    Returns ``(lane_invariant_indices, lane_varying_indices)`` as indices into
    ``body``.  The marker assignments themselves are excluded: their reduced
    scalars are finalized outside the lane loops, so consumers read them
    directly.

    A statement is lane-varying if it (transitively) reads the lane var.
    Statements that only depend on the finalized scalar(s) are lane-invariant and
    run once after the accumulate pass(es); lane-varying consumers run in a final
    lane pass, but only those that contribute to a side effect (a store, an
    ``if``-with-store, an in-place write).  Pure lane-varying producers that fed
    only the (now-removed) reduction markers are dropped.
    """
    from .ast_read_writes import ReadWrites

    consume_indices = [i for i in range(len(body)) if i not in marker_indices]
    consume_body = [body[i] for i in consume_indices]
    lane_varying_names = _lane_varying_names(consume_body, lane_var)
    keep_indices = _live_phase2_indices(consume_body)
    lane_invariant: list[int] = []
    lane_varying: list[int] = []
    for pos, body_idx in enumerate(consume_indices):
        reads = set(ReadWrites.from_ast(body[body_idx]).reads)
        if lane_var in reads or bool(reads & lane_varying_names):
            if pos in keep_indices:
                lane_varying.append(body_idx)
        else:
            lane_invariant.append(body_idx)
    return lane_invariant, lane_varying


def _backward_slice_to_boundary(
    body: list[ast.AST], roots: set[str], boundary_names: set[str]
) -> tuple[list[int], set[str]]:
    """Backward slice of ``roots`` that STOPS at ``boundary_names``.

    Like :func:`_backward_slice`, but names in ``boundary_names`` are treated as
    already defined (they are produced outside ``body``), so neither their
    defining statement nor its transitive producers are selected.

    Returns ``(selected_indices, boundaries_hit)`` — the latter is the subset of
    ``boundary_names`` the slice actually depends on.

    The lane-reduce split uses this with the marker result vars as boundaries: a
    marker's reduced scalar is finalized *outside* the lane loops, so a later
    pass that consumes it must read that scalar, not re-derive the per-lane input
    the reduction folded away.  Slicing without the boundary would drag every
    earlier marker's (now dead) producer chain into the later pass.
    """
    from .ast_read_writes import ReadWrites

    needed = set(roots) - boundary_names
    hit = set(roots) & boundary_names
    selected: list[int] = []
    for idx in range(len(body) - 1, -1, -1):
        rw = ReadWrites.from_ast(body[idx])
        if not (set(rw.writes) & needed):
            continue
        selected.append(idx)
        reads = set(rw.reads)
        hit |= reads & boundary_names
        needed |= reads - boundary_names
    selected.reverse()
    return selected, hit


def _lane_reduce_dependency_stages(
    body: list[ast.AST], markers: list[tuple[int, _LaneReduceMarker]]
) -> tuple[list[list[int]], dict[int, list[int]]] | None:
    """Group lane-reduce markers into dependency LAYERS.

    Returns ``(stages, slice_by_marker)`` where ``stages[k]`` is the list of body
    indices of the markers at depth ``k`` and ``slice_by_marker[i]`` is the
    minimal backward slice (body indices) that produces marker ``i``'s reduction
    input, stopping at every marker result (those are finalized scalars defined
    outside the lane loops).

    A marker's slice is taken over ``body[:i]`` — the marker reads a value that
    must already be defined where the marker stands — so a marker it depends on
    is necessarily *earlier*, and iterating markers in body order visits every
    dependency before its dependent.  Depth is ``1 + max(depth of the markers
    whose results the slice reads)``; markers reading no other marker's result
    (the mutually independent ones) all land in stage 0 and share one pass.

    Returns ``None`` when the marker results are not single-assignment within the
    body, which is what makes them safe slice boundaries.
    """
    from .ast_read_writes import ReadWrites

    result_index: dict[str, int] = {}
    for idx, m in markers:
        if m.result_var in result_index:
            return None
        result_index[m.result_var] = idx
    boundary_names = set(result_index)
    # A boundary is only sound when the marker statement is the name's ONLY
    # definition in this body; otherwise dropping the name from the slice's
    # needed set would drop a live producer.
    marker_indices = set(result_index.values())
    for idx, stmt in enumerate(body):
        if idx in marker_indices:
            continue
        if set(ReadWrites.from_ast(stmt).writes) & boundary_names:
            return None

    slice_by_marker: dict[int, list[int]] = {}
    depth: dict[int, int] = {}
    for idx, m in markers:
        selected, hit = _backward_slice_to_boundary(
            body[:idx], {m.input_name}, boundary_names
        )
        slice_by_marker[idx] = selected
        depth[idx] = 1 + max((depth[result_index[n]] for n in hit), default=-1)
    stages: list[list[int]] = [[] for _ in range(max(depth.values()) + 1)]
    for idx, _ in markers:
        stages[depth[idx]].append(idx)
    return stages, slice_by_marker


def _split_lane_loop_multi_stage(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
) -> list[ast.AST] | None:
    """Split a lane loop whose reduction markers are SEQUENTIALLY DEPENDENT.

    The plain two-pass split assumes every marker's reduction input is computable
    in one accumulate pass.  When one marker's input reads another marker's
    reduced scalar (an arange softmax denominator's ``sum(exp(v - amax(v)))``, or
    the two-marker expansion of an indexed reduction) that is impossible: the
    second input does not exist until the first reduction has been folded across
    the lanes *and* the thread axis.

    Emit one accumulate/finalize pass per dependency layer instead::

        acc_A = identity_A
        for LANE: <slice(A.input)>;  acc_A = combine(acc_A, A.input)
        A = warp_combine(acc_A)                 # A is now a finalized scalar
        acc_B = identity_B
        for LANE: <slice(B.input), reading A>;  acc_B = combine(acc_B, B.input)
        B = warp_combine(acc_B)
        <lane-invariant consumers>
        for LANE: <lane-varying consumers / stores>

    Mutually independent markers share a layer (and therefore a pass), so a body
    with no dependency chain reduces to exactly the single-pass structure.

    Returns ``None`` when the layered split cannot be emitted safely; the caller
    then falls back to ``_restore_per_lane_markers``.
    """
    body: list[ast.AST] = list(loop.body)
    marker_indices = {i for i, _ in markers}
    marker_by_index = dict(markers)
    staged = _lane_reduce_dependency_stages(body, markers)
    if staged is None:
        return None
    stages, slice_by_marker = staged

    # Plan the passes before emitting anything, so the duplication check below is
    # exact (it counts the statements this lowering would really emit twice)
    # rather than a guess from the slices.
    #
    # Each layer re-runs the producers its own input needs.  The lane-distributed
    # values cannot be hoisted out of the lane loop — they are per-lane registers,
    # one live value per lane — and staging them in a fragment costs
    # ``extent`` registers per value, which is why the register-stash lowering caps
    # itself at extent <= 256 while the extents reaching here go up to 1024.  So
    # re-running is the right default; what keeps it cheap is that
    # ``_lane_reduce_dependency_stages`` stops each slice at the finalized marker
    # results, so a layer re-derives only what it genuinely reads (never the
    # collapsed per-lane input of an earlier reduction), and mutually independent
    # markers share ONE pass instead of getting one each.
    pass_slices: list[list[int]] = [
        sorted({i for idx in stage for i in slice_by_marker[idx]} - marker_indices)
        for stage in stages
    ]
    invariant_indices, varying_indices = _lane_reduce_consume_tail(
        body, lane_var, marker_indices
    )

    # Re-running is only sound for side-effect-free producers, exactly as for the
    # single-pass split: a matmul / collective cannot be duplicated without racing
    # on shared memory.  A statement that this lowering would emit more than once
    # and that cannot be duplicated has no correct emission here, so decline and
    # let the caller fall back rather than silently emitting a racing collective.
    # (Emitting it once and stashing its result is the RIGHT answer in that case —
    # and it already exists: ``_split_lane_loop_with_register_stash``, which is
    # tried before this path and also emits one accumulate pass per marker.  This
    # path deliberately does not duplicate that machinery.)
    emit_count: collections.Counter[int] = collections.Counter()
    for indices in (*pass_slices, invariant_indices, varying_indices):
        emit_count.update(indices)
    for i, count in emit_count.items():
        if count > 1 and _contains_unduplicatable_op(body[i]):
            return None

    extent = _lane_loop_extent(loop)
    # A statement emitted in more than one pass must be a fresh node in each:
    # splicing one node object into two places in the tree breaks AST walking.
    # The first emission keeps the original node so its Helion AST metadata (e.g.
    # an already-split nested lane loop's marker attribute) survives.
    emitted: set[int] = set()

    def take(index: int) -> ast.AST:
        if index in emitted:
            return _clone_stmt(body[index])
        emitted.add(index)
        return body[index]

    result: list[ast.AST] = []
    for stage, stage_slice in zip(stages, pass_slices, strict=True):
        pass_body: list[ast.AST] = [take(i) for i in stage_slice]
        for idx in stage:
            m = marker_by_index[idx]
            acc_var = f"{m.result_var}_lane_acc"
            result.append(statement_from_string(f"{acc_var} = {m.identity_expr}"))
            # Cast the per-lane input to the accumulator dtype before combining
            # so the CUTLASS DSL's strict ternary type check (max/min emit a
            # Python ``a if a > b else b``) does not see mixed fp32/bf16 operands.
            # Same source of truth as the single-pass split: the marker's
            # explicit accumulator dtype, so an int32 sum still widens.
            ctor = _marker_acc_ctor(m)
            combine_val = (
                f"{ctor}({m.input_name})" if ctor is not None else m.input_name
            )
            pass_body.append(
                statement_from_string(
                    f"{acc_var} = {_combine_expr(m.reduction_type, acc_var, combine_val)}"
                )
            )
        result.append(_create_lane_loop(lane_var, extent, pass_body))
        for idx in stage:
            m = marker_by_index[idx]
            result.extend(_finalize_lane_reduce_marker(m, f"{m.result_var}_lane_acc"))

    result.extend(take(i) for i in invariant_indices)
    if varying_indices:
        result.append(
            _create_lane_loop(lane_var, extent, [take(i) for i in varying_indices])
        )
    return result


_LANE_STASH_COUNTER = itertools.count()


def _stash_dtype_for_value(
    value_name: str, body: list[ast.AST], markers: list[tuple[int, _LaneReduceMarker]]
) -> str:
    """Pick a CuTe scalar dtype constructor for a stashed lane value.

    Prefer the accumulator dtype of a marker whose input transitively reads the
    stashed value (matmul outputs feed an fp32 ``sum``), then any cast that
    wraps the value's defining assignment, then ``cutlass.Float32``.
    """
    from .ast_read_writes import ReadWrites

    for _idx, m in markers:
        slice_indices, _ = _backward_slice(body, {m.input_name})
        reads_value = any(
            value_name in ReadWrites.from_ast(body[i]).reads for i in slice_indices
        )
        if reads_value:
            ctor = _marker_acc_ctor(m)
            if ctor is not None:
                return ctor
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == value_name
            and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "cutlass"
        ):
            return f"cutlass.{stmt.value.func.attr}"
    return "cutlass.Float32"


def _strip_ssa_suffix(name: str) -> str:
    """Strip Helion's SSA / loop-carry suffixes from a variable name.

    ``acc_1`` / ``acc_copy`` / ``acc_copy_0`` all collapse to ``acc`` so a
    loop-carried accumulator can be matched to its per-iteration rewrites.
    """
    base = re.sub(r"(_copy)(_\d+)*$", "", name)
    return re.sub(r"(_\d+)+$", "", base)


def _assigns_simple_name(stmt: ast.AST, names: set[str]) -> bool:
    """Return True when ``stmt`` is ``X = ...`` for some ``X`` in ``names``."""
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id in names
    )


def _undup_stmt_rederives(stmt: ast.AST, name: str) -> bool:
    """Return True when ``stmt`` (an unduplicatable statement, typically the
    matmul K ``for`` loop) re-derives the loop-carried accumulator ``name``.

    The statement re-derives ``name`` when it both reads ``name`` (the live-in
    accumulator) and writes a value transitively dependent on ``name`` whose
    SSA-stripped name equals ``name`` (e.g. ``acc_1 = acc_copy_0 + ...``).  A
    plain input the matmul only reads (``indices_2``) is not re-derived.
    """
    from .ast_read_writes import ReadWrites

    if not isinstance(stmt, ast.For):
        rw = ReadWrites.from_ast(stmt)
        return name in rw.reads and any(_strip_ssa_suffix(w) == name for w in rw.writes)
    if name not in ReadWrites.from_ast(stmt).reads:
        return False
    # Forward slice from ``name`` within the loop body; True when it reaches a
    # write whose stripped name is ``name``.
    inner = list(stmt.body)
    tainted = {name}
    changed = True
    while changed:
        changed = False
        for s in inner:
            rw = ReadWrites.from_ast(s)
            if set(rw.reads) & tainted:
                for w in rw.writes:
                    if w not in tainted:
                        tainted.add(w)
                        changed = True
    return any(_strip_ssa_suffix(w) == name for w in tainted if w != name)


def _store_value_and_addr_reads(stmt: ast.AST) -> tuple[set[str], set[str]] | None:
    """For a statement containing a single ``(ADDR).store(VALUE)`` call, return
    ``(value_reads, addr_reads)`` — the names read in the stored VALUE and the
    names read in the ADDRESS expression (plus any guarding condition).

    Returns ``None`` when the statement has no ``.store(...)`` call (or has more
    than one, which this analysis does not attempt to characterize)."""
    stores = [
        node
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
        and len(node.args) == 1
    ]
    if len(stores) != 1:
        return None
    store = stores[0]
    value_reads = {n.id for n in ast.walk(store.args[0]) if isinstance(n, ast.Name)}
    # Everything read in the statement that is not part of the stored value
    # belongs to the address expression / guard (e.g. an enclosing ``if mask:``).
    all_reads = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    addr_reads = all_reads - value_reads
    return value_reads, addr_reads


def _lane_axis_wrongly_collapsed(
    tail: list[ast.AST],
    lane_var: str,
    lane_varying: set[str],
    stash_set: set[str],
    marker_result_names: set[str],
) -> bool:
    """Return True when the register-stash lowering would collapse the lane axis
    incorrectly.

    The stash lowering reduces each marker over the lane axis, which is only
    correct when the lane-distributed axis IS the reduced axis (matmul_layernorm:
    the reduced output free dim is broadcast back into a per-lane store whose
    VALUE still depends on the per-lane stash output).  When a side-effecting
    store writes to a lane-varying ADDRESS but its stored VALUE depends only on
    the reduced marker scalars (never on a per-lane stash output), each lane is a
    distinct, preserved output element that the lane reduction wrongly collapses
    (e.g. ``baddbmm(...).sum(-1)`` with the reduced dim folded into the matmul).

    ``lane_varying`` is the set of lane-derived names over the FULL lane-loop body
    (the per-lane output index, e.g. ``indices_2``, lives in the compute region
    rather than the tail, so it must be supplied by the caller).
    """
    # Names in the tail that (transitively) depend on a stashed per-lane value,
    # WITHOUT crossing a marker reduction.  A reduction marker collapses the lane
    # axis: its result is a cross-lane reduced scalar, no longer a per-lane value,
    # so the taint must STOP at marker results (a store of the reduced scalar is
    # exactly the collapse-bug signature, and must not count as stash-dependent).
    stash_tainted = _forward_taint_excluding_markers(
        tail, stash_set, marker_result_names
    )
    for stmt in tail:
        parsed = _store_value_and_addr_reads(stmt)
        if parsed is None:
            continue
        value_reads, addr_reads = parsed
        addr_lane_varying = lane_var in addr_reads or bool(addr_reads & lane_varying)
        value_depends_on_stash = bool(value_reads & stash_tainted)
        value_depends_on_marker = bool(value_reads & marker_result_names)
        if addr_lane_varying and value_depends_on_marker and not value_depends_on_stash:
            return True
    return False


def _forward_taint_excluding_markers(
    body: list[ast.AST], roots: set[str], marker_result_names: set[str]
) -> set[str]:
    """Forward-slice taint of ``roots`` through ``body`` that does NOT propagate
    across a lane-reduce marker.

    A marker assigns a cross-lane reduced scalar (``R = ..._helion_lane_reduce``);
    its result no longer depends on a single lane's value, so a statement that
    only writes a marker result must not inherit the taint even when its reduction
    input was tainted.
    """
    from .ast_read_writes import ReadWrites

    tainted = set(roots)
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if not (set(rw.reads) & tainted):
                continue
            for w in rw.writes:
                # A marker result is a reduction boundary: never taint it.
                if w in marker_result_names:
                    continue
                if w not in tainted:
                    tainted.add(w)
                    changed = True
    return tainted


#: Lane extent above which the register stash is not the PREFERRED lowering.
#:
#: The stash costs ``extent`` registers per stashed value, so a wide extent spills
#: to local memory.  ⚠ This is a PERFORMANCE preference, NOT a correctness bound --
#: the stash is correct at every extent (MEASURED: rel 5.7e-07 / 6.1e-07 / 5.5e-07
#: at extents 512 / 1024 / 2048 on ``matmul_layernorm``).  So it is applied only
#: where an alternative lowering exists.  When the alternative is a wrong answer or
#: a compile error, :func:`_decline_structural_lane_split` deliberately overrides it
#: -- correct-and-spilling beats silently-wrong, and it beats refusing to compile a
#: shape we can express.
_STASH_PREFERRED_MAX_EXTENT = 256


def _split_lane_loop_with_register_stash(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    *,
    extent_cap: int | None = _STASH_PREFERRED_MAX_EXTENT,
) -> list[ast.AST] | None:
    """Lower a lane loop whose reduction inputs depend on an unduplicatable op.

    The standard two-pass split re-runs the reduction-input producers in a
    second pass, which is unsafe when a matmul / cross-thread reduction is in
    the slice.  Instead, run the matmul-bearing *compute region* (the prefix of
    the body up to and including the last unduplicatable statement) exactly
    once, stashing each lane's unduplicatable-derived live-out values into
    per-thread register fragments.  Every downstream reduction marker and
    consumer is then re-derived from the stash (a plain register read, safely
    duplicatable), one accumulate/finalize pass per marker in dependency order,
    plus a final consume pass for the side-effecting statements.

    Returns the replacement statement list, or ``None`` when the pattern does
    not apply (the caller then chooses another lowering, or refuses).

    ``extent_cap`` bounds the lane extent this lowering is willing to take, and
    defaults to :data:`_STASH_PREFERRED_MAX_EXTENT` (a register-cost preference).
    Pass ``None`` to lift it when the only alternatives are a wrong answer or a
    compile error -- see :func:`_decline_structural_lane_split`.
    """
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    marker_indices = {i for i, _ in markers}
    extent = _lane_loop_extent(loop)

    # Compute region: prefix of the body up to (and including) the last
    # unduplicatable statement.  Everything after it must be free of
    # unduplicatable ops so it can be re-derived from the stash.
    undup_indices = [
        i for i in range(len(body)) if _contains_unduplicatable_op(body[i])
    ]
    if not undup_indices:
        return None
    region_end = max(undup_indices)
    region = body[: region_end + 1]
    tail = body[region_end + 1 :]
    tail_offset = region_end + 1
    if any(_contains_unduplicatable_op(s) for s in tail):
        return None
    # A marker inside the compute region cannot be re-derived from the stash
    # (its reduction would have to run before the matmul finishes).
    if any(i <= region_end for i in marker_indices):
        return None

    if extent <= 0 or (extent_cap is not None and extent > extent_cap):
        return None

    lane_varying = _lane_varying_names(body, lane_var)

    tail_reads: set[str] = set()
    for stmt in tail:
        tail_reads |= set(ReadWrites.from_ast(stmt).reads)

    # Identify the loop-carried accumulators produced by the unduplicatable
    # statements — values whose post-region magnitude depends on the matmul and
    # therefore cannot be recomputed.  These are exactly the names that MUST be
    # stashed.  Helion emits a matmul K loop as
    # ``acc = 0; for k: acc_copy = acc; ...; acc_<n> = acc_copy_0 + reduce`` and
    # an implicit phi makes the post-loop ``acc`` equal the loop output
    # ``acc_<n>`` (collapsed by a later rename pass).  A name X is carried when:
    #   * X is written by a non-unduplicatable region statement (its ``acc = 0``
    #     seed) and read after the region (lane-varying), and
    #   * an unduplicatable statement's body re-derives X — its forward slice
    #     from X reaches a write whose SSA-stripped name equals X (``acc_1`` /
    #     ``acc_copy`` -> ``acc``).
    # The forward-slice condition distinguishes a true accumulator (``acc``,
    # rewritten each K step) from a plain input that the matmul merely reads
    # (``indices_2``, unchanged through the loop).
    seed_writes: set[str] = set()
    for i, stmt in enumerate(region):
        if i in undup_indices:
            continue
        seed_writes |= set(ReadWrites.from_ast(stmt).writes)
    # The carried accumulator's *name* (``acc`` from ``acc = 0``) is not itself
    # lane-varying — only the loop output alias (``acc_1``) is — so do NOT filter
    # candidates by ``lane_varying`` here.  The re-derives check confirms the
    # matmul transforms the accumulator, and below we require the re-deriving
    # statement to be lane-varying so a genuinely lane-invariant accumulator is
    # left alone.
    candidate_carried = seed_writes & tail_reads
    carried: set[str] = set()
    for name in candidate_carried:
        for i in undup_indices:
            if not _undup_stmt_rederives(body[i], name):
                continue
            stmt_reads = set(ReadWrites.from_ast(body[i]).reads)
            if lane_var in stmt_reads or bool(stmt_reads & lane_varying):
                carried.add(name)
                break
    # Also stash any value DIRECTLY produced by an unduplicatable statement that
    # is read by the tail (e.g. a matmul whose output is a fresh name rather than
    # an accumulator phi).
    undup_writes: set[str] = set()
    for i in undup_indices:
        undup_writes |= set(ReadWrites.from_ast(body[i]).writes)
    direct = undup_writes & lane_varying & tail_reads
    stash_names = sorted(carried | direct)
    if not stash_names:
        return None

    stash_set = set(stash_names)

    # Correctness gate: the register-stash lowering reduces each marker OVER THE
    # LANE axis (stash per lane, then sum lanes + warp-reduce).  That is only
    # valid when the lane-distributed axis IS the reduced axis — i.e. the genuine
    # matmul_layernorm pattern, where the matmul OUTPUT free dim is split across
    # the lane and the layernorm reduces exactly that dim, then broadcasts the
    # reduced scalar back into a *per-lane* normalize+store (each lane keeps a
    # distinct, stash-derived output value).
    #
    # A different pattern — e.g. ``baddbmm(...).sum(-1)`` over a small static dim
    # — folds the reduced axis into the matmul itself, leaving each lane holding a
    # COMPLETE, distinct output element.  There the lane axis is a *preserved*
    # output dim, the spurious marker reduces over the wrong (lane) axis, and the
    # store writes the single reduced scalar to lane-varying addresses (every lane
    # storing the same collapsed value).  Detect that here and bail to the
    # correct per-lane path: a side-effecting store whose ADDRESS depends on the
    # lane var but whose stored VALUE depends only on reduced marker results (no
    # per-lane stash output) means the lane axis was wrongly collapsed.
    marker_result_names = {m.result_var for _, m in markers}
    if _lane_axis_wrongly_collapsed(
        tail, lane_var, lane_varying, stash_set, marker_result_names
    ):
        return None
    # Region statements that are *duplicatable* and can be recomputed cheaply in
    # the later passes (e.g. ``indices_2 = thread_idx[0] + lane * 4``).  Drop the
    # unduplicatable statements and any statement that produces a stashed name.
    recompute: list[ast.AST] = []
    for i, stmt in enumerate(region):
        if i in undup_indices:
            continue
        writes = set(ReadWrites.from_ast(stmt).writes)
        if writes & stash_set:
            continue
        recompute.append(stmt)
    # Keep only the recompute statements that (transitively) feed the tail.
    recompute_keep_idx, _ = _backward_slice(recompute, tail_reads)
    recompute_kept = [recompute[i] for i in recompute_keep_idx]
    # The recompute slice must not pull in an unduplicatable op or reference a
    # stashed name (which is only available from the fragment, not recomputable).
    for s in recompute_kept:
        if _contains_unduplicatable_op(s):
            return None

    # Allocate one register fragment per stashed value.
    uid = next(_LANE_STASH_COUNTER)
    frag_by_name: dict[str, str] = {}
    decls: list[ast.AST] = []
    for name in stash_names:
        frag = f"_lane_stash_{uid}_{name}"
        frag_by_name[name] = frag
        dtype = _stash_dtype_for_value(name, body, markers)
        decls.append(
            statement_from_string(f"{frag} = cute.make_fragment({extent}, {dtype})")
        )

    def read_stash_stmts() -> list[ast.AST]:
        return [
            statement_from_string(f"{name} = {frag_by_name[name]}[{lane_var}]")
            for name in stash_names
        ]

    # Phase 0: run the compute region once and stash the live-out values.
    phase0_body: list[ast.AST] = list(region)
    for name in stash_names:
        phase0_body.append(
            statement_from_string(f"{frag_by_name[name]}[{lane_var}] = {name}")
        )

    # The marker assignments within the tail produce already-finalized scalars
    # (computed once after each reduction pass), so they must NOT be re-run as
    # per-lane passthroughs inside any later pass.  Build the re-derivable tail
    # (every non-marker statement) and the set of finalized marker result vars
    # that downstream slices treat as pre-defined boundaries.
    marker_result_vars = {m.result_var for _, m in markers}
    rederivable_tail = [s for s in tail if _is_lane_reduce_marker_assign(s) is None]

    result: list[ast.AST] = []
    result.extend(decls)
    result.append(_create_lane_loop(lane_var, extent, phase0_body))

    # Process each marker in source order (they are sequentially dependent: a
    # later marker's input may read an earlier marker's finalized scalar).
    for _idx, m in markers:
        acc_var = f"{m.result_var}_lane_acc"
        result.append(statement_from_string(f"{acc_var} = {m.identity_expr}"))
        # Accumulate pass: recompute cheap region producers, read the stash,
        # then re-derive this marker's input and fold it into the accumulator.
        # The slice runs over the re-derivable tail only; references to other
        # markers' results stop at those finalized scalars.
        input_slice_idx, _ = _backward_slice(rederivable_tail, {m.input_name})
        input_stmts = [
            rederivable_tail[i]
            for i in input_slice_idx
            if not _assigns_simple_name(rederivable_tail[i], marker_result_vars)
        ]
        acc_body: list[ast.AST] = []
        acc_body.extend(_clone_stmt(s) for s in recompute_kept)
        acc_body.extend(read_stash_stmts())
        acc_body.extend(_clone_stmt(s) for s in input_stmts)
        ctor = _marker_acc_ctor(m)
        combine_val = f"{ctor}({m.input_name})" if ctor is not None else m.input_name
        acc_body.append(
            statement_from_string(
                f"{acc_var} = {_combine_expr(m.reduction_type, acc_var, combine_val)}"
            )
        )
        result.append(_create_lane_loop(lane_var, extent, acc_body))
        result.extend(_finalize_lane_reduce_marker(m, acc_var))

    # Final consume pass: everything in the tail except the marker assignments,
    # re-derived from the stash + finalized scalars.  Only keep statements that
    # feed a side effect (a store / in-place write).
    consume_candidates = [s for i, s in enumerate(body) if i >= tail_offset]
    consume_candidates = [
        s for s in consume_candidates if _is_lane_reduce_marker_assign(s) is None
    ]
    keep_idx = _live_phase2_indices(consume_candidates)
    consume_kept = [s for i, s in enumerate(consume_candidates) if i in keep_idx]
    if consume_kept:
        consume_body: list[ast.AST] = []
        consume_body.extend(_clone_stmt(s) for s in recompute_kept)
        consume_body.extend(read_stash_stmts())
        consume_body.extend(_clone_stmt(s) for s in consume_kept)
        result.append(_create_lane_loop(lane_var, extent, consume_body))
    return result


def _cute_lane_carried_records() -> dict[str, set[int]] | None:
    """The ``_phi``-recorded lane carries for the kernel being emitted, or None.

    ``None`` means "no producer ran" -- a non-cute backend, or no active DeviceFunction --
    and callers must fall back to their scan rather than reading it as "nothing is carried".
    See :func:`_markers_feed_cross_lane_carry` for why that distinction is load-bearing.

    ⚠ Read through the ambient ``DeviceFunction`` rather than threaded as an argument, and
    that is a deliberate trade: the consumers are module-level functions reached through
    ``split_lane_loop_reductions(body)`` from ``generate_ast``, so threading the record would
    change five signatures and the recursive walk between them.  The ambient read is scoped
    to a single ``current()`` lookup here, in one place, instead.
    """
    from .compile_environment import CompileEnvironment
    from .device_function import DeviceFunction

    # ⚠ ``NoCurrentEnvironment`` subclasses ``RuntimeError``, so both ``current()`` calls'
    # "nothing is active" signals are covered; ``AssertionError`` covers
    # ``DeviceFunction.current()``'s own assert.  Broad on purpose: this is a read of
    # OPTIONAL evidence, and failing to obtain it must degrade to the scan, never raise.
    try:
        if CompileEnvironment.current().backend.name != "cute":
            return None
        fn = DeviceFunction.current()
    except (AssertionError, RuntimeError):
        return None
    cute_state = getattr(fn, "_cute_state", None)
    if cute_state is None:
        return None
    return cute_state.lane_carried_fx_nodes


def _markers_feed_cross_lane_carry(
    body: list[ast.AST],
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
) -> bool:
    """Return True when the lane loop carries an accumulator across the lanes
    that a marker result feeds (online-softmax attention's ``m_i`` / ``l_i`` /
    ``acc`` recurrence).

    Helion represents a loop-carried value with a phi ``X_copy = X`` read at the
    TOP of the loop body (before the value is rewritten) and a corresponding
    output assignment renamed back to ``X`` by a later pass.  Such an
    accumulating lane loop must keep the existing per-lane restore lowering, so
    the register-stash path (which assumes each marker result is consumed only by
    per-lane / store consumers, never carried across lanes) must not fire.

    matmul_layernorm's N-output lane loop has no such top-level ``X_copy = X``
    carry (its ``acc`` is the matmul accumulator, carried by the *inner* K loop,
    not across the lane iterations), so this returns False and the stash path is
    free to run.

    ⭐⭐ THE PRODUCER NOW ANNOUNCES THIS, AND THE ANNOUNCEMENT WINS (task 2 step 2).
    ``_phi``'s codegen handler records every value carried across an OPEN lane loop on
    ``CuteDeviceFunctionState.lane_carried_fx_nodes``, keyed by the lane variable
    (``language/_tracing_ops.py::_record_cute_lane_carried_phi``).  When a record exists
    for ``lane_var`` this function reports it directly; the text scan below runs only as
    the fallback for a path that produced no record (a non-cute backend, or a lane loop
    built outside the phi handler's reach).

    ⛔ WHY THE RECORD IS STRICTLY BETTER, and it is measured, not aesthetic.  The scan is
    FLOW-INSENSITIVE answering a FLOW-SENSITIVE question: ``ReadWrites.from_ast`` flattens
    nested bodies, so on ``matmul_layernorm`` it returned True for ten hits that ALL came
    from a nested matmul K loop -- the K-loop temporaries read as lane-carried -- producing
    relerr **7.685** at N=512 on a kernel that compiled and looked plausible.  A record
    written at the phi is keyed to the loop that was actually open, so a K-loop phi names
    the K loop and can never be mistaken for a lane carry.

    ⚠ THE RECORD IS CONSULTED, NOT TRUSTED BLINDLY IN ONE DIRECTION.  An EMPTY record for a
    lane var that HAS an entry means "the phi handler saw this loop and it carries nothing",
    which is a positive statement and is honoured.  A MISSING entry means "no producer ran
    here", which is not evidence either way -- so it falls through to the scan rather than
    silently answering False.  Conflating those two is how a "carry the fact forward" change
    turns into a silent behaviour change on every path the producer does not cover.
    """
    from .ast_read_writes import ReadWrites

    if not markers:
        return False

    # ⛔⛔ THE RECORD IS AVAILABLE AND IS **NOT** CONSULTED HERE YET.  MEASURED, AND THIS IS
    # THE FINDING, NOT A TODO.
    #
    # Task 2 step 2 asks for "record lane-carried values at the phi, delete the scan".  The
    # producer is built and it works (``_record_cute_lane_carried_phi``: 0 records on a
    # no-carry kernel, 3 on attention's ``m_i``/``l_i`` recurrence).  But swapping the
    # consumer over is NOT a refactor -- it CHANGES THE ANSWER, and the changed answer is
    # WRONG on the one kernel that reaches this function:
    #
    #   ``matmul_layernorm``, the rel-7.685 counterexample's own shape, is the ONLY kernel in
    #   the example suite that calls this predicate (measured: attention and the P1
    #   arange-softmax shape never reach it).  The scan answers **False** there and the
    #   record answers **True**, because the kernel's ``acc`` phi IS carried across a loop
    #   that is open when the phi lowers -- but it is the matmul's K accumulator, and the
    #   question this predicate asks is specifically whether a carry crosses the *lane* axis.
    #   Consulting the record made ``test_matmul_layernorm_static_shapes`` FAIL (level 4:
    #   1 failed / 100 passed) and flipped ``test_consumed_collapse_fold_is_double_reduced``
    #   from xfail to a hard failure.
    #
    # ⚠ SO THE SCAN'S ANSWER IS RIGHT HERE AND THE RECORD'S IS WRONG -- the opposite of the
    # relationship the task statement predicts.  The doc's argument is that the scan's
    # measured bug was nested-K-loop temporaries scoring as lane-carried; the record does not
    # make that mistake, but it makes a NEW one: ``lane_loops`` is non-empty for the whole
    # device body once a lane loop has been registered, so a phi lowered anywhere inside it
    # is attributed to the lane loop even when its real carrier is an inner serial loop.
    #
    # ⇒ what the producer needs before the consumer can switch over is the loop that is
    # ACTUALLY carrying, not merely the innermost lane loop registered on the grid state --
    # i.e. the phi handler must know which loop node it is being emitted into.  That is a
    # larger change than "record at the phi", and shipping the swap without it trades a
    # measured-wrong scan for a measured-wrong record on the same kernel.
    #
    # The record is left in place and unconsulted deliberately: it is the evidence for the
    # above, and it is what a follow-up needs. ⛔ Do not "finish" this by deleting the scan
    # until the attribution question above is answered -- ``matmul_layernorm`` is the test.

    # Live-in names of the lane body (read before written), excluding the lane
    # var.  A loop-carried accumulator phi is live-in.
    written_so_far: set[str] = set()
    live_in: set[str] = set()
    for stmt in body:
        rw = ReadWrites.from_ast(stmt)
        for name in rw.reads:
            if name != lane_var and name not in written_so_far:
                live_in.add(name)
        written_so_far |= set(rw.writes)

    # Detect top-level phi copies ``X_copy = X`` of a live-in accumulator.
    for stmt in body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
        ):
            src = stmt.value.id
            dst = stmt.targets[0].id
            if src in live_in and _strip_ssa_suffix(dst) == _strip_ssa_suffix(src):
                return True
    return False


def _has_extra_cross_lane_carry(
    body: list[ast.AST], lane_var: str, marker_indices: set[int]
) -> bool:
    """Return True when ``body`` contains a loop-carried accumulator across the
    lanes that is INDEPENDENT of the reduction markers.

    Two-pass splitting is correct whenever every cross-lane carried value
    transitively consumes a marker result (e.g. an online-softmax
    ``mi = max(mi, local_amax)`` or ``di = di + sum``): after the split the
    carried update runs once per outer iteration on the fully-reduced scalar.
    But an *independent* carried accumulator — one that does not depend on any
    marker result, such as a matmul ``dot_acc += dot_product`` — must keep
    accumulating once per lane, so the split would drop or corrupt it. In that
    case the caller falls back to the single-pass per-lane behavior.

    A carried value is detected as a name that is *live-in* to the lane body
    (read before it is written within the body, directly or through a
    ``X_copy = X`` alias) and also written within the body.
    """
    from .ast_read_writes import ReadWrites

    written_so_far: set[str] = set()
    aliases: dict[str, str] = {}  # copy_var -> original carried name
    live_in: set[str] = set()
    for stmt in body:
        rw = ReadWrites.from_ast(stmt)
        for name in rw.reads:
            if name != lane_var and name not in written_so_far:
                live_in.add(name)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Name)
        ):
            aliases[stmt.targets[0].id] = stmt.value.id
        written_so_far |= set(rw.writes)

    def root(name: str) -> str:
        seen: set[str] = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return name

    # Names that (transitively) depend on a marker result. Carried values that
    # only depend on these are fine under the two-pass split.
    marker_results = {
        m.result_var
        for i, stmt in enumerate(body)
        if i in marker_indices
        and (m := _is_lane_reduce_marker_assign(stmt)) is not None
    }
    marker_tainted = set(marker_results)
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if set(rw.reads) & marker_tainted:
                for w in rw.writes:
                    if w not in marker_tainted:
                        marker_tainted.add(w)
                        changed = True

    # Names that (transitively) depend on the lane var. A carried accumulator
    # whose per-iteration update is lane-varying (e.g. a matmul
    # ``dot_acc += dot_product(lane)``) cannot move to the once-per-tile tail,
    # so the two-pass split would break it. A lane-invariant update (e.g.
    # welford's ``acc_cnt += block_size``) is fine in the tail.
    lane_varying = _lane_varying_names(body, lane_var)

    # Bail if some carried accumulator is independent of every marker AND its
    # update consumes a lane-varying value.
    for idx, stmt in enumerate(body):
        if idx in marker_indices:
            continue
        rw = ReadWrites.from_ast(stmt)
        reads = set(rw.reads)
        update_is_lane_varying = lane_var in reads or bool(reads & lane_varying)
        if not update_is_lane_varying:
            continue
        for w in rw.writes:
            if root(w) in live_in and root(w) not in marker_tainted:
                return True
    return False


def _matmul_already_folded_lane_axis(lane_var: str) -> bool:
    """Whether a matmul lowering already emitted a fold over ``lane_var``'s block.

    ⭐ **THIS IS THE CRITERION THAT SEPARATES THE TWO REGIMES** the decline in
    :func:`_decline_structural_lane_split` turns on.  It asks the one question that
    decides whether DELETING a lane-reduce marker's fold is correct: *has some other
    mechanism already reduced this marker's own axis?*

    ``_emit_cute_matmul_n_collapse`` (``cute/matmul_fallback.py``) is that other
    mechanism.  On a static-M==N-collapse ``baddbmm`` reduced over N it turns the
    shared M/N axis into a serial fold inside the matmul, so — its own words — "the
    downstream ``.sum(-1)``" is "a no-op".  It records the folded block id on the
    device-function state; this reads it back.

    ⭐ **WHY IT IS SOUND, and why it is a positive proof rather than a heuristic.**
    The producer asserts "I emitted a fold over this block", and it only does so after
    ``_cute_baddbmm_result_reduced_over_block`` has established that the block is
    genuinely reduced downstream and that the matmul result does not escape to any
    other consumer.  So a marker over that block is redundant BY CONSTRUCTION, and
    the per-lane restore reproduces the correct arithmetic rather than merely being
    "conservative".  Anything the producer has not vouched for keeps the ``raise``.

    ⛔ **THREE CHEAPER-LOOKING ROUTES ARE MEASURED-REFUTED.  DO NOT RE-DERIVE THEM.**
    All three fail on the same counterexample: ``addmm`` in a K loop then
    ``out[tile_m, :] = acc.sum(-1, keepdim=True)`` — a BROADCAST store, where the lane
    axis IS the reduced axis, so the restore is a silent wrong answer (MEASURED rel
    1.04 / 1.04 / 1.05 at N = 256 / 512 / 1024).  Against the ``baddbmm`` kernel,
    where the restore is CORRECT (maxabs 1.1e-05), it is indistinguishable by:

    * **the store signature** — both store a marker-only VALUE to a lane-varying
      ADDRESS.  (This is also why :func:`_lane_axis_wrongly_collapsed`, which is
      correct for gating the *stash*, must not be reused here: MEASURED, it says
      "restore" on the wrong-answer kernel too.)
    * **the block id** — both put the lane loop on ``block_index=2`` and reduce it.
    * **"does a fold exist"** — both contain a ``_cute_grouped_reduce``.  ⭐ This is
      why the brief's twice-warned "teach step 1 to recognise a grouped cross-warp
      fold" route was measured unsound: mere EXISTENCE of a fold does not
      discriminate.  The question has to be whether a fold over *this marker's own
      block*, licensed by the no-escape check, already happened.

    ⚠ Returns False with no live compilation.  These passes are pure AST functions
    and unit tests call them on hand-written bodies outside any ``DeviceFunction``
    context (see :func:`_tv_stage_read_by_frag`, which learned this the hard way).
    "Nobody recorded a fold" is the right answer for an absent compilation, not a
    crash.
    """
    from .device_function import DeviceFunction

    try:
        fn = DeviceFunction.current()
    except Exception:
        return False
    folded = getattr(
        getattr(fn, "cute_state", None), "matmul_folded_reduction_block_ids", None
    )
    if not folded:
        return False
    # ⭐ The lane loop's block id comes from the STRATEGY that owns the lane var, not
    # from parsing the var's name.  The name happens to end in the block index today
    # (``synthetic_lane_2``), but that is a naming convention, not an interface — and
    # ``fn.new_var`` is free to uniquify it.  The strategy holds the real answer.
    strategies = getattr(getattr(fn, "tile_strategy", None), "strategies", None) or ()
    env = CompileEnvironment.current()
    canonical = getattr(env, "canonical_block_id", lambda block_id: block_id)
    for strategy in strategies:
        if getattr(strategy, "_synthetic_cute_lane_var", None) != lane_var:
            continue
        block_index = getattr(strategy, "block_index", None)
        if isinstance(block_index, int) and canonical(block_index) in folded:
            return True
    return False


def _decline_structural_lane_split(
    loop: ast.For,
    lane_var: str,
    markers: list[tuple[int, _LaneReduceMarker]],
    *,
    reason: str,
) -> list[ast.AST]:
    """The SINGLE decision every structural-lane-split decline goes through.

    Each caller in :func:`_split_one_lane_loop` has established only that *its own*
    lowering cannot express this loop.  That is a statement about the lowering, and
    the three callers each learned it a different way.  It is **not** a statement
    about what is correct to emit instead, and conflating the two is what made this
    area produce silent wrong answers twice.

    So the decline is centralised here and asks the one question that actually
    decides legality, in a fixed order:

    1. **Does something else already fold the lane axis?**  If the lane loop carries
       an accumulator that consumes a marker result
       (:func:`_markers_feed_cross_lane_carry`), the per-lane restore is a genuine
       no-op: ``examples/attention.py``'s ``m_i``/``l_i``/``acc`` recurrence folds
       each per-lane reduced value on every lane iteration, so the lane axis IS
       combined, just not by the marker.  MEASURED: all eight attention examples
       reach a decline with this true on every compile, and raising unconditionally
       broke all eight.
    2. **Can the register stash express it?**  The stash runs the collective once
       and re-derives every reduction from a per-thread fragment, which is a
       *correct* cross-lane fold rather than a deleted one.  Two of the three
       callers never tried it, and the site the stash is tried before could still
       arrive here after the stash refused for an unrelated reason.  It is asked
       here with its extent cap LIFTED (``extent_cap=None``): that cap is a
       register-cost preference, and a lowering that spills is strictly better than
       one that returns wrong numbers or fails to compile.  MEASURED correct at the
       extents that reach here -- rel 5.7e-07 / 6.1e-07 / 5.5e-07 at 512 / 1024 /
       2048, emitting real ``_lane_acc`` folds and ``make_fragment(extent)`` stashes.
    2.5. **Did a matmul lowering already fold this marker's own axis?**  ⭐ If
       :func:`_matmul_already_folded_lane_axis` says yes, the marker's fold is
       redundant BY CONSTRUCTION and the per-lane restore is correct — see that
       function for why this is a positive proof and for the three cheaper-looking
       routes that are measured-refuted.  MEASURED: this is what
       ``test_indexing.py::test_full_slice_in_reduction_loop`` needs (maxabs 1.1e-05
       at ``block_sizes`` ``[16,16]`` / ``[16,8]`` / ``[8,16]`` / ``[16,4]``), and it
       does NOT admit the ``addmm`` + broadcast-store counterexample, which keeps
       raising.

       ⚠ It is asked AFTER the stash, not before: where the stash applies it is a
       real cross-lane fold and strictly more general, so it stays preferred.

    3. **Otherwise REFUSE.**  There is no correct emission, and a compile error is
       the only honest answer.

       ⛔ **AND THE REFUSAL IS LOAD-BEARING — DO NOT "SIMPLIFY" IT TO A RESTORE.**
       MEASURED, on ``addmm`` in a K loop followed by
       ``out[tile_m, :] = acc.sum(-1, keepdim=True)``: step 3 is reached (at every N
       tried, including 256 — so this is not the stash's extent cap) and the per-lane
       restore returns rel **1.04 / 1.04 / 1.05** at N = 256 / 512 / 1024, every lane
       writing its own partial sum.  A blanket restore here would trade this refusal
       for a silent wrong answer on a plain matmul + row-sum kernel.  Pinned by
       ``test/test_cute_lane_split_decline.py``.

    ⛔ **WHY THIS FUNCTION EXISTS AT ALL -- the bug it fixes was LIVE.**  Two of the
    three sites used to ``return [_restore_per_lane_markers(...)]`` unconditionally.
    ``examples/matmul_layernorm.py`` at ``block_sizes=[32,32]``, M=K=64, fp32 hit
    one of them for every ``N >= 512`` and returned WRONG NUMBERS, silently:

        N=128  rel 2.98e-07  (register stash absorbed it; extent 128 <= its cap)
        N=256  rel 3.31e-07  (same)
        N=512  rel **7.685**   row std 23.83 where layer-normed rows must be 1.51
        N=1024 rel **11.37**   row std 32.65
        (minimal one-marker form ``(x @ y).sum(-1)``: rel 1.009 / 0.9999)

    The emitted kernel contained ``sum_1 = cutlass.Float32(acc)`` -- the marker's
    fold DELETED -- and zero ``_lane_acc``.  The reachability threshold is the
    register stash's ``extent > 256`` refusal: below it the stash rescues the shape,
    above it control fell through to the restore.

    ⚠ **AND THE PREDICATE THAT SITE GATED ON CANNOT ANSWER THIS QUESTION.**  It
    consulted :func:`_has_extra_cross_lane_carry`, which returned True there -- but
    TRACED, all ten of its hits come from ONE statement: the nested matmul K loop.
    ``ReadWrites.from_ast`` flattens a nested loop into a single statement's
    reads+writes, so K-loop temporaries (``load``, ``dot_reduce_result``,
    ``acc_copy_0``, ...) are read-before-written *within that flattened view* and
    score as ``live_in``, i.e. as cross-lane carried accumulators.  The scan is
    flow-insensitive and cannot distinguish "carried across the lane iterations"
    from "written inside a nested loop", which is precisely the distinction this
    decline turns on.  (``aliases`` was empty at that invocation, so the
    ``X_copy = X`` ``root()`` machinery was not even involved -- do not re-derive
    the older "a K-loop phi is mistaken for a lane-loop phi" story from it.)

    That is why step 1 uses :func:`_markers_feed_cross_lane_carry` -- which asks
    whether a carry *consumes a marker result* -- and why a "cannot split" signal is
    never accepted as a licence to restore.
    """
    body = list(loop.body)
    if _markers_feed_cross_lane_carry(body, lane_var, markers):
        return [_restore_per_lane_markers(loop, markers)]
    stashed = _split_lane_loop_with_register_stash(
        loop, lane_var, markers, extent_cap=None
    )
    if stashed is not None:
        return stashed
    # Step 2.5 — see the docstring.  A matmul lowering that already folded this
    # marker's own axis makes the marker redundant, so the restore is correct here.
    if _matmul_already_folded_lane_axis(lane_var):
        return [_restore_per_lane_markers(loop, markers)]
    raise exc.BackendUnsupported(
        CompileEnvironment.current().backend.name,
        f"{reason}, with no loop-carried accumulator folding the lane axis. There "
        "is no correct lowering for this shape yet: the register-stash path "
        "declined it and no structural split applies, and the per-lane fallback "
        "would DELETE the reduction's cross-lane fold (a silent wrong answer -- "
        "measured rel 7.7 on matmul_layernorm before this refusal existed). Reduce "
        "the lane extent so the register stash applies, or split the kernel so each "
        "reduction reads the collective's output from memory.",
    )


def _restore_per_lane_markers(
    loop: ast.For, markers: list[tuple[int, _LaneReduceMarker]]
) -> ast.For:
    """Replace each ``_helion_lane_reduce`` marker in ``loop`` with its raw
    per-lane input, restoring the original single-pass behavior (used when the
    two-pass split is unsafe).

    **This is only semantically valid where the lane loop is not distributing the
    reduction axis.**  It drops the cross-lane accumulator, so it is correct only
    when some *other* mechanism already combines the lanes — in practice when the
    lane loop carries the reduction result forward itself (online-softmax
    attention's ``mi``/``di`` recurrence: the per-lane reduced value is folded into
    a carried accumulator on every lane iteration, so the lane axis IS combined,
    just not here).

    When a ``PersistentReductionStrategy`` synthesized the lane loop *for* this
    reduction, reverting is not conservative — it is a wrong answer, and a silent
    one (bug class 8 P1: the emitted kernel contained zero ``warp_reduction`` and
    every output was one element of the row).

    ⛔ **So this is NOT a general-purpose fallback.**  Every caller in
    :func:`_split_one_lane_loop` has a positive argument for why the lane axis is
    combined by something *else* — see the per-site comments.  The one site that
    had no such argument (sequentially-dependent markers whose layered split
    declined) raises ``BackendUnsupported`` instead of calling this.  Do not add a
    caller without stating which mechanism folds the lane axis.
    """
    body = list(loop.body)
    for idx, m in markers:
        body[idx] = statement_from_string(
            f"{m.result_var} = {m.finalize_expr(m.input_name)}"
        )
    loop.body = body
    return loop


_UNDUPLICATABLE_CALLS = (
    "_cute_grouped_reduce_shared_two_stage",
    "_cute_grouped_reduce_shared_tree",
    "_cute_grouped_reduce_warp",
    "cute.gemm",
    "warp_reduction",
)


def _contains_unduplicatable_op(stmt: ast.AST) -> bool:
    """Return True when ``stmt`` (or any nested statement) contains a matmul /
    collective whose shared-memory side effects make it unsafe to re-run in a
    second lane pass."""
    src = ast.unparse(stmt)
    return any(call in src for call in _UNDUPLICATABLE_CALLS)


def _has_side_effect(stmt: ast.AST) -> bool:
    """Return True when ``stmt`` produces an observable side effect (a store,
    an in-place / atomic write, or any non-plain-assignment statement such as
    an ``if mask: tensor[...].store(...)``)."""
    from .ast_read_writes import ReadWrites

    if isinstance(stmt, ast.Assign):
        return bool(ReadWrites.from_ast(stmt).inplace_writes)
    # Conservatively treat structured / expression statements as
    # side-effecting (store calls live inside ``if`` blocks / bare exprs).
    return True


def _live_phase2_indices(body: list[ast.AST]) -> set[int]:
    """Indices of statements in ``body`` that contribute to a side effect
    (directly, or by feeding a later side-effecting statement).

    ⭐⭐ WHY THIS LIVES HERE AND NOT IN A DEAD-CODE PASS -- the question was asked
    (``09_LANE_SPLIT_ENDGAME.md`` §4b: *"it is not a lane-split concern … the implementer
    should investigate where it belongs: its own pass, or fused into an existing one"*)
    and the answer is MEASURED: **neither**.

    ⛔ ``DeviceFunction.dead_code_elimination`` CANNOT SUBSUME IT, and no widening of
    ``dce_vars`` changes that.  Its rule is ``name in rw.writes and name not in rw.reads``
    over a read set flattened across the WHOLE function.  The statements this drops are the
    *phase-2 copies* of statements the *phase-1 copy* legitimately reads, so the name IS
    read and the rule can never fire::

        for lane: v_11 = w*g; v_12 = v_11*x; acc += v_12    # phase 1 -- v_12 IS read
        s = warp_reduce(acc)
        for lane: v_11 = w*g; v_12 = v_11*x; out.store(s*2) # phase 2 -- v_12 dead

    MEASURED with a MAXIMAL allow-list (``dce_vars`` widened from 19 names to every name
    the body writes, 19 -> 50 on ``rms_norm_bwd``): ``changed=False``, and the split-pass
    drop is not recovered on any arm.  Two further reasons: the general pass deletes
    single-target ``Assign`` nodes only, so it can never remove the whole extra
    ``for synthetic_lane`` nest that keeping one dead lane-varying statement forces; and
    ``dce_vars`` contains zero ``v_*`` temporaries in the first place.

    ⭐ THE CAPABILITY IS NOT "DCE".  It is: *given one ordered statement list about to be
    duplicated into N passes, which occurrences in THIS copy feed a side effect?*  That is
    a per-list, side-effect-rooted question, and it needs to know the pass structure -- so
    a standalone AST pass could only answer it by re-implementing this function verbatim.
    ⇒ its correct home is beside the code that DECIDES the duplication, which is exactly
    where its three callers are: :func:`_split_one_lane_loop`,
    :func:`_split_lane_loop_multi_stage`, and
    :func:`_split_lane_loop_with_register_stash`.

    ⚠ AND IT IS UPSTREAM CODE, byte-identical to ``a1e9642e5`` (as is
    :func:`_has_side_effect`).  The branch added only a *caller*
    (:func:`_lane_reduce_consume_tail`).  So there was never anything of this branch's to
    move out of here -- which is the part §4b did not know.

    **What it buys, measured** by neutering only this function (returning every index):
    ``cross_entropy`` loop-free/TV goes 91 -> 107 lines, 2 -> 3 lane nests, 2 -> 3
    ``cute.copy`` -- one extra row read and one extra nest, same answer.  Perf, not
    correctness, and invisible to every numeric check: exactly why it is pinned in prose
    here rather than trusted to a reviewer's memory.
    """
    from .ast_read_writes import ReadWrites

    needed_names: set[str] = set()
    keep: set[int] = set()
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        rw = ReadWrites.from_ast(stmt)
        writes = set(rw.writes)
        if _has_side_effect(stmt) or (writes & needed_names):
            keep.add(idx)
            needed_names |= set(rw.reads)
    return keep


def _lane_loop_extent(loop: ast.For) -> int:
    call = loop.iter
    assert isinstance(call, ast.Call)
    assert len(call.args) == 1
    return int(ast.literal_eval(call.args[0]))


def _lane_shared_producer_prelude(
    produced_inside: list[ast.AST], segment: list[ast.AST]
) -> list[ast.AST]:
    """The statements from earlier lane segments that ``segment`` still needs.

    ``produced_inside`` is every statement earlier segments emitted INSIDE a lane loop, in
    order.  A name written there is not in scope for a sibling loop, so any of it that
    ``segment`` (transitively) reads has to be re-emitted into ``segment``'s own loop.

    ⭐ THE SLICE IS TRANSITIVE AND RUNS BACKWARDS for the same reason
    :func:`_backward_slice` does: on the P1 shape, ``segment`` reads ``v_0``, whose
    producer reads ``load``, whose producer reads ``idx`` -- pulling only the direct
    producer leaves the next name free.  MEASURED as a DSL "cannot access free variable"
    when this was one level deep.

    ⚠ A ``cute.copy`` counts as writing its destination, which the plain ``ReadWrites`` view
    cannot see -- so the walk is delegated to :func:`_tv_backward_slice`, the existing helper
    for exactly this question.  ⭐ That function had ZERO callers before this; reusing it
    instead of writing a second near-identical walk is deliberate (two walks answering one
    dependence question is how they drift apart).
    """
    from .ast_read_writes import ReadWrites

    if not produced_inside:
        return []
    roots: set[str] = set()
    for stmt in segment:
        roots |= set(ReadWrites.from_ast(stmt).reads)
    return [produced_inside[i] for i in _tv_backward_slice(produced_inside, roots)]


def _flatten_lane_nest_stmts(stmt: ast.AST) -> list[ast.AST]:
    """Every statement in ``stmt``, including those nested inside loops.

    Used by :meth:`DeviceGridState._emit_lane_segment` to put a prebuilt nest's OWN
    statements into the lane-varying analysis.  The nest's ``lane_base = ... + lane *
    stride`` assignment lives inside its lane loop, so a flat scan of the top level
    would miss the very statement that links the lane var to the body -- and a body
    whose dependence on the lane var is invisible is hoisted out of the loop entirely.
    """
    out: list[ast.AST] = [stmt]
    for field in ("body", "orelse", "finalbody"):
        inner = getattr(stmt, field, None)
        if isinstance(inner, list):
            for child in inner:
                if isinstance(child, ast.AST):
                    out.extend(_flatten_lane_nest_stmts(child))
    return out


def _lane_varying_names(body: list[ast.AST], lane_var: str) -> set[str]:
    """Names whose values (transitively) depend on the lane var within ``body``."""
    from .ast_read_writes import ReadWrites

    varying = {lane_var}
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            # ⭐ A ``cute.copy``'s DESTINATION is an argument, so ``ReadWrites`` reports it
            # as a read and reports no write at all.  Counting it as a write is what makes
            # a fragment filled inside the lane loop read as lane-VARYING downstream.
            #
            # ⛔ MEASURED without this, on the persistent TV shape: the copies write
            # ``_tv_frag_K`` per lane iteration, the body reads
            # ``_tv_frag_K[reduction_vec_lane_N]`` (naming only the VEC var), so no chain
            # reached the body, every statement classified lane-invariant, and
            # ``_emit_lane_segment`` hoisted the ENTIRE body out of the loop -- 18
            # statements at the enclosing scope with no lane loop, fragments read before
            # they were filled.
            #
            # ⚠ ``is_varying`` in :meth:`DeviceGridState._emit_lane_segment` already takes
            # exactly this union for exactly this reason; the two views must agree, or the
            # transitive closure and the per-statement test disagree about the same copy.
            writes = set(rw.writes) | _tv_copy_dest_names(stmt)
            if set(rw.reads) & varying:
                for w in writes:
                    if w not in varying:
                        varying.add(w)
                        changed = True
    varying.discard(lane_var)
    return varying


def _lane_body_live_in(body: list[ast.AST], lane_var: str) -> set[str]:
    """Names read in ``body`` before they are written (live-in), excluding the
    lane var.  A loop-carried accumulator phi is live-in to the lane body."""
    from .ast_read_writes import ReadWrites

    written: set[str] = set()
    live_in: set[str] = set()
    for stmt in body:
        rw = ReadWrites.from_ast(stmt)
        for name in rw.reads:
            if name != lane_var and name not in written:
                live_in.add(name)
        written |= set(rw.writes)
    return live_in


def _is_serial_for(stmt: ast.AST) -> bool:
    """Return True when ``stmt`` is an ordinary serial ``for`` loop (a device
    serial loop), NOT a per-thread lane loop.

    ⚠ A ``cutlass.range_constexpr`` V-LOOP IS NOT A SERIAL LOOP EITHER, and excluding
    it is a correctness fix rather than a tidy-up.  ``interchange_lane_outside_serial_reductions``
    uses this predicate to find the ``mb`` loop of a ``for LANE: ... for MB: ...``
    nest and then REWRITES that nest, reverting the markers in one of the two copies
    it emits.  The TV protocol's constexpr V-loop matches the old spelling of this
    predicate (it is an ``ast.For`` carrying no lane-loop attribute), so a TV-planned
    persistent reduction was interchanged as if its fragment unroll were a serial
    device loop -- MEASURED: the pass returned 2 statements, and the marker was
    reverted to its raw per-lane input BEFORE ``split_lane_loop_reductions`` ever
    ran, so the emitted kernel read ``sum_1 = cutlass.Float32(v_0)`` with no fold at
    all.  The two loop kinds are genuinely different: an ``mb`` loop iterates a
    DEVICE-side extent and its trip count is a runtime bound, while a
    ``range_constexpr`` loop is a compile-time unroll over ONE register fragment's
    elements and carries no cross-iteration device state to interchange.
    """
    return (
        isinstance(stmt, ast.For)
        and getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None) is None
        and _constexpr_vec_loop(stmt) is None
    )


def _clone_stmt(stmt: ast.AST) -> ast.AST:
    """Return an independent copy of ``stmt`` via unparse + reparse.

    ``interchange_lane_outside_serial_reductions`` emits two loop nests that
    both re-run the shared (side-effect-free) producers. Splicing the same node
    objects into two places in the tree breaks AST walking, so each reused
    statement is rebuilt from its source text into a fresh ExtendedAST node.
    """
    return statement_from_string(ast.unparse(stmt))


def _clone_expr(node: ast.AST) -> ast.AST:
    return expr_from_string(ast.unparse(node))


def _forward_live_names(body: list[ast.AST], roots: set[str]) -> set[str]:
    """Names produced by the forward slice that (transitively) consumes any
    name in ``roots`` within ``body``."""
    from .ast_read_writes import ReadWrites

    tainted = set(roots)
    changed = True
    while changed:
        changed = False
        for stmt in body:
            rw = ReadWrites.from_ast(stmt)
            if set(rw.reads) & tainted:
                for w in rw.writes:
                    if w not in tainted:
                        tainted.add(w)
                        changed = True
    return tainted


def interchange_lane_outside_serial_reductions(
    body: list[ast.AST],
) -> list[ast.AST]:
    """Interchange a ``for LANE: ... for MB: ...`` nest whose inner serial loop
    contains ``_helion_lane_reduce`` markers.

    layer_norm_bwd / rms_norm_bwd compute, inside a serial ``mb`` loop, BOTH a
    per-feature accumulator that must keep the lane loop OUTSIDE the ``mb`` loop
    (``grad_w_acc += ...``) AND a feature reduction whose result is broadcast
    back into a per-row store (``grad_x``), which needs every lane summed *per*
    ``mb`` iteration (lane INSIDE ``mb``). A single lane loop cannot satisfy
    both nestings, so emit two specialized loop nests:

    * Nest B (grad_w): the original ``for LANE: ... for MB: ...`` loop with the
      lane-reduce markers and the reduction-consuming side effects removed —
      keeping only the per-feature accumulators and their stores.
    * Nest A (grad_x): a ``for MB: ... for LANE: ...`` loop carrying only the
      lane reduction and its broadcast consumer. Its inner lane loop still holds
      the markers so the subsequent ``split_lane_loop_reductions`` pass produces
      the per-``mb`` accumulate -> warp-combine -> consume structure.

    Returns ``body`` unchanged when no such pattern is present.
    """
    new_body: list[ast.AST] = []
    for stmt in body:
        new_body.extend(_interchange_stmt(stmt))
    return new_body


def _interchange_stmt(stmt: ast.AST) -> list[ast.AST]:
    for field in ("body", "orelse", "finalbody"):
        old = getattr(stmt, field, None)
        if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
            setattr(stmt, field, interchange_lane_outside_serial_reductions(old))
    lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
    if (
        lane_var is None
        or not isinstance(stmt, ast.For)
        or not isinstance(stmt.target, ast.Name)
        or stmt.target.id != lane_var
    ):
        return [stmt]
    return _interchange_one_lane_loop(stmt, lane_var)


def _interchange_one_lane_loop(loop: ast.For, lane_var: str) -> list[ast.AST]:
    from .ast_read_writes import ReadWrites

    body: list[ast.AST] = list(loop.body)
    # Find the single inner serial ``for`` loop that carries lane-reduce markers.
    mb_index: int | None = None
    for idx, stmt in enumerate(body):
        if _is_serial_for(stmt) and any(
            _is_lane_reduce_marker_assign(s) is not None
            for s in cast("ast.For", stmt).body
        ):
            if mb_index is not None:
                # More than one candidate serial loop: not the simple pattern.
                return [loop]
            mb_index = idx
    if mb_index is None:
        return [loop]

    mb_loop = cast("ast.For", body[mb_index])
    lane_prefix = body[:mb_index]
    lane_suffix = body[mb_index + 1 :]
    mb_body: list[ast.AST] = list(mb_loop.body)

    markers = [
        (i, m)
        for i, s in enumerate(mb_body)
        if (m := _is_lane_reduce_marker_assign(s)) is not None
    ]
    if not markers:
        return [loop]
    marker_indices = {i for i, _ in markers}
    marker_results = {m.result_var for _, m in markers}

    # The mb body's only side effect that consumes a marker result is the
    # broadcast-reduction store (e.g. ``grad_x[mb] = ...``). Everything else in
    # the mb body / suffix (the per-feature accumulators and their stores) is
    # independent of the reduction and is handled correctly by Nest B alone.
    grad_x_seed = {m.input_name for _, m in markers} | marker_results
    grad_x_live = _forward_live_names(mb_body, grad_x_seed)

    def is_reduction_store(stmt: ast.AST) -> bool:
        return _has_side_effect(stmt) and bool(
            set(ReadWrites.from_ast(stmt).reads) & grad_x_live
        )

    if not any(is_reduction_store(s) for s in mb_body):
        # The lane reduction inside the serial loop is not consumed by a
        # broadcast store, so the interchange does not apply. Markers nested in
        # a serial loop are not reachable by ``split_lane_loop_reductions`` (it
        # only rewrites top-level lane loops), so restore them to their raw
        # per-lane inputs to avoid leaving an unprocessed marker behind.
        restored: list[ast.stmt] = [cast("ast.stmt", s) for s in mb_body]
        for idx, m in markers:
            restored[idx] = statement_from_string(
                f"{m.result_var} = {m.finalize_expr(m.input_name)}"
            )
        mb_loop.body = restored
        return [loop]

    # A name is lane-varying if it (transitively) depends on the lane var across
    # the whole lane-loop body; lane-invariant names (mb bounds, masks, ...) are
    # recomputed once rather than per lane.
    lane_varying = _lane_varying_names([*lane_prefix, *mb_body, *lane_suffix], lane_var)

    def reads_lane(stmt: ast.AST) -> bool:
        reads = set(ReadWrites.from_ast(stmt).reads)
        return lane_var in reads or bool(reads & lane_varying)

    # --- Nest A (grad_x): for MB: for LANE: <reduction + broadcast store> ------
    # Backward slice from the reduction-broadcast stores and the marker inputs,
    # across both the mb body and the lane prefix. The per-feature accumulators
    # feed only the suffix stores (never the reduction store), so they are
    # naturally excluded — no carry analysis is required.
    mb_bound_names = set(ReadWrites.from_ast(mb_loop.iter).reads)
    needed = {m.input_name for _, m in markers} | mb_bound_names
    keep_mb_a: list[ast.AST] = []
    for idx in range(len(mb_body) - 1, -1, -1):
        stmt = mb_body[idx]
        rw = ReadWrites.from_ast(stmt)
        if idx in marker_indices:
            keep_mb_a.append(stmt)
            needed |= set(rw.reads) - marker_results
            continue
        if is_reduction_store(stmt) or (set(rw.writes) & needed):
            keep_mb_a.append(stmt)
            needed |= set(rw.reads)
    keep_mb_a.reverse()
    keep_prefix_a: list[ast.AST] = []
    for stmt in reversed(lane_prefix):
        rw = ReadWrites.from_ast(stmt)
        if set(rw.writes) & needed:
            keep_prefix_a.append(stmt)
            needed |= set(rw.reads)
    keep_prefix_a.reverse()

    # Partition kept statements into lane-invariant (run once per mb iteration)
    # vs lane-varying (recomputed per lane inside the inner lane loop).
    prefix_invariant_a = [_clone_stmt(s) for s in keep_prefix_a if not reads_lane(s)]
    prefix_varying_a = [_clone_stmt(s) for s in keep_prefix_a if reads_lane(s)]
    mb_head_a = [_clone_stmt(s) for s in keep_mb_a if not reads_lane(s)]
    mb_varying_a = [_clone_stmt(s) for s in keep_mb_a if reads_lane(s)]

    extent = _lane_loop_extent(loop)
    inner_lane_loop_a = _create_lane_loop(
        lane_var, extent, [*prefix_varying_a, *mb_varying_a]
    )
    mb_loop_a = create(
        ast.For,
        target=_clone_expr(mb_loop.target),
        iter=_clone_expr(mb_loop.iter),
        body=[*mb_head_a, inner_lane_loop_a],
        orelse=[],
        type_comment=None,
    )
    nest_a: list[ast.AST] = [*prefix_invariant_a, mb_loop_a]

    # --- Nest B (grad_w): the original lane loop with markers reverted to their
    # raw per-lane inputs. Its per-feature accumulators (lane-outside-mb) are
    # already correct; its reduction-broadcast store writes a partial (per-lane)
    # value that Nest A re-stores with the full reduction afterwards.
    restored_mb_body = list(mb_loop.body)
    for idx, m in markers:
        restored_mb_body[idx] = statement_from_string(
            f"{m.result_var} = {m.finalize_expr(m.input_name)}"
        )
    mb_loop.body = restored_mb_body
    nest_b = loop

    return [nest_b, *nest_a]


# ---------------------------------------------------------------------------
# Chunked-recurrence (GDN) lane-invariant accumulator hoist.
#
# A chunked recurrence such as gdn_fwd_h carries an accumulator ``b_h`` across a
# SERIAL chunk loop and, inside each chunk, contracts a matmul over the
# within-chunk position ``c`` (lowered as an inner lane loop).  ``matmul_fallback``
# emits the running-sum ``dot_acc`` form for this matmul (because the per-chunk
# rescale is lane-invariant), producing inside the lane loop:
#
#       dot_acc_base = <lane-invariant rescale of b_h>      # e.g. b_h * decay
#       dot_acc = dot_acc + <product(c)>                    # accumulate over c
#       b_h = dot_acc_base + dot_acc                        # WRONG: per-lane reassign
#
# with ``dot_acc = <identity>`` reset OUTSIDE the chunk loop.  Reassigning ``b_h``
# every lane iteration corrupts the recurrence (and any lane-invariant op that
# must read the chunk-ENTRY ``b_h``, e.g. the store of ``b_h`` and the matmul
# operand ``c_h = b_h``).  This pass restructures the nest to apply the rescale
# and the final add once per chunk:
#
#   for chunk:
#       dot_acc = <identity>                  # reset per chunk
#       <lane-invariant chunk-entry stores using b_h>
#       dot_acc_base = <rescale of frozen b_h>
#       for lane:
#           <producers; dot_acc = dot_acc + product(c)>
#       b_h = dot_acc_base + dot_acc          # once per chunk
# ---------------------------------------------------------------------------


def _find_dot_acc_recurrence(
    lane_loop: ast.For, lane_var: str
) -> tuple[str, str, str] | None:
    """If ``lane_loop`` ends with the chunked-recurrence ``dot_acc`` triple,
    return ``(acc_var, dot_acc_base_var, dot_acc_var)``; else ``None``.

    The triple (emitted by ``_emit_cute_matmul``'s lane-invariant ``dot_acc``
    path) is, as the LAST three statements of the lane-loop body:

        dot_acc_base = <expr>             # lane-invariant rescale of acc
        dot_acc = dot_acc + <product>    # running sum over the lane axis
        acc = dot_acc_base + dot_acc     # final combine
    """
    body = lane_loop.body
    if len(body) < 3:
        return None
    base_stmt, sum_stmt, final_stmt = body[-3], body[-2], body[-1]

    def assign_name(stmt: ast.AST) -> str | None:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            return stmt.targets[0].id
        return None

    base_var = assign_name(base_stmt)
    dot_acc_var = assign_name(sum_stmt)
    acc_var = assign_name(final_stmt)
    if base_var is None or dot_acc_var is None or acc_var is None:
        return None
    if not (base_var.startswith("dot_acc_base") and dot_acc_var.startswith("dot_acc")):
        return None
    # ``dot_acc = dot_acc + product``: a running self-sum over the lane axis.
    assert isinstance(sum_stmt, ast.Assign)
    if not (
        isinstance(sum_stmt.value, ast.BinOp)
        and isinstance(sum_stmt.value.op, ast.Add)
        and isinstance(sum_stmt.value.left, ast.Name)
        and sum_stmt.value.left.id == dot_acc_var
    ):
        return None
    # ``acc = dot_acc_base + dot_acc``: the final per-chunk combine.
    assert isinstance(final_stmt, ast.Assign)
    final_reads = {n.id for n in ast.walk(final_stmt.value) if isinstance(n, ast.Name)}
    if final_reads != {base_var, dot_acc_var}:
        return None
    # The rescale (``dot_acc_base``) must be lane-INVARIANT: it must not read the
    # lane var nor any value derived from it within the lane body.
    lane_body: list[ast.AST] = list(body)
    lane_varying = _lane_varying_names(lane_body, lane_var)
    from .ast_read_writes import ReadWrites

    base_reads = set(ReadWrites.from_ast(base_stmt).reads)
    if lane_var in base_reads or (base_reads & lane_varying):
        return None
    return acc_var, base_var, dot_acc_var


def _single_lane_loop_in_body(
    body: list[ast.AST],
) -> tuple[int, ast.For, str] | None:
    """If ``body`` contains exactly one direct lane loop, return its index, the
    loop node, and its lane var; else ``None``."""
    found: tuple[int, ast.For, str] | None = None
    for idx, stmt in enumerate(body):
        lane_var = getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
        if (
            lane_var is not None
            and isinstance(stmt, ast.For)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == lane_var
        ):
            if found is not None:
                return None
            found = (idx, stmt, lane_var)
    return found


def _find_reset_assign(body: list[ast.AST], var: str) -> int | None:
    """Index of the last ``var = <expr>`` plain assignment in ``body`` (the
    ``dot_acc`` reset emitted before the chunk loop), or ``None``."""
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == var
        ):
            return idx
    return None


def hoist_lane_invariant_chunk_recurrence(
    body: list[ast.AST],
) -> list[ast.AST]:
    """Restructure ``for chunk: for lane: <dot_acc recurrence>`` nests so the
    lane-invariant rescale, chunk-entry stores, and final accumulator combine
    run once per chunk (see the module comment above).

    Tightly gated: only fires on a serial chunk loop whose single inner lane
    loop ends with the ``dot_acc`` triple and whose ``dot_acc`` reset sits in
    the same statement list before the chunk loop.
    """
    new_body: list[ast.AST] = []
    for stmt in body:
        # Recurse into nested statement-bearing fields first.
        for field in ("body", "orelse", "finalbody"):
            old = getattr(stmt, field, None)
            if isinstance(old, list) and all(isinstance(s, ast.stmt) for s in old):
                setattr(stmt, field, hoist_lane_invariant_chunk_recurrence(old))

        info = _detect_chunk_recurrence(stmt)
        if info is None:
            new_body.append(stmt)
            continue
        dot_acc_var = info[0]
        reset_idx = _find_reset_assign(new_body, dot_acc_var)
        if reset_idx is None:
            # No relocatable reset found: bail out (leave nest unchanged).
            new_body.append(stmt)
            continue
        reset_stmt = new_body.pop(reset_idx)
        assert isinstance(stmt, ast.For)
        new_body.append(_rewrite_chunk_recurrence(stmt, info, reset_stmt))
    return new_body


def _detect_chunk_recurrence(
    stmt: ast.AST,
) -> tuple[str, int, ast.For, str, str, str, str] | None:
    """Return ``(dot_acc_var, lane_idx, lane_loop, lane_var, acc_var, base_var,
    dot_acc_var)`` when ``stmt`` is a serial chunk loop carrying the ``dot_acc``
    recurrence, else ``None``."""
    if not _is_serial_for(stmt) or not isinstance(stmt, ast.For):
        return None
    found = _single_lane_loop_in_body(list(stmt.body))
    if found is None:
        return None
    lane_idx, lane_loop, lane_var = found
    triple = _find_dot_acc_recurrence(lane_loop, lane_var)
    if triple is None:
        return None
    acc_var, base_var, dot_acc_var = triple
    return dot_acc_var, lane_idx, lane_loop, lane_var, acc_var, base_var, dot_acc_var


def _rewrite_chunk_recurrence(
    stmt: ast.For,
    info: tuple[str, int, ast.For, str, str, str, str],
    reset_stmt: ast.AST,
) -> ast.For:
    """Build the restructured chunk loop (see module comment)."""
    from .ast_read_writes import ReadWrites

    _dot_acc, lane_idx, lane_loop, lane_var, acc_var, _base_var, _dot_acc2 = info
    chunk_body: list[ast.AST] = list(stmt.body)
    lane_body: list[ast.AST] = list(lane_loop.body)
    base_stmt = lane_body[-3]
    sum_stmt = lane_body[-2]
    final_stmt = lane_body[-1]
    producers: list[ast.AST] = lane_body[:-3]

    lane_varying = _lane_varying_names(lane_body, lane_var)

    def reads_lane(s: ast.AST) -> bool:
        reads = set(ReadWrites.from_ast(s).reads)
        return lane_var in reads or bool(reads & lane_varying)

    # The "accumulator family": the names that all hold the chunk-ENTRY
    # accumulator value.  Helion may capture the loop-carried phi through a chain
    # of plain copy-aliases (``b_h_copy = b_h``; ``b_h_copy_0 = b_h_copy``) and
    # the rescale / chunk-entry store read those copies, not ``acc_var`` (which
    # is the chunk-EXIT result).  Build the copy-alias closure over the chunk
    # prefix and the lane producers, seeded from the loop-carried phi: a name
    # that is live-in to the lane body (read before written) and feeds the
    # lane-invariant rescale ``base_stmt``.
    chunk_prefix = chunk_body[:lane_idx]
    copy_src: dict[str, str] = {}
    for s in (*chunk_prefix, *producers):
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and isinstance(s.value, ast.Name)
        ):
            copy_src[s.targets[0].id] = s.value.id

    def copy_root(name: str) -> str:
        seen: set[str] = set()
        while name in copy_src and name not in seen:
            seen.add(name)
            name = copy_src[name]
        return name

    base_slice_indices, _ = _backward_slice(
        producers, set(ReadWrites.from_ast(base_stmt).reads)
    )
    base_slice_reads = set(ReadWrites.from_ast(base_stmt).reads)
    for i in base_slice_indices:
        base_slice_reads |= set(ReadWrites.from_ast(producers[i]).reads)
    lane_live_in = _lane_body_live_in(producers, lane_var)
    phi_roots = {
        copy_root(name) for name in base_slice_reads if copy_root(name) in lane_live_in
    }
    acc_family = set(phi_roots)
    for name in (*copy_src.keys(),):
        if copy_root(name) in phi_roots:
            acc_family.add(name)

    # Backward slice feeding the lane-invariant rescale ``base_stmt``: those
    # producers are lane-invariant and hoist before the lane loop with it.
    base_seed = set(ReadWrites.from_ast(base_stmt).reads)
    hoist_indices, _ = _backward_slice(producers, base_seed)
    hoist_set = set(hoist_indices)

    # Lane-invariant side-effecting statements whose backward slice reaches the
    # (frozen) chunk-ENTRY accumulator (the store of ``b_h``) must hoist before
    # the lane loop so they run once per chunk on the frozen value; bring their
    # producers along.  A store that does NOT read the accumulator stays in the
    # lane loop (it is a genuinely per-lane side effect).
    entry_indices: list[int] = []
    for idx, prod in enumerate(producers):
        if idx in hoist_set or reads_lane(prod) or not _has_side_effect(prod):
            continue
        slice_indices, _ = _backward_slice(
            producers[:idx], set(ReadWrites.from_ast(prod).reads)
        )
        slice_reads = set(ReadWrites.from_ast(prod).reads)
        for i in slice_indices:
            slice_reads |= set(ReadWrites.from_ast(producers[i]).reads)
        if not (acc_family & slice_reads):
            continue
        # The store + its lane-invariant producer slice all hoist.
        if any(reads_lane(producers[i]) for i in slice_indices):
            continue
        entry_indices.append(idx)
        hoist_set.update(slice_indices)

    hoist_pre = sorted(hoist_set | set(entry_indices))
    pre_lane = [producers[i] for i in hoist_pre]
    lane_kept = [
        producers[i]
        for i in range(len(producers))
        if i not in hoist_set and i not in set(entry_indices)
    ]

    new_lane_loop = _create_lane_loop(
        lane_var, _lane_loop_extent(lane_loop), [*lane_kept, sum_stmt]
    )
    new_chunk_body: list[ast.AST] = [
        *chunk_body[:lane_idx],
        reset_stmt,
        *pre_lane,
        base_stmt,
        new_lane_loop,
        final_stmt,
        *chunk_body[lane_idx + 1 :],
    ]
    return create(
        ast.For,
        target=stmt.target,
        iter=stmt.iter,
        body=new_chunk_body,
        orelse=stmt.orelse,
        type_comment=None,
    )


@dataclasses.dataclass
class LoopDimInfo:
    begin_var_name: str | None = None
    begin_expr: sympy.Expr | None = None
    end_var_name: str | None = None
    end_expr: sympy.Expr | None = None

    def is_end_matching(self, size: int | torch.SymInt) -> bool:
        expected = _to_sympy(size)
        if expected == self.end_expr:
            return True
        if (
            self.end_expr is None
            or _has_unbacked(self.end_expr)
            or _has_unbacked(expected)
        ):
            return False
        shape_env = CompileEnvironment.current().shape_env
        # TODO(jansel): current check is based on size hints, may need to guard here in the future
        return shape_env_size_hint(shape_env, expected) == shape_env_size_hint(
            shape_env, self.end_expr
        )


@dataclasses.dataclass
class DeviceLoopOrGridState:
    strategy: TileStrategy
    block_id_to_info: dict[int, LoopDimInfo]
    thread_axis_sizes: dict[int, int] = dataclasses.field(
        default_factory=dict, kw_only=True
    )
    block_thread_axes: dict[int, int] = dataclasses.field(
        default_factory=dict, kw_only=True
    )

    @property
    def block_ids(self) -> list[int]:
        return self.strategy.block_ids


@dataclasses.dataclass
class DeviceLoopState(DeviceLoopOrGridState):
    for_node: ast.For
    inner_statements: list[ast.AST]
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    # Block ids that this device loop distributes across a per-thread lane
    # loop (CuTe only). A reduction over one of these blocks needs the
    # two-pass lane structure (see ``split_lane_loop_reductions``).
    lane_loop_blocks: set[int] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class EmitPipelineLoopState(DeviceLoopOrGridState):
    """State for emit_pipeline-based loops on TPU (Pallas backend)."""

    body_fn_name: str
    body_fn_def: ast.FunctionDef | None = None
    inner_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    pipeline_call: ast.AST | None = None
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    _tensor_to_dma_scratch: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ForiLoopState(DeviceLoopOrGridState):
    """State for fori_loop-based loops on TPU (Pallas backend).

    Uses jax.lax.fori_loop with pltpu.make_async_copy for tensors whose
    inner-block shape passes ``_check_dma_alignment``; tensors that fail
    are kept on their outer BlockSpec and accessed via ``pl.ds`` from the
    body. Per-tensor pipelining membership lives in
    ``_tensor_to_dma_scratch``; input tensors with an overlapped prefetch are
    recorded in ``_prefetched_load_tensors``.
    """

    body_fn_name: str
    loop_var_name: str  # The fori_loop index variable (e.g., "_j")
    inner_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    _tensor_to_dma_scratch: dict[str, str] = dataclasses.field(default_factory=dict)
    _tensor_to_sem: dict[str, str] = dataclasses.field(default_factory=dict)
    _prefetched_load_tensors: set[str] = dataclasses.field(default_factory=set)


@dataclasses.dataclass
class DeviceGridState(DeviceLoopOrGridState):
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_loop_blocks: set[int] = dataclasses.field(default_factory=set)
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)
    # ⭐ A lane nest BUILT BY THE STRATEGY, not by :meth:`wrap_body`.
    #
    # ``(emit, sink)``: ``emit`` is what ``wrap_body`` returns verbatim, and ``sink``
    # is the list the user body is spliced into.  ``None`` -- the default -- means
    # "nobody owns a nest", which is the inert answer and keeps every existing
    # kernel on the loop-building path below, byte for byte.
    #
    # WHY THIS EXISTS.  ``wrap_body`` runs at the END of codegen, but the CuTe TV
    # protocol (``memory_ops._cute_tv_partition_hoist``) mutates the lane body
    # *while loads lower* -- it inserts the ``cute.copy`` next to the constexpr
    # V-loop it must precede, and the per-chunk ``local_tile``/``partition_*``
    # declarations just above the lane loop.  Both targets therefore have to be real
    # list objects before the first load is lowered, so a strategy that carries a TV
    # plan builds its own nest in ``codegen_preamble`` and registers it here.
    #
    # ⚠ IT IS A SENTINEL, NOT A CLASS TEST.  ``wrap_body`` asks whether a nest was
    # SUPPLIED; it never asks which strategy supplied it.  That is deliberate -- this
    # repo's enumeration antipattern is an ``isinstance`` ladder per strategy, and a
    # nest is exactly the capability the wrap needs to know about.
    prebuilt_lane_nest: tuple[list[ast.AST], list[ast.AST]] | None = None

    # ⭐⭐ THE SAME NEST, AS A FACTORY -- what makes a prebuilt nest SEGMENTABLE.
    #
    # ⛔ THE PROBLEM IT SOLVES.  ``prebuilt_lane_nest`` above is a nest that has already
    # been BUILT: ``sink`` is spliced exactly once, so the nest can wrap the body only
    # ONE time.  A reduction whose input depends on another reduction's finished scalar
    # needs the lane axis reopened -- one nest PER DEPENDENCY LAYER -- and a
    # single-splice nest structurally cannot express that.  So
    # ``_emit_inline_lane_reduce`` declined on ``prebuilt_lane_nest is not None`` and
    # fell back to the AST marker, which for the constexpr-V (TV) shape it cannot
    # restructure ⇒ ``BackendUnsupported: ... still owing its lane fold and its
    # cross-thread combine``.  THAT raise is the only reason
    # ``ConfigSpec.normalize`` force-rolled a persistent request into a looped one.
    #
    # ⇒ a strategy that can rebuild its nest registers this CALLABLE as well.  Given a
    # body it returns a fresh ``(emit, sink)`` pair -- new AST nodes, same recipe -- so
    # :meth:`_wrap_segmented_body` can mint one nest per segment instead of splicing one
    # nest once.  ``None`` means "this nest is not rebuildable", which keeps the
    # single-splice behaviour byte for byte.
    #
    # ⚠ WHY A CALLABLE AND NOT A COPY TAKEN AT REGISTRATION TIME: the TV protocol mutates
    # the nest *while loads lower* -- ``memory_ops._cute_tv_partition_hoist`` inserts the
    # ``cute.copy`` next to the constexpr V-loop and the ``local_tile``/``partition_*``
    # declarations above the lane loop -- so a copy taken when the nest is registered
    # (``codegen_preamble``, before any load lowers) would be missing the load entirely.
    # ``wrap_body`` runs at the END of codegen, when the nest is complete, so the factory
    # is invoked THERE and clones what is by then a finished tree.
    #
    # ⚠ Each clone redeclares the same variable NAMES in a sibling loop. That is the
    # established shape here, not a hazard: ``_emit_lane_segment`` already re-emits
    # ``lane_setup_statements`` per segment for the same reason (the CuTe DSL rejects a
    # name defined inside another loop), so every sibling recomputes its own.
    prebuilt_lane_nest_factory: (
        Callable[[list[ast.AST]], tuple[list[ast.AST], list[ast.AST]]] | None
    ) = None

    # ⭐⭐ THE LANE SEGMENTS -- G1's mechanism, and the reason the AST lane-split pass is
    # not needed for the shapes that reach here.
    #
    # ⛔ THE PROBLEM.  A reduction over a lane-distributed axis owes TWO combines: a
    # SERIAL LANE FOLD (within one thread, across the ``for lane in range(extent)``
    # iterations) and a CROSS-THREAD COMBINE (``warp_reduction_*`` /
    # ``_cute_grouped_reduce_*``).  The second one must land OUTSIDE the lane loop --
    # ``mx`` is not a finished scalar until every lane has been folded -- and a
    # subsequent reduction's input (``exp(v - mx)``) cannot be computed until it is.
    # So one lane loop cannot express two dependent reductions, and the old design
    # emitted a MARKER and had an AST post-pass cut the loop afterwards.  Read as IR,
    # ``MARKER(v,'max')`` is a valid ONE-element reduction, so ``mx = v`` was a
    # *faithful* lowering of it -- which is how that indirection produced kernels that
    # compiled, looked plausible, and were WRONG (``exp(v-v) == 1.0``: softmax rows
    # summing to 128.0 instead of 1.0; relerr 7.685 on ``matmul_layernorm`` at N=512).
    #
    # ⭐ THE FIX, AND WHY IT LIVES HERE RATHER THAN AT THE REDUCTION SITE.  MEASURED
    # (``_redfix2/repro/r5_sink_at_reduction.py``): at the moment ``codegen_reduction``
    # runs on this path there is **no lane ``For`` node at all** -- ``add_lane_loop``
    # has only recorded a ``(var, extent)`` tuple, the active sink is the FLAT
    # ``wrapped_body`` list, and the loop is minted at the END of codegen by
    # :meth:`wrap_body`.  ⇒ "close the enclosing lane loop" is not expressible at the
    # reduction site; the design note that said it was pointed at a different site
    # (``BlockReductionStrategy``, whose sink genuinely IS a live loop body).
    #
    # Because the wrap is deferred, the primitive that IS available is stronger:
    # **seal a segment**.  This list holds the lane body cut into consecutive pieces:
    #
    #     [ (body_0, seal_0), (body_1, seal_1), ... ]
    #
    # :meth:`wrap_body` emits one lane loop per segment and drops that segment's
    # ``seal`` statements BETWEEN the loops, at the enclosing scope.  A reduction, at
    # its own lowering site, calls :meth:`seal_lane_segment` -- its serial fold is
    # already in the open segment, and its cross-thread combine goes in the seal, where
    # it lands outside every lane loop *by construction*.  Everything lowered afterwards
    # flows into the next segment.
    #
    # ⇒ ``#lane nests == #segments == #dependency layers`` is an IDENTITY of the
    # mechanism, not a property to check after a rewrite.  That matters because the old
    # pass's own known defect was NUMERICALLY CORRECT while emitting three lane nests
    # where the algorithm has two and one extra ``cute.copy`` -- ~50% more traffic that
    # no numeric check can see.
    #
    # ⚠ ``None`` -- the default -- means "nobody sealed anything", which keeps every
    # existing kernel on the single-body path below, byte for byte.  It is a SENTINEL,
    # not a class test.
    lane_segment_seals: list[tuple[int, list[ast.AST], str]] | None = None
    # Statements that must be re-emitted at the TOP of every lane segment (the per-lane
    # index / mask setup).  ⚠ ``lane_setup_statements`` is injected by :meth:`wrap_body`,
    # so a SECOND lane loop that does not repeat it would reference ``indices_1`` as a
    # name defined only inside its sibling -- which the CuTe DSL rejects outright.
    # Re-materialising a pure index computation into each partition is exactly what the
    # reduction roller does one level up (its two subgraphs hold 6 and 9 nodes against 13
    # in the root, deliberately overlapping).
    lane_segment_prefix: list[ast.AST] = dataclasses.field(default_factory=list)

    def seal_lane_segment(self, at: int, seal: list[ast.AST], lane_var: str) -> None:
        """Close the lane segment that ends at statement index ``at`` (exclusive) and
        record the statements to emit between it and the next one.

        Called from a reduction's own lowering site with ``at`` = the current length of
        the active sink, so the fold it just emitted is the last thing inside the loop
        being closed and ``seal`` (the cross-thread combine) runs once, after it.

        ⭐⭐ ``lane_var`` IS THE AXIS THIS SEAL BELONGS TO, AND CARRYING IT IS A MEASURED FIX.
        :meth:`_wrap_segmented_body` used to segment ``lane_loops[-1]`` on the premise that a
        reduction's axis is the innermost registered one -- **false**, because ``lane_loops`` is
        one flat list fed by two unrelated producers, so ``[-1]`` is *the last registered*.
        MEASURED with a free ``hl.arange`` registering its own lane loop AFTER the reduction's:
        both accumulator seeds and both cross-thread combines landed inside the WRONG loop.

        ⚠ AND THE AXIS CANNOT BE RECOVERED AT THE WRAP.  Registration order is
        ``add_lane_loop(synthetic_lane_1)`` → ``SEAL`` → ``SEAL`` →
        ``add_lane_loop(arange_lane_0)``, so at wrap time nothing distinguishes which of the
        two axes the seals were recorded against.  ⇒ the producer must state it, which is
        exactly this parameter.
        """
        if self.lane_segment_seals is None:
            self.lane_segment_seals = []
        self.lane_segment_seals.append((at, seal, lane_var))

    def has_lane_loops(self) -> bool:
        # ⚠ A PREBUILT NEST **IS** A LANE LOOP, and this clause is load-bearing rather
        # than cosmetic: ``generate_ast`` calls ``wrap_body`` only when this answers
        # True, so a nest registered without it is built, filled by the hoist, and then
        # silently DISCARDED -- MEASURED, the emitted kernel lost its lane loop and its
        # copies and referenced ``_tv_frag_0`` / ``reduction_vec_lane_1`` as undefined
        # names.  The predicate's question is "does the body need wrapping", and a nest
        # is exactly that.
        return bool(self.lane_loops) or self.prebuilt_lane_nest is not None

    def add_lane_loop(self, block_id: int, lane_var: str, extent: int) -> None:
        self.lane_loops.append((lane_var, extent))
        self.lane_loop_blocks.add(block_id)

    def wrap_body(self, body: list[ast.AST]) -> list[ast.AST]:
        # ⭐ SEGMENTS WIN OVER A SINGLE SPLICE, *if* the nest can be rebuilt.
        #
        # Order matters and this is the whole of the A1 lowering change.  A prebuilt nest
        # can be spliced only once (``sink`` is one list), so when a reduction sealed a
        # segment -- i.e. a later reduction needs this one's FINISHED scalar -- one splice
        # cannot express it and the seals would be silently dropped.  With a factory the
        # segmented path mints a fresh nest per segment, so the dependent case lowers here
        # instead of falling back to the AST marker (which cannot restructure the
        # constexpr-V shape and raises).
        #
        # ⚠ Both conditions are required.  No seals ⇒ nothing to segment, take the cheap
        # splice and stay byte-identical.  No factory ⇒ the nest is not rebuildable, so
        # segmenting would have to alias one ``sink`` across segments; splice instead.
        if (
            self.lane_segment_seals
            and self.prebuilt_lane_nest is not None
            and self.prebuilt_lane_nest_factory is not None
        ):
            return self._wrap_segmented_body(body)
        if (nest := self.prebuilt_lane_nest) is not None:
            # ⚠ SPLICE, DO NOT REBUILD.  ``sink`` is the SAME list object the
            # constexpr V-loop holds by reference, so assigning a new list here
            # would leave the emitted loop pointing at the empty one.  The setup
            # statements lead, exactly as in the loop-building path below.
            emit, sink = nest
            sink[:0] = [*self.lane_setup_statements, *body]
            return emit
        if self.lane_segment_seals:
            return self._wrap_segmented_body(body)
        wrapped: list[ast.AST] = [*self.lane_setup_statements, *body]
        for lane_var, extent in reversed(self.lane_loops):
            wrapped = [_create_lane_loop(lane_var, extent, wrapped)]
        return wrapped

    def _wrap_segmented_body(self, body: list[ast.AST]) -> list[ast.AST]:
        """One lane loop per sealed segment, with each seal emitted BETWEEN them.

        See :attr:`lane_segment_seals` for why the segments exist.  The shape emitted
        for two dependent reductions over a lane-distributed axis is::

            mx_lane_acc = -inf                    # a segment PREFIX (before every loop)
            for lane in range(32):                # segment 0
                <producers>; mx_lane_acc = max(mx_lane_acc, v)
            mx = warp_reduction_max(mx_lane_acc)  # segment 0's SEAL -- outside the loop
            sum_lane_acc = 0
            for lane in range(32):                # segment 1, REOPENED
                <producers re-materialised>; sum_lane_acc += exp(v - mx)
            sum = warp_reduction_sum(sum_lane_acc)
            <the tail: elementwise consumers of a FINISHED scalar, at THIS scope, ONCE>

        ⛔⛔ **PLACEMENT IS PER-STATEMENT, AND GETTING IT BACKWARDS IS A SILENT WRONG
        ANSWER** -- not a compile error.  Two measured failures bound it from both sides:

        * leaving an elementwise consumer of a FINISHED scalar inside a lane loop --
          attention's ``l_i = l_i * alpha + l_ij`` -- gives **relerr 1.0**, the completed
          sum added 64 times;
        * hoisting a genuinely lane-VARYING consumer out -- ``rms``'s ``row * scale``
          normalize-and-store, which owes one write per lane -- would write one element
          of the row and drop the rest.

        ⇒ neither "all of the tail runs once" nor "all of the tail is a second lane loop"
        is correct.  Each statement is placed by **what it is**, using the predicate this
        tree already has: :func:`_lane_varying_names`, the lane-level analogue of the
        reduction roller's ``should_go_in_inner_graph`` ("nodes go in the inner graph if
        they use the reduction dimension").  ⭐ Reused rather than re-derived on purpose --
        the tree records **three cheaper consumer-side criteria that were tried and
        measured wrong on the same counterexample** (a store signature, a block id, and
        "does a fold exist"), so a fourth invented criterion is the likeliest way to
        reintroduce a wrong answer.

        ⚠ The SEAL positions, by contrast, involve no inference at all: each is a
        statement INDEX recorded by the reduction that closed the segment, at the moment
        it closed it.  That is the half that has to be exact, and it is exact by
        construction.

        ⛔⛔ THIS SEGMENTS ``lane_loops[-1]`` AND IS ONLY CORRECT WITH **ONE** LANE LOOP.
        The claim that used to stand here -- "a reduction seals against the lane axis it is
        distributed over, which is the innermost one" -- is **FALSE**.  ``lane_loops`` is one
        flat list fed by two unrelated producers (a reduction's synthetic lane, and
        ``generate_ast``'s free-``hl.arange`` lane), so ``[-1]`` is *the last registered*, not
        *the reduction's*.

        MEASURED with two: the seals segmented the arange's axis, so both accumulator seeds and
        both cross-thread combines landed INSIDE the reduction's own lane loop -- the
        accumulator re-initialised and the combine run on an unfinished partial, once per lane.
        ⇒ ``#lane nests == #segments == #dependency layers`` is an identity of the mechanism
        **only for a single lane axis**.

        ⭐ So the multi-axis case is REFUSED upstream, at ``_emit_inline_lane_reduce``'s
        ``len(grid.lane_loops) != 1`` guard, and that guard is what makes this method's
        ``[-1]`` sound rather than lucky.  Making it genuinely multi-axis means keying each
        seal by its lane var; that is an extension, not a repair, and it needs its own
        measurement.  ⚠ Do not relax that guard without doing it.

        ⛔⛔ AND THE REFUSAL CANNOT LIVE AT THE SEAL SITE, WHICH THE FIRST ATTEMPT ASSUMED.
        MEASURED registration order on the failing shape::

            add_lane_loop synthetic_lane_1   lane_loops=[]
            SEAL 3                           lane_loops=['synthetic_lane_1']
            SEAL 7                           lane_loops=['synthetic_lane_1']
            add_lane_loop arange_lane_0      lane_loops=['synthetic_lane_1']   <- AFTER both seals

        ⇒ at seal time there is exactly ONE lane loop and a ``len(lane_loops) != 1`` guard
        there cannot fire; the second axis appears later, during body lowering.  So the check
        has to be HERE, at the wrap, which is the first point that sees the final list -- and
        the honest response here is to fall back to the unsegmented wrap (dropping the seals)
        rather than emit against a false premise.  ⚠ Dropping the seals means the markers were
        never created for those reductions, so the fallback must ALSO be correct on its own;
        it is, because the unsegmented path is exactly what a single-lane-loop kernel without
        dependent reductions emits, and the reduction that sealed still has its fold and its
        combine -- they are simply all inside one loop, which is the pre-G1 shape.
        """
        assert self.lane_segment_seals is not None
        # ⭐ SEGMENT THE AXIS THE SEALS NAME, not ``lane_loops[-1]``.  Every seal in a given
        # wrap belongs to one reduction's lane axis (a second axis's reduction would seal
        # against its own var), so the axis is read off the seals themselves.
        sealed_vars = {lane_var for _at, _seal, lane_var in self.lane_segment_seals}
        extents = dict(self.lane_loops)
        if len(sealed_vars) != 1 or next(iter(sealed_vars)) not in extents:
            # More than one axis sealed, or an axis that never registered a loop: no single
            # correct segmentation exists here.  Fall back to the unsegmented nest rather than
            # emit against a false premise.
            wrapped: list[ast.AST] = [
                *self.lane_segment_prefix,
                *self.lane_setup_statements,
                *body,
                *(stmt for _at, seal, _v in self.lane_segment_seals for stmt in seal),
            ]
            for outer_var, outer_extent in reversed(self.lane_loops):
                wrapped = [_create_lane_loop(outer_var, outer_extent, wrapped)]
            return wrapped
        # ⚠ Guard the indices: a seal recorded past the end of the body means statements
        # it expected were DCE'd or never emitted, and silently clamping would put a
        # cross-thread combine inside a loop.  Clamp explicitly and keep the order.
        n = len(body)
        seals = sorted(
            ((min(at, n), seal) for at, seal, _v in self.lane_segment_seals),
            key=operator.itemgetter(0),
        )
        # ⭐ THE SEALED AXIS, not ``lane_loops[-1]`` -- see :meth:`seal_lane_segment`.
        lane_var = next(iter(sealed_vars))
        extent = extents[lane_var]
        # ⛔⛔ SPLIT THE SETUP BY AXIS: only the SEALED axis's setup may be re-materialised
        # inside each segment loop.  An OUTER axis's setup belongs at THIS scope.
        #
        # ``lane_setup_statements`` is one flat list fed per axis.  On a 2-D kernel it holds
        # both ``offsets_0 = ... + lane_0`` / ``mask_0 = offsets_0 < m`` (the OUTER row
        # axis) and ``indices_1 = tid + synthetic_lane_1 * threads`` (the SEALED reduction
        # axis).  ``_emit_lane_segment`` re-emits what it is given into every segment loop,
        # which is right for the sealed axis (sibling loops cannot reference a name defined
        # in another loop) and WRONG for an outer axis: a consumer at the outer level then
        # cannot see it.
        #
        # ⛔ MEASURED: ``row_sum`` at 129x130, ``block_sizes=[64] num_threads=[32]``,
        # ``reduction_loops=[None]`` put ``mask_0`` inside the ``synthetic_lane_1`` loop
        # while its store guard ``if mask_0:`` sits after that loop, at the ``lane_0``
        # level -- ``NameError: name 'mask_0' is not defined`` from the CuTe DSL.
        # ⭐ ``origin/main`` is correct here only because its ``wrap_body`` puts the WHOLE
        # setup list outside all lane loops, so every level sees it; the segmented path is
        # the one that has to distinguish the axes.
        #
        # Emitted ONCE, here, rather than inside :meth:`_emit_lane_segment` -- that runs
        # per segment, so hoisting from there would need cross-call bookkeeping to avoid
        # redeclaring the same name.  ``out`` is later wrapped by every non-sealed axis, so
        # a statement placed at its head lands inside the outer lane loop and outside the
        # sealed one, which is exactly the required scope.
        from .ast_read_writes import ReadWrites

        setup_varying = _lane_varying_names(list(self.lane_setup_statements), lane_var)
        outer_setup: list[ast.AST] = []
        inner_setup: list[ast.AST] = []
        for stmt in self.lane_setup_statements:
            reads = set(ReadWrites.from_ast(stmt).reads) | _tv_copy_dest_names(stmt)
            if reads & setup_varying or lane_var in reads:
                inner_setup.append(stmt)
            else:
                outer_setup.append(stmt)
        out: list[ast.AST] = [*self.lane_segment_prefix, *outer_setup]
        cursor = 0
        # ⭐⭐ SHARED PRODUCERS ARE RE-MATERIALISED INTO EACH SEGMENT THAT NEEDS THEM, and
        # this is a correctness requirement rather than a convenience.
        #
        # ⛔ MEASURED, without it: on the P1 shape (``amax`` then ``sum(exp(v - mx))``)
        # segment 1's ``v_1 = v_0 - mx`` referenced ``v_0``, which is written only inside
        # segment 0's lane loop.  Two failures at once -- the DSL rejects the free
        # variable, AND (worse, because it is silent) the lane-varying analysis cannot see
        # that ``v_1`` derives from the lane axis when its producer is invisible, so the
        # whole fold was hoisted OUT of the loop and would have run once.
        #
        # ⭐ The reduction roller solves the same problem one level up and it is worth
        # copying rather than re-deriving: its two subgraphs hold 6 and 9 nodes against 13
        # in the root -- deliberately OVERLAPPING, not a disjoint split -- and its
        # ``readd`` re-emits a shared producer into each partition.  MEASURED on
        # ``layer_norm``, 7 FX nodes belong to more than one layer.
        produced_inside: list[ast.AST] = []
        for at, seal in seals:
            self._emit_lane_segment(
                out, body[cursor:at], lane_var, extent, produced_inside, inner_setup
            )
            out.extend(seal)
            cursor = at
        self._emit_lane_segment(
            out, body[cursor:], lane_var, extent, produced_inside, inner_setup
        )
        # Every OTHER lane axis wraps the whole segmented run, in registration order.  ⚠ Keyed
        # by exclusion of the sealed var rather than by position, because the sealed axis is not
        # necessarily the last registered one -- that assumption was the F1 defect.
        for outer_var, outer_extent in reversed(self.lane_loops):
            if outer_var == lane_var:
                continue
            out = [_create_lane_loop(outer_var, outer_extent, out)]
        return out

    def _emit_lane_segment(
        self,
        out: list[ast.AST],
        segment: list[ast.AST],
        lane_var: str,
        extent: int,
        produced_inside: list[ast.AST],
        setup: list[ast.AST],
    ) -> None:
        """Append one segment to ``out``: its lane-INVARIANT statements at the enclosing
        scope, then a lane loop over the lane-VARYING ones (omitted when there are none).

        ``produced_inside`` accumulates, across segments, the lane-varying statements that
        earlier segments emitted INSIDE their loops.  A later segment reading one of those
        names gets the producer re-materialised into its own loop -- see the ⭐⭐ note in
        :meth:`_wrap_segmented_body` for the measurement that requires this.

        ⚠ THE SETUP STATEMENTS ARE INCLUDED IN THE ANALYSIS BUT NOT IN THE PARTITION, and
        that is what makes the answer right.  ``lane_setup_statements`` defines
        ``indices_N = tid + lane * threads`` -- so without them in the analysis body, a
        statement reading ``indices_N`` looks lane-INVARIANT (nothing visible derives it
        from ``lane_var``) and would be hoisted out of the loop.  They are then re-emitted
        into the loop rather than partitioned, because every sibling loop must recompute
        its own index instead of referencing a name defined in the other loop -- which the
        CuTe DSL rejects outright.

        ⚠ Hoisting preserves relative order within each group but does move an invariant
        statement above a varying one.  That mirrors the shipped
        :func:`_split_one_lane_loop` tail (``invariant_indices`` then the varying loop) and
        is why an invariant scalar epilogue stops being re-run once per lane -- the old
        pass's measured defect re-ran ``cross_entropy``'s epilogue 32x per row.
        """
        if not segment:
            return
        from .ast_read_writes import ReadWrites

        # Re-materialise the producers this segment needs from earlier segments, THEN
        # analyse -- the two are one question.  A statement whose producer is invisible
        # reads as lane-invariant and would be hoisted out of the loop, which is the
        # silent half of the failure this fixes.
        prelude = _lane_shared_producer_prelude(produced_inside, segment)
        # ⭐⭐ A1: THE NEST'S OWN STATEMENTS JOIN THE ANALYSIS, for exactly the reason the
        # docstring gives for ``lane_setup_statements`` -- and omitting them hoists the
        # WHOLE body out of the loop.
        #
        # On the TV shape the chain from the lane var to the body runs THROUGH the nest:
        # ``reduction_lane_base_N = ... + lane * (tpr*vec)`` (in the nest's lane body) and
        # ``indices_N = base + reduction_vec_lane_N`` (a setup statement).  The body then
        # reads ``_tv_frag_K[reduction_vec_lane_N]`` -- naming the VEC lane, never the
        # outer lane var.  So with only ``setup`` visible, nothing derives from
        # ``lane_var``, every statement reads as invariant, and all of them are hoisted.
        # ⛔ MEASURED before this line: 18 statements emitted at the enclosing scope with
        # NO lane loop at all -- the fragment reads unwrapped, the copies gone, and the
        # kernel then failed as "kernel launch without tensor args" because DCE removed
        # every tensor. A silent-shaped defect that happened to fail loudly.
        #
        # ⚠ Analysis only: these are NOT partitioned or emitted here (the factory rebuilds
        # them per segment), exactly as ``setup`` is analysed but re-emitted rather than
        # placed.
        nest_stmts: list[ast.AST] = []
        if (nest := self.prebuilt_lane_nest) is not None:
            emit, _sink = nest
            for stmt in emit:
                nest_stmts.extend(_flatten_lane_nest_stmts(stmt))
        analysis = [*nest_stmts, *setup, *prelude, *segment]
        varying_names = _lane_varying_names(analysis, lane_var)

        def is_varying(stmt: ast.AST) -> bool:
            rw = ReadWrites.from_ast(stmt)
            # A copy's destination is an ARGUMENT, so ``ReadWrites`` reports it as a read
            # and reports no write at all -- which would make a per-lane ``cute.copy``
            # read as invariant.  ``_tv_copy_dest_names`` is the one view that can see it.
            reads = set(rw.reads) | _tv_copy_dest_names(stmt)
            return bool(reads & varying_names) or lane_var in reads

        # ⛔⛔ A MEMORY BARRIER ON THE HOIST, AND ``_lane_varying_names`` CANNOT SUPPLY IT.
        #
        # That predicate answers a *VALUE* dependence question ("does this statement's value
        # derive from the lane var?").  Hoisting also reorders, so it needs a *MEMORY*
        # dependence answer too: an invariant statement that must OBSERVE a varying statement's
        # side effect must not be lifted above it.  Shape that shows why the value predicate is
        # not enough --
        #
        #     buf[tm, i] = v * 2.0      # lane-VARYING store   -> stays inside
        #     z = buf[tm, 0]            # lane-INVARIANT load   -> would be hoisted ABOVE it
        #
        # -- where ``z`` reads what the store just wrote, so hoisting reads stale memory.  ⚠ I
        # could NOT reproduce a wrong answer from this at the configs tried (both arms read
        # fresh), so this is a guard against a gap that is real in the ANALYSIS rather than a
        # fix for a measured failure -- stated that way deliberately, because overclaiming a
        # measurement is the defect this run has been correcting elsewhere.
        #
        # ⭐ ``_has_side_effect`` is the EXISTING predicate for this, already consulted by
        # ``_lane_reduce_consume_tail`` in this same file (OBLIGATION 1: reuse, do not
        # re-derive).  Once any side-effecting statement has been kept inside the loop, nothing
        # after it is hoisted -- a positional barrier rather than an alias analysis, which is
        # the conservative direction and cannot be wrong in the dangerous way.
        inside: list[ast.AST] = [_clone_stmt(s) for s in prelude]
        barrier = False
        for stmt in segment:
            if is_varying(stmt):
                inside.append(stmt)
                barrier = barrier or _has_side_effect(stmt)
            elif barrier:
                # A varying side effect has already been emitted inside the loop, so this
                # statement may observe it.  Keep the program order.
                inside.append(stmt)
            else:
                out.append(stmt)
        if not inside:
            return
        # ⛔ AN UNDUPLICATABLE OP MUST NOT BE RE-MATERIALISED.  A pure ``gmem -> rmem``
        # load is safe to re-run; a matmul or a cross-thread collective is not -- re-running
        # it duplicates a barrier or a shared-memory side effect.  ``_contains_unduplicatable_op``
        # is the existing predicate for exactly this and it errs toward refusing (it is a
        # substring match with a measured false positive on ``my_warp_reduction_helper()``),
        # which is the safe direction.  ⇒ REFUSE rather than guess: the caller's marker path
        # still exists, and every historical failure in this area was a wrong placement that
        # compiled.
        if any(_contains_unduplicatable_op(s) for s in prelude):
            raise exc.BackendUnsupported(
                CompileEnvironment.current().backend.name,
                "a reduction over a lane-distributed axis needs a producer "
                "re-materialised into a second lane loop, but that producer contains a "
                "matmul or a cross-thread collective which cannot be re-run without "
                "duplicating its barrier. Reduce the lane extent, or split the kernel so "
                "the second reduction reads the collective's output from memory.",
            )
        # ⚠ Cloned: the same node must not appear in two sibling loops, or the two
        # copies alias and a later in-place rewrite of one edits both.
        segment_body: list[ast.AST] = [*(_clone_stmt(s) for s in setup), *inside]
        # ⭐ A REBUILDABLE PREBUILT NEST SUPPLIES THIS SEGMENT'S LOOP.
        #
        # When the strategy owns the nest (the CuTe TV shape: a lane loop whose body holds
        # the ``lane_base`` assignment and a ``range_constexpr(vec)`` loop, with the
        # ``cute.copy`` and ``partition_*`` declarations the TV protocol inserted), the
        # segment cannot be wrapped in a bare ``_create_lane_loop`` -- that would drop the
        # vec loop and the copy, i.e. lose the load entirely.  Ask the factory for a FRESH
        # nest per segment instead, so each reopened lane loop carries its own copy and its
        # own partitions rather than aliasing one sink.
        # ⚠ The factory OWNS the splice -- it seats ``segment_body`` inside the vec loop
        # itself, so this must not splice again (doing so emitted every statement twice).
        if (factory := self.prebuilt_lane_nest_factory) is not None:
            emit, _sink = factory(segment_body)
            out.extend(emit)
        else:
            out.append(_create_lane_loop(lane_var, extent, segment_body))
        produced_inside.extend(inside)


@dataclasses.dataclass
class PersistentReductionState(DeviceLoopOrGridState):
    lane_loops: list[tuple[str, int]] = dataclasses.field(default_factory=list)
    lane_setup_statements: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_prefix: list[ast.AST] = dataclasses.field(default_factory=list)
    outer_suffix: list[ast.AST] = dataclasses.field(default_factory=list)

    def has_lane_loops(self) -> bool:
        return bool(self.lane_loops)

    def wrap_body(self, body: list[ast.AST]) -> list[ast.AST]:
        wrapped: list[ast.AST] = [*self.lane_setup_statements, *body]
        for lane_var, extent in reversed(self.lane_loops):
            wrapped = [_create_lane_loop(lane_var, extent, wrapped)]
        return wrapped


class TileStrategy:
    _fn: weakref.ReferenceType[DeviceFunction]
    block_ids: list[int]

    def cute_tv_capable(self) -> bool:
        """Does THIS strategy carry a live TV plan whose emission scaffolding exists?

        ⭐ DECLARED HERE, ON THE COMMON BASE OF *EVERY* STRATEGY, AND THAT PLACEMENT
        IS THE POINT.  ``ReductionStrategy`` overrides it with the real six-field
        conjunction; this default is ``False`` so that any strategy which does not
        carry the scaffolding answers honestly instead of raising ``AttributeError``.

        It was previously declared on ``ReductionStrategy``, which left the consumer
        in ``cute/memory_ops.py`` gating on ``isinstance(cand, ReductionStrategy)``
        before it could ask — i.e. the class test was removed one layer down and
        reintroduced one layer up, so a NON-reduction strategy was still structurally
        excluded from the capability whatever fields it had.  That mattered concretely:
        ``CuteNDTileStrategy`` already owns the two hardest pieces of the protocol
        (``_cute_lane_body_by_block``, whose own comment says it uses "the same
        protocol ``LoopedReductionStrategy`` uses", and ``_cute_vec_lane_var_by_block``,
        the constexpr-V-loop anchor), and MEASURED, ``chunk_plan`` succeeds for every
        real NDTile config — ``bs=512 nt=32 vw=8`` gives ``vec=8 lane_extent=2`` at the
        full 128-bit atom — so its geometry was never the obstacle.

        ⇒ The rule this encodes: **ask a strategy what it can do, never what it is.**
        A consumer that must first test the class has not removed the class test.
        """
        return False

    # ── CAPABILITY ③ SENTINELS, on the common base of EVERY strategy ──────────
    #
    # ⭐ THESE ARE HERE FOR THE SAME REASON THE NINE METHODS ARE, and skipping them cost
    # me a gate failure.  ``memory_ops._cute_tv_stage_slice`` dereferences the staging
    # state on whatever strategy answered ``cute_tv_capable()``.  With the methods moved
    # but the state left behind, a ``CuteNDTileStrategy`` reaching that function died on
    # ``AttributeError: ... has no attribute '_cute_tv_reload_from'`` inside ``codegen for
    # node load`` -- a missed optimisation turned into a compiler CRASH, which is precisely
    # what the capability query exists to prevent.
    #
    # ⚠ EVERY DEFAULT BELOW IS THE INERT ANSWER, so a strategy that never sets one reads as
    # "no staging" rather than as a crash.  Subclasses that DO stage override with per-block
    # state (``CuteNDTileStrategy``) or per-instance state (``ReductionStrategy``).
    # ⚠ IMMUTABLE DEFAULTS ONLY.  A mutable class attribute here would be ONE object shared
    # by every strategy instance in the process, and the emitted symbol names repeat across
    # kernels, so a leak would silently HIT rather than fail.
    #
    # The extent one "chunk" covers along the reduction axis.  ``0`` == "no chunk
    # geometry", which ``_cute_stage_num_chunks`` reads as a decline.
    _cute_tv_chunk: int = 0
    # The CAUSE of a residency decline, set by whichever site refused, so the canonical
    # marker names the real reason instead of a generic string.  ``None`` == not refused.
    _cute_row_residency_decline: str | None = None
    # Cache for the whole-device-IR multi-read walk.  ``None`` == not yet computed.
    _cute_tv_multi_read_cache: frozenset[str] | None = None
    # Cache for the whole-device-IR STORE walk -- the store-alias proof the ``registers``
    # residency owes (``memory_ops._cute_tv_stored_tensors``).  ``None`` == not yet computed,
    # which is distinct from ``frozenset()`` ("computed; nothing is written").  ⚠ Conflating
    # those two would make an uncomputed walk read as "no writes anywhere", i.e. as a licence
    # to cache a tensor across a store -- the one answer that is a silent wrong result.
    _cute_tv_stored_cache: frozenset[str] | None = None
    # ``tile id -> the rmem cache var holding it`` for the ``registers`` residency; see the
    # full declaration on ``ReductionStrategy``.
    #
    # ⭐⭐ DECLARED ON **THIS** BASE, NOT ONLY ON ``ReductionStrategy``, AND THE OMISSION WAS
    # MEASURED.  ``memory_ops._cute_tv_partition_hoist`` dereferences this on whatever
    # strategy answered ``cute_tv_capable()`` -- and ``CuteNDTileStrategy`` answers True.  With
    # the field on ``ReductionStrategy`` alone, a tiled reduction died with
    # ``AttributeError: 'CuteNDTileStrategy' object has no attribute
    # '_cute_tv_rmem_frag_by_tile'`` *inside* ``codegen for node load`` -- a missed
    # optimisation turned into a compiler CRASH, caught by
    # ``test_residency_axis_is_three_distinct_kernels``.
    #
    # ⚠ THIS IS THE EXACT HAZARD THE COMMENT BLOCK ABOVE ALREADY WARNS ABOUT ("a method
    # promoted to this base makes every field it reads part of the base's contract"), and the
    # ``_cute_row_residency_requested`` entry there records the same crash on 8 of 8
    # ``cross_entropy`` cells.  ⇒ the sentinel on the class is the honest fix, NOT a
    # ``getattr(self, ..., default)`` at the call site, which would hide the coupling.
    #
    # ⚠ ``None``, never ``{}``: a mutable class attribute here is ONE dict shared by every
    # strategy instance in the process, and the emitted cache names repeat across kernels, so
    # a leak would silently HIT rather than fail.
    _cute_tv_rmem_frag_by_tile: dict[tuple[object, ...], str] | None = None
    # ``fragment var -> staged READER partition var``, for a strategy whose later sweeps are
    # CLONED by the split pass rather than lowered.  ``None`` is the inert answer and means
    # "rewrite nothing"; a two-lowering-site strategy never populates it.
    _cute_tv_stage_read_by_frag: dict[str, str] | None = None
    # ``cute_cluster_n`` as EMITTED.  ``1`` == no cluster, which is the inert answer and the
    # only one a strategy without capability ② can honestly give: ``_cute_stage_num_chunks``
    # divides the extent by this, so a value the emission does not implement would size the
    # staged buffer for a CTA share no loop bound produces.
    _cute_cluster_n_emitted: int = 1
    # ⭐⭐ THE THREE SENTINELS THAT MUST TRAVEL WITH THE METHODS, AND THE ONE I MISSED BROKE
    # THE TREE.  ``cute_row_residency_forbids_sweep_cache`` reads
    # ``_cute_row_residency_requested``, and ``device_function.codegen_function_def`` calls
    # that method on **EVERY** strategy in the dispatcher -- not only reduction ones.  With
    # the method on this base and its default still on ``ReductionStrategy``, the first
    # non-reduction strategy to reach it raised::
    #
    #     AttributeError: 'CuteFlattenedTileStrategy' object has no attribute
    #                     '_cute_row_residency_requested'
    #
    # on 8 of 8 ``cross_entropy`` frozen cells -- i.e. that kernel could not compile at all.
    #
    # ⚠ THIS IS THE LATENT CASE ``cute_tv_set_active_block``'s DOCSTRING ALREADY NAMED
    # ("The same latent hazard still exists for
    # ``ReductionStrategy.cute_row_residency_forbids_sweep_cache``, which no caller asks of a
    # non-reduction strategy today").  Moving the method is exactly what made "today" end, so
    # the warning was a prediction and it came true.
    #
    # ⇒ THE RULE THE MOVE HAS TO OBEY: a method promoted to this base makes every field it
    # reads part of the base's contract.  A sentinel on the class is the honest fix -- NOT a
    # ``getattr(self, ..., default)`` at the call site, which would hide the coupling and is
    # against this codebase's convention.
    #
    # ``gmem`` == "no mechanism, the second read comes from global", which is exactly what a
    # strategy with no TV plan does, so it is both the inert answer and the true one.
    _cute_row_residency_requested: str = ROW_RESIDENCY_GMEM
    # The GRANTED half.  ``None`` == staging was not requested (or was refused), which every
    # consumer reads as "leave the GMEM read alone".
    _cute_tv_reload_from: str | None = None
    # The TV plan: the width, without which there is nothing to emit.  ``None`` == no plan, and
    # every capability query treats that as a decline.  Declared here because the staging
    # methods gate on it before touching anything else.
    _cute_tv_plan: ChunkTVPlan | None = None

    def cute_tv_rounded_extent(self) -> int | None:
        """``N'`` when this strategy's tile covers a ROUNDED-UP extent, else ``None``.

        ⭐ ON ``TileStrategy`` FOR THE SAME REASON AS ``cute_tv_tail_predicate`` BELOW, and
        the pair is not a coincidence: they are the two halves of one fact.  A tile that
        overshoots ``N`` needs the extra slots in the staged buffer (this method) AND a
        predicate on the copy (that one), so a strategy that answers ``None`` here must
        answer ``None`` there.  ``ReductionStrategy`` overrides both; a strategy whose tile
        cannot exceed the extent overrides neither.

        ``None`` -- "the tile is exactly ``N``" -- is the inert answer, and it is the right
        one for ``CuteNDTileStrategy``: that class DECLINES to build a plan at all when
        ``int(numel) % chunk`` (see ``_build_cute_tv_plan_for_block``), so a ragged extent
        never reaches staging there and there is nothing to round.  MEASURED at N=2000 and
        N=1543 with chunk=512: ``plan=None``, no chunk prefix, no TV copy.
        """
        return None

    def cute_stage_widest_dtype_bits(self) -> int:
        """Widest participant element size in bits, for the SMEM staging CHARGE (A7c).

        The charge must cover the dtype the staged tile is ACTUALLY allocated at, and with
        mixed dtypes that is the retyped one -- see ``cute_stage_feasible``'s note for the
        measured ptxas failure when it under-charged.  0 means "no participant information",
        and the caller then falls back to the plan's own dtype (the pre-A7c behaviour).

        Overridden by the strategies that know their participant set; the base cannot, and a
        base that guessed would be a second opinion able to disagree with the allocation.
        """
        return 0

    def _cute_tv_shared_for_dtype(self, dtype_str: str) -> tuple[str, str] | None:
        """The ``(atom_var, thr_var)`` pair for ONE participant dtype, or ``None`` (A7c).

        ⭐ WHY PER DTYPE.  A copy atom's element type must match the tensor being copied, so a
        mixed-dtype access group (fp8 in, fp32 out) needs one atom per distinct dtype.  The
        GEOMETRY is shared -- ``ChunkTVPlan.for_dtype`` changes only the element type, and
        ``emit_tiled_copy``'s ``thr_layout``/``val_layout`` do not mention it -- so every
        per-dtype atom tiles the chunk identically and the legs cannot disagree.

        ⚠ Defined on the BASE and layered over the existing single-valued ``_cute_tv_shared``
        rather than replacing it, so a strategy that never sees a second dtype behaves exactly
        as before and the single-dtype emission stays byte-identical.  The dict is created
        lazily for the same reason: no second dtype, no second entry, no diff.
        """
        cache = self._cute_tv_shared_by_dtype
        if dtype_str in cache:
            return cache[dtype_str]
        # First dtype seen: adopt whatever the single-valued field already holds, so a
        # strategy that emitted its atom before this accessor existed is not re-emitted.
        primary = self._cute_tv_shared
        if primary is not None and not cache:
            cache[dtype_str] = primary
            return primary
        return None

    def _cute_tv_set_shared_for_dtype(
        self, dtype_str: str, value: tuple[str, str]
    ) -> None:
        """Record this dtype's atom/slice pair; the FIRST also fills ``_cute_tv_shared``.

        Keeping the single-valued field in step matters: ``memory_ops`` reads it directly in
        two places (the staging and store-forwarding legs) to recover ``thr_var``, and those
        want *a* valid slice for this layout -- any dtype's will do, because all of them are
        ``get_slice`` of an identically-shaped tiled copy.
        """
        self._cute_tv_shared_by_dtype[dtype_str] = value
        if self._cute_tv_shared is None:
            self._cute_tv_shared = value

    @property
    def _cute_tv_shared_by_dtype(self) -> dict[str, tuple[str, str]]:
        """Per-dtype atom cache, scoped like ``_cute_tv_shared`` on this receiver.

        ``CuteNDTileStrategy`` overrides this to key by the ACTIVE block as well, because one
        instance drives up to three axes and each has its own layout.
        """
        cache = getattr(self, "_cute_tv_shared_by_dtype_store", None)
        if cache is None:
            cache = {}
            self._cute_tv_shared_by_dtype_store = cache
        return cache

    def cute_tv_thread_axis(self) -> int:
        """Which CUDA thread axis indexes the TV copy's ``get_slice`` (A7a).

        ⭐ THE COPY AXIS'S THREAD AXIS, WHICH IS NOT ALWAYS 0.  The tiled copy's
        ``thr_layout`` distributes ``threads_per_row`` threads along the axis being copied, so
        ``get_slice`` must be handed *this thread's index on that axis*.  For every reduction
        the answer is 0 -- helion's row axis stays out of the layout, so the reduction thread
        axis IS axis 0 -- and that is why the emission site hardcoded it.

        ⛔ IT WAS WRONG THE MOMENT A SECOND TILED AXIS APPEARED.  On root-grid 2-D pointwise
        (``out[tm, tn] = x[tm, tn] + y[tm, tn]``) the copied axis is ``tn``, whose thread axis
        is **1**, while ``tm`` owns axis 0.  Slicing by axis 0 hands every thread in a row the
        SAME fragment coordinate and different threads of a column different ones -- MEASURED
        maxdiff 6.8 on bf16 at ``vec=8``, i.e. a silent wrong answer, not a compile error.

        Default 0 keeps every reduction path byte-identical; ``CuteNDTileStrategy`` overrides
        it with the axis of the block whose plan is active.
        """
        return 0

    def cute_stage_block_id(self) -> int | None:
        """The block id whose ROW the SMEM staging methods describe (capability ③).

        ⭐ A DISTINCT ACCESSOR RATHER THAN ``block_index``, AND THAT IS THE WHOLE POINT.
        The nine staging methods below used to read ``self.block_index`` -- a
        ``ReductionStrategy`` property returning ``block_ids[0]``, which is well defined
        there because a reduction owns exactly ONE axis.  It is NOT well defined on a
        strategy that drives up to three (``CuteNDTileStrategy``), where
        ``block_ids[0]`` names whichever axis happened to sort first rather than the axis
        the copy addresses.  Giving that class a ``block_index`` property with different
        semantics would be a name meaning two things depending on the receiver -- exactly
        the aliasing trap this codebase has been bitten by repeatedly.

        So the methods ask THIS question instead, and every strategy answers it in its own
        terms: a reduction returns its one axis, an NDTile strategy returns the ACTIVE
        one.  The methods then stop caring which class they are on, which is the point of
        moving them to this base at all.

        ``None`` == "no row to describe", which every caller reads as a decline.  The
        default is ``None`` and not ``block_ids[0]``: a strategy that does not participate
        in the residency axis must not be handed a plausible-looking axis id.
        """
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # CAPABILITY ③ -- SMEM STAGING OF THE REDUCTION ROW, ON THE BASE CLASS.
    #
    # ⭐ EVERY METHOD BELOW WAS ON ``LoopedReductionStrategy`` AND IS NOW HERE, for the
    # same reason capabilities ① and ② moved: a consumer had to test the CLASS to
    # discover whether a CAPABILITY existed.  ``ReductionStrategy.cute_stage_feasible``
    # used to be a stub returning ``None`` -- "this strategy has no SMEM staging" --
    # which is a decline BY CLASS, not by geometry, and it made ③ vacuous on every
    # loop-free strategy however well its arithmetic worked out.
    #
    # ⭐ THE COUPLING WAS ONE FIELD, exactly as it was for ①.  The looped versions read
    # ``self._loop_block_size`` ("the extent one ``for roffset`` iteration covers");
    # they now read :attr:`_cute_tv_chunk`, whose contract is the strictly more general
    # "the extent ONE CTA covers in one pass".  ``LoopedReductionStrategy.__init__``
    # assigns ``_cute_tv_chunk = self._loop_block_size``, and persistent assigns its
    # padded (or per-CTA) extent, so the two are ONE reading of one number and the
    # looped path's emitted text is unchanged.
    #
    # ⭐ AND THE LOOP-FREE ARITHMETIC WORKS OUT, VERIFIED against the emitted CuTe
    # rather than argued: with no outer loop ``num_chunks == 1`` and ``chunk`` is the
    # padded per-CTA extent, so ``stage_smem_elems == rows_per_cta * 1 * padded`` --
    # the same ``rows_per_cta * (N / cluster_n)`` the looped path allocates -- and the
    # chunk coordinate is the literal ``0`` that ``codegen_preamble`` already installs
    # in ``_cute_tv_chunk_index_var``.  MEASURED on a persistent two-pass kernel at
    # N=1024 bs=1 tpr=32 vec=8: ``alloc_smem(BFloat16, 1024)`` == 1 row x 1024 cols.
    #
    # ⚠⚠ BUT THE BUFFER IS ONLY HALF OF ③, AND THE OTHER HALF IS NOT HERE.  On the
    # looped path the row's SECOND read is a second LOWERING SITE (two ``for roffset``
    # device loops in the device IR), so ``memory_ops._cute_tv_stage_slice`` is called
    # TWICE and returns ``first_read=True`` then ``False`` -- and the ``False`` call is
    # what emits the staged READ that REPLACES a gmem copy.  A loop-free strategy has
    # exactly ONE lowering site; its several visible sweeps are manufactured LATER, by
    # ``tile_strategy._split_lane_loop_over_constexpr_vec`` cloning one lane body.  So
    # the read half is emitted by THAT pass (``_tv_restage_cloned_loads``), not here.
    # MEASURED with only the methods below moved: one ``autovec_copy`` WRITE, ZERO
    # ``partition_S`` readers, and the write landing in the wrong sweep.
    # ══════════════════════════════════════════════════════════════════════════

    # ── THE RAGGED TAIL: one accessor, read by the tiler, the cluster, the ────
    #    staging sizing and the copy guard, so they cannot disagree.

    def _cute_reload_from_config(
        self, fn: DeviceFunction, plan: ChunkTVPlan, block_index: int
    ) -> str | None:
        """Read ``cute_reduction_reload``: the REQUEST half of the decision.

        Split deliberately in two.  This runs from ``__init__``, where the
        thread-block dims are not final yet (other strategies have not been
        constructed, so ``thread_block_dims()`` still reports 1 on the row axis --
        MEASURED, it is what made the first version of this decline every config).
        So only knob-and-plan facts are checked here; everything that needs real
        geometry is in :meth:`cute_stage_feasible`, called at codegen time.

        ⭐ THERE IS NO LONGER A ``lane_extent`` DECLINE HERE, AND BOTH HALVES OF THE
        ONE THAT USED TO BE ARE MEASURED FALSE.  It read
        ``if plan.lane_extent != 1: return None``, justified as "with more than one
        lane iteration a thread owns several fragments per chunk and the staged tile
        would have to be indexed by lane as well, which is a different (and strictly
        bigger) buffer than quack's."  Refuted by symbols in its own call chain:

        * **"would have to be indexed by lane"** -- it ALREADY is.
          ``memory_ops._cute_tv_stage_slice`` returns
          ``plan.emit_lane_slice(cached, lane_var)``, and ``emit_lane_slice`` emits
          ``[None, 0, <lane>]``: one spelling, used for the gmem operand and the
          staged operand alike, because the staged tile is partitioned through the
          SAME ``thr_var`` slice and the SAME ``(1, chunk)`` tiler.  MEASURED in the
          emitted CuTe, both legs lane-indexed:
              cute.autovec_copy(_tv_frag_0, _tv_spart_0[None, 0, reduction_lane_1])
              cute.autovec_copy(_tv_spart_1[None, 0, reduction_lane_1], _tv_frag_1)
        * **"strictly bigger buffer"** -- it is INVARIANT.  ``stage_smem_elems`` is
          ``rows_per_cta * num_chunks * chunk`` and ``num_chunks * chunk == N`` (per
          CTA, ``N/cluster_n``), while ``lane_extent = chunk // (tpr * vec)`` lives
          *inside* ``chunk``.  So raising ``lane_extent`` at fixed ``chunk`` lowers
          ``num_chunks`` and the product does not move.  MEASURED: the alloc tracks
          ``rows_per_cta`` only -- ``bs=1 -> 4096``, ``bs=2 -> 8192``,
          ``bs=4 -> 16384`` elements, independent of ``lane_extent``.

        Removing it is safe because it made staging *unavailable*, not *declined*:
        a config can always decline staging by NAMING its residency, and VERIFIED an
        explicit ``cute_reduction_reload=[None]`` survives ``normalize()`` unchanged
        while only an ABSENT slot is refilled from the ladder.  So a cell where
        staging loses -- e.g. ``rms_norm`` N=16384, measured 10.8% worse than no
        mechanism -- is handled by pinning ``[None]``.  ⚠ What the gate DID do
        incidentally was keep ~22 of 24 declining cells from ever reaching
        :meth:`cute_stage_feasible`; they now reach it and decline on the real
        conditions (SMEM budget, geometry), which is the point of Task 2.
        """
        residency = self._cute_row_residency_config(fn, block_index)
        self._cute_row_residency_requested = residency
        if residency != ROW_RESIDENCY_SMEM:
            log.debug(
                "cute staging DECLINE block=%s: row residency %r is not 'smem' "
                "(plan vec=%s lane_extent=%s chunk=%s)",
                block_index,
                residency,
                plan.vec,
                plan.lane_extent,
                plan.chunk,
            )
            return None
        return ROW_RESIDENCY_SMEM

    def cute_row_residency_veto_offset_vars(self) -> set[str]:
        """⭐ THE ONE DECISION: which sweep loops may NOT use the register cache.

        Returns the set of loop OFFSET VARIABLES this strategy vetoes, i.e. the loops whose
        ``fuse_tv_copy_sweeps`` budget must be forced to 0 because their row residency is an
        EXPLICIT ``gmem``.  ``device_function.codegen_function_def`` unions this over every
        strategy and hands the result to the pass.

        ⭐⭐ WHY THIS METHOD EXISTS IN THIS SHAPE (task 4, FIXLIST item 4).  FIXLIST item 4
        says the three residencies are enforced in three different places and asks for ONE
        decision site.  The ``gmem`` arm was the worst of the three, and not for the reason
        the item gives -- it had a *predicate* method,
        ``cute_row_residency_forbids_sweep_cache``, whose docstring was the SPECIFICATION for
        the veto, and **that method was never called**.  MEASURED:

            $ grep -rn 'forbids_sweep_cache(' helion/ test/ _redfix2/
            helion/_compiler/tile_strategy.py:3684:    def cute_row_residency_forbids_sweep_cache(...)

        one hit, and it was the ``def``.  The veto was implemented INLINE TWICE in
        ``device_function.py`` -- once off the config key, once off the per-strategy record --
        and the second leg RE-IMPLEMENTED the predicate (``if value != "gmem": continue``)
        rather than calling it.  So the specification and the behaviour were different code,
        which is exactly the failure mode item 4 is about, in its sharpest form: a
        documented decision site that decides nothing.

        ⇒ this method now RETURNS THE ANSWER instead of answering a question nobody asked,
        so the two legs are one implementation and the docstring governs the code that runs.
        The base implementation covers a strategy with per-block resolved residencies
        (``CuteNDTileStrategy``); ``ReductionStrategy`` overrides it for the single-axis case.

        ⚠ IT VETOES ON ``gmem`` ALONE, AND THAT ASYMMETRY IS MEASURED, NOT TIDINESS.
        The tempting rule -- "veto whenever the residency is not ``registers``" -- MOVES
        A CELL THAT EXISTS TODAY.  MEASURED on ``rms_norm``-shaped M=2048 N=8192 with
        ``cute_row_residency=["smem"], cute_stage_smem_kb=[0]``: staging DECLINES on
        the zero budget and the register cache then fires (1 rmem declaration), which is
        today's behaviour for every cell whose staging is refused -- and after task 1's
        migration that is 12 of the 25 cells that used to carry the legacy key, i.e. ~30%
        of the frozen table.  Vetoing on the ``smem`` REQUEST would re-emit all of them.

        ⛔⛔ AND IT MUST ASK "WAS A RESIDENCY *RESOLVED*", NOT ONLY "IS IT gmem".  ``gmem``
        is ALSO the base-class default -- it means "no mechanism", the honest answer for a
        strategy that never participated in the axis -- so a predicate over the VALUE alone
        cannot tell an EXPLICIT ``gmem`` (which must veto, because that is the only thing
        making ``gmem`` a distinct kernel rather than a synonym for ``registers``) from an
        ABSENT request (which must NOT veto, or every kernel in the tree silently loses
        ``fuse_tv_copy_sweeps``).  ⇒ the answer needs PROVENANCE, not just a value, which is
        why this returns a set built from a per-block record rather than a bool from a field.
        That is also why the FIXLIST's proposed ``resolve_row_residency(request) -> granted``
        signature cannot reproduce the truth table: the request's three values do not carry
        the distinction.
        """
        resolved = self._cute_row_residency_requested_by_block
        if not resolved:
            # Never resolved a residency for any block => the ``gmem`` this strategy reports
            # is the base default ("no mechanism"), not a request.  Nothing to veto.
            return set()
        out: set[str] = set()
        for block_id, value in resolved.items():
            if value != ROW_RESIDENCY_GMEM:
                continue
            try:
                out.add(self.offset_var(block_id))
            except (NotImplementedError, AssertionError, KeyError):
                # No per-block offset variable => not a device loop => neither pass can fire
                # on it, so there is nothing to veto.
                continue
        return out

    # ``block_id -> resolved residency``, for a strategy that resolves one PER AXIS.  Empty
    # is the inert answer and means "this strategy never participated in the axis", which
    # ``cute_row_residency_veto_offset_vars`` above reads as "nothing to veto".
    # ⚠ IMMUTABLE DEFAULT: a mutable class attribute would be ONE dict shared by every
    # strategy in the process, and the emitted offset-var names repeat across kernels, so a
    # leak would silently HIT rather than fail.  ``CuteNDTileStrategy`` mints a real dict
    # per instance in its ``__init__``.
    _cute_row_residency_requested_by_block: Mapping[int, str] = MappingProxyType({})

    def cute_row_residency_forbids_sweep_cache(self) -> bool:
        """⭐ Does this reduction's residency FORBID the register sweep cache?

        ⚠ THE PREDICATE FORM, KEPT FOR THE SINGLE-AXIS CASE and for the tests that assert
        it.  ``cute_row_residency_veto_offset_vars`` above is what codegen calls; this is
        the value-level question it is built out of.  See that method for why the veto is
        asymmetric and why a value alone is not enough at the codegen site.

        The one thing that makes ``"gmem"`` a distinct kernel rather than a synonym for
        ``"registers"``.  ``gmem`` means "no mechanism -- sweep 2 re-reads global", so the
        ``fuse_tv_copy_sweeps`` register cache must not fire; without this veto the two
        values emit byte-identical code and the axis has two names for one kernel.

        ⚠ IT VETOES ON ``gmem`` ALONE, AND THAT ASYMMETRY IS MEASURED, NOT TIDINESS.
        The tempting rule -- "veto whenever the residency is not ``registers``" -- MOVES
        A CELL THAT EXISTS TODAY.  MEASURED on ``rms_norm``-shaped M=2048 N=8192 with
        ``cute_reduction_reload=["smem"], cute_stage_smem_kb=[0]``: staging DECLINES on
        the zero budget and the register cache then fires (1 rmem declaration), which is
        today's behaviour for every cell whose staging is refused.  Vetoing on the
        ``smem`` REQUEST would suppress that fallback and re-emit those cells.

        So the rule is: a request for ``smem`` that is refused still falls back to
        whatever the budget allows, exactly as before; only an EXPLICIT ``gmem`` -- a
        value that was unreachable before this axis existed -- suppresses the cache.
        That is what makes this edit inert on all 40 frozen cells by construction.
        """
        return self._cute_row_residency_requested == ROW_RESIDENCY_GMEM

    def _cute_row_residency_config(self, fn: DeviceFunction, block_index: int) -> str:
        """⭐ THE ONE PLACE THE ROW-RESIDENCY AXIS IS READ.  Returns the REQUEST.

        ``cute_row_residency`` (``CuteRowResidencySpec``) over
        ``("registers", "smem", "gmem")`` -- one three-valued axis where the decision
        used to be a conjunction over ``cute_reduction_reload`` (an enum, per reduction
        block) and ``cute_tv_sweep_cache`` (an int slot budget, per DEVICE LOOP), with
        ``gmem`` unnameable.  Every arm downstream reads THIS answer, so "exactly one
        mechanism is in effect" is true by construction rather than by convention.

        ⚠ THIS IS THE REQUEST, NOT THE GRANT, and the split is the same one
        :meth:`_cute_reload_from_config` documents: this runs from ``__init__`` where
        ``thread_block_dims()`` is not final, so only knob facts are read here.  Both
        optimising residencies can still be refused by geometry or budget
        (:meth:`cute_stage_feasible` for ``smem``, the ``cute_tv_sweep_cache`` budget for
        ``registers``), and a refusal falls to ``gmem`` -- which cannot itself be refused,
        because re-reading the row from global is what the kernel does when no mechanism
        fires.  The EFFECTIVE answer is recorded on the emitted artifact by
        ``memory_ops._cute_tv_record_residency``; this method never claims it.

        ⭐⭐ THE LEGACY FALLBACK IS GONE (task 1), AND THAT IS THE POINT OF TASK 1.  This
        method used to read ``cute_row_residency`` and then, when no slot answered, fall
        back to ``row_residency_from_legacy(cute_reduction_reload, cute_tv_sweep_cache)``
        -- i.e. the old two-knob conjunction was still a live SELECTOR here, so the
        ambiguity the axis exists to remove was still expressible at codegen time.  It is
        now a one-shot CONFIG MIGRATION instead: ``ConfigSpec._normalize_cute_row_residency``
        translates an old spelling into this key and then STRIPS it, so by the time any
        config reaches codegen the residency is carried by exactly one key.

        ⇒ the only remaining fallback is ``ROW_RESIDENCY_GMEM``, and it is not a
        compatibility path -- it is the honest answer for a block that owns no residency
        slot at all.  A TILED reduction axis is exactly that case (measured: 0 slots in
        the per-reduction-block specs), and ``gmem`` -- "no mechanism, the second read
        comes from global" -- is what such a strategy does.  It is also the base-class
        sentinel, so the two agree by construction.

        ⚠ MEASURED that this is inert on the whole frozen table: all 25 cells that named
        the legacy key were migrated to ``cute_row_residency`` first
        (``_notes_codereview/migrate_frozen_task1.py``), and the emitted CuTe is
        byte-identical on 40/40 cells before and after -- which is only true because the
        migration ran BEFORE this fallback was deleted.  Deleting it first would have
        silently re-read 25 cells as ``gmem``.
        """
        env = CompileEnvironment.current()
        residency = env.config_spec.cute_row_residency.config_get(
            cast("list[str]", fn.config.config.get("cute_row_residency", []) or []),
            block_index,
            None,
        )
        if isinstance(residency, str):
            return residency
        return ROW_RESIDENCY_GMEM

    def _cute_stage_smem_capacity_bytes(self) -> int:
        """⭐ WHAT THE **DEVICE** HAS for a staged row tile, in BYTES.  Hardware only.

        ⛔⛔ THIS USED TO BE A CONFIG BUDGET AND THE BUDGET IS GONE (task 1 steps 2-3).
        It read the ``cute_stage_smem_kb`` knob (default 64 KiB) and clamped that to the
        device limit; now there is no knob and the device limit is the whole answer.

        WHY THE KNOB WENT.  ``cute_row_residency`` names WHERE the second read of a
        reduction row comes from.  A *performance* budget that can overrule it means a
        config can name one memory and emit another -- MEASURED on the frozen table, 13 of
        40 cells recorded a residency the kernel did not use.  A capacity limit is a
        different thing: it is the hardware refusing, not a policy preferring, and that is
        one of the two refusals a named residency may legitimately get (the other being
        geometry).

        ⚠ WHAT IS LOST, STATED PLAINLY RATHER THAN GLOSSED.  The knob's 64 KiB default
        encoded a measured OCCUPANCY judgement -- 32/64 KiB win by 2-8%, 128 KiB loses by
        24-28% (``launch__occupancy_limit_shared_mem``: 6 blocks/SM unstaged, 3 at 64 KiB,
        1 at 128 KiB).  Occupancy is NOT expressible as a size limit: a tile can fit the
        device and still cost 93% (measured on ``rms_norm`` 8192x100000, which asks 112 KiB
        against a 227 KiB device).  ⇒ removing the cap means a shape whose row is large but
        device-legal will now STAGE where it previously declined, and may be slower.
        That is the intended trade: the residency the config names is the residency it
        gets, and a shape that does not want staging says ``gmem``.  ⭐ The frozen table
        pays nothing for it -- ``rms_norm/8192x100000`` was re-frozen to the ``gmem`` it was
        already emitting, so no cell asks for a tile the old cap refused.

        ⚠ ``getattr``/``or 0`` is load-bearing: the attribute is absent on some devices, and
        a missing limit must not become a ZERO capacity (which would silently forbid all
        staging).  On a device that cannot report it, staging is admitted and the DSL's own
        allocation check remains the backstop.
        """
        device_limit = int(
            getattr(
                torch.cuda.get_device_properties(torch.cuda.current_device()),
                "shared_memory_per_block_optin",
                0,
            )
            or 0
        )
        # No reported limit => do not constrain (see the ⚠ above).  A very large sentinel
        # rather than 0, so the caller's ``already + want > capacity`` test cannot refuse
        # every tile on a device that simply did not answer.
        return device_limit if device_limit > 0 else 1 << 62

    def cute_stage_feasible(self) -> tuple[int, int] | None:
        """``(rows_per_cta, num_chunks)`` when SMEM staging is affordable, else None.

        Called at codegen time, where the geometry is final.  Three declines, all
        measured or structural:

        1. **The DEVICE CAPACITY.**  The staged tile is ``rows_per_cta * N`` elements (see
           the sizing trap in ``ChunkTVPlan`` for why it is not ``* chunk``), and
           :meth:`_cute_stage_smem_capacity_bytes` says how much shared memory the device
           has.

           ⛔ THIS ITEM USED TO SAY "the occupancy budget ... the ``cute_stage_smem_kb``
           knob (:meth:`_cute_stage_smem_budget_bytes`) says how much the kernel may spend.
           Its default reproduces the measured 64 KB cliff exactly."  Task 1 deleted BOTH
           the knob and that method: the rule is now *honour the request and let it spill*,
           and only capacity and geometry may refuse.  ⚠ The 64 KiB default encoded a
           measured OCCUPANCY judgement that a capacity limit cannot express, so a
           wide-but-device-legal row now stages where it used to decline and may be slower --
           a real, recorded cost, not an oversight.  Keeping the old text here would send a
           reader to a method that does not exist and to a knob they cannot set.
        2. **⚠ The budget is charged PER KERNEL, not per reduction.**  This is
           class-9 item S1 made non-latent: ``alloc_smem`` SUMS across inlined
           call sites, so layer_norm's and cross_entropy's *two* reductions would
           each pass a per-reduction check and jointly allocate twice the cap --
           landing exactly on the measured losing size.  The running total lives
           on the DeviceFunction's ``cute_state`` so the second reduction sees the
           first one's charge.
        3. **No row axis in the thread block.**  Staging is per-CTA, so the tile's
           row coordinate must be a real ``thread_idx()[1]``.

        Idempotent: re-charging is keyed on ``self``, so the many load sites of one
        reduction all see the same answer and only the first pays.

        ⭐ EVERY DECLINE BELOW LOGS ITS REASON AND ITS GEOMETRY, at DEBUG
        (``HELION_LOGS=+helion._compiler.reduction_strategy``).  Before that, asking
        for staging on a cell that refuses it produced ZERO diagnostic output while the
        ``Config`` object still read ``['smem']`` -- the only way to find out was to
        grep the emitted CuTe for ``_tv_spart``.  ⚠ A decline must stay a DECLINE and
        never become a raise: an unconditional ``raise`` at a decline site broke all 8
        attention examples.  These declines are mostly legitimate; the defect was that
        they were invisible.
        """
        if self._cute_tv_reload_from != "smem":
            # A consequence of the request half, already logged there.
            log.debug(
                "cute staging DECLINE block=%s: reload_from=%r (not requested, or a "
                "previous site already exhausted the SMEM budget)",
                self.cute_stage_block_id(),
                self._cute_tv_reload_from,
            )
            return None
        plan = self._cute_tv_plan
        if plan is None:
            log.debug(
                "cute staging DECLINE block=%s: no TV plan, so nothing partitions "
                "through a layout that could be staged",
                self.cute_stage_block_id(),
            )
            return None
        rows_per_cta = self._cute_stage_rows_per_cta()
        if rows_per_cta is None:
            log.debug(
                "cute staging DECLINE block=%s: no usable row axis in the thread block "
                "(thread_block_dims=%s); the staged tile's row coordinate must be a "
                "real thread_idx()[1]",
                self.cute_stage_block_id(),
                self.fn.tile_strategy.thread_block_dims(),
            )
            return None
        num_chunks = self._cute_stage_num_chunks()
        if num_chunks is None:
            return None  # _cute_stage_num_chunks logs its own four reasons
        # ⛔⛔ CHARGE FOR THE **WIDEST** PARTICIPANT, NOT THE PLAN'S OWN DTYPE (A7c fix).
        #
        # ``stage_smem_bytes`` multiplies by ``plan.dtype_bits``, and ``plan`` here still
        # carries the FIRST participant's element type.  The ALLOCATION, however, is emitted
        # from the RETYPED plan (``memory_ops`` rebinds ``plan = dtype_plan`` before
        # ``emit_stage_smem_alloc``), so in a mixed-dtype group whose staged tensor is the
        # WIDER one the charge is an UNDERCOUNT and a tile that this budget admits can still
        # exceed the device.
        #
        # MEASURED (found by adversarial review): a bf16-weight / fp32-row reduction at
        # ``n=16384 rows=4 bs=512 residency=smem`` charges at 16 bits, passes, then allocates
        # at 32 and dies in **ptxas** -- ``NVPTX compiler invocation failed ... ptxas error``.
        # The pre-A7c tree declines the same config cleanly (no TV plan at all), so this is
        # A7c's regression, not pre-existing.
        #
        # ⇒ scale the charge to the widest participant.  Over-charging is the safe direction:
        # it can only DECLINE a tile that would not have fit anyway, and the alloc is bounded
        # by the same number.  Asking the participants (rather than trusting the plan's own
        # dtype) is also what keeps the charge correct if the retyping order ever changes.
        widest_bits = max(self.cute_stage_widest_dtype_bits(), plan.dtype_bits)
        want = plan.stage_smem_bytes(rows_per_cta, num_chunks)
        if widest_bits > plan.dtype_bits:
            want = want * widest_bits // max(1, plan.dtype_bits)
        charged = self.fn.cute_state.reduction_stage_smem_bytes
        if self not in charged:
            capacity = self._cute_stage_smem_capacity_bytes()
            already = sum(charged.values())
            if already + want > capacity:
                # ⛔ OVER WHAT THE DEVICE HAS.  Decline permanently for this reduction so a
                # later site cannot get a different answer.
                #
                # ⭐ THIS IS NOW A HARDWARE REFUSAL, NOT A POLICY ONE (task 1 steps 2-3).
                # It used to read "over the per-kernel SMEM budget ... raise
                # cute_stage_smem_kb to admit it" -- a *tunable* refusal, which is exactly
                # what made a named residency overridable.  There is nothing to raise now:
                # the tile does not fit the device.
                #
                # ⚠ THE CHARGE IS STILL PER KERNEL, and that half is inherited rather than
                # introduced: ``alloc_smem`` SUMS across inlined call sites, so
                # ``layer_norm``'s and ``cross_entropy``'s two reductions must be charged
                # against ONE capacity or they jointly allocate twice what they checked.
                log.debug(
                    "cute staging DECLINE block=%s: the staged tile exceeds DEVICE SMEM -- "
                    "want=%d B (rows_per_cta=%d x num_chunks=%d x chunk=%d x %d bits) "
                    "+ already_charged=%d B > device capacity=%d B",
                    self.cute_stage_block_id(),
                    want,
                    rows_per_cta,
                    num_chunks,
                    plan.chunk,
                    plan.dtype_bits,
                    already,
                    capacity,
                )
                self._cute_tv_reload_from = None
                # ⭐ And record it as the CAUSE, so the canonical row-residency marker (and
                # the raise built on it) names DEVICE CAPACITY rather than a generic
                # "declined".  Two different declines must give two different reasons, or
                # one hardcoded string passes for a diagnosis.
                self._cute_row_residency_decline = (
                    f"the staged row tile exceeds the device's shared memory "
                    f"(want={want} B + already={already} B > capacity={capacity} B)"
                )
                return None
            charged[self] = want
        log.debug(
            "cute staging ENGAGED block=%s: rows_per_cta=%d num_chunks=%d chunk=%d "
            "vec=%d lane_extent=%d -> %d B",
            self.cute_stage_block_id(),
            rows_per_cta,
            num_chunks,
            plan.chunk,
            plan.vec,
            plan.lane_extent,
            want,
        )
        return (rows_per_cta, num_chunks)

    def _cute_stage_rows_per_cta(self) -> int | None:
        """Rows one CTA stages == the thread-block extent on the row axis.

        ``None`` when this reduction's own thread axis is the only one, i.e. there
        is no separate row axis to index the staged tile with.
        """
        dims = self.fn.tile_strategy.thread_block_dims()
        rows = int(dims[1])
        if rows < 1:
            log.debug(
                "cute staging DECLINE block=%s: thread_block_dims=%s has no row axis "
                "(dims[1]=%d < 1)",
                self.cute_stage_block_id(),
                dims,
                rows,
            )
            return None
        if dims[2] != 1:
            # A third thread axis would need a third staging mode; decline.
            log.debug(
                "cute staging DECLINE block=%s: a third thread axis is in play "
                "(thread_block_dims=%s, dims[2]=%d != 1) and would need a third "
                "staging mode",
                self.cute_stage_block_id(),
                dims,
                dims[2],
            )
            return None
        return rows

    def _cute_stage_num_chunks(self) -> int | None:
        """Outer chunk iterations, i.e. how many chunks the staged tile must hold.

        See the sizing trap in ``ChunkTVPlan``: helion runs the two sweeps as two
        separate ``for roffset`` loops, so sweep 2 chunk ``c`` needs sweep 1 chunk
        ``c`` and the buffer must span the whole row.

        ⭐ IT MUST SPAN THE **ROUNDED** ROW, and this is the sibling of the tiler
        edit above.  Both sweeps walk ``[0, N')``, so the buffer needs
        ``ceil(N'/chunk)`` slots; sizing it from the TRUE ``N`` would leave the last
        (tail) chunk of sweep 1 writing past the end of the allocation.  Reading the
        SAME ``cute_tv_rounded_extent`` the loop bound reads is what makes the two
        unable to drift apart -- the failure mode E014 item 1 describes.  The tail
        slots hold whatever the guarded copy left in the fragment, and the op
        identity at the combine discards them (``ragged_tail`` I4/I6).

        ⭐ ``num_chunks == 1`` IS THE LOOP-FREE CASE, AND IT FALLS OUT OF THIS
        ARITHMETIC RATHER THAN BEING SPECIAL-CASED.  A strategy with no outer loop sets
        ``_cute_tv_chunk`` to the whole (per-CTA) padded extent, so ``total // chunk``
        is exactly 1 and ``stage_smem_elems`` reduces to ``rows_per_cta * padded``.
        Nothing below tests for a loop; the ONE thing that had to change is that the
        chunk is read from ``_cute_tv_chunk`` and not from ``_loop_block_size``.
        """
        env = CompileEnvironment.current()
        numel = env.block_sizes[self.cute_stage_block_id()].numel
        # ⚠ ``_cute_tv_chunk``, NOT ``_loop_block_size``: this was the ONE
        # loop-specific input of the whole staging block, exactly as it was for the TV
        # plan (see ``_build_cute_tv_plan``'s ``chunk`` parameter).  The looped path
        # assigns ``_cute_tv_chunk = self._loop_block_size``, so its emitted text is
        # unchanged; a loop-free strategy assigns its own padded per-CTA extent.
        chunk = self._cute_tv_chunk
        if chunk <= 0:
            log.debug(
                "cute staging DECLINE block=%s: chunk=%d <= 0 (defensive)",
                self.cute_stage_block_id(),
                chunk,
            )
            return None
        total = shape_env_size_hint(env.shape_env, numel)
        rounded = self.cute_tv_rounded_extent()
        if rounded is not None:
            total = rounded
        elif not env.known_multiple(numel, chunk):
            # No round-up in play (no TV plan) and the extent is ragged: the staged
            # tile's last chunk would be partial with nothing predicating it.
            log.debug(
                "cute staging DECLINE block=%s: ragged extent numel=%s with no "
                "round-up (chunk=%d does not divide it, cute_tv_rounded_extent=None), "
                "so the staged tile's last chunk would be partial and unpredicated",
                self.cute_stage_block_id(),
                numel,
                chunk,
            )
            return None
        if total <= 0:
            log.debug(
                "cute staging DECLINE block=%s: total=%d <= 0 (defensive)",
                self.cute_stage_block_id(),
                total,
            )
            return None
        # ⚠ TRAP 1, the flatness half.  With a cluster this CTA only ever sees
        # ``total // cluster_n`` of the row, so the staged tile must be that size --
        # NOT the whole row.  This is the single line that makes the SMEM footprint go
        # FLAT in N (PORT_SPEC §9c): as N doubles the ladder doubles ``cluster_n``, the
        # quotient is unchanged, and the same 64 KB cap that declined N>=16384 before
        # now admits it.  It is also the reason the tiler edit and the staging size
        # cannot drift apart -- they read the same ``cluster_n``.
        cluster_n = self._cute_cluster_n_emitted
        if cluster_n > 1:
            if total % cluster_n:
                log.debug(
                    "cute staging DECLINE block=%s: extent total=%d is not divisible by "
                    "cluster_n=%d, so this CTA's share of the row is not a whole number "
                    "of elements",
                    self.cute_stage_block_id(),
                    total,
                    cluster_n,
                )
                return None
            total //= cluster_n
        return max(1, total // chunk)

    def _cute_stage_row_axis_expr(self) -> str:
        """The CTA-LOCAL row coordinate for the staged tile.

        Deliberately ``thread_idx()[1]`` and NOT the clamped global row that
        ``_cute_tv_row_index_expr`` builds for ``local_tile`` on GMEM: the staging
        buffer is per-CTA and only has ``rows_per_cta`` rows, so the global row
        would be out of range.  No clamp is needed for the same reason -- the
        axis extent IS the tile's row extent, so every value it can take is in
        bounds by construction.
        """
        return "cutlass.Int32(cute.arch.thread_idx()[1])"

    def cute_stage_restages_cloned_sweeps(self) -> bool:
        """Does this strategy need the SPLIT PASS to emit staging's READ half?

        ⭐ THE ONE STRUCTURAL DIFFERENCE BETWEEN ③ ON A LOOPED AND A LOOP-FREE
        STRATEGY, AND IT IS NOT ABOUT THE BUFFER.  ``cute_stage_feasible`` and
        ``stage_smem_elems`` are already class-independent (see the capability-③
        banner), so the buffer, its size and its writer are emitted identically for
        both.  What differs is WHERE THE SECOND READ COMES FROM:

        * **Looped.** The row's several sweeps are several LOWERING SITES -- one
          ``for roffset`` device loop per sweep in the device IR.  So
          ``memory_ops._cute_tv_stage_slice`` is called once per sweep and returns
          ``first_read=True``, then ``False``; the ``False`` call is what emits
          ``autovec_copy(_tv_spart, _tv_frag)`` INSTEAD OF a gmem ``cute.copy``.
          Nothing else has to happen, which is why ③ worked there from the start.

        * **Loop-free.** There is exactly ONE lowering site.  MEASURED on a
          persistent two-pass kernel: ``_cute_tv_stage_slice`` is called ONCE, with
          ``first_read=True``.  The several sweeps a human reads in the emitted CuTe
          do not exist yet at lowering time -- they are manufactured afterwards by
          ``tile_strategy._split_lane_loop_over_constexpr_vec``, which CLONES one
          lane body per reduction phase.  So there is no second call to return
          ``False`` from, and moving the staging methods to the base class alone
          yields a buffer that is written and never read (MEASURED: 1 ``autovec_copy``
          write, 0 ``partition_S`` readers, and the write emitted in the wrong sweep).

        ⇒ for a loop-free strategy the read half must be emitted by the pass that
        creates the second sweep, which is what ``_tv_restage_cloned_loads`` does.

        ⚠ ANSWERED FROM THE STRATEGY'S OWN GEOMETRY, NOT FROM ITS CLASS.  "One
        lowering site per row" is exactly "there is no outer reduction loop", and this
        repo's enumeration antipattern is what an ``isinstance`` here would be.  The
        question is asked of ``offset_var``: a strategy with a real per-block offset
        variable has a ``for OFFSET in ...`` sweep loop and therefore one site per
        sweep, while a loop-free strategy's offset is the literal ``"0"`` -- which is
        the same property ``codegen_preamble`` relies on when it emits the chunk
        coordinate as a literal ``0``.
        """
        if (
            self._cute_tv_plan is None
            or self._cute_tv_reload_from != ROW_RESIDENCY_SMEM
        ):
            return False
        try:
            offset = self.offset_var(self.cute_stage_block_id())
        except (AssertionError, KeyError):
            return False
        return offset == "0"

    def cute_tv_lane_block_id(self) -> int | None:
        """Which block id this strategy's TV plan addresses along the copy axis.

        ``None`` == "no TV plan, or none this question applies to", which every
        consumer reads as a decline.

        ⭐ DECLARED HERE FOR THE SAME REASON AS ``cute_tv_capable``, and it exists
        because the two independent TV gates -- the plan's participant selector and
        ``memory_ops._cute_tv_site_eligible`` -- must agree about WHICH AXIS the copy
        addresses.  Before this, each recognised the axis by its *syntactic* form (a
        literal ``slice(None)``), which silently excluded a **tiled** reduction axis
        (``x[tile_m, tile_n]``, whose trailing index is a ``SymInt``) from both.
        Asking the strategy which block it owns, instead of pattern-matching the
        subscript, is what lets a tiled axis be admitted by both gates through ONE
        answer -- so they cannot drift, and a plan can never be built at a width no
        load site honours (that is bug class 1 exactly).

        ⚠ It is NOT "the block whose size matches", and must never become that.
        Resolving a lane axis by size equality is LEDGER E052/E053: MEASURED, the
        size scan mis-binds at 3 of 23 slice sites, and the axis selects the
        strategy that decides the access width.
        """
        return None

    def cute_tv_set_active_block(self, block_id: int) -> bool:
        """Tell this strategy which axis the CALLER is addressing; True iff usable.

        The default is a pure AGREEMENT CHECK -- it stores nothing, because a
        strategy that owns one axis has nothing to select.  ``CuteNDTileStrategy``
        overrides it to actually point its per-block state at ``block_id``.

        ⭐ DECLARED ON THE COMMON BASE FOR THE SAME REASON AS ``cute_tv_capable``: the
        consumer asks EVERY candidate strategy, so a strategy that does not
        participate must answer honestly rather than raise ``AttributeError``.  (The
        same latent hazard still exists for
        ``ReductionStrategy.cute_row_residency_forbids_sweep_cache``, which no caller
        asks of a non-reduction strategy today.)
        """
        return self.cute_tv_lane_block_id() == block_id

    def cute_tv_tail_predicate(self) -> str | None:
        """``base < N`` guarding a ``cute.copy`` whose tile can exceed ``N``, else None.

        ⭐ ON ``TileStrategy``, NOT ``ReductionStrategy``, and that placement is the
        whole point: ``memory_ops._cute_tv_partition_hoist`` calls this on WHATEVER
        strategy answered ``cute_tv_capable()`` True.  Leaving it on
        ``ReductionStrategy`` made the hoist crash with ``AttributeError`` for exactly
        the non-reduction strategies the capability query exists to admit -- the same
        "a missed optimisation becomes a compiler crash" failure the class tests were
        removed to prevent, one method further in.

        ``None`` -- "nothing to predicate" -- is the inert answer and the correct one
        for any strategy whose tile cannot exceed the extent.  A strategy that CAN
        overshoot must override this; ``ReductionStrategy`` does.
        """
        return None

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
    ) -> None:
        self._fn = weakref.ref(fn)
        self.block_ids = block_ids
        self.index_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"indices_{block_idx}", dce=True)
            for block_idx in block_ids
        }
        # CuTe DSL preprocessor counter collision: the preprocessor's
        # negative-step machinery (``_handle_negative_step`` in
        # ``cutlass.base_dsl.ast_preprocessor.DSLPreprocessor``) emits
        # ``offset_<counter>`` / ``start_<counter>`` / ``stop_<counter>`` /
        # ``step_<counter>`` / ``isNegative_<counter>`` helpers at the enclosing
        # scope of every for-loop whose step is not a positive Python literal.
        # Helion's tile-offset names share the same ``offset_<n>`` namespace —
        # Python's name-binding rule sees the late preprocessor assignment and
        # treats the variable as local for the whole function body, turning
        # earlier reads into ``UnboundLocalError``. The ``tile_`` prefix moves
        # Helion's names out of the reserved CuTe DSL namespace. Of the five
        # reserved suffixes, only ``offset_`` and ``step_`` are emitted by
        # Helion (``offset_<bid>`` here; ``step_<n>`` via ``codegen.lift(...,
        # prefix='step')`` in ``codegen_grid_loops`` / ``codegen_lane_loops``);
        # both are renamed on cute. ``start_/stop_/isNegative_`` collisions are
        # not currently emitted by Helion. Non-CuTe backends keep the
        # historical short name to preserve existing goldens — this is a
        # deliberate trade-off (see ``cute_plan.md`` §7.6.5.2 for the trade-off
        # rationale; search "CuTe DSL preprocessor counter collision" for the
        # diagnosis).
        env = CompileEnvironment.current()
        offset_prefix = "tile_offset" if env.backend.name == "cute" else "offset"
        self.offset_vars: dict[int, str] = {
            block_idx: self.fn.new_var(f"{offset_prefix}_{block_idx}", dce=True)
            for block_idx in block_ids
        }

    @property
    def fn(self) -> DeviceFunction:
        fn = self._fn()
        assert fn is not None
        return fn

    def offset_var(self, block_idx: int) -> str:
        return self.offset_vars[block_idx]

    def index_var(self, block_idx: int) -> str:
        return self.index_vars[block_idx]

    def mask_var(self, block_idx: int) -> str | None:
        raise NotImplementedError

    def load_mask_var(self, block_idx: int) -> str | None:
        """The mask a pure-READ consumer of ``block_idx`` must apply.

        Defaults to ``mask_var`` — i.e. reads and writes are masked identically, which is
        the behaviour every strategy had before this hook existed and is what every
        backend other than cute still gets.

        A strategy overrides this to return ``None`` when it can prove that *every thread
        the launch creates* has an in-range index on this axis, so the bounds compare is
        vacuous for reads.  ``mask_var`` itself is deliberately NOT overridden: a store's
        predication is chosen by ``mask_var`` and is therefore preserved by construction,
        because no store consumer calls this method.

        ⚠ WHY THE SPLIT IS ASYMMETRIC AND NOT MERELY CAUTIOUS.  A bounds mask on this path
        has two jobs that a single ``index < end`` compare cannot distinguish:

        * on a READ it selects which lanes' loaded values are *used* (an unused lane's
          value is discarded downstream by ``_mask_to``'s reduction identity), and
        * on a WRITE it selects which lanes *have an effect on memory*.

        Dropping the first is a no-op whenever the read is in bounds.  Dropping the second
        is never a no-op, because an unowned address written is an unowned address
        corrupted — and it is not recoverable downstream.  ``734eea2d9`` dropped the mask
        at its *definition*, which necessarily dropped both, and ``examples/split_k_barrier``
        then wrote 12.3% of its output from threads that did not own it.

        ⚠ AND THE READ SIDE STILL NEEDS THE INDEX IN RANGE, not merely "unused".  On this
        backend the mask is threaded into the *address* of a vec load
        (an inline anchor-pointer select), so an elided read mask is a real load from a real
        address.  That is why the proof below is about the launch's thread extent and not
        about whether the value is consumed.
        """
        return self.mask_var(block_idx)

    def ragged_peel_plan(self, block_idx: int) -> tuple[int, int, int, str] | None:
        """``(numel, bulk_end, block_size, mask_var)`` if this axis's TAIL can be peeled.

        ``None`` -- i.e. "decline" -- for every strategy that does not override this, which
        is every strategy other than ``NDTileStrategy``.  See that override for the
        arithmetic; see ``cute/peel_ragged_tile.py`` for the consumer.
        """
        return None

    def block_size_var(self, block_idx: int) -> str | None:
        return self.fn.block_size_var_cache.get((block_idx,))

    def supports_index_rank_expansion(self) -> bool:
        """Whether index expressions produced by this strategy are tensor-shaped."""
        return True

    def thread_axes_used(self) -> int:
        return 0

    def thread_block_sizes(self) -> list[int]:
        """Return the thread block size for each thread axis this strategy uses."""
        return []

    def thread_block_size_exprs(self) -> list[str]:
        """Return per-axis thread block sizes as launch-time expressions."""
        return [str(size) for size in self.thread_block_sizes()]

    @staticmethod
    def get_tl_range_kwargs(config: Config, block_idx: int) -> list[str]:
        """Get the range_extra string for loop unroll factor and num_stages based on config."""
        env = CompileEnvironment.current()
        kwargs = []

        range_unroll_factor = env.config_spec.range_unroll_factors.config_get(
            config.range_unroll_factors, block_idx, 0
        )
        range_warp_specialize = env.config_spec.range_warp_specialize.config_get(
            config.range_warp_specializes, block_idx, None
        )
        range_num_stages = env.config_spec.range_num_stages.config_get(
            config.range_num_stages, block_idx, 0
        )
        num_stages = config.num_stages

        if "tensor_descriptor" in config.indexing:
            # Tensor descriptor + multi-stage pipelines in addition to unrolling tend to cause
            # CUDA "misaligned address" or "unspecified launch failure" errors.
            if range_num_stages > 0:
                range_num_stages = 0
            if range_unroll_factor > 0 and num_stages > 1:
                range_unroll_factor = 0
        elif (
            range_num_stages > 1
            and range_unroll_factor > 1
            and env.block_sizes[block_idx].size
            and env.block_sizes[block_idx].numel.is_number
        ):
            # Unrolling can cause CUDA IMA with pipelining
            # We want to ensure new step size + pipeline is within bounds
            loop_numel = int(env.block_sizes[block_idx].numel)
            block_size = int(env.block_sizes[block_idx].from_config_assert(config))
            step = range_unroll_factor * block_size
            last_offset = ((loop_numel - 1) // block_size) * block_size
            remainder = loop_numel - last_offset
            range_num_stages = min(
                max(1, int(math.ceil(remainder / step))), range_num_stages
            )

        if range_unroll_factor > 0:
            kwargs.append(f"loop_unroll_factor={range_unroll_factor}")
        if range_warp_specialize is not None:
            kwargs.append(f"warp_specialize={range_warp_specialize}")
        if range_num_stages > 0:
            kwargs.append(f"num_stages={range_num_stages}")

        range_multi_buffer = env.config_spec.range_multi_buffers.config_get(
            config.range_multi_buffers, block_idx, None
        )
        if range_multi_buffer is not None:
            kwargs.append(f"disallow_acc_multi_buffer={not range_multi_buffer}")

        range_flatten = env.config_spec.range_flattens.config_get(
            config.range_flattens, block_idx, None
        )
        if range_flatten is not None:
            kwargs.append(f"flatten={range_flatten}")

        dpf_range = config.get("_triton_range_id_data_partition_factor", None)
        dpf_value = config.get("_triton_range_value_data_partition_factor", None)

        if dpf_range is not None and dpf_value is not None and dpf_range == block_idx:
            kwargs.append(f"data_partition_factor={dpf_value}")

        return kwargs

    @staticmethod
    def get_range_call_str(
        config: Config,
        block_ids: list[int],
        *,
        begin: str | None = None,
        end: str,
        step: str | None = None,
    ) -> str:
        env = CompileEnvironment.current()

        # Allow backend to override the range expression entirely
        backend_range = env.backend.range_str(begin, end, step)
        if backend_range is not None:
            return backend_range

        use_static_range = all(
            env.config_spec.static_ranges.config_get(
                config.static_ranges, block_idx, None
            )
            is True
            for block_idx in block_ids
        )

        range_args = []
        if begin is not None:
            range_args.append(begin)
        range_args.append(end)
        if step is not None and step != "1":
            range_args.append(step)

        if use_static_range:
            return f"tl.static_range({', '.join(range_args)})"

        range_kwargs = TileStrategy.get_tl_range_kwargs(config, block_ids[0])
        return f"tl.range({', '.join(range_args + range_kwargs)})"

    def user_size(self, block_index: int) -> sympy.Expr:
        raise NotImplementedError

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        raise NotImplementedError

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        raise NotImplementedError

    def codegen_preamble(self, state: CodegenState) -> None:
        """Called after a *different* strategy has been used to generate the grid."""

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        raise NotImplementedError

    def _create_block_id_info_dict(
        self,
        state: CodegenState,
        use_proxy_ends: bool = False,
        ends_override: list[object] | None = None,
    ) -> dict[int, LoopDimInfo]:
        """Helper to create block_id_to_info dictionary with end bounds.

        Args:
            state: The codegen state
            use_proxy_ends: If True, use proxy_ends from state.proxy_args (for device loops)
            ends_override: If provided, use these ends instead of block_sizes.numel (for data-dependent bounds)
        """
        env = CompileEnvironment.current()
        block_id_to_info = {}

        def begin_to_ast(value: object) -> ast.AST:
            if isinstance(value, ast.AST):
                return value
            if isinstance(value, int):
                return expr_from_string(repr(value))
            if isinstance(value, sympy.Expr):
                return expr_from_string(DeviceFunction.current().sympy_expr(value))
            if isinstance(value, torch.SymInt):
                return begin_to_ast(value._sympy_())
            if isinstance(value, torch.Tensor):
                tensor_arg = DeviceFunction.current().tensor_arg(value)
                return expr_from_string(env.backend.scalar_load_expr(tensor_arg.name))
            raise NotImplementedError(f"{type(value)} is not implemented.")

        def normalize_dim_values(value: object) -> list[object]:
            if isinstance(value, (list, tuple, torch.Size)):
                return list(value)
            return [value]

        begin_values: list[object] | None = None
        proxy_begins: list[object] | None = None
        if isinstance(state.ast_args, (list, tuple)):
            if len(state.ast_args) >= 2 and isinstance(state.ast_args[1], list):
                begin_values = state.ast_args[1]
        if isinstance(state.proxy_args, (list, tuple)):
            if len(state.proxy_args) >= 2 and isinstance(
                state.proxy_args[1], (list, tuple, torch.Size)
            ):
                proxy_begins = normalize_dim_values(state.proxy_args[1])
                if begin_values is None:
                    begin_values = proxy_begins
            elif len(state.proxy_args) >= 2:
                begin_arg, end_arg = state.proxy_args[:2]
                if end_arg is None:
                    proxy_begins = [0] * len(normalize_dim_values(begin_arg))
                else:
                    proxy_begins = normalize_dim_values(begin_arg)
                if begin_values is None:
                    begin_values = proxy_begins

        if use_proxy_ends:
            _, _, proxy_ends, _, _ = state.proxy_args
            assert isinstance(proxy_ends, list)
            for idx, (block_idx, end) in enumerate(
                zip(self.block_ids, proxy_ends, strict=True)
            ):
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                else:
                    end_expr = None
                block_id_to_info[block_idx] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=None,
                    end_expr=end_expr,
                )
        elif ends_override is not None:
            # Data-dependent bounds: use the provided ends
            for idx, (block_id, end) in enumerate(
                zip(self.block_ids, ends_override, strict=True)
            ):
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if isinstance(end, (int, torch.SymInt)):
                    end_expr = _to_sympy(end)
                    end_var_name = state.sympy_expr(end_expr)
                else:
                    # Tensor (data-dependent) - end_expr is None, but we still need end_var
                    end_expr = None
                    end_var_name = None
                block_id_to_info[block_id] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=end_var_name,
                    end_expr=end_expr,
                )
        else:
            for idx, block_id in enumerate(self.block_ids):
                block_size_info = env.block_sizes[block_id]
                begin_expr = None
                begin_var_name = None
                if proxy_begins is not None:
                    begin = proxy_begins[idx]
                    if isinstance(begin, (int, torch.SymInt)):
                        begin_expr = _to_sympy(begin)
                if begin_values is not None:
                    begin_var_name = state.codegen.lift(
                        begin_to_ast(begin_values[idx]),
                        dce=True,
                        prefix="begin",
                    ).id
                if block_size_info.size is None:
                    # Data-dependent bound - skip numel, it will be handled elsewhere
                    end_expr = None
                    end_var_name = None
                else:
                    end_expr = block_size_info.numel
                    end_var_name = state.sympy_expr(end_expr)
                block_id_to_info[block_id] = LoopDimInfo(
                    begin_var_name=begin_var_name,
                    begin_expr=begin_expr,
                    end_var_name=end_var_name,
                    end_expr=end_expr,
                )

        return block_id_to_info

    def _setup_block_size_constexpr(
        self, state: CodegenState, block_size_var: str, block_size: SymIntLike
    ) -> None:
        """Helper to setup constexpr block size variable on host."""
        state.device_function.constexpr_arg_with_host_def(block_size_var, block_size)


class BlockSizeTileStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        super().__init__(
            fn=fn,
            block_ids=block_ids,
        )
        self.block_size = block_size
        self.loop_order = loop_order

    def _reorder(self, block_ids: list[_T]) -> list[_T]:
        if len(block_ids) <= 1:
            return block_ids
        order = self.loop_order
        assert len(order) == len(block_ids), (
            f"Invalid order length: {len(order)} != {len(block_ids)}"
        )
        assert {*order} == {*range(len(order))}, f"Invalid permutation: {order}"
        return [block_ids[i] for i in reversed(order)]

    def _get_data_dependent_numel(
        self, state: CodegenState, end: object, begin: object
    ) -> sympy.Expr | str:
        """Get numel for data-dependent bounds using the tensor end value.

        When the tile bound is a tensor (data-dependent), we need to pass
        the tensor to the kernel and use it to compute the number of elements.
        Returns either a sympy.Expr or a string expression.
        """
        from .device_function import DeviceFunction

        device_function = DeviceFunction.current()

        if isinstance(end, torch.Tensor):
            # For tensor bounds, we need to add it as a kernel argument
            # and load the scalar value
            tensor_arg = device_function.tensor_arg(end)
            end_expr = CompileEnvironment.current().backend.scalar_load_expr(
                tensor_arg.name
            )
        elif isinstance(end, (int, torch.SymInt)):
            end_expr = device_function.sympy_expr(_to_sympy(end))
        else:
            raise NotImplementedError(f"Unsupported end type: {type(end)}")

        if begin == 0:
            # Simple case: numel = end
            return end_expr  # type: ignore[return-value]
        if isinstance(begin, torch.Tensor):
            begin_arg = device_function.tensor_arg(begin)
            begin_expr = CompileEnvironment.current().backend.scalar_load_expr(
                begin_arg.name
            )
            return f"({end_expr} - {begin_expr})"  # type: ignore[return-value]
        if isinstance(begin, (int, torch.SymInt)):
            begin_expr = device_function.sympy_expr(_to_sympy(begin))
            return f"({end_expr} - {begin_expr})"  # type: ignore[return-value]
        raise NotImplementedError(f"Unsupported begin type: {type(begin)}")

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].symbol()

    def _fold_tile_end_op(
        self,
        state: CodegenState,
        end: object,
        block_size: int | torch.SymInt,
    ) -> sympy.Expr | None:
        """
        Compute more precise end bound for the pattern:

            for outer in hl.tile(...):
                for inner in hl.tile(outer.begin, outer.end):
                    ...
        """
        if isinstance(end, (int, torch.SymInt)):
            end = _to_sympy(end)
        elif not isinstance(end, sympy.Expr):
            return None

        var_info = state.device_function.expr_to_var_info.get(end)
        if var_info is None or not isinstance(block_size, int):
            return end

        from ..language.tile_ops import tile_end

        env = CompileEnvironment.current()
        fx_node = var_info.fx_node
        # check for the case where we have the same end bound a parent loop
        if (
            fx_node is not None
            and fx_node.target is tile_end
            and isinstance(arg := fx_node.args[0], torch.fx.Node)
            and (block_id := env.get_block_id(arg.meta["val"])) is not None
            and (device_loops := state.codegen.active_device_loops.get(block_id))
            and (loop_info := device_loops[-1].block_id_to_info.get(block_id))
            is not None
            # TODO(jansel): when parent block size is a SymInt, we fail to apply this optimization should fix this
            and isinstance(
                parent_block_size := state.device_function.resolved_block_size(
                    block_id
                ),
                int,
            )
            # If our block size is larger than the parent, then their will be gaps in the iteration space
            and block_size <= parent_block_size
        ):
            # Replace our end bound (a SymInt) will the parent loop's end bound
            return loop_info.end_expr
        return end

    def _compute_thread_axis_offset(
        self,
        active_device_loops: dict[int, list[DeviceLoopOrGridState]],
    ) -> int:
        """Compute the starting thread axis for the next strategy.

        Counts axes already claimed by active device loops, reserving at
        least one axis for reduction strategies when the backend places
        reductions first.

        When a ``CuTeGridExecutionPlan`` with ``block_axis_priority`` is
        in scope for this strategy's blocks, the offset is instead
        derived from ``thread_axis_for_strategy`` so the M/N axis order
        is dictated by the plan (e.g. the warp-per-row layout swaps the
        outer M-grid and inner N-tile axes so each warp owns one row).
        """
        from .reduction_strategy import ReductionStrategy

        env = CompileEnvironment.current()

        # Plan-driven path: honor ``block_axis_priority`` so the outer
        # grid loop can reserve an axis for a lower-priority inner tile
        # loop even when that inner loop has not yet entered
        # ``active_device_loops``.  Used by the warp-per-row layout where
        # the outer M-grid must take a HIGHER thread-axis index than the
        # inner N-tile so 32 contiguous threads on axis 0 form one warp
        # per row.
        plan = self.fn.tile_strategy.current_cute_grid_execution_plan(
            block_ids=self.block_ids
        )
        if plan is not None and any(
            plan.priority_for_block(block_id) is not None for block_id in self.block_ids
        ):
            offset = self.fn.tile_strategy.thread_axis_for_strategy(self)
            if offset is not None:
                return offset

        seen: set[int] = set()
        active_reduction_axes = 0
        active_non_reduction_axes = 0
        for loops in active_device_loops.values():
            for loop_state in loops:
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                axes = loop_state.strategy.thread_axes_used()
                if env.backend.reduction_axis_first() and isinstance(
                    loop_state.strategy, ReductionStrategy
                ):
                    active_reduction_axes += axes
                else:
                    active_non_reduction_axes += axes

        if not env.backend.reduction_axis_first():
            return active_non_reduction_axes + active_reduction_axes

        has_reduction_strategy = any(
            isinstance(strategy, ReductionStrategy) and strategy.thread_axes_used() > 0
            for strategy in self.fn.tile_strategy.strategies
        )
        if plan is not None and any(
            plan.disables_reduction_axis_reservation(block_id)
            for block_id in self.block_ids
        ):
            return active_non_reduction_axes + active_reduction_axes
        reserved_reduction_axes = max(
            1 if has_reduction_strategy else 0, active_reduction_axes
        )
        return reserved_reduction_axes + active_non_reduction_axes

    def select_pid_strategy(self) -> ProgramIDs:
        env = CompileEnvironment.current()
        if env.compact_worklist_plan is not None:
            # Compact worklist: the owner hl.grid becomes the work-item grid.
            from .program_id import WorklistProgramIDs

            return WorklistProgramIDs(upper_expr=str(env.compact_worklist_upper))
        backend_name = env.backend.name
        pid_type = self.fn.config.pid_type
        if pid_type == "xyz":
            assert 1 < len(self.block_ids) <= 3
            return XYZProgramIDs()
        use_tcgen05_scheduler = self._use_tcgen05_persistent_scheduler(
            pid_type, backend_name
        )
        if pid_type == "persistent_blocked":
            if use_tcgen05_scheduler:
                return Tcgen05PersistentProgramIDs(is_blocked=True)
            return PersistentBlockedProgramIDs()
        if pid_type == "persistent_interleaved":
            if use_tcgen05_scheduler:
                return Tcgen05PersistentProgramIDs(is_blocked=False)
            return PersistentInterleavedProgramIDs()
        assert pid_type == "flat"
        return FlatProgramIDs()

    def _use_tcgen05_persistent_scheduler(
        self, pid_type: str, backend_name: str
    ) -> bool:
        if backend_name != "cute" or not pid_type.startswith("persistent"):
            return False
        from .backend import _kernel_specialized_mma_impl

        return _kernel_specialized_mma_impl(self.fn, config=self.fn.config) == "tcgen05"


class FlattenedTileStrategy(BlockSizeTileStrategy):
    """Collapse all dimensions into single flat iteration space."""

    # pyrefly: ignore [bad-override]
    block_size: SymIntLike

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, (int, torch.SymInt))
        super().__init__(fn, block_ids, block_size, loop_order)
        env = CompileEnvironment.current()
        flat_numel = functools.reduce(  # pyrefly: ignore[incompatible-overload-residual]
            operator.mul, [env.block_sizes[i].numel for i in block_ids]
        )
        self._mask_var_can_prove_vacuous = env.known_multiple(
            flat_numel, block_size
        ) and not any(env.is_jagged_tile(i) for i in block_ids)
        if self._mask_var_can_prove_vacuous and not env.backend.force_tile_mask():
            self._mask_var = None
        else:
            self._mask_var: str | None = self.new_var("mask", dce=True)
        self._offsets_var = self.new_var("offsets", dce=True)

        key = (*self.block_ids,)
        assert key not in fn.block_size_var_cache
        fn.block_size_var_cache[key] = bs_var = self.new_var("_BLOCK_SIZE")
        for block_index in block_ids:
            fn.block_size_var_cache[(block_index,)] = bs_var

    def new_var(self, prefix: str, dce: bool = False) -> str:
        return self.fn.new_var(
            f"{prefix}_{'_'.join(map(str, self.block_ids))}", dce=dce
        )

    def offset_var(self, block_idx: int) -> str:
        raise NotImplementedError("offset_var not used in FlattenedTileStrategy")

    def mask_var(self, block_idx: int) -> str | None:
        if (
            self._mask_var is not None
            and self._mask_var_can_prove_vacuous
            and not self._uses_thread_axis()
        ):
            return None
        return self._mask_var

    def block_size_var(self, block_idx: int) -> str:
        return self.fn.block_size_var_cache[tuple(self.block_ids)]

    def thread_axes_used(self) -> int:
        return int(self._uses_thread_axis())

    def thread_block_sizes(self) -> list[int]:
        if not self._uses_thread_axis() or not isinstance(self.block_size, int):
            return []
        return [self.block_size]

    def thread_block_size_exprs(self) -> list[str]:
        if not self._uses_thread_axis():
            return []
        if isinstance(self.block_size, int):
            return [str(self.block_size)]
        bs_var = self.block_size_var(-1)
        if bs_var is None:
            return []
        return [bs_var]

    def _uses_thread_axis(self) -> bool:
        return not (isinstance(self.block_size, int) and self.block_size == 1)

    def _numel_str(self, state: CodegenState, value: sympy.Expr | str) -> str:
        if isinstance(value, str):
            return value
        return state.sympy_expr(value)

    def _range_trip_count(
        self,
        begin: object,
        end: object,
        step: object | None,
    ) -> sympy.Expr | str:
        return self._range_numel_expr(begin, end, step)

    def _range_numel_expr(
        self, begin: object, end: object, step: object | None
    ) -> sympy.Expr | str:
        begin_expr = (
            _to_sympy(begin)
            if isinstance(begin, (int, torch.SymInt, sympy.Expr))
            else None
        )
        end_expr = (
            _to_sympy(end) if isinstance(end, (int, torch.SymInt, sympy.Expr)) else None
        )
        diff_expr = (
            sympy.Add(end_expr, sympy.Mul(-1, begin_expr))
            if begin_expr is not None and end_expr is not None
            else None
        )
        if step is None or step == 1:
            if diff_expr is not None:
                return diff_expr
            return f"(({self._expr_str(end)}) - ({self._expr_str(begin)}))"
        assert isinstance(step, (int, torch.SymInt, sympy.Expr))
        step_expr = _to_sympy(step)
        if getattr(step_expr, "free_symbols", None):
            return (
                f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
                f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
            )
        if diff_expr is not None:
            return sympy.ceiling(sympy.Mul(diff_expr, sympy.Pow(step_expr, -1)))
        return (
            f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
            f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
        )

    def _expr_str(self, value: object) -> str:
        if isinstance(value, (int, torch.SymInt, sympy.Expr)):
            return self.fn.sympy_expr(_to_sympy(value))
        if isinstance(value, torch.Tensor):
            tensor_arg = DeviceFunction.current().tensor_arg(value)
            return CompileEnvironment.current().backend.scalar_load_expr(
                tensor_arg.name
            )
        if isinstance(value, str):
            return value
        raise NotImplementedError(f"{type(value)} is not implemented.")

    def _normalize_loop_steps(
        self, step_arg: object | None, ndim: int
    ) -> list[object | None]:
        if step_arg is None:
            return [None] * ndim
        if isinstance(step_arg, (list, tuple)):
            steps = list(step_arg)
            assert len(steps) == ndim
            return steps
        return [step_arg] * ndim

    def _extract_root_bounds(
        self, state: CodegenState
    ) -> tuple[list[object], list[object], list[object | None]]:
        assert len(state.proxy_args) == 3
        if state.proxy_args[1] is None:
            begins: list[object] = [0] * len(self.block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins_arg = state.proxy_args[0]
            begins = (
                list(begins_arg)
                if isinstance(begins_arg, (list, tuple))
                else [begins_arg]
            )
            ends_arg = state.proxy_args[1]
        ends = list(ends_arg) if isinstance(ends_arg, (list, tuple)) else [ends_arg]
        steps = self._normalize_loop_steps(state.proxy_args[2], len(self.block_ids))
        assert len(begins) == len(self.block_ids)
        assert len(ends) == len(self.block_ids)
        return begins, ends, steps

    def _extract_device_loop_bounds(
        self, state: CodegenState
    ) -> tuple[list[object], list[object], list[object | None]]:
        if len(state.ast_args) == 5:
            _, begins_arg, ends_arg, _, steps_arg = state.ast_args
        else:
            _, begins_arg, ends_arg, _ = state.ast_args
            steps_arg = None
        begins = (
            list(begins_arg) if isinstance(begins_arg, (list, tuple)) else [begins_arg]
        )
        ends = list(ends_arg) if isinstance(ends_arg, (list, tuple)) else [ends_arg]
        steps = self._normalize_loop_steps(steps_arg, len(self.block_ids))
        assert len(begins) == len(self.block_ids)
        assert len(ends) == len(self.block_ids)
        return begins, ends, steps

    def _codegen_common(
        self,
        state: CodegenState,
        *,
        begins: list[object] | None = None,
        ends: list[object] | None = None,
        steps: list[object | None] | None = None,
    ) -> tuple[str, str, sympy.Expr | str, list[ast.AST]]:
        offsets_var = self._offsets_var
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        if begins is None:
            begins = [0] * len(block_ids)
        if ends is None:
            ends = [env.block_sizes[block_id].numel for block_id in block_ids]
        if steps is None:
            steps = [None] * len(block_ids)
        total_numel: sympy.Expr | str = sympy.S.One
        statements = []

        # pyrefly: ignore [bad-assignment]
        for i, (block_idx, begin, end, step) in enumerate(
            self._reorder([*zip(block_ids, begins, ends, steps, strict=True)])
        ):
            cute_scalar_tile = (
                CompileEnvironment.current().backend.name == "cute"
                and len(block_ids) == 1
                and self._uses_thread_axis()
                and step not in (None, 1)
            )
            numel = (
                self._range_numel_expr(begin, end, None)
                if cute_scalar_tile
                else self._range_trip_count(begin, end, step)
            )
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({self._numel_str(state, total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({self._numel_str(state, numel)})"
            step_expr = self._expr_str(step) if step not in (None, 1) else None
            if step_expr is not None and not (
                CompileEnvironment.current().backend.name == "cute"
                and len(block_ids) == 1
                and self._uses_thread_axis()
            ):
                expr = f"({expr}) * ({step_expr})"
            if begin != 0:
                expr = f"({self._expr_str(begin)}) + ({expr})"
            statements.append(statement_from_string(f"{block_index_var} = {expr}"))
            if isinstance(total_numel, str) or isinstance(numel, str):
                total_numel = (
                    f"({self._numel_str(state, total_numel)})"
                    f" * ({self._numel_str(state, numel)})"
                )
            else:
                assert isinstance(total_numel, sympy.Expr)
                assert isinstance(numel, sympy.Expr)
                total_numel = sympy.Mul(total_numel, numel)

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            mask_terms = [f"{offsets_var} < ({self._numel_str(state, total_numel)})"]
            # Skip the ``thread_idx[axis] < block_size`` term for a CuTe
            # block-size-1 axis (see ``codegen_grid``): the axis is not a thread
            # axis, so this term would otherwise pin the launch dim to 1 and
            # block a synthetic free-``hl.arange`` axis from reusing it.
            if not (env.backend.name == "cute" and not self._uses_thread_axis()):
                thread_mask = env.backend.thread_in_tile_mask_expr(
                    block_size_var, axis=self._flat_thread_axis()
                )
                if thread_mask is not None:
                    mask_terms.insert(0, f"({thread_mask})")
            mask_expr = " and ".join(mask_terms)
            statements.append(statement_from_string(f"{mask_var} = {mask_expr}"))
        # pyrefly: ignore [bad-return]
        return block_size_var, offsets_var, total_numel, statements

    def _flat_thread_axis(self) -> int:
        """Compute the thread axis for this flattened strategy.

        For CuTe, reduction strategies occupy earlier axes.
        """
        return self._compute_thread_axis_offset(self.fn.codegen.active_device_loops)

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        assert state.ast_args is None

        from .ast_extension import ExtendedAST
        from .type_info import GridIndexType
        from .type_info import IterType
        from .type_info import SequenceType

        type_info = ExtendedAST.current()[-1]._type_info
        scalar_grid_loop = False
        if isinstance(type_info, IterType):
            inner = (
                type_info.inner.unpack()
                if isinstance(type_info.inner, SequenceType)
                else [type_info.inner]
            )
            scalar_grid_loop = len(inner) == 1 and isinstance(inner[0], GridIndexType)

        if (
            scalar_grid_loop
            and len(self.block_ids) == 1
            and len(state.proxy_args) == 3
            and not isinstance(state.proxy_args[0], (list, tuple))
            and (
                state.proxy_args[1] is None
                or not isinstance(state.proxy_args[1], (list, tuple))
            )
            and not isinstance(state.proxy_args[2], (list, tuple))
        ):

            def _range_bound_to_sympy(value: object) -> sympy.Expr:
                assert isinstance(value, (int, torch.SymInt, sympy.Expr))
                return _to_sympy(value)

            step = state.proxy_args[2]
            if step not in (None, 1):
                block_id = self.block_ids[0]
                if state.proxy_args[1] is None:
                    begin = 0
                    end = state.proxy_args[0]
                else:
                    begin = state.proxy_args[0]
                    end = state.proxy_args[1]
                    if isinstance(begin, (list, tuple)):
                        assert len(begin) == 1
                        begin = begin[0]
                    if isinstance(end, (list, tuple)):
                        assert len(end) == 1
                        end = end[0]
                begin_expr = _range_bound_to_sympy(begin)
                end_expr = _range_bound_to_sympy(end)
                step_expr = _range_bound_to_sympy(step)
                trip_count = (
                    f"(({state.sympy_expr(end_expr)}) - ({state.sympy_expr(begin_expr)}) + "
                    f"({state.sympy_expr(step_expr)}) - 1) // ({state.sympy_expr(step_expr)})"
                )

                env = CompileEnvironment.current()
                dtype = env.index_type()
                pid_var = state.device_function.new_var("pid_flat", dce=True)
                offsets_var = self._offsets_var
                block_size_var = self.block_size_var(-1)
                self._setup_block_size_constexpr(state, block_size_var, self.block_size)
                pids = self.select_pid_strategy()
                if isinstance(state.device_function.pid, ForEachProgramID):
                    pids.shared_pid_var = state.device_function.pid.shared_pid_var
                pids.append(PIDInfo(pid_var, block_size_var, trip_count, block_id))
                state.add_statement(
                    env.backend.arange_expr(
                        offsets_var,
                        pid_var,
                        block_size_var,
                        dtype,
                        axis=self._flat_thread_axis(),
                    )
                )
                index_var = self.index_var(block_id)
                state.add_statement(
                    f"{index_var} = ({state.sympy_expr(begin_expr)}) + ({offsets_var}) * ({state.sympy_expr(step_expr)})"
                )
                mask_var = self.mask_var(-1)
                if mask_var is not None:
                    mask_terms = [f"{offsets_var} < ({trip_count})"]
                    thread_mask = env.backend.thread_in_tile_mask_expr(
                        block_size_var, axis=self._flat_thread_axis()
                    )
                    if thread_mask is not None:
                        mask_terms.insert(0, f"({thread_mask})")
                    state.add_statement(
                        statement_from_string(
                            f"{mask_var} = {' and '.join(mask_terms)}"
                        )
                    )
                pids.codegen(state)
                if isinstance(state.device_function.pid, ForEachProgramID):
                    shared_pid = state.device_function.pid
                    shared_pid.cases.append(pids)
                    shared_pid.codegen(state)
                else:
                    state.device_function.set_pid(pids)
                tracker = ThreadAxisTracker()
                if self._uses_thread_axis() and isinstance(self.block_size, int):
                    tracker.record_all(
                        self.block_ids, self._flat_thread_axis(), self.block_size
                    )
                return DeviceGridState(
                    self,
                    block_id_to_info=self._create_block_id_info_dict(
                        state, ends_override=[end]
                    ),
                    thread_axis_sizes=tracker.sizes,
                    block_thread_axes=tracker.block_axes,
                )
        begins, ends, steps = self._extract_root_bounds(state)
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state,
            begins=begins,
            ends=ends,
            steps=steps,
        )
        env = CompileEnvironment.current()
        dtype = env.index_type()

        pid_var = state.device_function.new_var("pid_flat", dce=True)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))

        # A CuTe grid whose block size is 1 does not claim a thread axis: its
        # ``offsets = pid * 1 + thread_idx[axis]`` term is always 0 (launch dim
        # for the axis is 1). Emit ``offsets = pid * 1`` instead so the axis is
        # genuinely free for a synthetic free-``hl.arange`` thread axis to reuse
        # without the grid's ``thread_idx[axis] < 1`` mask filtering its lanes.
        if env.backend.name == "cute" and not self._uses_thread_axis():
            state.add_statement(
                statement_from_string(
                    f"{offsets_var} = ({pid_var}) * ({block_size_var})"
                )
            )
        else:
            state.add_statement(
                env.backend.arange_expr(
                    offsets_var,
                    pid_var,
                    block_size_var,
                    dtype,
                    axis=self._flat_thread_axis(),
                )
            )
        state.codegen.statements_stack[-1].extend(statements)

        pids.codegen(state)

        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        tracker = ThreadAxisTracker()
        if self._uses_thread_axis():
            thread_size: int | None = None
            if isinstance(self.block_size, int):
                thread_size = self.block_size
            elif isinstance(self.block_size, torch.SymInt):
                if (block_size_id := env.get_block_id(self.block_size)) is not None:
                    config_block_size = env.config_spec.block_sizes.config_get(
                        state.config.block_sizes,
                        block_size_id,
                    )
                    if isinstance(config_block_size, int):
                        thread_size = config_block_size
            if thread_size is not None:
                tracker.record_all(
                    self.block_ids, self._flat_thread_axis(), thread_size
                )
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        begins, ends, steps = self._extract_device_loop_bounds(state)
        block_size_var, offsets_var, total_numel, statements = self._codegen_common(
            state,
            begins=begins,
            ends=ends,
            steps=steps,
        )
        env = CompileEnvironment.current()
        dtype = env.index_type()
        lid = self.new_var("lid")
        numel_str = self._numel_str(state, total_numel)
        end_var = env.backend.cdiv_expr(numel_str, block_size_var, is_device=True)
        # Mirror ``codegen_grid``: a CuTe block-size-1 loop axis does not claim a
        # thread axis, so drop the always-zero ``+ thread_idx[axis]`` term so the
        # axis stays free (and consistent with the mask emitted by
        # ``_codegen_common``, which also drops its thread term for this case).
        if env.backend.name == "cute" and not self._uses_thread_axis():
            arange_expr = f"{offsets_var} = ({lid}) * ({block_size_var})"
        else:
            arange_expr = env.backend.arange_expr(
                offsets_var, lid, block_size_var, dtype, axis=self._flat_thread_axis()
            )
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
            ),
            body=(
                body := [
                    statement_from_string(arange_expr),
                    *statements,
                ]
            ),
            orelse=[],
            type_comment=None,
        )
        block_id_to_info = self._create_block_id_info_dict(state, ends_override=ends)
        tracker = ThreadAxisTracker()
        if self._uses_thread_axis():
            thread_size: int | None = None
            if isinstance(self.block_size, int):
                thread_size = self.block_size
            elif isinstance(self.block_size, torch.SymInt):
                if (block_size_id := env.get_block_id(self.block_size)) is not None:
                    config_block_size = env.config_spec.block_sizes.config_get(
                        state.config.block_sizes,
                        block_size_id,
                    )
                    if isinstance(config_block_size, int):
                        thread_size = config_block_size
            if thread_size is not None:
                tracker.record_all(
                    self.block_ids, self._flat_thread_axis(), thread_size
                )
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    @classmethod
    def update_allow_flattened(cls, shape: Sequence[sympy.Expr]) -> None:
        env = CompileEnvironment.current()
        used_indices = {}
        for i, x in enumerate(shape):
            block_idx = env.get_block_id(x)
            if block_idx is not None:
                used_indices[block_idx] = i
        flatten_loops = env.config_spec.flatten_loops
        for spec in [*flatten_loops]:
            block_ids = spec.block_ids
            if not (
                all(x in used_indices for x in block_ids)
                or all(x not in used_indices for x in block_ids)
            ):
                flatten_loops.disable_block_id(block_ids[0])
                continue
            for i, j in itertools.pairwise(block_ids):
                if i in used_indices and used_indices[i] + 1 != used_indices[j]:
                    # The block indices must be contiguous
                    flatten_loops.disable_block_id(block_ids[0])
                    break

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # Keep axis structure intact for multi-phase kernels (e.g., barrier) to
        # avoid mismatched ranks in downstream reductions.
        if len(HostFunction.current().device_ir.root_ids) > 1:
            return shapes

        env = CompileEnvironment.current()
        # Filter out unit-sized blocks that don't need compacting
        compact_block_ids = [
            block_id
            for block_id in self.block_ids
            if not (
                isinstance(env.block_sizes[block_id].size, int)
                and env.block_sizes[block_id].size == 1
            )
        ]
        if not compact_block_ids:
            return shapes

        output = []
        shape_queue = collections.deque(shapes)
        while shape_queue:
            shape = shape_queue.popleft()
            # Check if this starts our flattened sequence
            if len(shape.block_ids) != 1 or shape.block_ids[0] != compact_block_ids[0]:
                output.append(shape)
                continue

            # Try to collect the full sequence
            group_shapes = [shape]
            found_complete_sequence = True
            for expected in compact_block_ids[1:]:
                if (
                    shape_queue
                    and len(shape_queue[0].block_ids) == 1
                    and shape_queue[0].block_ids[0] == expected
                ):
                    group_shapes.append(shape_queue.popleft())
                else:
                    # Partial match - don't combine
                    found_complete_sequence = False
                    output.extend(group_shapes)
                    break

            if found_complete_sequence:
                # Full match - combine into one
                for s in group_shapes[1:]:
                    shape = shape.combine(s)
                output.append(shape)
        return output


class _BaseNDTileStrategy(BlockSizeTileStrategy):
    # pyrefly: ignore [bad-override]
    block_size: list[SymIntLike]

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
    ) -> None:
        assert isinstance(block_size, list)
        super().__init__(fn, block_ids, block_size, loop_order)
        for bs, block_idx in zip(block_size, block_ids, strict=True):
            if (block_idx,) not in fn.block_size_var_cache and bs != 1:
                fn.block_size_var_cache[(block_idx,)] = fn.new_var(
                    f"_BLOCK_SIZE_{block_idx}"
                )

    def _uses_thread_axis(self, block_size: SymIntLike) -> bool:
        return not (isinstance(block_size, int) and block_size == 1)

    def _uses_thread_axis_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> bool:
        """Hook: does ``block_id`` claim a CUDA thread axis under this strategy?

        Defaults to ``_uses_thread_axis(block_size)``. Subclasses that
        track per-block-id state (e.g. ``CuteNDTileStrategy``'s
        ``inactive_block_ids``) override this to return False for
        block_ids that don't claim an axis so the grid / device-loop
        codegen does not emit ``thread_idx[axis]`` for them.
        """
        return self._uses_thread_axis(block_size)

    def thread_axes_used(self) -> int:
        return sum(
            1 for block_size in self.block_size if self._uses_thread_axis(block_size)
        )

    def thread_block_sizes(self) -> list[int]:
        sizes: list[int] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if self._uses_thread_axis(bs) and isinstance(bs, int):
                sizes.append(bs)
        return sizes

    def thread_block_size_exprs(self) -> list[str]:
        exprs: list[str] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if not self._uses_thread_axis(bs):
                continue
            if isinstance(bs, int):
                exprs.append(str(bs))
            else:
                bs_var = self.block_size_var(block_id)
                if bs_var is None:
                    return []
                exprs.append(bs_var)
        return exprs

    def _thread_axis_offset(self, state: CodegenState) -> int:
        return self._compute_thread_axis_offset(state.codegen.active_device_loops)

    def _thread_axis_map(self) -> dict[int, int]:
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        axis_order = [self.block_ids[i] for i in self.loop_order]
        axis = 0
        mapping: dict[int, int] = {}
        for block_id in axis_order:
            mapping[block_id] = axis
            if self._uses_thread_axis(block_size_by_id[block_id]):
                axis += 1
        return mapping

    def _normalize_loop_steps(
        self, step_arg: object | None, ndim: int
    ) -> list[object | None]:
        if step_arg is None:
            return [None] * ndim
        if isinstance(step_arg, (list, tuple)):
            steps = list(step_arg)
            assert len(steps) == ndim
            return steps
        return [step_arg] * ndim

    def _root_grid_steps(self, state: CodegenState) -> list[object | None]:
        from .ast_extension import ExtendedAST
        from .type_info import GridIndexType
        from .type_info import IterType
        from .type_info import SequenceType

        type_info = ExtendedAST.current()[-1]._type_info
        assert isinstance(type_info, IterType)
        inner = (
            type_info.inner.unpack()
            if isinstance(type_info.inner, SequenceType)
            else [type_info.inner]
        )
        if not all(isinstance(value, GridIndexType) for value in inner):
            return [None] * len(self.block_ids)
        return self._normalize_loop_steps(state.proxy_args[2], len(self.block_ids))

    def _range_numel_expr(
        self, begin: object, end: object, step: object | None
    ) -> sympy.Expr | str:
        begin_expr = (
            _to_sympy(begin)
            if isinstance(begin, (int, torch.SymInt, sympy.Expr))
            else None
        )
        end_expr = (
            _to_sympy(end) if isinstance(end, (int, torch.SymInt, sympy.Expr)) else None
        )
        diff_expr = (
            sympy.Add(end_expr, sympy.Mul(-1, begin_expr))
            if begin_expr is not None and end_expr is not None
            else None
        )
        if step is None or step == 1:
            if diff_expr is not None:
                return diff_expr
            return f"(({self._expr_str(end)}) - ({self._expr_str(begin)}))"
        assert isinstance(step, (int, torch.SymInt, sympy.Expr))
        step_expr = _to_sympy(step)
        if getattr(step_expr, "free_symbols", None):
            return (
                f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
                f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
            )
        if diff_expr is not None:
            return sympy.ceiling(sympy.Mul(diff_expr, sympy.Pow(step_expr, -1)))
        return (
            f"((({self._expr_str(end)}) - ({self._expr_str(begin)})) + "
            f"({self._expr_str(step)}) - 1) // ({self._expr_str(step)})"
        )

    def _expr_str(self, value: object) -> str:
        if isinstance(value, (int, torch.SymInt, sympy.Expr)):
            return self.fn.sympy_expr(_to_sympy(value))
        return ast.unparse(self._to_ast(value))

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var
        elif (
            isinstance(pids, FlatProgramIDs)
            and env.backend.name == "pallas"
            and len(block_ids) >= 2
        ):
            pids = XYZProgramIDs()

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        ends: list[object]
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins = state.proxy_args[0]
            ends_arg = state.proxy_args[1]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)
        if isinstance(ends_arg, (list, tuple)):
            ends = list(ends_arg)
        else:
            ends = [ends_arg]
        assert len(ends) == len(block_ids)
        steps = self._root_grid_steps(state)

        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        self._cute_tv_thread_axis_offset = thread_axis_offset
        thread_axis_map = self._thread_axis_map()
        for i, (block_idx, block_size, begin, end, step) in enumerate(
            reversed(
                self._reorder(
                    [*zip(block_ids, block_sizes, begins, ends, steps, strict=True)]
                )
            )
        ):
            numel = self._range_numel_expr(begin, end, step)
            device_function = state.device_function
            dtype = env.index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if step not in (None, 1):
                step_ast = self._to_ast(step, to_dtype=dtype)
                # CuTe DSL preprocessor reserves ``step_<counter>`` (see comment
                # in ``TileStrategy.__init__``) — rename our lifted step var to
                # avoid the same UnboundLocalError that drove the offset rename.
                step_prefix = "tile_step" if env.backend.name == "cute" else "step"
                step_var = state.codegen.lift(step_ast, dce=True, prefix=step_prefix).id
                block_size_var = "1"
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}({pid_var}) * {step_var}"
                )
            elif block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")
            axis = thread_axis_offset + thread_axis_map[block_idx]
            # Inactive block_ids never claim a CUDA thread axis (per
            # ``_thread_axis_map``); without the polymorphic
            # ``_uses_thread_axis_for_block`` hook the grid would emit
            # ``thread_idx[axis]`` for them and collide with the inner
            # device-loop on the same axis.
            uses_thread_axis = step in (
                None,
                1,
            ) and self._uses_thread_axis_for_block(block_idx, block_size)
            bs = block_size_var if uses_thread_axis else "1"
            idx_expr = env.backend.grid_index_expr(offset_var, bs, dtype, axis=axis)
            if uses_thread_axis and isinstance(block_size, int):
                tracker.record(block_idx, axis, block_size)
            state.add_statement(f"{index_var} = {idx_expr}")
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                state.add_statement(mask_statement)
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        # Only use ends_override if there are data-dependent (tensor) bounds
        has_tensor_ends = any(isinstance(e, torch.Tensor) for e in ends)
        if has_tensor_ends:
            block_id_to_info = self._create_block_id_info_dict(
                state, ends_override=ends
            )
        else:
            block_id_to_info = self._create_block_id_info_dict(state)
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def _to_ast(self, x: object, to_dtype: str | None = None) -> ast.AST:
        if isinstance(x, ast.AST):
            if to_dtype:
                cast_expr = CompileEnvironment.current().backend.ast_to_dtype_expr(
                    "{value}", to_dtype
                )
                return expr_from_string(cast_expr, value=x)
            return x
        if isinstance(x, int):
            return expr_from_string(repr(x))
        if isinstance(x, sympy.Expr):
            from .device_function import DeviceFunction

            return expr_from_string(DeviceFunction.current().sympy_expr(x))
        if isinstance(x, torch.SymInt):
            return self._to_ast(x._sympy_())
        if isinstance(x, torch.Tensor):
            # Handle tensor values (for data-dependent bounds)
            # For scalar tensors, we need to load the value using tl.load
            from .device_function import DeviceFunction

            tensor_arg = DeviceFunction.current().tensor_arg(x)
            return expr_from_string(
                CompileEnvironment.current().backend.scalar_load_expr(tensor_arg.name)
            )
        if isinstance(x, str):
            # Already a string expression (for data-dependent numel)
            return expr_from_string(x)
        raise NotImplementedError(f"{type(x)} is not implemented.")

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        # TODO(jansel): refactor this to share code with codegen_grid
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        dtype = env.index_type()
        block_sizes = self.block_size
        body = innermost_body = []
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        if len(state.ast_args) == 5:
            _, begins, ends, _, steps = state.ast_args
        else:
            _, begins, ends, _ = state.ast_args
            steps = None
        _, _, proxy_ends, *_ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        if steps is None:
            steps = [None] * len(block_ids)
        assert isinstance(steps, list)
        assert isinstance(proxy_ends, list)
        block_id_to_info = {}
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        self._cute_tv_thread_axis_offset = thread_axis_offset
        thread_axis_map = self._thread_axis_map()
        for block_idx, block_size, begin, end, step, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, steps, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if step in (None, 1) and block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            begin_var_name = state.codegen.lift(
                self._to_ast(begin, to_dtype=dtype), dce=True, prefix="begin"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                begin_var_name=begin_var_name,
                begin_expr=_to_sympy(begin)
                if isinstance(begin, (int, torch.SymInt))
                else None,
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            # When the backend uses Python range() (e.g. Pallas), range
            # bounds must be plain Python ints — skip the dtype cast so
            # that concrete values stay as ints and are not wrapped in
            # backend-traced dtype conversions.
            range_dtype = None if env.backend.range_requires_python_int else dtype
            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=(
                            ast.unparse(self._to_ast(step, to_dtype=range_dtype))
                            if step not in (None, 1)
                            else block_size_var
                        ),
                    ),
                    begin=self._to_ast(begin, to_dtype=range_dtype),
                    end=self._to_ast(end, to_dtype=range_dtype),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            assert for_node.body is body
            # Inactive block_ids never claim a CUDA thread axis (per
            # ``_thread_axis_map``); see ``codegen_grid`` above for the
            # collision this guards against.
            uses_thread_axis = step in (
                None,
                1,
            ) and self._uses_thread_axis_for_block(block_idx, block_size)
            axis = thread_axis_offset + thread_axis_map[block_idx]
            bs = block_size_var if uses_thread_axis else "1"
            idx_expr = env.backend.loop_index_expr(offset_var, bs, dtype, axis=axis)
            if uses_thread_axis and isinstance(block_size, int):
                tracker.record(block_idx, axis, block_size)
            extra_body = [
                statement_from_string(f"{index_var} = {idx_expr}"),
            ]
            # pyrefly: ignore [missing-attribute]
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                extra_body.append(mask_statement)
            # pyrefly: ignore [unsupported-operation]
            body[:] = [*extra_body, *body]
            body = [for_node]
        assert for_node is not None
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=innermost_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        # TODO(jansel): we should combine size==1 dimensions here
        return shapes


class NDTileStrategy(_BaseNDTileStrategy):
    """Do up to 3D tiling using the kernel grid."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
        self.mask_vars: dict[int, str | None] = {}
        self.l2_grouping = l2_grouping

    def mask_var(self, block_idx: int) -> str | None:
        return self.mask_vars[block_idx]

    def load_mask_var(self, block_idx: int) -> str | None:
        """``mask_var``, except elided for reads when every launched thread is in range.

        See ``TileStrategy.load_mask_var`` for why reads and writes are treated
        asymmetrically.  This override supplies the *proof* for this strategy; it never
        touches ``mask_var``, so no store's predication can change.
        """
        if self.mask_vars.get(block_idx) is None:
            return None
        if not self._read_index_always_in_range(block_idx):
            return self.mask_vars[block_idx]
        return None

    def ragged_peel_plan(self, block_idx: int) -> tuple[int, int, int, str] | None:
        """``(numel, bulk_end, block_size, mask_var)`` if the axis's TAIL is peelable.

        ⭐ WHAT THIS ANSWERS, and why it is a *different* question from
        ``load_mask_var``.  ``_read_index_always_in_range`` proves the read mask vacuous
        over the WHOLE loop, and needs BOTH ``numel % B == 0`` and ``SPAN == B``.  When only
        the second holds, the mask is vacuous over the *prefix* ``[0, floor(numel/B)*B)``
        and non-vacuous only in the final, ragged iteration -- so the loop can be split into
        a mask-free bulk and a masked one-iteration tail.

        ``bulk_end = floor(numel / B) * B``.  For every ``tile_offset`` the bulk loop
        visits, ``tile_offset + B <= bulk_end <= numel``, and ``SPAN == B`` bounds a
        thread's displacement by ``B - 1``, so ``index <= tile_offset + B - 1 < numel``.
        That is the same arithmetic as ``_read_index_always_in_range``, with
        ``max(tile_offset) == bulk_end - B`` supplied by the split instead of by
        divisibility.

        Declines (returns ``None``) whenever a split would be pointless or unsound:
          * ``mask_var`` is already absent -- nothing to elide;
          * ``SPAN != B`` -- the surplus-thread problem, which no bound rewrite fixes;
          * no static extent (data-dependent / jagged) -- ``_static_read_numel`` screens it;
          * ``numel % B == 0`` -- ``load_mask_var`` already elides; a split would be dead
            code and would make the emitted source differ for no reason;
          * ``bulk_end == 0`` (``B > numel``) -- the *only* iteration is the ragged one, so
            there is no bulk to peel and the mask must stay.
        """
        mask_var = self.mask_vars.get(block_idx)
        if mask_var is None:
            return None
        numel = self._static_read_numel(block_idx)
        if numel is None:
            return None
        block_size = dict(zip(self.block_ids, self.block_size, strict=True))[block_idx]
        if not isinstance(block_size, int) or block_size <= 0:
            return None
        if numel % block_size == 0:
            return None
        if not self._read_index_span_covers_block(block_idx, block_size):
            return None
        bulk_end = (numel // block_size) * block_size
        if bulk_end <= 0:
            return None
        return (numel, bulk_end, block_size, mask_var)

    def _lane_elements_per_thread(self, block_idx: int, block_size: int) -> int | None:
        """How many consecutive elements of this axis does ONE thread cover?

        1 for this strategy: the index is ``tile_offset + thread_idx()[axis]``, so
        consecutive threads take consecutive elements.  ``CuteNDTileStrategy`` overrides
        this because it may wrap the tile in a *lane loop*, where one thread covers
        ``block_size // num_threads`` consecutive elements.  ``None`` means "cannot
        determine", which makes the caller decline.
        """
        return 1

    def _read_index_always_in_range(self, block_idx: int) -> bool:
        """Is ``index_var(block_idx) < numel`` true for EVERY THREAD THE LAUNCH CREATES?

        This is the whole soundness argument for the read-side elision, so it is stated as
        arithmetic rather than as a pattern match.

        **The index this strategy emits.**  ``codegen_grid`` / ``codegen_device_loop`` build
        the axis index as a tile offset plus a per-thread displacement::

            tile_offset in {0, B, 2B, ...}          # B = block_size
            index       = tile_offset + d,  0 <= d < SPAN

        where ``SPAN`` is the number of distinct displacements *threads that actually exist*
        can produce: ``LAUNCH_EXTENT * ELEMENTS_PER_THREAD``, i.e. ``blockDim[axis]`` times
        however many consecutive elements one thread covers (1 here; a lane loop's
        ``block_size // num_threads`` on ``CuteNDTileStrategy``).

        **The claim.**  If ``numel % B == 0`` *and* ``SPAN == B`` then every index an
        existing thread can form is in ``[0, numel)``.
        Proof: divisibility gives ``max(tile_offset) == numel - B``; ``SPAN == B`` gives
        ``d <= B - 1``; so ``index <= numel - 1``.  ∎
        Both conjuncts are load-bearing and neither implies the other.

        ⚠ ``SPAN == B`` IS THE CONJUNCT ``734eea2d9`` DID NOT HAVE, AND ITS ABSENCE IS THE
        WHOLE BUG.  ``thread_block_dims`` sizes each axis as the **elementwise max over
        every strategy that shares it** — a thread block is sized for its widest user — so a
        block of 8 on an axis another loop drives at 16 launches 16 threads, and the surplus
        8 form indices past their own tile.  MEASURED, not argued: an in-kernel violation
        counter on ``examples/split_k_barrier``
        (``_redfix2/repro/r3_a1_splitk_detect.py``) shows ``_BLOCK_SIZE_1 == 8`` against
        ``blockDim.z == 16`` and ``indices_1 < 64`` FIRING.  That mask is genuinely
        non-vacuous, so it was never elidable on any consumer.  ``734eea2d9`` instead asked
        ``block_size % thread_extent_for_block_id == 0``, which reads that block's *own*
        tracked extent (8, not 16) — a different quantity, and one that says nothing about
        the surplus threads.  This also explains why E070's proposed repair ("require exact
        coverage on every axis") changed nothing: it tightened a test on the wrong number.

        Conservative by construction; anything it cannot prove keeps the mask:
          * jagged tiles decline (the bound is a runtime parent extent, not ``numel``);
          * a non-``int`` block size, a symbolic ``numel``, an axis that resolves to no
            launch axis, or an unknown elements-per-thread all decline.

        ⭐ THE TWO CONJUNCTS ARE SEPARATELY NAMED, and deliberately so.  Only the *first*
        (``numel % B == 0``) is a property of the loop BOUNDS, and a loop split can
        manufacture it on a sub-range; the second (``SPAN == B``) is a property of the
        LAUNCH and no rewrite of the bounds can change it.  See
        ``_read_index_span_covers_block`` and ``cute/peel_ragged_tile.py``.
        """
        numel = self._static_read_numel(block_idx)
        if numel is None:
            return False
        block_size = dict(zip(self.block_ids, self.block_size, strict=True))[block_idx]
        if not isinstance(block_size, int) or block_size <= 0:
            return False
        if numel % block_size:
            return False
        return self._read_index_span_covers_block(block_idx, block_size)

    def _static_read_numel(self, block_idx: int) -> int | None:
        """The axis's static extent, or ``None`` when there is provably not one.

        Split out of ``_read_index_always_in_range`` so that every consumer of the
        elision proof screens a data-dependent extent the same way.

        ⚠ ``BlockSizeInfo.numel`` ASSERTS on a block whose ``size`` is an ``AutoSize`` or
        ``None`` -- a data-dependent extent -- so the size must be screened FIRST.  A tile
        over a data-dependent range has no static extent to compare against and can never
        be proved, so declining is both correct and the only option.  MEASURED: reading
        ``numel`` unguarded crashes ``examples/flex_attention`` and
        ``examples/jagged_dense_add`` in codegen (``test_examples.py`` step 6), neither of
        which is reduction-shaped -- which is exactly why that step exists.
        """
        env = CompileEnvironment.current()
        if env.is_jagged_tile(block_idx):
            return None
        block_info = env.block_sizes[block_idx]
        if not isinstance(block_info.size, (int, torch.SymInt)):
            return None
        numel_expr = block_info.numel
        if not isinstance(numel_expr, (int, sympy.Integer)):
            return None
        return int(numel_expr)

    def _read_index_span_covers_block(self, block_idx: int, block_size: int) -> bool:
        """The ``SPAN == B`` conjunct of ``_read_index_always_in_range``, alone.

        ``SPAN = LAUNCH_EXTENT * ELEMENTS_PER_THREAD`` is the number of distinct
        displacements a thread *that actually exists* can add to the tile offset.  This is
        a property of the LAUNCH, not of the loop bounds, so it is exactly the part of the
        proof that a loop split cannot help with and must therefore still hold.
        """
        if not self._uses_thread_axis_for_block(block_idx, block_size):
            # No thread axis: the emitters pass ``bs = "1"`` to ``grid_index_expr``, so the
            # index is ``tile_offset`` alone and the displacement is identically 0.  That is
            # ``SPAN == 1``, which meets the claim only for ``block_size == 1`` -- and for
            # ``block_size == 1`` the tile offsets already enumerate ``[0, numel)`` exactly.
            return block_size == 1
        tile_strategy = self.fn.tile_strategy
        axis = tile_strategy.thread_axis_for_block_id(block_idx)
        if axis is None:
            return False
        dims = tile_strategy.thread_block_dims()
        if axis >= len(dims):
            return False
        launch_extent = dims[axis]
        elements_per_thread = self._lane_elements_per_thread(block_idx, block_size)
        if (
            launch_extent <= 0
            or elements_per_thread is None
            or elements_per_thread <= 0
        ):
            return False
        return launch_extent * elements_per_thread == block_size

    def _setup_mask(
        self,
        state: CodegenState,
        block_idx: int,
        block_size: SymIntLike,
        index_var: str,
        end: object,
    ) -> ast.stmt | None:
        env = CompileEnvironment.current()
        if (
            not env.backend.force_tile_mask()
            and env.block_sizes[block_idx].known_multiple(block_size)
            and not env.is_jagged_tile(block_idx)
        ):
            self.mask_vars[block_idx] = None
            return None
        self.mask_vars[block_idx] = mask_var = self.fn.new_var(
            f"mask_{block_idx}", dce=True
        )

        if env.is_jagged_tile(block_idx):
            jagged_tile_parents_ast = state.ast_args[3]
            jagged_tile_parents_proxy = state.proxy_args[3]
            assert isinstance(jagged_tile_parents_ast, list)
            assert isinstance(jagged_tile_parents_proxy, list)
            # We guarantee the first lifted loop input is the jagged_tile parent tensor.
            jagged_tile_parent = jagged_tile_parents_ast[0]
            jagged_tile_block_size = env.block_sizes[block_idx].var
            jagged_tile_parent_proxy = jagged_tile_parents_proxy[0]
            assert isinstance(jagged_tile_parent_proxy, torch.Tensor)
            parent_dims: list[torch.SymInt] = []
            for d in jagged_tile_parent_proxy.size():
                assert isinstance(d, torch.SymInt)
                parent_dims.append(d)
            assert len(parent_dims) >= 1
            env.jagged_tile_mask_shapes[block_idx] = [
                *parent_dims,
                jagged_tile_block_size,
            ]
            if not self.supports_index_rank_expansion():
                return statement_from_string(
                    f"{mask_var} = ({index_var}) < {{parent}}",
                    parent=self._to_ast(jagged_tile_parent),
                )
            k = len(parent_dims)
            child_expand = "[" + ", ".join(["None"] * k + [":"]) + "]"
            parent_expand = "[" + ", ".join([":"] * k + ["None"]) + "]"
            return statement_from_string(
                f"{mask_var} = ({index_var}){child_expand} < {{parent}}{parent_expand}",
                parent=self._to_ast(jagged_tile_parent),
            )

        return statement_from_string(
            f"{mask_var} = ({index_var}) < {{end}}", end=self._to_ast(end)
        )

    def select_pid_strategy(self) -> ProgramIDs:
        if self.l2_grouping > 1:
            return L2GroupingProgramIDs(
                group_size=self.l2_grouping,
                parent_strategy=super().select_pid_strategy(),
            )
        return super().select_pid_strategy()


class CuteNDTileStrategy(NDTileStrategy):
    """CuTe N-D tile strategy using the standard tile pipeline."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        l2_grouping: int,
        num_threads: list[int] | None = None,
        mma_mode: bool = False,
        inactive_block_ids: set[int] | None = None,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order, l2_grouping)
        assert isinstance(block_size, list)
        if num_threads is None:
            num_threads = [0 for _ in block_ids]
        assert len(num_threads) == len(block_ids)
        self.num_threads = num_threads
        self.mma_mode = mma_mode
        self.inactive_block_ids = inactive_block_ids or set()
        self._lane_var_by_block: dict[int, str] = {}
        # Per-block vec width for the lane loop (1 = scalar).  Populated
        # from the autotuner-selected ``cute_vector_widths`` config when
        # the block has a lane loop and its ``elements_per_thread`` is
        # divisible by the picked V.  When > 1, ``codegen_device_loop``
        # partitions the lane loop into outer (epT/V) x inner constexpr V
        # so memory_ops can hoist a single ``cute.arch.load(..., V)`` per
        # outer-lane iter (LDG.64 / LDG.128).
        self._cute_lane_vec_width_by_block: dict[int, int] = {}
        # Per-block constexpr V-loop var (only set when the lane loop is
        # vec-partitioned). Used by memory_ops to find the inner loop's
        # target var when emitting per-lane bitcasts.
        self._cute_vec_lane_var_by_block: dict[int, str] = {}
        # Per-block lane-base index var (the per-thread base of a V-wide
        # contiguous chunk).  Set when lane vec is in play; used by
        # memory_ops to compute the vec load pointer once per outer-lane
        # iter.
        self._cute_lane_base_index_var_by_block: dict[int, str] = {}
        # Per-block lane body (list of AST statements inside the outer
        # lane loop, ending in the constexpr V-loop). memory_ops uses
        # ``insert(len(lane_body)-1, hoist_stmt)`` to splice the vec
        # load just before the inner V-loop.
        self._cute_lane_body_by_block: dict[int, list] = {}
        # Shared per-block hoist cache: (tensor_name, base_ptr_expr) ->
        # (hoist_var, dtype).  Same shape as
        # ``LoopedReductionStrategy._cute_lane_vec_loads``.
        self._cute_lane_vec_loads_by_block: dict[int, dict] = {}
        # ── the RAGGED-N round-up, per participating block ─────────────────────────
        #
        # ``block_id -> the axis's TRUE extent N``, recorded by
        # ``_build_cute_tv_plan_for_block`` for exactly the blocks whose tile it let
        # OVERSHOOT.  Absence means "this axis's tile is exact", which is the inert
        # answer and is what every non-ragged block reads.
        #
        # ⚠ IT RECORDS ``N``, NOT "IS RAGGED", because the predicate needs the number.
        # A bool would force ``cute_tv_tail_predicate`` to re-derive ``N`` from the
        # block sizes, i.e. two readings of one fact -- and the rolled path's own
        # ``cute_tv_rounded_extent`` docstring is about exactly that hazard (the mask is
        # created from the REQUESTED cluster while the loop bound comes from the EMITTED
        # one, and only a single reading keeps them from disagreeing).
        self._cute_tv_tail_numel_by_block: dict[int, int] = {}
        # ── capability ①: the TV plan, per participating block ────────────────
        #
        # Populated in ``codegen_device_loop`` (NOT here): a plan is only useful
        # alongside the lane body / constexpr-V loop / chunk prefix it is emitted
        # against, and those three exist only once that method has built them.
        # Building the plan here and the scaffolding there is exactly how the two
        # could come to disagree, so they are produced together, in one place.
        self._cute_tv_plan_by_block: dict[int, ChunkTVPlan] = {}
        # Per-block emission scaffolding for the TV protocol.  Same roles as
        # ``ReductionStrategy``'s single-valued ``_cute_tv_chunk_prefix`` /
        # ``_cute_tv_constexpr_loop`` / ``_cute_tv_chunk_index_var``; keyed by
        # block because ONE NDTile strategy drives several axes and only some of
        # them carry a lane loop.  The computed properties below project them onto
        # the single-valued protocol for the ACTIVE block.
        self._cute_tv_chunk_prefix_by_block: dict[int, list] = {}
        self._cute_tv_constexpr_loop_by_block: dict[int, ast.For] = {}
        self._cute_tv_chunk_index_var_by_block: dict[int, str] = {}
        # The chunk index EXPRESSION, per block -- see the assignment site for why the
        # variable name alone cannot identify a re-read of the same tile.
        self._cute_tv_chunk_index_expr_by_block: dict[int, str] = {}
        # The OUTER lane loop node, held by reference so the chunk prefix can be
        # located by IDENTITY rather than by list position: the serial-loop body is
        # assembled after the lane loops and other statements are prepended to it,
        # so any index recorded here would move.
        self._cute_tv_outer_lane_loop_by_block: dict[int, ast.For] = {}
        # ⭐ THE ACTIVE BLOCK, set by the CONSUMER at the load site and by nothing
        # else.  See ``cute_tv_lane_block_id`` for why this is not inferred.
        self._cute_tv_active_block: int | None = None
        # (atom_var, thr_var) for THE one layout, and the per-(tensor, S|D)
        # fragment cache -- both per block, for the same reason as the rest.
        self._cute_tv_shared_by_block: dict[int, tuple[str, str]] = {}
        # A7c: per-(block, dtype) atom/slice cache.  Keyed by block for the same reason
        # ``_cute_tv_shared_by_block`` is -- one instance, up to three independent layouts.
        self._cute_tv_shared_by_dtype_by_block: dict[
            int, dict[str, tuple[str, str]]
        ] = {}
        # ⛔ The CUDA thread-axis OFFSET in force at emission time, recorded rather than
        # recomputed: ``cute_tv_thread_axis`` needs it and has no ``CodegenState``.  See that
        # method for the wrong answer its absence caused (253952/262144 elements unwritten).
        # 0 is correct until an emission site records otherwise -- a kernel with nothing
        # reserving a thread axis has offset 0, which is every pure-pointwise shape.
        self._cute_tv_thread_axis_offset: int = 0
        self._cute_tv_partitions_by_block: dict[int, dict[tuple[str, str], str]] = {}
        # ── capability ③: the SMEM staging state, per block ────────────────────
        #
        # ⭐ THE NINE STAGING METHODS LIVE ON ``TileStrategy`` NOW, BUT THEIR STATE DOES
        # NOT COME FOR FREE.  ``memory_ops._cute_tv_stage_slice`` dereferences these on
        # whatever strategy it is handed, and its first act is to read
        # ``_cute_tv_reload_from``.  MEASURED when the class gate was removed before these
        # existed: ``AttributeError: 'CuteNDTileStrategy' object has no attribute
        # '_cute_tv_reload_from'`` inside ``codegen for node load`` -- i.e. exactly the
        # "a missed optimisation becomes a compiler CRASH" failure the capability query
        # exists to prevent.  A strategy that can be ASKED must be able to ANSWER.
        #
        # ⚠ PER BLOCK, like the rest of this class's TV state, because one NDTile strategy
        # drives up to three axes and staging describes ONE row.  The single-valued
        # protocol names below are computed properties projecting these onto the ACTIVE
        # block, exactly as ``_cute_tv_partitions`` already does.
        #
        # ⚠ MINTED HERE AND NOT AS CLASS ATTRIBUTES.  A mutable class-level default is ONE
        # object shared by every strategy instance in the process, so a partition recorded
        # by one kernel would be visible to the next -- and the emitted symbol names
        # (``_tv_spart_0``) repeat across kernels, so it would silently HIT.
        # ``ReductionStrategy`` documents the same trap for the same fields.
        self._cute_tv_stage_partitions_by_block: dict[
            int, dict[tuple[str, str], str]
        ] = {}
        self._cute_tv_staged_tensors_by_block: dict[int, set[str]] = {}
        # The ``registers`` analogue, block-keyed for the same reason: this strategy serves
        # several reduction blocks and each owns its own tiles.
        self._cute_tv_rmem_frag_by_tile_by_block: dict[
            int, dict[tuple[object, ...], str]
        ] = {}
        self._cute_tv_stage_smem_var_by_block: dict[int, str] = {}
        self._cute_tv_reload_from_by_block: dict[int, str | None] = {}
        # The REQUESTED residency, resolved once per block in
        # ``_setup_cute_tv_chunk_prefix`` (where the plan and the geometry are both final),
        # and the CAUSE of a decline.  Seeded absent; the properties below report the
        # base-class sentinels until then, which read as "no mechanism".
        self._cute_row_residency_requested_by_block: dict[int, str] = {}
        if not mma_mode:
            env_local = CompileEnvironment.current()
            cute_vec_widths_cfg = cast(
                "list[int]",
                fn.config.config.get("cute_vector_widths", []) or [],
            )
            for block_id, nt, bs in zip(
                block_ids, num_threads, block_size, strict=True
            ):
                if block_id in self.inactive_block_ids:
                    continue
                static_bs = self._configured_block_size_int(bs)
                if (
                    nt > 0
                    and static_bs is not None
                    and static_bs > nt
                    and static_bs % nt == 0
                ):
                    self._lane_var_by_block[block_id] = self.fn.new_var(
                        f"lane_{block_id}"
                    )
                    elements_per_thread = static_bs // nt
                    # Vec slot is registered eagerly in device-IR analysis; read
                    # the tuned V.  Never append here — growing the spec during
                    # codegen breaks the autotuner's fixed-width unflatten.
                    if (
                        block_id
                        in env_local.config_spec.cute_vector_widths.valid_block_ids()
                    ):
                        vec_width = env_local.config_spec.cute_vector_widths.config_get(
                            cute_vec_widths_cfg,
                            block_id,
                            1,
                        )
                        # NARROW to the widest V that divides EPT, rather than
                        # dropping the vector load entirely.
                        #
                        # This used to be ``elements_per_thread % vec_width == 0`` with no
                        # else-branch, i.e. a requested V that did not divide EPT left the
                        # block ABSENT from ``_cute_lane_vec_width_by_block`` -- and
                        # ``_cute_vector_load_ctx``'s tile branch then bails on
                        # ``vec_width <= 1`` (``cute/memory_ops.py:2582``), so the whole
                        # load degrades to per-element work.  MEASURED (LEDGER E048) on
                        # ``cross_entropy_online`` at ``bi=512, V=8``:
                        #
                        #     nt    EPT   8 | EPT?   cute.arch.load sites   ratio
                        #     32     16    yes              2              0.815
                        #     64      8    yes              3              0.657
                        #     128     4    NO               0              0.215  <- cliff
                        #     256     2    NO               0              0.202
                        #
                        # A 3.8x cliff off the best, with nothing reporting it -- the same
                        # silent-scalar failure mode as LEDGER E012 (where the ROLLED path's
                        # width was discarded and two 400-config sweeps measured a flat
                        # surface as a result), reached here by a different trigger.
                        #
                        # The ROLLED path already does the right thing: ``chunk_plan``
                        # (``cute/tv_layout.py:880``) runs ``while vec > 1 and chunk %
                        # (tpr * vec): vec //= 2``, i.e. it NARROWS to what is legal.  This
                        # is that same loop, on the tile path's own divisor (EPT).  ``V`` is
                        # documented as a CAP, not a request (``PORT_SPEC_layout.md`` 5c), so
                        # lowering it is exactly the contract -- and every V it can settle on
                        # is a power of two dividing EPT, so the lane arithmetic stays exact.
                        if isinstance(vec_width, int) and vec_width > 1:
                            while vec_width > 1 and elements_per_thread % vec_width:
                                vec_width //= 2
                            if vec_width > 1:
                                self._cute_lane_vec_width_by_block[block_id] = vec_width
                    else:
                        # Metal shares this strategy without vec-width tuning,
                        # and non-static sizes are eager-skipped — both run
                        # scalar.  A missing static slot on cute is a bug.
                        assert env_local.backend_name != "cute" or not isinstance(
                            env_local.block_sizes[block_id].size,
                            (int, torch.SymInt),
                        ), (
                            f"cute_vector_widths slot missing for static-size "
                            f"block_id={block_id}; it must be registered during "
                            f"device-IR analysis, not lazily during codegen"
                        )

    # ══════════════════════════════════════════════════════════════════════════
    # CuTe TV-layout capability ① on a NON-REDUCTION looping strategy.
    #
    # ⭐ THE ONE STRUCTURAL PROBLEM, AND HOW IT IS SOLVED.  The TV emission
    # protocol that ``memory_ops._cute_tv_partition_hoist`` dereferences is
    # SINGLE-VALUED (``_cute_tv_plan``, ``_cute_lane_body``, ...), because a
    # ``ReductionStrategy`` owns exactly ONE axis.  This strategy owns up to three,
    # and only some of them carry a lane loop -- so all of its state is keyed by
    # block id.
    #
    # The resolution is a set of COMPUTED PROPERTIES that project the per-block
    # dicts onto the protocol for ONE block: ``_cute_tv_active_block``.  Two
    # properties of that choice are what make it sound rather than convenient:
    #
    #   * ⭐ THE ACTIVE BLOCK IS SET BY THE CONSUMER, NOT INFERRED HERE.  The load
    #     site has already resolved which axis it is addressing -- that is
    #     ``_cute_vector_load_ctx``'s ``inner_block_id``, which comes from the
    #     INDEXER's own binding (``slice_block_ids``), MEASURED 23/23 correct where
    #     a size scan mis-binds 3/23 (LEDGER E052/E053).  Re-deriving it here would
    #     be a second opinion that can disagree with the address actually emitted,
    #     and the axis is what selects the width.  So the consumer states it and
    #     this class stores it.
    #   * ⚠ NOT ``isinstance`` ANYWHERE.  The protocol stays exactly the six fields
    #     plus the tail predicate; this class supplies each one, so the consumer
    #     still asks a QUESTION.  Widening the consumer with a branch per strategy
    #     class is this repo's documented enumeration antipattern (six soundness
    #     bugs in one week, every one "correct for every case the author
    #     enumerated").
    # ══════════════════════════════════════════════════════════════════════════

    def cute_tv_lane_block_id(self) -> int | None:
        """The block whose TV plan the protocol properties below resolve to.

        ``None`` when the consumer has not stated an active block, or has stated one
        with no plan -- both of which read as a decline, so nothing is emitted
        against scaffolding that does not exist.
        """
        block_id = self._cute_tv_active_block
        if block_id is None or block_id not in self._cute_tv_plan_by_block:
            return None
        return block_id

    def cute_tv_set_active_block(self, block_id: int) -> bool:
        """Point the protocol properties at ``block_id``; True iff it has a plan.

        Called by the load/store site with the block IT resolved (see the block
        comment above on why the direction is consumer -> strategy and never the
        reverse).  Returning a bool rather than raising keeps a stated-but-planless
        block a DECLINE: the caller falls back to its legacy mode.
        """
        if block_id not in self._cute_tv_plan_by_block:
            return False
        self._cute_tv_active_block = block_id
        return True

    @property
    def _cute_tv_plan(self) -> ChunkTVPlan | None:
        block_id = self.cute_tv_lane_block_id()
        return None if block_id is None else self._cute_tv_plan_by_block[block_id]

    def cute_tv_rounded_extent(self) -> int | None:
        """``N'`` when THIS axis's tile was rounded up past ``N``, else ``None``.

        The tile-path analogue of ``ReductionStrategy.cute_tv_rounded_extent``.
        ``None`` -- "the tile is exact, nothing to predicate" -- is what every
        non-ragged block reads, which keeps the divisible path untouched.

        ``cluster_n=1``: this strategy emits no cluster, so the tile granularity is
        ``chunk`` alone.
        """
        block_id = self.cute_tv_lane_block_id()
        if block_id is None:
            return None
        numel = self._cute_tv_tail_numel_by_block.get(block_id)
        if numel is None:
            return None
        plan = self._cute_tv_plan_by_block[block_id]
        rounded = rounded_extent(numel, plan.chunk, 1)
        return None if rounded == numel else rounded

    def cute_tv_tail_predicate(self) -> str | None:
        """``base < N`` for the current lane, or ``None`` when no tail exists.

        ⭐ THE HOOK THIS CLASS USED TO LEAVE INERT, and the in-code comment claiming
        that was "not an omission" was measured FALSE -- it was the sole reason two of
        the 40 frozen cells could not reach TV.  Deliberately the SAME shape as
        ``ReductionStrategy.cute_tv_tail_predicate``: source text comparing the emitted
        per-thread column base against the true extent, spliced around the
        ``cute.copy`` by ``memory_ops._cute_tv_partition_hoist`` -- which asks whatever
        strategy answered ``cute_tv_capable()``, so no consumer change was needed.

        ⭐ ONE SCALAR COMPARE IS EXACT FOR THE WHOLE FRAGMENT (``ragged_tail``
        invariant I2): the variable compared is a multiple of ``vec``, so a vector block
        is wholly in or wholly out of bounds -- which matters because ``cute.copy``'s
        ``pred`` granularity is one whole block and cannot mask part of one.
        :func:`build_tv_plan`'s ``tail_predicated`` arm is what guarantees ``vec | N``;
        the assert below states it at the point of emission.

        ⚠ Returns ``None`` if the lane base var does not exist yet.  That is not
        defensive: the var is minted LATER in ``codegen_device_loop`` than the plan is
        built, so a caller that asks too early must get "nothing to predicate" rather
        than a reference to an undeclared name.  The plan builder refuses the round-up
        unless the constexpr-V var (which IS set before it) is present, so the two
        checks agree.
        """
        if self.cute_tv_rounded_extent() is None:
            return None
        block_id = self.cute_tv_lane_block_id()
        assert block_id is not None
        base_var = self._cute_lane_base_index_var_by_block.get(block_id)
        if base_var is None:
            return None
        numel = self._cute_tv_tail_numel_by_block[block_id]
        assert_vec_divides_extent(self._cute_tv_plan_by_block[block_id].vec, numel)
        return f"{base_var} < {numel}"

    @property
    def _cute_lane_body(self) -> list[ast.AST] | None:
        block_id = self.cute_tv_lane_block_id()
        return None if block_id is None else self._cute_lane_body_by_block.get(block_id)

    @property
    def _cute_tv_constexpr_loop(self) -> ast.For | None:
        block_id = self.cute_tv_lane_block_id()
        if block_id is None:
            return None
        return self._cute_tv_constexpr_loop_by_block.get(block_id)

    @property
    def _cute_tv_chunk_prefix(self) -> list[ast.AST] | None:
        block_id = self.cute_tv_lane_block_id()
        if block_id is None:
            return None
        return self._cute_tv_chunk_prefix_by_block.get(block_id)

    @property
    def _cute_tv_chunk_index_var(self) -> str | None:
        block_id = self.cute_tv_lane_block_id()
        if block_id is None:
            return None
        return self._cute_tv_chunk_index_var_by_block.get(block_id)

    @property
    def _cute_tv_chunk_index_expr(self) -> str | None:
        """The chunk index as an EXPRESSION rather than this sweep's variable name.

        ⭐ ``tv_tile_ids`` prefers this precisely so two sweeps of one row produce the SAME
        tile id: every sweep mints its own ``_tv_chunk_N`` holding the same value, so a
        name-keyed id reports N distinct tiles and no re-read can ever be recognised.
        """
        block_id = self.cute_tv_lane_block_id()
        if block_id is None:
            return None
        return self._cute_tv_chunk_index_expr_by_block.get(block_id)

    @property
    def _cute_reduction_lane_var(self) -> str | None:
        """The OUTER lane loop's var -- the copies' lane slice.

        ⚠ THE OUTER LOOP, NOT THE CONSTEXPR-V ONE.  ``plan.emit_lane_slice`` indexes
        the partition's REST mode, whose extent is ``plan.lane_extent`` --
        i.e. exactly the outer lane loop's trip count (``EPT // V``), which
        ``codegen_device_loop`` builds from the same ``vec_width``.  The inner
        ``range_constexpr(V)`` var indexes the FRAGMENT, and is what
        ``_cute_tv_vec_lane_var`` reads off ``_cute_tv_constexpr_loop``.  Swapping
        them would slice one mode with the other's index.
        """
        block_id = self.cute_tv_lane_block_id()
        return None if block_id is None else self._lane_var_by_block.get(block_id)

    @property
    def _cute_tv_shared(self) -> tuple[str, str] | None:
        block_id = self.cute_tv_lane_block_id()
        return None if block_id is None else self._cute_tv_shared_by_block.get(block_id)

    @_cute_tv_shared.setter
    def _cute_tv_shared(self, value: tuple[str, str]) -> None:
        block_id = self.cute_tv_lane_block_id()
        assert block_id is not None, (
            "_cute_tv_shared was assigned with no active TV block; the hoist must "
            "have been reached without ``cute_tv_capable()`` returning True"
        )
        self._cute_tv_shared_by_block[block_id] = value

    @property
    def _cute_tv_shared_by_dtype(self) -> dict[str, tuple[str, str]]:
        """Per-(ACTIVE BLOCK, dtype) atom cache -- see the base for why it is per dtype.

        Scoped by block as well because this strategy drives up to three tiled axes and each
        has its OWN layout (different ``chunk``/``tpr``/``vec``), so an atom minted for one
        axis must never be reused on another -- the base's flat dict would do exactly that.
        """
        block_id = self.cute_tv_lane_block_id()
        assert block_id is not None, (
            "_cute_tv_shared_by_dtype was read with no active TV block; the hoist must "
            "have been reached without ``cute_tv_capable()`` returning True"
        )
        return self._cute_tv_shared_by_dtype_by_block.setdefault(block_id, {})

    @property
    def _cute_tv_partitions(self) -> dict[tuple[str, str], str]:
        block_id = self.cute_tv_lane_block_id()
        assert block_id is not None, (
            "_cute_tv_partitions was read with no active TV block; the hoist must "
            "have been reached without ``cute_tv_capable()`` returning True"
        )
        return self._cute_tv_partitions_by_block.setdefault(block_id, {})

    # ══════════════════════════════════════════════════════════════════════════
    # CAPABILITY ③ -- the staging protocol, projected onto the ACTIVE block.
    #
    # Same pattern and same reason as the block above: ``memory_ops`` reads
    # single-valued names, this class owns one row per axis.  Every one of these is a
    # field ``_cute_tv_stage_slice`` dereferences.
    # ══════════════════════════════════════════════════════════════════════════

    def cute_stage_widest_dtype_bits(self) -> int:
        """The ACTIVE block's participants -- see the base for why the charge needs this."""
        block_id = self.cute_tv_lane_block_id()
        if block_id is None:
            return 0
        return max(
            (b for b in self._cute_tv_participants(block_id).dtype_bits if b > 0),
            default=0,
        )

    def cute_tv_thread_axis(self) -> int:
        """The ACTIVE block's thread axis -- see ``TileStrategy.cute_tv_thread_axis``.

        A reduction can hardcode 0 because its row axis is outside the layout; this strategy
        drives up to three tiled axes, so the copy's slice index must name the axis of the
        block whose plan is active.

        ⛔⛔ IT IS ``thread_axis_offset + _thread_axis_map[bid]``, AND THE OFFSET TERM IS THE
        WHOLE BUG THIS METHOD ONCE HAD.  This returned ``_thread_axis_map()[bid]`` alone while
        both emission sites compute ``axis = thread_axis_offset + thread_axis_map[block_idx]``.
        The offset is 0 for a pure pointwise kernel -- so the omission was arithmetically
        invisible on every shape the A7a tests cover -- and NON-ZERO as soon as anything
        reserves a thread axis, e.g. any reduction in the same kernel.  MEASURED on
        ``out[tm,tn] = x[tm,tn] + y[tm,tn] + w[tm,:].sum(-1)[:,None]`` at
        ``block_sizes=[32,512]``: the copy was sliced by ``thread_idx()[1]`` (``tile_m``'s axis)
        while its own addresses counted on ``thread_idx()[2]`` (``tile_n``'s), and **253952 of
        262144 output elements were never written** -- a silent wrong answer, autotuner
        reachable through the searchable ``cute_ndtile_tv``.

        ⇒ the offset is RECORDED at emission time (``_cute_tv_thread_axis_offset``) rather than
        recomputed here, because ``_thread_axis_offset`` needs the ``CodegenState`` this method
        does not have.  Reading a value the emission site stored is also what makes the two
        agree by construction instead of by two parallel derivations -- which is exactly the
        failure this docstring used to claim was impossible.
        """
        block_id = self.cute_tv_lane_block_id()
        if block_id is None:
            return 0
        return self._cute_tv_thread_axis_offset + self._thread_axis_map().get(
            block_id, 0
        )

    def cute_stage_block_id(self) -> int | None:
        """⭐ THE ACTIVE block, not ``block_ids[0]`` -- see ``TileStrategy``'s default.

        This is the override that makes the nine moved staging methods FUNCTIONAL here:
        they read this instead of ``self.block_index`` precisely because ``block_ids[0]``
        would name whichever of this strategy's up-to-three axes sorted first rather than
        the axis the copy addresses.  Answering with the active block means the row the
        staging methods describe is the row the emission actually reads.
        """
        return self.cute_tv_lane_block_id()

    @property
    def _cute_tv_stage_partitions(self) -> dict[tuple[str, str], str]:
        block_id = self.cute_stage_block_id()
        assert block_id is not None, (
            "_cute_tv_stage_partitions was read with no active TV block; staging must "
            "have been reached without ``cute_tv_capable()`` returning True"
        )
        return self._cute_tv_stage_partitions_by_block.setdefault(block_id, {})

    @property
    def _cute_tv_staged_tensors(self) -> set[str]:
        """⚠ NOT reset per chunk body, unlike the partitions -- see the reset site.

        This records that a tensor was PUBLISHED in an earlier sweep, which is exactly what
        makes a later sweep's read legal.  Clearing it would make every sweep believe it
        was the first, so each would re-publish and none would read.
        """
        block_id = self.cute_stage_block_id()
        assert block_id is not None, (
            "_cute_tv_staged_tensors was read with no active TV block; staging must "
            "have been reached without ``cute_tv_capable()`` returning True"
        )
        return self._cute_tv_staged_tensors_by_block.setdefault(block_id, set())

    @property
    def _cute_tv_rmem_frag_by_tile(self) -> dict[tuple[object, ...], str]:
        """The ``registers`` analogue of :attr:`_cute_tv_staged_tensors`, block-keyed.

        ⚠ NOT reset per chunk body, for exactly the reason given above: it records that a
        tile's lanes were PUBLISHED into the register cache by an earlier sweep, which is what
        makes a later sweep's read legal.

        ⛔ THE CLASS-LEVEL SENTINEL IS ``None`` AND THAT IS NOT ENOUGH ON ITS OWN.  MEASURED:
        with only the sentinel on ``TileStrategy``, this strategy reached the rmem branch with
        ``by_tile=None``, so the branch DECLINED every time and ``fuse_tv_copy_sweeps`` was
        still doing the work -- the mechanism silently not firing on the tile path while
        reporting no error at all.  ⇒ a sentinel keeps a missing attribute from crashing; only
        a real per-block dict makes the capability WORK here.
        """
        block_id = self.cute_stage_block_id()
        assert block_id is not None, (
            "_cute_tv_rmem_frag_by_tile was read with no active TV block; the register "
            "residency must have been reached without ``cute_tv_capable()`` returning True"
        )
        return self._cute_tv_rmem_frag_by_tile_by_block.setdefault(block_id, {})

    @property
    def _cute_tv_stage_smem_var(self) -> str | None:
        block_id = self.cute_stage_block_id()
        return (
            None
            if block_id is None
            else self._cute_tv_stage_smem_var_by_block.get(block_id)
        )

    @_cute_tv_stage_smem_var.setter
    def _cute_tv_stage_smem_var(self, value: str) -> None:
        block_id = self.cute_stage_block_id()
        assert block_id is not None, (
            "_cute_tv_stage_smem_var was assigned with no active TV block"
        )
        self._cute_tv_stage_smem_var_by_block[block_id] = value

    @property
    def _cute_tv_reload_from(self) -> str | None:
        block_id = self.cute_stage_block_id()
        return (
            None
            if block_id is None
            else self._cute_tv_reload_from_by_block.get(block_id)
        )

    @_cute_tv_reload_from.setter
    def _cute_tv_reload_from(self, value: str | None) -> None:
        block_id = self.cute_stage_block_id()
        assert block_id is not None, (
            "_cute_tv_reload_from was assigned with no active TV block"
        )
        self._cute_tv_reload_from_by_block[block_id] = value

    @property
    def _cute_row_residency_requested(self) -> str:
        """The REQUEST.  ``gmem`` -- "no mechanism" -- until a chunk prefix resolves it.

        Matches ``ReductionStrategy``'s class default, so a strategy whose plan was never
        built reads as a decline rather than as a missing attribute.
        """
        block_id = self.cute_stage_block_id()
        if block_id is None:
            return ROW_RESIDENCY_GMEM
        return self._cute_row_residency_requested_by_block.get(
            block_id, ROW_RESIDENCY_GMEM
        )

    @_cute_row_residency_requested.setter
    def _cute_row_residency_requested(self, value: str) -> None:
        block_id = self.cute_stage_block_id()
        assert block_id is not None, (
            "_cute_row_residency_requested was assigned with no active TV block"
        )
        self._cute_row_residency_requested_by_block[block_id] = value

    @property
    def _cute_tv_stage_chunk_index_var(self) -> str | None:
        """⭐ THE CTA-LOCAL CHUNK COORDINATE, WHICH HERE IS THE GLOBAL ONE (plan item 4).

        The staged tile's ``local_tile`` column coordinate.  On the looped path this is a
        distinct ``_tv_chunklocal_N`` var whenever a cluster is in play, because a clustered
        staged tile spans only this CTA's share of the row and a global chunk number would
        index past its end.  **This strategy has no cluster mechanism at all** (capability ②
        is not implemented here and ``_cute_cluster_n_emitted`` stays at its base 1), so the
        CTA's share IS the whole row and the two coordinates coincide -- exactly the looped
        path's no-cluster branch, which reuses the same var for the same reason.

        ⚠ IF ② EVER ARRIVES HERE, THIS MUST GROW THE SUBTRACTION (``chunk - cluster_y *
        chunks_per_cta``).  Returning the global var under a cluster would index off the end
        of the buffer for every peer of rank > 0 -- a silent wrong answer, and a FAST one.
        The gate test ``reload_smem_is_chunk_indexed`` asserts the staged ``local_tile``'s
        column is a ``_tv_chunk*``-prefixed var, which this satisfies either way, so that
        guard would NOT catch the mistake.
        """
        return self._cute_tv_chunk_index_var

    def cute_tv_capable(self) -> bool:
        """The SAME six-field conjunction ``ReductionStrategy`` answers.

        Deliberately spelled as the identical question rather than as "am I an
        NDTile strategy with vec widths": every conjunct is a field
        ``_cute_tv_partition_hoist`` dereferences, resolved through the computed
        properties above, so a True here means an insert would land somewhere that
        exists -- which is all the consumer needs to know.
        """
        return (
            self._cute_tv_plan is not None
            and self._cute_lane_body is not None
            and self._cute_tv_constexpr_loop is not None
            and self._cute_tv_chunk_prefix is not None
            and self._cute_tv_chunk_index_var is not None
            and self._cute_reduction_lane_var is not None
        )

    def _cute_tv_participants(self, block_id: int) -> TVParticipants:
        """The tensors this kernel accesses along ``block_id``, for the plan builder.

        The NDTile analogue of ``ReductionStrategy._cute_layout_participants``, and
        it is a SEPARATE walk rather than a widening of that one on purpose.

        ⭐ IT RETURNS THE SHARED :class:`tv_layout.TVParticipants`, so the two walks
        differ only in WHICH accesses they select -- never in how the selection is
        then interpreted.  The interpretation (one dtype? which stride fold? which
        width bound?) lives once, in :func:`tv_layout.build_tv_plan`.

        ⚠ WHY NOT WIDEN THE SHARED ONE.  That selector recognises its axis by the
        subscript's *syntax* -- ``any(isinstance(idx, slice) and idx ==
        slice(None))`` -- which is unambiguous for a rolled reduction, where the
        axis is the only bare slice in play.  Here the axis is a ``SymInt`` naming a
        specific block, and several of this strategy's own blocks appear as
        ``SymInt``s in the same subscript.  So the predicate has to be
        ``env.get_block_id(idx) == block_id`` -- a question about identity, not
        form.  Making the shared walk answer both would give it two notions of "its
        axis" and one caller for each, which is precisely the coupling that made
        widening this path hard in the first place.

        ⚠ IN THE DEVICE IR A SUBSCRIPT ENTRY IS AN ``fx.Node``, NOT A ``SymInt``.
        MEASURED on this exact kernel: the entries arrive as
        ``[Node:sym_size_int, Node:block_size_1]`` whose ``meta["val"]`` are the
        SymInts ``u0``/``u1``.  A bare ``isinstance(idx, torch.SymInt)`` test
        therefore matches NOTHING here, and this walk would return ``([], [])`` for
        every kernel -- indistinguishable, from the caller, from an honest decline.
        Hence the unwrapping lives in :func:`_tiled_axis_block_id`, shared with the
        codegen-time gate that sees the SymInt directly: one function, both
        representations, so neither caller can be silently vacuous.

        ⭐ AND IT REQUIRES THE AXIS TO BE THE **TRAILING** DIM.  ``val_layout = (1,
        vec)`` puts a thread's ``vec`` elements contiguous along the copy axis, so a
        participant indexed on ``block_id`` at any other position is not addressable
        by this layout at that width and the whole plan must decline -- narrowing
        per-site instead would leave the lane loop's trip count assuming the wider
        width, which is bug class 1.  Returning ``([], [])`` makes the caller
        decline, exactly as an empty participant list does on the rolled path.
        """
        from ..language import memory_ops as _memory_ops

        try:
            hf = HostFunction.current()
        except Exception:
            return TVParticipants.empty()
        if hf._device_ir is None:
            return TVParticipants.empty()
        # ``(dtype_str, dtype_bits, strides)`` per participating access.  The
        # row-stride reduction and the width folds are ``TVParticipants``'; this
        # walk's only job is deciding WHICH accesses participate.
        accesses: list[tuple[str, int, tuple[int, ...], object]] = []
        for graph_info in hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if node.target not in (_memory_ops.load, _memory_ops.store):
                    continue
                if len(node.args) < 2:
                    continue
                tensor_node = node.args[0]
                subscript = node.args[1]
                if not isinstance(subscript, (list, tuple)):
                    continue
                val = (
                    tensor_node.meta.get("val")
                    if isinstance(tensor_node, torch.fx.Node)
                    else tensor_node
                )
                if not isinstance(val, torch.Tensor):
                    continue
                # Which subscript position (ignoring ``None`` inserts, which add no
                # tensor dim) names THIS axis?
                axis_pos = None
                pos = -1
                for idx in subscript:
                    if idx is None:
                        continue
                    pos += 1
                    if _tiled_axis_block_id(idx) == block_id:
                        axis_pos = pos
                if axis_pos is None:
                    continue
                # Trailing dim only -- see the docstring.  A rank mismatch means the
                # subscript is not a plain per-dim index list and this walk cannot
                # reason about which dim the axis lands on, so it declines too.
                if pos + 1 != val.ndim or axis_pos != val.ndim - 1:
                    return TVParticipants.empty()
                if val.stride(val.ndim - 1) != 1:
                    return TVParticipants.empty()
                accesses.append(
                    (
                        CompileEnvironment.current().backend.dtype_str(val.dtype),
                        val.element_size() * 8,
                        tuple(val.stride()),
                        val.dtype,
                    )
                )
        return TVParticipants.from_accesses(accesses)

    def _cute_ndtile_tv_requested(self, env: CompileEnvironment, block_id: int) -> bool:
        """Is the TV copy path requested for ``block_id``?  Config knob, env override.

        ⭐ THE CONFIG IS THE SOURCE OF TRUTH; the env var is a debugging override.
        ``cute_ndtile_tv`` (``CuteNDTileTvSpec``) is one slot per device loop, so
        different loops in one kernel can differ -- which is the point:  MEASURED, the
        sign of this trade tracks LANE EXTENT, so three of the eight
        ``cross_entropy_online`` cells prefer OFF at their current geometry and ON once
        re-tuned.  A single default cannot express that.

        ⛔ THE "extent 8-16 wins +81..110%" FIGURE THIS DOCSTRING USED TO CARRY IS
        REFUTED (corrected 2026-08-01, run 2 T0).  Those magnitudes are gate-OFF LOSSES
        from raising ``chunk``; with the arm ON extent 8-16 is worth -0.5%..-6.9% and
        **extent 4 beats extent 16** on 3 of 4 cells measured in both arms.  See
        ``cute/tv_layout.py::ndtile_tv_for`` for the table -- do not tune toward +81%.

        ⚠ ``HELION_CUTE_NDTILE_TV=1`` still forces it ON everywhere, and that is kept
        deliberately rather than for compatibility: it is how you run the
        one-variable A/B on a config that does not name the key, which is exactly how
        this knob's ladder was measured.  It is an override, so it can only turn the
        path ON -- a config that explicitly asks ``False`` and an env var that says
        ``1`` is a contradiction, and the env var wins loudly (it is the debugging
        tool) rather than silently.

        ⚠ NO SLOT MEANS OFF, and that is not the same rule ``_fill_missing`` follows.
        ``_fill_missing`` answers "the key is absent from a config that HAS a slot" and
        returns the ladder.  Here the question is "this loop has no slot at all" --
        which happens for any strategy that is not an NDTile loop, and for a
        ``ConfigSpec`` built before this knob existed.  Off is the inert answer, and
        it keeps this a pure reachability change.
        """
        if cute_ndtile_tv_enabled():
            return True
        spec_seq = getattr(env.config_spec, "cute_ndtile_tv", None)
        if spec_seq is None or not len(spec_seq):
            return False
        values = self.fn.config.config.get("cute_ndtile_tv")
        if not isinstance(values, (list, tuple)):
            return False
        return bool(spec_seq.config_get(cast("list[bool]", values), block_id, False))

    def _build_cute_tv_plan_for_block(
        self, block_id: int, vec_cap: int
    ) -> ChunkTVPlan | None:
        """The ONE place this strategy's TV access width is decided, per axis.

        ``chunk`` is the block size and ``threads_per_row`` the axis's thread count,
        so ``lane_extent`` comes out as ``EPT // vec`` -- which is *by construction*
        the outer lane loop's trip count that ``codegen_device_loop`` computes from
        the same ``vec_width``.  That identity is the point: the loop bound and the
        copy width are two readings of one number, so "index and access width
        disagree" is not expressible (``ChunkTVPlan``'s own docstring, class 1).

        ⚠ ``vec_cap`` IS THE ALREADY-NARROWED ``_cute_lane_vec_width_by_block``
        ENTRY, not the raw config value.  ``__init__`` has already lowered the
        requested V to one that divides EPT; passing the raw request would let
        ``chunk_plan`` return a wider ``vec`` than the lane loop was built for.

        ⭐ OPT-IN, VIA ``HELION_CUTE_NDTILE_TV=1``, AND THE DEFAULT-OFF IS A
        MEASURED DECISION RATHER THAN TIMIDITY.  The capability is complete and
        correct (see the tests in ``test_cute_ndtile_tv_copy.py``), but switching it
        on by DEFAULT is not a widening -- it is a REPLACEMENT.  MEASURED on this
        tree, with no gate:

          * 3 tests in ``test_cute_tile_loop_vec_hoist.py`` fail, because the TV copy
            displaces the ``cute.arch.load`` emission they pin -- they are not wrong,
            they pin the path that is still the default;
          * 8 of the 40 frozen perf cells change emission, ALL of them
            ``cross_entropy_online/*`` -- attributed exactly, by an A/B on ONE tree
            toggling only this gate (30 -> 38 CHANGED at gate level 1).

        Those 8 cells are benchmarked kernels whose configs were tuned against the
        ``tile_unroll`` emission.  Re-routing them onto a different instruction mix
        is a PERF question -- a 128-bit ``cute.copy`` per outer iter versus two
        overlapped ``LDG.64``s -- and it must be answered by measurement, not by a
        default.  So the two paths coexist, the legacy one stays the default, and
        flipping this gate is a one-line, measurable experiment.

        ⇒ ``cute_vector_widths`` is deliberately NOT overloaded to carry this.  It is
        documented as a WIDTH CAP, and both paths honour it identically; making one
        of its values also select an emission STRATEGY would conflate two questions
        in one knob, which is how a width came to be silently discarded before
        (LEDGER E012/E048).

        ⭐ AND THE ONLY LEGAL OUTCOMES ARE "A PLAN AT EXACTLY ``vec_cap``" OR
        ``None``.  ``chunk_plan`` may legitimately NARROW (an odd row stride, the
        pointer-alignment rule), and a narrowed plan is unusable here: the outer lane
        loop's trip count is already fixed at ``EPT // vec_cap`` by the caller, so a
        narrower copy would visit fewer elements than the loop assumes -- bug class 1
        exactly.  Declining is safe and costs nothing: the caller keeps the legacy
        per-element ``tile_unroll`` enumeration, which is already complete.  This is
        the same "DECLINE, never narrow" rule the legacy branch of
        ``_cute_vector_load_ctx`` states for its own clamp, for the same reason.
        """
        from ..language.memory_ops import _cute_is_byte_packed
        from .cute.tv_layout import build_tv_plan

        # ⚠ LOCAL, and it must stay local: reduction_strategy imports this module at
        # module scope, so a top-level import here is a cycle.
        from .reduction_strategy import _cute_tv_forwardable_raw_keys
        from .reduction_strategy import _cute_tv_has_store_then_load_alias

        env = CompileEnvironment.current()
        if env.backend.name != "cute":
            return None
        if not self._cute_ndtile_tv_requested(env, block_id):
            return None
        idx = self.block_ids.index(block_id)
        threads_per_row = self.num_threads[idx]
        chunk = self._configured_block_size_int(self.block_size[idx])
        if chunk is None or chunk <= 0 or threads_per_row <= 0:
            return None
        # ── RAGGED N: ROUND THE TILE UP AND PREDICATE THE TAIL, same as the rolled
        #    path.  This used to be a hard decline. ─────────────────────────────────
        #
        # ⛔ THE COMMENT THAT USED TO STAND HERE WAS MEASURED FALSE.  It read: "this
        # strategy has no such predicate on the copy … which is why
        # ``cute_tv_tail_predicate`` stays at its inert ``None`` for this strategy and
        # is NOT AN OMISSION."  It was an omission.  The hook is called on WHATEVER
        # strategy answered ``cute_tv_capable()`` (``memory_ops.py:1039``), so the
        # emission side was already path-agnostic; only this class's override was
        # missing.  ``ragged_tail`` is likewise path-agnostic int arithmetic.
        #
        # ⭐ AND THE DECLINE WAS LOAD-BEARING IN THE WRONG DIRECTION.  MEASURED over the
        # 40 frozen cells: it is the sole reason
        # ``cross_entropy_online/{32768x12000, 8192x100000}`` are the ONLY two cells
        # that (a) still emit a classic vec load and reach the branch's OOB-guard
        # machinery, and (b) still need ``peel_ragged_tile``.  Both of Task 6's
        # deletions were blocked on this one gate.
        #
        # ``cluster_n=1``: this strategy emits no cluster (``_cute_cluster_n_emitted``
        # stays 1), so the tile granularity is ``chunk`` alone.  Asked through the
        # SHARED predicate rather than by re-deriving ``numel % chunk``, so the
        # overshoot budget and the divisibility rule are one decision for both paths.
        numel = env.block_sizes[block_id].numel
        tail_predicated = not isinstance(numel, (int, sympy.Integer)) or bool(
            int(numel) % chunk
        )
        if tail_predicated and not ragged_tile_admissible(
            env, numel, chunk, cluster_n=1, vec=1
        ):
            return None
        # ⚠ fp8 IS EXCLUDED, and not for tidiness.  ⭐ AND THE GATE BELONGS TO THIS
        # CALLER, NOT TO THE PLAN BUILDER, which is why it is asked HERE and not
        # passed to :func:`build_tv_plan` as a policy flag.  The exclusion is a
        # property of this strategy's CONSUMER, not of the layout: the legacy
        # ``tile_unroll`` path loads a byte-packed dtype as ONE packed integer and
        # shift-extracts the elements, so its per-element pipeline is a different
        # shape from the ``fragment[vi]`` read the TV path emits.  Admitting it
        # would give a ``cute.copy`` whose fragment the consume pipeline does not
        # know how to read element-wise -- so this declines to the path that
        # already handles it.  A TV layout over fp8 is perfectly constructible.
        # ⛔⛔ THE STORE-THEN-LOAD RAW VETO, NOW ASKED ON THIS PATH TOO.
        #
        # ``_cute_tv_partition_hoist`` anchors a load copy ABOVE the per-element loop and a
        # store flush BELOW it.  That order is exactly right when the load precedes the
        # store in program order and exactly WRONG when it follows one: the load then reads
        # pre-store gmem.  MEASURED historically on the rolled path -- ``buf[m,:] = 2.0``
        # then ``sum(buf[m,:])`` summed the OLD values (N=2048 gave 4096.0 instead of
        # 2048.0) at every ``vec > 1``.
        #
        # ⭐ THE EMISSION MECHANISM IS SHARED, SO THE GATE MUST BE.  Only the ROLLED path
        # asked this, and that asymmetry was an accident of coverage rather than a
        # statement about the tile path: the hoist that creates the hazard is the same
        # function for both.  The gate is module-level and reads no ``self`` (it walks
        # ``HostFunction.current().device_ir``), so sharing it costs one call.
        #
        # ⚠ AND IT IS ASKED HERE, AT PLAN CONSTRUCTION, for the reason the rolled path's
        # own comment gives: a per-SITE decline would leave the lane loop's trip count
        # derived from ``plan.vec`` while an individual access narrowed -- bug class 1.
        # With no plan, ``_cute_lane_vec_width_by_block`` stays unset for this block and
        # every vec mode is unreachable, so the scalar enumeration (which observes the
        # store) is what emits.
        #
        # ⚠ THIS IS A NARROWING, and I could not produce a wrong answer without it: both
        # hazard shapes I built were bit-exact on the tile path, because B3's
        # ``_cute_tv_forwardable_raw_keys`` CLASSIFIED the hazard as forwardable rather
        # than the veto firing (measured: ``fwd=['buf'], veto=False``).  ⇒ this closes an
        # audited hole rather than fixing a measured failure, and it is stated that way
        # deliberately -- but the hole is real, this run WIDENED this path's coverage
        # (the ragged round-up above), and the failure mode is order- and vec-sensitive,
        # i.e. exactly the kind that hides.  Failing closed is the cheap side of that bet.
        if _cute_tv_has_store_then_load_alias():
            return None
        # ⛔⛔ AND ANY STORE-THEN-LOAD RAW AT ALL IS FATAL **ON THIS PATH**, not just an
        #    unforwardable one.
        #
        # The gate above declines when NO raw can forward.  That is the right test for the
        # ROLLED path, which then records the forwardable keys in
        # ``_cute_tv_forwarded_raw_keys`` and reads the value out of the store's fragment.
        # ⭐ ``CuteNDTileStrategy`` NEVER POPULATES THAT SET (``grep`` shows assignments only in
        # ``reduction_strategy.py``), so at emission ``_cute_tv_forwards_store_fragment`` reads
        # an empty set, returns False, and the load emits its own ``partition_S`` copy hoisted
        # ABOVE the constexpr V-loop -- i.e. ahead of the store's post-loop flush.  It reads
        # STALE GMEM.
        #
        # MEASURED (found by adversarial review, reproduced here): on
        # ``out[t] = x[t].to(f32); out[t] = out[t] * 3.0`` at ``bs=[1,512] vw=[1,4]``, A7c emits
        # ``tv=2`` and is WRONG by maxdiff 300 on {-1,+1} data, while the pre-A7c tree emits
        # ``tv=0`` and is EXACT.  The hole is pre-existing (fp32->fp32 is wrong on the baseline
        # too), but the baseline declined every mixed set, so admitting mixed dtypes is what
        # makes it FIRE -- turning the audited hole the gate's own comment describes ("I could
        # not produce a wrong answer without it") into a measured one.
        #
        # ⇒ decline the plan outright here.  Correct-and-scalar beats fast-and-wrong, and the
        # legacy per-element enumeration observes the store.  Forwarding on this path needs the
        # keys to be recorded first; that is a separate change with its own proof obligation.
        if _cute_tv_forwardable_raw_keys():
            return None
        participants = self._cute_tv_participants(block_id)
        if any(
            isinstance(d, torch.dtype) and _cute_is_byte_packed(d)
            for d in participants.torch_dtypes
        ):
            # ⭐ fp8 IS ADMITTED (A7c).  This used to be an unconditional decline, justified as
            # "the ``tile_unroll`` consume pipeline shift-extracts raw bytes and cannot read a
            # typed ``frag[vi]``".  The premise was right and the conclusion did not follow: the
            # fp8 leg now hands the SAME raw-``Uint8`` representation downstream (see the
            # normalisation at the TV load site in ``cute/memory_ops.py``), so the byte decode
            # -- which is what every cute ``fp8 -> f32`` lowers to -- gets exactly what it
            # expects.  MEASURED bit-exact: pure fp8 at vw=4/8, and fp8+bf16 mixed emitting
            # ``Float8E4M3FN/32`` + ``BFloat16/64`` atoms with zero ``cute.arch.load``.
            #
            # ⚠ ``HELION_CUTE_FP8_TV=0`` forces the old decline, which is the A/B that attributes
            # any future fp8 emission change to this path rather than to the plan or the atom
            # cache.
            if os.environ.get("HELION_CUTE_FP8_TV") == "0":
                return None
        # ── THE WIDTH DECISION ITSELF IS NOT HERE ──────────────────────────────
        #
        # ⭐ ONE builder AND ONE POLICY for every strategy -- see :func:`build_tv_plan`.
        #
        # This call used to pass ``require_exact_vec_cap=True``, because ``__init__`` had
        # already fixed the outer lane loop's trip count at ``EPT // vec_cap`` and this site
        # could not reshape it, so a narrowed copy would visit fewer elements than the loop
        # assumed (bug class 1).  ⇒ **A7b removed that asymmetry at the source**:
        # ``_build_cute_vec_lane_loop`` now asks for the plan BEFORE it builds the loop and
        # derives the trip count from ``plan.vec``, which is what the reduction path always
        # did.  So a layout-imposed narrowing becomes a narrower VECTORISED copy instead of a
        # decline, both callers pass the same arguments, and the flag has no caller left.
        plan = build_tv_plan(
            chunk=chunk,
            threads_per_row=threads_per_row,
            participants=participants,
            vec_cap=vec_cap,
            # ⭐ A7c: MIXED DTYPES ARE ADMITTED.  The emission site mints one copy atom per
            # distinct dtype (``_cute_tv_shared_for_dtype`` + ``ChunkTVPlan.for_dtype``), which
            # is what this flag's docstring said a caller must be able to do before passing it.
            # The GEOMETRY stays single-sourced: ``for_dtype`` changes only the element type and
            # ``emit_tiled_copy``'s layouts do not mention it, so every atom tiles the chunk
            # identically.  ``vec`` is still bounded by the WIDEST participant, so one width
            # serves the group -- which is the invariant, not the single-dtype restriction.
            #
            # ⚠ Unblocks fp8, and not as a special case: ANY realistic fp8 kernel is mixed-dtype
            # because its output is fp32/bf16 (measured: even ``out_f32[t] = x_fp8[t] * 2.0`` has
            # participants ``{fp8, f32}``), which is why the fp8 decline lower down was never
            # what actually stopped fp8.
            allow_mixed_dtypes=True,
            # ⚠ ``numel`` is REQUIRED when ``tail_predicated``, and the builder raises
            # rather than guessing if it is missing -- a caller that predicates a tail
            # without saying how long the row is cannot be served.
            numel=int(numel) if tail_predicated else None,
            tail_predicated=tail_predicated,
        )
        if plan is None:
            return None
        # ⛔ THE ROUNDED TILE IS ONLY SOUND IF THE COPY IS ACTUALLY PREDICATED, so the
        # predicate's OWN precondition is checked HERE, where the round-up is admitted,
        # rather than trusted at the emission site.  If the guard silently vanished, the
        # rounded tile's last chunk would write its tail into the NEXT ROW --
        # ``local_tile`` on a row-major tensor resolves column ``N + k`` of row ``m`` to
        # column ``k`` of row ``m + 1`` (the same hazard class as the reverted ND mask
        # elision, which shipped an unconditional store).
        #
        # ⚠ THE PRECONDITION IS THE CONSTEXPR-V VAR, **NOT** THE LANE BASE VAR, and the
        # difference is a lifecycle fact rather than a preference.  MEASURED:
        # ``_cute_vec_lane_var_by_block[block_id]`` is assigned at
        # ``codegen_device_loop`` line ~8024, BEFORE this method is called at ~8052;
        # ``_cute_lane_base_index_var_by_block`` is assigned at ~8177, AFTER it.  So
        # testing the base var here would decline EVERY ragged plan -- a widening that
        # reads as landed and is inert, which is the failure mode this tree keeps
        # hitting.  ``cute_tv_tail_predicate`` reads the base var by the time IT runs
        # (codegen of the copy, later still) and returns ``None`` if absent, so the two
        # checks agree without either guessing.
        #
        # ⇒ declines the ROUND-UP, not the plan: an exact tile needs no predicate and is
        # unaffected, so this cannot narrow existing coverage.
        if tail_predicated and block_id not in self._cute_vec_lane_var_by_block:
            return None
        if tail_predicated:
            self._cute_tv_tail_numel_by_block[block_id] = int(numel)
        return plan

    def _setup_cute_tv_chunk_prefix(
        self,
        block_id: int,
        chunk_body: list[ast.AST],
        offset_var: str,
        block_size_var: str,
    ) -> None:
        """Record where per-chunk TV declarations go, and the chunk's coordinate.

        ``chunk_body`` is the serial loop's body list, whose LAST element must be
        THIS axis's outer lane loop -- ``_cute_tv_partition_hoist`` inserts its
        ``local_tile`` / ``partition_*`` / fragment declarations at ``len - 1``, so a
        list whose tail is something else would place them after the loop that reads
        them (or, worse, inside a sibling axis's loop).

        ⭐ THE TAIL IS CHECKED BY **IDENTITY**, NOT ASSUMED, and the check is the
        reason this is a method rather than two assignments at the call site.  With
        two tiled axes the lane loops nest at the innermost level, so the OUTER
        axis's serial-loop body ends in the inner axis's ``for``, not in a lane loop
        -- and inserting a ``local_tile`` there would emit it outside the loop whose
        lane var slices it.  A block whose tail does not match simply gets no chunk
        prefix, which makes ``cute_tv_capable()`` False for it and the load site
        falls back to its legacy mode.  Declining beats emitting into the wrong
        scope, and an ORDER-based test could not tell the two cases apart.

        ``chunk_index_var`` is ``offset // chunk``: ``offset_var`` steps by the block
        size, so this is the tile coordinate ``local_tile``'s column argument wants.
        The same expression the rolled path emits, for the same reason.

        ⛔⛔ AND IT RESETS THE PARTITION CACHE, WHICH FIXES A WRONG-ANSWER BUG.  See the
        long note at the reset below; in short, this method runs ONCE PER SWEEP over the
        axis, and a partition minted in the previous sweep is out of scope in this one.
        """
        if block_id not in self._cute_tv_plan_by_block:
            return
        outer_lane_loop = self._cute_tv_outer_lane_loop_by_block.get(block_id)
        if outer_lane_loop is None or not chunk_body:
            return
        if chunk_body[-1] is not outer_lane_loop:
            return
        # ⛔⛔ THE PARTITION CACHE IS PER **CHUNK BODY**, NOT PER BLOCK, AND GETTING THAT
        # WRONG EMITTED A DANGLING SYMBOL.
        #
        # ``_cute_tv_partition_hoist`` caches ``(tensor, "S"|"D") -> fragment`` and
        # early-returns the cached fragment (``cute/memory_ops.py``, the ``if key in
        # cache`` at the top of its partition section).  The cached fragment is declared
        # by a ``local_tile`` / ``partition_*`` / ``make_rmem_tensor_like`` trio inserted
        # into THIS ``chunk_body`` -- i.e. inside THIS serial ``for`` loop -- so it is
        # only in scope for this sweep.
        #
        # A user kernel with two ``hl.tile`` loops over one registered block size (the
        # tile-regime analogue of a two-sweep norm) drives ``codegen_device_loop`` TWICE
        # on ONE strategy instance, so this method runs twice and the second sweep's load
        # of the same tensor HIT the first sweep's entry.  MEASURED, and it is not a
        # missed optimisation -- it is a compile failure at DSL runtime::
        #
        #     for tile_offset_2 in range(0, 2048, _BLOCK_SIZE_0):    # SWEEP 1
        #         _tv_part_0 = _tv_thr.partition_S(_tv_tile_0)
        #         _tv_frag_0 = cute.make_rmem_tensor_like(_tv_part_0[None, 0, 0])
        #     for tile_offset_2 in range(0, 2048, _BLOCK_SIZE_0):    # SWEEP 2
        #         ...                                               # no tile/part/frag
        #         load_1 = _tv_frag_0[vec_lane_1]                    # ⛔ OUT OF SCOPE
        #
        #     DSLRuntimeError: name '_tv_frag_0' is not defined
        #
        # ⭐ THE ROLLED PATH ALREADY DOES THIS, and its own comment states the reason in
        # the same words: ``reduction_strategy.py`` resets ``_cute_tv_partitions`` at the
        # top of every ``for roffset`` body because "the staging partitions are
        # ``local_tile``s of THIS chunk body, so a partition from the previous sweep must
        # not be reused (it would be out of scope)".  This strategy's cache is keyed by
        # block and was never reset, so it was the SAME defect with the per-block
        # indirection hiding it.
        #
        # ⚠ ``_cute_tv_staged_tensors`` MUST NOT BE RESET HERE, and the asymmetry is
        # load-bearing rather than an oversight.  That set records that a tensor was
        # PUBLISHED in an earlier sweep, which is exactly what makes a later sweep's read
        # legal -- clearing it would make every sweep believe it was the first, so each
        # would re-publish and none would read.  The rolled path documents the same
        # asymmetry at its own reset site.  (The set lives on the reduction strategies
        # today; this class does not carry one yet, so there is nothing to skip -- the
        # note is here for whoever adds it.)
        #
        # ⚠ INERT WHEN THE TV PATH IS OFF, by construction and not by luck: the guard
        # above returns before this line unless ``_cute_tv_plan_by_block`` has an entry,
        # and that dict is only populated when ``_build_cute_tv_plan_for_block`` returns
        # a plan, which requires ``cute_ndtile_tv_enabled()``.  MEASURED with the gate
        # unset: this line executes ZERO times across the frozen cells and all 40 emit
        # byte-identical source.
        self._cute_tv_partitions_by_block[block_id] = {}
        # ── capability ③: resolve this axis's staging geometry and request ─────────
        #
        # ⚠ THE STAGE PARTITIONS ARE RESET WITH THE GMEM ONES, for the identical reason:
        # they are ``local_tile``s of THIS chunk body.  ``_cute_tv_staged_tensors`` is
        # deliberately NOT reset -- it is what makes the later sweep's read legal.
        self._cute_tv_stage_partitions_by_block[block_id] = {}
        # ``_cute_tv_chunk``'s contract is "the extent ONE CTA covers in one pass".  On
        # this path there is no cluster, so that is exactly the block size -- read, not
        # re-derived, from the same accessor the plan used, so the staged buffer's size and
        # the copy's width cannot disagree.
        chunk = self._configured_block_size_int(
            self.block_size[self.block_ids.index(block_id)]
        )
        if chunk is not None:
            self._cute_tv_chunk = chunk
        # ⭐ RESOLVED HERE, NOT IN ``__init__``, and the split is the same one
        # ``_cute_reload_from_config`` documents: at ``__init__`` the thread-block dims are
        # not final (``thread_block_dims()`` still reports 1 on the row axis), so a
        # residency decided there declines every config.  By this point the plan exists and
        # the geometry is final.
        #
        # ⚠ AND IT IS READ THROUGH THE SHARED ``_cute_row_residency_config``, not from a
        # second parse of the knobs.  That method already handles the authoritative
        # ``cute_row_residency`` key AND the legacy ``cute_reduction_reload`` /
        # ``cute_tv_sweep_cache`` composition, so a config written either way means the same
        # thing here as it does on the rolled path.  ⚠ A TILED axis owns no slot in the two
        # per-reduction-block specs today (measured: 0 slots), so this resolves to the
        # ``gmem`` fallback -- i.e. today's behaviour -- until those slots exist.  That is
        # the deliberate scope line: the MECHANISM is complete and the KNOB is a separate,
        # riskier change (see the item-6 write-up in _notes2/T2_SCRATCH.md).
        # ⚠ THE ACTIVE BLOCK MUST BE POINTED AT ``block_id`` ACROSS THIS CALL.
        # ``_cute_reload_from_config`` ASSIGNS ``_cute_row_residency_requested`` (it resolves
        # the request and records it in one step, so the two cannot disagree), and that
        # setter -- like every property in the block above -- routes through
        # ``cute_stage_block_id()``.  During ``codegen_device_loop`` the consumer has not
        # stated an active block yet (that happens later, at the load site), so without this
        # the assignment would assert.  Saved and restored rather than left set: the active
        # block belongs to the CONSUMER, and leaving it pointed here would be this class
        # inferring it, which is exactly what ``cute_tv_lane_block_id``'s contract forbids.
        prev_active = self._cute_tv_active_block
        self._cute_tv_active_block = block_id
        try:
            self._cute_tv_reload_from_by_block[block_id] = (
                self._cute_reload_from_config(
                    self.fn, self._cute_tv_plan_by_block[block_id], block_id
                )
            )
        finally:
            self._cute_tv_active_block = prev_active
        chunk_index_var = self.fn.new_var(f"_tv_chunk_{block_id}", dce=False)
        # ⚠ UNDER ``SyntheticLocation``: a statement emitted without one inherits the
        # ambient user source location, which makes the unparser add a ``# src[...]``
        # comment AND shift its neighbours -- inflating the codegen diff so an inert
        # change reads as an emission change.  The rolled path's own chunk-index
        # statement is emitted from a context that already carries one.
        with SyntheticLocation():
            chunk_index_stmt = statement_from_string(
                f"{chunk_index_var} = {offset_var} // {block_size_var}"
            )
        chunk_body.insert(0, chunk_index_stmt)
        self._cute_tv_chunk_index_var_by_block[block_id] = chunk_index_var
        # ⭐⭐ AND THE CHUNK INDEX **EXPRESSION**, WHICH IS WHAT MAKES TWO SWEEPS OF ONE ROW
        # COMPARE EQUAL.  ``tv_tile_ids`` prefers this over the per-sweep VARIABLE NAME, and
        # ``memory_ops``' comment at that site already states why: each sweep mints its own
        # ``_tv_chunk_N`` while all of them hold the SAME value, so keying on the name reports
        # N distinct tiles where there is one and can never identify a re-read.
        #
        # ⛔ ONLY ``LoopedReductionStrategy`` SET THIS; THIS PATH DID NOT, AND THE OMISSION WAS
        # SILENT.  MEASURED on the tile path under ``cute_row_residency='registers'``: the two
        # sweeps produced ids differing only in ``_tv_chunk_0`` vs ``_tv_chunk_1``, so the
        # register-residency lookup saw **2 first-reads and 0 later-reads** -- it published the
        # row twice and served it never, and ``fuse_tv_copy_sweeps`` was still doing the real
        # work.  Nothing failed; the mechanism just did not fire.
        self._cute_tv_chunk_index_expr_by_block[block_id] = (
            f"{offset_var} // {block_size_var}"
        )
        self._cute_tv_chunk_prefix_by_block[block_id] = chunk_body

    def _configured_block_size_int(self, block_size: SymIntLike) -> int | None:
        if isinstance(block_size, int):
            return block_size
        env = CompileEnvironment.current()
        resolved_block_id = env.resolve_block_id(block_size)
        if resolved_block_id is not None:
            configured_size = self.fn.resolved_block_size(resolved_block_id)
            if isinstance(configured_size, int):
                return configured_size
        block_size_expr = _to_sympy(block_size)
        block_size_expr = env.specialize_expr(block_size_expr)
        if getattr(block_size_expr, "free_symbols", None):
            return None
        return int(block_size_expr)

    def _elements_per_thread_for_block(self, block_id: int) -> int:
        """Elements per thread for *block_id* (derived from num_threads)."""
        if block_id in self.inactive_block_ids:
            return 1
        idx = self.block_ids.index(block_id)
        nt = self.num_threads[idx]
        if nt == 0:
            return 1
        bs = self._configured_block_size_int(self.block_size[idx])
        assert isinstance(bs, int)  # validated by _thread_extent_for_axis
        return bs // nt

    def _lane_elements_per_thread(self, block_idx: int, block_size: int) -> int | None:
        """See ``NDTileStrategy._read_index_always_in_range``.

        This strategy may wrap the tile in a lane loop, so one thread covers
        ``block_size // num_threads`` consecutive elements rather than exactly one.  MMA
        mode distributes elements through the MMA atom rather than a lane loop, so its span
        is not the product this proof assumes -- decline there rather than guess.
        """
        if self.mma_mode:
            return None
        return self._elements_per_thread_for_block(block_idx)

    def _thread_extent_for_axis(
        self, block_id: int, block_size: SymIntLike
    ) -> SymIntLike:
        if block_id in self.inactive_block_ids:
            return 1
        if self.mma_mode:
            return 1  # MMA handles element distribution, no CUDA threads needed
        idx = self.block_ids.index(block_id)
        nt = self.num_threads[idx]
        if nt == 0:
            return block_size
        resolved_block_size = block_size
        if not isinstance(resolved_block_size, int):
            static_block_size = self._configured_block_size_int(resolved_block_size)
            if static_block_size is None:
                raise exc.BackendUnsupported(
                    "cute",
                    "num_threads requires static ND block sizes for cute",
                )
            resolved_block_size = static_block_size
        if resolved_block_size % nt != 0:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "block size must be divisible by num_threads for cute axis "
                    f"{block_id}: {resolved_block_size} is not divisible by {nt}"
                ),
            )
        return nt

    def _uses_thread_axis_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> bool:
        if block_id in self.inactive_block_ids:
            return False
        thread_extent = self._thread_extent_for_axis(block_id, block_size)
        return not (isinstance(thread_extent, int) and thread_extent == 1)

    def _thread_axis_map(self) -> dict[int, int]:
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        axis_order = [self.block_ids[i] for i in self.loop_order]
        axis = 0
        mapping: dict[int, int] = {}
        for block_id in axis_order:
            mapping[block_id] = axis
            if self._uses_thread_axis_for_block(block_id, block_size_by_id[block_id]):
                axis += 1
        return mapping

    def thread_axes_used(self) -> int:
        return sum(
            1
            for block_idx, block_size in zip(
                self.block_ids, self.block_size, strict=True
            )
            if self._uses_thread_axis_for_block(block_idx, block_size)
        )

    def _static_thread_extent_for_block(
        self, block_id: int, block_size: SymIntLike
    ) -> int | None:
        thread_extent = self._thread_extent_for_axis(block_id, block_size)
        if isinstance(thread_extent, int):
            return thread_extent
        return self._configured_block_size_int(thread_extent)

    def thread_block_sizes(self) -> list[int]:
        sizes: list[int] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            thread_extent = self._thread_extent_for_axis(
                block_id, block_size_by_id[block_id]
            )
            if self._uses_thread_axis_for_block(block_id, block_size_by_id[block_id]):
                static_extent = thread_extent
                if not isinstance(static_extent, int):
                    static_extent = self._configured_block_size_int(static_extent)
                if isinstance(static_extent, int):
                    sizes.append(static_extent)
        return sizes

    def thread_block_size_exprs(self) -> list[str]:
        exprs: list[str] = []
        block_size_by_id = dict(zip(self.block_ids, self.block_size, strict=True))
        for block_id in (self.block_ids[i] for i in self.loop_order):
            bs = block_size_by_id[block_id]
            if not self._uses_thread_axis_for_block(block_id, bs):
                continue
            thread_extent = self._thread_extent_for_axis(block_id, bs)
            if isinstance(thread_extent, int):
                exprs.append(str(thread_extent))
                continue
            if not isinstance(bs, torch.SymInt):
                return []
            bs_var = self.block_size_var(block_id)
            if bs_var is None:
                return []
            elements_per_thread = self._elements_per_thread_for_block(block_id)
            if elements_per_thread == 1:
                exprs.append(bs_var)
            else:
                exprs.append(f"({bs_var}) // {elements_per_thread}")
        return exprs

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        if not self._lane_var_by_block:
            return super().codegen_grid(state)

        block_ids = self.block_ids
        env = CompileEnvironment.current()
        block_sizes = self.block_size
        assert len(block_sizes) == len(block_ids)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var

        assert state.ast_args is None
        assert len(state.proxy_args) == 3
        ends: list[object]
        if state.proxy_args[1] is None:
            begins = [0] * len(block_ids)
            ends_arg = state.proxy_args[0]
        else:
            begins = state.proxy_args[0]
            ends_arg = state.proxy_args[1]
            if not isinstance(begins, (list, tuple)):
                begins = [begins]
            assert len(begins) == len(block_ids)
        if isinstance(ends_arg, (list, tuple)):
            ends = list(ends_arg)
        else:
            ends = [ends_arg]
        assert len(ends) == len(block_ids)
        steps = self._root_grid_steps(state)

        lane_setup_statements: list[ast.AST] = []
        outer_setup_statements: list[ast.AST] = []
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        self._cute_tv_thread_axis_offset = thread_axis_offset
        thread_axis_map = self._thread_axis_map()
        for i, (block_idx, block_size, begin, end, step) in enumerate(
            reversed(
                self._reorder(
                    [*zip(block_ids, block_sizes, begins, ends, steps, strict=True)]
                )
            )
        ):
            numel = self._range_numel_expr(begin, end, step)
            device_function = state.device_function
            dtype = env.index_type()
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            pid_var = device_function.new_var(f"pid_{i}", dce=True)

            begin_offset_expr = ""
            if begin != 0:
                begin_ast = self._to_ast(begin, to_dtype=dtype)
                begin_offset_expr = (
                    f"{state.codegen.lift(begin_ast, dce=True, prefix='begin').id} + "
                )

            if step not in (None, 1):
                step_ast = self._to_ast(step, to_dtype=dtype)
                # CuTe DSL preprocessor reserves ``step_<counter>`` (see comment
                # in ``TileStrategy.__init__``) — rename our lifted step var to
                # avoid the same UnboundLocalError that drove the offset rename.
                step_prefix = "tile_step" if env.backend.name == "cute" else "step"
                step_var = state.codegen.lift(step_ast, dce=True, prefix=step_prefix).id
                block_size_var = "1"
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}({pid_var}) * {step_var}"
                )
            elif block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
                state.add_statement(
                    f"{offset_var} = {begin_offset_expr}{pid_var} * {block_size_var}"
                )
            else:
                block_size_var = "1"
                state.add_statement(f"{offset_var} = {begin_offset_expr}{pid_var}")

            elements_per_thread = self._elements_per_thread_for_block(block_idx)
            uses_thread_axis = step in (None, 1) and self._uses_thread_axis_for_block(
                block_idx, block_size
            )
            axis = thread_axis_offset + thread_axis_map[block_idx]
            if uses_thread_axis:
                idx_expr = env.backend.lane_index_expr(
                    offset_var, elements_per_thread, axis=axis
                )
                thread_extent = self._thread_extent_for_axis(block_idx, block_size)
                static_extent = (
                    thread_extent
                    if isinstance(thread_extent, int)
                    else self._static_thread_extent_for_block(block_idx, block_size)
                )
                if isinstance(static_extent, int):
                    tracker.record(block_idx, axis, static_extent)
            else:
                idx_expr = offset_var
            if lane_var := self._lane_var_by_block.get(block_idx):
                # ⚠⚠ THE 2-D ROOT-GRID PATH IS DELIBERATELY SCALAR, both for TV and for
                # classic vec, matching ``origin/main``.  ``cute_vector_widths`` is inert here.
                #
                # A vectorised version existed briefly (A7a) and was DELETED, for two reasons
                # worth keeping apart:
                #
                # 1. ⛔ IT WAS BROKEN, and MODE-INDEPENDENTLY so.  ``DeviceGridState.wrap_body``
                #    splices ``lane_setup_statements`` WHOLESALE into the INNERMOST loop, so
                #    with more than one lane axis the OUTER axis's ``indices_``/``mask_`` land
                #    inside the inner loop while the hoisted wide load above it reads them:
                #    ``cannot access free variable 'indices_0'``.  MEASURED only 3 of 9
                #    ``block_sizes[0]`` x ``vec`` combinations compiling, at a 13.3% autotuner
                #    draw rate, needing no ``cute_ndtile_tv`` key.  ⚠ Invisible at
                #    ``block_sizes[0] == 32`` -- the value its own tests pinned -- because the
                #    row axis has EPT=1 there, gets no lane var, and its setup is hoisted to
                #    top level.
                # 2. It vectorised through CLASSIC VEC (hand-built integer addresses), the path
                #    this backend is trying to converge away from in favour of TV-or-scalar.
                #
                # ⇒ KNOWN, ACCEPTED HOLE: 2-D root-grid pointwise
                # (``for tm, tn in hl.tile(x.shape)``) runs at the scalar floor.
                # ⚠ The INNER-LOOP 2-D shape (``for tm: for tn:``) is a DIFFERENT entry point
                # on this same class (``codegen_device_loop``) and is UNAFFECTED: it vectorises
                # today and reaches a TV copy under ``cute_ndtile_tv``.
                #
                # HOW TO CLOSE IT, in order:
                # (a) SHARED PREREQUISITE, needed by either mode: partition
                #     ``lane_setup_statements`` PER AXIS and expose the scalar lane-loop bodies
                #     as insertion points, so each axis's setup lands in its own loop.
                #     Precedent: ``46ee1b15d`` did exactly this for ``_wrap_segmented_body``
                #     (the ``mask_0`` scope fix) -- same defect, different consumer.
                # (b) THEN, for TV rather than classic vec: register the grid blocks'
                #     ``cute_ndtile_tv`` slots (``device_ir.py`` skips them today, so no plan is
                #     ever built here) and call ``_build_cute_tv_plan_for_block``.  Doing (a)
                #     alone would only re-enable the classic-vec version.
                idx_expr = f"{idx_expr} + {env.backend.lane_offset_expr(lane_var)}"
                target = lane_setup_statements
            else:
                # Setup that does not depend on a lane variable can be hoisted
                # out of the lane loops. This avoids reassignments inside the
                # lane-loop body that confuse the CuTe DSL preprocessor when
                # its internal negative-step machinery emits identifiers like
                # ``offset_<n>`` that collide with helion's tile offsets.
                target = outer_setup_statements
            target.append(statement_from_string(f"{index_var} = {idx_expr}"))

            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                target.append(mask_statement)
            pid = PIDInfo(pid_var, block_size_var, numel, block_idx)
            pids.append(pid)
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)

        has_tensor_ends = any(isinstance(e, torch.Tensor) for e in ends)
        if has_tensor_ends:
            block_id_to_info = self._create_block_id_info_dict(
                state, ends_override=ends
            )
        else:
            block_id_to_info = self._create_block_id_info_dict(state)
        lane_loops = [
            (
                self._lane_var_by_block[block_id],
                self._elements_per_thread_for_block(block_id),
            )
            for block_id in (self.block_ids[i] for i in self.loop_order)
            if block_id in self._lane_var_by_block
        ]
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_loop_blocks=set(self._lane_var_by_block),
            lane_setup_statements=lane_setup_statements,
            outer_prefix=outer_setup_statements,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def _cute_lane_loops_meta(self) -> list[tuple[int, str, int, int]]:
        """Per-block ``(block_id, lane_var, full_extent, vec_width)``, in loop order.

        Factored out of :meth:`codegen_device_loop` so :meth:`codegen_grid` builds its lane
        nest from the SAME description (A7a).  Returning a list rather than looping in place
        is what lets both callers share ``_build_cute_lane_nest`` below.
        """
        meta: list[tuple[int, str, int, int]] = []
        for block_id in (self.block_ids[i] for i in self.loop_order):
            if block_id not in self._lane_var_by_block:
                continue
            meta.append(
                (
                    block_id,
                    self._lane_var_by_block[block_id],
                    self._elements_per_thread_for_block(block_id),
                    self._cute_lane_vec_width_by_block.get(block_id, 1),
                )
            )
        return meta

    def _build_cute_vec_lane_loop(
        self,
        block_id: int,
        lane_var: str,
        extent: int,
        vec_width: int,
        body: list[ast.AST],
    ) -> list[ast.AST]:
        """Split one axis's lane loop into ``outer (EPT // V) x inner constexpr V``.

        Verbatim the logic that used to be inline in :meth:`codegen_device_loop`; moved so
        :meth:`codegen_grid` can build the identical nest (A7a).  Records the four pieces the
        memory-op dispatcher and the TV site gate read back:
        ``_cute_vec_lane_var_by_block``, ``_cute_lane_body_by_block``, and (when a plan is
        available) ``_cute_tv_plan_by_block`` plus the two loop REFERENCES.
        """
        # ── A7b: ASK FOR THE PLAN **BEFORE** COMMITTING THE TRIP COUNT ──────────────────
        #
        # ⭐ THIS IS THE WHOLE OF A7b, and it is what collapses the two ``build_tv_plan``
        # callers into one policy.  The two used to differ only in WHEN they committed:
        #
        #   reduction path  build first, then read ``lane_extent`` back off the plan, so a
        #                   narrowing re-derives the trip count for free.
        #   this path       fixed the trip count at ``EPT // vec_cap`` first, so a narrowed
        #                   plan was unusable and had to be rejected outright
        #                   (``require_exact_vec_cap=True``).
        #
        # The reduction order is strictly better: nothing here needs the trip count before
        # the plan exists, and asking first means a layout-imposed narrowing (an odd row
        # stride, a pointer-alignment clamp, a ragged tail bound) turns into a NARROWER
        # VECTORISED copy instead of a decline to the per-element fallback.  ⇒ this path now
        # GAINS coverage it used to refuse, and ``require_exact_vec_cap`` has no remaining
        # caller.
        #
        # ⚠ The plan is still authoritative for the width, exactly as before -- the change is
        # only that its answer is obtained before anything is built from it, so the two can no
        # longer disagree BY CONSTRUCTION rather than by an assertion after the fact.
        #
        # ⛔⛔ THE CONSTEXPR-V VAR MUST BE MINTED **BEFORE** ASKING.  ``_build_cute_tv_plan_for_block``
        # gates its RAGGED (round-up + tail-predicate) arm on
        # ``block_id in self._cute_vec_lane_var_by_block`` -- a deliberate LIFECYCLE proxy for
        # "has this axis's constexpr-V nest been started yet", because the tail predicate is
        # emitted against that variable.  Asking before minting it therefore declined EVERY
        # ragged plan: MEASURED, the two non-power-of-2 ``cross_entropy_online`` frozen cells
        # (32768x12000, 8192x100000) fell from a 128-bit ``cute.copy`` to 5 ``cute.arch.load``s
        # while ``build_tv_plan`` itself was still happily returning ``vec=8`` -- the decline
        # was entirely in the caller's own precondition.
        #
        # ⇒ mint the var first.  It is a pure name allocation with no dependence on the width,
        # so hoisting it above the plan request is safe, and it keeps that gate's lifecycle
        # meaning intact rather than weakening the gate to accommodate the new order.
        vec_lane_var = self.fn.new_var(f"vec_lane_{block_id}", dce=False)
        self._cute_vec_lane_var_by_block[block_id] = vec_lane_var
        plan = self._build_cute_tv_plan_for_block(block_id, vec_width)
        if plan is not None and plan.vec != vec_width:
            # The layout narrowed.  Re-derive the emitted geometry from the plan's width so
            # the loop visits exactly the elements the copies cover, and record the narrowed
            # width so the load site's ``plan.vec == vec_width`` authority test agrees.
            if plan.vec <= 1 or extent % plan.vec:
                # Not expressible as an ``outer x V`` split (a V that does not divide EPT
                # would leave a partial final iteration): keep the legacy per-element
                # enumeration, which is already complete.
                plan = None
            else:
                vec_width = plan.vec
                self._cute_lane_vec_width_by_block[block_id] = vec_width
        # The inner constexpr-V loop's body re-runs the user body for each of the V lanes
        # (the user body's per-lane ``index_var = ...`` setup keys off the COMPOSITE lane =
        # outer*V + inner so the per-element index is correct).  ``vec_lane_var`` was minted
        # above, before the plan request -- see the lifecycle note there.
        inner_for = cast(
            "ast.For",
            ast.parse(
                f"for {vec_lane_var} in cutlass.range_constexpr({vec_width}):\n    pass"
            ).body[0],
        )
        inner_for.body = body  # type: ignore[assignment]
        # ``lane_body`` is what's INSIDE the outer lane loop: statements above the constexpr
        # V-loop, plus the loop itself (last entry).  memory_ops.py splices a hoisted wide
        # copy into ``lane_body[-1:]`` via the same protocol ``LoopedReductionStrategy`` uses.
        lane_body: list[ast.AST] = [inner_for]
        self._cute_lane_body_by_block[block_id] = lane_body
        outer_extent = extent // vec_width
        # Always emit the outer lane loop even when ``outer_extent == 1`` (i.e. EPT == V):
        # the lane-base index expression references ``lane_var``, which must be defined in
        # scope.  The CuTe DSL constant-folds the 1-iter loop away.
        outer_for = _create_lane_loop(lane_var, outer_extent, lane_body)
        if plan is not None:
            # ⭐ STILL ASSERTED.  ``plan.lane_extent`` and the emitted trip count are two
            # readings of one number; if they ever differ the copies cover a different element
            # set than the loop visits -- bug class 1, silently.  It is now a tautology on the
            # happy path (both derive from ``plan.vec``), which is the point: the invariant
            # moved from "a decline enforces it" to "the construction cannot violate it", and
            # the assert stays as the tripwire for a future edit that reintroduces a second
            # source for either number.
            assert plan.lane_extent == outer_extent, (
                f"TV plan lane_extent={plan.lane_extent} disagrees with the "
                f"emitted outer lane extent {outer_extent} for block "
                f"{block_id}: {plan.describe()}"
            )
            # The constexpr-V loop is recorded by REFERENCE (not by position): both TV legs
            # mutate ``lane_body``, so a stored index would move.
            self._cute_tv_plan_by_block[block_id] = plan
            self._cute_tv_constexpr_loop_by_block[block_id] = inner_for
            self._cute_tv_outer_lane_loop_by_block[block_id] = outer_for
        return [outer_for]

    def _build_cute_lane_nest(self, body: list[ast.AST]) -> list[ast.AST]:
        """Wrap ``body`` in this strategy's lane loops, vectorising where possible.

        ⭐ THE ONE LANE-NEST BUILDER, shared by ``codegen_device_loop`` and (since A7a)
        ``codegen_grid``.  Both used to have to agree about a five-part protocol -- the
        outer lane loop, the inner ``range_constexpr(V)`` loop, ``_cute_lane_body_by_block``,
        ``_cute_vec_lane_var_by_block``, and the TV plan + its two loop references -- and
        only the device-loop path implemented it, which is exactly why root-grid pointwise
        emitted a scalar load at every vector width (A7a's measured 2x bandwidth gap).
        Sharing the builder means a future sixth part cannot be added to one path only.

        When ``vec_width > 1`` and it divides the per-thread extent, the lane loop is split
        into ``outer (EPT // V) x inner constexpr V``: the memory-op dispatcher then splices
        ONE wide copy between the outer lane setup and the inner loop, so per-thread bytes
        per load grow from ``sizeof(dtype)`` to ``V * sizeof(dtype)`` (LDG.64 / LDG.128).
        Otherwise the axis gets a plain scalar lane loop, exactly as before.
        """
        for block_id, lane_var, extent, vec_width in reversed(
            self._cute_lane_loops_meta()
        ):
            if vec_width > 1 and extent > 0 and extent % vec_width == 0:
                body = self._build_cute_vec_lane_loop(
                    block_id, lane_var, extent, vec_width, body
                )
            else:
                body = [_create_lane_loop(lane_var, extent, body)]
        return body

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        if not self._lane_var_by_block and not self.mma_mode:
            return super().codegen_device_loop(state)

        block_ids = self.block_ids
        env = CompileEnvironment.current()
        dtype = env.index_type()
        block_sizes = self.block_size
        user_body: list[ast.AST] = []
        body: list[ast.AST] = user_body
        # Capture per-block (lane_var, full_extent, vec_width).  When
        # vec_width > 1, the outer lane runs (full_extent // vec_width)
        # iters; the inner constexpr-V loop handles V elements per outer
        # iter.  The memory_ops vec-load dispatcher splices a single
        # ``cute.arch.load(..., V)`` between the outer lane setup and the
        # inner constexpr loop, so per-thread bytes-per-load grow from
        # ``sizeof(dtype)`` to ``V * sizeof(dtype)`` (LDG.64 / LDG.128).
        body = self._build_cute_lane_nest(body)
        for_node: ast.For | None = None
        assert len(block_sizes) == len(block_ids)
        if len(state.ast_args) == 5:
            _, begins, ends, _, steps = state.ast_args
        else:
            _, begins, ends, _ = state.ast_args
            steps = None
        _, _, proxy_ends, *_ = state.proxy_args
        assert isinstance(begins, list)
        assert isinstance(ends, list)
        if steps is None:
            steps = [None] * len(block_ids)
        assert isinstance(steps, list)
        assert isinstance(proxy_ends, list)
        block_id_to_info = {}
        tracker = ThreadAxisTracker()
        thread_axis_offset = self._thread_axis_offset(state)
        self._cute_tv_thread_axis_offset = thread_axis_offset
        thread_axis_map = self._thread_axis_map()
        index_setup: list[ast.stmt] = []
        for block_idx, block_size, begin, end, step, proxy_end in self._reorder(
            [*zip(block_ids, block_sizes, begins, ends, steps, proxy_ends, strict=True)]
        ):
            offset_var = self.offset_var(block_idx)
            index_var = self.index_var(block_idx)
            if step in (None, 1) and block_size != 1:
                block_size_var = self.block_size_var(block_idx)
                assert block_size_var is not None
                self._setup_block_size_constexpr(state, block_size_var, block_size)
            else:
                block_size_var = "1"
            end_var_name = state.codegen.lift(
                self._to_ast(end, to_dtype=dtype), dce=True, prefix="end"
            ).id
            begin_var_name = state.codegen.lift(
                self._to_ast(begin, to_dtype=dtype), dce=True, prefix="begin"
            ).id
            block_id_to_info[block_idx] = LoopDimInfo(
                begin_var_name=begin_var_name,
                begin_expr=_to_sympy(begin)
                if isinstance(begin, (int, torch.SymInt))
                else None,
                end_var_name=end_var_name,
                end_expr=self._fold_tile_end_op(state, proxy_end, block_size),
            )

            # When the backend uses Python range() (e.g. Pallas), range
            # bounds must be plain Python ints — skip the dtype cast so
            # that concrete values stay as ints and are not wrapped in
            # backend-traced dtype conversions.
            range_dtype = None if env.backend.range_requires_python_int else dtype
            for_node = create(
                ast.For,
                target=create(ast.Name, id=offset_var, ctx=ast.Store()),
                iter=expr_from_string(
                    self.get_range_call_str(
                        state.config,
                        [block_idx],
                        begin="{begin}",
                        end="{end}",
                        step=(
                            ast.unparse(self._to_ast(step, to_dtype=range_dtype))
                            if step not in (None, 1)
                            else block_size_var
                        ),
                    ),
                    begin=self._to_ast(begin, to_dtype=range_dtype),
                    end=self._to_ast(end, to_dtype=range_dtype),
                ),
                body=body,
                orelse=[],
                type_comment=None,
            )
            elements_per_thread = self._elements_per_thread_for_block(block_idx)
            uses_thread_axis = step in (None, 1) and self._uses_thread_axis_for_block(
                block_idx, block_size
            )
            axis = thread_axis_offset + thread_axis_map[block_idx]
            if uses_thread_axis:
                idx_expr = env.backend.lane_index_expr(
                    offset_var, elements_per_thread, axis=axis
                )
                thread_extent = self._thread_extent_for_axis(block_idx, block_size)
                static_extent = (
                    thread_extent
                    if isinstance(thread_extent, int)
                    else self._static_thread_extent_for_block(block_idx, block_size)
                )
                if isinstance(static_extent, int):
                    tracker.record(block_idx, axis, static_extent)
            else:
                idx_expr = offset_var
            block_vec_width = self._cute_lane_vec_width_by_block.get(block_idx, 1)
            vec_lane_var = self._cute_vec_lane_var_by_block.get(block_idx)
            if lane_var := self._lane_var_by_block.get(block_idx):
                if block_vec_width > 1 and vec_lane_var is not None:
                    # Composite per-element lane index = outer*V + inner.
                    # outer (``lane_var``) ranges [0, EPT/V); inner
                    # (``vec_lane_var``) ranges [0, V).  Per-thread base
                    # (the start of the V-wide chunk this thread owns
                    # for this outer iter) is stashed in
                    # ``_cute_lane_base_index_var_by_block`` so the vec
                    # load can use it directly (mirrors the
                    # ``LoopedReductionStrategy`` unroll path).
                    base_index_var = self.fn.new_var(
                        f"lane_base_{block_idx}", dce=False
                    )
                    self._cute_lane_base_index_var_by_block[block_idx] = base_index_var
                    # ``base = offset + tid*EPT + outer*V``  (per-thread
                    # V-aligned base) — emitted INSIDE the outer lane
                    # loop's body (above the constexpr V-loop) so a
                    # single ``cute.arch.load(..., V)`` can be hoisted
                    # at the same level by memory_ops.
                    lane_body_list = self._cute_lane_body_by_block.get(block_idx)
                    if lane_body_list is not None:
                        # ⛔⛔ THE TWO PATHS NEED DIFFERENT FORMULAS, AND CONFLATING
                        # THEM RETURNED ``nan``.  A TV site takes its ADDRESSES from
                        # ``partition_S`` (interleaved: thread stride ``vec``, lane
                        # stride ``tpr*vec``), while the legacy per-element path
                        # derives them from THIS expression and is blocked (thread
                        # stride ``EPT``, lane stride ``vec``).  Emitting the blocked
                        # form for a TV site addresses a layout the copy does not
                        # use: harmlessly permuted on a divisible extent, off the end
                        # of the row on a ragged one.
                        #
                        # ⭐ The TV arm does not spell the formula here -- the PLAN
                        # owns it (``ChunkTVPlan.emit_lane_base``), so this site
                        # cannot transpose the strides again, which is exactly how
                        # the bug happened.
                        if (
                            plan := self._cute_tv_plan_by_block.get(block_idx)
                        ) is not None:
                            base_expr = plan.emit_lane_base(
                                offset_var,
                                lane_var,
                                f"cutlass.Int32(cute.arch.thread_idx()[{axis}])",
                            )
                        else:
                            base_expr = (
                                f"{idx_expr} + {env.backend.lane_offset_expr(lane_var)} "
                                f"* {block_vec_width}"
                            )
                        lane_body_list.insert(
                            0,
                            statement_from_string(f"{base_index_var} = {base_expr}"),
                        )
                    # The user-body's per-element index uses the base +
                    # the inner constexpr-V var so the existing scalar
                    # pipeline (mask + cast + reduce-or-store) keeps
                    # working unchanged.
                    idx_expr = f"{base_index_var} + cutlass.Int32({vec_lane_var})"
                    self._setup_cute_tv_chunk_prefix(
                        block_idx, body, offset_var, block_size_var
                    )
                else:
                    idx_expr = f"{idx_expr} + {env.backend.lane_offset_expr(lane_var)}"
            index_setup.append(statement_from_string(f"{index_var} = {idx_expr}"))
            mask_statement = self._setup_mask(
                state, block_idx, block_size, index_var, end
            )
            if mask_statement is not None:
                index_setup.append(mask_statement)
                # ⭐ RAGGED-TAIL PEEL.  Record that this axis's *bounds* mask is vacuous on
                # every iteration but the last, so ``cute/peel_ragged_tile.py`` can split
                # the loop after all the other AST passes have run.  Registering the
                # ``offset_var`` here rather than pattern-matching a ``for`` later means
                # the pass never has to guess which loop belongs to which axis.  The plan
                # is *resolved* (not merely recorded) now, because ``block_size`` and the
                # thread geometry are both final at this point; ``ragged_peel_plan``
                # declines for every shape it cannot prove.
                self.fn.register_ragged_peel(offset_var, self, block_idx)
            # ⭐ AND RECORD THAT THIS IS A DEVICE LOOP, for ``cute/defer_online_merge.py``
            # (task 3).  Registered here, beside the ragged peel, for the same reason and
            # by the same argument: the pass must not have to GUESS which ``for`` is the
            # recurrence's outer loop.  The inner lane loop matches every structural
            # predicate the outer one does, and deferring at it is relerr 5.18 -- so the
            # loop that owns an offset var says so, and the lane loop cannot be chosen.
            #
            # ⚠ Unconditional and outside the ragged-peel ``if``: this records only "a
            # strategy emitted a device loop with this offset var", which is true whether or
            # not that loop is peelable or carries a recurrence.  The recurrence's own
            # algebra is still checked by the pass from the emitted body, because the
            # cross-lane reduces it matches on are created by codegen.
            self.fn.register_online_defer(offset_var)
            body = [for_node]
        assert for_node is not None
        # Run index/mask setup once per loop-offset and per-lane before user body.
        user_body[:0] = index_setup
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
            lane_loop_blocks=set(self._lane_var_by_block),
        )

    def supports_index_rank_expansion(self) -> bool:
        return False


class CuteFlattenedTileStrategy(FlattenedTileStrategy):
    """Flattened CuTe strategy: scalar index per thread over a flattened tile."""

    def __init__(
        self,
        fn: DeviceFunction,
        block_ids: list[int],
        block_size: list[SymIntLike] | SymIntLike,
        loop_order: list[int],
        num_threads: int = 0,
    ) -> None:
        super().__init__(fn, block_ids, block_size, loop_order)
        self._num_threads = num_threads
        self._lane_var: str | None = None
        if num_threads > 0 and isinstance(block_size, int) and num_threads < block_size:
            self._lane_var = self.new_var("lane", dce=False)
        # ⚠⚠ THE 1-D FLATTENED PATH IS DELIBERATELY SCALAR, AND ``cute_vector_widths`` IS
        # INERT HERE.  This matches ``origin/main``, where every width emitted a scalar
        # ``.load()`` on this path.
        #
        # A vectorised version existed briefly (A7a): populating the four dicts
        # ``memory_ops._cute_vector_load_ctx`` reads -- ``_cute_lane_vec_width_by_block``,
        # ``_cute_vec_lane_var_by_block``, ``_cute_lane_base_index_var_by_block`` and
        # ``_cute_lane_body_by_block`` -- reached ``cute.arch.load`` and MEASURED
        # ``scalar 2 -> 0`` / ``arch.load 0 -> 4`` at ``vw=8``.  It was DELETED on purpose:
        # it vectorised through the CLASSIC-VEC path (hand-built integer addresses), not
        # through a ``ChunkTVPlan``, so it added a new classic-vec consumer to a backend
        # converging on TV-or-scalar at every load site.
        #
        # ⇒ KNOWN, ACCEPTED HOLE: 1-D pointwise (``vector_add``-shaped) runs at the scalar
        # floor -- measured 2337 GB/s, ~50% of the torch ceiling, and no config can reach a
        # wide load.  Roughly 2x bandwidth is on the table.
        #
        # HOW TO CLOSE IT PROPERLY (do this, not the classic-vec version): give this class a
        # TV plan.  It has none today -- no ``_cute_tv_plan``, ``_cute_tv_chunk_prefix``,
        # ``_cute_lane_body`` or ``_cute_tv_chunk_index_var``, and it never calls
        # ``_build_cute_tv_plan_for_block``, so ``_cute_tv_site_eligible`` declines at its
        # first gate.  The open design question is what ``threads_per_row`` means for a
        # single flat axis, since ``ChunkTVPlan``'s geometry is (row, column) with ``tpr``
        # threads covering ``tpr*vec`` consecutive columns.

    @property
    def _elements_per_thread(self) -> int:
        """Elements per thread (derived from num_threads and block_size)."""
        if self._num_threads == 0:
            return 1
        assert isinstance(self.block_size, int)
        return self.block_size // self._num_threads

    def _thread_extent(self) -> SymIntLike:
        if self._num_threads == 0:
            return self.block_size
        if not isinstance(self.block_size, int):
            raise exc.BackendUnsupported(
                "cute",
                "num_threads requires static flattened block sizes for cute",
            )
        if self.block_size % self._num_threads != 0:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "block size must be divisible by num_threads for cute: "
                    f"{self.block_size} is not divisible by {self._num_threads}"
                ),
            )
        return self._num_threads

    def thread_block_sizes(self) -> list[int]:
        if not self._uses_thread_axis():
            return []
        thread_extent = self._thread_extent()
        if not isinstance(thread_extent, int):
            return []
        return [thread_extent]

    def thread_block_size_exprs(self) -> list[str]:
        if not self._uses_thread_axis():
            return []
        thread_extent = self._thread_extent()
        if isinstance(thread_extent, int):
            return [str(thread_extent)]
        if not isinstance(self.block_size, torch.SymInt):
            return []
        bs_var = self.block_size_var(-1)
        if bs_var is None:
            return []
        if self._num_threads == 0:
            return [bs_var]
        return [f"({bs_var}) // {self._elements_per_thread}"]

    def _uses_thread_axis(self) -> bool:
        thread_extent = self._thread_extent()
        return not (isinstance(thread_extent, int) and thread_extent == 1)

    def codegen_grid(self, state: CodegenState) -> DeviceGridState:
        if self._lane_var is None:
            return super().codegen_grid(state)

        offsets_var = self._offsets_var
        offsets_base_var = self.new_var("offsets_base", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        env = CompileEnvironment.current()
        total_numel = sympy.S.One
        lane_setup_statements: list[ast.AST] = []

        # ⚠ SCALAR, deliberately -- see the note in ``__init__``.  ``cute_vector_widths``
        # is inert on this path, as on ``origin/main``.
        lane_setup_statements.append(
            statement_from_string(
                f"{offsets_var} = {offsets_base_var} + {env.backend.lane_offset_expr(self._lane_var)}"
            )
        )
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            lane_setup_statements.append(
                statement_from_string(f"{block_index_var} = {expr}")
            )
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({state.sympy_expr(total_numel)})"
                )
            )

        pid_var = state.device_function.new_var("pid_flat", dce=True)
        pids = self.select_pid_strategy()
        if isinstance(state.device_function.pid, ForEachProgramID):
            pids.shared_pid_var = state.device_function.pid.shared_pid_var
        pids.append(PIDInfo(pid_var, block_size_var, total_numel, self.block_ids[0]))
        axis = self._flat_thread_axis()
        state.add_statement(
            f"{offsets_base_var} = {env.backend.lane_index_expr(f'({pid_var}) * ({block_size_var})', self._elements_per_thread, axis=axis)}"
        )
        pids.codegen(state)
        if isinstance(state.device_function.pid, ForEachProgramID):
            shared_pid = state.device_function.pid
            shared_pid.cases.append(pids)
            shared_pid.codegen(state)
        else:
            state.device_function.set_pid(pids)
        block_id_to_info = self._create_block_id_info_dict(state)
        lane_loops = []
        if self._lane_var is not None:
            lane_loops = [(self._lane_var, self._elements_per_thread)]
        tracker = ThreadAxisTracker()
        thread_extent = self._thread_extent()
        if self._uses_thread_axis() and isinstance(thread_extent, int):
            tracker.record_all(self.block_ids, axis, thread_extent)
        return DeviceGridState(
            self,
            block_id_to_info=block_id_to_info,
            lane_loops=lane_loops,
            lane_loop_blocks=set(self.block_ids) if lane_loops else set(),
            lane_setup_statements=lane_setup_statements,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        if self._lane_var is None:
            return super().codegen_device_loop(state)

        env = CompileEnvironment.current()
        offsets_var = self._offsets_var
        offsets_base_var = self.new_var("offsets_base", dce=True)
        block_size_var = self.block_size_var(-1)
        self._setup_block_size_constexpr(state, block_size_var, self.block_size)
        block_ids = self.block_ids
        total_numel = sympy.S.One
        lane_setup_statements: list[ast.AST] = []

        lane_setup_statements.append(
            statement_from_string(
                f"{offsets_var} = {offsets_base_var} + {env.backend.lane_offset_expr(self._lane_var)}"
            )
        )
        for i, block_idx in enumerate(self._reorder(block_ids)):
            numel = env.block_sizes[block_idx].numel
            block_index_var = self.index_var(block_idx)
            expr = offsets_var
            if total_numel != sympy.S.One:
                expr = f"({expr}) // ({state.sympy_expr(total_numel)})"
            if i + 1 < len(block_ids):
                expr = f"({expr}) % ({state.sympy_expr(numel)})"
            lane_setup_statements.append(
                statement_from_string(f"{block_index_var} = {expr}")
            )
            total_numel = total_numel * numel

        mask_var = self.mask_var(-1)
        if mask_var is not None:
            lane_setup_statements.append(
                statement_from_string(
                    f"{mask_var} = {offsets_var} < ({state.sympy_expr(total_numel)})"
                )
            )

        lid = self.new_var("lid")
        end_var = env.backend.cdiv_expr(
            state.sympy_expr(total_numel), block_size_var, is_device=True
        )
        axis = self._flat_thread_axis()
        user_body: list[ast.AST] = []
        body: list[ast.AST] = user_body
        user_body[:0] = lane_setup_statements
        if self._lane_var is not None:
            lane_for = _create_lane_loop(
                self._lane_var,
                self._elements_per_thread,
                body,
            )
            body = [lane_for]
        body[:0] = [
            statement_from_string(
                f"{offsets_base_var} = {env.backend.lane_index_expr(f'{lid} * ({block_size_var})', self._elements_per_thread, axis=axis)}"
            )
        ]
        for_node = create(
            ast.For,
            target=create(ast.Name, id=lid, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(state.config, self.block_ids, end=end_var)
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        block_id_to_info = self._create_block_id_info_dict(state, use_proxy_ends=True)
        tracker = ThreadAxisTracker()
        thread_extent = self._thread_extent()
        if self._uses_thread_axis() and isinstance(thread_extent, int):
            tracker.record_all(self.block_ids, axis, thread_extent)
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=user_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def offset_var(self, block_idx: int) -> str:
        return self._offsets_var

    def supports_index_rank_expansion(self) -> bool:
        return False


class CompactedShape(NamedTuple):
    size_str: str
    user_indices: list[int]
    block_ids: list[int]

    def combine(self, other: CompactedShape) -> CompactedShape:
        size_str = self.size_str
        if size_str == "1":
            size_str = other.size_str
        else:
            assert other.size_str in ("1", size_str)
        return CompactedShape(
            size_str=size_str,
            user_indices=[*self.user_indices, *other.user_indices],
            block_ids=[*self.block_ids, *other.block_ids],
        )
