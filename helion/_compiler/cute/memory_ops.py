"""CuTe-backend codegen for ops defined in ``helion.language.memory_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``memory_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import math
import operator
import os
import re
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch.fx.node import map_arg

from ... import exc
from ...language import _decorators
from ...language.memory_ops import _CUTE_VECTOR_DTYPES
from ...language.memory_ops import _CUTE_VECTOR_UNROLL_DTYPES
from ...language.memory_ops import _codegen_cute_store_permute_lane_loops
from ...language.memory_ops import _codegen_cute_store_tcgen05_tile
from ...language.memory_ops import _cute_active_index_var
from ...language.memory_ops import _cute_active_mask_var
from ...language.memory_ops import _cute_axis_mask_var
from ...language.memory_ops import _cute_combined_mask
from ...language.memory_ops import _cute_index_exprs
from ...language.memory_ops import _cute_index_tuple
from ...language.memory_ops import _cute_is_byte_packed
from ...language.memory_ops import _cute_is_unroll_dtype
from ...language.memory_ops import _cute_register_tile_unroll_vec_hoist
from ...language.memory_ops import _cute_register_tile_unroll_vec_hoist_split2
from ...language.memory_ops import _cute_row_safe_index_expr
from ...language.memory_ops import _cute_scalar_load_expr
from ...language.memory_ops import _cute_scalar_pointer_expr
from ...language.memory_ops import _cute_tensor_dim_size_expr
from ...language.memory_ops import _cute_unique_graph_block_id
from ...language.memory_ops import _cute_unroll_vec_elem_type
from ...language.memory_ops import _matching_block_ids
from ...language.memory_ops import _maybe_codegen_cute_packed_affine_lhs_load
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .cute_epilogue import analyze_tcgen05_unary_epilogue_chain
from .cute_fx_walk import reach_tcgen05_matmul_anchors
from .tv_layout import ROW_RESIDENCY_REGISTERS
from .tv_layout import ROW_RESIDENCY_SMEM
from .tv_layout import legal_vec

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState
    from .tv_layout import ChunkTVPlan

log = logging.getLogger(__name__)


def _log_cute_layout(state: CodegenState, op_name: str) -> None:
    """Log the CuTe layout annotation for the current node, if any.

    This is used during CuTe load/store codegen to make layout info
    visible for debugging and future codegen integration.
    """
    layout = state.cute_layout
    if layout is None:
        return
    node_name = state.fx_node.name if state.fx_node else "?"
    log.debug(
        "cute %s %s: layout tag=%s thread=%s value=%s",
        op_name,
        node_name,
        layout.tag.value,
        layout.thread_shape,
        layout.value_shape,
    )


def _maybe_codegen_cute_packed_rhs_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
) -> ast.AST | None:
    from .indexing import match_cute_duplicate_stack_reshape_rhs

    fx_node = state.fx_node
    if fx_node is None or len(subscript) not in (2, 3) or len(fx_node.users) != 1:
        return None

    user = next(iter(fx_node.users))
    if user.op != "call_function" or user.target is not torch.ops.aten.stack.default:
        return None
    stack_users = list(user.users)
    if len(stack_users) != 1 or not isinstance(stack_users[0], torch.fx.Node):
        return None
    rhs_node = stack_users[0]
    packed_rhs = match_cute_duplicate_stack_reshape_rhs(rhs_node)
    if packed_rhs != (
        fx_node,
        len(user.args[0]) if isinstance(user.args[0], (list, tuple)) else 0,
    ):
        return None

    packed_block_id = _cute_unique_graph_block_id(state)
    if packed_block_id is None:
        return None
    packed_index = _cute_active_index_var(state, packed_block_id)
    if packed_index is None:
        return None

    leading_subscript = [*subscript[:-2]]
    col_index_exprs = _cute_index_exprs(
        state,
        [subscript[-1]],
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(col_index_exprs) != 1:
        return None
    (col_index,) = col_index_exprs
    leading_index_exprs = _cute_index_exprs(
        state,
        leading_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(leading_index_exprs) != len(leading_subscript):
        return None
    tensor_name = state.device_function.tensor_arg(tensor).name
    load_index_expr = ", ".join([*leading_index_exprs, packed_index, col_index])
    load_expr: ast.AST = expr_from_string(f"{tensor_name}[{load_index_expr}]")
    mask_terms: list[str] = []
    col_mask = _cute_combined_mask(
        state,
        [*leading_subscript, subscript[-1]],
        extra_mask,
        tensor=tensor,
    )
    if col_mask is not None:
        mask_terms.append(col_mask)
    if packed_mask := _cute_active_mask_var(state, packed_block_id):
        mask_terms.append(f"({packed_mask})")
    if not mask_terms:
        return load_expr
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    return expr_from_string(
        f"({{value}} if {' and '.join(mask_terms)} else {zero}(0))",
        value=load_expr,
    )


def _cute_scalar_storage_dtype(dtype: torch.dtype) -> str:
    if dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return "cutlass.Uint8"
    return CompileEnvironment.current().backend.dtype_str(dtype)


def _cute_scalar_store_expr(
    tensor_name: str, index_exprs: list[str], value: str
) -> str:
    if "None" in index_exprs:
        return f"{tensor_name}.__setitem__({_cute_index_tuple(index_exprs)}, {value})"
    return f"{_cute_scalar_pointer_expr(tensor_name, index_exprs)}.store({value})"


def _cute_unroll_vec_load_dtype_arg(dtype: torch.dtype, vec_width: int) -> str:
    """The dtype argument to ``cute.arch.load`` for an unroll-mode hoist.

    fp8 loads ``vec_width`` contiguous bytes as ONE packed scalar integer
    (no ``VectorType`` — avoids the V=8 ``nvvm.load.ext`` ICE and emits a
    single LDG).  bf16/fp16 load a ``Uint16`` vector of width ``vec_width``.
    """
    if _cute_is_byte_packed(dtype):
        return _cute_unroll_vec_elem_type(dtype, vec_width) + ".mlir_type"
    return f"ir.VectorType.get([{vec_width}], cutlass.Uint16.mlir_type)"


def _cute_tv_site_eligible(
    state: CodegenState,
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
) -> bool:
    """Can THIS access site be addressed by the reduction's TV layout?

    Deliberately narrow, and note what the narrowness costs: nothing.  An
    ineligible site does not degrade to a narrower copy at a wider stride (that
    would be class 1); it means the plan should never have been built, which
    :meth:`LoopedReductionStrategy._cute_tv_reduction_eligible` decides once,
    at ``__init__``, for the whole reduction.  This function is the per-site
    half of the same predicate and exists so the two cannot drift: if it ever
    returns False where the plan exists, that is a bug, and the caller falls
    back to a legacy mode which -- because ``plan.vec == vec_width`` gates
    entry -- can only be narrower, so the assert below is load-bearing.

    Requirements:
      * a 2-D row-major tensor indexed ``[<row>, <copy axis>]``, or a rank-1
        tensor indexed ``[<copy axis>]`` (the M-broadcast weight);
      * the copy axis is the trailing, stride-1 dim;
      * the chunk body's TV scaffolding exists (set up in
        ``codegen_device_loop``).

    ⭐ "THE COPY AXIS" IS ASKED OF THE STRATEGY, NOT PATTERN-MATCHED.  This used to
    require the trailing subscript to be a literal ``slice(None)``, which is the
    form a ROLLED reduction's axis takes -- and it silently excluded a **tiled**
    axis (``x[tile_m, tile_n]``, whose trailing index is a ``SymInt``) even when
    that axis carried a complete plan.  The predicate is now "the trailing index
    names the axis this strategy's plan addresses", answered by
    ``cute_tv_lane_block_id()``, with the bare-slice form still accepted for the
    rolled path where no block id is on offer.

    ⚠ THE TWO GATES MUST AGREE ABOUT THE AXIS, and this is one half of that.  The
    other is whichever selector built the plan (``_cute_layout_participants`` for a
    reduction, ``CuteNDTileStrategy._cute_tv_participants`` for a tile axis).  Both
    now key on the SAME question -- which axis does the copy address -- so a plan
    cannot be built at a width this function then refuses, which would leave the
    lane loop's trip count assuming a width no access uses (bug class 1).
    """
    if strategy is None:
        return False
    if strategy._cute_tv_chunk_prefix is None:  # pyrefly: ignore [missing-attribute]
        return False
    if strategy._cute_lane_body is None:  # pyrefly: ignore [missing-attribute]
        return False
    if strategy._cute_tv_chunk_index_var is None:  # pyrefly: ignore [missing-attribute]
        return False
    if tensor.ndim not in (1, 2):
        return False
    # The copy axis must be the trailing dim and contiguous, because the TV
    # layout's ``val_layout=(1, vec)`` puts a thread's elements contiguous
    # along it.
    trailing_stride = tensor.stride(tensor.ndim - 1)
    if not (isinstance(trailing_stride, int) and trailing_stride == 1):
        return False
    non_none = [idx for idx in subscript if idx is not None]
    if len(non_none) != tensor.ndim:
        return False
    if not _cute_tv_indexes_copy_axis(strategy, non_none[-1]):
        return False
    # ``x[:, :]`` (both axes tiled) would need the row as its own layout mode;
    # that is not this commit's shape.
    return not (tensor.ndim == 2 and isinstance(non_none[0], slice))


def _cute_tv_tile_site_takes_over(
    state: CodegenState,
    strategy: object,  # a TileStrategy with cute_tv_capable() at runtime
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    lane_block_id: int,
    vec_width: int,
) -> bool:
    """Should THIS tile-strategy site emit a TV ``cute.copy`` instead of a
    ``cute.arch.load``?

    ⭐ THIS FUNCTION IS WHERE THE ACTIVE BLOCK IS STATED.  The load site has already
    resolved which axis it addresses -- ``lane_block_id``, which came from the
    INDEXER's own binding and not from a size scan (see the long note in
    ``_cute_vector_load_ctx`` on why: MEASURED 23/23 correct vs 3/23 mis-bound).
    That block is handed to the strategy, whose single-valued protocol properties
    then resolve to it.  The direction is deliberately consumer -> strategy: the
    strategy re-deriving it would be a second opinion able to disagree with the
    address actually being emitted, and the axis is what selects the width.

    Declines -- returning False, so the caller keeps its legacy per-element
    enumeration -- whenever the plan does not exist, does not own this width, or
    cannot address this site.  Every one of those is a missed optimisation and none
    is a correctness question, because the legacy path is already complete.
    """
    if not strategy.cute_tv_set_active_block(lane_block_id):  # pyrefly: ignore
        return False
    if not strategy.cute_tv_capable():  # pyrefly: ignore
        return False
    plan = strategy._cute_tv_plan  # pyrefly: ignore
    # ``plan.vec == vec_width`` is the same authority test the rolled path applies:
    # the lane loop's trip count was derived from ``plan.vec``, so a site that
    # cannot do a width-``plan.vec`` copy must NOT fall back to a narrower one.
    if plan is None or plan.vec != vec_width:
        return False
    return _cute_tv_site_eligible(state, strategy, tensor, subscript)


def _cute_tv_indexes_copy_axis(strategy: object, idx: object) -> bool:
    """Does subscript entry ``idx`` name the axis ``strategy``'s TV plan addresses?

    Two accepted forms, and they are not interchangeable:

    * a bare ``slice(None)`` -- a ROLLED reduction's axis.  The strategy owns
      exactly one axis and the bare slice is it, so no identity check is available
      or needed.
    * a ``SymInt`` whose block id is ``strategy.cute_tv_lane_block_id()`` -- a
      TILED axis.  Here identity is the whole content of the test: several of one
      NDTile strategy's own axes appear as ``SymInt``s in the same subscript, so
      "is a SymInt" would admit the wrong one.

    ⚠ NEVER BY SIZE.  Binding a lane axis by size equality is LEDGER E052/E053:
    MEASURED, the size scan mis-binds at 3 of 23 slice sites, and the axis is what
    selects the access width.

    The block id is read through ``_tiled_axis_block_id``, the SAME helper the
    plan-side participant walk uses, so the two gates cannot disagree about which
    axis an entry names -- see that function on why one shared unwrapping is
    load-bearing rather than tidy.
    """
    from ..tile_strategy import _tiled_axis_block_id

    if isinstance(idx, slice):
        return idx == slice(None)
    lane_block_id = strategy.cute_tv_lane_block_id()  # pyrefly: ignore
    if lane_block_id is None:
        return False
    return _tiled_axis_block_id(idx) == lane_block_id


def _maybe_codegen_cute_tv_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    index_exprs: list[str],
    tensor_name: str,
    value: ast.AST,
    extra_mask: ast.AST | None,
) -> ast.AST | None:
    """Emit a TV-layout store, or None to leave the scalar path alone.

    The store writes into the SAME fragment the load reads through, then one
    ``cute.copy`` per lane iteration flushes it via ``partition_D``.  Emission
    order matters and is enforced structurally: fragment writes go inside the
    constexpr V-loop (so ``range_constexpr`` still visits every element -- §8b
    Rule 1), while the flush is appended AFTER that loop by
    ``_cute_tv_partition_hoist``.
    """
    if extra_mask is not None:
        return None
    env = CompileEnvironment.current()
    if env.backend.name != "cute":
        return None
    # Find the reduction strategy whose TV plan owns the trailing axis.
    #
    # ⭐ ``cute_tv_capable()``, NOT ``isinstance(cand, LoopedReductionStrategy)``.
    # The question here is whether a CAPABILITY is present, and the class was only
    # ever a proxy for "these six fields exist" -- see
    # ``ReductionStrategy.cute_tv_capable``.  Any reduction strategy that provides
    # the plan and its emission scaffolding is served, whatever its class.
    #
    # ⚠ AND THE CLASS TEST IS GONE FROM *BOTH* LAYERS.  This loop used to read
    # ``isinstance(cand, ReductionStrategy) and cand.cute_tv_capable()`` -- which
    # removed the class test from the predicate and then reinstated it as the
    # predicate's own precondition, so a non-reduction strategy stayed structurally
    # excluded no matter which fields it had.  ``cute_tv_capable`` is now declared on
    # ``TileStrategy`` (defaulting to False), so every strategy can be ASKED, and the
    # answer -- not the class -- decides.
    strategy = None
    for block_id, loops in state.codegen.active_device_loops.items():
        if not loops:
            continue
        cand = getattr(loops[-1], "strategy", None)
        if cand is not None and cand.cute_tv_capable():
            strategy = cand
            del block_id
            break
    if strategy is None:
        return None
    plan = strategy._cute_tv_plan
    assert plan is not None
    if not _cute_tv_site_eligible(state, strategy, tensor, subscript):
        return None
    # The row predicate gates BOTH the fragment writes (so a phantom row's
    # value is never computed into the fragment) and the flush copy itself (so
    # the clamped row-0 address is never written).  Only the second is
    # load-bearing for correctness -- see the comment at the guard site.
    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    frag_var = _cute_tv_partition_hoist(
        state,
        strategy,
        plan,
        tensor,
        tensor_name,
        _cute_tv_row_index_expr(state, tensor, subscript, index_exprs),
        is_store=True,
        store_mask_expr=mask_expr,
    )
    vi = _cute_tv_vec_lane_var(strategy)
    assign_stmt = statement_from_string(f"{frag_var}[{vi}] = {{value}}", value=value)
    if mask_expr is None:
        state.add_statement(assign_stmt)
    else:
        mask_ast = expr_from_string(mask_expr)
        assert isinstance(mask_ast, ast.expr)
        state.add_statement(
            ast.fix_missing_locations(
                ast.If(test=mask_ast, body=[assign_stmt], orelse=[])
            )
        )
    return ast.Constant(value=None)


def _cute_tv_vec_lane_var(strategy: object) -> str:
    """The constexpr V-loop's target var, read off the emitted loop itself.

    ``PORT_SPEC_layout.md`` §8b Rule 2: the V-loop's bound comes from the
    fragment's own extent, so nothing here tracks a separate ``V``.  Reading the
    loop var back off the AST keeps that true for the subscript as well.
    """
    constexpr_loop = strategy._cute_tv_constexpr_loop  # type: ignore[attr-defined]
    assert isinstance(constexpr_loop, ast.For)
    assert isinstance(constexpr_loop.target, ast.Name)
    return constexpr_loop.target.id


def _cute_tv_row_index_expr(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    index_exprs: list[str],
) -> str | None:
    """The CLAMPED row index for ``local_tile``'s row coordinate, or None for a
    rank-1 (M-broadcast) tensor.

    Reuses the class-2/3 clamp verbatim (``row if mask else 0``,
    ``_cute_row_safe_index_expr``) rather than quack's ``if row < M:`` branch:
    helion's row axis is ``thread_idx()[1]`` and the reduction body contains
    ``sync_threads()``, so a branch there hangs (LEDGER E005).

    ⚠ The row is returned as a tile COORDINATE for ``local_tile``.  It must not
    be folded into the tensor's iterator: MEASURED, that makes the DSL reject
    every ``vec > 1`` copy with "S ptr alignment (16 bits) does not meet
    requirement (128 bits)", because a dynamic pointer offset defeats its
    alignment analysis.
    """
    if tensor.ndim == 1:
        return None
    env = CompileEnvironment.current()
    row_idx = next(idx for idx in subscript if idx is not None)
    row_expr = index_exprs[0]
    block_id = env.get_block_id(row_idx) if isinstance(row_idx, torch.SymInt) else None
    if block_id is None:
        return row_expr
    mask_var = _cute_axis_mask_var(state, block_id)
    if mask_var is None:
        # No mask var => the row tile exactly covers M, so every coordinate this
        # axis can produce is in bounds and no clamp is needed.
        return row_expr
    return _cute_row_safe_index_expr(row_expr, mask_var)


def _cute_tv_forwards_store_fragment(
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
    tensor: torch.Tensor,
) -> bool:
    """Does a load of ``tensor`` read the STORE's fragment instead of GMEM?

    Reads the decision recorded on the strategy by ``_build_cute_tv_plan``; it is
    NOT re-derived here, so the emission and the plan cannot drift about which
    RAWs were resolved by forwarding and which by declining.
    """
    from ..host_function import HostFunction
    from ..reduction_strategy import _cute_tv_alias_key

    keys = getattr(strategy, "_cute_tv_forwarded_raw_keys", None)
    if not keys:
        return False
    return _cute_tv_alias_key(HostFunction.current(), tensor) in keys


def _cute_tv_rmem_slice(
    state: CodegenState,
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
    plan: ChunkTVPlan,
    tensor: torch.Tensor,
    tensor_name: str,
    tile_id: tuple[object, ...],
    frag_var: str,
) -> tuple[str, bool] | None:
    """``(cache_slice, is_first_read)`` for the ``registers`` residency, or ``None``.

    Deliberately the same signature and contract as :func:`_cute_tv_stage_slice`, so the two
    residency arms read as one shape at the call site: ``first_read`` means "read gmem AND
    publish"; ``False`` means "the cache already holds this tile, read it instead".

    ⭐ WHAT THE CACHE IS.  A flat per-thread rmem array of ``num_chunks * lane_extent * vec``
    elements, declared ONCE at a scope enclosing every sweep, indexed
    ``chunk * lane_extent * vec + lane * vec + vi``.  The flat shape is forced, not chosen:
    a fragment is shaped ``_like`` its sweep's partition and a partition is a ``local_tile``
    of its own chunk body, so neither survives into a later sweep (see the ⚠⚠ note at the
    call site for the two NameErrors that established this).

    ⭐⭐ AND EVERY INPUT IS ALREADY IN HAND HERE, WHICH IS THE WHOLE POINT.  ``fuse_tv_copy_sweeps``
    reconstructs each of these after the fact:

    ======================  =========================================  ==========================
    quantity                the AST pass gets it by                    here it is just
    ======================  =========================================  ==========================
    tile identity           unparse + inline temporaries + strcmp      ``tile_id``
    sweep count (``trip``)  parsing the sweep loop's range             ``_cute_stage_num_chunks()``
    ``lane_trip``/``vec``   re-deriving the lane loop's shape           ``plan.lane_extent`` / ``plan.vec``
    element dtype           GUESSING from surrounding typed text        ``tensor.dtype``
    ======================  =========================================  ==========================

    ⚠ The dtype row is the sharpest one: that pass has a helper whose docstring says it
    declines rather than "guessing a width" when the emitter wrote no typed expression around
    the fragment.  At this site the tensor carries its dtype, so the question cannot arise.
    """
    if not _cute_tv_rmem_reuse_enabled(state, strategy, tensor, tensor_name):
        return None
    # ⛔⛔ DECLINE WHERE THE LOOP STRUCTURE IS NOT YET FINAL, i.e. where an AST pass will still
    # RESTRUCTURE the nest this cache is indexed against.
    #
    # The cache index is ``chunk*lane_extent*vec + lane*vec + vi``, computed HERE against the
    # nest as it stands.  On the LOOP-FREE path (``cute_stage_restages_cloned_sweeps()``: one
    # lowering site, remaining sweeps CLONED) the lane-split pass still has to cut that nest into
    # accumulate/finalize/consume -- so an index minted now describes a structure that is about
    # to change.
    #
    # ⚠ AND IT IS WORSE THAN "the index might be stale": the per-element publish
    # (``cache[base+vi] = frag[vi]``) lands INSIDE the constexpr V-loop, and
    # ``_split_lane_loop_over_constexpr_vec`` requires exactly one V-loop with every marker as a
    # DIRECT child of it.  MEASURED: with the publish present that pass returns ``None``, the
    # marker reaches the safety net, and ``partial_fold`` RAISES -- so the emission does not
    # merely risk a wrong index, it **turns working kernels into `BackendUnsupported`**.  Caught
    # by gate level 2 (phase A: 17 passed / 1 failed) after I minted this strategy's cache dict.
    #
    # ⭐ THIS IS THE SAME PREREQUISITE THE ``smem`` ARM HAS, and stating it here is the point:
    # a residency mechanism decided at lowering needs the loop structure to be FINAL at lowering.
    # It is final on the looped/tile regimes (N re-lowered sweeps) and NOT on the loop-free one.
    # ⇒ the loop-free regime is blocked on raising the lane split to lowering -- exactly like
    # ``smem`` -- and until then this declines and the row is re-read from gmem, which is correct
    # and merely unoptimised.
    # ⚠ TESTED VIA THE GRID STATE'S **PREBUILT NEST**, NOT ``cute_stage_restages_cloned_sweeps()``
    # and NOT an ``isinstance`` ladder.  MEASURED: that predicate answers ``False`` on
    # ``PersistentReductionStrategy`` -- it keys on ``offset_var == "0"``, which is a statement
    # about the STAGING coordinate, not about whether an AST pass will restructure the nest -- so
    # gating on it left this shape firing and still raising.
    #
    # ⭐ ``DeviceGridState.prebuilt_lane_nest`` is the honest sentinel and is already the
    # tree's own name for this regime: it is set exactly when a strategy built ONE lane nest up
    # front because it has a single lowering site (``reduction_strategy.py``'s ``codegen_preamble``
    # TV branch), and it is what ``wrap_body`` keys on.  ⇒ asking the state "was a nest supplied?"
    # is the same question, asked of a capability rather than of a class -- which is this repo's
    # documented rule (the enumeration antipattern: ask what a strategy can do, never what it is).
    grid_state = state.codegen.current_grid_state
    if getattr(grid_state, "prebuilt_lane_nest", None) is not None:
        return None
    num_chunks = strategy._cute_stage_num_chunks()  # pyrefly: ignore
    if num_chunks is None or num_chunks < 1:
        # No chunk geometry: ``_cute_stage_num_chunks`` logs its own reasons.  A decline
        # here leaves the gmem read in place, which is correct and merely unoptimised.
        return None
    # ⭐ Both the slot COUNT and the slot INDEX come from the plan
    # (``rmem_slots`` / ``emit_rmem_sweep_base``) rather than being re-multiplied
    # here.  This index is per-THREAD storage and so has no ``tid`` term, unlike the
    # global column index -- a distinction worth stating once, in the plan, instead of
    # at each call site: getting the two confused is the same class of defect as the
    # transposed strides ``ChunkTVPlan.emit_lane_base`` documents.
    slots = plan.rmem_slots(num_chunks)
    if slots < 1:
        return None
    by_tile = strategy._cute_tv_rmem_frag_by_tile  # pyrefly: ignore
    if by_tile is None:
        return None
    lane_var = strategy._cute_reduction_lane_var  # pyrefly: ignore
    chunk_expr = strategy._cute_tv_chunk_index_var  # pyrefly: ignore
    if not isinstance(lane_var, str) or not isinstance(chunk_expr, str):
        return None
    # ⚠ The BASE offset, not the element: the caller adds ``vi`` inside the constexpr V-loop,
    # exactly as the fragment's own ``[vi]`` indexing does.  Keeping the shapes parallel is
    # what lets the consumer substitution be a name swap rather than a re-index.
    base = plan.emit_rmem_sweep_base(chunk_expr, lane_var)
    cached = by_tile.get(tile_id)
    if cached is not None:
        return (f"{cached}[{base}]", False)
    cache_var = state.device_function.new_var(
        f"_tv_rmem_cache_{len(by_tile)}", dce=False
    )
    # ⛔ DECLARED AT THE ENCLOSING SCOPE, NOT IN THE CHUNK BODY -- that is the entire reason
    # this is a flat array.  ``_cute_tv_emit_cross_sweep_stmt`` puts it in the device loop's
    # ``outer_prefix`` (above the ``for roffset`` sweep loop), falling back to the chunk body
    # for the loop-FREE shape, which has one sweep and therefore no scope problem.
    _cute_tv_emit_cross_sweep_stmt(
        state,
        strategy,
        f"{cache_var} = cute.make_rmem_tensor({slots}, "
        f"{CompileEnvironment.current().backend.dtype_str(tensor.dtype)})",
        None,
    )
    by_tile[tile_id] = cache_var
    _cute_tv_record_residency(
        state,
        strategy,
        tensor_name,
        ROW_RESIDENCY_REGISTERS,
        "the row's lanes are cached in registers and re-read from there",
    )
    return (f"{cache_var}[{base}]", True)


def _cute_tv_emit_cross_sweep_stmt(
    state: CodegenState,
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
    text: str,
    emit_chunk_stmt: object,  # Callable[[str], None] | None
) -> None:
    """Emit ``text`` at a scope that ENCLOSES every sweep, falling back to the chunk body.

    The rmem arm needs a fragment declared once, above the per-sweep chunk bodies, because a
    later sweep reads it.  On the LOOPED path the enclosing scope is the device loop's
    ``outer_prefix`` -- the same list ``LoopedReductionStrategy.codegen_reduction`` puts its
    accumulator seed in, and it is flushed above the ``for roffset`` loop, so a declaration
    there is visible to every sweep inside it.

    ⚠ FALLS BACK TO THE CHUNK BODY RATHER THAN ASSERTING.  A path with no active device loop
    for this block (the loop-FREE persistent shape) has only one sweep, so a chunk-body
    declaration is correct there and the fallback is the right answer rather than a
    degradation.  ⛔ It must not raise: reaching this with no loop is a legitimate shape, and
    turning it into a crash would trade a working kernel for an error -- the failure mode this
    area has produced twice.
    """
    block_index = getattr(strategy, "block_index", None)
    if block_index is not None:
        loops = state.codegen.active_device_loops.get(block_index) or []
        for loop_state in reversed(loops):
            outer_prefix = getattr(loop_state, "outer_prefix", None)
            if isinstance(outer_prefix, list):
                outer_prefix.append(statement_from_string(text))
                return
    # ⚠ No enclosing device loop for this block: the loop-FREE persistent shape, which has a
    # SINGLE sweep -- so the chunk body IS the enclosing scope there and the fallback is the
    # right answer, not a degradation.  ``None`` means the caller has no chunk-body emitter to
    # fall back to (the rmem cache decl), in which case the kernel PREAMBLE is the widest
    # scope available and is always correct for a declaration with no local dependencies.
    if emit_chunk_stmt is None:
        state.device_function.preamble.append(statement_from_string(text))
        return
    emit_chunk_stmt(text)  # pyrefly: ignore [not-callable]


def _cute_tv_rmem_reuse_enabled(
    state: CodegenState,
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
    tensor: torch.Tensor,
    tensor_name: str,
) -> bool:
    """May a LATER read of this tensor's tile reuse the fragment the first read filled?

    The three conditions, each a requirement rather than a preference:

    1. ``registers`` must actually be the REQUESTED residency.  Read off
       ``_cute_row_residency_requested`` -- the recorded decision -- rather than re-derived,
       so the emission and the plan cannot drift about which arm is in play.  ⚠ This is the
       same reason :func:`_cute_tv_forwards_store_fragment` reads a recorded key set.
    2. The tensor must be read more than once along the axis
       (:func:`_cute_tv_multi_read_tensors`).  Reusing a fragment for a single-read tensor is
       vacuous; more importantly, the walk is the thing that knows a *second* read exists,
       which nothing local at the first read site does.
    3. ⛔ **Nothing may WRITE the tensor along that axis** (:func:`_cute_tv_stored_tensors`).
       A fragment held across an intervening store serves a PRE-WRITE value -- the one
       failure mode here that is a silent wrong answer rather than a missed optimisation.
       This is the whole-kernel store-alias proof, answered from the device IR at the
       lowering site; the AST pass answers the same question by scanning the emitted body for
       TV store copies (``fuse_tv_copy_sweeps``' ``stored_bases``), and the two were measured
       to agree (``_redfix2/repro/r5_store_walk_agreement.py``).

    ⚠ The env switch is honoured here too, so the fail-capability arm ("the redundant copies
    come back when the cache is withheld") has one instrument for both the AST pass and this
    path.  Without that, "no second copy" is indistinguishable from "the mechanism silently
    stopped firing".
    """
    if os.environ.get("HELION_TV_SWEEP_FUSE", "auto") == "disabled":
        return False
    requested = strategy._cute_row_residency_requested  # pyrefly: ignore
    if requested != ROW_RESIDENCY_REGISTERS:
        # ⭐⭐ THE ONE EXCEPTION, AND IT IS THE SECOND JOB THE AST PASS WAS DOING.
        #
        # Under an explicit ``smem`` request the staged tile admits exactly ONE tensor -- it
        # has a row mode and a chunk mode and NO tensor mode, so a second tensor would alias
        # the first at identical coordinates (a measured pre-existing wrong answer, which is
        # why the gate below refuses).  The REFUSED tensor then "falls back to a gmem
        # re-read"... except that ``fuse_tv_copy_sweeps`` was quietly picking it up and
        # serving it from registers instead.
        #
        # ⇒ that is a REAL job and it belongs here for the same reason the rest does: the
        # decision needs to know a tensor was refused staging, and this site is where the
        # refusal happened.  MEASURED on ``_two_tensor_norm``: ``x`` takes the staged tile and
        # ``y`` was the pass's only remaining fusion in the whole suite.
        #
        # ⚠ SCOPED TIGHTLY: only under a ``smem`` request, and only for a tensor staging has
        # actually DECLINED.  A registers mechanism firing for the tensor that *did* get the
        # tile would be the residency-confusion this axis exists to prevent, and a ``gmem``
        # request must keep meaning "no mechanism" -- it is the explicit opt-OUT.
        if requested != ROW_RESIDENCY_SMEM:
            return False
        # ⛔⛔ AND "HAS STAGING DECLINED?" MUST BE ASKED OF THE **PLAN**, NOT OF
        # ``_cute_tv_staged_tensors``.  That set is populated as sweeps LOWER, so at the FIRST
        # read of the first tensor it is still empty -- the test "not in staged_tensors" is
        # trivially true for *every* tensor at that moment, including the one about to be
        # staged.  MEASURED on ``cross_entropy/32768x1024``: ``logits`` was admitted to the
        # register cache with ``staged=[]``, then staged as well -- two mechanisms on one
        # tensor, and it moved 14 of the 40 frozen cells.
        #
        # ⭐ ``_cute_tv_reload_from`` is the DECISION, taken once per strategy before any sweep
        # lowers, so it does not depend on lowering order.  When it is ``"smem"`` this
        # reduction WILL stage its eligible tensor, and the register arm must stand aside; the
        # genuinely-refused-tensor case (a second multi-read tensor, which the one-tensor tile
        # cannot hold) is the one where staging has already committed to another name.
        if strategy._cute_tv_reload_from == ROW_RESIDENCY_SMEM:  # pyrefly: ignore
            staged = strategy._cute_tv_staged_tensors  # pyrefly: ignore
            # Stand aside unless staging has already committed the tile to a DIFFERENT tensor.
            if not staged or tensor_name in staged:
                return False
    if tensor_name not in _cute_tv_multi_read_tensors(state, strategy):
        return False
    return tensor_name not in _cute_tv_stored_tensors(state, strategy)


def _cute_tv_partition_hoist(
    state: CodegenState,
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
    plan: ChunkTVPlan,
    tensor: torch.Tensor,
    tensor_name: str,
    row_index_expr: str | None,
    *,
    is_store: bool,
    store_mask_expr: str | None = None,
) -> str:
    """Partition ``tensor`` through THE reduction's one TV layout.

    Returns the per-lane fragment variable.  The load leg reads it, the store
    leg writes it; both are ``partition_*`` of the SAME ``get_slice``, so the
    two legs cannot address different elements -- which is what makes class 1
    unrepresentable rather than merely fixed (``PORT_SPEC_layout.md`` §7).

    Emission layout, per chunk (see ``ChunkTVPlan`` for why the row is a tile
    coordinate and not part of ``thr_layout``)::

        # once per kernel, in the preamble
        _tv_atom_0  = cute.make_copy_atom(CopyUniversalOp(), <dtype>,
                                          num_bits_per_copy=<vec*bits>)
        _tv_tiled_0 = cute.make_tiled_copy_tv(_tv_atom_0,
                          make_ordered_layout((1, tpr), order=(1,0)),
                          make_layout((1, vec)))
        _tv_thr_0   = _tv_tiled_0.get_slice(thread_idx()[0])   # ONE slice

        # once per (tensor, chunk), at the top of the chunk body
        _tv_part_x_0 = _tv_thr_0.partition_S(
                           cute.local_tile(x, (1, chunk), (<row>, <chunkidx>)))
        _tv_frag_x_0 = cute.make_rmem_tensor_like(_tv_part_x_0[None, 0, 0])

        # once per lane iteration
        cute.copy(_tv_atom_0, _tv_part_x_0[None, 0, lane], _tv_frag_x_0)   # load
        cute.copy(_tv_atom_0, _tv_frag_o_0, _tv_part_o_0[None, 0, lane])   # store

    The innermost ``range_constexpr(vec)`` loop then reads/writes
    ``_tv_frag_*[vi]``.  That loop is NOT elided (``PORT_SPEC_layout.md`` §8b
    Rule 1): it iterates the FRAGMENT's own extent, so its trip count comes
    from the layout and cannot disagree with it (Rule 2), and every element the
    stride assumes is therefore visited by loop structure as well as by the
    copy's width.

    ⭐ THE PRECONDITION IS ``cute_tv_capable()``, NOT A CLASS.  This used to be
    ``assert isinstance(strategy, LoopedReductionStrategy)``, which on a widened
    path turns a missed optimisation into a compiler CRASH.  It is now an assert
    on the CAPABILITY -- the six fields this function actually dereferences -- so
    the only way to reach it is for a caller to have skipped the same predicate it
    is required to ask.  Every caller checks ``cute_tv_capable()`` (or
    ``_cute_tv_site_eligible``, which subsumes it) and DECLINES on False.
    """
    # ⚠ NO ``isinstance`` HERE EITHER -- the assert is on the CAPABILITY alone.
    # Keeping ``isinstance(strategy, ReductionStrategy) and ...`` would have made this
    # crash for exactly the strategies the capability query exists to admit: a
    # non-reduction strategy that answers True would fail the class half and die on the
    # assert, which is the same "missed optimisation becomes a compiler crash" failure
    # the class test was removed to prevent, one layer further in.
    assert strategy.cute_tv_capable(), (  # pyrefly: ignore [missing-attribute]
        "TV partition hoist reached on a strategy without the TV emission "
        f"scaffolding: {type(strategy).__name__}.  Callers must decline on "
        "``cute_tv_capable()`` being False rather than reach this."
    )
    lane_body = strategy._cute_lane_body
    assert isinstance(lane_body, list)
    fn = state.device_function

    def emit_chunk_stmt(text: str) -> None:
        """Add a per-chunk declaration just ABOVE the lane loop.

        ``_cute_tv_chunk_prefix`` is the chunk body list whose LAST element is
        the lane loop, so declarations insert at ``len - 1``.  Appending would
        place them after the loop that reads them.
        """
        chunk_body = strategy._cute_tv_chunk_prefix
        assert isinstance(chunk_body, list)
        chunk_body.insert(len(chunk_body) - 1, statement_from_string(text))

    # -- ONE atom / tiled_copy / slice PER DTYPE, hoisted to the kernel preamble ----
    #
    # ⭐ A7c: PER DTYPE, not per strategy.  A copy atom's element type must match the tensor
    # being copied, so a mixed-dtype access group (fp8 input + fp32 output is the common one)
    # needs one atom per distinct dtype.  It must keep ONE shared GEOMETRY, and it does:
    # ``plan.for_dtype`` replaces only ``dtype_str``/``dtype_bits``, and ``emit_tiled_copy``
    # builds ``thr_layout=(1,tpr) order=(1,0)`` x ``val_layout=(1,vec)`` -- neither mentions
    # the dtype -- so every per-dtype atom tiles the chunk IDENTICALLY and the legs address the
    # same elements.  ``vec`` is unchanged and still bounded by the WIDEST participant, so
    # there is exactly one answer to "how wide is one copy".
    #
    # ⚠ The single-dtype case is BYTE-IDENTICAL: one distinct dtype means one cache entry,
    # emitted from the same ``plan`` with the same var names in the same order.
    tensor_dtype_str = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    dtype_plan = (
        plan
        if tensor_dtype_str == plan.dtype_str
        else plan.for_dtype(tensor_dtype_str, tensor.element_size() * 8)
    )
    shared = strategy._cute_tv_shared_for_dtype(tensor_dtype_str)
    plan = dtype_plan
    if shared is None:
        atom_var = fn.new_var("_tv_atom", dce=False)
        tiled_var = fn.new_var("_tv_tiled", dce=False)
        thr_var = fn.new_var("_tv_thr", dce=False)
        fn.preamble.append(statement_from_string(plan.emit_copy_atom(atom_var)))
        fn.preamble.append(
            statement_from_string(plan.emit_tiled_copy(tiled_var, atom_var))
        )
        # ⭐ THE SLICE INDEX IS THE **COPY AXIS'S** THREAD AXIS, ASKED FOR RATHER THAN
        # ASSUMED.  This was hardcoded to ``thread_idx()[0]``, which is right for every
        # reduction (helion's row axis stays out of the layout, so the reduction thread axis
        # IS axis 0) and WRONG as soon as a second tiled axis exists: on root-grid 2-D
        # pointwise the copied axis is ``tn``, whose thread axis is 1, and slicing by axis 0
        # gives a silent wrong answer (MEASURED maxdiff 6.8, bf16 vec=8).  The strategy owns
        # the mapping, so it answers -- see ``TileStrategy.cute_tv_thread_axis``.
        tv_axis = strategy.cute_tv_thread_axis()  # pyrefly: ignore [missing-attribute]
        fn.preamble.append(
            statement_from_string(
                plan.emit_get_slice(
                    thr_var,
                    tiled_var,
                    f"cutlass.Int32(cute.arch.thread_idx()[{tv_axis}])",
                )
            )
        )
        shared = (atom_var, thr_var)
        strategy._cute_tv_set_shared_for_dtype(tensor_dtype_str, shared)
    atom_var, thr_var = shared

    # -- one partition per (tensor, direction) within this chunk body ---------
    cache = strategy._cute_tv_partitions
    key = (tensor_name, "D" if is_store else "S")
    if key in cache:
        return cache[key]
    # ⭐ B3 STORE-TO-LOAD FORWARDING.  A load of a tensor whose RAW was classified
    # forwardable reads the value straight out of the STORE's fragment: it is
    # already in a register, the two legs partition through the SAME ``get_slice``
    # so they address identical elements, and the dependence is intra-thread (the
    # TV slice is per-thread), so no barrier is needed.  Returning the ``"D"``
    # entry here also emits NO load copy -- which is the point: it is one fewer
    # gmem read per lane than a correct non-forwarding implementation would do.
    #
    # ⚠ The store leg must already have been emitted, which program order
    # guarantees: forwarding is only classified for a key whose every load follows
    # a store (``_cute_tv_forwardable_raw_keys`` condition 2), and codegen walks
    # the graph in that order.  If it has NOT been emitted, fall through and
    # partition normally -- that cannot be reached today, and reaching it would
    # produce today's (correct, un-forwarded) shape rather than a wrong answer.
    if not is_store and _cute_tv_forwards_store_fragment(strategy, tensor):
        store_frag = cache.get((tensor_name, "D"))
        if store_frag is not None:
            cache[key] = store_frag
            return store_frag
    slot = len(cache)
    part_var = fn.new_var(f"_tv_part_{slot}", dce=False)
    frag_var = fn.new_var(f"_tv_frag_{slot}", dce=False)
    tile_var = fn.new_var(f"_tv_tile_{slot}", dce=False)

    chunk_idx_var = strategy._cute_tv_chunk_index_var
    assert isinstance(chunk_idx_var, str)
    if row_index_expr is None:
        # A rank-1 tensor (``weight[:]``): view it as (1, N) with row stride 0
        # so it partitions through the SAME slice as the row tensors.  quack
        # does this host-side (``rmsnorm.py:124-127``).
        view_var = fn.new_var(f"_tv_bcast_{slot}", dce=False)
        emit_chunk_stmt(
            plan.emit_row_broadcast_view(
                view_var,
                tensor_name,
                f"{tensor_name}.shape[0]",
                f"{tensor_name}.layout.stride[0]",
            )
        )
        emit_chunk_stmt(plan.emit_local_tile(tile_var, view_var, "0", chunk_idx_var))
    else:
        emit_chunk_stmt(
            plan.emit_local_tile(tile_var, tensor_name, row_index_expr, chunk_idx_var)
        )
    emit_partition = (
        plan.emit_partition_dest if is_store else plan.emit_partition_source
    )
    emit_chunk_stmt(emit_partition(part_var, thr_var, tile_var))
    emit_chunk_stmt(plan.emit_fragment_like(frag_var, part_var))
    # ⭐ RECORD THE TILE'S IDENTITY, keyed on the fragment variable (task 4).  Every
    # component is already in hand HERE, at the one site that emits the tile -- which is the
    # whole point: ``fuse_tv_copy_sweeps`` must prove two copies address the same tile in
    # order to delete the second, and it currently recovers that by unparsing, inlining
    # single-assignment temporaries and STRING-COMPARING the reconstructed source, because
    # each sweep mints its own ``_tv_tile_N`` / ``_tv_part_N`` names.  The identity was
    # known at emission and discarded into a variable name.
    #
    # ⚠ ROW COORD ``"0"`` FOR THE RANK-1 BROADCAST VIEW, matching what was emitted above --
    # the id must describe the tile that was actually emitted, not the call's arguments, or
    # a broadcast weight and a real row could collide on the same id.
    #
    # ⚠⚠ THE COLUMN COORDINATE IS THE CHUNK INDEX **EXPRESSION**, NOT ITS VARIABLE NAME.
    # MEASURED: each sweep of one row mints its own ``_tv_chunk_N`` and all of them hold the
    # SAME value (``roffset_1 // _REDUCTION_BLOCK_1``), so keying on the name reports three
    # distinct tiles where there is one and can never identify a re-read -- which is exactly
    # what ``fuse_tv_copy_sweeps`` works around by inlining single-assignment temporaries
    # before comparing text.  Caught by ``TestTvTileIdChannel``'s discrimination arm, which
    # is why that arm asserts a SHARED id and not merely that ids exist.
    tile_id = (
        tensor_name,
        plan.chunk,
        "0" if row_index_expr is None else row_index_expr,
        getattr(strategy, "_cute_tv_chunk_index_expr", None) or chunk_idx_var,
    )
    fn.cute_state.tv_tile_ids[frag_var] = tile_id

    # ⭐⭐ ``reload_from="registers"`` AT LOWERING (G2's `registers` arm).
    #
    # The FIRST read of a tile reads gmem and additionally publishes its lanes into a
    # per-thread register array; every LATER read of the SAME tile reads that array instead.
    # Structurally identical to the ``smem`` arm below -- same first/later shape, one level
    # closer to the ALU -- which is why the two are handled side by side here.
    #
    # ⛔ WHY THIS IS NOT A PORT OF ``fuse_tv_copy_sweeps``.  That pass must PROVE two copies
    # address the same tile before it may delete the second, and after the fact it can only
    # do that by unparsing each copy, inlining single-assignment temporaries and
    # STRING-COMPARING the reconstructed source -- because every sweep mints its own
    # ``_tv_tile_N`` / ``_tv_part_N`` / ``_tv_chunk_N``.  Its sweep grouping and its
    # producer/consumer key matching exist ONLY to recover information that was known here
    # and, in the words of the comment above, "discarded into a variable name".  ⇒ at this
    # site the tile id is in hand, so the match is one dict lookup and there is nothing to
    # re-derive.
    #
    # ⚠⚠ AND THE CACHE CANNOT SIMPLY BE THE FRAGMENT, WHICH THE FIRST ATTEMPT HERE ASSUMED.
    # ``emit_fragment_like`` shapes the fragment ``_like`` THIS sweep's partition, and the
    # partition is a ``local_tile`` of THIS chunk body -- so the fragment is out of scope in
    # a later sweep and hoisting its declaration is impossible (it would reference the
    # partition above the partition's own definition).  MEASURED, in that order:
    # ``NameError: name '_tv_frag_0' is not defined``, then after hoisting,
    # ``NameError: name '_tv_part_0' is not defined``.  ⇒ the cache must be a FLAT,
    # explicitly-sized rmem array declared at a scope enclosing every sweep -- which is
    # exactly the shape ``fuse_tv_copy_sweeps`` emits, and now the reason for that shape is
    # recorded rather than rediscovered.
    rmem = (
        None
        if is_store
        else _cute_tv_rmem_slice(
            state, strategy, plan, tensor, tensor_name, tile_id, frag_var
        )
    )

    # -- one copy per lane iteration -----------------------------------------
    lane_var = strategy._cute_reduction_lane_var
    assert isinstance(lane_var, str)
    lane_slice = plan.emit_lane_slice(part_var, lane_var)
    # ``reload_from="smem"``: the FIRST read of a tensor still comes from GMEM and
    # additionally writes the staged tile; every LATER read of the same tensor
    # comes from the tile instead.  ``stage`` is None when staging does not apply.
    stage = (
        None
        if is_store
        else _cute_tv_stage_slice(
            state, strategy, plan, tensor, tensor_name, emit_chunk_stmt, frag_var
        )
    )
    if stage is not None:
        stage_slice, first_read = stage
        if first_read:
            # First sweep: read GMEM, then publish to SMEM.  Two statements, and
            # the publish must follow the read of the SAME fragment, so it is
            # appended right after the load copy below.
            copy_text = plan.emit_copy(lane_slice, frag_var, atom_var)
        else:
            # Second sweep: quack ``rmsnorm.py:334`` -- ``autovec_copy(tXsX,
            # tXrX)`` REPLACES the gmem copy.  This is the whole point of the
            # knob: the row is not re-read from DRAM.
            copy_text = plan.emit_stage_copy(stage_slice, frag_var)
    elif rmem is not None:
        # ⭐⭐ ``reload_from="registers"``, the SAME first/later shape as ``smem`` above and
        # deliberately written to look like it -- one level closer to the ALU.
        rmem_slice, rmem_first = rmem
        if rmem_first:
            # First sweep: read GMEM as usual.  The per-element publish into the cache is
            # emitted inside the constexpr V-loop by the caller (the cache is indexed per
            # element, so it cannot be one whole-fragment copy the way SMEM's publish is).
            copy_text = plan.emit_copy(lane_slice, frag_var, atom_var)
        else:
            # Later sweep: the cache holds this tile, so there is NO gmem copy at all.  The
            # fragment is still declared (the V-loop's reads are rewritten onto the cache by
            # the caller), and this statement is the one the mechanism exists to delete.
            copy_text = None
        # ⚠ THE ARTIFACT MARK.  ``cute_observed_row_residency`` reads the residency off the
        # EMITTED source; ``registers`` is defined by an ABSENCE (no second gmem copy) and a
        # regex cannot match an absence.  MEASURED without it: the second read WAS eliminated
        # and the marker still said ``gmem``, so an explicit request raised
        # ``CuteRowResidencyUnavailable`` -- the artifact contradicting the emission.
        emit_chunk_stmt(repr(_RMEM_REUSE_MARK))
    else:
        copy_text = (
            plan.emit_copy(frag_var, lane_slice, atom_var)
            if is_store
            else plan.emit_copy(lane_slice, frag_var, atom_var)
        )
    # ── THE TWO PREDICATES ON A COPY, and they guard DIFFERENT axes ────────────
    #
    # * ``store_mask_expr``  -- the ROW axis (``row < M``), store leg only.
    # * ``tail_pred``        -- the REDUCTION axis (``base < N``), BOTH legs, and
    #                           only when the tile was rounded up past ``N``.
    #
    # They are ANDed rather than nested so the emitted form stays one ``if`` and
    # ``fuse_tv_copy_sweeps`` sees the same statement shape either way.
    guards: list[str] = []
    tail_pred = strategy.cute_tv_tail_predicate()
    if tail_pred is not None:
        # ⚠ THE STORE LEG IS WHY THIS IS A CORRECTNESS PREDICATE AND NOT AN
        # OPTIMISATION.  The rounded-up tile's last chunk addresses columns
        # ``>= N``, and ``local_tile`` on a row-major tensor resolves column
        # ``N + k`` of row ``m`` to column ``k`` of row ``m + 1``.  So an
        # unguarded flush writes the tail fragment straight into the NEXT ROW --
        # the same hazard class as the reverted ND mask elision (E070/E071),
        # which shipped an unconditional store.  On the LOAD leg the same guard
        # is memory safety: the final row's tail would read past the allocation.
        #
        # By ``ragged_tail`` invariant I2 a fragment is wholly in or wholly out of
        # bounds, so this ONE scalar compare is exact for all ``vec`` elements --
        # no ``predicate_k`` Boolean tensor is needed.
        guards.append(tail_pred)
    if is_store and store_mask_expr is not None:
        # quack ``rmsnorm.py:351-352`` -- ``if row < shape[0]: copy(tXrO, tXgO)``.
        #
        # ⚠ The GUARD MUST BE ON THE COPY, not only on the fragment writes.  The
        # row clamp folds a phantom row's coordinate to 0, so an unguarded flush
        # copies that thread's (unwritten) fragment straight over row 0 --
        # MEASURED as ``class3_row_tail_correct`` relerr 110025 at M=33.  The
        # clamp keeps the ADDRESS legal; only this predicate keeps the WRITE
        # from happening.
        #
        # Safe despite E005's "a branch HANGS" warning, which is specifically
        # about a barrier under divergent control flow: this branch wraps one
        # ``cute.copy`` and contains no ``sync_threads``.  quack nests it the
        # same way, and it is MEASURED correct at M=255/253 in
        # ``_redfix/repro/probe_target_body.py``.
        guards.append(store_mask_expr)
    # Order is correctness here, and it is anchored to the constexpr loop NODE
    # rather than to a list position, because both legs mutate ``lane_body``:
    #   * a LOAD copy must precede the loop that reads the fragment;
    #   * a STORE copy must follow the loop that fills it.
    constexpr_loop = strategy._cute_tv_constexpr_loop
    assert isinstance(constexpr_loop, ast.For)
    loop_pos = next(i for i, stmt in enumerate(lane_body) if stmt is constexpr_loop)
    if copy_text is None:
        # ⭐ THE DELETED COPY: the rmem cache already holds this tile, so no load is emitted
        # at all.  Nothing is inserted, and the fragment the V-loop reads is filled by the
        # per-element cache reads added below.  ⚠ This is the ONE path that reaches here with
        # no copy; every other arm has one.
        pass
    else:
        copy_stmt: ast.AST
        if guards:
            copy_stmt = ast.fix_missing_locations(
                ast.If(
                    test=cast("ast.expr", expr_from_string(" and ".join(guards))),
                    body=[cast("ast.stmt", statement_from_string(copy_text))],
                    orelse=[],
                )
            )
        else:
            copy_stmt = statement_from_string(copy_text)
        lane_body.insert(loop_pos + 1 if is_store else loop_pos, copy_stmt)
    if rmem is not None:
        # ⭐⭐ THE PER-ELEMENT LEG, and it must be per-element rather than one whole-fragment
        # copy the way SMEM's publish is.  The cache is a FLAT array indexed
        # ``chunk*lane_extent*vec + lane*vec + vi``, so its element ``vi`` and the fragment's
        # element ``vi`` correspond one-for-one but the two objects have different shapes --
        # there is no single ``autovec_copy`` between them.  Emitted INSIDE the constexpr
        # V-loop, where ``vi`` exists:
        #   * first read  -> ``cache[base + vi] = frag[vi]``  (publish, after the gmem read)
        #   * later read  -> ``frag[vi] = cache[base + vi]``  (serve, with no gmem read)
        #
        # ⚠ SERVING BY FILLING THE FRAGMENT, rather than rewriting every downstream reader to
        # index the cache, is what keeps this a LOCAL change.  ``fuse_tv_copy_sweeps`` has to
        # do the rewrite (its ``frag[vi]`` redirect) because by the time it runs the readers
        # already exist; here they have not been emitted yet, so filling the fragment they are
        # about to read is both simpler and impossible to get partially wrong.
        rmem_slice, rmem_first = rmem
        vec_var = constexpr_loop.target
        assert isinstance(vec_var, ast.Name)
        elem = f"{rmem_slice[:-1]} + {vec_var.id}]"
        constexpr_loop.body.insert(
            0 if not rmem_first else len(constexpr_loop.body),
            cast(
                "ast.stmt",
                statement_from_string(
                    f"{elem} = {frag_var}[{vec_var.id}]"
                    if rmem_first
                    else f"{frag_var}[{vec_var.id}] = {elem}"
                ),
            ),
        )
    if stage is not None and stage[1]:
        # The publish, immediately AFTER the gmem read that filled the fragment
        # and still BEFORE the constexpr loop.  quack ``rmsnorm.py:233`` stages
        # from gmem straight into ``sX`` with cp.async; helion cannot, because its
        # fragment is also the reduction's input -- so the fragment is the source
        # and this is rmem->smem.  Same tile, same layout, so the round trip is
        # element-for-element (asserted by ``reload_smem_is_chunk_indexed`` in the
        # gate, which also proves no barrier is needed: writer and reader are the
        # SAME thread through the SAME slice, so the dependency is intra-thread).
        # The publish targets SMEM, whose tile is sized from the SAME rounded
        # extent (``_cute_stage_num_chunks``), so the tail slot exists and writing
        # it is in bounds -- no guard needed, and adding one would only cost a
        # branch.  What it publishes for a tail lane is the fragment's stale
        # content, which the second sweep reads back and the op identity at the
        # combine then discards (``ragged_tail`` I4/I6).
        lane_body.insert(
            loop_pos + 1,
            statement_from_string(plan.emit_stage_copy(frag_var, stage[0])),
        )
    cache[key] = frag_var
    return frag_var


def _cute_tv_record_residency(
    state: CodegenState,
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
    tensor_name: str,
    effective: str,
    why: str,
) -> None:
    """Record where ``tensor_name``'s row ACTUALLY lives between sweeps.

    Two outputs, deliberately:

    * a ``log.debug`` line, for a human running with
      ``HELION_LOGS=+helion._compiler.reduction_strategy``;
    * ⭐ a line **on the emitted artifact**, so the record cannot drift from reality
      and a harness can grep it.  Produced at the emission site for exactly that
      reason: the ``Config`` object is not evidence -- MEASURED, a cell can carry
      ``cute_reduction_reload=['smem']`` while the kernel stages nothing, which is
      how ``frozen_configs.json`` came to record residencies that were never used.

    ⚠ ``ast`` HAS NO COMMENT NODE.  ``statement_from_string("# ...")`` parses to zero
    statements and raises -- an in-tree comment at
    ``cute/peel_ragged_tile.py`` already records this.  So the marker is emitted as a
    module-level string expression (a docstring-shaped no-op), which survives the
    unparser verbatim and does not depend on ``settings.output_origin_lines`` the way
    the ``# src[...]`` location comments do.

    Idempotent per ``(tensor, effective)``: one reduction has many load sites and they
    must not each append a line.

    ⚠ THIS FUNCTION NO LONGER WRITES THE ARTIFACT, and that is a fix rather than a
    reduction in scope.  It used to append one marker line PER TENSOR, which produced

        row residency: x -> smem  (staged: ...)
        row residency: weight -> gmem  (rank-1 M-broadcast tensor; ...)

    on a single kernel -- two statements that do not answer "what is this kernel's row
    residency", because the reduced row and a broadcast weight vector are not the same
    question.  The per-tensor facts are still recorded, at ``log.debug``, where the
    detail is useful and the ambiguity is harmless.  The ONE canonical line is written
    by :func:`cute_emit_row_residency_marker` below, from the FINAL emitted body.
    """
    key = f"{tensor_name}\x00{effective}"
    seen = getattr(strategy, "_cute_tv_residency_recorded", None)
    if seen is None:
        seen = set()
        strategy._cute_tv_residency_recorded = seen  # type: ignore[attr-defined]
    if key in seen:
        return
    seen.add(key)
    block_index = getattr(strategy, "block_index", "?")
    log.debug(
        "cute row residency block=%s tensor=%s -> %s (%s)",
        block_index,
        tensor_name,
        effective,
        why,
    )
    # The DECLINE REASONS, accumulated per strategy so the canonical marker can name
    # the cause without re-deriving it.  Only the reduced row's own declines are
    # interesting; a rank-1 broadcast weight is never staged by design and its
    # "decline" is not a diagnosis of anything.
    if (
        effective != "smem"
        and why != "not requested"
        and tensor_name in _cute_tv_multi_read_tensors(state, strategy)
    ):
        reasons = getattr(strategy, "_cute_row_residency_reasons", None)
        if reasons is None:
            reasons = []
            strategy._cute_row_residency_reasons = reasons  # type: ignore[attr-defined]
        if why not in reasons:
            reasons.append(why)


# The three EMITTED signatures of a row residency, as regexes over the final source.
# ⭐ These are read off the ARTIFACT rather than off the config, which is the whole
# point: MEASURED, a cell can carry ``cute_reduction_reload=['smem']`` while staging
# nothing, so the ``Config`` is not evidence.  Counts measured on this tree for
# ``rms_norm``-shaped M=2048 N=8192 bf16 -- registers 0/1/3, smem 4/0/3, gmem 0/0/4
# (``_tv_spart`` / rmem-cache-decl / ``cute.copy(``).
_RESIDENCY_SMEM_RE = re.compile(r"_tv_spart_[0-9]+")
_RESIDENCY_REGISTERS_RE = re.compile(r"_tv_sweep_cache_[0-9]+ = cute\.make_rmem_tensor")
# The LOWERING path's signature: the emitter leaves this mark where it declined to emit a
# second copy because the fragment already holds the tile.
#
# ⚠ A MARK IS NEEDED AT ALL because the emission's whole point is that there is NO new
# statement to look for -- ``registers`` at lowering is defined by an ABSENCE (the second
# gmem copy), and a regex cannot match an absence.  The alternative, counting copies here,
# would have to re-derive which tensor is the reduced row, which is exactly the config-vs-
# artifact drift ``cute_observed_row_residency`` exists to prevent.
#
# ⚠ Emitted as a bare STRING-EXPRESSION statement, matching how the canonical residency
# marker itself is emitted (``statement_from_string(repr(text))``), because a ``#`` comment
# is not representable in the AST the emitter builds.
#
# ⛔⛔ AND IT MUST **NOT** CONTAIN THE TOKEN ``row residency:``.  That token is the CANONICAL
# marker's, and the contract is **exactly one ``row residency:`` statement per emitted kernel**
# -- a rule that exists because an earlier version appended one line PER TENSOR and produced two
# statements that did not answer "what is this kernel's row residency".  MEASURED: with this mark
# spelled ``row residency: registers (fragment reused...)`` the count went to **3** and
# ``_notes/tests/test_residency_enum.py`` went green -> RED on exactly that assertion.  ⇒ the
# mark says the same thing in a spelling that cannot be mistaken for the canonical line.
_RMEM_REUSE_MARK = "tv rmem reuse: second read served from the register cache"
_RESIDENCY_RMEM_REUSE_RE = re.compile(re.escape(_RMEM_REUSE_MARK))


def cute_observed_row_residency(source: str) -> str:
    """Which residency the EMITTED INSTRUCTIONS show.  Independent of any config.

    Order matters and is not arbitrary: staging and the register cache are mutually
    exclusive in the emission (MEASURED -- with ``reload="smem"`` and a positive budget
    the staged tile is emitted and the register cache never fires), so ``smem`` is
    tested first and ``gmem`` is the residual.  ``gmem`` being the fallthrough is
    correct because it is defined by the ABSENCE of both caches plus the second gmem
    read, and the second read is implied once neither cache exists.

    ⭐⭐ ``registers`` NOW HAS **TWO** EMITTED SIGNATURES, and reading only the first one
    made the better emission read as a DECLINE.

    * ``fuse_tv_copy_sweeps`` (the AST post-pass) materialises an explicit rmem cache
      tensor -- ``_tv_sweep_cache_N = cute.make_rmem_tensor`` -- copies the producer's
      fragment into it, and redirects the consumer's element reads at it.
    * The LOWERING path (``_cute_tv_partition_hoist``'s rmem branch) does not need a cache
      tensor at all: the second read simply **reuses the fragment the first read filled**
      and emits no copy.  ⇒ strictly less code and one fewer register-to-register copy,
      but it leaves NO ``_tv_sweep_cache_N`` for a regex to find.

    ⛔ MEASURED, and this is why the second signature exists: with only the cache-tensor
    regex, ``rms_norm`` at ``cute_row_residency=['registers']`` reused the fragment
    correctly (the row's second gmem read WAS eliminated) and the marker still reported
    ``gmem``, so the explicit-request check raised ``CuteRowResidencyUnavailable`` -- the
    artifact contradicting the emission it was describing.

    ⇒ the second signature is the ABSENCE of a second gmem read of a multi-read row, which
    is what ``registers`` *means*; it is supplied by the emitter as an explicit marker
    comment rather than inferred by counting copies here, because a count would have to
    re-derive which tensor is the row -- exactly the drift this function exists to avoid.
    """
    if _RESIDENCY_SMEM_RE.search(source):
        return "smem"
    if _RESIDENCY_REGISTERS_RE.search(source) or _RESIDENCY_RMEM_REUSE_RE.search(
        source
    ):
        return "registers"
    return "gmem"


def cute_emit_row_residency_marker(
    fn: object,  # DeviceFunction at runtime
    body: list[ast.stmt],
) -> ast.stmt | None:
    """⭐ THE ONE canonical row-residency statement, derived from the FINAL body.

    Returns the marker statement, or ``None`` when this kernel has no reduction row to
    describe (no strategy participates in the residency axis).

    ⭐ WHY IT IS COMPUTED HERE AND NOT AT THE LOAD SITE.  The choice between
    ``registers`` and ``gmem`` is not made at any load site: it is made by
    ``fuse_tv_copy_sweeps``, an AST pass that runs AFTER every load has lowered and can
    still decline on its register budget.  A marker written at the load site therefore
    could not know which of the two it got, and the honest thing it could say is
    "gmem/registers" -- which is exactly what the previous version said, and which
    names two values where the reader needs one.  Reading the FINAL body removes the
    guess: the statement is a function of the emitted instructions, so
    "stated == observed" holds by construction rather than by discipline.

    ⚠ It is still produced AT THE EMISSION SITE in the sense that matters -- from the
    artifact, never from the ``Config``.  Moving it later makes that property stronger,
    not weaker: it now sees the last pass's decision too.

    The wording is ``row residency: <effective>`` plus, only when the request was NOT
    granted, ``(requested <r>; declined: <reason>)``.  The token ``row residency:`` is
    load-bearing -- ``_notes/probe_2c_residency_report.py`` and the gate harness grep
    for it -- so it is spelled exactly once, here.
    """
    # ⚠ THE PREDICATE IS "OWNS A RESIDENCY SLOT", NOT ``cute_tv_capable()``.  Gating on
    # the TV plan looks natural and is wrong in the one direction that matters: a
    # reduction whose plan was DECLINED (e.g. ``cute_vector_widths=[1]``, where no plan
    # is built at all) is precisely the case where a ``smem`` request cannot be honoured
    # -- so gating on the plan made the loudest decline the only SILENT one.  MEASURED
    # before this predicate widened: ``vw=1`` with ``cute_row_residency=["smem"]``
    # emitted ZERO marker lines while the config still read ``['smem']``, which is the
    # exact defect the marker exists to remove, reintroduced one level up.
    #
    # ⚠ AND THE SECOND CONJUNCT ASKS ``cute_stage_block_id()``, NOT ``block_index``.  The
    # ``block_index`` spelling silently excluded ``CuteNDTileStrategy``, which deliberately
    # does NOT define that property: ``block_ids[0]`` is meaningless on a strategy driving
    # three axes, so the staging protocol asks ``cute_stage_block_id()`` instead.  MEASURED:
    # a tile-regime kernel that STAGED (``alloc_smem`` 1, one ``partition_D`` + one
    # ``partition_S``, a staged read replacing a gmem load) emitted ZERO marker lines -- the
    # exact "a cell records a residency the kernel never used" defect this marker exists to
    # remove, reappearing on a newly admitted path.
    # ⚠ THE FIRST CONJUNCT USED TO BE
    # ``getattr(s, "_cute_row_residency_requested", None) is not None`` AND IT WAS
    # ALWAYS TRUE (task 6a).  ``_cute_row_residency_requested`` is declared on
    # ``TileStrategy`` -- the common base of EVERY strategy -- with the value
    # ``ROW_RESIDENCY_GMEM``, so the getattr never took its default and the test read
    # ``"gmem" is not None``.  It looked like a guard against a strategy that never
    # participated in the axis; it filtered nothing.  ⭐ The REAL predicate is the second
    # conjunct, which is why the comment above is written about that one.
    strategies = [
        s
        for s in fn.tile_strategy.strategies  # type: ignore[attr-defined]
        if s.cute_stage_block_id() is not None
    ]
    if not strategies:
        return None
    # The reduced row is one row however many load sites read it, so one marker per
    # kernel.  With several participating reductions the first is reported and the rest
    # stay in the DEBUG log -- ONE canonical line is the contract, and a kernel whose
    # two reductions disagree is a case no shape in the tree produces today.
    strategy = strategies[0]
    effective = cute_observed_row_residency(
        "\n".join(ast.unparse(stmt) for stmt in body)
    )
    # ⚠⚠ THIS getattr's DEFAULT DID NOT MATCH THE DECLARED ONE, and that is why it is
    # called out rather than quietly deleted (task 6a).  The field is declared
    # ``ROW_RESIDENCY_GMEM`` on ``TileStrategy``, but the getattr's fallback was
    # ``effective`` -- and those two behave DIFFERENTLY one line below: ``effective`` makes
    # the ``requested == effective`` branch taken (so the marker reads a bare
    # ``row residency: <arm>``), while ``"gmem"`` would make an observed ``registers`` or
    # ``smem`` read as ``(requested gmem; declined: ...)``.  The mismatch was unreachable
    # ONLY because the comprehension above guaranteed the attribute exists -- i.e. it was
    # dead code protected by an always-true filter, and loosening either one would have
    # changed the marker text the gate harness greps for.
    requested = strategy._cute_row_residency_requested  # pyrefly: ignore [missing-attribute]
    if requested == effective:
        text = f"row residency: {effective}"
    else:
        # ⭐ MOST SPECIFIC CAUSE FIRST.  A site that knows exactly why it refused
        # (``cute_stage_feasible``'s SMEM-budget arithmetic) sets
        # ``_cute_row_residency_decline``; the per-tensor staging chain contributes
        # coarser reasons; and the register budget leaves no trace at all, because
        # ``fuse_tv_copy_sweeps`` declines inside an AST pass that never sees a
        # strategy.  Two different declines MUST read differently -- one hardcoded
        # string is not a diagnosis, it just moves the silence.
        specific = strategy._cute_row_residency_decline  # pyrefly: ignore [missing-attribute]
        reasons = list(getattr(strategy, "_cute_row_residency_reasons", None) or [])
        if specific:
            why = specific
        elif reasons:
            why = "; ".join(reasons)
        elif requested == ROW_RESIDENCY_REGISTERS:
            # ⛔⛔ THIS USED TO NAME THE BUDGET UNCONDITIONALLY, AND IT WAS WRONG ON 10 OF
            # THE 13 CELLS IT FIRED ON.  It read:
            #
            #     "the row's cache footprint exceeds the cute_tv_sweep_cache register budget"
            #
            # -- a hardcoded string asserting the budget as the cause, which is the exact
            # thing the ⭐ comment above forbids ("one hardcoded string is not a diagnosis,
            # it just moves the silence").
            #
            # MEASURED 2026-08-01 on all 40 frozen cells, by re-emitting the whole table
            # with the cap lifted through the pass's own override
            # (``HELION_TV_SWEEP_FUSE_SLOTS=1000000``) -- the fail-capability arm for "is
            # the budget what refuses these?":
            #   * 11 cells requested ``registers`` and emitted ``gmem``;
            #   * at an INFINITE budget only **ONE** of them flipped
            #     (``cross_entropy/8192x100000``, which needs 784 slots > 128);
            #   * the other 10 still emitted ``gmem``, so the budget was NOT their cause,
            #     while every one of them carried the budget string above.
            #
            # The real cause for those 10 is structural, inside ``fuse_tv_copy_sweeps``:
            # ``_resolve_sweep`` refuses a sweep whose lane trip exceeds 64
            # (``fuse_tv_copy_sweeps.py`` ~:331), among ~13 other shape declines, and none
            # of them records anything -- the pass runs on the AST and never sees a
            # strategy.  ⇒ the honest answer names the two candidates and says which
            # instrument separates them, rather than picking one and sounding certain.
            # ⛔⛔ THIS STRING IS SHIPPED TO USERS AND IT NAMED TWO THINGS THAT NO LONGER
            # EXIST.  It used to read "either the row's footprint exceeds the
            # cute_tv_sweep_cache budget, or fuse_tv_copy_sweeps declined on the sweep's
            # shape ...; re-run with HELION_TV_SWEEP_FUSE_SLOTS raised to tell the two
            # apart" -- but task 1 deleted BOTH the ``cute_tv_sweep_cache`` budget and the
            # ``HELION_TV_SWEEP_FUSE_SLOTS`` override.  ⇒ it offered a diagnosis that cannot
            # happen and an action that cannot be taken, which is worse than saying less:
            # a user following it gets "unrecognised env var" and concludes the tool is
            # broken.  With the budget gone there is only ONE cause left, so the string can
            # be both shorter and TRUE.
            #
            # ⚠ The remaining instrument is the whole-pass kill switch
            # (``HELION_TV_SWEEP_FUSE=disabled``), which is an A/B rather than a way to
            # widen a budget -- so it is offered as attribution, not as a fix.
            why = (
                "fuse_tv_copy_sweeps declined this loop on the sweep's shape (e.g. lane "
                "trip > 64, fewer than two sibling sweeps, or a store to the same tensor "
                "between the reads); set HELION_TV_SWEEP_FUSE=disabled to confirm by A/B "
                "that the pass is what changes the emitted kernel"
            )
        else:
            why = "no reason recorded"
        text = f"row residency: {effective}  (requested {requested}; declined: {why})"
        # ⭐⭐ AN *EXPLICIT* REQUEST THAT WAS NOT GRANTED IS AN ERROR, NOT A FOOTNOTE.
        # This is the whole content of "make ``cute_row_residency`` authoritative": if the
        # user names a residency they get it or a hard error, never a different one.
        #
        # ⚠ ONLY WHEN THE CALLER *WROTE* THE KEY.  A residency that ``_fill_missing``
        # supplied must still DECLINE, because the ladder provably cannot predict the
        # decline -- it sees only ``block_ids`` + ``size_hint``, while the outcome depends
        # on ``ChunkTVPlan.lane_extent``, ``thread_block_dims()``, the emitted
        # ``cluster_n`` and the running SMEM charge, none of which exist at normalize
        # time.  Raising there would fail a kernel over a default the user never saw.
        # ⇒ the question is USER-provenance, and the carrier for it is
        # ``ConfigSpec.cute_row_residency_is_explicit`` (which is where the measurement
        # showing ``_cute_row_residency_requested_by_block`` CANNOT answer it lives).
        #
        # ⚠⚠ AND IT MUST RAISE HERE -- AT CODEGEN -- NOT IN ``normalize``.  MEASURED:
        # ``benchmark_provider.py`` ~:744 wraps ``kernel.compile_config`` in
        # ``except Exception`` and SKIPS the config ("Skipping config that failed to
        # compile"), so a codegen raise is absorbed by the autotuner.  ``normalize`` is
        # NOT so wrapped -- ``ConfigGeneration.unflatten`` calls it with
        # ``_fix_invalid=True`` and ``base_search.benchmark_flat`` catches only
        # ``InvalidConfig`` -- which is why an earlier attempt at this rule (recorded at
        # ``ConfigSpec._reconcile_cute_residency_budget``) was unsound and had to become a
        # reconcile.  The polarity is: raise at codegen, never at normalize.
        #
        # ⚠ AND A DECLINE MUST STAY A DECLINE FOR EVERYTHING ELSE.  An unconditional
        # raise at a decline site broke all 8 attention examples, twice
        # (``cute_stage_feasible``'s docstring records it).  The guard here is
        # provenance, so a kernel that never named the key cannot reach the raise.
        if _cute_row_residency_request_was_explicit(fn):
            raise exc.CuteRowResidencyUnavailable(requested, effective, why)
    return statement_from_string(repr(text))


def _cute_row_residency_request_was_explicit(fn: object) -> bool:
    """Did the CALLER write ``cute_row_residency``, per the normalizer's record?

    ⚠ Fail-safe FALSE.  Any path that cannot answer (no env, no spec, a config that did
    not come through this spec's ``normalize``) degrades to today's silent decline rather
    than to a new raise -- the polarity that keeps this change from failing a kernel it
    has no evidence about.

    ⛔⛔ ``HELION_CUTE_ROW_RESIDENCY_HINT=1`` RESTORES THE PRE-TASK-1 SEMANTICS (report the
    decline on the artifact, do not raise), AND IT EXISTS FOR ONE SPECIFIC REASON: to make
    a REAL SPEC CONFLICT falsifiable instead of asserted.

    Three GRADED contract tests in ``_notes/tests/`` -- which the run protocol forbids
    editing -- assert the OPPOSITE polarity to task 1's rule.  They were GREEN at baseline
    and they specify that an explicit-but-unfulfillable request must COMPILE and report:
        test_residency_enum.py::test_declined_request_is_stated_with_a_reason
        test_residency_enum.py::test_a_second_decline_cause_is_distinguishable
        test_residency_enum.py::test_registers_over_budget_declines_observably
        test_staging_lane_extent.py::test_smem_budget_decline_still_fires
        test_staging_on_persistent.py::test_smem_budget_decline_still_fires_on_persistent
    while ``_notes_codereview2/04_OVERNIGHT_TASKS.md`` §"TASK 1 step 4" specifies the raise:
    *"the user wrote the key and it cannot be emitted -> raise"*.  Both cannot hold.

    ⇒ the conflict is a QUESTION FOR THE HUMAN, not something to resolve by quietly
    picking one and letting the other go red.  With this flag those five tests pass
    unmodified, so the reviewer can run BOTH specifications on ONE tree and decide which is
    the intended contract -- rather than being handed a tree where the evidence for the
    losing side has been deleted.  ⚠ It is NOT a compatibility shim and should not survive
    that decision: whichever way it goes, one of the two behaviours becomes dead and this
    branch should be removed with it.
    """
    import os

    from ..compile_environment import CompileEnvironment

    if os.environ.get("HELION_CUTE_ROW_RESIDENCY_HINT") == "1":
        return False
    config = getattr(getattr(fn, "config", None), "config", None)
    if not isinstance(config, dict):
        return False
    env = CompileEnvironment.current()
    return env.config_spec.cute_row_residency_is_explicit(config)


def _cute_tv_stage_slice(
    state: CodegenState,
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
    plan: ChunkTVPlan,
    tensor: torch.Tensor,
    tensor_name: str,
    emit_chunk_stmt: object,  # Callable[[str], None]
    frag_var: str,
) -> tuple[str, bool] | None:
    """``reload_from="smem"`` staging for ``tensor``, or None when it does not apply.

    Returns ``(lane_slice_into_sX, is_first_read)``.  ``is_first_read`` is True
    for the sweep that must WRITE the tile and False for every later sweep, which
    READS it instead of touching GMEM.  quack's structure exactly
    (``rmsnorm.py:233`` stages, ``:334`` re-reads).

    Returns None -- i.e. leaves the GMEM read alone -- for five cases:

    * the knob is off, or the strategy declined it for budget/shape reasons
      (``LoopedReductionStrategy._cute_reload_from_config``);
    * a rank-1 M-broadcast tensor (``weight[:]``).  Staging it would be pure
      loss: it is ``N`` elements shared by every row, so L2 already serves the
      second read, and it is not what quack stages either (quack stages ``mX``
      and ``mRes``, never ``mW``);
    * the tensor is read only ONCE in the whole kernel, so there is no second
      read to make cheaper.  Detected structurally: the write is emitted lazily
      on the first read and the tile is only *consulted* on a later one, so a
      single-read tensor emits a write nobody reads -- which DCE cannot remove
      because SMEM stores are side-effecting.  Hence the explicit check.
    * ⛔ ANOTHER TENSOR ALREADY OWNS THIS REDUCTION'S TILE.  The tile has a row mode
      and a chunk mode and no TENSOR mode, so a second tensor would alias the first
      at identical coordinates and clobber it -- MEASURED on the looped path in an
      unmodified tree.  See the long comment at the gate for the emitted evidence.
    * a ragged row, where the staged tile would need ``predicate_k``/``fill_oob``
      to avoid publishing garbage.  Unreachable today (the plan declines a ragged
      N outright) but checked so it stays unreachable.

    ⭐ AND IT IS A DECLINE, NOT AN ASSERT, FOR A STRATEGY WITHOUT STAGING.  This
    used to be ``assert isinstance(strategy, LoopedReductionStrategy)``, then
    ``if not isinstance(strategy, ReductionStrategy): return None``.  The question is
    whether SMEM staging exists, and it is now asked as a CAPABILITY: the nine staging
    methods live on ``TileStrategy`` (with ``cute_stage_block_id()`` defaulting to
    ``None``), so every strategy can be asked and a strategy that cannot stage takes the
    ordinary "leave the GMEM read alone" path.

    ⭐ THE CLASS TEST WAS THE LAST GATE ON CAPABILITY ③, and removing it is the same edit
    that was already made twice one layer out: ``cute_tv_capable`` moved to
    ``TileStrategy`` (see its docstring there) for exactly this reason, and
    ``_cute_tv_partition_hoist``'s precondition became the capability rather than the
    class.  Leaving this one in place meant ③ was declined BY CLASS for every non-reduction
    strategy however well its geometry worked out -- which is what made the capability
    vacuous on ``CuteNDTileStrategy`` even though that class owns the lane body, the
    constexpr-V loop and the chunk coordinate the protocol needs.
    """
    # ⚠ ``cute_tv_capable()`` IS THE RIGHT PREDICATE AND ``cute_stage_block_id()`` IS NOT
    # ENOUGH ON ITS OWN.  Everything below dereferences the TV emission scaffolding (the
    # plan, the shared slice, the chunk coordinate, the lane var), which is precisely the
    # six-field conjunction ``cute_tv_capable`` answers -- so a strategy that owns a
    # residency slot but has no live plan must decline HERE rather than crash further in.
    # ``cute_tv_capable`` is declared on ``TileStrategy`` with a ``False`` default, so this
    # is a question every strategy can answer.
    if not strategy.cute_tv_capable():  # pyrefly: ignore [missing-attribute]
        return None
    # ⭐ Every decline here is logged with its geometry, and the EFFECTIVE residency is
    # recorded on the emitted artifact by ``_cute_tv_record_residency`` below -- the
    # request half was previously silent even though the ``Config`` object still read
    # ``['smem']``, so a ``frozen_configs.json`` entry recorded a residency the kernel
    # never used.  ⚠ A decline stays a decline; making one a raise broke all 8
    # attention examples.
    if strategy._cute_tv_reload_from != "smem":
        # ⚠ NOT A DECLINE REASON when smem was never asked for.  This branch fires for
        # every ``registers``/``gmem`` config too, where "reload_from is not 'smem'" is
        # a restatement of the request rather than a diagnosis of a refusal -- and
        # recording it as a cause made the canonical marker blame the wrong thing (a
        # register-BUDGET decline read "declined: reload_from is not 'smem'").  The
        # genuine smem refusals all have their own reason strings below and in
        # ``cute_stage_feasible``; a request that was simply not for smem has none.
        _cute_tv_record_residency(
            state,
            strategy,
            tensor_name,
            strategy._cute_row_residency_requested,
            "not requested",
        )
        return None
    if tensor.ndim != 2:
        # Rank-1 broadcast weights: see the docstring.
        _cute_tv_record_residency(
            state,
            strategy,
            tensor_name,
            "gmem",
            f"rank-{tensor.ndim} M-broadcast tensor; L2 already serves the second read",
        )
        return None
    if tensor_name not in _cute_tv_multi_read_tensors(state, strategy):
        _cute_tv_record_residency(
            state,
            strategy,
            tensor_name,
            "gmem",
            "read only once along the reduction axis, so there is no second read to "
            "make cheaper",
        )
        return None
    # ⛔⛔ ONE STAGED TENSOR PER REDUCTION, AND THIS IS A SOUNDNESS GATE, NOT A POLICY.
    #
    # The staged tile is minted ONCE PER STRATEGY (below, keyed on
    # ``_cute_tv_stage_smem_var``) and sized ``rows_per_cta * N`` by
    # ``ChunkTVPlan.stage_smem_elems`` -- which has a ROW mode and a CHUNK mode and **NO
    # TENSOR MODE**.  ``_cute_tv_staged_tensors`` is a set of NAMES, though, so without
    # this gate an arbitrary number of tensors is admitted into one buffer and every one
    # of them partitions ``local_tile(_tv_smem, (1, chunk), (row, chunk_idx))`` at the
    # SAME coordinates.  The second tensor's publish then overwrites the first's and both
    # staged reads return the second -- a silent wrong answer.
    #
    # ⭐ MEASURED ON THE **LOOPED** PATH, in an unmodified tree, so this is a
    # pre-existing bug and not a consequence of widening ③.  A rolled kernel reducing
    # two multi-read row tensors (``x[tile_m, :]`` and ``y[tile_m, :]``) at
    # ``cute_reduction_reload=["smem"]`` emits ONE ``alloc_smem(BFloat16, 2048)`` and
    # then::
    #
    #     _tv_stile_0 = cute.local_tile(_tv_smem, (1, 512), (thread_idx()[1], _tv_chunk_1))
    #     _tv_spart_0 = _tv_thr.partition_D(_tv_stile_0)          # x's writer
    #     _tv_stile_1 = cute.local_tile(_tv_smem, (1, 512), (thread_idx()[1], _tv_chunk_1))
    #     _tv_spart_1 = _tv_thr.partition_D(_tv_stile_1)          # y's writer, SAME SLOT
    #     cute.autovec_copy(_tv_frag_0, _tv_spart_0[None, 0, reduction_lane_1])
    #     cute.autovec_copy(_tv_frag_1, _tv_spart_1[None, 0, reduction_lane_1])  # clobbers
    #
    # ⚠ WHY NO TEST CAUGHT IT: every cell that stages today stages exactly ONE 2-D
    # tensor.  ``rms_norm`` / ``layer_norm`` / ``cross_entropy`` stage ``x`` and nothing
    # else, because ``weight`` is declined as rank-1 by the check above.  The bug needs
    # TWO multi-read 2-D tensors on one reduction axis to become reachable, which no
    # frozen cell has -- so it is invisible to the 40-cell table by construction rather
    # than by luck.
    #
    # ⇒ DECLINING is the fail-closed answer and it is also what quack does (it stages
    # ``mX`` alone).  The refused tensor falls back to a gmem re-read, which is exactly
    # the behaviour it has today on every kernel where staging is not requested.  The
    # alternative -- a third tile mode indexed by tensor, plus a proportionally larger
    # buffer and budget charge -- is a real capability extension and belongs in its own
    # change, not smuggled in behind a correctness fix.
    #
    # ⚠ ASKED OF ``_cute_tv_staged_tensors`` AND NOT OF A COUNTER, because that set is
    # the SAME state the ``first_read`` discriminator below reads.  A separate counter
    # could disagree with it; this cannot.  It also makes the gate stable across sweeps:
    # the set is deliberately NOT reset per chunk body (unlike
    # ``_cute_tv_stage_partitions``), so a tensor already staged in sweep 1 still reads
    # as "the owner" in sweep 2 and takes the staged-read path rather than being refused.
    if (
        strategy._cute_tv_staged_tensors
        and tensor_name not in strategy._cute_tv_staged_tensors
    ):
        owner = ", ".join(sorted(strategy._cute_tv_staged_tensors))
        log.debug(
            "cute staging DECLINE block=%s tensor=%s: %r already owns this reduction's "
            "staged tile, which has no tensor mode (rows_per_cta x N), so a second "
            "tensor would alias it at identical (row, chunk) coordinates",
            strategy.block_index,
            tensor_name,
            owner,
        )
        _cute_tv_record_residency(
            state,
            strategy,
            tensor_name,
            "gmem",
            f"another tensor ({owner}) already owns this reduction's staged tile; the "
            f"tile has no tensor mode, so sharing it would alias",
        )
        return None
    feasible = strategy.cute_stage_feasible()
    if feasible is None:
        # cute_stage_feasible logged the specific reason and its geometry.
        _cute_tv_record_residency(
            state,
            strategy,
            tensor_name,
            "gmem",
            "cute_stage_feasible declined (budget or geometry; see the DEBUG log)",
        )
        return None
    rows_per_cta, num_chunks = feasible
    _cute_tv_record_residency(
        state,
        strategy,
        tensor_name,
        "smem",
        f"staged: rows_per_cta={rows_per_cta} num_chunks={num_chunks} "
        f"chunk={plan.chunk} lane_extent={plan.lane_extent}",
    )
    fn = state.device_function
    emit = cast("object", emit_chunk_stmt)
    assert callable(emit)

    # -- the ONE staging buffer, hoisted to the kernel preamble --------------
    smem_var = strategy._cute_tv_stage_smem_var
    if smem_var is None:
        ptr_var = fn.new_var("_tv_smem_ptr", dce=False)
        smem_var = fn.new_var("_tv_smem", dce=False)
        fn.preamble.append(
            statement_from_string(
                plan.emit_stage_smem_alloc(ptr_var, rows_per_cta, num_chunks)
            )
        )
        fn.preamble.append(
            statement_from_string(
                plan.emit_stage_smem_tensor(smem_var, ptr_var, rows_per_cta, num_chunks)
            )
        )
        strategy._cute_tv_stage_smem_var = smem_var

    # -- one staging partition per (tensor, chunk body) ----------------------
    first_read = tensor_name not in strategy._cute_tv_staged_tensors
    key = (tensor_name, "W" if first_read else "R")
    cached = strategy._cute_tv_stage_partitions.get(key)
    if cached is None:
        slot = len(strategy._cute_tv_stage_partitions)
        stile_var = fn.new_var(f"_tv_stile_{slot}", dce=False)
        spart_var = fn.new_var(f"_tv_spart_{slot}", dce=False)
        # The staging tile's row is the CTA-LOCAL row, i.e. this reduction's
        # sibling thread axis -- NOT the clamped global row.  The buffer is
        # per-CTA, so a global row index would be out of range.
        row_axis = strategy._cute_stage_row_axis_expr()
        # SAME chunk coordinate as the gmem ``local_tile``: sweep 2's chunk c must
        # read what sweep 1 wrote at chunk c.  Using 0 here instead makes every
        # chunk alias the last one -- MEASURED relerr 261.6.
        chunk_idx_var = cast("str", strategy._cute_tv_chunk_index_var)
        emit(plan.emit_stage_local_tile(stile_var, smem_var, row_axis, chunk_idx_var))
        # ``partition_D`` for the writer and ``partition_S`` for the reader, both
        # off THE shared slice, so the two land on identical elements -- the same
        # property that makes the gmem legs agree.
        thr_var = cast("tuple[str, str]", strategy._cute_tv_shared)[1]
        emit(
            plan.emit_partition_dest(spart_var, thr_var, stile_var)
            if first_read
            else plan.emit_partition_source(spart_var, thr_var, stile_var)
        )
        strategy._cute_tv_stage_partitions[key] = spart_var
        cached = spart_var
    if first_read:
        strategy._cute_tv_staged_tensors.add(tensor_name)
    lane_var = strategy._cute_reduction_lane_var
    assert isinstance(lane_var, str)
    # ⭐ CAPABILITY ③ ON A LOOP-FREE STRATEGY: EMIT THE READER PARTITION **NOW**, even
    # though no second read has been lowered and none ever will be.
    #
    # ⚠ THIS IS NOT SPECULATION, IT IS THE ONLY MOMENT THE READER CAN BE DECLARED.  A
    # loop-free strategy lowers the row's load ONCE (measured: this function is called
    # a single time, ``first_read=True``); the second sweep is CLONED later by
    # ``tile_strategy._split_lane_loop_over_constexpr_vec``, long after every
    # ``emit_chunk_stmt`` target has been sealed and after the ``local_tile`` /
    # ``partition_*`` declarations the reader needs are already in the chunk prefix.
    # A reader partition minted by that pass would have nowhere legal to go, so it is
    # declared here, next to the writer, off the SAME tile and the SAME slice -- which
    # is also what makes writer and reader address identical elements.
    #
    # The pass then rewrites the CLONED gmem copy to read this partition; see
    # ``_tv_restage_cloned_loads``.  If the pass declines (an unsupported nest shape),
    # nothing reads ``_tv_spart_R`` and ``dead_assignment_elimination`` removes it, so
    # the failure direction is "staging did not engage", never a wrong answer.
    if first_read and strategy.cute_stage_restages_cloned_sweeps():
        read_key = (tensor_name, "R")
        if read_key not in strategy._cute_tv_stage_partitions:
            slot = len(strategy._cute_tv_stage_partitions)
            # ⭐ ``dce=True`` ON BOTH, UNLIKE EVERY OTHER TV DECLARATION.  These two are
            # the only staging statements emitted SPECULATIVELY: they are declared at the
            # single load site for a reader the split pass has not created yet, so if
            # that pass declines (an unsupported nest shape) nothing will reference them.
            # Marking them DCE-eligible is what makes the decline path emit today's
            # unstaged shape instead of leaving two dangling declarations in the kernel.
            #
            # ⚠ Safe because they are PURE: ``local_tile`` and ``partition_S`` compute
            # addresses and have no side effects, so dropping them when unread changes
            # nothing.  The PUBLISH is deliberately NOT dce-eligible -- an SMEM store IS
            # a side effect and ``dead_assignment_elimination`` must not remove it.
            r_stile = fn.new_var(f"_tv_stile_{slot}", dce=True)
            r_spart = fn.new_var(f"_tv_spart_{slot}", dce=True)
            emit(
                plan.emit_stage_local_tile(
                    r_stile,
                    smem_var,
                    strategy._cute_stage_row_axis_expr(),
                    cast("str", strategy._cute_tv_stage_chunk_index_var),
                )
            )
            emit(
                plan.emit_partition_source(
                    r_spart,
                    cast("tuple[str, str]", strategy._cute_tv_shared)[1],
                    r_stile,
                )
            )
            strategy._cute_tv_stage_partitions[read_key] = r_spart
        # Record the rewrite the split pass must perform: this tensor's gmem load
        # fragment, re-read from THIS partition.  Keyed on the FRAGMENT rather than on
        # the tensor because that is what the cloned copy statement names, so the pass
        # matches on a symbol it can actually see rather than re-deriving eligibility.
        strategy._cute_tv_stage_read_by_frag[frag_var] = (
            strategy._cute_tv_stage_partitions[read_key]
        )
    return (plan.emit_lane_slice(cached, lane_var), first_read)


def _cute_tv_multi_read_tensors(
    state: CodegenState,
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
) -> frozenset[str]:
    """Names of tensors this kernel reads along the reduction axis MORE THAN ONCE.

    Staging a single-read tensor is a pure loss (an SMEM write nobody reads), so
    the decision has to be made from the device IR rather than from emission
    order -- at the first load site nothing local distinguishes "there will be a
    second read" from "this is the only one".

    Cached on the strategy: the walk is over the whole device IR and every load
    site would otherwise repeat it.

    ⭐ THE AXIS IS RECOGNISED BY *IDENTITY*, NOT BY SYNTACTIC FORM, and that is a fix
    rather than a generalisation.  This walk used to require a bare ``slice(None)``
    somewhere in the subscript -- the form a ROLLED reduction's axis takes -- which
    silently excluded a **TILED** axis (``x[tile_m, tile_n]``, whose entries are ``SymInt``s,
    and in the device IR ``fx.Node``s wrapping them).  MEASURED on a two-sweep tile kernel:
    the walk returned ``frozenset()`` for every tensor, so the gate that consumes it
    ("read only once, no second read to make cheaper") refused EVERY tensor and capability
    ③ could not engage on that path however well the rest of it worked.

    ⚠ THIS IS THE SAME DEFECT ``_cute_tv_site_eligible`` ALREADY FIXED ONE LAYER OUT, in
    this file -- see its docstring: *"This used to require the trailing subscript to be a
    literal ``slice(None)`` … and it silently excluded a tiled axis … even when that axis
    carried a complete plan."*  Reusing :func:`_cute_tv_indexes_copy_axis` here is what
    makes the two unable to drift: both now ask the SAME question of the SAME helper
    (``_tiled_axis_block_id`` vs ``cute_tv_lane_block_id()``), so a plan cannot be built at
    a width one gate admits and the other refuses.
    """
    from ...language import memory_ops as _memory_ops
    from ..host_function import HostFunction

    cached = strategy._cute_tv_multi_read_cache  # pyrefly: ignore [missing-attribute]
    if cached is not None:
        return cast("frozenset[str]", cached)
    counts: dict[str, int] = {}
    hf = HostFunction.current()
    if hf._device_ir is not None:
        for graph_info in hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if node.target is not _memory_ops.load:
                    continue
                if len(node.args) < 2:
                    continue
                tensor_node, subscript = node.args[0], node.args[1]
                if not isinstance(subscript, (list, tuple)):
                    continue
                # ⚠ ANY entry may name the axis, not only the trailing one: this walk is
                # counting READS OF A ROW and does not care where in the subscript the axis
                # sits.  (The trailing-dim requirement is the SITE gate's job -- it is a
                # fact about what the TV layout can address, not about how many times the
                # row is read.)  Keeping the two separate is why widening this one cannot
                # admit an access the emission then refuses.
                if not any(
                    _cute_tv_indexes_copy_axis(strategy, idx) for idx in subscript
                ):
                    continue
                val = (
                    tensor_node.meta.get("val")
                    if isinstance(tensor_node, torch.fx.Node)
                    else tensor_node
                )
                if not isinstance(val, torch.Tensor) or val.ndim != 2:
                    continue
                name = state.device_function.tensor_arg(val).name
                counts[name] = counts.get(name, 0) + 1
    result = frozenset(name for name, n in counts.items() if n > 1)
    strategy._cute_tv_multi_read_cache = result  # pyrefly: ignore [missing-attribute]
    return result


def _cute_tv_stored_tensors(
    state: CodegenState,
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
) -> frozenset[str]:
    """Names of tensors this kernel WRITES along the reduction axis.

    ⭐⭐ THIS IS THE STORE-ALIAS PROOF THE ``registers`` RESIDENCY OWES, ANSWERED FROM THE
    DEVICE IR INSTEAD OF FROM THE EMITTED AST.

    ⛔ THE HAZARD IT CLOSES.  Serving a tensor's SECOND read from a register cache is only
    sound if nothing WROTE that tensor in between: with an intervening store the cache holds
    a pre-write value and the kernel silently computes with stale data.  The register-fusing
    AST pass answers this by scanning the emitted body for TV store copies
    (``fuse_tv_copy_sweeps``' ``stored_bases``) and refusing those tensors outright.  That
    answer is only available AFTER the whole body exists, which is exactly why the decision
    lives in a post-pass -- and moving the decision to the lowering site means the proof has
    to be available there too.

    ⚠ IT IS A WHOLE-KERNEL QUESTION ASKED AT A LOCAL SITE, which is the same shape of problem
    :func:`_cute_tv_multi_read_tensors` already solves for loads: at the first read site
    nothing local distinguishes "a later statement will write this tensor" from "nothing
    will".  So this is deliberately its mirror image -- same graph walk, same axis-identity
    predicate, same per-strategy cache -- and the pair should be read together.

    ⚠⚠ BUT IT IS **NOT** MERELY THAT WALK WITH ``load`` -> ``store``, and the difference is
    load-bearing rather than pedantic:

    * ``store``'s value is its THIRD arg, so a rank check on ``args[0]``'s val is the right
      test but the multiplicity is not -- ONE write is already disqualifying, where one read
      is not.  ⇒ this returns a set of "was written at all", not "written more than once".
      A count here would admit exactly the aliasing case it exists to refuse.
    * ``_ATOMIC_OPS`` write too.  An atomic is a write for this purpose even though it is not
      ``store``, so they are included -- omitting them would leave a cache serving stale data
      across an ``atomic_add`` to the same row.

    ⇒ so the plan's "the store version is the same walk with load -> store" is right about the
    SHAPE and wrong about the PREDICATE; the difference is written out above rather than left
    for the next reader to rediscover.
    """
    from ...language import _MEMORY_OPS
    from ...language import memory_ops as _memory_ops
    from ..host_function import HostFunction

    cached = strategy._cute_tv_stored_cache  # pyrefly: ignore [missing-attribute]
    if cached is not None:
        return cast("frozenset[str]", cached)
    # Every op that WRITES memory: the plain store plus the atomics.  ``_MEMORY_OPS``
    # includes ``load``, so it cannot be used as the write set directly.
    write_targets = {op for op in _MEMORY_OPS if op is not _memory_ops.load}
    names: set[str] = set()
    hf = HostFunction.current()
    if hf._device_ir is not None:
        for graph_info in hf.device_ir.graphs:
            for node in graph_info.graph.nodes:
                if node.target not in write_targets:
                    continue
                if len(node.args) < 2:
                    continue
                tensor_node, subscript = node.args[0], node.args[1]
                if not isinstance(subscript, (list, tuple)):
                    continue
                # ANY entry may name the axis -- same reasoning as the read walk: this is a
                # question about whether the ROW was written, not about where in the
                # subscript the axis sits.
                if not any(
                    _cute_tv_indexes_copy_axis(strategy, idx) for idx in subscript
                ):
                    continue
                val = (
                    tensor_node.meta.get("val")
                    if isinstance(tensor_node, torch.fx.Node)
                    else tensor_node
                )
                if not isinstance(val, torch.Tensor) or val.ndim != 2:
                    continue
                names.add(state.device_function.tensor_arg(val).name)
    result = frozenset(names)
    strategy._cute_tv_stored_cache = result  # pyrefly: ignore [missing-attribute]
    return result


def _cute_register_unroll_vec_hoist(
    state: CodegenState,
    strategy: object,  # a ReductionStrategy with cute_tv_capable() at runtime
    lane_block_id: int,
    tensor: torch.Tensor,
    tensor_name: str,
    index_exprs: list[str],
    vec_width: int,
    subscript: list[object] | tuple[object, ...],
) -> str:
    """Register a Uint16 vec load to be hoisted above the constexpr V-loop
    in the active lane body and return the per-element extract expression.

    The hoist runs once per outer-lane iter; the constexpr V-loop's body
    receives ``hoist_var[vi].bitcast(dtype)`` (a scalar) so the existing
    cast/mul/accumulate pipeline keeps working unchanged.

    The address is main's: ``_cute_scalar_pointer_expr`` with the lane axis swapped for
    the per-thread V-aligned base, wrapped in main's own lane-axis anchor guard.

    ⛔ THE BRANCH'S ROW-AXIS CLAMP IS GONE (T6, see the comment at the pointer below).  Two
    GPU-measured over-reads motivated it: the software pipeliner re-emits this load one
    chunk ahead, so an unguarded LANE index reads past the row (main's inline guard covers
    that), and an unguarded ROW index reads past the tensor on a tail CTA (that was the
    branch's addition).  ⚠ The second is a real hazard of the CLASSIC VEC path, and it is
    retired by the path no longer being reached rather than by being guarded: measured, all
    40 frozen cells emit a TV copy and ZERO reach this emitter.
    """
    elem_dtype = _CUTE_VECTOR_UNROLL_DTYPES[tensor.dtype]
    base_index_var = strategy._cute_lane_base_index_var  # pyrefly: ignore [missing-attribute]
    lane_body = strategy._cute_lane_body  # pyrefly: ignore [missing-attribute]
    assert isinstance(base_index_var, str)
    assert isinstance(lane_body, list)
    # The inner reduction-axis index_expr is the last entry; swap it with
    # the per-lane base so the vec load points at the start of the V-wide
    # chunk this thread owns.
    # ⛔ THE BRANCH'S OOB-GUARD LAYER IS GONE FROM HERE (T6).  This is main's pointer,
    # verbatim: swap the reduction axis's index_expr (the last entry) for the per-thread
    # V-aligned base and build the address with ``_cute_scalar_pointer_expr``.
    #
    # ⭐ WHY REVERTING IS RIGHT RATHER THAN A REGRESSION.  The branch replaced this with
    # ``_cute_vec_load_desc`` + ``_cute_guarded_pointer_expr``, a ~260-line descriptor that
    # clamps every axis of a vec address.  It reads as correctness and is not: it guards a
    # **vec** load, i.e. correctness CONDITIONAL ON AN OPTIMISATION.  The scalar floor it
    # degrades to has no address to guard, and the TV path carries its own bounds argument.
    # ⇒ the layer only ever protected the classic vec path, which is what T6 removes.
    #
    # ⚠ MEASURED before deleting it: all 40 frozen cells now emit ``make_tiled_copy_tv``
    # and ZERO reach this hoist.
    base_exprs = list(index_exprs)
    base_exprs[-1] = base_index_var
    base_ptr_expr = _cute_scalar_pointer_expr(tensor_name, base_exprs)
    cache_key = (tensor_name, base_ptr_expr)
    cache = getattr(strategy, "_cute_lane_vec_loads", None)
    if cache is None:
        cache = {}
        # pyrefly: ignore [missing-attribute]
        strategy._cute_lane_vec_loads = cache
    if cache_key not in cache:
        hoist_var = state.device_function.new_var(
            f"_unroll_vec_{len(cache)}", dce=False
        )
        cache[cache_key] = (hoist_var, tensor.dtype)
        hoist_stmt = statement_from_string(
            f"{hoist_var} = cute.arch.load({base_ptr_expr}, "
            f"ir.VectorType.get([{vec_width}], cutlass.Uint16.mlir_type))"
        )
        # Insert the hoist just BEFORE the constexpr V-loop (the last entry
        # in lane_body).  ``lane_body[-1]`` is the constexpr loop.
        lane_body.insert(len(lane_body) - 1, hoist_stmt)
    else:
        hoist_var, _ = cache[cache_key]
    # The constexpr V-loop's target var is the last element's loop var.
    constexpr_loop = lane_body[-1]
    assert isinstance(constexpr_loop, ast.For)
    assert isinstance(constexpr_loop.target, ast.Name)
    vec_lane_var = constexpr_loop.target.id
    return f"cutlass.Uint16({hoist_var}[{vec_lane_var}]).bitcast({elem_dtype})"


def _cute_stack_tensor_offset_expr(
    state: CodegenState,
    tensor_like: torch.Tensor,
    subscript: list[object],
    ast_subscript: list[object] | tuple[object, ...],
) -> str:
    env = CompileEnvironment.current()
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor_like,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if "None" in index_exprs:
        raise exc.BackendUnsupported("cute", "inactive stack tensor load dimension")
    index_dtype = env.index_type()
    terms = []
    for dim, index in enumerate(index_exprs):
        stride = tensor_like.stride(dim)
        stride_expr = (
            str(stride) if isinstance(stride, int) else state.sympy_expr(stride)
        )
        terms.append(f"({index_dtype}({index}) * {index_dtype}({stride_expr}))")
    return " + ".join(terms) if terms else "0"


def _cute_stack_tensor_mask_expr(
    state: CodegenState,
    tensor_like: torch.Tensor,
    dev_ptrs: torch.Tensor,
    subscript: list[object],
    extra_mask: ast.AST | None,
) -> str | None:
    terms = []
    tensor_mask = _cute_combined_mask(
        state,
        subscript,
        extra_mask,
        tensor=tensor_like,
        include_tensor_index_masks=False,
    )
    if tensor_mask is not None:
        terms.append(tensor_mask)
    stack_mask = _cute_combined_mask(
        state,
        [slice(None)] * dev_ptrs.ndim,
        None,
        tensor=dev_ptrs,
    )
    if stack_mask is not None and stack_mask not in terms:
        terms.append(stack_mask)
    if not terms:
        return None
    return " and ".join(f"({term})" for term in terms)


def _cute_stack_tensor_pointer_expr(
    target_dtype: str,
    dev_ptrs_ast: ast.AST,
    offset_expr: str,
) -> ast.AST:
    return expr_from_string(
        f"(cute.make_ptr({target_dtype}, cutlass.Int64({{base}}), "
        f"cute.AddressSpace.gmem) + ({offset_expr}))",
        base=dev_ptrs_ast,
    )


def _codegen_cute_store_stack_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: tuple[object, ...] | list[object],
    ast_subscript: tuple[object, ...] | list[object],
    value: ast.AST,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    if value_node.op != "call_function" or value_node.target is not load:
        return None
    stack_arg = value_node.args[0]
    if not isinstance(stack_arg, tuple) or len(stack_arg) != 2:
        return None
    ptr_node = stack_arg[1]
    if (
        not isinstance(ptr_node, torch.fx.Node)
        or ptr_node.op != "call_function"
        or ptr_node.target is not load
        or len(ptr_node.args) < 2
    ):
        return None
    dev_ptrs = (
        ptr_node.args[0].meta.get("val")
        if isinstance(ptr_node.args[0], torch.fx.Node)
        else None
    )
    ptr_subscript = ptr_node.args[1]
    if not isinstance(dev_ptrs, torch.Tensor) or not isinstance(
        ptr_subscript, (list, tuple)
    ):
        return None
    tensor_like_node = stack_arg[0]
    tensor_like = (
        tensor_like_node.meta.get("val")
        if isinstance(tensor_like_node, torch.fx.Node)
        else tensor_like_node
    )
    if not isinstance(tensor_like, torch.Tensor):
        return None

    if (
        dev_ptrs.ndim == 2
        and len(ptr_subscript) == 2
        and all(isinstance(idx, slice) and idx == slice(None) for idx in ptr_subscript)
        and len(subscript) >= 3
        and isinstance(subscript[0], slice)
        and subscript[0] == slice(None)
        and isinstance(subscript[1], slice)
        and subscript[1] == slice(None)
    ):
        stack_value_subscript = value_node.args[1]
        if not isinstance(stack_value_subscript, (list, tuple)):
            return None
        stack_value_subscript_proxy = map_arg(
            stack_value_subscript, lambda arg: arg.meta["val"]
        )
        stack_value_subscript_ast = map_arg(
            stack_value_subscript, lambda arg: state.env[arg]
        )
        tensor_offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*stack_value_subscript_proxy],
            [*stack_value_subscript_ast],
        )
        target_index_exprs = _cute_index_exprs(
            state,
            [*subscript],
            ast_subscript,
            tensor=tensor,
            inactive_singleton_slice_expr="0",
        )
        if len(target_index_exprs) != tensor.ndim:
            return None
        first_stack_index = target_index_exprs[0]
        target_tail = target_index_exprs[2:]
        loop_var = state.device_function.new_var("stack_dim", dce=True)
        env = CompileEnvironment.current()
        index_dtype = env.index_type()
        dev_ptrs_name = state.device_function.tensor_arg(dev_ptrs).name
        tensor_name = state.device_function.tensor_arg(tensor).name
        target_dtype = env.backend.dtype_str(tensor.dtype)
        dev_ptr_offset = (
            f"{index_dtype}({first_stack_index}) * "
            f"{index_dtype}({dev_ptrs.stride(0)}) + "
            f"{index_dtype}({loop_var}) * {index_dtype}({dev_ptrs.stride(1)})"
        )
        stack_ptr_expr = (
            f"(cute.make_ptr({target_dtype}, "
            f"cutlass.Int64(({dev_ptrs_name}.iterator + {dev_ptr_offset}).load()), "
            f"cute.AddressSpace.gmem) + ({tensor_offset_expr}))"
        )
        target_indices = [first_stack_index, loop_var, *target_tail]
        store_expr = _cute_scalar_store_expr(
            tensor_name,
            target_indices,
            f"({stack_ptr_expr}).load()",
        )
        mask_expr = _cute_combined_mask(state, [*subscript], extra_mask, tensor=tensor)
        if mask_expr is None:
            body = f"    {store_expr}"
        else:
            body = f"    if {mask_expr}:\n        {store_expr}"
        state.add_statement(
            statement_from_string(
                f"for {loop_var} in range({dev_ptrs.size(1)}):\n{body}"
            )
        )
        return ast.Constant(value=None)

    ptr_subscript_proxy = map_arg(ptr_subscript, lambda arg: arg.meta["val"])
    ptr_subscript_ast = map_arg(ptr_subscript, lambda arg: state.env[arg])
    ptr_index_exprs = _cute_index_exprs(
        state,
        [*ptr_subscript_proxy],
        [*ptr_subscript_ast],
        tensor=dev_ptrs,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if "None" in ptr_index_exprs:
        return None

    target_index_exprs = _cute_index_exprs(
        state,
        [*subscript],
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    ptr_pos = 0
    rewritten_index_exprs = []
    for idx, index_expr in zip(subscript, target_index_exprs, strict=True):
        if isinstance(idx, slice) and idx == slice(None):
            replacement = (
                ptr_index_exprs[ptr_pos] if ptr_pos < len(ptr_index_exprs) else None
            )
            ptr_pos += 1
            rewritten_index_exprs.append(
                replacement if replacement is not None else index_expr
            )
        else:
            if ptr_pos < len(ptr_subscript_proxy) and not (
                isinstance(ptr_subscript_proxy[ptr_pos], slice)
                and ptr_subscript_proxy[ptr_pos] == slice(None)
            ):
                ptr_pos += 1
            rewritten_index_exprs.append(index_expr)

    tensor_name = state.device_function.tensor_arg(tensor).name
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(tensor.dtype)
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    store_expr = expr_from_string(
        _cute_scalar_store_expr(tensor_name, rewritten_index_exprs, "{value}"),
        value=value,
    )
    mask_expr = _cute_combined_mask(state, [*subscript], extra_mask, tensor=tensor)
    if mask_expr is None:
        return store_expr
    mask_ast = expr_from_string(mask_expr)
    assert isinstance(mask_ast, ast.expr)
    assert isinstance(store_expr, ast.expr)
    state.add_statement(
        ast.fix_missing_locations(
            ast.If(
                test=mask_ast,
                body=[ast.Expr(value=store_expr)],
                orelse=[],
            )
        )
    )
    return ast.Constant(value=None)


def _cute_affine_range_block_id(state: CodegenState, affine: object) -> int | None:
    from .indexing import CuteAffineRangeIndex

    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    env = CompileEnvironment.current()
    base_meta = getattr(affine.base, "meta", {})
    base_val = base_meta.get("val") if isinstance(base_meta, dict) else None
    block_id = env.resolve_block_id(base_val) if base_val is not None else None
    if block_id is None:
        codegen = base_meta.get("codegen") if isinstance(base_meta, dict) else None
        if isinstance(codegen, ast.Name) and codegen.id.startswith("_BLOCK_SIZE_"):
            with contextlib.suppress(ValueError):
                block_id = int(codegen.id.removeprefix("_BLOCK_SIZE_"))
    if block_id is None:
        return None
    if state.fx_node is not None:
        return env.resolve_codegen_block_id(
            block_id, state.codegen, state.fx_node.graph
        )
    return block_id


def _cute_affine_range_expr(
    state: CodegenState,
    affine: object,
    lane_var: str,
    *,
    dtype: torch.dtype | None = None,
) -> str | None:
    from .indexing import CuteAffineRangeIndex

    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    if affine.step != 1 or affine.factor <= 0:
        return None
    block_id = _cute_affine_range_block_id(state, affine)
    if block_id is None:
        return None
    index_var = _cute_active_index_var(state, block_id)
    if index_var is None:
        return None
    expr = f"({affine.factor}) * ({index_var}) + cutlass.Int32({lane_var})"
    if dtype is not None:
        expr = f"{CompileEnvironment.current().backend.dtype_str(dtype)}({expr})"
    return expr


def _codegen_cute_affine_range_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    value: object,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None = None,
) -> ast.AST | None:
    from ..ast_extension import create
    from .indexing import CuteAffineRangeIndex

    affine_positions = [
        (pos, idx)
        for pos, idx in enumerate(ast_subscript)
        if isinstance(idx, CuteAffineRangeIndex)
    ]
    if len(affine_positions) != 1 or len(subscript) != 1 or extra_mask is not None:
        return None
    _pos, affine = affine_positions[0]
    block_id = _cute_affine_range_block_id(state, affine)
    if block_id is None:
        return None

    lane_var = state.device_function.new_var("affine_lane", dce=True)
    index_expr = _cute_affine_range_expr(
        state, affine, lane_var, dtype=CompileEnvironment.current().index_dtype
    )
    if index_expr is None:
        return None
    backend = CompileEnvironment.current().backend
    if (
        value_node is not None
        and value_node.op == "call_function"
        and value_node.target is load
    ):
        source_tensor_node = value_node.args[0]
        if not isinstance(source_tensor_node, torch.fx.Node):
            return None
        source_tensor = source_tensor_node.meta.get("val")
        if not isinstance(source_tensor, torch.Tensor):
            return None
        source_subscript = value_node.args[1]
        if (
            not isinstance(source_subscript, (list, tuple))
            or len(source_subscript) != 1
        ):
            return None
        source_subscript_args = tuple(cast("Any", source_subscript))
        ast_source_subscript = list(
            map_arg(source_subscript_args, lambda arg: state.env[arg])
        )
        (source_affine,) = ast_source_subscript
        if not isinstance(source_affine, CuteAffineRangeIndex):
            return None
        if source_affine.factor != affine.factor:
            return None
        source_index_expr = _cute_affine_range_expr(
            state,
            source_affine,
            lane_var,
            dtype=CompileEnvironment.current().index_dtype,
        )
        if source_index_expr is None:
            return None
        source_name = state.device_function.tensor_arg(source_tensor).name
        value_expr = f"{source_name}[{source_index_expr}]"
        if source_tensor.dtype is torch.bool:
            value_expr = f"({value_expr} != cutlass.Uint8(0))"
    elif isinstance(value, CuteAffineRangeIndex):
        value_expr = _cute_affine_range_expr(state, value, lane_var, dtype=value.dtype)
        if value_expr is None:
            return None
    elif isinstance(value, ast.AST):
        value_expr = ast.unparse(value)
    elif isinstance(value, (int, float, bool)):
        value_expr = repr(value)
    else:
        return None

    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(value_expr, target_dtype)
    tensor_name = state.device_function.tensor_arg(tensor).name
    store_expr = (
        f"{tensor_name}.__setitem__({_cute_index_tuple([index_expr])}, {value_expr})"
    )
    mask_var = _cute_active_mask_var(state, block_id)
    if mask_var is not None:
        store_expr = f"{store_expr} if {mask_var} else None"

    return create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({affine.factor})"),
        body=[create(ast.Expr, value=expr_from_string(store_expr))],
        orelse=[],
        type_comment=None,
    )


def _codegen_cute_affine_reshape_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """Lower a 2-D affine-row store fed by a reshape/stack chain.

    Handles ``out[(begin*K):(begin*K + block*K), tile_n] = reshaped`` where the
    leading index is a ``CuteAffineRangeIndex`` (factor ``K``) over the m-tile,
    the trailing index is the n-tile, and the value is a row-major shape chain
    (e.g. ``stack([a, b], dim=1).reshape(block*K, block_n)``).

    Each m-tile thread owns row ``m_local`` of the source; the reshaped tensor
    has ``K`` rows per source row, so the thread loops ``s in range(K)`` and
    writes the value resolved at flat index ``(K*m_local + s)*block_n + n_local``
    to output row ``K*m_global + s``, column ``n_global``.
    """
    from ..ast_extension import create
    from ..generate_ast import GenerateAST
    from .cute_reshape import _get_block_local_coord
    from .cute_reshape import resolve_cute_shape_chain_value_at
    from .indexing import CuteAffineRangeIndex
    from .indexing import is_cute_shape_chain_target

    if (
        tensor.ndim != 2
        or len(subscript) != 2
        or len(ast_subscript) != 2
        or extra_mask is not None
        or value_node is None
        or not isinstance(state.codegen, GenerateAST)
    ):
        return None
    affine = ast_subscript[0]
    if not isinstance(affine, CuteAffineRangeIndex):
        return None
    if affine.step != 1 or affine.factor <= 0:
        return None
    n_index = subscript[1]
    if not isinstance(n_index, torch.SymInt):
        return None
    env = CompileEnvironment.current()
    block_id_n = env.get_block_id(n_index)
    if block_id_n is None:
        return None
    block_id_m = _cute_affine_range_block_id(state, affine)
    if block_id_m is None:
        return None

    if value_node.op != "call_function" or not is_cute_shape_chain_target(
        value_node.target
    ):
        return None
    value_val = value_node.meta.get("val")
    if not isinstance(value_val, torch.Tensor) or value_val.ndim != 2:
        return None

    m_global = _cute_active_index_var(state, block_id_m)
    n_global = _cute_active_index_var(state, block_id_n)
    if m_global is None or n_global is None:
        return None
    m_local = _get_block_local_coord(state.codegen, block_id_m)
    n_local = _get_block_local_coord(state.codegen, block_id_n)
    if m_local is None or n_local is None:
        return None
    block_n = state.device_function.resolved_block_size(block_id_n)
    if not isinstance(block_n, int):
        return None

    factor = affine.factor
    lane_var = state.device_function.new_var("affine_lane", dce=True)
    row_local = f"cutlass.Int32({factor}) * ({m_local}) + cutlass.Int32({lane_var})"
    flat_index = (
        f"(({row_local}) * cutlass.Int32({block_n})) + ({n_local})"
        if block_n != 1
        else f"({row_local}) + ({n_local})"
    )
    value_ast = resolve_cute_shape_chain_value_at(state, value_node, flat_index)
    if value_ast is None:
        return None

    backend = env.backend
    index_dtype = backend.dtype_str(env.index_dtype)
    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(ast.unparse(value_ast), target_dtype)

    # Bind the resolved (possibly select-based) value to a variable so the CuTe
    # DSL sees the stack `ifexp` as its own assignment rather than nested inside
    # the `.store(...)` call / masked store ternary.
    value_var = state.device_function.new_var("affine_value", dce=True)

    row_index = (
        f"{index_dtype}(cutlass.Int32({factor}) * ({m_global}) "
        f"+ cutlass.Int32({lane_var}))"
    )
    col_index = f"{index_dtype}({n_global})"
    tensor_name = state.device_function.tensor_arg(tensor).name
    store_expr = _cute_scalar_store_expr(tensor_name, [row_index, col_index], value_var)

    store_stmt: ast.stmt = create(ast.Expr, value=expr_from_string(store_expr))
    mask_parts = [
        mask
        for mask in (
            _cute_active_mask_var(state, block_id_m),
            _cute_active_mask_var(state, block_id_n),
        )
        if mask is not None
    ]
    if mask_parts:
        # Use a guard statement (not a ternary) so the CuTe DSL accepts the
        # device-value mask condition.
        mask_ast = expr_from_string(" and ".join(mask_parts))
        assert isinstance(mask_ast, ast.expr)
        store_stmt = ast.fix_missing_locations(
            ast.If(test=mask_ast, body=[store_stmt], orelse=[])
        )

    return create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({factor})"),
        body=[
            statement_from_string(f"{value_var} = {value_expr}"),
            store_stmt,
        ],
        orelse=[],
        type_comment=None,
    )


def _is_cute_affine_range_load_for_store(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
) -> bool:
    from .indexing import CuteAffineRangeIndex
    from .indexing import match_cute_affine_range_iota

    def compatible_store_user(user: torch.fx.Node) -> bool:
        if (
            user.op != "call_function"
            or user.target is not store
            or len(user.args) < 4
            or user.args[2] is not state.fx_node
            or user.args[3] is not None
        ):
            return False
        store_subscript = user.args[1]
        return (
            isinstance(store_subscript, (list, tuple))
            and len(store_subscript) == 1
            and isinstance(store_subscript[0], torch.fx.Node)
            and match_cute_affine_range_iota(store_subscript[0]) is not None
        )

    return (
        state.fx_node is not None
        and len(state.fx_node.users) > 0
        and all(compatible_store_user(user) for user in state.fx_node.users)
        and len(subscript) == 1
        and len(ast_subscript) == 1
        and isinstance(ast_subscript[0], CuteAffineRangeIndex)
    )


def _cute_positive_1d_slice_bounds(
    tensor: torch.Tensor, index: object
) -> tuple[int, int, int, int] | None:
    if not isinstance(index, slice) or index == slice(None):
        return None
    with contextlib.suppress(TypeError):
        dim_size = int(tensor.shape[0])
        start, stop, step = index.indices(dim_size)
        if step <= 0:
            return None
        length = max(0, (stop - start + step - 1) // step)
        return start, stop, step, length
    return None


def _is_cute_strided_slice_load_for_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
) -> bool:
    def compatible_store_user(user: torch.fx.Node) -> bool:
        if (
            user.op != "call_function"
            or user.target is not store
            or len(user.args) < 4
            or user.args[2] is not state.fx_node
            or user.args[3] is not None
        ):
            return False
        target_node = user.args[0]
        if not isinstance(target_node, torch.fx.Node):
            return False
        target_tensor = target_node.meta.get("val")
        if not isinstance(target_tensor, torch.Tensor) or target_tensor.ndim != 1:
            return False
        store_subscript = user.args[1]
        return (
            isinstance(store_subscript, (list, tuple))
            and len(store_subscript) == 1
            and _cute_positive_1d_slice_bounds(target_tensor, store_subscript[0])
            is not None
        )

    return (
        state.fx_node is not None
        and len(state.fx_node.users) > 0
        and all(compatible_store_user(user) for user in state.fx_node.users)
        and tensor.ndim == 1
        and len(subscript) == 1
        and _cute_positive_1d_slice_bounds(tensor, subscript[0]) is not None
    )


def _codegen_cute_strided_slice_store(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    value: object,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None = None,
) -> ast.AST | None:
    from ..ast_extension import create

    if tensor.ndim != 1 or len(subscript) != 1 or extra_mask is not None:
        return None
    target_bounds = _cute_positive_1d_slice_bounds(tensor, subscript[0])
    if target_bounds is None:
        return None
    target_start, _target_stop, target_step, target_length = target_bounds

    env = CompileEnvironment.current()
    backend = env.backend
    index_dtype = backend.dtype_str(env.index_dtype)
    loop_var = state.device_function.new_var("slice_idx", dce=True)
    target_index = f"{index_dtype}({target_start} + {loop_var} * {target_step})"

    if (
        value_node is not None
        and value_node.op == "call_function"
        and value_node.target is load
    ):
        source_tensor_node = value_node.args[0]
        if not isinstance(source_tensor_node, torch.fx.Node):
            return None
        source_tensor = source_tensor_node.meta.get("val")
        if not isinstance(source_tensor, torch.Tensor) or source_tensor.ndim != 1:
            return None
        source_subscript = value_node.args[1]
        if (
            not isinstance(source_subscript, (list, tuple))
            or len(source_subscript) != 1
        ):
            return None
        source_bounds = _cute_positive_1d_slice_bounds(
            source_tensor, source_subscript[0]
        )
        if source_bounds is None:
            return None
        source_start, _source_stop, source_step, source_length = source_bounds
        if source_length != target_length:
            return None
        source_index = f"{index_dtype}({source_start} + {loop_var} * {source_step})"
        source_name = state.device_function.tensor_arg(source_tensor).name
        value_expr = f"{source_name}[{source_index}]"
        if source_tensor.dtype is torch.bool:
            value_expr = f"({value_expr} != cutlass.Uint8(0))"
    elif isinstance(value, ast.AST):
        value_expr = ast.unparse(value)
    elif isinstance(value, (int, float, bool)):
        value_expr = repr(value)
    else:
        return None

    target_name = state.device_function.tensor_arg(tensor).name
    target_dtype = backend.dtype_str(tensor.dtype)
    value_expr = backend.ast_to_dtype_expr(value_expr, target_dtype)
    store_expr = f"{target_name}.__setitem__(({target_index},), {value_expr})"
    return create(
        ast.For,
        target=create(ast.Name, id=loop_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({target_length})"),
        body=[create(ast.Expr, value=expr_from_string(store_expr))],
        orelse=[],
        type_comment=None,
    )


def _codegen_cute_store_loaded_index_trailing_slices(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    from ..ast_extension import create

    if value_node.target is not load or len(value_node.args) < 2:
        return None
    source_tensor_node = value_node.args[0]
    if not isinstance(source_tensor_node, torch.fx.Node):
        return None
    source_tensor = source_tensor_node.meta.get("val")
    if not isinstance(source_tensor, torch.Tensor):
        return None
    source_subscript = value_node.args[1]
    if not isinstance(source_subscript, (list, tuple)) or not source_subscript:
        return None
    indexer = source_subscript[0]
    if not isinstance(indexer, torch.fx.Node):
        return None
    indexer_value = indexer.meta.get("val")
    if not isinstance(indexer_value, torch.Tensor) or indexer_value.ndim == 0:
        return None
    source_subscript_args = tuple(cast("Any", source_subscript))
    trailing_source = list(source_subscript_args[1:])
    if not trailing_source or not all(idx == slice(None) for idx in trailing_source):
        return None
    if len(subscript) != indexer_value.ndim + len(trailing_source):
        return None
    trailing_store = subscript[indexer_value.ndim :]
    if not all(idx == slice(None) for idx in trailing_store):
        return None

    ast_source_subscript = list(
        map_arg(source_subscript_args, lambda arg: state.env[arg])
    )
    index_exprs = _cute_index_exprs(
        state,
        [indexer_value],
        [ast_source_subscript[0]],
        tensor=source_tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(index_exprs) != 1:
        return None

    prefix_subscript = [*subscript[: indexer_value.ndim]]
    prefix_ast_subscript = [*ast_subscript[: indexer_value.ndim]]
    target_prefix = _cute_index_exprs(
        state,
        prefix_subscript,
        prefix_ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(target_prefix) != indexer_value.ndim:
        return None

    env = CompileEnvironment.current()
    index_dtype = env.backend.dtype_str(env.index_dtype)
    source_loop_vars = [
        state.device_function.new_var("slice_idx", dce=True) for _ in trailing_source
    ]
    source_indices = [
        index_exprs[0],
        *[f"{index_dtype}({var})" for var in source_loop_vars],
    ]
    target_indices = [
        *target_prefix,
        *[f"{index_dtype}({var})" for var in source_loop_vars],
    ]
    if len(source_indices) != source_tensor.ndim or len(target_indices) != tensor.ndim:
        return None

    source_name = state.device_function.tensor_arg(source_tensor).name
    target_name = state.device_function.tensor_arg(tensor).name
    source_dtype = env.backend.dtype_str(source_tensor.dtype)
    target_dtype = env.backend.dtype_str(tensor.dtype)
    source_mask = _cute_combined_mask(
        state,
        [indexer_value],
        None,
        tensor=source_tensor,
    )
    target_mask = _cute_combined_mask(
        state,
        prefix_subscript,
        extra_mask,
        tensor=tensor,
    )
    masks = [mask for mask in (source_mask, target_mask) if mask is not None]
    mask_expr = " and ".join(f"({mask})" for mask in masks) if masks else None
    load_expr = f"{source_name}[{', '.join(source_indices)}]"
    if mask_expr is not None:
        load_expr = f"({load_expr} if {mask_expr} else {source_dtype}(0))"
    store_expr = (
        f"{target_name}.__setitem__({_cute_index_tuple(target_indices)}, "
        f"{env.backend.ast_to_dtype_expr(load_expr, target_dtype)})"
    )
    if mask_expr is not None:
        store_expr = f"{store_expr} if {mask_expr} else None"

    tensor_dim = 0
    for idx in prefix_subscript:
        block_id = None
        if isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
        elif idx == slice(None) and tensor_dim < tensor.ndim:
            block_id = next(
                (
                    candidate
                    for candidate in _matching_block_ids(env, tensor.shape[tensor_dim])
                    if candidate in state.codegen.active_device_loops
                ),
                None,
            )
        tensor_dim += 1
        if block_id is None:
            continue
        axis = None
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            axis = grid_state.block_thread_axes.get(block_id)
        if axis is None:
            loops = state.codegen.active_device_loops.get(block_id)
            if loops:
                axis = loops[-1].block_thread_axes.get(block_id)
        if axis is None or not (0 <= axis < 3):
            continue
        block_size = state.device_function.resolved_block_size(block_id)
        if not isinstance(block_size, int):
            continue
        state.codegen.max_thread_block_dims[axis] = max(
            state.codegen.max_thread_block_dims[axis],
            block_size,
        )
        state.codegen.referenced_thread_block_dims[axis] = max(
            state.codegen.referenced_thread_block_dims[axis],
            block_size,
        )

    stmt: ast.stmt = create(ast.Expr, value=expr_from_string(store_expr))
    for loop_var, source_pos in reversed(
        [*zip(source_loop_vars, range(1, len(source_subscript)), strict=True)]
    ):
        extent = _cute_tensor_dim_size_expr(state, source_tensor, source_pos)
        stmt = create(
            ast.For,
            target=create(ast.Name, id=loop_var, ctx=ast.Store()),
            iter=expr_from_string(f"range({extent})"),
            body=[stmt],
            orelse=[],
            type_comment=None,
        )
    state.add_statement(stmt)
    return ast.Constant(value=None)


def _cute_expand_broadcast_dim(value_node: torch.fx.Node) -> int | None:
    """Return the dim an ``aten.expand`` broadcasts (input size 1 -> >1).

    Returns ``None`` unless ``value_node`` is an ``aten.expand`` whose value has
    exactly one broadcast dimension — i.e. the expanded value carries a stride-0
    mode at exactly one position whose pre-expand extent was 1. This is the
    signal that the stored value replicates one source element across that dim.
    """
    if value_node.target is not torch.ops.aten.expand.default:
        return None
    input_arg = value_node.args[0]
    if not isinstance(input_arg, torch.fx.Node):
        return None
    out_val = value_node.meta.get("val")
    in_val = input_arg.meta.get("val")
    if not isinstance(out_val, torch.Tensor) or not isinstance(in_val, torch.Tensor):
        return None
    if out_val.ndim != in_val.ndim:
        return None
    env = CompileEnvironment.current()
    broadcast_dims = [
        dim
        for dim in range(out_val.ndim)
        if env.known_equal(in_val.shape[dim], 1)
        and not env.known_equal(out_val.shape[dim], 1)
        and out_val.stride(dim) == 0
    ]
    if len(broadcast_dims) != 1:
        return None
    return broadcast_dims[0]


def _cute_block_tile_begin_expr(state: CodegenState, block_id: int) -> str | None:
    """Return the *per-block* tile start for a tile mapped onto a thread axis.

    In the CuTe SIMT model a tile dimension is spread across a thread axis, so
    the strategy's ``index_var`` is the per-*thread* global index
    (``pid * block + thread_idx[axis]``). Subtracting the thread-local coordinate
    yields the per-*block* tile base (``pid * block``), shared by every thread in
    the tile — the correct anchor for a broadcast lane loop. Returns ``None`` when
    the block id has no active thread axis in this scope.
    """
    from .cute_reshape import _grid_local_coord_expr

    loops = state.codegen.active_device_loops.get(block_id)
    if not loops:
        return None
    loop_state = loops[-1]
    thread_axis = loop_state.block_thread_axes.get(block_id)
    global_index = loop_state.strategy.index_var(block_id)
    if thread_axis is None or global_index is None:
        return None
    local_coord = _grid_local_coord_expr(state.codegen, block_id, thread_axis)
    return state.codegen.lift(
        expr_from_string(f"({global_index}) - ({local_coord})"),
        dce=True,
        prefix="tile_begin",
    ).id


def _cute_unsqueeze_expand_load_source(
    value_node: torch.fx.Node, broadcast_dim: int
) -> torch.fx.Node | None:
    """Return the ``hl.load`` feeding ``expand(val[..., None, ...])``.

    Walks ``value_node`` (an ``aten.expand``) back through a single
    unsqueeze-style subscript op (``val[:, None, :]`` inserting the broadcast dim)
    to the originating ``hl.load``. Returns ``None`` unless the chain is exactly
    that shape, so the caller falls back to the load-agnostic path.
    """
    from ...language.view_ops import subscript as subscript_op

    inner = value_node.args[0]
    if not isinstance(inner, torch.fx.Node):
        return None
    if inner.op == "call_function" and inner.target is subscript_op:
        index_arg = inner.args[1] if len(inner.args) > 1 else None
        if not isinstance(index_arg, (list, tuple)):
            return None
        # Exactly one ``None`` (the inserted broadcast dim) at ``broadcast_dim``.
        index_arg_entries = tuple(cast("Any", index_arg))
        none_positions = [
            pos for pos, entry in enumerate(index_arg_entries) if entry is None
        ]
        if none_positions != [broadcast_dim]:
            return None
        load_node = inner.args[0]
    else:
        load_node = inner
    if (
        isinstance(load_node, torch.fx.Node)
        and load_node.op == "call_function"
        and load_node.target is load
        and len(load_node.args) >= 2
    ):
        return load_node
    return None


def _codegen_cute_store_expand_broadcast_tile(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    value: ast.AST,
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node,
) -> ast.AST | None:
    """Lower a store whose value is broadcast across a reused tile dimension.

    Handles the pattern::

        val = hl.load(src, [tile, hl.arange(k)])  # (block, k)
        val_3d = val[:, None, :].expand(block, block, k)  # stride-0 middle dim
        hl.store(out, [idx[tile], tile.index, hl.arange(k)], val_3d)

    Here ``tile`` appears twice in the store index — once as a tensor indexer
    (``idx[tile]``) and once as the bare tile index (``tile.index``) — while the
    value is broadcast (stride 0) along the second (``tile.index``) position. The
    generic SIMT store lowers both positions onto ``tile``'s single thread axis,
    so each thread only writes the ``a == b`` diagonal of the ``(block, block)``
    block. Instead emit a sequential lane loop over the broadcast position so a
    thread holding ``val[a]`` writes the full ``out[idx[a], begin+b, :]`` row for
    every ``b`` in the tile, filling the block. ``val`` is broadcast, so every
    lane reads the same per-thread register.

    Returns ``None`` (a strict no-op) unless every gate matches, so existing
    kernels are byte-for-byte unchanged.
    """
    env = CompileEnvironment.current()
    broadcast_dim = _cute_expand_broadcast_dim(value_node)
    if broadcast_dim is None:
        return None
    if broadcast_dim >= len(subscript):
        return None
    broadcast_idx = subscript[broadcast_dim]
    # The broadcast position must be a bare tile index (a SymInt block id), and
    # that same block id must be reused by another (tensor) index position — the
    # collision the generic path mis-handles.
    if not isinstance(broadcast_idx, torch.SymInt):
        return None
    broadcast_block_id = env.get_block_id(broadcast_idx)
    if broadcast_block_id is None:
        return None
    block_size = state.device_function.resolved_block_size(broadcast_block_id)
    if not isinstance(block_size, int) or block_size <= 1:
        return None
    reused = False
    for pos, idx in enumerate(subscript):
        if pos == broadcast_dim:
            continue
        if isinstance(idx, torch.Tensor):
            for dim_size in idx.shape:
                if broadcast_block_id in _matching_block_ids(env, dim_size):
                    reused = True
                    break
        if reused:
            break
    if not reused:
        return None

    # Walk the value chain ``expand -> unsqueeze(None) -> load`` to recover the
    # source load. The stored value is a per-thread register holding ``val[a, c]``
    # whose coordinates live on the *load*'s thread axes; the store's own free
    # ``hl.arange`` index entries are distinct nodes that the synthetic-axis
    # machinery assigns to *different* axes. Reusing the load's coordinate for
    # those non-broadcast positions keeps the register and the store address on
    # the same thread axis (otherwise thread ``(a, c_load, c_store)`` would write
    # ``out[..., c_store] = val[a, c_load]`` for ``c_load != c_store``).
    load_node = _cute_unsqueeze_expand_load_source(value_node, broadcast_dim)
    load_coords: list[str] | None = None
    load_subscript_proxy: tuple[object, ...] | None = None
    if load_node is not None:
        load_tensor_node = load_node.args[0]
        load_subscript = load_node.args[1]
        if isinstance(load_tensor_node, torch.fx.Node) and isinstance(
            load_subscript, (list, tuple)
        ):
            load_tensor = load_tensor_node.meta.get("val")
            if isinstance(load_tensor, torch.Tensor):
                load_subscript_args = tuple(cast("Any", load_subscript))
                load_subscript_proxy = tuple(
                    map_arg(load_subscript_args, lambda arg: arg.meta["val"])
                )
                load_subscript_ast = map_arg(
                    load_subscript_args, lambda arg: state.env[arg]
                )
                load_coords = _cute_index_exprs(
                    state,
                    [*load_subscript_proxy],
                    [*load_subscript_ast],
                    tensor=load_tensor,
                    inactive_singleton_slice_expr="0",
                )
                if len(load_coords) != load_tensor.ndim:
                    load_coords = None
                    load_subscript_proxy = None

    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    if len(index_exprs) != tensor.ndim or "None" in index_exprs:
        return None

    # Re-align each non-broadcast free-``hl.arange`` store position onto the
    # load's matching coordinate. Value dim ``d`` maps to load dim ``d`` before
    # the unsqueezed broadcast dim and ``d - 1`` after it. Only positions where
    # *both* the store and the matching load entry are free ``hl.arange`` index
    # tensors are remapped — a tensor *indexer* (``idx[tile]``) keeps its own
    # coordinate.
    if load_coords is not None and load_subscript_proxy is not None:
        for pos, idx in enumerate(subscript):
            if pos == broadcast_dim or not isinstance(idx, torch.Tensor):
                continue
            load_dim = pos if pos < broadcast_dim else pos - 1
            if not (0 <= load_dim < len(load_coords)):
                continue
            if isinstance(load_subscript_proxy[load_dim], torch.Tensor):
                index_exprs[pos] = load_coords[load_dim]

    # Replace the broadcast position's coordinate (currently the reused tile's
    # per-thread global index) with ``block_begin + lane`` so the lane loop sweeps
    # the full tile block, identically for every thread in the tile. ``block_begin``
    # is the *per-block* tile start (``global_index - local_coord``); in the CuTe
    # SIMT model the tile is mapped onto a thread axis, so the bare offset var
    # still carries the per-thread ``thread_idx`` lane and must be stripped.
    block_begin = _cute_block_tile_begin_expr(state, broadcast_block_id)
    if block_begin is None:
        return None
    lane_var = state.device_function.new_var("bcast_lane", dce=True)
    index_dtype = env.index_type()
    broadcast_coord = f"({block_begin}) + {index_dtype}({lane_var})"
    index_exprs[broadcast_dim] = broadcast_coord

    backend = env.backend
    target_dtype = backend.dtype_str(tensor.dtype)
    tensor_name = state.device_function.tensor_arg(tensor).name
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    store_expr = expr_from_string(
        _cute_scalar_store_expr(tensor_name, index_exprs, "{value}"),
        value=value,
    )

    # Base mask excludes the broadcast position (its bound is enforced by the lane
    # bound below); other positions keep their tile/tensor masks.
    base_subscript = [
        slice(None) if pos == broadcast_dim else idx
        for pos, idx in enumerate(subscript)
    ]
    mask_expr = _cute_combined_mask(state, base_subscript, extra_mask, tensor=tensor)
    dim_size = _cute_tensor_dim_size_expr(state, tensor, broadcast_dim)
    lane_bound = f"({broadcast_coord}) < {dim_size}"
    mask_expr = lane_bound if mask_expr is None else f"({mask_expr}) and {lane_bound}"

    from ..ast_extension import create

    mask_ast = expr_from_string(mask_expr)
    assert isinstance(mask_ast, ast.expr)
    assert isinstance(store_expr, ast.expr)
    body_stmt: ast.stmt = ast.fix_missing_locations(
        ast.If(
            test=mask_ast,
            body=[ast.Expr(value=store_expr)],
            orelse=[],
        )
    )
    loop_stmt = create(
        ast.For,
        target=create(ast.Name, id=lane_var, ctx=ast.Store()),
        iter=expr_from_string(f"range({block_size})"),
        body=[body_stmt],
        orelse=[],
        type_comment=None,
    )
    state.add_statement(loop_stmt)
    return ast.Constant(value=None)


def _try_splice_tcgen05_unary_epilogue(
    state: CodegenState,
    tensor: object,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """Splice attempt for ``out[tile] = chain(acc)[.to(x.dtype)]``.

    Returns the splice-completion sentinel (``ast.Constant(value=None)``)
    on a successful splice (the caller should return it directly), and
    ``None`` if the splice did not fire — the caller should continue to
    the loud-failure backstop or the SIMT fallback.

    Splice is attempted only when the kernel has a tcgen05-registered
    matmul fx_node (``cute_state.matmul_fx_nodes`` non-empty), the
    store value has a backing FX node, the store target is a 2-D
    ``torch.Tensor``, and the chain analyzer accepts the value chain
    (returning ``(chain, anchor)`` for a non-empty chain rooted at
    a tcgen05 matmul). Chains the whitelist rejects (broadcast aux
    loads, reductions, kwarg-bearing binaries, etc.) leave the
    analyzer returning ``None`` and the splice does not fire — the
    loud-failure backstop then catches them.
    """
    cute_state = state.device_function.cute_state
    if not cute_state.matmul_fx_nodes:
        return None
    if value_node is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    analyzed = analyze_tcgen05_unary_epilogue_chain(
        state, value_node, output_global_shape=tuple(tensor.shape)
    )
    if analyzed is None:
        return None
    chain, anchor = analyzed
    if not chain.steps:
        return None
    anchor_result_var = cute_state.matmul_fx_node_result_vars.get(anchor)
    if anchor_result_var is None:
        return None
    rewritten_stmt = _codegen_cute_store_tcgen05_tile(
        state,
        tensor,
        subscript,
        ast_subscript,
        extra_mask,
        anchor_result_var,
        epilogue_chain=chain,
    )
    if rewritten_stmt is None:
        return None
    stmts = rewritten_stmt if isinstance(rewritten_stmt, list) else [rewritten_stmt]
    for stmt in stmts:
        state.add_statement(stmt)
    return ast.Constant(value=None)


def _try_splice_tcgen05_grouped_tail_epilogue(
    state: CodegenState,
    tensor: object,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    value_node: torch.fx.Node | None,
) -> ast.AST | None:
    """Splice grouped preserve-output M/N tail stores into tcgen05."""
    cute_state = state.device_function.cute_state
    if not cute_state.matmul_fx_nodes:
        return None
    if value_node is None or state.fx_node is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return None
    grouped_tail = cute_state.grouped_tail_proof_for_store(state.fx_node)
    if grouped_tail is None:
        return None
    anchor_result_var = cute_state.matmul_fx_node_result_vars.get(grouped_tail.anchor)
    if anchor_result_var is None:
        return None
    rewritten_stmt = _codegen_cute_store_tcgen05_tile(
        state,
        tensor,
        subscript,
        ast_subscript,
        extra_mask,
        anchor_result_var,
        grouped_tail_epilogue=grouped_tail,
    )
    if rewritten_stmt is None:
        return None
    state.codegen.remove_statements_owned_by_nodes(grouped_tail.producer_nodes)
    stmts = rewritten_stmt if isinstance(rewritten_stmt, list) else [rewritten_stmt]
    for stmt in stmts:
        state.add_statement(stmt)
    return ast.Constant(value=None)


@_decorators.codegen(store, "cute")
def _(state: CodegenState) -> ast.AST:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    raw_value = state.ast_args[2]
    extra_mask = state.ast_args[3]
    assert isinstance(extra_mask, (type(None), ast.AST))
    value_node = None
    if state.fx_node is not None and len(state.fx_node.args) > 2:
        maybe_value_node = state.fx_node.args[2]
        if isinstance(maybe_value_node, torch.fx.Node):
            value_node = maybe_value_node

    if isinstance(tensor, torch.Tensor):
        affine_range_store = _codegen_cute_affine_range_store(
            state,
            tensor,
            subscript,
            ast_subscript,
            raw_value,
            extra_mask,
            value_node,
        )
        if affine_range_store is not None:
            state.add_statement(affine_range_store)
            return ast.Constant(value=None)
        affine_reshape_store = _codegen_cute_affine_reshape_store(
            state,
            tensor,
            subscript,
            ast_subscript,
            extra_mask,
            value_node,
        )
        if affine_reshape_store is not None:
            state.add_statement(affine_reshape_store)
            return ast.Constant(value=None)
        strided_slice_store = _codegen_cute_strided_slice_store(
            state,
            tensor,
            subscript,
            raw_value,
            extra_mask,
            value_node,
        )
        if strided_slice_store is not None:
            state.add_statement(strided_slice_store)
            return ast.Constant(value=None)

    value = state.ast_arg(2)

    if value_node is not None:
        if value_node.op == "call_function":
            if isinstance(tensor, torch.Tensor):
                rewritten_stmt = _codegen_cute_store_stack_load(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    value,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
                rewritten_stmt = _codegen_cute_store_loaded_index_trailing_slices(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
                rewritten_stmt = _codegen_cute_store_expand_broadcast_tile(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    value,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
                rewritten_stmt = _codegen_cute_store_permute_lane_loops(
                    state,
                    tensor,
                    subscript,
                    ast_subscript,
                    value,
                    extra_mask,
                    value_node,
                )
                if rewritten_stmt is not None:
                    return rewritten_stmt
            from .cute_reshape import codegen_cute_store_permute

            rewritten = codegen_cute_store_permute(state, value, value_node)
            if rewritten is not None:
                value = rewritten

    if isinstance(tensor, tuple):
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        _tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        assert isinstance(dev_ptrs_ast, ast.AST)
        tensor_like, dev_ptrs = tensor
        offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*subscript],
            ast_subscript,
        )
        backend = CompileEnvironment.current().backend
        target_dtype = backend.dtype_str(tensor_like.dtype)
        value = expr_from_string(
            backend.ast_to_dtype_expr("{value}", target_dtype),
            value=value,
        )
        ptr_expr = _cute_stack_tensor_pointer_expr(
            target_dtype, dev_ptrs_ast, offset_expr
        )
        store_expr = expr_from_string(
            "({ptr}).store({value})", ptr=ptr_expr, value=value
        )
        mask_expr = _cute_stack_tensor_mask_expr(
            state,
            tensor_like,
            dev_ptrs,
            [*subscript],
            extra_mask,
        )
        if mask_expr is None:
            return store_expr
        mask_ast = expr_from_string(mask_expr)
        assert isinstance(mask_ast, ast.expr)
        assert isinstance(store_expr, ast.expr)
        state.add_statement(
            ast.fix_missing_locations(
                ast.If(
                    test=mask_ast,
                    body=[ast.Expr(value=store_expr)],
                    orelse=[],
                )
            )
        )
        return ast.Constant(value=None)
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"store target type: {type(tensor)}")

    _log_cute_layout(state, "store")

    if isinstance(value, ast.Name):
        rewritten_stmt = _codegen_cute_store_tcgen05_tile(
            state,
            tensor,
            subscript,
            ast_subscript,
            extra_mask,
            value.id,
        )
        if rewritten_stmt is not None:
            stmts = (
                rewritten_stmt if isinstance(rewritten_stmt, list) else [rewritten_stmt]
            )
            for stmt in stmts:
                state.add_statement(stmt)
            return ast.Constant(value=None)

    # Try to splice a whitelisted chain epilogue
    # (`out[tile] = chain(acc)[.to(x.dtype)]`) into the role-local
    # tcgen05 epilogue's per-thread T2R loop. Implementation in
    # ``_try_splice_tcgen05_unary_epilogue``. Chains the whitelist
    # rejects (broadcast aux loads, reductions, etc.) leave the
    # splice off and fall through to the loud-failure backstop
    # below.
    spliced = _try_splice_tcgen05_unary_epilogue(
        state, tensor, subscript, ast_subscript, extra_mask, value_node
    )
    if spliced is not None:
        return spliced
    spliced = _try_splice_tcgen05_grouped_tail_epilogue(
        state, tensor, subscript, ast_subscript, extra_mask, value_node
    )
    if spliced is not None:
        return spliced

    # Loud-failure backstop for fused-epilogue stores that follow a
    # tcgen05 matmul. The tcgen05 grid-emission path (in `program_id.py`)
    # does not bind the per-block-id `indices_<n>` / `mask_<n>` variable
    # names that the SIMT-fallback store path expects, so falling through
    # here would emit a kernel that crashes inside the cute DSL with
    # `name 'mask_0' is not defined`. Detect the pattern here — any
    # store value whose FX user chain transitively reaches a
    # tcgen05-registered matmul fx node — and raise a structured error
    # so the caller sees the actionable message instead of a cute-DSL
    # crash. Fixing this requires either (a) extending the tcgen05 grid
    # to emit per-block-id index/mask vars, or (b) per-subtile lambda
    # emission in `_codegen_cute_store_tcgen05_tile`.
    if (
        state.device_function.cute_state.matmul_fx_nodes
        and value_node is not None
        and reach_tcgen05_matmul_anchors(state, value_node)
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 MMA path does not yet emit per-block-id indices "
            "and masks for non-whitelisted fused epilogues that follow "
            "the MMA. The store target's value chain depends on a "
            "tcgen05 matmul result through ops the chain analyzer "
            "rejects (e.g. aux tensors with a 3-D underlying shape "
            "and a static collapse like `aux3d[tile_m, tile_n, 0]`, "
            "loads whose index expression is not exactly the "
            "carrier tile-id symbol, non-scalar binary ops, "
            "`aten.add.Tensor` with `alpha=k`, or an intermediate "
            "`.to(d_inter)` cast where `d_inter` differs from the "
            "store-target dtype). Identity stores "
            "(`out[tile] = acc.to(x.dtype)`), whitelisted unary chains "
            "(relu/tanh/exp/log/sqrt/abs/neg + scalar add/sub/mul/div "
            "on the accumulator carrier), exact-shape 2-D "
            "auxiliary-tensor binary ops (`acc + residual[tile_m, "
            "tile_n]`), and rank-1 trailing-axis (rowvec) broadcast "
            "aux loads (`acc + bias[tile_n]`) all work via the "
            "fused-epilogue splice path. The leading-axis rank-1 "
            "form (`acc + bias[tile_m]`) is rejected because a bare "
            "rank-1 RHS aligns to the trailing axis under PyTorch "
            "broadcasting; an explicit colvec broadcast must be "
            "written with `bias[tile_m][:, None]` / "
            "`.unsqueeze(-1)`.",
        )

    tensor_name = state.device_function.tensor_arg(tensor).name
    backend = CompileEnvironment.current().backend
    target_dtype = backend.dtype_str(tensor.dtype)
    value = expr_from_string(
        backend.ast_to_dtype_expr("{value}", target_dtype),
        value=value,
    )
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_singleton_slice_expr="0",
    )
    topk_lane_expr: object | None = None
    topk_k: object | None = None
    if state.fx_node is not None and len(state.fx_node.args) > 2:
        value_node = state.fx_node.args[2]
        if (
            isinstance(value_node, torch.fx.Node)
            and value_node.target is operator.getitem
            and isinstance(value_node.args[0], torch.fx.Node)
            and value_node.args[0].target is torch.ops.aten.topk.default
        ):
            topk_lane_expr = value_node.args[0].meta.get("cute_topk_lane_expr")
            topk_k = value_node.args[0].meta.get("cute_topk_k")
    if isinstance(topk_lane_expr, str) and isinstance(topk_k, int):
        index_exprs[-1] = topk_lane_expr
    # ── the TV store leg (``PORT_SPEC_layout.md`` §3a, D2 in §4c) ────────────
    #
    # Before the rework there was no store-side width concept AT ALL: every
    # store went through ``_cute_scalar_store_expr``'s one-element ``.store()``
    # at a V-strided address, which is the whole of 02_PERF's 4.03x sector
    # inflation (LEDGER E001).  Here the store is ``partition_D`` off the same
    # ``get_slice`` as the load, so the width is the atom's and the addresses
    # are the layout's -- the two legs have no way to hold different opinions.
    tv_store = _maybe_codegen_cute_tv_store(
        state, tensor, subscript, index_exprs, tensor_name, value, extra_mask
    )
    if tv_store is not None:
        return tv_store
    store_uses_pointer = "None" not in index_exprs
    store_expr = _cute_scalar_store_expr(tensor_name, index_exprs, "{value}")
    assign_expr = expr_from_string(store_expr, value=value)

    mask_expr = _cute_combined_mask(state, subscript, extra_mask, tensor=tensor)
    if isinstance(topk_lane_expr, str) and isinstance(topk_k, int):
        topk_mask = f"({topk_lane_expr}) < {topk_k}"
        mask_expr = topk_mask if mask_expr is None else f"({mask_expr}) and {topk_mask}"
    if mask_expr is None:
        return assign_expr
    if store_uses_pointer:
        mask_ast = expr_from_string(mask_expr)
        assert isinstance(mask_ast, ast.expr)
        assert isinstance(assign_expr, ast.expr)
        state.add_statement(
            ast.fix_missing_locations(
                ast.If(
                    test=mask_ast,
                    body=[ast.Expr(value=assign_expr)],
                    orelse=[],
                )
            )
        )
        return ast.Constant(value=None)
    return expr_from_string(
        f"({store_expr} if {mask_expr} else None)",
        value=value,
    )


def _cute_load_feeds_sort_or_scan(load_node: object) -> bool:
    """Return True if ``load_node`` feeds a sort/topk/_associative_scan.

    Direct users (sort/topk and the scalar ``_associative_scan`` path) are
    matched immediately.  For a tuple ``_associative_scan`` the index stream is
    typically a ``load`` that flows through a chain of dtype-cast / shape ops
    (e.g. ``indices[tile].float().unsqueeze(1).expand_as(vals)``) before
    reaching the scan.  To recover a scalar load for that stream we follow the
    forward chain through those pass-through ops.
    """
    from torch.fx.node import Node

    from .indexing import is_cute_shape_chain_target

    if not isinstance(load_node, Node):
        return False

    passthrough_targets = (torch.ops.prims.convert_element_type.default,)
    seen: set[Node] = set()
    stack: list[Node] = [load_node]
    while stack:
        node = stack.pop()
        for user in node.users:
            if not isinstance(user, Node):
                continue
            target = user.target
            if (
                target in (torch.ops.aten.sort.default, torch.ops.aten.topk.default)
                or getattr(target, "__name__", None) == "_associative_scan"
            ):
                return True
            if (
                is_cute_shape_chain_target(target) or target in passthrough_targets
            ) and user not in seen:
                seen.add(user)
                stack.append(user)
    return False


def _cute_vector_load_ctx(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    index_exprs: list[str],
    extra_mask: ast.AST | None,
    slice_block_ids: dict[int, int],
) -> tuple[int, int, str, int] | None:
    """Return (vec_width, lane_block_id, mode, lane_axis_pos) when a vec load
    may be emitted.

    ``mode`` is one of ``"vec"`` (explicit ``cute.arch.load(..., V)``) or
    ``"unroll"`` (per-element scalar bitcast inside a constexpr V-loop).
    ``lane_axis_pos`` is the ``index_exprs`` position of the stride-1 lane axis;
    the caller needs it to build the load descriptor's per-axis bounds.
    Returns None when any predicate for a 128-bit gmem load fails, in which
    case the caller falls back to ``_cute_scalar_load_expr``.

    ``slice_block_ids`` is the ``{index_exprs position: block_id}`` map
    ``_cute_index_exprs`` recorded while it built ``index_exprs`` -- the caller
    must pass the map from the SAME call that produced ``index_exprs``.
    """
    from ..reduction_strategy import ReductionStrategy

    env = CompileEnvironment.current()
    if env.backend.name != "cute":
        return None
    if extra_mask is not None:
        return None
    if "None" in index_exprs:
        return None
    if tensor.dtype not in _CUTE_VECTOR_DTYPES and not _cute_is_unroll_dtype(
        tensor.dtype
    ):
        return None
    # ⛔ THE ``feeds_reduction`` FX-USER WALK IS GONE WITH THE ``"vec"`` MODE.
    #
    # It answered "does this load's result eventually reach a reduction op?" by a
    # transitive walk over ``fx_node.users`` with a SUBSTRING test on the target name
    # (``"reduction" in target_name or ...``).  Its own comment said so: *"``feeds_
    # reduction`` is required ONLY for the ``vec`` mode below; the ``unroll`` mode
    # also applies to the consume sweep where the load result feeds an elementwise
    # pipeline (no reduction)"*.  The reason it existed is that ``vec`` handed the
    # combine a whole length-V vector, and the consume sweep mixes a loaded vector
    # with a post-reduction SCALAR, which the DSL cannot broadcast.
    #
    # ⇒ every surviving mode returns a per-element scalar, so no mode has any use
    # for the question.  Deleting the walk removes a transitive FX traversal per load
    # site, and one more name-substring test from a tree that is trying to stop
    # recovering facts from text.
    #
    # ⚠ The ``fx_node is None`` guard it also served is NOT dropped -- ``state.fx_node``
    # is still dereferenced downstream (the sort/scan check at the caller), and a load
    # lowered outside an FX node must still decline here.
    if state.fx_node is None:
        return None
    # The lane/vec axis must be a tensor dim that is stride-1 so that
    # consecutive lane iters fetch consecutive bytes.  For a row-major lhs
    # the reduction axis is the LAST subscript position; for a column-major
    # rhs (e.g. the K-major ``y`` of a tcgen05 fp8 matmul) it is the FIRST.
    # ``_cute_lane_axis_pos`` records the index_exprs position of that
    # stride-1 lane axis so the hoist substitutes the per-lane base there
    # (not blindly at ``[-1]``).
    # Find the stride-1 dim WITHOUT forcing specialization of a symbolic
    # stride: a contiguous dim has a concrete ``int`` stride of 1, so only
    # accept plain ints here.  Calling ``int()`` on a ``SymInt`` stride would
    # bake the (otherwise-dynamic) size into the kernel — see the
    # ``test_mark_static`` regression where ``int(stride(0))`` specialized
    # ``n``.
    stride1_tensor_dim: int | None = None
    for d in range(tensor.ndim):
        s = tensor.stride(d)
        if isinstance(s, int) and s == 1:
            stride1_tensor_dim = d
            break
    if stride1_tensor_dim is None:
        return None
    # Locate the non-None subscript carrying an active lane block.  Slices
    # resolve to the matching tensor-dim block via the strategy that's
    # currently active for that block.  Prefer the block sitting on the
    # stride-1 tensor dim (the true lane axis), and record its index_exprs
    # position.
    inner_block_id: int | None = None
    lane_axis_pos: int | None = None
    # ── CARRY THE INDEXER'S BINDING; DO NOT RE-DERIVE IT ────────────────────────
    #
    # A bare ``slice(None)`` names no loop, so *something* has to decide which
    # block drives that axis.  ``_cute_index_exprs`` already decided -- it called
    # ``resolve_active_slice_block_id`` to pick the block whose ``index_var`` it
    # then wrote into ``index_exprs`` -- and ``slice_block_ids`` is that decision,
    # keyed by ``index_exprs`` position.  So this loop READS it.
    #
    # It used to re-derive the binding here instead, with two layered heuristics
    # (an ``index_exprs[expr_pos]`` string match against every active block's
    # ``index_var``, then a first-match ``known_equal(block.numel, dim_size)``
    # size scan guarded by a ``bound_block_ids`` set).  Both are strictly weaker
    # than the resolver they were shadowing, which filters by activeness, by a
    # LIVE ``used_block_ids`` set (so an earlier slice of the same subscript is
    # excluded -- the static SymInt-only ``bound_block_ids`` could not do that),
    # and breaks a remaining tie on the ``reduction`` flag.  MEASURED over 23
    # slice sites in 11 kernels: the indexer is 23/23 correct and the size scan
    # alone mis-binds at 3/23.
    #
    # ⚠ THE STRING MATCH IS NOT A SAFETY NET, it is a third opinion.  It asked
    # ``cand_strategy.index_var(cand_bid)`` DIRECTLY, so it could not see
    # ``matmul_operand_index_override`` -- under which the indexer legitimately
    # emits a serial-loop temp that equals no strategy's ``index_var``.  MEASURED
    # on ``test_indexing.py::test_full_slice_in_reduction_loop`` (a ``baddbmm``
    # that sets the override): identity MISSED at 4 of 8 slice sites, handing
    # those axes to the size scan -- which at ``N == C == D == 16`` is ambiguous.
    #
    # The mis-binding matters because ``inner_block_id`` selects the STRATEGY
    # below, and the strategy is what decides whether a vector load is emitted
    # and how wide (LEDGER E052/E053/E054; ``_notes/EXPLAIN_slice_resolver.md``
    # has the full derivation and the per-cell measurements).
    expr_pos = -1
    tensor_dim = 0
    for idx in subscript:
        if idx is None:
            continue
        expr_pos += 1
        if isinstance(idx, torch.SymInt):
            bid = env.get_block_id(idx)
            if bid is not None and state.codegen.active_device_loops.get(bid):
                if tensor_dim == stride1_tensor_dim or inner_block_id is None:
                    inner_block_id = bid
                    lane_axis_pos = expr_pos
        elif isinstance(idx, slice) and idx == slice(None):
            # A recorded block still has to be one THIS site can use: the
            # branches below read ``active_device_loops[inner_block_id][-1]``, so
            # a block the indexer resolved through the grid state (or through the
            # index override, which consults no loop at all) is not a lane
            # candidate here.  Same precondition the old scan applied to every
            # candidate it considered.
            rec_bid = slice_block_ids.get(expr_pos)
            if rec_bid is not None and state.codegen.active_device_loops.get(rec_bid):
                if tensor_dim == stride1_tensor_dim or inner_block_id is None:
                    inner_block_id = rec_bid
                    lane_axis_pos = expr_pos
        tensor_dim += 1
    if inner_block_id is None or lane_axis_pos is None:
        return None
    loops = state.codegen.active_device_loops.get(inner_block_id)
    if not loops:
        return None
    strategy = getattr(loops[-1], "strategy", None)
    # ⭐ ``ReductionStrategy``, not ``LoopedReductionStrategy``.  Every field read
    # in this branch now has a class-default sentinel on the base
    # (``_cute_reduction_vec_width = 1``, ``_cute_tv_plan = None``, ...), so a
    # reduction strategy that does not carry the capability declines on the very
    # next line instead of being excluded by its CLASS.  ``BlockReductionStrategy``
    # reaches here too and returns ``None`` at ``vec_width <= 1``, which is exactly
    # what the class test gave it before.
    if isinstance(strategy, ReductionStrategy):
        vec_width = strategy._cute_reduction_vec_width
        if vec_width <= 1:
            return None
        # ── A LIVE REDUCTION-AXIS MASK IS FATAL TO THE **LEGACY** VEC MODES, NOT
        #    TO THE TV PATH ────────────────────────────────────────────────────
        #
        # ``vec`` mode addresses the lane axis with ``rindex`` directly and
        # DEFERS the mask to a post-fold scalar (see the ``vec_mode == "vec"``
        # branch below), which is only sound when every element it reads is in
        # bounds -- hence the blanket decline here, historically.
        #
        # The TV path is different in kind: the mask is NOT deferred.  Its
        # per-element index is ``rindex = lane_base + vi`` inside
        # ``range_constexpr(vec)``, i.e. exactly the element fragment slot ``vi``
        # holds, so the caller's ``x if mask else <identity>`` wrapper still sits
        # between this load and the combine and ``_mask_to``'s per-op identity is
        # untouched (``cute/ragged_tail.py`` invariant I4).  The out-of-bounds
        # ADDRESS is handled separately, by the guard on the copy itself (I3).
        #
        # So the decline is scoped to the modes whose soundness argument actually
        # needs it.  Fail-closed shape: a plan is required, and it must be a plan
        # this site can honour, or we fall through to the same decline as before.
        tv_plan = strategy._cute_tv_plan  # pyrefly: ignore [missing-attribute]
        if strategy._mask_var is not None and not (
            tv_plan is not None and tv_plan.vec == vec_width
        ):
            return None
        if strategy._cute_reduction_lane_extent <= 0:
            return None
        mode = strategy._cute_reduction_vec_mode  # pyrefly: ignore [missing-attribute]
        # ── the TV path (``PORT_SPEC_layout.md`` §3a / §6d) ─────────────────
        #
        # Checked BEFORE the legacy modes, and it takes over whenever the
        # strategy holds a plan whose ``vec`` this site can honour.  The check
        # is ``plan.vec == vec_width``, i.e. *the plan is authoritative for the
        # width*: the strategy's trip count was derived from ``plan.vec``, so
        # any site that cannot do a width-``plan.vec`` copy must not fall back
        # to a narrower one -- it would leave the stride assuming the wider
        # width, which is class 1 exactly.  Instead ``_cute_tv_load_eligible``
        # returns False for the whole reduction at ``__init__`` time and the
        # plan is never built, so the strategy stays scalar.
        if (
            (plan := strategy._cute_tv_plan) is not None  # pyrefly: ignore [missing-attribute]
            and plan.vec == vec_width
            and _cute_tv_site_eligible(state, strategy, tensor, subscript)
        ):
            return vec_width, inner_block_id, "tv", lane_axis_pos
        # ⛔ THE ``"vec"`` MODE IS GONE, and with it the branch's OOB-guard machinery.
        #
        # ``vec`` mode issued one explicit ``cute.arch.load(ptr, V x elem)`` and
        # DEFERRED its mask entirely to a post-fold scalar -- so nothing gated the
        # ADDRESS, which is why it needed ``CuteVecLoadDesc`` /
        # ``_cute_guarded_pointer_expr`` (a whole descriptor + guard-rendering layer,
        # 252 lines, all branch-new) to keep its pointer in bounds on every axis.
        #
        # ⭐ THAT LAYER WAS CORRECTNESS *CONDITIONAL ON AN OPTIMISATION*, never
        # correctness as such: the scalar floor it falls back to has no address to
        # guard.  MEASURED before deleting it, two ways:
        #  * over all 40 frozen cells, ZERO reach the guard machinery and 40/40 emit
        #    the TV layout instead (the last two were re-freezed onto TV once the tile
        #    path learned the ragged round-up);
        #  * over a config sweep with the TV arm FORCED OFF (so the legacy modes are
        #    visible rather than merely shadowed), ``"vec"`` fires at ZERO sites --
        #    only ``"unroll"`` and the scalar floor do.
        #
        # ⇒ every shape it served now reaches either the TV copy or the scalar floor,
        # both of which carry their own bounds argument.  ``feeds_reduction`` above is
        # kept: it is still the ``unroll``-vs-scalar question for a load whose user is
        # a cast.
        if mode == "unroll":
            if tensor.dtype not in _CUTE_VECTOR_UNROLL_DTYPES:
                return None
            # The CuTe DSL's ``nvvm.load.ext`` only supports vec sizes 2
            # and 4 for bf16/fp16 (V=8 raises ICE).  Cap effective V
            # here so the autotuner's V=8 seed still compiles instead
            # of crashing.
            #
            # NOTE this cap is why V=8 was unreachable on this path.  The TV
            # route above does not need it: MEASURED, ``cute.copy`` through a
            # 128-bit atom is fine at V=8 where ``cute.arch.load`` with an
            # explicit ``VectorType([8], Uint16)`` ICEs
            # (``_redfix/repro/probe_load_widths.py``).
            if vec_width > 4:
                return None
            # Need a lane base index var + a constexpr V-loop var; both
            # are set up by the strategy's codegen_device_loop.
            if (
                strategy._cute_lane_base_index_var is None  # pyrefly: ignore [missing-attribute]
                or strategy._cute_lane_body is None  # pyrefly: ignore [missing-attribute]
            ):
                return None
            return vec_width, inner_block_id, "unroll", lane_axis_pos
        return None
    # CuTe N-D tile strategy with lane loops: vec is set up per-block in
    # ``CuteNDTileStrategy.__init__`` when the autotuner picks
    # ``cute_vector_widths[block_id]`` > 1 and EPT is divisible by V.  Mode
    # is forced to ``"unroll"`` (per-element bitcast) for fp16/bf16 since
    # subscripting a bf16/fp16 vector in the CuTe DSL is unsafe; fp32
    # could in principle use ``"vec"`` but the per-element pipeline runs
    # most of the consume-sweep code after a cast, so unroll is the
    # robust choice.
    from ..tile_strategy import BlockSizeTileStrategy

    if isinstance(strategy, BlockSizeTileStrategy):
        vec_by_block = getattr(strategy, "_cute_lane_vec_width_by_block", None)
        if not isinstance(vec_by_block, dict):
            return None
        vec_width = vec_by_block.get(inner_block_id, 1)
        if vec_width <= 1:
            return None
        # ── THE TV PATH, CHECKED FIRST -- SO THE TWO PATHS CANNOT BOTH FIRE ─────
        #
        # ⭐ THE EXCLUSIVITY IS STRUCTURAL, NOT A CONVENTION.  ``mode`` is a single
        # return value: returning ``"tv"`` here means the caller's ``tile_unroll``
        # branches are unreachable for this site, and the ``tile_unroll`` returns
        # below are reached only when this declined.  There is no ordering the
        # caller has to respect and no flag either path could forget to check.
        #
        # ⚠ AND EVERY PREDICATE BELOW THIS POINT IS LEFT EXACTLY AS IT WAS -- in
        # particular the row-stride clamp, which fixes a live misaligned-address
        # fault.  This block adds an EARLIER exit; it does not weaken a later one.
        # A site that takes TV is clamped by the plan instead, through
        # ``chunk_plan``'s own ``row_stride_elems=gcd(strides)`` (and it must
        # DECLINE rather than narrow, which ``_build_cute_tv_plan_for_block``
        # enforces by requiring ``plan.vec == vec_cap``).
        if _cute_tv_tile_site_takes_over(
            state, strategy, tensor, subscript, inner_block_id, vec_width
        ):
            return vec_width, inner_block_id, "tv", lane_axis_pos
        if not _cute_is_unroll_dtype(tensor.dtype):
            return None
        # The CuTe DSL's ``nvvm.load.ext`` ICEs at V=8 for fp16/bf16 (and
        # for the V=8 ``Uint8`` vector used by fp8), so widths > 4 cannot
        # use a single ``cute.arch.load``.  V=8 still
        # gets full LDG.128 throughput via the ``tile_unroll_split2``
        # mode: two back-to-back ``cute.arch.load(..., V=4)`` calls
        # (covering vec lanes 0-3 and 4-7) emit as two LDG.64s that the
        # SASS scheduler can overlap.  Wider Vs (16, 32, ...) are not
        # supported.
        if vec_width > 8:
            return None
        if vec_width == 8 and vec_width % 4 != 0:
            return None
        base_var_by_block = getattr(
            strategy, "_cute_lane_base_index_var_by_block", None
        )
        lane_body_by_block = getattr(strategy, "_cute_lane_body_by_block", None)
        vec_lane_var_by_block = getattr(strategy, "_cute_vec_lane_var_by_block", None)
        if (
            not isinstance(base_var_by_block, dict)
            or not isinstance(lane_body_by_block, dict)
            or not isinstance(vec_lane_var_by_block, dict)
            or inner_block_id not in base_var_by_block
            or inner_block_id not in lane_body_by_block
            or inner_block_id not in vec_lane_var_by_block
        ):
            return None
        # When the per-thread vec base could straddle the tensor edge
        # (e.g. ``numel`` not a multiple of V), the masked-tail iter
        # could load garbage in some lanes.  Gate the per-element mask
        # path correctly by requiring ``numel % V == 0`` so partial-vec
        # straddles are impossible.
        numel = env.block_sizes[inner_block_id].numel
        if not env.known_multiple(numel, vec_width):
            return None
        # ⭐ CLAMP ON THE NON-LANE (ROW) STRIDE.  A ``V``-wide load moves
        # ``issue * dtype_bits / 8`` bytes from ``base + row * row_stride``, so it
        # needs ``issue | row_stride`` -- and nothing above checks that.  Without
        # this the emitted ``cute.arch.load`` FAULTS at runtime with ``CUDA error:
        # misaligned address`` on any sliced input whose row stride is not a
        # multiple of the issue width.  MEASURED on a 64x512 bf16 ``base[:, :512]``
        # of a ``(64, 512 + pad)`` base: 5 of 12 ``(pad, V)`` cells faulted, and the
        # failing set is exactly ``gcd(row_stride, issue) < issue``.
        #
        # The ROLLED (``LoopedReductionStrategy``) path already narrows through
        # ``tv_layout.legal_vec`` with ``row_stride_elems=gcd(strides)`` and handles
        # the identical input at every pad; this branch simply never called it.
        #
        # Four things here are load-bearing:
        #
        # * DECLINE, never narrow.  The outer lane loop's trip count ``EPT / V`` is
        #   fixed in ``CuteNDTileStrategy.__init__`` and this site cannot reshape
        #   it, so a narrowed load would visit fewer elements than the loop assumes
        #   (bug class 1 exactly).  Declining is safe: the scalar enumeration
        #   ``lane_base + vec_lane`` is already complete.
        # * ``gcd`` over the strides, NOT ``min``.  "stride is a multiple of v" is a
        #   conjunction over the participating dims and ``gcd`` is its exact fold.
        #   ``min`` is the bug ``af73ba169`` fixed on the rolled path (a contiguous
        #   output's stride masked a sliced input's and the DSL ICEd).
        # * ``s > 1`` drops the stride-1 lane dim (never a constraint) and a
        #   0-stride broadcast dim from ``.expand()``.
        # * The ISSUE width, not the logical ``V``.  bf16/fp16 ``V=8`` emits
        #   ``tile_unroll_split2`` -- TWO 4-element ``cute.arch.load``s -- so it
        #   needs only 4-element alignment, which is why ``row_stride=516, V=8``
        #   runs correctly and must keep vectorising.  ⚠ fp8 ``V=8`` is NOT split:
        #   it is a single packed ``Uint64``, i.e. 8 bytes, so its requirement
        #   really is 8 elements.  Halving unconditionally would under-clamp it.
        issue_width = (
            vec_width // 2
            if vec_width == 8 and not _cute_is_byte_packed(tensor.dtype)
            else vec_width
        )
        # Symbolic strides are dropped rather than treated as unaligned, matching
        # ``LoopedReductionStrategy._cute_layout_participants``: this site must not
        # invent a new decline class for every dynamic-shape kernel.
        stride_cand = [
            abs(int(s))
            for d in range(tensor.ndim)
            if isinstance(s := tensor.stride(d), int) and s > 1
        ]
        # ``numel`` is an ``int``/``sympy.Integer`` by construction here -- the
        # ``known_multiple`` check above returns False for anything symbolic, so a
        # non-static extent has already declined.  (``env.size_hint`` would
        # ``assert isinstance(n, int)`` on a ``sympy.Integer``.)
        if (
            legal_vec(
                int(numel),
                tensor.element_size() * 8,
                row_stride_elems=math.gcd(*stride_cand) if stride_cand else None,
            )
            < issue_width
        ):
            return None
        # Record the index_exprs position of the stride-1 lane axis so the
        # hoist substitutes the per-lane base there.  Row-major lhs loads
        # use the last position; a column-major rhs (K-major ``y``) uses
        # position 0.
        pos_by_block = getattr(strategy, "_cute_lane_axis_pos_by_block", None)
        if not isinstance(pos_by_block, dict):
            pos_by_block = {}
            # pyrefly: ignore [missing-attribute]
            strategy._cute_lane_axis_pos_by_block = pos_by_block
        pos_by_block[inner_block_id] = lane_axis_pos
        # fp8 loads a packed Uint64 (V=8) / Uint32 (V=4) in the regular
        # ``tile_unroll`` path — no ``VectorType`` so no V=8 ICE, hence no
        # split2 needed.  bf16/fp16 V=8 still needs the 2x V=4 split.
        if vec_width == 8 and not _cute_is_byte_packed(tensor.dtype):
            return vec_width, inner_block_id, "tile_unroll_split2", lane_axis_pos
        return vec_width, inner_block_id, "tile_unroll", lane_axis_pos
    return None


@_decorators.codegen(load, "cute")
def _(state: CodegenState) -> object:
    tensor = state.proxy_arg(0)
    subscript = state.proxy_arg(1)
    assert isinstance(subscript, (list, tuple))
    ast_subscript = state.ast_args[1]
    assert isinstance(ast_subscript, (list, tuple))
    extra_mask = state.ast_args[2]
    assert isinstance(extra_mask, (type(None), ast.AST))

    if isinstance(tensor, tuple):
        stack_tensor_ast = state.ast_args[0]
        assert isinstance(stack_tensor_ast, tuple)
        assert len(stack_tensor_ast) == 2
        tensor_like_ast, dev_ptrs_ast = stack_tensor_ast
        assert isinstance(dev_ptrs_ast, ast.AST)
        tensor_like, dev_ptrs = tensor
        offset_expr = _cute_stack_tensor_offset_expr(
            state,
            tensor_like,
            [*subscript],
            ast_subscript,
        )
        backend = CompileEnvironment.current().backend
        target_dtype = backend.dtype_str(tensor_like.dtype)
        ptr_expr = _cute_stack_tensor_pointer_expr(
            target_dtype, dev_ptrs_ast, offset_expr
        )
        load_expr = f"({ast.unparse(ptr_expr)}).load()"
        mask_expr = _cute_stack_tensor_mask_expr(
            state,
            tensor_like,
            dev_ptrs,
            [*subscript],
            extra_mask,
        )
        if tensor_like.dtype is torch.bool:
            load_expr = f"({load_expr} != cutlass.Uint8(0))"
            if mask_expr is None:
                return expr_from_string(load_expr)
            return expr_from_string(
                f"({load_expr} if {mask_expr} else cutlass.Boolean(0))"
            )
        if mask_expr is None:
            return expr_from_string(load_expr)
        return expr_from_string(f"({load_expr} if {mask_expr} else {target_dtype}(0))")
    if not isinstance(tensor, torch.Tensor):
        raise exc.BackendUnsupported("cute", f"load tensor type: {type(tensor)}")

    _log_cute_layout(state, "load")

    from ...language import tile_index

    tensor_node = state.fx_node.args[0] if state.fx_node is not None else None
    if (
        isinstance(tensor_node, torch.fx.Node)
        and tensor_node.op == "call_function"
        and tensor_node.target == tile_index
    ):
        env = CompileEnvironment.current()
        block_id = env.get_block_id(tensor.size(0))
        if block_id is None:
            raise exc.BackendUnsupported("cute", "tile_index load block id")
        index_var = _cute_active_index_var(state, block_id)
        if index_var is None:
            raise exc.BackendUnsupported("cute", "inactive tile_index load")
        for idx in subscript:
            if idx is None or idx == slice(None):
                continue
            raise exc.BackendUnsupported(
                "cute", f"tile_index load index type: {type(idx)}"
            )
        return expr_from_string(index_var)

    cute_state = state.device_function.cute_state
    if cute_state.suppress_root_lane_loops or (
        state.fx_node is not None
        and cute_state.is_collective_handled_load(state.fx_node.name)
    ):
        zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
        return expr_from_string(f"{zero}(0)")

    packed_affine_lhs = _maybe_codegen_cute_packed_affine_lhs_load(
        state, tensor, subscript, extra_mask
    )
    if packed_affine_lhs is not None:
        return packed_affine_lhs

    packed_rhs_load = _maybe_codegen_cute_packed_rhs_load(
        state, tensor, subscript, extra_mask
    )
    if packed_rhs_load is not None:
        return packed_rhs_load

    if _is_cute_affine_range_load_for_store(state, subscript, ast_subscript):
        zero = _cute_scalar_storage_dtype(tensor.dtype)
        return expr_from_string(f"{zero}(0)")
    if _is_cute_strided_slice_load_for_store(state, tensor, subscript):
        zero = _cute_scalar_storage_dtype(tensor.dtype)
        return expr_from_string(f"{zero}(0)")

    tensor_name = state.device_function.tensor_arg(tensor).name
    # ``slice_block_ids`` is filled in by the call below and consumed by
    # ``_cute_vector_load_ctx``: it is the ``{index_exprs position: block_id}``
    # binding the indexer resolved for each bare-slice axis.  It MUST come from
    # the same call that produced ``index_exprs`` (see that function's docstring
    # on why this is an out-parameter and not a module global).
    slice_block_ids: dict[int, int] = {}
    index_exprs = _cute_index_exprs(
        state,
        subscript,
        ast_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
        slice_block_ids=slice_block_ids,
    )
    # ``for_load``: this is the codegen for ``hl.load`` / a tile read.  It produces value
    # selects (``x if mask else 0``) and, through the vec hoists' inline anchor guard,
    # address clamps --
    # never an ``if`` around a store.  So a bounds term the strategy can prove vacuous for
    # every launched thread is droppable here; see ``TileStrategy.load_mask_var``.
    mask_expr = _cute_combined_mask(
        state,
        subscript,
        extra_mask,
        tensor=tensor,
        include_tensor_index_masks=False,
        for_load=True,
    )
    vec_ctx = _cute_vector_load_ctx(
        state, tensor, subscript, index_exprs, extra_mask, slice_block_ids
    )
    if vec_ctx is not None:
        vec_width, vec_block_id, vec_mode, lane_axis_pos = vec_ctx
        from ..reduction_strategy import ReductionStrategy

        loops = state.codegen.active_device_loops.get(vec_block_id)
        strategy = loops[-1].strategy if loops else None
        if vec_mode == "tv":
            # THE TV path.  ``partition_S`` off the reduction's one
            # ``get_slice``; the store leg's ``partition_D`` comes off the same
            # slice in ``_codegen_cute_store``, which is what makes the index
            # and the access width structurally impossible to disagree.
            #
            # The assert is on the CAPABILITY, not the class: ``_cute_vector_load_ctx``
            # only returns ``"tv"`` after ``_cute_tv_site_eligible`` passed, which
            # requires exactly these fields, so a failure here means the two
            # predicates drifted -- which is worth crashing over.  A strategy merely
            # LACKING the capability never reaches this branch.
            # ⚠ CAPABILITY ONLY, no ``isinstance`` -- see the note at
            # ``_cute_tv_partition_hoist``: pairing the class with the capability makes
            # this crash for precisely the non-reduction strategies the query admits.
            assert strategy.cute_tv_capable()  # pyrefly: ignore [missing-attribute]
            plan = strategy._cute_tv_plan
            assert plan is not None and plan.vec == vec_width, (
                "TV load site reached with a width the plan does not own: "
                f"plan={plan.describe() if plan else None} site_vec={vec_width}"
            )
            frag_var = _cute_tv_partition_hoist(
                state,
                strategy,
                plan,
                tensor,
                tensor_name,
                _cute_tv_row_index_expr(state, tensor, subscript, index_exprs),
                is_store=False,
            )
            # The per-element read.  ``vec_lane_var`` is the constexpr V-loop's
            # target, so the fragment subscript is a compile-time constant and
            # the existing scalar cast/mul/accumulate pipeline is unchanged --
            # including the ``if mask else 0`` gate the caller appends, so
            # ``_mask_to``'s identity rescue (LEDGER E005 item 4) still sits
            # between this load and the combine.
            load_expr = f"{frag_var}[{_cute_tv_vec_lane_var(strategy)}]"
        elif vec_mode == "unroll":
            # Register (or reuse) a hoisted U16 vec load for this (tensor,
            # base_index) pair, then return ``hoist_var[vi].bitcast(dtype)``
            # so the existing scalar pipeline sees a scalar of the original
            # dtype.
            #
            # ``_cute_vector_load_ctx`` returns ``"unroll"`` only after checking
            # ``_cute_lane_base_index_var`` and ``_cute_lane_body`` are both set,
            # which is precisely what the hoist dereferences -- so this is a
            # coherence assert between two predicates, not a class filter.
            assert isinstance(strategy, ReductionStrategy)
            load_expr = _cute_register_unroll_vec_hoist(
                state,
                strategy,
                vec_block_id,
                tensor,
                tensor_name,
                index_exprs,
                vec_width,
                subscript,
            )
        elif vec_mode == "tile_unroll":
            # Same hoist protocol as ``LoopedReductionStrategy``'s
            # ``unroll`` mode but for ``CuteNDTileStrategy`` lane loops.
            from ..tile_strategy import BlockSizeTileStrategy

            assert isinstance(strategy, BlockSizeTileStrategy)
            load_expr = _cute_register_tile_unroll_vec_hoist(
                state,
                strategy,
                vec_block_id,
                tensor,
                tensor_name,
                index_exprs,
                vec_width,
                subscript,
            )
        else:
            assert vec_mode == "tile_unroll_split2"
            # V=8 fp16/bf16: emit two back-to-back ``cute.arch.load(...,
            # V=4)`` calls (lanes 0-3 and 4-7).  Works around the CuTe
            # DSL's ``nvvm.load.ext`` ICE on V=8 while still issuing the
            # full LDG.128 of bytes-per-thread-per-outer-iter.
            from ..tile_strategy import BlockSizeTileStrategy

            assert isinstance(strategy, BlockSizeTileStrategy)
            load_expr = _cute_register_tile_unroll_vec_hoist_split2(
                state,
                strategy,
                vec_block_id,
                tensor,
                tensor_name,
                index_exprs,
                vec_width,
                subscript,
            )
    else:
        load_expr = _cute_scalar_load_expr(tensor_name, index_exprs, tensor.dtype)
    if tensor.dtype is torch.bool:
        load_expr = f"({load_expr} != cutlass.Uint8(0))"
        if mask_expr is None:
            return expr_from_string(load_expr)
        return expr_from_string(f"({load_expr} if {mask_expr} else cutlass.Boolean(0))")
    if state.fx_node is not None and _cute_load_feeds_sort_or_scan(state.fx_node):
        from .indexing import CuteSortableLoad

        tensor_dim = 0
        sort_index_pos = -1
        for idx in subscript:
            if idx is None:
                continue
            if tensor_dim == tensor.ndim - 1:
                sort_index_pos = tensor_dim
                break
            tensor_dim += 1
        if sort_index_pos < 0:
            raise exc.BackendUnsupported("cute", "sort/topk input rank")
        sortable_load = CuteSortableLoad(
            expr=expr_from_string(
                load_expr
                if mask_expr is None
                else f"({load_expr} if {mask_expr} else {_cute_scalar_storage_dtype(tensor.dtype)}(0))"
            ),
            tensor_name=tensor_name,
            index_exprs=tuple(index_exprs),
            sort_index_pos=sort_index_pos,
            mask_expr=mask_expr,
            dtype=tensor.dtype,
        )
        state.fx_node.meta["cute_sortable_load"] = sortable_load
        return sortable_load.expr
    if mask_expr is None:
        return expr_from_string(load_expr)
    zero = _cute_scalar_storage_dtype(tensor.dtype)
    return expr_from_string(f"({load_expr} if {mask_expr} else {zero}(0))")
