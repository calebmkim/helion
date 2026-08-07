from __future__ import annotations

import ast
import logging
import operator
import os
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._inductor import ir
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch._prims_common import get_computation_dtype

from .. import exc
from .._compat import shape_env_size_hint
from .ast_extension import create
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .compile_environment import CompileEnvironment
from .cute.layout import LayoutTag as _CuteLayoutTag
from .cute.layout_propagation import META_KEY as _CUTE_LAYOUT_META_KEY
from .cute.ragged_tail import assert_vec_divides_extent
from .cute.ragged_tail import ragged_tile_admissible
from .cute.ragged_tail import rounded_extent
from .cute.ragged_tail import tile_granularity
from .cute.tv_layout import ROW_RESIDENCY_GMEM
from .cute.tv_layout import ChunkTVPlan
from .cute.tv_layout import TVParticipants
from .cute.tv_layout import build_tv_plan
from .cute.tv_layout import emit_lane_base_for
from .cute.tv_layout import max_cluster_n_for_arch
from .device_function import find_block_size_symbols
from .host_function import HostFunction
from .inductor_lowering import ReductionLowering
from .inductor_lowering import install_inductor_kernel_handlers
from .tile_strategy import CompactedShape
from .tile_strategy import DeviceGridState
from .tile_strategy import DeviceLoopState
from .tile_strategy import LoopDimInfo
from .tile_strategy import PersistentReductionState
from .tile_strategy import ThreadAxisTracker
from .tile_strategy import TileStrategy
from .tile_strategy import _to_sympy

if TYPE_CHECKING:
    from .device_function import DeviceFunction
    from .inductor_lowering import CodegenState

log = logging.getLogger(__name__)


def _dtype_str(dtype: torch.dtype) -> str:
    return CompileEnvironment.current().backend.dtype_str(dtype)


# Reduction ops whose combine ACCUMULATES, i.e. repeatedly folds many elements
# into one running value, so the running value can exceed the range of the input
# dtype even when the mathematically-correct answer fits.  These are the ops that
# need a widened integer accumulator.  ``max``/``min`` are deliberately absent:
# they only ever SELECT one of their inputs, so the running value is always an
# element of the input and can never leave the input's range -- widening them
# would cost registers/SMEM and buy nothing.  Indexed reductions (``argmin`` /
# ``argmax``) are also absent: their *index* is already ``int64``, and the *value*
# they compare is a select, not an accumulation.
_ACCUMULATING_REDUCTION_TYPES = frozenset({"sum", "prod"})


def widen_integer_acc_dtype(
    reduction_type: str, base_dtype: torch.dtype
) -> torch.dtype:
    """Widen ``base_dtype`` to ``int64`` when it is an integral accumulator of an
    ACCUMULATING reduction.

    ``torch.sum`` / ``torch.prod`` promote every integral (and ``bool``) input to
    ``int64`` internally, so a reduction that accumulates in the input's own
    integer width diverges from the reference by silently wrapping
    (``_redfix/01_BUGS.md`` class 5).  Measured with torch: ``sum``/``prod`` over
    ``bool``/``int8``/``uint8``/``int16``/``int32``/``int64`` all give ``int64``,
    while ``amax``/``amin`` keep the input dtype.

    A no-op for float bases (their widening is ``get_computation_dtype``'s job)
    and for selecting ops, whose running value is always one of their inputs and
    therefore cannot leave the input's range.
    """
    if (
        reduction_type in _ACCUMULATING_REDUCTION_TYPES
        and not base_dtype.is_floating_point
        and not base_dtype.is_complex
        and base_dtype != torch.int64
    ):
        return torch.int64
    return base_dtype


def reduction_acc_dtype(reduction_type: str, input_dtype: torch.dtype) -> torch.dtype:
    """The dtype a reduction's accumulator must use.

    The accumulator is **not** a memory object: it does not inherit the input
    tensor's (or a copy atom's) dtype.  It is chosen to match torch's reference
    semantics for ``reduction_type`` over ``input_dtype``:

    * float inputs -> ``get_computation_dtype``, i.e. fp16/bf16 accumulate in
      fp32 (the pre-existing, already-correct behaviour);
    * integral inputs of ``sum``/``prod`` -> ``int64`` (see
      :func:`widen_integer_acc_dtype`);
    * integral inputs of ``max``/``min``/``argmax``/``argmin`` -> unchanged.

    ``get_computation_dtype`` alone is not enough: it is the identity on
    integers (``int32 -> int32``), which is exactly the class-5 bug.
    """
    return widen_integer_acc_dtype(reduction_type, get_computation_dtype(input_dtype))


def _cute_shared_memory_budget_bytes() -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    default_shared = int(props.shared_memory_per_block)
    optin_shared = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
    return max(default_shared, optin_shared)


def _log_cute_reduction_layout(state: CodegenState) -> None:
    """Log the CuTe layout annotation for the current reduction node, if any."""
    if state.fx_node is None:
        return
    constraint = state.fx_node.meta.get(_CUTE_LAYOUT_META_KEY)
    if constraint is None or constraint.input_layout is None:
        return
    layout = constraint.input_layout
    log.debug(
        "cute reduction %s: layout tag=%s thread=%s value=%s",
        state.fx_node.name,
        layout.tag.value,
        layout.thread_shape,
        layout.value_shape,
    )


def _reduction_threads_from_annotation(state: CodegenState) -> int | None:
    """Read reduction thread count from the layout annotation, if available.

    Returns the thread count from the layout annotation when the node has
    a REDUCTION-tagged layout with a concrete integer thread count.
    Falls back to ``None`` so the caller can use ``reduction_threads_hint()``.
    """
    if state.fx_node is None:
        return None
    constraint = state.fx_node.meta.get(_CUTE_LAYOUT_META_KEY)
    if constraint is None or constraint.input_layout is None:
        return None
    layout = constraint.input_layout
    if layout.tag != _CuteLayoutTag.REDUCTION:
        return None
    nt = layout.num_threads()
    if isinstance(nt, int) and nt > 0:
        return nt
    return None


def _cute_reduction_smem_bytes(num_elements: int, dtype: torch.dtype) -> int:
    return num_elements * torch.empty((), dtype=dtype).element_size()


_CUTE_LOOPED_REDUCTION_MAX_ELEMENTS_PER_THREAD = 256
_CUTE_WARP_REDUCTION_THREADS = 32

# ⭐ ``_CUTE_STAGE_SMEM_MAX_BYTES = 64 * 1024`` USED TO LIVE HERE.  It is now the
# ``cute_stage_smem_kb`` CONFIG KNOB (``CuteStageSmemKbSpec``), whose default ladder
# ``tv_layout.stage_smem_kb_for`` returns the same 64 for every ``n`` -- so the emitted
# kernel is unchanged and the promotion is a pure reachability change.
#
# THE MEASUREMENT THE CONSTANT CARRIED, kept because it is the knob's whole
# justification.  rms_norm bf16, cold-L2 CUDA-graph, ratio = quack_ms/helion_ms, one
# process per arm (the same process serves a cached kernel and hides the effect
# entirely -- that cost one round):
#
#   staged tile | cell                  | reload=None | reload="smem"
#   ------------|-----------------------|-------------|---------------
#      32 KB    | N=4096  bs=4  rl=1024 |    0.867    |  0.917   WIN
#      64 KB    | N=4096  bs=8  rl=1024 |    0.999    |  1.067   WIN
#      64 KB    | N=8192  bs=4  rl=1024 |    0.960    |  1.033   WIN
#      64 KB    | N=8192  bs=4  rl=2048 |    1.016    |  1.077   WIN
#      64 KB    | N=16384 bs=2  rl=2048 |    0.931    |  1.005   WIN
#     128 KB    | N=8192  bs=8  rl=1024 |    0.987    |  0.743   LOSS
#     128 KB    | N=16384 bs=4  rl=1024 |    0.912    |  0.707   LOSS
#     128 KB    | N=16384 bs=4  rl=2048 |    1.002    |  0.722   LOSS
#
# Every tile <= 64 KB wins (by 2-8%); every tile at 128 KB loses (by 24-28%).  The
# mechanism is occupancy: ncu ``launch__occupancy_limit_shared_mem`` is 6 blocks/SM
# without staging and **1** with a 128 KB tile, on a 232 KB/CTA device.  64 KB admits 3
# blocks, which is enough to keep the DRAM pipe fed.
#
# ⚠ AND THAT IS AN EXCHANGE RATE, WHICH IS WHY IT IS A KNOB.  "Enough resident CTAs to
# keep the DRAM pipe fed" depends on how much work each CTA does, so the winning budget
# is a function of the row width -- and the table above was measured at N=4096..16384
# only, i.e. exactly the band where the frozen table stages.  MEASURED at the wide end,
# the constant is refusing requests rather than sizing them: ``cross_entropy``
# 8192x100000 asks for 196 KB and is declined, ``cross_entropy`` 32768x32768 is admitted
# at EXACTLY 64 KB.
#
# Why the budget is still ABSOLUTE and not a fraction of
# ``_cute_shared_memory_budget_bytes()``: class-9 item S1 records that helion's reduction
# SMEM check is PER reduction while ``alloc_smem`` SUMS across inlined call sites.  A
# kernel with two reductions (layer_norm, cross_entropy) allocates two tiles against one
# budget, so 64 KB each is already 128 KB total -- exactly the losing size.  See
# ``cute_stage_feasible``, which charges the budget per *kernel* for that reason.


def cute_looped_reduction_block_size(size_hint: int, max_threads: int) -> int:
    """Pick the default CuTe loop chunk for reductions wider than one warp."""
    return min(size_hint, max_threads * _CUTE_LOOPED_REDUCTION_MAX_ELEMENTS_PER_THREAD)


def cute_live_reduction_threads(max_threads: int) -> int:
    # Persistent reductions on CuTe can recruit threads beyond a single warp
    # (cross-warp combining uses _cute_grouped_reduce_shared_two_stage). The
    # autotuner / config_spec keeps the size_hint <= max_threads case here so
    # no synthetic lane wrap is required.
    return max_threads


def _strategies_concurrent_with_block(
    tile_dispatch: object,
    block_index: int,
) -> list[TileStrategy]:
    """Return strategies that can co-execute with reduction ``block_index``.

    Drops reduction strategies that live in a control-flow branch mutually
    exclusive with ``block_index``'s branch so the per-block thread budget is
    not over-counted (CuTe branch-by-pid kernels). Outside that pattern (no
    branch paths) this returns every strategy unchanged.
    """
    from .device_ir import DeviceIR
    from .host_function import HostFunction

    strategies = list(getattr(tile_dispatch, "strategies", []))
    device_ir = HostFunction.current().device_ir
    red_paths = device_ir.reduction_block_id_branch_paths()
    own_paths = red_paths.get(block_index)
    if not own_paths:
        return strategies
    own_path = own_paths[0]
    result: list[TileStrategy] = []
    for strategy in strategies:
        other_path = None
        for other_block in strategy.block_ids:
            paths = red_paths.get(other_block)
            if paths:
                other_path = paths[0]
                break
        if DeviceIR.branch_paths_mutually_exclusive(own_path, other_path):
            continue
        result.append(strategy)
    return result


def _cute_vec_kernel_mode() -> str:
    """Return ``"vec"`` when all reduction-feeding loads are vec-eligible
    without a degrading dtype cast (i.e. fp32 -> fp32 pipeline), ``"unroll"``
    when at least one load uses the bf16/fp16 -> fp32 cast pattern that the
    CuTe DSL's ``Float32(vec)`` constructor would silently scalarise, or
    ``"none"`` when there's no looped reduction at all.

    ``"vec"`` lets the strategy emit a single ``cute.arch.load(..., V)`` +
    V-fold per lane iter.  ``"unroll"`` falls back to a constexpr V-loop
    around per-element scalar loads — the CUTLASS DSL cannot iterate the
    elements of a bf16/fp16 vector without crashing during compile.
    """
    from .host_function import HostFunction
    from .host_function import NoCurrentFunction

    try:
        hf = HostFunction.current()
    except NoCurrentFunction:
        return "none"
    if hf._device_ir is None:
        return "none"
    cast_targets = {
        "convert_element_type.default",
        "convert_element_type",
        "_to_copy.default",
        "_to_copy",
    }
    from ..language import memory_ops as _memory_ops

    load_target = _memory_ops.load
    has_cast = False
    for graph_info in hf.device_ir.graphs:
        for node in graph_info.graph.nodes:
            if node.target is not load_target:
                continue
            for user in node.users:
                target_name = getattr(user.target, "__name__", "") or ""
                if target_name in cast_targets:
                    has_cast = True
                    break
            if has_cast:
                break
        if has_cast:
            break
    return "unroll" if has_cast else "vec"


def _block_has_indexed_reduction(fn: DeviceFunction, block_index: int) -> bool:
    """Return True when ``block_index`` is the reduction axis of any
    argmin/argmax in the device IR.

    Populated by :meth:`DeviceIR.register_rollable_reductions` so this is
    just a set lookup on ConfigSpec.

    Used to cap CuTe reduction strategies' thread_count at the warp width
    when an indexed reduction is present — CuTe argreduce uses
    cute.arch.warp_reduction which is only correct for threads_in_group<=32.
    """
    env = CompileEnvironment.current()
    return block_index in env.config_spec.cute_indexed_reduction_block_ids


def _cute_tv_alias_key(host: HostFunction, val: torch.Tensor) -> object:
    """A comparable identity for the memory ``val`` names.

    Prefers the host buffer name, so two fx nodes that reach the same host
    tensor compare equal even if they are distinct ``_host_tensor`` nodes in
    different graphs (which they are: the rolled reduction body is a *copy* of
    the root's nodes).  Falls back to object identity for device-internal
    temporaries, which have no host name but are the same fake tensor object
    everywhere they appear.
    """
    origin = host.tensor_to_origin.get(val)
    name = origin.root_rw_name() if origin is not None else None
    return name if name is not None else id(val)


def _cute_tv_subscript_key(subscript: object) -> object:
    """A comparable identity for a load/store subscript.

    Two accesses with equal keys address the SAME elements of the same tensor:
    same row (or the same broadcast row), same trailing extent.  That is the
    predicate store-to-load forwarding needs, and it is what separates the three
    shapes that must keep declining (a different row, a column sub-slice, a
    scalar element) from the one that can forward.

    A ``SymInt`` is canonicalised through its block id where it has one, so the
    key does not depend on the ``SymInt`` object's identity -- the rolled
    reduction body is a *copy* of the root graph's nodes, so the same access can
    appear as distinct objects in different graphs.
    """
    if not isinstance(subscript, (list, tuple)):
        return None
    env = CompileEnvironment.current()
    parts: list[object] = []
    for idx in subscript:
        if idx is None:
            parts.append(("none",))
        elif isinstance(idx, slice):
            parts.append(("slice", str(idx.start), str(idx.stop), str(idx.step)))
        elif isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
            parts.append(
                ("blk", block_id) if block_id is not None else ("sym", str(idx))
            )
        else:
            parts.append(("other", str(idx)))
    return tuple(parts)


def _cute_tv_store_node_forwardable(node: torch.fx.Node) -> bool:
    """Can a TV ``cute.copy`` store be emitted for this ``hl.store`` node?

    ⭐ Condition 4 of :func:`_cute_tv_forwardable_raw_keys`, and the reason it is a
    separate function: forwarding makes the LOAD read the STORE's fragment, so it
    is only sound when the store actually emits one.  Conditions 1-3 are decided
    from the IR; this is decided by the emitter, and the gap between the two was a
    silent wrong answer (see that function's docstring for the measurement).

    ⚠ **This is deliberately a conservative POSITIVE check, not a list of known
    bad spellings.**  It vouches for the shapes the TV store path provably
    handles, and everything else is refused -- so a store-leg decline added to
    ``_maybe_codegen_cute_tv_store`` or ``_cute_tv_site_eligible`` in future
    degrades this to "do not forward" (i.e. the pre-B3 blanket decline, which is
    correct but unvectorised) rather than to a stale read.  Enumerating the bad
    cases instead would be correct only for the cases enumerated today.

    ⚠ The emitter's own eligibility test needs a live ``strategy`` and a
    ``CodegenState``, neither of which exists during this IR walk, so it cannot be
    called here.  What is checkable at IR time is the store's ARGUMENT shape, and
    that is where the reachable declines live: ``extra_mask`` (``args[3]`` /
    ``kwargs``, which ``_maybe_codegen_cute_tv_store`` refuses outright) and a
    subscript naming fewer axes than the tensor has.
    """
    if len(node.args) > 3 and node.args[3] is not None:
        return False  # extra_mask positionally
    if node.kwargs.get("extra_mask") is not None:
        return False  # extra_mask by keyword
    val = (
        node.args[0].meta.get("val")
        if isinstance(node.args[0], torch.fx.Node)
        else None
    )
    subscript = node.args[1]
    if not isinstance(val, torch.Tensor) or not isinstance(subscript, (list, tuple)):
        return False
    # Mirrors ``_cute_tv_site_eligible``'s rank and subscript-arity conditions,
    # which are the two it can decide without a strategy.  A store the emitter
    # would refuse on a stride or copy-axis ground still reaches it, and is then
    # refused there -- but by then the plan exists, so those must be caught by the
    # arity/rank test or by the store simply not being the trailing-axis shape.
    if val.ndim not in (1, 2):
        return False
    return len([idx for idx in subscript if idx is not None]) == val.ndim


def _cute_tv_forwardable_raw_keys() -> frozenset[object] | None:
    """Classify every store-then-load RAW: which can FORWARD, or None if any cannot.

    ⭐ This is what replaced B3's blanket decline.  The hazard
    (:func:`_cute_tv_has_store_then_load_alias`) is real, but declining the whole
    TV path is far coarser than it: one RAW anywhere in any graph used to kill
    vectorisation for EVERY tensor in the reduction -- including tensors never
    stored to -- and took the sweep cache, the SMEM staging and the cluster down
    with it.

    ⭐ THE VALUE THE LOAD WANTS IS ALREADY IN A REGISTER.  The store leg fills
    ``_tv_frag_D[vi]`` inside the constexpr V-loop; a load sited after it in
    program order can read that same element instead of re-reading GMEM.  Both
    legs partition through the SAME ``get_slice`` (that is the TV plan's core
    invariant), so ``partition_D``'s fragment and ``partition_S``'s fragment
    provably address identical elements.  ⭐ No barrier is needed, which is the
    non-obvious part: the TV slice is per-thread, so the dependence is
    intra-thread.  The result is one FEWER gmem read per lane than even a
    correct non-forwarding implementation -- an optimisation the bug concealed.

    Returns
    -------
    ``frozenset()``
        No RAW at all.  The plan is unconstrained; nothing forwards.
    a non-empty ``frozenset``
        Every RAW in the IR forwards, and these are the alias keys
        (:func:`_cute_tv_alias_key`) whose load leg must read the store's
        fragment.
    ``None``
        Some RAW cannot forward.  The caller declines the plan, exactly as
        before -- so every kernel that declines today still declines.

    THE THREE CONDITIONS A RAW MUST MEET TO FORWARD, and what each rules out:

    1. **Identical coverage.**  Every access of the tensor -- load and store --
       shares one :func:`_cute_tv_subscript_key`.  A store to ``buf[tile_m, :]``
       and a load of ``buf[0, :]`` differ in the ROW, so the store's fragment is
       not the load's data; ``buf[tile_m, 0:N//2]`` and ``buf[tile_m, 0]`` differ
       in the trailing EXTENT, so the load needs the *merge* of stored and
       pre-existing data, which one fragment cannot hold.  ⚠ Coverage cannot be
       decided from the alias key alone: :func:`_cute_tv_alias_key` compares HOST
       BUFFER NAMES and all four of those shapes share one name.
    2. **Store-first.**  No load of the tensor may precede its first store.  This
       is not about the dependence -- it is about *nameability*: the emitted
       fragments are cached per ``(tensor_name, "S" | "D")``, so a pre-store load
       and a post-store load of one tensor are indistinguishable at the emission
       site and the second would silently inherit the first's (stale) fragment.
       Declining keeps that shape on today's path.  ⇒ ``v = x[t,:]`` ...
       ``x[t,:] = f(v)`` ... ``sum(x[t,:])`` still declines.
    3. **No cycle.**  Guaranteed by 2 rather than checked: with no pre-store load
       of the tensor, the stored value cannot depend on a forwarded read of it.
       fx node order is program order, so a value cannot flow backwards from the
       later load into the earlier store.
    4. ⭐ **The store leg must itself be TV-emittable.**  Forwarding reads the
       *store's* fragment, so it presupposes the store emits one.  Conditions 1-3
       are all properties of the IR; this one is a property of EMISSION, and the
       two are decided in different places -- which is exactly how a wrong answer
       got in.  MEASURED before this condition existed:
       ``hl.store(buf, [t, :], 1.0, extra_mask=...)`` followed by
       ``sum(buf[t, :])`` returned every row wrong (a 4x2048 bf16 case gave
       ``[0, 0, 0, 0]`` where the correct answer is ``[2048, 0, 2048, 0]``),
       because ``_maybe_codegen_cute_tv_store`` declines on ``extra_mask`` while
       conditions 1-3 saw only the subscript and let the plan proceed.  The load
       leg then got a ``partition_S`` with its ``cute.copy`` hoisted ABOVE the
       loop and the store degraded to an in-loop scalar pointer store, so the
       reduction summed STALE PRE-STORE GMEM: B3's original bug, reached through a
       spelling the coverage key cannot see.

       ⚠ **NOT enumerated as "extra_mask, and the others".**  ``extra_mask`` is
       the cheapest instance, not the only one: ``_cute_tv_site_eligible``
       independently declines a store on tensor rank, a non-unit trailing stride,
       a subscript that does not name the copy axis, and a 2-D both-axes-sliced
       access.  Any of them reproduces the same stale read.  So the predicate asks
       the store-side question *positively* -- "can a TV store be emitted for this
       node at all?" -- via :func:`_cute_tv_store_node_forwardable`, and anything
       it cannot vouch for is disqualified.  A new store-leg decline added later
       is therefore safe by default (it disqualifies, i.e. falls back to the
       blanket decline) instead of silently becoming a wrong answer.

    Scanned per graph, and a key must satisfy the conditions in EVERY graph that
    touches it (the rolled body duplicates the root's nodes, so one access is
    seen more than once and a per-graph verdict would be ambiguous at emission).
    """
    from ..language import memory_ops as _memory_ops

    host = HostFunction.current()
    if host._device_ir is None:
        return frozenset()
    raw_keys: set[object] = set()
    # alias key -> the one subscript key every access of it uses, or None once a
    # second, different subscript key has been seen.
    coverage: dict[object, object] = {}
    disqualified: set[object] = set()
    for graph_info in host.device_ir.graphs:
        stored: set[object] = set()
        for node in graph_info.graph.nodes:
            if node.target not in (_memory_ops.load, _memory_ops.store):
                continue
            if len(node.args) < 2:
                continue
            tensor_node, subscript = node.args[0], node.args[1]
            if not isinstance(subscript, (list, tuple)):
                continue
            val = (
                tensor_node.meta.get("val")
                if isinstance(tensor_node, torch.fx.Node)
                else tensor_node
            )
            if not isinstance(val, torch.Tensor):
                continue
            key = _cute_tv_alias_key(host, val)
            sub_key = _cute_tv_subscript_key(subscript)
            if key in coverage:
                if coverage[key] != sub_key:
                    disqualified.add(key)  # condition 1: coverage differs
            else:
                coverage[key] = sub_key
            if node.target is _memory_ops.store:
                # Condition 4: a store the TV path cannot emit has no fragment for
                # a later load to forward from.  Disqualify rather than decline
                # here, so the verdict is per-KEY: an unrelated tensor's RAW in
                # the same graph can still forward.
                if not _cute_tv_store_node_forwardable(node):
                    disqualified.add(key)
                stored.add(key)
            else:
                if key in stored:
                    raw_keys.add(key)
                else:
                    # A load with no preceding store, in this graph: condition 2.
                    disqualified.add(key)
    if raw_keys & disqualified:
        return None
    return frozenset(raw_keys)


def _cute_unduplicatable_producer_in_graph(state: CodegenState) -> bool:
    """True when this reduction's graph contains a producer that must run EXACTLY ONCE.

    ⭐⭐ THE SAME QUESTION ``_contains_unduplicatable_op`` ANSWERS, ASKED AT LOWERING AND BY
    IDENTITY.  A lane-reduce lowering that re-materialises a shared producer into each
    sibling lane loop is sound for a pure ``gmem -> rmem`` load and UNSOUND for a matmul or a
    cross-thread collective: re-running one duplicates a barrier.  So the emitter has to know
    whether such a producer is present.

    ⛔ WHAT IT REPLACES, AND WHY THAT IS A CORRECTNESS IMPROVEMENT RATHER THAN A REFACTOR.
    The AST-side predicate does ``ast.unparse(stmt)`` and then a SUBSTRING match against five
    hard-coded names -- i.e. a fact known perfectly here (the FX node *is* an ``addmm``) is
    discarded into source text and recovered with ``str.__contains__``.  Its measured false
    positives, every one of which this function rejects::

        x = my_warp_reduction_helper()  # a helper whose NAME contains a call name
        x = warp_reduction_count  # a bare NAME, no call at all
        z = foo.cute.gemm2(a)  # a substring of a longer attribute
        w = "warp_reduction"  # a STRING LITERAL

    Each of those declines the inline emission and falls back to a marker for no reason,
    which is the "known at emission and discarded into text" antipattern this tree records
    elsewhere.

    ⚠ THE TARGET SET IS SHARED WITH ``roll_reduction.has_matmul_with_rdim``, deliberately
    imported rather than restated: two lists of matmul targets would drift, and the roller's
    is the one already trusted to decide whether a reduction is rollable at all.

    ⚠ AND IT IS DELIBERATELY *NOT* rdim-SCOPED, unlike the roller's version.  The roller asks
    "is there a matmul over THIS reduction axis" (a question about slicing); the emitter asks
    "is there a matmul anywhere in the producer chain I would re-run" -- and re-running a
    matmul over a *different* axis duplicates its barrier just as badly.  Narrowing to the
    rdim here would silently re-admit exactly the ``matmul_rowsum`` shape this gate exists
    for, whose ``addmm`` is over the K axis, not the reduced N axis.

    Returns False when there is no FX graph to inspect, so a caller outside a graph context
    falls through to the text scan rather than being wrongly cleared.
    """
    fx_node = state.fx_node
    if fx_node is None:
        return False
    graph = getattr(fx_node, "graph", None)
    if graph is None:
        return False
    from ..language._tracing_ops import _for_loop
    from ..language._tracing_ops import _for_loop_step
    from ..language._tracing_ops import _if
    from ..language.matmul_ops import dot as hl_dot
    from ..language.matmul_ops import dot_scaled as hl_dot_scaled

    targets = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
        hl_dot,
        hl_dot_scaled,
    }
    # ⚠⚠ THE WALK MUST FOLLOW SUBGRAPH EDGES, AND FINDING THAT OUT IS THE WHOLE REASON THIS
    # FUNCTION IS NOT A ONE-LINER.  MEASURED on ``matmul_rowsum`` (the shape this gate exists
    # for): the reduction's OWN graph has 10 nodes and NO matmul among them --
    #     ['_get_symnode', 'full', '_for_loop', 'getitem', '_phi', '_mask_to',
    #      'dim_IntList', '_host_tensor', 'store']
    # -- because the ``addmm`` sits inside the K loop, which FX represents as a
    # ``_for_loop(graph_id, ...)`` node referring to a SEPARATE graph.  A single-graph walk
    # therefore returned False on the one kernel it had to catch, while the AST scan found it
    # trivially (it sees the *flattened emitted body*, where the K loop is inlined).
    #
    # ⇒ that asymmetry is the real content of "the information is available at lowering": it
    # is available, but one level of indirection away, and a port that skips the indirection
    # is silently weaker than the text scan it replaces.  Control-flow ops carry their
    # subgraph as ``args[0]`` (a graph id into ``HostFunction.device_ir.graphs``), so the walk
    # resolves those and recurses, with a ``seen`` set because a graph id may appear twice.
    control_flow = {_for_loop, _for_loop_step, _if}
    try:
        all_graphs = HostFunction.current().device_ir.graphs
    except Exception:
        all_graphs = []

    def walk(g: torch.fx.Graph, seen: set[int]) -> bool:
        for node in g.nodes:
            if node.op != "call_function":
                continue
            if node.target in targets:
                return True
            if node.target in control_flow and node.args:
                gid = node.args[0]
                if isinstance(gid, int) and gid not in seen and gid < len(all_graphs):
                    seen.add(gid)
                    if walk(all_graphs[gid].graph, seen):
                        return True
        return False

    return walk(graph, set())


def _cute_tv_has_store_then_load_alias() -> bool:
    """True when some graph has a store-then-load RAW that cannot be FORWARDED.

    ⚠ SEMANTICS CHANGED BY B3, and the name is kept deliberately: this is the
    plan-level gate, so "make it True" is still the way to force the blanket
    decline.  The RAW *detection* now lives in
    :func:`_cute_tv_forwardable_raw_keys`, which classifies each hazard instead
    of declining on the first one; this function reports only the residue that
    classification could not resolve.

    ⚠ This is a MEMORY-DEPENDENCE (RAW) check, and it is a prerequisite for the
    TV path because the TV path's emission order is FROZEN relative to the
    reduction's per-element loop.  Read off the emitted code for
    ``buf[tile_m, :] = 1.0`` followed by ``sum(buf[tile_m, :])``::

        _tv_part_0 = _tv_thr.partition_D(_tv_tile_0)   # the store's leg
        _tv_part_1 = _tv_thr.partition_S(_tv_tile_1)   # the load's leg
        for lane ...:
            cute.copy(_tv_atom, _tv_part_1[..., lane], _tv_frag_1)  # LOAD, pre-loop
            for vi in cutlass.range_constexpr(4):
                _tv_frag_0[vi] = 1.0                   # the store fills its frag
                acc += _tv_frag_1[vi]                  # ... reads the OLD gmem
            cute.copy(_tv_atom, _tv_frag_0, _tv_part_0[..., lane])  # FLUSH, post-loop

    The two legs get their own partition and their own fragment (so this is not
    one shared fragment), but ``_cute_tv_partition_hoist`` anchors a load copy
    BEFORE the constexpr loop and a store flush AFTER it -- which is exactly
    right when the load precedes the store in program order, and exactly wrong
    when it follows one.  MEASURED: N=2048 sums to 4096.0 (the pre-store 2.0s)
    instead of 2048.0, at every ``vec > 1``.

    There is a SECOND instance of the same mistake one layer down:
    ``_cute_tv_partitions`` caches a fragment per ``(tensor, S|D)``, so a second
    load of a stored tensor re-reads the first load's fragment
    (``load_1 = _tv_frag_0[vi]`` in the two-load variant).  Invalidating that
    cache alone would NOT fix this: a re-emitted load copy lands at the same
    frozen position, still ahead of the flush.  Declining the plan is what makes
    both unrepresentable -- with no plan, ``_cute_reduction_vec_width`` stays 1,
    every vec mode in ``_cute_vector_load_ctx`` is unreachable, and the scalar
    path emits an in-place ``(buf.iterator + ...).load()`` per access, which
    observes the store.  MEASURED correct at ``vec == 1`` on this very repro.

    Deliberately ORDER-SENSITIVE, and deliberately asymmetric:

    * store-then-load is the hazard, so it declines;
    * load-then-store is FINE and must stay fast -- an in-place normalisation
      (``v = x[tile, :]`` ... ``x[tile, :] = v * k``) wants the pre-store values,
      which is precisely what the frozen order delivers.  A symmetric "T is both
      read and written" test would decline it and cost throughput for nothing.

    Any store to the tensor counts, not only a row store through the TV leg: a
    scalar store's write is emitted inside the lane body too, so a later row load
    whose copy is hoisted above it is stale for the same reason.

    Scanned PER GRAPH, in node order (fx preserves tracing order, so node order
    is program order).  The root graph already contains every access of a
    single-tile-loop kernel, so this also covers a store and a load that end up
    in different rolled bodies.
    """
    return _cute_tv_forwardable_raw_keys() is None


class ReductionStrategy(TileStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        mask_var: str | None,
        block_size_var: str | None,
    ) -> None:
        super().__init__(
            fn=fn,
            block_ids=[block_index],
        )
        self._mask_var = mask_var
        if block_size_var is not None:
            fn.block_size_var_cache[(block_index,)] = block_size_var
        # Per-INSTANCE, for the reason spelled out at the class-level declaration: a
        # shared dict would carry one kernel's fragment->partition rewrites into the
        # next, and the emitted fragment names repeat across kernels.
        self._cute_tv_stage_read_by_frag: dict[str, str] = {}

    # ── CuTe capability STATE, on the base class ──────────────────────────────
    #
    # ⭐ CLASS-DEFAULT SENTINELS, NOT ``hasattr``.  Every field below is declared
    # here with the value that means "this capability is not in play", so a
    # consumer never has to test the CLASS to find out whether the ATTRIBUTE
    # exists.  That distinction is the whole point of the rework: a class test
    # standing in for "this field exists" is what turned ten call sites into
    # ``isinstance(..., LoopedReductionStrategy)`` and four of them into bare
    # ``assert``s -- i.e. a missed optimisation became a compiler CRASH.
    #
    # ⚠ These are CLASS attributes, deliberately, so a subclass that never runs
    # ``ReductionStrategy.__init__``'s capability block (or that is constructed
    # by a test double) still reads the sentinel rather than raising
    # ``AttributeError``.  A subclass that DOES use a capability rebinds the
    # field on the instance in its own ``__init__``.

    # The TV layout that owns this reduction's access width, or None when the
    # reduction is not on the TV path.  See ``_build_cute_tv_plan``.
    _cute_tv_plan: ChunkTVPlan | None = None
    # The ONE access width, read back off the plan.  ``1`` == scalar, which is
    # what every consumer already treats as "no vector path".
    _cute_reduction_vec_width: int = 1
    # ``"vec"`` (fp32 fast path) or ``"unroll"`` (bf16/fp16 fallback).
    _cute_reduction_vec_mode: str = "vec"
    # Lane-loop trip count, also read back off the plan.
    _cute_reduction_lane_extent: int = 1
    # The extent one "chunk" covers along the reduction axis: the looped path's
    # ``_loop_block_size``, or a loop-free strategy's whole padded extent.
    # ``0`` means "no chunk geometry", which every capability reads as a decline.
    _cute_tv_chunk: int = 0
    # ``cute_cluster_n``: request (knob-and-shape facts) and emitted decision.
    _cute_cluster_n_requested: int = 1
    _cute_cluster_n_emitted: int = 1
    # fx-node id -> its kernel-preamble mbarrier var, and the ``block_idx()[1]``
    # var naming this CTA's rank along the cluster's N axis.
    #
    # ⭐ ON THE BASE CLASS because ``_cute_cluster_mbar_var`` / ``_cute_cluster_y_var``
    # are, and for the same reason the rest of this block is: they are the state the
    # cluster EMITTERS dereference, and a strategy that reaches an emitter without
    # them would raise ``AttributeError`` rather than decline.  Per NODE, not per
    # strategy: one strategy can carry several reduction nodes (layer_norm's mean +
    # variance) and each needs its own barrier phase.
    _cute_cluster_mbar_names: dict[int, str] | None = None
    _cute_cluster_y_name: str | None = None
    # ⭐ The columns THIS CTA owns when a SUBDIVIDING cluster is in play (a loop-free
    # strategy: the cluster partitions one fixed swept extent rather than multiplying a
    # per-iteration chunk).  ``0`` -- the sentinel every field in this block uses -- means
    # "no subdividing split", which is both the no-cluster case and the looped path,
    # whose split lives in its ``for roffset`` bound instead.  Read by the index-offset
    # emitter so the value that sized the split and the value that offsets the index are
    # ONE number.
    _cute_cluster_per_cta_columns: int = 0
    # The per-thread column base var, read by ``cute_tv_tail_predicate``.
    _cute_lane_base_index_var: str | None = None
    # The lane var the copies are sliced by (``plan.emit_lane_slice``).
    _cute_reduction_lane_var: str | None = None
    # The chunk body list whose LAST element is the lane loop; per-chunk
    # declarations insert at ``len - 1``.
    _cute_tv_chunk_prefix: list[ast.AST] | None = None
    # The list of statements INSIDE the lane loop, ending in the constexpr
    # V-loop.  ``memory_ops._cute_tv_partition_hoist`` inserts into it.
    _cute_lane_body: list[ast.AST] | None = None
    # The emitted ``for vi in range_constexpr(vec)`` node, held by reference.
    _cute_tv_constexpr_loop: ast.For | None = None
    # The chunk's tile coordinate along N, and its CTA-LOCAL twin.
    _cute_tv_chunk_index_var: str | None = None
    # The EXPRESSION ``_cute_tv_chunk_index_var`` is assigned, e.g.
    # ``roffset_1 // _REDUCTION_BLOCK_1``.  ⚠ Declared beside the var because the two are
    # NOT interchangeable for identity: every sweep of one row mints a fresh var holding
    # the SAME expression, so only the expression identifies the tile.  Read by the
    # tile-id channel (``cute/memory_ops.py``, task 4); ``None`` where no chunk loop
    # exists (the persistent path sets the var to a literal "0").
    _cute_tv_chunk_index_expr: str | None = None
    _cute_tv_stage_chunk_index_var: str | None = None
    # (atom_var, thr_var) for THE one layout, once emitted.
    _cute_tv_shared: tuple[str, str] | None = None
    # ``cute_reduction_reload``: where the SECOND read of the row comes from.
    _cute_tv_reload_from: str | None = None
    # ⭐ ``cute_row_residency``: the REQUESTED residency of the reduction row, one of
    # ``("registers", "smem", "gmem")``.  Declared on the BASE for the same reason as
    # the rest of this block -- ``memory_ops`` reads it at every load site, and a
    # strategy that never set it must read as a decline rather than crash.  The base
    # value is ``gmem``: no mechanism, i.e. the second read comes from global, which is
    # exactly what a strategy with no TV plan does.
    _cute_row_residency_requested: str = ROW_RESIDENCY_GMEM
    # The CAUSE of a residency decline, set by whichever site refused, so the canonical
    # marker names the real reason instead of a generic string.  ``None`` = not refused.
    _cute_row_residency_decline: str | None = None
    # B3: alias keys whose store-then-load RAW is resolved by FORWARDING.
    # (``cute_row_residency_forbids_sweep_cache`` is defined below, next to
    # ``cute_stage_feasible``, so the two residency capabilities read together.)
    _cute_tv_forwarded_raw_keys: frozenset[object] = frozenset()
    # ``reload_from="smem"`` staging state (capability ③).  Declared here for the
    # same reason as the rest: ``memory_ops._cute_tv_stage_slice`` reads them, and
    # a missing attribute there would be a crash rather than "no staging".
    _cute_tv_stage_smem_var: str | None = None
    _cute_tv_staged_tensors: frozenset[str] | set[str] = frozenset()
    _cute_tv_multi_read_cache: frozenset[str] | None = None
    # ⭐ ``fragment var -> staged READER partition var``, for a strategy whose second
    # sweep is CLONED by the split pass rather than lowered (see
    # ``cute_stage_restages_cloned_sweeps``).  Written by
    # ``memory_ops._cute_tv_stage_slice`` at the ONE load site; read by
    # ``tile_strategy._tv_restage_cloned_loads``, which rewrites the CLONED gmem copy
    # of each fragment into an ``autovec_copy`` off the recorded partition.
    #
    # ⚠ EMPTY IS THE INERT ANSWER, and every consumer must treat it as "rewrite
    # nothing".  Keyed on the FRAGMENT because that is the symbol the cloned copy
    # statement actually names -- keying on the tensor would force the pass to
    # re-derive eligibility from the IR, which is exactly the drift the residency
    # marker exists to prevent.
    #
    # ⚠ ``None``, NOT ``{}``: a mutable class-level default is ONE dict shared by every
    # strategy instance in the process, so a rewrite recorded by one kernel's reduction
    # would be visible to the next -- a cross-kernel leak, and the emitted symbol names
    # (``_tv_frag_0``) repeat across kernels, so it would silently HIT.  The real dict
    # is minted per instance in :meth:`__init__`; this default only guarantees the
    # attribute exists for a strategy that never reaches that code.
    _cute_tv_stage_read_by_frag: dict[str, str] | None = None
    # ⭐⭐ ``reload_from="registers"`` state -- the RMEM analogue of
    # ``_cute_tv_staged_tensors``, and G2's `registers` arm at LOWERING.
    #
    # ``tile id -> the fragment var already holding it``.  The FIRST read of a tile emits
    # its gmem copy as usual and records the fragment here; a LATER read of the SAME tile
    # emits **no copy at all** and reuses that fragment.  ⇒ the "second read comes from
    # registers" decision is a lookup at the site that owns the fragment, instead of an AST
    # post-pass re-discovering it.
    #
    # ⚠ KEYED ON THE TILE ID, NOT THE TENSOR NAME, and that is strictly better than the
    # ``smem`` arm's key.  ``tv_tile_ids`` records
    # ``(tensor, chunk, row_coord, chunk_index_EXPRESSION)`` at the hoist, so two reads
    # match only when they address the SAME tile of the same row -- which is the question.
    # A tensor-name key cannot distinguish two chunks of one tensor, which is the
    # documented one-staged-tensor-per-reduction aliasing gate the ``smem`` arm carries.
    #
    # ⚠ AND IT MUST PERSIST ACROSS SWEEPS, exactly like ``_cute_tv_staged_tensors``:
    # ``_cute_tv_partitions`` is reset per chunk body (its partitions are ``local_tile``s
    # of THAT body and would be out of scope later), but "this tile is already in a
    # fragment" is precisely a cross-sweep fact.  Resetting it would make every sweep
    # re-read gmem, i.e. silently disable the mechanism while still reporting ``registers``.
    #
    # ⚠ ``None``, NOT ``{}`` -- a mutable class default is ONE dict shared by every strategy
    # instance in the process, and the emitted fragment names (``_tv_frag_0``) repeat across
    # kernels, so a leak would silently HIT rather than fail.  Same reasoning as
    # ``_cute_tv_stage_read_by_frag`` above.
    _cute_tv_rmem_frag_by_tile: dict[tuple[object, ...], str] | None = None

    def cute_tv_capable(self) -> bool:
        """Does THIS strategy carry a live TV plan whose emission scaffolding
        exists?

        ⭐ THE COMPUTED PROPERTY THAT REPLACED ``isinstance(strategy,
        LoopedReductionStrategy)``.  A consumer wants to know whether a
        CAPABILITY is present, and the class is the wrong question: it was only
        ever a proxy for "these fields exist", and the proxy is what made
        widening the path a crash instead of a decline.

        Every conjunct is a field the emission protocol actually dereferences:

        * ``_cute_tv_plan`` -- the width, without which there is nothing to emit;
        * ``_cute_lane_body`` -- the mutable list the hoist inserts into;
        * ``_cute_tv_constexpr_loop`` -- the node the insert position is
          anchored to (a list POSITION would move, because the store leg appends
          after it);
        * ``_cute_tv_chunk_prefix`` -- where per-chunk declarations go;
        * ``_cute_tv_chunk_index_var`` -- ``local_tile``'s column coordinate;
        * ``_cute_reduction_lane_var`` -- the copy's lane slice.

        So a False here means "an insert would have gone somewhere that does not
        exist", and the caller declines.  Any strategy that provides all six is
        served, whatever its class -- which is the goal.
        """
        return (
            self._cute_tv_plan is not None
            and self._cute_lane_body is not None
            and self._cute_tv_constexpr_loop is not None
            and self._cute_tv_chunk_prefix is not None
            and self._cute_tv_chunk_index_var is not None
            and self._cute_reduction_lane_var is not None
        )

    def cute_tv_lane_block_id(self) -> int | None:
        """This reduction's own axis, when it carries a plan.

        A reduction owns exactly ONE axis, so there is nothing to select -- but the
        question still has to be ANSWERED, because ``memory_ops`` now asks it of
        whichever strategy answered ``cute_tv_capable()`` in order to decide which
        subscript entry names the copy axis.  Gated on the plan so a scalar reduction
        reports ``None`` (a decline) rather than naming an axis it does not address.
        """
        return None if self._cute_tv_plan is None else self.block_index

    def mask_var(self, block_idx: int) -> str | None:
        assert block_idx == self.block_index
        return self._mask_var

    @property
    def block_index(self) -> int:
        return self.block_ids[0]

    def cute_stage_block_id(self) -> int:
        """A reduction's staged row is its ONE axis, so this is ``block_index``.

        The override that keeps the nine staging methods (now on ``TileStrategy``) emitting
        byte-identical text on this path: they read ``cute_stage_block_id()`` where they
        used to read ``self.block_index``, and here the two are the same number by
        definition.  ⚠ The base default is ``None`` -- a decline -- so this override is
        what makes the methods FUNCTIONAL for a reduction rather than merely present.
        """
        return self.block_index

    def user_size(self, block_index: int) -> sympy.Expr:
        return CompileEnvironment.current().block_sizes[block_index].numel

    def compact_shape(self, shapes: list[CompactedShape]) -> list[CompactedShape]:
        return shapes

    def _reduction_thread_count(self) -> int:
        """Return threads used for this reduction on thread-aware backends."""
        return 0

    def thread_axes_used(self) -> int:
        return 1 if self._reduction_thread_count() > 0 else 0

    def thread_block_sizes(self) -> list[int]:
        count = self._reduction_thread_count()
        return [count] if count > 0 else []

    def _reduction_block_has_lane_loops(self) -> bool:
        """Return True when this reduction block is being traversed via a
        lane loop on the cute backend (synthetic per-thread iteration
        inside a ``DeviceGridState`` that does not have a live thread for
        every logical lane).

        Lane loops serialize part of the logical tile in Python rather
        than mapping it to actual threads, so reductions over the looped
        block cannot be fast-pathed via a warp-level reduction (every
        participating axis must be backed by a live thread).
        """
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        current_grid = codegen.current_grid_state
        if (
            isinstance(current_grid, DeviceGridState)
            and current_grid.has_lane_loops()
            and self.block_index in current_grid.lane_loop_blocks
        ):
            return True
        for loops in codegen.active_device_loops.values():
            for loop_state in loops:
                if (
                    isinstance(loop_state, DeviceGridState)
                    and loop_state.has_lane_loops()
                    and self.block_index in loop_state.lane_loop_blocks
                ):
                    return True
        return False

    def _reduction_block_in_device_lane_loop(self) -> bool:
        """Return True when a ``DeviceLoopState`` distributes this reduction
        block across a per-thread lane loop (CuteNDTileStrategy lanes).

        Unlike :meth:`_reduction_block_has_lane_loops`, this does NOT feed
        :meth:`_needs_loop_carried_accumulator` — it is a dedicated signal for
        the two-pass marker so the existing warp / vec-fold paths keep their
        tuned behavior.
        """
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        for loops in codegen.active_device_loops.values():
            for loop_state in loops:
                if (
                    isinstance(loop_state, DeviceLoopState)
                    and self.block_index in loop_state.lane_loop_blocks
                ):
                    return True
        return False

    def _lane_reduce_threads_in_group(self) -> int | None:
        """Return ``threads_in_group`` for a two-pass lane reduction over this
        block, or ``None`` when this reduction is not over a lane-distributed
        block.

        When the block is split across a per-thread lane loop, the per-lane
        partials must first be accumulated across the lane loop and then
        combined across the live thread axis (``threads_in_group``). A value of
        1 means the block has no live thread axis (a pure lane loop), so the
        accumulator alone is the result.
        """
        # A synthetic reduction lane (PersistentReductionStrategy) always
        # distributes the reduction axis across a lane loop; the live thread
        # axis is ``_reduction_thread_count`` wide.
        if getattr(self, "_synthetic_cute_lane_var", None) is not None:
            return max(1, self._reduction_thread_count())
        if not (
            self._reduction_block_has_lane_loops()
            or self._reduction_block_in_device_lane_loop()
        ):
            return None
        threads = self._reduction_thread_count()
        return max(1, threads)

    def _reshape_merged_reduction_group_params(
        self,
    ) -> tuple[int, int, str] | None:
        """Return ``(pre, group_span, lane_expr)`` for a reshape-merged
        reduction whose live thread axis is interleaved with a *sibling*
        thread axis, or ``None`` when no such interleaving exists.

        When ``x[tile0, tile1, tile2].reshape(tile0, -1).sum(-1)`` merges
        ``tile1`` (a live thread axis) and ``tile2`` (a lane loop) into a
        single synthetic reduction block, the reduction's live thread axis
        (``tile1``) shares the launch warp with the *unrelated* ``tile0``
        row axis. A plain ``cute.arch.warp_reduction_*(threads_in_group=N)``
        folds together CONSECUTIVE warp lanes, so it would sum across both
        ``tile1`` AND ``tile0`` (cross-contaminating rows). Instead the
        reduction must be grouped/strided so each lane only combines the
        lanes that share its ``tile0`` coordinate.

        This computes the ``pre`` (product of live thread extents on axes
        *below* the reduce axis) and ``group_span`` (``pre`` times the
        reduce axis extent) used by ``_cute_grouped_reduce_warp``. Returns
        ``None`` when ``pre == 1`` (no sibling axis below the reduce axis),
        in which case the plain consecutive-lane warp reduction is already
        correct.
        """
        env = CompileEnvironment.current()
        backend = env.backend
        if backend.name != "cute":
            return None
        numel = env.block_sizes[self.block_index].numel
        if not isinstance(numel, sympy.Expr):
            return None
        # Source block ids merged into this reduction dim by the reshape.
        source_block_ids: set[int] = set()
        for symbol in numel.free_symbols:
            if not isinstance(symbol, sympy.Symbol):
                return None
            block_id = env.get_block_id(symbol)
            if block_id is None:
                return None
            source_block_ids.add(env.canonical_block_id(block_id))
        if len(source_block_ids) < 2:
            # A single source block needs no de-interleaving.
            return None
        tile_strategy = self.fn.tile_strategy
        # The reduce axis is the (single) live thread axis spanned by the
        # source blocks. A lane-looped source block has ``extent is None``.
        reduce_axis: int | None = None
        reduce_extent = 1
        for block_id in source_block_ids:
            axis = tile_strategy.thread_axis_for_block_id(block_id)
            extent = tile_strategy.thread_extent_for_block_id(block_id)
            if axis is None or extent is None or extent <= 1:
                continue
            if reduce_axis is not None and reduce_axis != axis:
                # More than one live thread axis among the source blocks is
                # not expressible as a single grouped warp reduce.
                return None
            reduce_axis = axis
            reduce_extent = max(reduce_extent, extent)
        if reduce_axis is None:
            return None
        # Live thread extents of ALL blocks (siblings included) so the linear
        # lane index strides are computed correctly. The reduction block's own
        # synthetic thread axis is excluded -- it is fictional (no real warp
        # lanes back it); the source live axis carries the actual data.
        logical_axis_sizes: dict[int, int] = {reduce_axis: reduce_extent}
        for info in env.block_sizes:
            block_id = info.block_id
            if block_id == self.block_index or block_id in source_block_ids:
                continue
            axis = tile_strategy.thread_axis_for_block_id(block_id)
            extent = tile_strategy.thread_extent_for_block_id(block_id)
            if axis is None or extent is None or extent <= 1:
                continue
            logical_axis_sizes[axis] = max(logical_axis_sizes.get(axis, 1), extent)
        pre = 1
        for axis in range(reduce_axis):
            pre *= logical_axis_sizes.get(axis, 1)
        if pre <= 1:
            # No sibling thread axis below the reduce axis: the reduce axis is
            # already at the bottom of the linear lane index, so consecutive
            # warp lanes belong to the reduction and the plain warp reduce is
            # correct.
            return None
        group_span = pre * reduce_extent
        if group_span > 32:
            # Cross-warp grouped reduction is not handled by the marker path.
            return None
        lane_expr = backend.thread_linear_index_expr(logical_axis_sizes)
        if lane_expr is None:
            return None
        return pre, group_span, lane_expr

    def _emit_inline_lane_reduce(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        identity_expr: str,
        threads: int,
        *,
        acc_dtype_str: str,
        group_pre: int = 1,
        group_span: int = 0,
        group_lane_expr: str = "",
        group_count: int = 1,
        result_hint: str = "lane_reduce",
    ) -> str | None:
        """⭐⭐ G1: emit a lane-distributed reduction's BOTH combines HERE, at lowering.

        Returns the name of the finished reduced scalar, or ``None`` when this site
        cannot do it inline (the caller then falls back to emitting a marker).

        The two combines a lane-distributed reduction owes are emitted as:

        * ``acc = identity`` into the grid state's **segment prefix** -- above every lane
          loop, so it is initialised once rather than per lane;
        * ``acc = combine(acc, v)`` into the **currently open segment**, i.e. right here,
          which is inside the lane loop :meth:`DeviceGridState.wrap_body` will mint;
        * the **cross-thread combine** into this segment's **seal**, which
          :meth:`DeviceGridState._wrap_segmented_body` emits *between* the sibling lane
          loops -- outside every one of them, by construction.

        ⇒ the reduction is COMPLETE where it appears.  No marker, no deferred structure,
        and no AST pass that could discharge the obligation by deleting it.

        ⛔ WHY THIS IS NOT MERELY TIDIER THAN THE MARKER.  Read as IR,
        ``MARKER(v,'max')`` is a valid ONE-element reduction, so ``mx = v`` was a
        *faithful* lowering of it -- which is how the deferred design produced kernels
        that compiled, looked plausible, and were WRONG (``exp(v-v) == 1.0``: softmax rows
        summing to 128.0 instead of 1.0; ``got[0]=1.0`` against ``ref[0]=59.4``; relerr
        7.685 on ``matmul_layernorm`` at N=512).  Emitting the structure at the site where
        the obligation is KNOWN removes the window in which it can be lost.

        ⚠ THE CROSS-THREAD COMBINE IS BUILT BY THE SAME HELPER THE AST PASS USED
        (``_finalize_lane_reduce_marker``), through a ``_LaneReduceMarker`` used as a
        plain parameter record rather than as an emitted call.  That helper picks between
        four cross-thread forms -- two-stage cross-warp, strided grouped single-warp,
        plain consecutive-lane, and none -- and getting that choice wrong
        cross-contaminates rows.  ⭐ Reusing it means the inline path and the (still
        reachable) marker path cannot DISAGREE about the combine for the same layout,
        which is the property that makes this replacement safe to land incrementally.
        """
        from .tile_strategy import _combine_expr
        from .tile_strategy import _contains_unduplicatable_op
        from .tile_strategy import _finalize_lane_reduce_marker
        from .tile_strategy import _LaneReduceMarker

        grid = state.codegen.current_grid_state
        # ⚠ EVERY CONDITION HERE IS A REQUIREMENT OF THE MECHANISM, NOT A PREFERENCE:
        #  * a ``DeviceGridState`` is what owns the segments and mints the lane loops;
        #  * ``lane_loops`` must be non-empty, because the seal is emitted BETWEEN loops
        #    that ``wrap_body`` builds from that list -- with no registered lane loop
        #    there is nothing to seal and the fold would run once instead of per lane;
        #  * a prebuilt nest (the TV protocol) holds ONE ``(emit, sink)`` pair and is
        #    structurally unable to express two sibling lane loops, so that path keeps the
        #    marker.  ⇒ each of these is a decline, never a silent best effort.
        # ⭐⭐ THE WITHHOLDING SWITCH -- THE FAIL-CAPABILITY ARM'S ONLY INSTRUMENT.
        # ``HELION_INLINE_LANE_REDUCE=0`` withholds the inline emission, restoring the
        # marker path.  Required rather than convenient: G1's success criterion is "markers
        # go to ZERO", and zero is ALSO what a dead hook, a kernel that stopped compiling,
        # and a shape that never emitted all report.  Without a way to turn the new
        # emission off in the same process, "the markers came back when it is withheld" --
        # the arm that distinguishes those cases -- has no instrument at all.  Four
        # investigations in this tree were fooled by an unfalsifiable zero.
        if os.environ.get("HELION_INLINE_LANE_REDUCE", "1").strip() == "0":
            return None
        if not isinstance(grid, DeviceGridState):
            return None
        if not grid.lane_loops:
            return None
        # ⭐⭐ A1: A PREBUILT NEST NO LONGER DECLINES *IF IT CAN BE REBUILT*.
        #
        # This used to read ``or grid.prebuilt_lane_nest is not None``, whose stated reason
        # was that a prebuilt nest "holds ONE ``(emit, sink)`` pair and is structurally
        # unable to express two sibling lane loops".  That is true of a nest that can only
        # be SPLICED, and false once its owner also registers
        # ``prebuilt_lane_nest_factory``, which mints a fresh nest per segment.
        #
        # ⛔ WHY THE DECLINE WAS EXPENSIVE.  Declining here falls back to the AST marker,
        # and the marker path cannot restructure this shape: the marker lands inside a
        # ``range_constexpr(vec)`` loop, which ``split_lane_loop_reductions`` only
        # recognises as a DIRECT child of a lane loop, so it survives to
        # ``restore_unprocessed_lane_reduce_markers`` and -- having declared
        # ``partial_fold`` -- correctly RAISES rather than silently dropping both combines.
        # MEASURED with the force-roll bypassed: ``reduction_loops=[None]`` +
        # ``cute_vector_widths=[8,1]`` raised ``BackendUnsupported: a lane reduction ('sum'
        # over 'v_1') reached the end of codegen still owing its lane fold and its
        # cross-thread combine`` at N=256 and N=1024.  ⇒ persistent + TV was UNEMITTABLE,
        # and ``ConfigSpec.normalize``'s force-roll existed only to dodge that.
        #
        # ⚠ A nest with no factory still declines: segmenting it would have to alias one
        # ``sink`` across sibling loops, so the marker fallback stays the honest answer.
        if grid.prebuilt_lane_nest is not None and (
            grid.prebuilt_lane_nest_factory is None
        ):
            return None
        # ⛔⛔ THIS DECLINE MUST HAPPEN BEFORE THE EMISSIONS BELOW, AND IT USED TO HAPPEN
        # AFTER THEM -- WHICH TURNED A CLEAN FALLBACK INTO A COMPILE FAILURE.
        #
        # The condition is the one stated further down (a reduction with no synthetic axis
        # of its own can only seal against the single registered lane loop; with two it
        # cannot know which is its own).  It was tested at the SEAL, i.e. after
        # ``lane_segment_prefix.append(acc = identity)`` and after
        # ``state.add_statement(acc = combine(acc, v))`` had already been emitted.  So the
        # decline left BOTH halves of a half-built reduction in the body and returned
        # before ``seal_lane_segment``: no seal ⇒ ``lane_segment_seals`` stays ``None`` ⇒
        # ``wrap_body`` takes its unsegmented path ⇒ the seed and the cross-thread combine
        # are never emitted, leaving an accumulator read-and-written with no initialiser.
        # ``_has_extra_cross_lane_carry`` then correctly reports an independent carry and
        # raises ``BackendUnsupported``.  ⇒ THE RAISE WAS THIS DECLINE'S OWN DEBRIS: the
        # diagnostic described what it saw accurately and blamed the wrong thing.
        #
        # MEASURED on ``out[tm,tn] = v - sum(v,-1)`` at ``block_sizes=[32,32]``,
        # ``num_threads=[8,8]``, ``cute_vector_widths=[1,1]`` -- a kernel that COMPILES ON
        # ``origin/main`` and raised here; reported at ~19% of ``random_config()`` draws.
        # Declining early instead restores the marker path, which handles two lane loops
        # correctly (verified bit-exact at every ``num_threads`` tried).
        #
        # ⭐ Reachable because of a THIRD ``lane_loops`` producer that
        # ``grep 'add_lane_loop('`` misses: ``CuteNDTileStrategy.codegen_grid`` builds
        # ``DeviceGridState`` with ``lane_loops`` PRE-POPULATED via the constructor, one
        # entry per tile block whose ``block_size > num_threads``.  So a 2-D ``hl.tile``
        # arrives here with ``len(lane_loops) == 2`` before the grid body lowers at all --
        # MEASURED ``[('lane_0', 4), ('lane_1', 4)]``, no registration-order race.
        # ⚠ That REFUTES the "this test never fires" note below, which is right for its own
        # shape (a free ``hl.arange`` registering after the seals) and generalises from a
        # census that missed the constructor producer.
        #
        # ⚠ ORDER MATTERS WITH THE ``group_span`` FIX: the marker path this restores was
        # itself silently wrong for strided single-warp groups until
        # ``_lane_loop_cross_warp_group_params`` stopped declining them, so moving this
        # decline up ALONE would have traded a loud raise for a quiet wrong answer at
        # ``num_threads`` products <= 32.  Both changes land together, deliberately.
        if (
            getattr(self, "_synthetic_cute_lane_var", None) is None
            and len(grid.lane_loops) != 1
        ):
            return None
        # ⛔⛔ EXACTLY ONE REGISTERED LANE LOOP, AND THIS GUARD IS A MEASURED FIX, NOT CAUTION.
        #
        # ``_wrap_segmented_body`` segments ``lane_loops[-1]`` and wraps the rest around the
        # whole run.  Its docstring used to justify that with "a reduction seals against the
        # lane axis it is distributed over, which is the innermost one" -- **which is false**.
        # ``lane_loops`` is ONE FLAT LIST fed by two unrelated producers (this strategy's
        # synthetic reduction lane, and ``generate_ast``'s free-``hl.arange`` lane), so
        # ``[-1]`` is *the last registered*, not *this reduction's*.
        #
        # MEASURED, on a lane-distributed reduction pair followed by a free
        # ``hl.arange(0, 2048)`` (which registers its own lane loop AFTER the reduction's):
        # the seals segmented the arange's axis, so both accumulator seeds AND both
        # ``warp_reduction_*`` landed INSIDE the reduction's own lane loop -- the accumulator
        # re-initialised and the cross-thread combine run on an unfinished partial, once per
        # lane.  ⚠ On the marker path the same shape is a SILENT WRONG ANSWER
        # (``maxerr 211.6``, got 6.11 against ref 103.19); on this path it emitted invalid
        # code (a free ``indices_1``), because ``lane_setup_statements`` is flat across axes
        # too and the wrong axis's index got cloned into the wrong loop.
        #
        # ⇒ FIXED by keying each seal to its OWN axis (``seal_lane_segment``'s ``lane_var``),
        # so the wrap segments the reduction's loop and nests the others around it.  ⚠ The
        # guard that used to stand here (``len(grid.lane_loops) != 1 -> decline``) could not
        # work: MEASURED, the second axis registers AFTER both seals, so at this point there is
        # always exactly one lane loop and the test never fired.  That is why the axis is
        # carried as data instead of being inferred from a count.
        # ⚠ The sink must be the list ``wrap_body`` will wrap, or the recorded seal index
        # refers to a different list than the one it is later applied to -- a
        # cross-thread combine emitted inside a lane loop, i.e. a wrong answer.  The grid
        # body is the sink at the BOTTOM of the stack under the host statements; a
        # reduction lowered inside a nested sink (a serial device loop's body, an ``if``)
        # is not on that list, so it declines and keeps the marker.
        sink = state.codegen.statements_stack[-1]
        if sink is not state.codegen.active_grid_body_statements:
            return None
        # ⛔⛔ THE INPUT'S PRODUCER MUST BE RE-RUNNABLE, AND THIS DECLINE IS MEASURED, NOT
        # DEFENSIVE.  The segment mechanism re-materialises a shared producer into each
        # sibling lane loop that needs it (that is what makes two dependent reductions
        # expressible at all).  A pure ``gmem -> rmem`` load is safe to re-run; a matmul or
        # a cross-thread collective is NOT -- re-running it duplicates a barrier.
        #
        # MEASURED on ``matmul_rowsum`` (``addmm`` in a K loop, then ``acc.sum(-1)`` over
        # the lane-distributed N axis) with this gate absent: the fold read ``acc`` BEFORE
        # the K loop that computes it -- ``sum_1_lane_acc + acc`` was emitted above the
        # loop, with ``acc`` still 0.0 -- giving **rel 1.0** and ``got[0] = 0.000``.  The
        # producer is ``_cute_grouped_reduce_shared_two_stage``, i.e. exactly an
        # unduplicatable collective, and the statement carrying it is a nested serial
        # ``for`` this segment analysis treats as one opaque unit.
        #
        # ⭐ Declining here is not a fallback to something wrong: the marker path this
        # returns to routes such a shape to ``_split_lane_loop_with_register_stash``, which
        # runs the collective ONCE and re-derives the reduction from a per-thread fragment
        # -- a genuine cross-lane fold rather than a duplicated barrier.  ⇒ this is the
        # boundary between the two mechanisms, and the existing predicate draws it.
        #
        # ⭐⭐ ASKED AT LOWERING, BY FX NODE-TARGET IDENTITY, WITH THE TEXT SCAN AS A
        # FALLBACK.  ``_cute_unduplicatable_producer_in_graph`` walks this reduction's own FX
        # graph and tests ``node.target in {aten.mm, aten.addmm, aten.bmm, aten.baddbmm,
        # hl.dot, hl.dot_scaled}`` -- the same identity test ``roll_reduction`` already uses.
        #
        # ⛔ WHY IDENTITY BEATS THE SCAN, and it is a correctness argument rather than a
        # taste one.  ``_contains_unduplicatable_op`` does ``ast.unparse(stmt)`` and then
        # ``str.__contains__`` against five hard-coded names, so a fact known PERFECTLY at
        # lowering (the FX node IS an ``addmm``) is thrown away into source text and
        # recovered with grep.  MEASURED false positives, all of which the identity test
        # rejects: ``x = my_warp_reduction_helper()``, a bare name ``warp_reduction_count``,
        # ``foo.cute.gemm2(a)``, and even the STRING LITERAL ``w = 'warp_reduction'``.  Each
        # of those declines the inline emission and falls back to a marker for no reason.
        #
        # ⚠ THE SCAN IS KEPT AS AN ``or`` EVEN THOUGH IT IS MEASURABLY REDUNDANT HERE, and
        # the reason is a coverage gap I could not close by measurement.  MEASURED over 27
        # configs x 3 matmul-reduction shapes (row-sum, row-max, broadcast-store), with the
        # scan NEUTERED and only the FX identity test live: **identical results in every
        # cell** (6 OK / 3 BackendUnsupported per shape, both arms).  So at THIS call site the
        # identity test subsumes it.
        #
        # ⛔ BUT ``_UNDUPLICATABLE_CALLS`` also names the three ``_cute_grouped_reduce_*``
        # collectives, which are NOT matmuls and are emitted as helper CALLS with no FX node
        # of their own -- so no node-target test can see them.  I could not construct a
        # kernel that reaches this gate needing only that half, which means I cannot prove
        # the half is dead; and the failure it would allow is a DUPLICATED CROSS-THREAD
        # BARRIER, i.e. a wrong answer rather than a missed optimisation.  ⇒ identity FIRST,
        # so its precision decides every case it can see; text second, so an unproven gap
        # fails closed.
        if _cute_unduplicatable_producer_in_graph(state) or any(
            _contains_unduplicatable_op(stmt) for stmt in sink
        ):
            return None
        acc_var = self.fn.new_var(f"{result_hint}_lane_acc", dce=False)
        result_var = self.fn.new_var(f"{result_hint}_lane_result", dce=False)
        record = _LaneReduceMarker(
            result_var=result_var,
            input_name=input_name,
            reduction_type=reduction_type,
            identity_expr=identity_expr,
            threads_in_group=threads,
            wrap_template="__HELION_FINALIZED__",
            group_pre=group_pre,
            group_span=group_span,
            group_lane_expr=group_lane_expr,
            group_count=group_count,
            acc_dtype_str=acc_dtype_str,
            # ⭐ The obligation is DISCHARGED HERE, so the record is not partial: this
            # object never becomes an emitted marker and nothing downstream may revert it.
            partial_fold=False,
        )
        grid.lane_segment_prefix.append(
            statement_from_string(f"{acc_var} = {identity_expr}")
        )
        # Cast to the accumulator dtype before combining, for the same reason the AST
        # split did: the CUTLASS DSL's ternary type check is strict (max/min emit
        # ``a if a > b else b``), and an int32 sum must accumulate in int64 rather
        # than wrap.
        combine_val = f"{acc_dtype_str}({input_name})" if acc_dtype_str else input_name
        state.add_statement(
            f"{acc_var} = {_combine_expr(reduction_type, acc_var, combine_val)}"
        )
        # Seal AFTER the fold: ``len(sink)`` is now one past it, so the fold is the last
        # statement inside the loop being closed and the combine lands just outside.
        # ⭐ THE AXIS THIS REDUCTION IS DISTRIBUTED OVER, stated rather than inferred.  A
        # ``PersistentReductionStrategy`` carries it as ``_synthetic_cute_lane_var``; a
        # ``BlockReductionStrategy`` has none of its own and is distributed over whichever lane
        # loop the grid registered for its block, so fall back to the single registered var.
        # ⚠ The wrap CANNOT recover this later -- registration order interleaves the axes -- so
        # passing it here is what makes multi-axis kernels safe (see ``seal_lane_segment``).
        # ⚠ NO DECLINE HERE.  The ``seal_lane_var is None and len(lane_loops) != 1`` case
        # is rejected by the early predicate at the top of this method, so by this point
        # the fallback below is always well defined.  It must NOT be re-tested here: this
        # point is past two side-effecting emissions, and declining after them is exactly
        # the leak that made a compilable kernel raise (see that predicate's note).
        # An assert rather than a branch, so a future edit that weakens the early check
        # fails loudly here instead of silently reintroducing the half-built reduction.
        seal_lane_var = getattr(self, "_synthetic_cute_lane_var", None)
        if seal_lane_var is None:
            assert len(grid.lane_loops) == 1, (
                "unreachable: the early predicate declines a non-synthetic reduction "
                f"with {len(grid.lane_loops)} lane loops before any emission"
            )
            seal_lane_var = grid.lane_loops[0][0]
        grid.seal_lane_segment(
            len(sink), _finalize_lane_reduce_marker(record, acc_var), seal_lane_var
        )
        return result_var

    def _lane_reduce_marker_unsupported(self, state: CodegenState) -> bool:
        """Return True when the two-pass lane-reduction marker cannot be
        handled by the ``split_lane_loop_reductions`` post-pass, so the caller
        must fall back to the existing single-pass path.

        Two situations are unsupported:

        * An active *serial* device loop (over a different block) wraps this
          reduction inside the lane scope. The post-pass splits the lane loop
          at its top level, but here the reduction needs the lanes reduced
          *per* serial iteration (the lane loop is outside the serial loop).
        * An active ``LoopedReductionStrategy`` already rolls this block: the
          rolled loop carries its own accumulator (and vec-fold) for the lane
          reduction, so emitting a marker would double-handle it.
        """
        for block_id, loops in state.codegen.active_device_loops.items():
            for loop_state in loops:
                if not isinstance(loop_state, DeviceLoopState):
                    continue
                if isinstance(loop_state.strategy, LoopedReductionStrategy):
                    # The rolled reduction owns the lane reduction over its
                    # block; defer to its accumulate / vec-fold machinery.
                    if block_id == self.block_index:
                        return True
                    continue
                if (
                    block_id != self.block_index
                    and block_id not in loop_state.block_thread_axes
                    and block_id not in loop_state.lane_loop_blocks
                ):
                    # The reduction's lane loop is OUTSIDE this serial device
                    # loop. A synthetic-lane PersistentReductionStrategy can be
                    # repaired by the ``interchange_lane_outside_serial_reductions``
                    # post-pass (it splits the lane loop into a lane-inside-mb
                    # two-pass nest for the broadcast consumer plus a
                    # lane-outside-mb nest for any per-feature accumulators), so
                    # keep emitting the marker in that case. Other (non-synthetic)
                    # situations remain unsupported. ``getattr`` guards strategies
                    # without a synthetic lane var (e.g. BlockReductionStrategy).
                    if getattr(self, "_synthetic_cute_lane_var", None) is not None:
                        continue
                    return True
        return False

    def _reduction_block_is_serial(self) -> bool:
        """Return True when this reduction block is being traversed by a
        serial ``DeviceLoopState`` (a Python ``for`` loop) rather than a
        live thread axis.

        Reductions over a serially-iterated block cannot be fast-pathed
        via a warp-level reduction; the surrounding loop has to carry the
        accumulator.
        """
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        for loop_state in codegen.active_device_loops.get(self.block_index, []):
            if (
                isinstance(loop_state, DeviceLoopState)
                and self.block_index not in loop_state.block_thread_axes
            ):
                return True
        return False

    def _reduction_block_has_live_thread_axis(self) -> bool:
        """Return True when this reduction block is mapped to a live thread
        axis in the active loop nest (in either the current grid or any
        active device loop).

        A ``False`` return on the cute backend means a warp-level reduction
        across this block would fold together unrelated tensor elements,
        because no real threads back the block. The caller falls back to
        loop-carried accumulation.
        """
        codegen = getattr(self, "_codegen", None)
        if codegen is None:
            return False
        current_grid = codegen.current_grid_state
        if (
            isinstance(current_grid, DeviceGridState)
            and self.block_index in current_grid.block_thread_axes
        ):
            return True
        for loop_state in codegen.active_device_loops.get(self.block_index, []):
            if self.block_index in loop_state.block_thread_axes:
                return True
        for loops in codegen.active_device_loops.values():
            for loop_state in loops:
                if self.block_index in loop_state.block_thread_axes:
                    return True
        return False

    def _needs_loop_carried_accumulator(self) -> bool:
        """Return True when the surrounding loop nest must perform the
        reduction via loop-carried accumulation instead of a warp-level
        reduction across threads.

        This consolidates the three "no live thread axis" conditions:

        * :meth:`_reduction_block_is_serial` — the block is iterated by
          a serial ``DeviceLoopState`` rather than a thread axis;
        * :meth:`_reduction_block_has_lane_loops` — the block is
          iterated by a lane loop (synthetic per-thread iteration);
        * ``not _reduction_block_has_live_thread_axis()`` — the block
          is not mapped to any thread axis at all.

        In every case the conclusion is the same: there is no live
        thread axis to reduce across, so the surrounding loop must
        accumulate the partial values across iterations.

        Always returns False for tile-level backends (Triton / Pallas /
        TileIR) which use their native reduction primitives.
        """
        if CompileEnvironment.current().backend.max_reduction_threads() is None:
            return False
        return (
            self._reduction_block_is_serial()
            or self._reduction_block_has_lane_loops()
            or not self._reduction_block_has_live_thread_axis()
        )

    def _planned_thread_dims(self) -> tuple[int, int, int]:
        return self.fn.tile_strategy.thread_block_dims()

    def _get_thread_axis(self) -> int:
        """Compute the thread axis index for this reduction strategy.

        Some backends place reduction strategies first so reduction threads share
        a warp. Others keep the natural strategy order.
        """
        env = CompileEnvironment.current()
        if (axis := self.fn.tile_strategy.thread_axis_for_strategy(self)) is not None:
            return axis
        if env.backend.reduction_axis_first():
            axis = 0
            for strategy in self.fn.tile_strategy.strategies:
                if strategy is self:
                    break
                if isinstance(strategy, ReductionStrategy):
                    axis += strategy.thread_axes_used()
            return axis
        axis = 0
        for strategy in self.fn.tile_strategy.strategies:
            if strategy is self:
                break
            axis += strategy.thread_axes_used()
        return axis

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        raise NotImplementedError

    def call_reduction_function(
        self,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> str:
        backend = CompileEnvironment.current().backend
        if backend.is_indexed_reduction(reduction_type):
            index_var = self.index_var(self.block_index)
            return self.call_indexed_reduction(
                input_name,
                self.broadcast_str(index_var, fake_input, dim),
                reduction_type,
                dim,
                fake_output,
            )
        return backend.reduction_expr(
            input_name,
            reduction_type,
            dim,
            block_size_var=self.block_size_var(self.block_index),
        )

    def _index_init_expr(self, block_size_var: str, dtype: str, block_idx: int) -> str:
        env = CompileEnvironment.current()
        backend = env.backend
        size = env.block_sizes[block_idx].size
        if isinstance(size, int) and size == 0:
            return backend.reduction_index_zero_expr(dtype)
        if isinstance(size, torch.SymInt) and env.known_equal(size, 0):
            return backend.reduction_index_zero_expr(dtype)
        return backend.reduction_index_expr(
            block_size_var, dtype, block_idx, axis=self._get_thread_axis()
        )

    def call_indexed_reduction(
        self,
        input_name: str,
        index_value: str,
        reduction_type: str,
        dim: int,
        fake_output: torch.Tensor,
    ) -> str:
        env = CompileEnvironment.current()
        return env.backend.argreduce_result_expr(
            input_name,
            index_value,
            reduction_type,
            dim,
            fake_output.dtype,
            block_size_var=self.block_size_var(self.block_index),
            index_dtype=env.index_dtype,
        )

    def _indexed_lane_reduce_expr(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
        threads: int,
        *,
        group_pre: int = 1,
        group_span: int = 0,
        group_lane_expr: str = "",
        group_count: int = 1,
    ) -> str:
        """Lower ``argmin``/``argmax`` over a lane-distributed axis into TWO
        plain (non-indexed) lane-reduce markers.

        The two-pass lane machinery has no *paired* (value, index) accumulator:
        ``_combine_expr``/``_warp_reduce_expr`` only know scalar ``sum``/
        ``prod``/``max``/``min``.  Rather than teach them a second, tie-breaking
        accumulator shape, decompose the indexed reduction the same way
        :meth:`CuteBackend.argreduce_result_expr` already does for the
        non-lane path — into a value reduction followed by a ``min`` over the
        candidate indices of the winners:

        1. ``value = max(input)`` — marker 1;
        2. ``cand = index if input == value else INDEX_MAX`` — a per-lane select;
        3. ``result = min(cand)`` — marker 2.

        The second marker's input reads the first marker's reduced scalar, so
        this is exactly the sequentially-dependent marker shape that
        ``_split_lane_loop_multi_stage`` emits one accumulate/finalize pass per
        dependency layer for.  Nothing new is needed in the marker vocabulary.

        Semantics, matching ``argreduce_result_expr`` (and therefore the rolled
        and non-lane CuTe paths) exactly:

        * **ties** resolve to the LOWEST index, because the second reduction is a
          ``min`` over the indices whose value equals the winner — the same rule
          ``torch.argmax``/``torch.argmin`` use;
        * **dtypes stay distinct**: the value half accumulates in
          ``reduction_acc_dtype`` (fp32 for a bf16/fp16 input) while the index
          half accumulates in ``env.index_dtype``, each carried through its own
          marker's explicit ``acc_dtype`` (class 5's ABI).  The per-lane input is
          cast to the value accumulator dtype before the comparison so the
          CUTLASS DSL's strict ternary type check never sees mixed widths;
        * **identities**: ``-inf``/``+inf`` (``default_accumulator``) for the
          value half and ``iinfo(index_dtype).max`` for the index half — the
          latter is the identity of a ``min``, and is also what
          ``reduction_index_init_expr`` seeds the rolled path's index
          accumulator with.
        """
        from .tile_strategy import _lane_reduce_marker_expr

        env = CompileEnvironment.current()
        backend = env.backend
        name_hint = state.fx_node.name if state.fx_node is not None else reduction_type

        # --- marker 1: reduce the VALUES -------------------------------------
        value_reduction = "min" if reduction_type == "argmin" else "max"
        value_acc_dtype = reduction_acc_dtype(value_reduction, fake_input.dtype)
        value_acc_dtype_str = _dtype_str(value_acc_dtype)
        value_identity = ir.Reduction.default_accumulator(
            value_reduction, value_acc_dtype
        )
        assert isinstance(value_identity, (float, int, bool))
        value_identity_expr = backend.cast_expr(
            constant_repr(value_identity), value_acc_dtype_str
        )
        # ⭐⭐ G1 ON THE ARGMAX PAIR, AND IT NEEDS NO NEW VOCABULARY -- which is the point.
        #
        # ``argmax`` is ONE FX node that lowers to TWO dependent reductions (reduce the
        # values; then ``min`` over the indices of the winners, which needs the value
        # FINALIZED).  Any design that decides "how many lane loops" by counting FX
        # reduction nodes sees **1** and builds one loop where two are required.  The
        # segment mechanism does no counting: each reduction seals its own segment where it
        # appears, so the two loops fall out of the sequence.
        #
        # ⚠ BOTH HALVES MUST TAKE THE SAME ROUTE.  A mixed pair -- value inline, index as a
        # marker -- would leave the AST pass looking at a lane loop holding one marker whose
        # dependency layer was already cut out from under it.  So the value half's result
        # decides, and the index half below follows it.
        inline_value = self._emit_inline_lane_reduce(
            state,
            input_name,
            value_reduction,
            value_identity_expr,
            threads,
            acc_dtype_str=value_acc_dtype_str,
            group_pre=group_pre,
            group_span=group_span,
            group_lane_expr=group_lane_expr,
            group_count=group_count,
            result_hint=f"{name_hint}_value",
        )
        if inline_value is not None:
            value_var = inline_value
        else:
            value_var = self.fn.new_var(f"{name_hint}_lane_value", dce=True)
            state.add_statement(
                f"{value_var} = "
                + _lane_reduce_marker_expr(
                    input_name,
                    value_reduction,
                    value_identity_expr,
                    threads,
                    acc_dtype_str=value_acc_dtype_str,
                    group_pre=group_pre,
                    group_span=group_span,
                    group_lane_expr=group_lane_expr,
                    group_count=group_count,
                )
            )

        # --- the per-lane candidate index ------------------------------------
        index_dtype = env.index_dtype
        index_dtype_str = backend.index_type_str(index_dtype)
        index_identity_expr = backend.cast_expr(
            repr(torch.iinfo(index_dtype).max), index_dtype_str
        )
        index_value = self.broadcast_str(
            self.index_var(self.block_index), fake_input, dim
        )
        candidate_var = self.fn.new_var(f"{name_hint}_lane_index", dce=True)
        state.add_statement(
            f"{candidate_var} = "
            + backend.where_expr(
                f"({backend.cast_expr(input_name, value_acc_dtype_str)}) "
                f"== ({value_var})",
                f"{index_value}",
                index_identity_expr,
            )
        )

        # --- reduction 2: reduce the candidate INDICES ------------------------
        # ⚠ The candidate select above lowered into the segment the value half REOPENED, so
        # it is inside the second lane loop and reads the FINALIZED value scalar -- which is
        # exactly the dependency that made this pair need two loops.  Sealing here closes
        # that loop and puts the index combine after it.
        if inline_value is not None:
            inline_index = self._emit_inline_lane_reduce(
                state,
                candidate_var,
                "min",
                index_identity_expr,
                threads,
                acc_dtype_str=index_dtype_str,
                group_pre=group_pre,
                group_span=group_span,
                group_lane_expr=group_lane_expr,
                group_count=group_count,
                result_hint=f"{name_hint}_index",
            )
            # ⛔ The value half already sealed a segment, so falling back to a marker HERE
            # would strand the index reduction in a body whose layering has been cut.
            # Nothing can produce that state -- both halves see the same grid state and the
            # same sink -- so assert it rather than inventing a repair for an unreachable
            # case.
            assert inline_index is not None, (
                "the argmax value half emitted inline but the index half declined; "
                "both see the same grid state, so this should be unreachable"
            )
            return backend.cast_expr(inline_index, _dtype_str(fake_output.dtype))
        index_marker = _lane_reduce_marker_expr(
            candidate_var,
            "min",
            index_identity_expr,
            threads,
            acc_dtype_str=index_dtype_str,
            group_pre=group_pre,
            group_span=group_span,
            group_lane_expr=group_lane_expr,
            group_count=group_count,
        )
        return backend.cast_expr(index_marker, _dtype_str(fake_output.dtype))

    def maybe_reshape(
        self,
        expr: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> str:
        size = [*fake_input.size()]
        size.pop(dim)
        if [*fake_output.size()] == size:
            return expr
        backend = CompileEnvironment.current().backend
        shape = self.fn.tile_strategy.shape_str([*fake_output.size()])
        return backend.maybe_reshape_reduction(
            expr,
            source_shape=size,
            target_shape=[*fake_output.size()],
            target_shape_expr=shape,
        )

    def broadcast_str(self, base: str, fake_input: torch.Tensor, dim: int) -> str:
        input_size = [*fake_input.size()]
        expand = self.fn.tile_strategy.expand_str(input_size, dim)
        shape = self.fn.tile_strategy.shape_str(input_size)
        return CompileEnvironment.current().backend.broadcast_to_expr(
            f"{base}{expand}", shape
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CuTe reduction CAPABILITIES.  On the BASE class deliberately: these five
    # blocks (the TV plan, the cluster/DSMEM combine, SMEM staging, the ragged
    # tail, mask elision) used to live inside ``LoopedReductionStrategy``, so a
    # consumer had to test the CLASS to discover whether a CAPABILITY existed.
    #
    # ⭐ THE COUPLING WAS ONE FIELD.  ``_build_cute_tv_plan``'s only
    # loop-specific input was ``self._loop_block_size`` ("the extent one
    # iteration covers"), so it is now a ``chunk`` PARAMETER: the looped path
    # passes ``self._loop_block_size``, and a strategy with no outer loop passes
    # its own padded extent.  ``_cute_layout_participants`` reads no ``self``
    # state at all (it walks ``HostFunction.current().device_ir``).
    #
    # ⚠ ``cute_tv_capable()`` -- NOT ``isinstance`` -- is what a consumer asks.
    # See its docstring for why (this repo's own enumeration antipattern).
    # ══════════════════════════════════════════════════════════════════════════

    def cute_tv_rounded_extent(self) -> int | None:
        """``N'`` when this reduction covers a ROUNDED-UP tile, else ``None``.

        ``None`` means "the tile is exactly ``N``" -- either there is no TV plan
        (scalar path) or ``N`` already divides the tile granularity.  Every caller
        therefore reads ``None`` as "nothing to predicate", and the divisible path
        stays byte-identical because ``None`` is what it saw before this existed.

        The granularity is ``chunk * cluster_n``, not ``chunk``: see
        :func:`ragged_tail.tile_granularity`.

        ⚠ ``cluster_n`` is the **EMITTED** one, deliberately.  The requested cluster
        can still be declined at codegen time (``cute_cluster_feasible``), and the
        loop bound, the per-CTA split and the staging size are all emitted from
        THIS number -- so reading the request here would round to a granularity the
        emitted kernel does not use.  The mask, by contrast, is created from the
        REQUEST (in ``__init__``, before the emitted value exists), which is the
        safe asymmetry: request >= emitted, so the mask can only be redundant,
        never missing.
        """
        if self._cute_tv_plan is None:
            return None
        env = CompileEnvironment.current()
        numel = env.block_sizes[self.block_index].numel
        if not isinstance(numel, (int, sympy.Integer)):
            return None
        cluster_n = self._cute_cluster_emitted_n()
        rounded = rounded_extent(int(numel), self._cute_tv_chunk, cluster_n)
        if rounded == int(numel):
            return None
        # A rounded tile is only sound if the identity gate exists at the combine
        # (invariant I4), and that gate is ``self._mask_var``.  If it is absent the
        # phantom columns would reach the accumulator ungated -- MEASURED wrong at
        # N=12288/chunk=4096/cluster=2.  ``__init__`` creates the mask against the
        # requested cluster, which dominates the emitted one, so this assert should
        # be unreachable; it is here because the alternative to crashing is a silent
        # wrong answer.
        assert self._mask_var is not None, (
            f"rounded tile {int(numel)} -> {rounded} without a reduction-axis mask: "
            "the out-of-range lanes would reach the combine ungated.  See "
            "ragged_tail invariant I4."
        )
        return rounded

    def cute_tv_tail_predicate(self) -> str | None:
        """``base < N`` for the current lane, or ``None`` when no tail exists.

        Returned as source text because it is spliced around a ``cute.copy``
        statement by ``memory_ops._cute_tv_partition_hoist``.  The variable it
        compares is the emitted per-thread column base
        (``reduction_lane_base_*``), which is a multiple of ``vec`` -- which is
        what makes one scalar compare exact for the whole fragment
        (``ragged_tail`` invariant I2).
        """
        if self.cute_tv_rounded_extent() is None:
            return None
        base_var = self._cute_lane_base_index_var
        if base_var is None:
            return None
        env = CompileEnvironment.current()
        numel = env.block_sizes[self.block_index].numel
        plan = self._cute_tv_plan
        assert plan is not None
        # I2's precondition, asserted at the point of emission: no vector block
        # may straddle the row end, because ``cute.copy``'s ``pred`` granularity
        # is one whole block.
        assert_vec_divides_extent(plan.vec, int(numel))
        return f"{base_var} < {int(numel)}"

    def _build_cute_tv_plan(
        self,
        fn: DeviceFunction,
        *,
        chunk: int,
        state_free: bool,
        vec_cap: int | None,
        block_index: int,
    ) -> ChunkTVPlan | None:
        """The ONE place a reduction's access width is decided.

        Returns ``None`` when this reduction is not a candidate for the TV path
        at all, in which case the caller keeps ``vec == 1`` -- i.e. today's
        scalar codegen.  Returning a plan with ``vec == 1`` and returning
        ``None`` are deliberately equivalent in effect: neither can produce a
        strided index without a matching width.

        ``state_free`` marks that this runs from ``__init__``, before any
        ``CodegenState`` exists, so only the device IR and the config may be
        consulted (that is the *point*: the width must be known before any
        address is emitted).  ``fn`` is passed explicitly for the same reason
        ``block_index`` is: this runs BEFORE ``super().__init__``, so ``self.fn``
        does not exist yet.  It is needed to read ``cute_cluster_n``, which sets
        the tile granularity the round-up must respect.

        ⭐ ``chunk`` IS A PARAMETER, NOT ``self._loop_block_size``.  That single
        field was this method's ONLY loop-specific input -- "the reduction-axis
        extent one iteration covers".  ``LoopedReductionStrategy`` passes its
        ``_loop_block_size``; a strategy with no outer reduction loop passes its
        own padded extent.  Everything else here is ``env``, the config, and
        ``_cute_layout_participants`` (a device-IR walk that reads no ``self``
        state), which is what makes the whole capability class-independent.

        Per ``PORT_SPEC_layout.md`` §2a / E010 trap 3 the width is bounded by
        the WIDEST dtype partitioned through the layout, and per §6c / E010
        trap 1 by each tensor's REAL row stride -- not by ``N``.  A sliced input
        whose row stride is not a multiple of ``vec`` is an IR-verification
        failure at compile time, so the clamp is what turns "ICE" into
        "narrower copy".
        """
        assert state_free
        env = CompileEnvironment.current()
        if env.backend.name != "cute":
            return None
        if chunk <= 0 or self._reduction_thread_count() <= 0:
            return None
        # ── RAGGED N: round the TILE up, predicate the TAIL ────────────────────
        #
        # ``numel`` need NOT be an exact multiple of the chunk.  When it is not,
        # the loop walks the rounded-up extent ``N' = ceil(N/G)*G`` and every
        # ``cute.copy`` on a chunk that can exceed ``N`` is guarded by
        # ``if base < N:``.  ``ragged_tail`` carries the whole argument, including
        # why one scalar compare per lane is exact (a fragment is wholly in or
        # wholly out of bounds) and why no per-op identity fill is needed here
        # (helion's ``_mask_to`` already supplies it, per op, AT THE COMBINE --
        # which is also what makes ``layer_norm``'s post-centering case correct).
        #
        # The historical comment here claimed "the per-element mask would have to
        # gate a whole fragment".  That was FALSE: on the TV path the per-element
        # index ``rindex = base + vi`` IS the element fragment slot ``vi`` holds,
        # so the existing per-element mask is already per-element.
        #
        # ``block_index`` is passed explicitly: this runs BEFORE
        # ``super().__init__``, so ``self.block_ids`` does not exist yet.
        numel = env.block_sizes[block_index].numel
        # ``cluster_n`` is read here rather than taken from
        # ``self._cute_cluster_n_requested`` because that field is assigned AFTER
        # this method runs (it depends on ``_loop_block_size``).  Both read the same
        # knob through the same method, so the granularity used to decide
        # admissibility and the granularity the cluster later demands cannot
        # disagree.
        #
        # ``rounded=True`` is required, not incidental: without it this asks about a
        # cluster that must divide ``N`` exactly, gets 1 at a ragged extent, and
        # therefore validates the round-up against granularity ``chunk`` while the
        # emitted kernel later rounds to ``chunk * cluster_n``.  Admissibility is
        # monotone DECREASING in ``cluster_n`` (a bigger granularity can only
        # overshoot more), so asking at the largest cluster in play is the
        # conservative direction: every cluster codegen can still settle on is a
        # divisor of it and is therefore also admissible.
        cluster_req = self._cute_cluster_n_config(
            fn, block_index, chunk=chunk, rounded=True
        )
        if not ragged_tile_admissible(env, numel, chunk, cluster_n=cluster_req, vec=1):
            return None
        # A read-after-write on the SAME tensor is a hazard: the TV path's load
        # copy is anchored ABOVE the per-element loop and its store flush BELOW
        # it, so a load that follows a store in program order reads the pre-store
        # gmem -- see ``_cute_tv_has_store_then_load_alias`` for the emitted code.
        #
        # ⭐ B3: it is resolved by FORWARDING where that is provably sound, and
        # only declined where it is not.  Both decisions are made HERE, in the ONE
        # place the width is decided, and never per-site: a per-site decline would
        # leave the trip count derived from ``plan.vec`` while an individual access
        # narrowed, which is class 1 exactly (see this method's docstring).  The
        # forwardable set is recorded on the strategy so the emission site cannot
        # re-derive it and drift.
        # ⚠ The VETO is asked through ``_cute_tv_has_store_then_load_alias`` rather
        # than by testing the classifier's ``None`` directly, even though the two
        # are the same predicate by construction.  That keeps ONE monkeypatchable
        # gate for "force the blanket decline", which is how the fail-capability
        # controls force this path off (``_notes/tests/test_b3_store_load_forward.py``
        # patches exactly this symbol, and so does the level-5 wiring).  The extra
        # IR walk is compile-time only and runs once per reduction.
        if _cute_tv_has_store_then_load_alias():
            return None
        # ``or frozenset()``, not an assert: the line above normally guarantees a
        # non-None result here, but a caller that patched the gate to False must get
        # today's un-forwarded emission rather than a compiler crash.
        self._cute_tv_forwarded_raw_keys = (
            _cute_tv_forwardable_raw_keys() or frozenset()
        )
        # ── THE WIDTH DECISION ITSELF IS NOT HERE ──────────────────────────────
        #
        # ⭐ Everything above this line is what makes THIS strategy's inputs; the
        # decision is :func:`tv_layout.build_tv_plan`, which every CuTe strategy
        # wanting a TV copy calls.  There used to be a second copy of the decision
        # on ``CuteNDTileStrategy``, and the two carried gates that differed by
        # accident of coverage rather than by design -- see that function's
        # docstring for the list.
        #
        # ``tail_predicated`` is the ragged-N policy: this path rounds the tile up
        # and predicates the tail, so ``vec`` must additionally divide ``numel``.
        # It is asked ONLY in the non-divisible case so the divisible path's width
        # bound is untouched, byte for byte.
        ragged = not env.known_multiple(numel, chunk) and isinstance(
            numel, (int, sympy.Integer)
        )
        return build_tv_plan(
            chunk=chunk,
            threads_per_row=self._reduction_thread_count(),
            participants=self._cute_layout_participants(),
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
            numel=int(numel) if ragged else None,
            tail_predicated=ragged,
            # ⚠ NOT ``require_exact_vec_cap``.  This caller reads ``lane_extent``
            # back OFF the returned plan (``self._cute_reduction_lane_extent =
            # plan.lane_extent``), so a narrowing re-derives the trip count with
            # the width and cannot skew.  A caller that fixed its trip count from
            # ``vec_cap`` BEFORE asking must pass it; this one must not.
        )

    def _cute_cluster_n_config(
        self,
        fn: DeviceFunction,
        block_index: int,
        *,
        chunk: int,
        rounded: bool = False,
        subdivides: bool = False,
    ) -> int:
        """Read ``cute_cluster_n``: the REQUEST half of the cluster decision.

        Split from :meth:`cute_cluster_feasible` for exactly the reason E017 trap 4
        records for ``reload_from``: this runs from ``__init__``, where
        ``thread_block_dims()`` is not final (sibling strategies do not exist yet, so
        the row axis still reports 1).  A feasibility test here would decline every
        config.  So this reads only knob-and-shape facts, and everything needing real
        geometry lives in the codegen-time check.

        Two declines are made here because both are pure functions of the knob:

        * ``cluster_n <= 1`` -- no cluster requested, nothing to do;
        * ``cap_cluster_n`` (quack ``reduction_base.py:28-40``) -- when
          ``threads_per_row * cluster_n`` exceeds the row's vector-block count, one
          CTA tile already spans the whole row, every peer would reduce the SAME
          columns, and the cluster combine would multiply the answer by
          ``cluster_n``.  Capping is not an optimisation; without it the result is
          silently wrong.

        ``rounded`` says the caller will cover a **rounded-up** extent
        ``N' = ceil(N/(chunk*cluster_n))*chunk*cluster_n`` and predicate the tail
        (``ragged_tail``).  Then the two divisibility declines below are vacuous by
        construction -- ``N'`` is a multiple of ``chunk * cluster_n`` -- so they are
        skipped rather than answered.  ⚠ It must stay a parameter and not be
        inferred: without the round-up the loop bound is the TRUE extent, the
        per-CTA share ``N // cluster_n`` is then ragged, and each CTA would silently
        cover the wrong columns.  Defaulting to ``False`` keeps every non-TV caller
        on the exact-division rule, so "I forgot to thread it through" fails closed.

        ⭐ WHY THIS IS NOT CIRCULAR even though the plan gate calls it.  Admissibility
        of the round-up is *monotone decreasing* in ``cluster_n`` (a smaller cluster
        means a smaller granularity, hence no more overshoot).  So the plan gate asks
        with ``rounded=True`` to learn the LARGEST granularity in play, and any
        cluster the codegen later settles on is a divisor of it and therefore also
        admissible.  Nothing here consults the plan.

        ``chunk`` is a PARAMETER for the same reason it is one on
        :meth:`_build_cute_tv_plan`: it was the only loop-specific input, and the
        two must read the SAME number or the granularity used to decide
        admissibility and the granularity the cluster later demands disagree.

        ⭐ ``subdivides`` -- THE THIRD SHAPE OF THE SAME QUESTION, AND THE ONE A
        LOOP-FREE STRATEGY ASKS.  Both branches below assume the caller's ``chunk`` is
        a *per-iteration* extent that the cluster MULTIPLIES: the total swept is
        ``chunk * iters * cluster_n``, so ``cluster_n`` adds a second divisibility
        requirement and a wider round-up granularity.  A PERSISTENT reduction is the
        other way round.  Its ``chunk`` is the whole padded extent
        ``next_pow2(N) == thread_count * lane_extent``, swept once, and a cluster
        **subdivides** that fixed total into ``cluster_n`` shares of
        ``chunk // cluster_n`` -- it does not enlarge it.  Passing that geometry to
        the non-rounded branch asks the WRONG question and always answers 1:
        MEASURED at N=1024, chunk=1024, cluster_n=2 the ``cap_cluster_n`` loop tests
        ``(1024 // 2) % 1024 == 512 != 0`` and halves to 1, i.e. it reads
        "one CTA tile already spans the row" from a chunk that describes ``cluster_n``
        CTAs' work rather than one CTA's.

        So under ``subdivides`` the requirement is the one that is actually load-bearing
        for a subdividing cluster, and it is stated on the quantity being subdivided:
        ``cluster_n`` must divide ``chunk``, so every CTA's share
        ``per_cta = chunk // cluster_n`` is a whole number of threads' worth of
        columns.  ⚠ AND THE SHARE MUST STILL BE ADDRESSABLE BY ONE CTA -- ``per_cta``
        threads is capped by the launch, so a share below one thread per column is
        declined rather than emitted.  Narrowing (halving) rather than refusing keeps
        this monotone in ``cluster_n`` like the other two branches, so the codegen-time
        decision can only ever be a divisor of what is returned here.

        ⚠ ``subdivides`` MUST BE A PARAMETER, not inferred from ``rounded`` or from the
        class.  It names a property of the CALLER'S LOOP NEST ("is my chunk the whole
        extent or one iteration of it?"), which no amount of shape inspection recovers:
        ``chunk == padded`` is *also* true of a looped strategy whose single iteration
        happens to cover the row, and there the multiplying reading is the correct one.
        Defaulting to ``False`` keeps every existing caller on the multiplying rule, so
        "I forgot to thread it through" fails closed -- the same convention ``rounded``
        uses one paragraph above.
        """
        env = CompileEnvironment.current()
        cluster_cfg = cast(
            "list[int]", fn.config.config.get("cute_cluster_n", []) or []
        )
        requested = env.config_spec.cute_cluster_n.config_get(
            cluster_cfg, block_index, 1
        )
        if not isinstance(requested, int) or requested <= 1:
            return 1
        numel = env.block_sizes[block_index].numel
        total = shape_env_size_hint(env.shape_env, numel)
        if total <= 0:
            return 1
        capability = env.config_spec.target_device_capability
        capped = min(
            requested,
            max_cluster_n_for_arch(capability[0] if capability is not None else None),
        )
        if chunk <= 0:
            return 1
        if subdivides:
            # The cluster PARTITIONS a fixed swept extent (see the ``subdivides``
            # paragraph).  ``chunk`` here is the whole padded extent, so the only
            # divisibility that matters is of ``chunk`` by ``cluster_n`` -- and because
            # ``chunk`` is a power of two (``thread_count * lane_extent``, both powers
            # of two) and ``cluster_n`` is too, this can only fail when the cluster
            # exceeds the extent.  Halve until the share is a whole number of columns
            # AND at least one column per thread of the CTA's own share.
            while capped > 1 and (chunk % capped or chunk // capped < 1):
                capped //= 2
            return max(1, capped)
        if rounded:
            # Every divisibility requirement below is satisfied by ``N'``.  What is
            # left is the *cost* of the round-up, which grows with the granularity,
            # so narrow the cluster until the overshoot is admissible.
            while capped > 1 and not ragged_tile_admissible(
                env, numel, chunk, cluster_n=capped, vec=1
            ):
                capped //= 2
            return max(1, capped)
        # The cluster splits the REDUCTION extent across CTAs, so the row must divide
        # evenly among them; a ragged split would need the tail predicated, which
        # only the TV path does (``rounded=True`` above).
        if total % requested:
            return 1
        # quack's ``_cap_cluster_n``, on the chunk the strategy actually runs: the
        # per-CTA share of the row must stay a whole number of chunks.
        while capped > 1 and (total // capped) % chunk:
            capped //= 2
        return max(1, capped)

    def cute_cluster_feasible(self) -> int:
        """``cluster_n`` when a clustered launch is affordable AND safe, else 1.

        Called at codegen time, where the geometry is final.  Every decline below is
        either a correctness requirement or a **hang** requirement -- per the brief, a
        hang is worse than a slow kernel, so an infeasible request is declined at
        compile time rather than emitted and discovered at runtime:

        1. **The grid must be a clean product.**  A clustered launch requires the grid
           to be a multiple of the cluster shape; CUDA fails the launch otherwise.
           Because the cluster occupies grid.y and grid.y IS ``cluster_n`` here, that
           holds by construction -- the grid is widened to ``(..., cluster_n, 1)`` in
           ``DeviceFunction.codegen_function_call`` -- so there is no rounding that
           could leave a partial cluster.
        2. **The intra-CTA combine must be the grouped one.**  The cluster step
           composes ON TOP of ``_cute_grouped_reduce_shared_two_stage``, so that path
           must have been taken (``thread_count > 32``, linear lane index).  Below a
           warp there is no group geometry to hang the exchange off.

        ⚠ NOT a decline: several reductions in one kernel.  Each clustered reduction
        needs its OWN mbarrier -- ``mbarrier_wait(mbar, 0)`` waits for phase 0, so a
        second reduction reusing one barrier either hangs or falls straight through
        with a stale buffer.  MEASURED: layer_norm emits **2** ``_cute_cluster_reduce``
        calls against **1** ``_cute_cluster_mbar_alloc``, because it is ONE
        ``LoopedReductionStrategy`` with two reduction *nodes* (mean, then variance) --
        so a guard counting reduction *strategies* does not see it.  That is why the
        barrier is allocated per REDUCTION NODE (see ``_cute_cluster_mbar_var``) rather
        than per strategy, which is quack's own answer to the same problem
        (``rmsnorm.py:302`` passes ``mbar_ptr + 1`` to its second moment).
        """
        if self._cute_cluster_n_requested <= 1:
            return 1
        env = CompileEnvironment.current()
        if env.backend.name != "cute":
            return 1
        # (2) the grouped intra-CTA combine must be reachable.
        if self._reduction_thread_count() <= _CUTE_WARP_REDUCTION_THREADS:
            return 1
        return self._cute_cluster_n_requested

    def cute_stage_widest_dtype_bits(self) -> int:
        """This reduction's participants -- see the base for why the SMEM charge needs this."""
        return max(
            (b for b in self._cute_layout_participants().dtype_bits if b > 0), default=0
        )

    def _cute_layout_participants(self) -> TVParticipants:
        """The tensors partitioned through the layout, for :func:`build_tv_plan`.

        Walks the device IR for ``hl.load``/``hl.store`` sites whose subscript
        reaches this reduction's axis.  Rank-1 side outputs indexed only on the
        row (``inv_rms[tile_m]``) are excluded: they are not partitioned through
        the row layout, and including their (often fp32) width would silently
        halve ``vec`` on a mixed-dtype kernel (E010 trap 3).

        The *stride* recorded for each tensor is the stride of its stride-1-most
        non-contiguous dim -- i.e. the row stride for a 2-D row-major tensor,
        which is the quantity ``legal_vec`` clamps on.

        ⭐ THE RETURN TYPE IS SHARED (:class:`tv_layout.TVParticipants`) WHILE THE
        WALK IS NOT, and that split is the whole point.  This walk recognises its
        axis by the subscript's *syntax* (the bare ``slice(None)``);
        ``CuteNDTileStrategy._cute_ndtile_layout_participants`` recognises its
        axis by ``block_id`` *identity*.  Those are genuinely different questions
        and one walk answering both is what made this path hard to widen.  What
        was duplicated was never the walk -- it was the *interpretation* of what
        the walk returned (one dtype? which stride fold? which width bound?), and
        that now lives once, in :func:`tv_layout.build_tv_plan`.

        ⚠ IT SELECTS ON ``slice(None)``, WHICH IS WHY MOVING IT TO THE BASE CLASS
        DOES NOT BY ITSELF MAKE THE TV PLAN AVAILABLE TO A **TILED** REDUCTION
        AXIS.  The predicate below is ``any(isinstance(idx, slice) and idx ==
        slice(None) for idx in subscript)``: it recognises ``x[tile_m, :]`` and
        rejects ``x[tile_m, tile_n]``, because an explicit inner tile index is a
        ``SymInt``, not a slice.

        MEASURED (2026-07-29), calling this method on a live
        ``CuteNDTileStrategy`` from inside its own ``codegen_device_loop``, for
        ``for tile_m: for tile_n: acc += x[tile_m, tile_n].sum(-1)`` at
        ``block_sizes=[1, 512] num_threads=[0, 32] cute_vector_widths[n]=8``:

            participants -> ([], [])          # <- EMPTY
            lane-body scaffolding present     # _cute_lane_body_by_block={1},
                                              # _cute_vec_lane_var_by_block={1},
                                              # vec width 8
            emitted: cute.copy=0  make_tiled_copy_tv=0  cute.arch.load=2

        vs the looped reference at ``x[tile_m, :]``, same probe:

            site eligible -> True for subscript ('u0', 'slice(None, ...)')
            emitted: cute.copy=1  make_tiled_copy_tv=1  cute.arch.load=0

        ⇒ ``_build_cute_tv_plan`` returns ``None`` at its ``if not dtypes`` guard
        for an NDTile axis EVEN THOUGH the lane-body owner, the constexpr-V-loop
        var and the coverage bijection are all already in place -- i.e. "the only
        missing piece is a plan-construction site" is FALSE for that strategy.
        ``memory_ops._cute_tv_site_eligible`` carries the same restriction
        independently (it requires the last non-``None`` subscript to be
        ``slice(None)``), so both the plan and every emission site would have to
        learn the tiled-axis shape together -- and they must learn it TOGETHER,
        because a plan built for a width no site can honour is class 1 exactly.
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
                # Only tensors indexed along the reduction axis participate.  A
                # subscript that never mentions that axis (``inv_rms[tile_m]``) is a
                # rank-1 side output and must NOT contribute its (often fp32) width,
                # which would silently halve ``vec`` on a mixed-dtype kernel.
                #
                # ⭐ TWO SPELLINGS OF "THIS AXIS", AND BOTH ARE THE AXIS:
                #
                #  * ``slice(None)`` -- the ROLLED spelling, ``x[tile_m, :]``, where the
                #    reduction owns the whole trailing dim and no inner tile index exists;
                #  * an explicit ``SymInt`` tile index whose block id IS this reduction's
                #    axis -- the TILED spelling, ``x[tile_m, tile_n]``.
                #
                # Accepting only the first is what made this method return ``([], [])``
                # for a tiled axis, so ``_build_cute_tv_plan`` bailed at its
                # ``if not dtypes`` guard and no TV plan was ever built for
                # ``CuteNDTileStrategy`` -- even though its lane-body owner, its
                # constexpr-V-loop var and the coverage bijection were all already in
                # place.  MEASURED before this change, on a live ``CuteNDTileStrategy``
                # at ``block_sizes=[1, 512] num_threads=[0, 32] vw=8``:
                #     participants -> ([], []) ; emitted cute.copy=0 tiled_copy_tv=0
                # against the rolled reference at ``x[tile_m, :]``: cute.copy=1, tv=1.
                #
                # ⚠ THE BLOCK-ID TEST IS LOAD-BEARING, not a formality.  Matching any
                # ``SymInt`` would admit the ROW index (``tile_m``) and every unrelated
                # tile axis, so a rank-1 side output would start contributing its width
                # -- E010 trap 3, the mixed-dtype ``vec`` halving. Only the id equal to
                # this reduction's own axis counts.
                #
                # ⚠ AND ``memory_ops._cute_tv_site_eligible`` CARRIES THE SAME
                # RESTRICTION INDEPENDENTLY.  The two must agree: a plan built at a width
                # no emission site honours is bug class 1 exactly (the trip count would
                # assume a width the access does not use), so that predicate is widened
                # in the same change.
                # ⚠ THIS SELECTOR IS DELIBERATELY SYNTAX-BASED AND ROLLED-ONLY.  It
                # recognises its axis as "the bare ``slice(None)``", which is unambiguous
                # for a rolled reduction because the axis is the only bare slice in play.
                #
                # ⛔ DO NOT WIDEN IT TO ACCEPT A TILED (``SymInt``) AXIS.  I tried, and it
                # was wrong twice over: (a) in the DEVICE IR a subscript entry is an
                # ``fx.Node`` whose ``meta["val"]`` holds the SymInt -- entries arrive as
                # ``[Node:sym_size_int, Node:block_size_1]`` -- so a bare
                # ``isinstance(idx, torch.SymInt)`` test matches NOTHING and the widening
                # is a SILENT NO-OP indistinguishable from an honest decline; and (b) even
                # unwrapped correctly it is the wrong home, because a tiled axis must be
                # recognised by IDENTITY (``block_id`` equality) while this one recognises
                # by FORM, and one walk answering both gives it two notions of "its axis"
                # with one caller for each -- exactly the coupling that made this path hard
                # to widen in the first place.
                #
                # ⇒ The tiled analogue is a SEPARATE walk,
                # ``CuteNDTileStrategy._cute_ndtile_layout_participants``, which shares only
                # the ``_tiled_axis_block_id`` unwrapping with the codegen-time gate so that
                # neither can be silently vacuous.  See its docstring.
                if not any(
                    isinstance(idx, slice) and idx == slice(None) for idx in subscript
                ):
                    continue
                val = None
                if isinstance(tensor_node, torch.fx.Node):
                    val = tensor_node.meta.get("val")
                elif isinstance(tensor_node, torch.Tensor):
                    val = tensor_node
                if not isinstance(val, torch.Tensor):
                    continue
                accesses.append(
                    (
                        _dtype_str(val.dtype),
                        val.element_size() * 8,
                        tuple(val.stride()),
                        val.dtype,
                    )
                )
        return TVParticipants.from_accesses(accesses)

    def _cute_cluster_emitted_n(self) -> int:
        """``cluster_n`` for THIS kernel, decided once and cached.

        Idempotent and keyed on the strategy, like ``cute_stage_feasible``: the
        combine emitter, the loop-bound emitter and the launch all ask, and they must
        all get the same answer or the kernel and its launch geometry disagree -- which
        is the failure mode that hangs rather than the one that computes wrongly.
        """
        if self._cute_cluster_n_emitted > 1:
            return self._cute_cluster_n_emitted
        cluster_n = self.cute_cluster_feasible()
        if cluster_n > 1:
            self._cute_cluster_n_emitted = cluster_n
            # The launch side already exists end-to-end: ``cute_state.cluster_shape``
            # is emitted as ``_helion_cute_cluster_shape`` by
            # ``generate_ast.py:1530-1536`` and consumed by
            # ``runtime/__init__.py:3292-3309`` -> ``:3414-3416``.  Setting it is the
            # whole of the plumbing this leg needed.
            self.fn.cute_state.cluster_shape = (1, cluster_n, 1)
        return cluster_n

    def _cute_cluster_y_var(self, state: CodegenState) -> str:
        """``block_idx()[1]`` -- this CTA's rank along the cluster's N axis.

        quack ``rmsnorm.py:162`` (``cluster_y``).  Deliberately ``block_idx()[1]`` and
        NOT ``block_idx_in_cluster()``: the two agree only because the cluster shape is
        ``(1, cluster_n, 1)`` and grid.y IS ``cluster_n``, but the *tile* coordinate is
        a grid property while ``block_idx_in_cluster`` is a cluster property, and they
        would diverge the moment grid.y carried anything else.  The exchange in
        ``reduce_helpers`` uses ``block_idx_in_cluster()`` because there the cluster
        rank is what indexes the buffer; here the grid coordinate is what indexes the
        DATA.

        ⚠ NOT reusable as the staging row coordinate (E017 trap 6 / trap 4 in this
        commit's notes): DSMEM peer addressing and the per-CTA staging buffer are
        different address spaces, and the staging tile's row stays ``thread_idx()[1]``.

        ⭐ ON THE BASE CLASS, with ``_cute_cluster_mbar_var`` below it, because the
        cluster is a capability and not a class: BOTH looped and persistent need this
        CTA's rank (looped to offset its ``for roffset`` bound, persistent to offset
        its ``thread_idx()``-derived index), and the var must be allocated ONCE per
        strategy whichever asks.
        """
        existing = self._cute_cluster_y_name
        if existing is not None:
            return existing
        fn = state.device_function
        cy_var = fn.new_var("_cluster_y", dce=False)
        fn.preamble.append(
            statement_from_string(f"{cy_var} = cutlass.Int32(cute.arch.block_idx()[1])")
        )
        self._cute_cluster_y_name = cy_var
        return cy_var

    def _cute_cluster_mbar_var(self, state: CodegenState) -> str:
        """Allocate a mbarrier for THIS reduction node in the kernel preamble.

        ⚠ ONE BARRIER PER REDUCTION NODE, keyed on the fx node -- NOT one per
        strategy.  ``mbarrier_wait(mbar, 0)`` waits for phase 0, so two reductions
        sharing one barrier either hang or fall straight through onto a stale buffer.
        MEASURED: layer_norm emitted **2** ``_cute_cluster_reduce`` calls against
        **1** barrier before this was keyed, because it is ONE
        ``LoopedReductionStrategy`` carrying TWO reduction nodes (mean, then variance)
        -- so a guard that counts reduction *strategies* cannot see it, which is
        exactly what the first version of ``cute_cluster_feasible`` tried to do.
        quack has the identical constraint and the identical answer: allocate ``stage``
        barriers and hand the second moment ``mbar_ptr + 1`` (``rmsnorm.py:302``).

        In the PREAMBLE, and unconditionally, because ``alloc_smem`` is a trace-order
        bump allocator: every CTA of the cluster must compute the SAME shared address
        for a given barrier.  ``mapa.shared::cluster`` maps a local address into a
        peer, so a divergent allocation order would silently map to a different object
        there.  Allocating per node in one fixed order keeps every CTA's SMEM layout
        identical.

        ⚠ THE PER-NODE DICT IS LAZILY CREATED HERE, not in a subclass ``__init__``.
        The class default is ``None`` (the "no cluster state" sentinel every field in
        that block uses) so a strategy that never clusters carries no per-instance
        dict; the first emitter to ask materialises one on the instance.  A MUTABLE
        class-level ``{}`` default would be shared by every strategy in the process,
        which is how one kernel's barrier name leaks into another's.
        """
        node = state.fx_node
        key = id(node) if node is not None else 0
        names = self._cute_cluster_mbar_names
        if names is None:
            names = {}
            self._cute_cluster_mbar_names = names
        existing = names.get(key)
        if existing is not None:
            return existing
        fn = state.device_function
        mbar_var = fn.new_var("_cluster_mbar", dce=False)
        fn.preamble.append(
            statement_from_string(f"{mbar_var} = _cute_cluster_mbar_alloc()")
        )
        fn.preamble.append(
            statement_from_string(
                f"_cute_cluster_mbar_init({mbar_var}, "
                "cutlass.Int32(cute.arch.thread_idx()[0]) "
                "+ cutlass.Int32(cute.arch.thread_idx()[1]))"
            )
        )
        names[key] = mbar_var
        return mbar_var

    def _cute_cluster_per_cta_extent(self) -> int | None:
        """The reduction extent ONE CTA of the cluster covers, or None if not static.

        This is quack's ``tiler_mn[1]``: ``N' // cluster_n``, a whole number of chunks.

        ⭐ IT IS THE **ROUNDED** EXTENT THAT IS SPLIT, not the true one.  quack
        ``reduction_base.py:47`` divides the ``ceil_div``-ed block count, so its
        per-CTA tile is a whole number of vector blocks even at ragged ``N``; the
        tail then falls inside the LAST CTA's tile and is handled by the same
        predicate as the non-clustered case.  Splitting the *true* extent instead
        would give a ragged per-CTA share, and CTA ``cy``'s loop would start
        mid-chunk -- every CTA of rank > 0 covering the wrong columns, silently.
        Since ``N'`` is a multiple of ``chunk * cluster_n`` by construction
        (:func:`ragged_tail.tile_granularity`), this division is always exact.
        """
        env = CompileEnvironment.current()
        cluster_n = self._cute_cluster_n_emitted
        if cluster_n <= 1:
            return None
        total = shape_env_size_hint(
            env.shape_env, env.block_sizes[self.block_index].numel
        )
        if total <= 0:
            return None
        rounded = self.cute_tv_rounded_extent()
        if rounded is not None:
            total = rounded
        if total % cluster_n:
            return None
        return total // cluster_n


class PersistentReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
    ) -> None:
        from .device_ir import ReductionLoopGraphInfo

        env = CompileEnvironment.current()
        numel = env.block_sizes[block_index].numel
        # Skip the mask when RDIM_SIZE == numel (no padding needed).
        # This is true when numel is a power of 2 (Triton doesn't round),
        # or when the backend uses exact RDIM sizes (e.g., Pallas).
        needs_mask = True
        # Guard numel > 0: on PyTorch 2.9, next_power_of_2(0) returns 0
        # (the n <= 0 guard was added later), so static_rdim_size(0) == 0
        # would incorrectly skip the mask for zero-size reductions.
        if isinstance(numel, (int, sympy.Integer)) and int(numel) > 0:
            needs_mask = env.backend.static_rdim_size(int(numel)) != int(numel)
        mask_var: str | None = (
            fn.new_var(f"mask_{block_index}", dce=True) if needs_mask else None
        )
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_RDIM_SIZE_{block_index}"),
        )
        self.offset_vars[block_index] = "0"
        # Compute thread count for warp-level reductions
        max_threads = env.backend.max_reduction_threads()
        if max_threads is not None:
            if env.backend.name == "cute":
                max_threads = cute_live_reduction_threads(max_threads)
                # Indexed reductions (argmin/argmax) on CuTe only have a
                # warp-level reduction primitive that takes
                # ``threads_in_group <= 32``. Cap the persistent thread
                # count to the warp size so the emitted
                # ``cute.arch.warp_reduction`` is correct.
                if _block_has_indexed_reduction(fn, block_index):
                    max_threads = min(max_threads, _CUTE_WARP_REDUCTION_THREADS)
            if isinstance(numel, (int, sympy.Integer)):
                size_hint = int(numel)
            elif isinstance(numel, sympy.Expr):
                size_hint = shape_env_size_hint(env.shape_env, numel)
            else:
                size_hint = env.size_hint(numel)
            self._thread_count = next_power_of_2(min(size_hint, max_threads))
        else:
            self._thread_count = 0
        # On cute, the launch block dim is capped at MAX_THREADS_PER_BLOCK.
        # If the existing tile strategies already claim that budget, the
        # reduction's Y/Z axis silently collapses to 1, producing kernels
        # whose ``thread_idx[axis] + synthetic_lane * thread_count`` indexing
        # only covers ``padded_size // thread_count`` of the reduction extent.
        # Shrink ``_thread_count`` here so the full extent stays addressable
        # via the synthetic lane loop.
        # Tile strategies are added before reduction strategies, so they are
        # already on the dispatcher by the time we get here.
        tile_dispatch = getattr(fn, "tile_strategy", None)
        if tile_dispatch is not None:
            # Reductions in mutually-exclusive control-flow branches share a
            # thread axis (see ``TileStrategyDispatch._branch_by_control_flow``),
            # so they never co-execute and must not be multiplied into this
            # reduction's thread budget. Drop them before adjusting.
            concurrent = _strategies_concurrent_with_block(tile_dispatch, block_index)
            self._thread_count = env.backend.adjust_reduction_thread_count(
                self._thread_count, concurrent
            )
        # ⛔⛔ DO NOT ADD A ``cute_threads_per_row`` CLAMP HERE.  IT WAS TRIED, AND IT
        # PRODUCES SILENT WRONG ANSWERS.  Written out in full because the brief that asked
        # for it presents the opposite as VERIFIED, so the next reader will be tempted too.
        #
        # THE ASK ("step 0" of the capability rework): make
        # ``PersistentReductionStrategy`` read ``cute_threads_per_row`` and lower
        # ``_thread_count``, on the stated grounds that ``_thread_count`` is the whole
        # padded reduction extent, hence ``lane_extent == 1``, hence ``ChunkTVPlan``'s
        # coverage bijection admits only ``vec == 1`` and any TV-plan hoist is inert.
        #
        # ⭐ THE PREMISE IS FALSE, but ⚠ SO IS THE REBUTTAL THAT REPLACED IT.  Both
        # earlier claims here generalized from ONE kernel to "always", in opposite
        # directions.  RE-MEASURED (2026-07-29) by reading THIS OBJECT's
        # ``_thread_count`` / ``_synthetic_cute_lane_extent`` right after
        # ``__init__``, over a grid of (kernel, N, block_sizes, cute_vector_widths)
        # -- see ``_notes/tests/test_capability_matrix.py`` for the arm this pins:
        #
        #   kernel   N     bs  | thread_count  synthetic_lane_extent  emitted lane loop
        #   row_sum  256   1   |    256              1                    none
        #   row_sum  512   1   |    512              1                    none
        #   row_sum  768   1   |   1024              1                    none
        #   row_sum  1024  1   |   1024              1                    none
        #   rms      256   1/4 |    256              1                    none
        #   rms      1024  1   |   1024              1                    none
        #   (bs >= 4 at N >= 512, or bs=8 at N=256: PERSISTENT IS NOT USED AT ALL --
        #    ``normalize()`` rolls the reduction and the emitted kernel has
        #    ``for roffset``, i.e. ``LoopedReductionStrategy``.)
        #
        # ⇒ ON EVERY SHAPE WHERE PERSISTENT IS ACTUALLY THE STRATEGY,
        # ``_thread_count == next_power_of_2(N)`` and ``lane_extent == 1``, exactly
        # as the ORIGINAL brief said -- ``create_synthetic_reduction_lanes`` returns
        # ``None`` when ``padded_size <= thread_count``, so the warp cap 60 lines
        # above never fires (it is guarded by ``lane_extent is not None``).  The
        # "8/16/32" table above this was measured on a config that had been ROLLED
        # into the looped strategy, so it was reporting the looped path's lane
        # extent, not persistent's.  ``lane_extent > 1`` on persistent requires a
        # LOWER ``_thread_count``, and there is no shape at which it comes for free.
        #
        # ⛔ AND THAT DOES NOT MAKE THE CLAMP RIGHT -- the wrongness below is
        # independently measured and still stands.  What it means is that capability
        # ① on ``PersistentReductionStrategy`` is blocked on a CORRECTNESS story for
        # the warp reduction at a lowered thread count (see the ⇒ at the end), which
        # is a bigger piece of work than a config read, and NOT on the plan
        # construction site (that now exists on the base class:
        # ``ReductionStrategy._build_cute_tv_plan``, parameterized by ``chunk``).
        #
        # The two OTHER blockers are real and independent of the thread count:
        # (a) a lane-body owner -- ``_cute_lane_body`` /
        # ``_cute_tv_constexpr_loop`` / ``_cute_tv_chunk_prefix``, which
        # ``memory_ops._cute_tv_partition_hoist`` mutates DURING load lowering,
        # whereas persistent's lane loop is built at the END of codegen by
        # ``DeviceGridState.wrap_body``; and (b) ⭐ NEWLY FOUND, not in any
        # document: ``split_lane_loop_reductions`` finds ``_helion_lane_reduce``
        # markers only among the lane loop's DIRECT children
        # (``_split_one_lane_loop`` enumerates ``loop.body``), and the TV protocol
        # nests the marker inside ``for vi in cutlass.range_constexpr(vec)``.  A
        # nested marker is not rewritten; it is reverted to the raw per-lane input
        # by ``restore_unprocessed_lane_reduce_markers``, which DROPS the cross-lane
        # accumulator -- a silent wrong answer of exactly the class that comment
        # calls "bug class 8 P1".  So giving persistent a TV plan requires teaching
        # that pass the nested shape FIRST.
        #
        # ⭐ AND THE CLAMP IS ACTIVELY WRONG.  MEASURED with it in place:
        # ``test_attention_block_pointer`` returns 683292 of 4194304 elements wrong (max
        # abs err 0.923 vs SDPA), and 16 of 118 ``test_examples.py`` tests fail -- 7 of the
        # 8 attention examples.  Bisected to this clamp ALONE: reverting only it restores
        # ``mismatched=0/4194304`` with every other change of that session still in place.
        # MECHANISM: attention's reduction arrives with ``cute_threads_per_row=[8]``, so
        # the clamp cut its thread count 32 -> 8 and the warp reduction then covered a
        # fraction of the axis -- exactly the #2643 hazard the cap above exists to prevent.
        #
        # ⚠ AND THE OBVIOUS SAFEGUARD DOES NOT WORK EITHER.  Gating on "the caller asked
        # explicitly", i.e. ``"cute_threads_per_row" in fn.config.config``, looks like it
        # confines the change to opt-in callers.  It does not: ``fn.config.config`` is the
        # NORMALIZED config, not the caller's.  MEASURED -- the attention test passes only
        # ``block_sizes`` / ``num_stages`` / ``indexing``, yet inside this ``__init__`` the
        # key is present with the ladder value ``[8]``.  ⇒ there is no cheap way to tell
        # "requested" from "filled in" here; that is the same trap LEDGER E067 records on
        # the looped path, which is why the looped clamp also gates on ``requested_vec > 1``.
        #
        # ⇒ The prerequisite for persistent honouring this knob is a correctness story for
        # the warp reduction at a lowered thread count -- NOT a config read.
        self._synthetic_cute_lane_var: str | None = None
        self._synthetic_cute_lane_extent = 1
        is_graph_reduction_dim = any(
            isinstance(graph, ReductionLoopGraphInfo) and block_index in graph.block_ids
            for graph in fn.codegen.codegen_graphs
        )
        if self._thread_count > 0:
            if isinstance(numel, (int, sympy.Integer)):
                size_hint = int(numel)
            elif isinstance(numel, sympy.Expr):
                size_hint = shape_env_size_hint(env.shape_env, numel)
            else:
                size_hint = env.size_hint(numel)
            # For a non-graph-reduction dim we always try to recover the
            # full extent through a synthetic lane loop. For a graph
            # reduction dim we only need the synthetic lane loop when
            # ``adjust_reduction_thread_count`` shrank ``thread_count``
            # below the (padded) reduction extent — otherwise every logical
            # lane is already backed by a live thread and the warp/cross-warp
            # reduction covers the whole axis. Without this, a shrunk
            # graph-reduction dim only addresses the first ``thread_count``
            # elements (e.g. layer_norm_bwd's feature axis), leaving the
            # remaining columns/partial sums uncomputed.
            # ⚠ NOTE THE POSITION: this must NOT be nested under the ``#2643`` cap's
            # ``lane_extent is not None`` guard.  ``create_synthetic_reduction_lanes``
            # returns ``None`` when ``padded <= thread_count`` -- which is precisely
            # persistent's own case (``thread_count == padded``), so at N=1024
            # ``lane_extent`` is ``None``, the cap above never fires, and a clamp placed
            # inside that branch is unreachable.  MEASURED on the first attempt at this
            # edit: ``thread_count=1024 synthLE=1 plan=None``, i.e. it silently did
            # nothing.  Applied here it runs unconditionally, then re-derives the loop.
            requested_tpr = env.config_spec.cute_threads_per_row.config_get(
                cast(
                    "list[int]",
                    fn.config.config.get("cute_threads_per_row", []) or [],
                ),
                block_index,
                None,
            )
            requested_vec = env.config_spec.cute_vector_widths.config_get(
                cast(
                    "list[int]",
                    fn.config.config.get("cute_vector_widths", []) or [],
                ),
                block_index,
                1,
            )
            if (
                env.backend.name == "cute"
                and isinstance(requested_tpr, int)
                and isinstance(requested_vec, int)
                and requested_vec > 1
                and 0 < requested_tpr < self._thread_count
                # A power of two, so ``padded // tpr`` is exact and the lane
                # arithmetic stays integral.  ``_normalize`` already enforces
                # membership in ``THREADS_PER_ROW_CHOICES`` (all powers of two);
                # this is belt-and-braces for a hand-written config.
                and requested_tpr & (requested_tpr - 1) == 0
            ):
                self._thread_count = requested_tpr
                # ⭐ THE COVERAGE ARGUMENT, and it is why this is sound where the
                # earlier attempt was not: ``create_synthetic_reduction_lanes`` returns
                # ``padded // thread_count``, so the loop grows by exactly the factor
                # the count shrank and ``thread_count * lane_extent == padded`` still
                # holds.  Every element of the axis is still visited, and the warp
                # reduction now spans ``requested_tpr <= 32``-style groups it can
                # actually cover rather than the whole padded extent.
                lane_extent = env.backend.create_synthetic_reduction_lanes(
                    self._thread_count, size_hint
                )
                assert lane_extent is not None and (
                    self._thread_count * lane_extent
                    >= next_power_of_2(max(1, size_hint))
                ), (
                    "lowered thread count does not cover the padded axis: "
                    f"tpr={self._thread_count} lane_extent={lane_extent} "
                    f"padded={next_power_of_2(max(1, size_hint))}"
                )
            # ⛔⛔ COMPARE AGAINST THE **REAL** EXTENT, NOT THE THREAD-CAPPED ONE.
            #
            # This used to read ``next_power_of_2(min(size_hint, max_threads))``, and the
            # ``min`` made the test unable to notice the very case it exists to catch.
            # ``_thread_count`` is itself capped at ``max_threads`` (1024 for cute), so at
            # any ``size_hint > 1024`` both sides collapse to 1024, the test is
            # ``1024 < 1024`` == False, NO synthetic lane loop is created, and the
            # reduction silently visits only the first 1024 elements.
            #
            # ⛔ MEASURED as a SILENT WRONG ANSWER, with {-1,+1} integer data so a correct
            # kernel must be bit-exact: an inner ``sum(x[tm, :])`` at
            # ``reduction_loops=[None]``, ``cute_vector_widths=[1,1]`` returned
            # **exactly the sum of the first 1024 columns** -- ``torch.equal(got,
            # x[:, :1024].sum(-1))`` was True at N=2048 and N=4096 (``got[0]=10`` vs
            # ``ref[0]=46``). N<=1024 was correct, so the defect begins exactly where the
            # cap binds. 31744 of 32768 elements were dropped at N=32768.
            #
            # ⚠ WHY NOBODY SAW IT: the four ``reduction_loops=[None]`` clamp sites
            # (``config_spec`` x3, ``tile_dispatch`` x1) roll any config whose per-CTA
            # extent exceeds the thread budget, so this branch is unreachable on
            # ``origin/main`` and on this branch -- VERIFIED, both are correct here. It is
            # reachable only once those clamps are removed, which is why it is fixed FIRST:
            # the comment at ``config_spec.py``'s site 2 cites exactly this hazard
            # ("a persistent reduction would hit the synthetic-lane-loop bug") as the
            # reason those sites must roll.
            #
            # ⭐ ``create_synthetic_reduction_lanes`` was always right -- ``(1024, 2048) ->
            # 2``, ``(1024, 4096) -> 4``. Only this caller declined to ask it. With the
            # ``min`` gone the lane loop fires (``tc=32 lane_ext=64/128/1024`` at
            # N=2048/4096/32768) and every N tested is bit-exact.
            #
            # INERT on everything currently reachable: 0 of the 40 frozen cells change
            # emission (0 errors), the cute suites give an identical failure set, and
            # ``test_examples.py`` is 101 passed / 0 failed -- including all 8 attention
            # examples, which are the historically fragile consumer of this block.
            needs_synthetic = not is_graph_reduction_dim or (
                self._thread_count < next_power_of_2(size_hint)
                if max_threads is not None
                else False
            )
            if needs_synthetic:
                lane_extent = env.backend.create_synthetic_reduction_lanes(
                    self._thread_count, size_hint
                )
                # The synthetic lane loop folds each lane into a per-thread
                # accumulator that is then combined with a single warp reduction
                # over ``_reduction_thread_count`` threads. That warp reduction
                # is only correct within one warp, so cap the thread count to
                # the warp size and let the lane loop grow to cover the rest;
                # otherwise a multi-warp group silently drops lanes (#2643).
                if (
                    lane_extent is not None
                    and self._thread_count > _CUTE_WARP_REDUCTION_THREADS
                ):
                    self._thread_count = _CUTE_WARP_REDUCTION_THREADS
                    lane_extent = env.backend.create_synthetic_reduction_lanes(
                        self._thread_count, size_hint
                    )
                # ⭐ CAPABILITY ① PREREQUISITE: HONOUR ``cute_threads_per_row`` **HERE**,
                # INSIDE THIS BLOCK, SO THE LANE LOOP IS RE-DERIVED TO COVER THE REMAINDER.
                #
                # Why persistent needs this at all: ``_thread_count`` above is the whole
                # padded extent, so ``chunk == threads_per_row`` and
                # ``lane_extent = chunk // (tpr * vec)`` collapses to 1 -- and
                # ``ChunkTVPlan.__post_init__``'s coverage bijection
                # (``chunk % (threads_per_row * vec) == 0``) then admits only ``vec == 1``,
                # i.e. a scalar load.  A TV plan of width 1 is indistinguishable from no
                # plan, so ① is unreachable until ``lane_extent > 1``.
                #
                # ⛔ AND THIS IS WHERE AN EARLIER ATTEMPT WENT WRONG -- read before editing.
                # Lowering ``_thread_count`` *outside* this block (before the
                # ``needs_synthetic`` computation, as a standalone config read) produced
                # SILENT WRONG ANSWERS: ``test_attention_block_pointer`` returned
                # 683292/4194304 elements wrong, and 16 of 118 ``test_examples.py`` tests
                # failed (7 of the 8 attention examples), bisected to that change alone.
                # The cause is exactly the invariant the ``#2643`` cap above maintains:
                # the warp reduction covers ``_thread_count`` CONSECUTIVE lanes, so cutting
                # the count without regrowing the lane loop leaves the remaining
                # ``padded/new_count - padded/old_count`` lanes' worth of the axis
                # **never reduced**.
                #
                # ⇒ THE FIX IS NOT A DIFFERENT GATE, IT IS A DIFFERENT PLACE.  Applied here,
                # ``create_synthetic_reduction_lanes(new_count, size_hint)`` re-derives
                # ``lane_extent = padded // new_count`` immediately below, so the loop grows
                # by exactly the factor the count shrank and total coverage
                # (``thread_count * lane_extent == padded``) is preserved by construction --
                # which is the same argument that makes the ``#2643`` cap itself sound.
                #
                # ⚠ ``requested_vec > 1`` IS REQUIRED, and not for symmetry with the looped
                # clamp.  ``CuteThreadsPerRowSpec._fill_missing`` returns a real ladder value
                # (``threads_per_row_for(n)``), and ``fn.config.config`` is the NORMALIZED
                # config -- MEASURED, attention passes only ``block_sizes``/``num_stages``/
                # ``indexing`` yet the key is present here as ``[8]``.  So "is the key
                # present" cannot distinguish requested from filled-in, and honouring it
                # unconditionally would re-geometry every persistent kernel in the tree
                # (LEDGER E067's trap).  Gating on ``vec > 1`` keeps the lever where a TV
                # plan is actually wanted: at ``vec == 1`` there is nothing to enable, so
                # the ladder default stays inert and every existing kernel is untouched.
                if lane_extent is not None:
                    self._synthetic_cute_lane_var = fn.new_var(
                        f"synthetic_lane_{block_index}",
                        dce=False,
                    )
                    self._synthetic_cute_lane_extent = lane_extent
        # ── CAPABILITY GEOMETRY: what "one chunk" means with no outer loop ─────
        #
        # ``_cute_tv_chunk`` is the base class's single loop-specific input (see
        # ``ReductionStrategy._build_cute_tv_plan``).  Persistent covers the whole
        # padded extent in ONE pass, so its chunk IS that extent -- which is also
        # what ``_index_init_expr`` and the launch block dim are built from.
        #
        # Recorded even when no capability engages, because it is the number every
        # base-class capability would read and it must not be inferred at two sites.
        if env.backend.name == "cute" and self._thread_count > 0:
            padded = self._thread_count * self._synthetic_cute_lane_extent
            self._cute_tv_chunk = padded
            # ⭐ CAPABILITY ② (cluster/DSMEM): the REQUEST half, on persistent.
            #
            # Safe to read unconditionally here, unlike ``cute_threads_per_row``:
            # ``CuteClusterNSpec._fill_missing`` returns **1** for an omitted slot,
            # and its own comment records WHY that asymmetry exists ("clustering
            # changes the launch geometry, so a hand-written config that never
            # named it must not get a cluster it did not ask for (run-2 E067)").
            # VERIFIED by reading that spec: the E067 trap that makes a
            # ``cute_threads_per_row`` read here unsound does not apply.
            #
            # ⭐ ``subdivides=True`` -- AND IT IS THE WHOLE REASON THIS REQUEST WAS
            # PREVIOUSLY ALWAYS 1.  Persistent sweeps ``padded`` columns ONCE, so a
            # cluster PARTITIONS that fixed total; the looped path's ``chunk`` is a
            # per-ITERATION extent that the cluster MULTIPLIES.  Handing persistent's
            # geometry to the multiplying rule asks the wrong question and always
            # declines -- MEASURED at N=1024/chunk=1024/cluster_n=2: ``cap_cluster_n``
            # tests ``(1024 // 2) % 1024 == 512 != 0`` and halves to 1, i.e. it read
            # "one CTA tile already spans the row" off a chunk that describes
            # ``cluster_n`` CTAs' work.  See the ``subdivides`` paragraph on
            # ``_cute_cluster_n_config``.
            self._cute_cluster_n_requested = self._cute_cluster_n_config(
                fn, block_index, chunk=padded, rounded=False, subdivides=True
            )
            # ── THE PER-CTA EXTENT SPLIT, WHICH IS WHAT MAKES THE EXCHANGE SOUND ────
            #
            # Persistent has no ``for roffset`` whose bound the cluster can edit (that
            # is the looped path's ``loop_begin = cy * per_cta``).  Its index is
            # ``thread_idx()[0] + synthetic_lane * thread_count``, which covers the
            # WHOLE extent in EVERY CTA -- so emitting the exchange without splitting
            # would make every peer reduce the SAME columns and multiply the answer by
            # ``cluster_n``.  The split therefore happens in the GEOMETRY: this CTA
            # owns ``per_cta = padded // cluster_n`` columns, and ``codegen_preamble``
            # offsets the index by ``cluster_y * per_cta`` so the ``cluster_n`` shares
            # tile ``[0, padded)`` exactly once.
            #
            # ⭐ WHY LOWERING ``_thread_count`` IS SOUND HERE, WHERE THE
            # ``cute_threads_per_row`` CLAMP WAS NOT.  The ⛔ block above records that
            # cutting the count leaves ``padded/new - padded/old`` lanes' worth of the
            # axis NEVER REDUCED (#2643, MEASURED: 683292/4194304 elements wrong).  That
            # argument turns on WHO covers the columns the shrunken count drops.  For
            # the clamp the answer was "nobody", which is the bug.  Here it is "the
            # other CTAs of the cluster, and the DSMEM combine folds their partials",
            # so total coverage is
            #
            #     cluster_n * thread_count' * lane_extent' == padded
            #
            # which is asserted below rather than trusted.  ⇒ the cluster is not an
            # exception to #2643; it SATISFIES it, with the cross-CTA combine playing
            # the role the lane loop plays there.
            #
            # ⚠ ``cluster_n`` READ HERE IS THE **REQUEST**, and the asymmetry is
            # deliberate and fail-safe in ONE direction only.  ``cute_cluster_feasible``
            # runs at codegen and can only DECLINE (request >= emitted), so a geometry
            # split for a cluster that is later declined would leave each CTA covering
            # ``padded/cluster_n`` with no peers to supply the rest -- a WRONG ANSWER,
            # not a slow one.  So the two must agree, and they are made to agree by
            # construction: every decline in ``cute_cluster_feasible`` is re-checked
            # HERE, against this same geometry, before the split is committed.  The
            # ``_thread_count > 32`` one is the only geometric decline it makes, and it
            # is evaluated on the POST-split count because that is the count the
            # emitted combine will use.
            if self._cute_cluster_n_requested > 1:
                cluster_n = self._cute_cluster_n_requested
                # ⚠⚠ THE EXTENT SPLIT IS TAKEN FROM ``next_power_of_2(size_hint)``, NOT
                # FROM THE LOCAL ``padded`` ABOVE, AND THE DIFFERENCE IS A MEASURED
                # SILENT WRONG ANSWER.  ``padded`` is
                # ``_thread_count * _synthetic_cute_lane_extent``, which EQUALS the
                # padded extent only when a synthetic lane loop was created to cover the
                # surplus.  It is not always: ``needs_synthetic`` is
                # ``not is_graph_reduction_dim or thread_count < next_pow2(min(N, max))``,
                # and for a GRAPH reduction dim at N > max_reduction_threads both
                # disjuncts are false (``1024 < next_pow2(min(2048, 1024)) == 1024`` is
                # False), so ``lane_extent`` stays 1 and ``padded`` reports 1024 for a row
                # of 2048.
                #
                # That was HARMLESS before, and only because the force-roll guaranteed
                # persistent never saw N > max_reduction_threads.  Honouring
                # ``reduction_loops=[None]`` removes that guarantee, so the identity has
                # to be established here instead of assumed.  MEASURED
                # with the local ``padded``: at N=2048/cluster_n=2 the split gave
                # ``per_cta=512`` and total coverage ``2 * 512 * 1 = 1024`` -- HALF THE ROW
                # NEVER REDUCED.  The exact-coverage conjunct below is what turns that from
                # a wrong answer into a decline, and it is why that conjunct is an
                # equality and not an inequality.
                true_padded = next_power_of_2(max(1, size_hint))
                per_cta = true_padded // cluster_n
                split_threads = min(self._thread_count, per_cta)
                split_lanes = per_cta // split_threads if split_threads > 0 else 0
                if (
                    # The grouped two-stage intra-CTA combine the exchange composes on
                    # top of needs more than a warp; this is
                    # ``cute_cluster_feasible``'s only geometric decline, asked on the
                    # POST-split count so the two cannot disagree.
                    split_threads > _CUTE_WARP_REDUCTION_THREADS
                    # Coverage, as an exact identity rather than an inequality, and
                    # against the TRUE padded extent -- see the ⚠⚠ note above.  This is
                    # the conjunct that declines the N=2048 half-coverage geometry.
                    and cluster_n * split_threads * split_lanes == true_padded
                    # An indexed reduction's warp primitive takes <= 32 lanes, so it
                    # never reaches the grouped combine and must not be split.
                    and not _block_has_indexed_reduction(fn, block_index)
                ):
                    self._thread_count = split_threads
                    self._synthetic_cute_lane_extent = split_lanes
                    if split_lanes > 1 and self._synthetic_cute_lane_var is None:
                        self._synthetic_cute_lane_var = fn.new_var(
                            f"synthetic_lane_{block_index}",
                            dce=False,
                        )
                    elif split_lanes == 1:
                        # The share fits one thread per column, so there is no lane
                        # loop left to emit.  Dropping the var (rather than emitting a
                        # trip-count-1 loop) keeps ``codegen_reduction``'s marker/
                        # cross-warp routing reading the same "is there a lane loop"
                        # question it does without a cluster.
                        self._synthetic_cute_lane_var = None
                    # ⭐ THE PER-CTA CHUNK.  ``_cute_tv_chunk``'s contract is "the extent
                    # ONE CTA covers in one pass" (see the ⭐ note at capability ①'s plan
                    # build, which READS this field rather than re-deriving ``padded``
                    # for exactly this reason).  Narrowed HERE, after the split, so the
                    # TV plan below and every base-class capability get the per-CTA
                    # number instead of the whole row.
                    self._cute_tv_chunk = per_cta
                    self._cute_cluster_per_cta_columns = per_cta
                else:
                    # Fail CLOSED: forget the request entirely rather than carry one the
                    # emit site would honour on an unsplit geometry.  ``_cute_cluster_n_emitted``
                    # is derived from this field, so zeroing it here is what keeps the
                    # emitted combine and the launch cluster shape consistent.
                    log.debug(
                        "cute cluster DECLINED on a persistent reduction block=%s: "
                        "cluster_n=%s true_padded=%s chunk_padded=%s per_cta=%s "
                        "split_threads=%s split_lanes=%s (needs threads>%s and "
                        "cluster_n*threads*lanes == true_padded).",
                        block_index,
                        cluster_n,
                        true_padded,
                        padded,
                        per_cta,
                        split_threads,
                        split_lanes,
                        _CUTE_WARP_REDUCTION_THREADS,
                    )
                    self._cute_cluster_n_requested = 1
            # ── CAPABILITY ①: BUILD THE TV PLAN, HERE, FROM ``_cute_tv_chunk`` ─────
            #
            # ⭐ THE CHUNK IS **READ**, NOT RE-DERIVED.  It is ``self._cute_tv_chunk``
            # above and deliberately not the local ``padded``: the field's contract is
            # "the extent ONE CTA covers in one pass", which is ``padded`` without a
            # cluster and ``padded // cluster_n`` with one.  A capability that recomputed
            # ``padded`` here would keep working today (``cluster_n == 1``) and silently
            # build a plan for twice the columns a CTA owns the moment a cluster lands --
            # the trip count assuming a width the access does not use, bug class 1.  One
            # field, one reading.
            #
            # ⚠ ``requested_vec`` GATES THIS, for the SAME reason it gates the thread-count
            # clamp above and the looped path's (LEDGER E067): ``fn.config.config`` is the
            # NORMALIZED config, so "the key is present" cannot distinguish a caller's
            # request from a ladder default.  At ``vec == 1`` a plan is indistinguishable
            # from no plan, so the ladder default stays inert and every existing
            # persistent kernel is untouched.
            requested_vec_plan = env.config_spec.cute_vector_widths.config_get(
                cast(
                    "list[int]",
                    fn.config.config.get("cute_vector_widths", []) or [],
                ),
                block_index,
                1,
            )
            # ⭐⭐ NO GATE HERE, AND THAT IS THE POINT.  A loop-free reduction that takes a
            # TV plan emits its reduction inside ``for vi in cutlass.range_constexpr(vec)``
            # and therefore owes a lane-reduce MARKER, which used to need ~670 lines of AST
            # rewrite to discharge.  An earlier version of this work gated the plan OFF to
            # retire that rewrite -- which retired the CAPABILITY too, on 144 measured
            # configs.  ⛔ That was the wrong trade and it is undone.
            #
            # ⭐ The marker is now avoided at a HIGHER level: ``ConfigSpec.normalize`` rolls
            # a persistent reduction that would take a TV plan (see its "A PERSISTENT
            # REDUCTION THAT WOULD TAKE A TV PLAN IS ROLLED INSTEAD" block), so the shape
            # reaches ``LoopedReductionStrategy`` -- which gets one subgraph per dependency
            # layer from the reduction roller and needs no marker at all.  MEASURED over 216
            # loop-free configs: ZERO raises, ZERO markers, 198 carrying a TV layout.
            #
            # ⇒ this branch is still reachable (a persistent reduction whose extent cannot
            # be rolled below itself -- ``numel <= 3`` -- stays persistent), and it must be:
            # such a reduction has one dependency layer by construction, so the single nest
            # ``codegen_preamble`` builds is sufficient and no marker is owed.
            if isinstance(requested_vec_plan, int) and requested_vec_plan > 1:
                plan = self._build_cute_tv_plan(
                    fn,
                    chunk=self._cute_tv_chunk,
                    state_free=True,
                    vec_cap=requested_vec_plan,
                    block_index=block_index,
                )
                if plan is not None and plan.vec > 1:
                    # ⭐ THE COVERAGE POST-CONDITION, ASSERTED.  ``lane_extent`` is
                    # derived by the plan (``chunk // (tpr * vec)``) and the emitted lane
                    # loop's trip count is read back OFF the plan in ``codegen_preamble``,
                    # so these are two readings of one number rather than two numbers that
                    # must agree.  Asserting it here makes a future edit that breaks the
                    # identity a compile error instead of a wrong answer.
                    assert plan.covers_chunk(), (
                        f"persistent TV plan does not cover its chunk: {plan.describe()}"
                    )
                    self._cute_tv_plan = plan
                    self._cute_reduction_vec_width = plan.vec
                    self._cute_reduction_vec_mode = "unroll"
                    self._cute_reduction_lane_extent = plan.lane_extent
                    # ── CAPABILITY ③ (SMEM STAGING): THE REQUEST HALF ──────────
                    #
                    # Read HERE, next to the plan, for the same reason the looped
                    # path reads it next to its own: the request is only meaningful
                    # when the TV layout owns the access, and ``cute_stage_feasible``
                    # (the DECISION half) needs the plan to size anything.
                    #
                    # ⚠ E013 trap 6: this runs BEFORE ``super().__init__``, so
                    # ``self.fn`` / ``self.block_ids`` / ``self.block_index`` do not
                    # exist yet.  ``fn`` and ``block_index`` are passed explicitly for
                    # exactly that reason -- do not "simplify" them away.
                    #
                    # ⚠ Safe to read unconditionally, unlike ``cute_threads_per_row``:
                    # ``CuteRowResidencySpec``/``row_residency_from_legacy`` spell
                    # ``gmem`` -- no mechanism -- for a config that named nothing, so a
                    # kernel that never asked cannot acquire staging from a ladder
                    # default.  VERIFIED INERT: 40/40 frozen cells hash identical.
                    self._cute_tv_reload_from = self._cute_reload_from_config(
                        fn, plan, block_index
                    )
                    # Per-chunk staging state.  A loop-free strategy has exactly ONE
                    # chunk body, so unlike the looped path (which resets these per
                    # ``for roffset`` body in ``codegen_device_loop``) these are
                    # initialised once and never reset.
                    self._cute_tv_stage_partitions = {}
                    self._cute_tv_staged_tensors = set()
                    # ⭐ THE ``registers`` CACHE MAP, and OMITTING IT SILENTLY DISABLED THE
                    # WHOLE ARM ON THIS STRATEGY.
                    #
                    # ``_cute_tv_rmem_slice`` declines when this is ``None`` -- the class-level
                    # sentinel on ``TileStrategy``, which exists so a strategy that never
                    # participates reads as "no cache" instead of crashing.  I minted the real
                    # dict on ``LoopedReductionStrategy`` and ``CuteNDTileStrategy`` and MISSED
                    # this one, so a loop-free kernel asking for ``registers`` had every
                    # precondition satisfied (``enabled=True``, ``num_chunks=1``,
                    # ``lane_extent=4``, ``vec=8``) and still declined on the sentinel -- then
                    # raised ``CuteRowResidencyUnavailable`` because the request could not be
                    # honoured.  ⇒ it read like a capability boundary and was a missing line.
                    #
                    # ⚠ SAME CLASS OF DEFECT AS THE NDTILE ONE (a sentinel keeps a missing
                    # attribute from crashing; only a real per-instance dict makes the
                    # capability WORK), and it is the second time in this run -- so the rule
                    # is: every strategy that answers ``cute_tv_capable()`` needs the real
                    # object, not just the inert default.
                    self._cute_tv_rmem_frag_by_tile = {}

    def _reduction_thread_count(self) -> int:
        return self._thread_count

    def offset_var(self, block_idx: int) -> str:
        assert block_idx == self.block_index
        return "0"

    def _cute_cluster_column_offset(self, state: CodegenState, *, scale: int) -> str:
        """``+ cluster_y * (per_cta // scale)`` -- this CTA's column base, or ``""``.

        ⭐ THE ENTIRE PER-CTA EXTENT SPLIT, AS ONE TERM.  Persistent has no loop bound
        for the cluster to edit, so what makes the ``cluster_n`` peers cover DISJOINT
        columns is this offset on their index.  ``__init__`` has already narrowed
        ``_thread_count`` / ``_synthetic_cute_lane_extent`` so each CTA's index spans
        exactly ``[0, per_cta)``; adding ``cluster_y * per_cta`` tiles the ``cluster_n``
        shares over ``[0, padded)`` once each, which is the invariant the DSMEM combine's
        correctness rests on (without it every peer folds the SAME partial and the answer
        is multiplied by ``cluster_n``).

        ⚠ ``scale`` EXISTS BECAUSE THE THREE CALLERS MULTIPLY BY DIFFERENT FACTORS, and
        getting it wrong is a silent wrong answer rather than a crash.  The TV branch
        builds ``base = (thread_expr) * vec + ...``, so a term folded into
        ``thread_expr`` is scaled by ``vec`` and must be pre-divided by it; the scalar
        branches add the term to the index directly and pass ``scale=1``.  It is a
        PARAMETER rather than inferred from ``self._cute_tv_plan`` because the caller
        knows its own arithmetic and the plan does not know where it is being spliced.
        The division is exact: ``per_cta == thread_count * lane_extent * vec`` by the
        coverage identity asserted in ``__init__``, so ``vec | per_cta``.

        Returns ``""`` -- not ``None`` -- so callers can concatenate unconditionally and
        the no-cluster path stays BYTE-IDENTICAL to what it emits today.
        """
        per_cta = self._cute_cluster_per_cta_columns
        if per_cta <= 0 or self._cute_cluster_emitted_n() <= 1:
            return ""
        assert per_cta % scale == 0, (
            f"cluster column offset does not divide the split: per_cta={per_cta} "
            f"scale={scale}"
        )
        return f" + {self._cute_cluster_y_var(state)} * {per_cta // scale}"

    def codegen_preamble(self, state: CodegenState) -> None:
        env = CompileEnvironment.current()
        backend = env.backend
        block_idx = self.block_index
        numel = env.block_sizes[block_idx].numel
        index_var = self.index_var(block_idx)
        mask_var = self._mask_var
        block_size_var = self.block_size_var(self.block_index)
        assert block_size_var is not None
        if state.device_function.constexpr_arg(block_size_var):
            if isinstance(numel, sympy.Integer):
                # Static size - issue statement immediately
                stmt = statement_from_string(
                    f"{block_size_var} = {backend.static_rdim_size(int(numel))}"
                )
                state.codegen.host_statements.append(stmt)
            else:
                # Check for block size dependencies
                block_mapping, _ = find_block_size_symbols(numel)
                if block_mapping:
                    # Defer issuing statement until block sizes are known
                    state.device_function.deferred_rdim_defs.append(
                        (block_size_var, numel)
                    )
                else:
                    # No dependencies - issue statement immediately
                    expr_str = HostFunction.current().sympy_expr(numel)
                    stmt = statement_from_string(
                        f"{block_size_var} = {backend.dynamic_rdim_size_expr(expr_str)}"
                    )
                    state.codegen.host_statements.append(stmt)
        current_grid = state.codegen.current_grid_state
        synthetic_lane_var = self._synthetic_cute_lane_var
        if (
            synthetic_lane_var is not None
            and current_grid is not None
            and self._cute_tv_plan is not None
            and isinstance(current_grid, DeviceGridState)
        ):
            # ── CAPABILITY ①: THE TV LANE NEST, BUILT HERE ─────────────────────────
            #
            # ⭐ WHY THE NEST IS BUILT NOW AND NOT BY ``wrap_body``.  The TV emission
            # protocol mutates two list objects WHILE loads lower -- ``_cute_lane_body``
            # (the ``cute.copy`` goes next to the constexpr V-loop it must precede) and
            # ``_cute_tv_chunk_prefix`` (the per-chunk ``local_tile`` /
            # ``partition_S`` / fragment declarations go just above the lane loop).
            # ``wrap_body`` runs at the END of codegen, so a nest built there would not
            # exist yet at the moment the hoist needs to insert into it.  So the nest is
            # built here, handed to the grid state as a sentinel, and ``wrap_body``
            # splices the user body into its sink.
            #
            # ⚠ THE EMITTED GEOMETRY IS THE **PLAN'S**, NOT THE SCALAR LANE LOOP'S, and
            # conflating the two is bug class 1.  ``_synthetic_cute_lane_extent`` is the
            # SCALAR loop's trip count (``padded // thread_count`` == 32 at N=1024,
            # one element per iteration); ``plan.lane_extent`` is the VECTOR loop's
            # (``chunk // (tpr * vec)`` == 4, ``vec`` elements per iteration).  Emitting
            # the scalar extent around a ``vec``-wide copy would read 32 * 8 elements
            # where the row has 1024 / 32 per thread -- i.e. a trip count assuming a
            # width the access does not use.  Coverage is asserted below from the plan's
            # own identity rather than trusted.
            plan = self._cute_tv_plan
            lane_extent = plan.lane_extent
            assert lane_extent * self._thread_count * plan.vec == self._cute_tv_chunk, (
                "persistent TV nest does not cover its chunk: "
                f"lane_extent={lane_extent} threads={self._thread_count} "
                f"vec={plan.vec} chunk={self._cute_tv_chunk}"
            )
            axis = self._get_thread_axis()
            current_grid.thread_axis_sizes[axis] = max(
                current_grid.thread_axis_sizes.get(axis, 1),
                self._thread_count,
            )
            current_grid.block_thread_axes[block_idx] = axis
            current_grid.lane_loop_blocks.add(block_idx)
            # ⭐ THE EXISTING SYNTHETIC LANE VAR IS REUSED, not replaced by a new name.
            # This IS persistent's synthetic lane loop -- the only thing the TV plan
            # changes is its EXTENT (the plan's ``lane_extent``, with ``vec`` elements
            # per iteration instead of one).  Keeping the var also keeps
            # ``_synthetic_cute_lane_var`` non-None, which is what routes
            # ``codegen_reduction`` to the marker path that the constexpr-V split then
            # rewrites -- minting a second name would leave two notions of "this
            # reduction's lane" for the marker and the copies to disagree about.
            lane_var = synthetic_lane_var
            vec_lane_var = self.fn.new_var(f"reduction_vec_lane_{block_idx}", dce=False)
            base_var = self.fn.new_var(f"reduction_lane_base_{block_idx}", dce=False)
            self._cute_reduction_lane_var = lane_var
            self._cute_lane_base_index_var = base_var
            # ``base = tid * vec + lane * (threads * vec)``, and the per-element index
            # is ``base + vi`` inside the constexpr loop -- so the existing scalar
            # pipeline (mask, cast, combine) keeps working per element, unchanged.
            # This is the persistent analogue of the looped path's ``base_expr``,
            # without the ``roffset`` term it has no loop to supply.
            thread_expr = self._index_init_expr(
                block_size_var, env.index_type(), block_idx
            )
            from .tile_strategy import _clone_stmt
            from .tile_strategy import _constexpr_vec_loop
            from .tile_strategy import _create_lane_loop

            vec_for = cast(
                "ast.For",
                ast.parse(
                    f"for {vec_lane_var} in cutlass.range_constexpr({plan.vec}):\n"
                    f"    pass"
                ).body[0],
            )
            # The SINK: the user body is spliced in here by ``wrap_body``. It must be
            # the very list the emitted loop holds, so the reference is kept, never
            # rebuilt.
            sink: list[ast.AST] = []
            vec_for.body = sink  # type: ignore[assignment]
            # ⭐ THE PLAN OWNS THE FORMULA (``ChunkTVPlan.emit_lane_base``).  This site
            # was already correct, but it is routed through the plan anyway: the same
            # expression hand-written at several sites is how the transposed-stride
            # wrong answer happened, so being correct today is not a reason to keep a
            # private copy.
            #
            # ``offset_expr=None``: this strategy is persistent -- the chunk IS the row,
            # so there is no outer chunk offset to add (its own ``offset_var`` returns
            # ``"0"``), and passing a literal ``"0 + "`` would only add a dead term.
            # ``scale=1``: the cluster term is in ELEMENT units already, like the two
            # summands it follows, so it is appended to the base rather than folded
            # into ``thread_expr``.
            lane_base_expr = plan.emit_lane_base(
                None,
                lane_var,
                f"({thread_expr})",
                extra_terms=self._cute_cluster_column_offset(state, scale=1),
            )
            lane_body: list[ast.AST] = [
                statement_from_string(f"{base_var} = {lane_base_expr}"),
                vec_for,
            ]
            self._cute_tv_constexpr_loop = vec_for
            self._cute_lane_body = lane_body
            chunk_body: list[ast.AST] = [
                _create_lane_loop(lane_var, lane_extent, lane_body)
            ]

            # ⭐⭐ A1: REGISTER THE AXIS, AND MAKE THE NEST REBUILDABLE.
            #
            # Two halves of one change, and both are needed:
            #
            # 1. ``add_lane_loop`` -- this branch previously registered only
            #    ``lane_loop_blocks`` (line ~3180) and never the ``(var, extent)`` pair,
            #    because the prebuilt nest emits the loop itself so ``wrap_body`` must not
            #    build a second one.  But ``_emit_inline_lane_reduce`` declines outright on
            #    ``not grid.lane_loops`` (MEASURED: that was the exit taken, not the
            #    ``prebuilt_lane_nest`` one), and ``_wrap_segmented_body`` reads the axis's
            #    EXTENT out of the same dict.  So the axis must be visible.  Double-building
            #    is prevented by ``wrap_body``: with a nest registered it never reaches the
            #    ``for lane_var, extent in reversed(self.lane_loops)`` path.
            # 2. The FACTORY -- ``chunk_body`` above can be spliced only once.  When one
            #    reduction's input needs another's FINISHED scalar (``amax`` then
            #    ``sum(exp(v - amax))``) the lane axis must be reopened per dependency
            #    layer, which one splice cannot express.  ``_wrap_segmented_body`` calls
            #    this once per segment, at the END of codegen -- i.e. after the TV protocol
            #    inserted the ``cute.copy`` and the ``partition_*`` declarations -- so the
            #    clone reproduces a COMPLETE nest.
            #
            # ⚠ The clone must be deep: two sibling loops holding the same node objects
            # alias, and a later in-place rewrite of one would edit both.
            current_grid.add_lane_loop(block_idx, lane_var, lane_extent)

            def _rebuild_nest(
                segment_body: list[ast.AST],
                _lane_body: list[ast.AST] = lane_body,
                _lane_var: str = lane_var,
                _lane_extent: int = lane_extent,
                _vec_lane_var: str = vec_lane_var,
                _vec: int = plan.vec,
                _chunk_body: list[ast.AST] = chunk_body,
                _outer_lane_loop: ast.AST = chunk_body[0],
            ) -> tuple[list[ast.AST], list[ast.AST]]:
                # ⚠ THE VEC LOOP IS REBUILT FROM ITS RECIPE, NOT CLONED.  ``_clone_stmt``
                # round-trips through ``ast.unparse``, and an ``ast.For`` whose body list is
                # the live sink unparses to ``for v in ...:`` with NOTHING under it -- which
                # re-parses as an ``IndentationError``.  MEASURED as exactly that.  Cloning
                # is still right for every OTHER statement (the hoisted copies and the base
                # assignment are complete), so only the loop itself is re-minted.
                fresh_sink: list[ast.AST] = [*segment_body]
                fresh_vec_for = cast(
                    "ast.For",
                    ast.parse(
                        f"for {_vec_lane_var} in cutlass.range_constexpr({_vec}):\n"
                        f"    pass"
                    ).body[0],
                )
                fresh_vec_for.body = fresh_sink  # type: ignore[assignment]
                fresh: list[ast.AST] = []
                seen_vec_loop = False
                for stmt in _lane_body:
                    if (
                        isinstance(stmt, ast.For)
                        and _constexpr_vec_loop(stmt) is not None
                    ):
                        fresh.append(fresh_vec_for)
                        seen_vec_loop = True
                    else:
                        fresh.append(_clone_stmt(stmt))
                assert seen_vec_loop, (
                    "prebuilt nest rebuild found no constexpr vec loop in the lane body"
                )
                # ⛔⛔ THE NEST IS TWO LEVELS, AND REBUILDING ONLY THE INNER ONE LOSES THE
                # TENSORS.  ``chunk_body`` is ``[<per-chunk declarations>, <lane loop>]``:
                # ``_cute_tv_partition_hoist`` inserts the ``local_tile`` /
                # ``partition_S`` / ``make_rmem_tensor_like`` declarations at
                # ``len(chunk_body) - 1`` (see its docstring -- appending would put them
                # after the loop that reads them), and THOSE are the only statements that
                # name ``x`` / ``w`` / ``out``.
                #
                # MEASURED when this returned just the lane loop: the declarations were
                # never emitted, so after DCE the surviving reads were ``_tv_part_*`` with
                # no tensor names at all, every ``TensorArg`` was dropped as unused, and
                # codegen failed with ``BackendUnsupported: kernel launch without tensor
                # args``.  The body was structurally correct and the kernel had no inputs.
                fresh_chunk: list[ast.AST] = []
                for stmt in _chunk_body:
                    if stmt is _outer_lane_loop:
                        fresh_chunk.append(
                            _create_lane_loop(_lane_var, _lane_extent, fresh)
                        )
                    else:
                        fresh_chunk.append(_clone_stmt(stmt))
                return fresh_chunk, fresh_sink

            current_grid.prebuilt_lane_nest_factory = _rebuild_nest
            # The chunk coordinate is the literal 0: persistent covers the whole
            # extent in ONE pass, so there is exactly one chunk along N and
            # ``local_tile``'s column coordinate is its index.  (The looped path emits
            # ``roffset // block``, which is this same quantity where a loop exists.)
            self._cute_tv_chunk_index_var = "0"
            # ⭐ CAPABILITY ③: THE STAGED TILE'S CHUNK COORDINATE, AND IT IS THE SAME
            # LITERAL ``0`` -- which is the whole of the "loopless body is the
            # ``num_chunks == 1`` case" argument, at the emission site.
            #
            # ⚠ It is the CTA-LOCAL chunk number, and here that is ``0`` even WITH a
            # cluster, where the looped path has to subtract ``cluster_y *
            # chunks_per_cta`` from a global chunk index.  The reason is that
            # persistent's cluster split lives in the per-CTA COLUMN OFFSET on the
            # index (``_cute_cluster_column_offset``) rather than in a loop bound, so
            # this CTA's ``_cute_tv_chunk`` already IS its share: chunk 0 of a
            # ``per_cta``-wide tile.  ``_cute_stage_num_chunks`` divides the extent by
            # the same ``cluster_n``, so buffer and coordinate cannot drift.
            self._cute_tv_stage_chunk_index_var = "0"
            self._cute_tv_chunk_prefix = chunk_body
            self._cute_tv_partitions = {}
            current_grid.prebuilt_lane_nest = (chunk_body, sink)
            # The per-element index, in terms of the plan's own vars.
            current_grid.lane_setup_statements.append(
                statement_from_string(
                    f"{index_var} = {base_var} + cutlass.Int32({vec_lane_var})"
                )
            )
            if mask_var is not None:
                current_grid.lane_setup_statements.append(
                    statement_from_string(
                        f"{mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
                    )
                )
        elif synthetic_lane_var is not None and current_grid is not None:
            axis = self._get_thread_axis()
            current_grid.add_lane_loop(
                block_idx,
                synthetic_lane_var,
                self._synthetic_cute_lane_extent,
            )
            current_grid.thread_axis_sizes[axis] = max(
                current_grid.thread_axis_sizes.get(axis, 1),
                self._thread_count,
            )
            current_grid.block_thread_axes[block_idx] = axis
            index_expr = (
                f"({self._index_init_expr(block_size_var, env.index_type(), block_idx)})"
                f" + cutlass.Int32({synthetic_lane_var}) * {self._thread_count}"
                f"{self._cute_cluster_column_offset(state, scale=1)}"
            )
            current_grid.lane_setup_statements.append(
                statement_from_string(f"{index_var} = {index_expr}")
            )
            if mask_var is not None:
                current_grid.lane_setup_statements.append(
                    statement_from_string(
                        f"{mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
                    )
                )
        else:
            # No lane loop: one thread per column of THIS CTA's share.  The cluster
            # offset is still required -- it is what distinguishes the peers.
            state.add_statement(
                f"{index_var} = "
                f"{self._index_init_expr(block_size_var, env.index_type(), block_idx)}"
                f"{self._cute_cluster_column_offset(state, scale=1)}"
            )
            if mask_var is not None:
                state.add_statement(
                    f"{mask_var} = {index_var} < {self.fn.sympy_expr(numel)}"
                )
        # Extract end_var_name from the numel expression
        from .tile_strategy import LoopDimInfo

        end_var_name = self.fn.sympy_expr(numel)
        block_id_to_info = {
            self.block_index: LoopDimInfo(end_var_name=end_var_name, end_expr=numel)
        }
        tracker = ThreadAxisTracker()
        if self._thread_count > 0:
            tracker.record(
                self.block_index, self._get_thread_axis(), self._thread_count
            )
        state.codegen.push_active_loops(
            PersistentReductionState(
                self,
                block_id_to_info=block_id_to_info,
                thread_axis_sizes=tracker.sizes,
                block_thread_axes=tracker.block_axes,
            )
        )

    def _cute_cross_warp_reduction_expr(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        default_value: float | bool,
        dtype: torch.dtype,
    ) -> str | None:
        env = CompileEnvironment.current()
        backend = env.backend
        if (
            backend.name != "cute"
            or self._thread_count <= 32
            or self._synthetic_cute_lane_var is not None
            or backend.is_indexed_reduction(reduction_type)
        ):
            return None

        current_grid = state.codegen.current_grid_state
        axis_sizes: dict[int, int] = {}
        if isinstance(current_grid, DeviceGridState):
            for axis, size in current_grid.thread_axis_sizes.items():
                axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        reduction_axis = self._get_thread_axis()
        axis_sizes[reduction_axis] = max(
            axis_sizes.get(reduction_axis, 1), self._thread_count
        )

        num_threads = 1
        for size in axis_sizes.values():
            num_threads *= size
        group_span = self._thread_count
        if num_threads % group_span != 0:
            return None

        identity_expr = backend.cast_expr(
            constant_repr(default_value), _dtype_str(dtype)
        )
        # ``dtype`` is the accumulation dtype (``reduction_acc_dtype``), passed to
        # the two-stage shared reduce EXPLICITLY as ``acc_dtype`` -- it is never
        # inferred from the identity.  Upcast the (possibly narrower) masked input
        # to the same dtype so the helper's ``input if mask else identity``
        # selection unifies cleanly.
        input_expr = backend.cast_expr(input_name, _dtype_str(dtype))

        if reduction_axis == 0:
            # ``axis_sizes`` only reflects the thread axes discovered so far. A
            # sibling control-flow branch can still introduce a *redundant*
            # thread axis later in codegen -- e.g. a free ``hl.arange`` that
            # another (mutually-exclusive) branch maps onto thread axis 1/2 --
            # which enlarges the launch block beyond ``num_threads``. Those extra
            # threads re-run this reduction; if every redundant row keyed its
            # shared memory on the same slots the cross-warp combine would race
            # (producing intermittently wrong partial reductions). Key the
            # per-group shared memory on the FULL flattened thread id (from the
            # runtime block dims) so each redundant row reduces into its own
            # region. When the reduction owns thread axis 0
            # (``blockDim.x == group_span``) this is identical to the
            # single-axis path whenever there is no redundancy; the extra
            # groups simply go unused.
            #
            # ⚠ ``MAX_THREADS_PER_BLOCK`` WAS IMPORTED HERE AND IS NO LONGER NEEDED (task 5):
            # the group count is now derived from the launch geometry rather than from the
            # hardware maximum.  See the group-count block below.
            index_type = backend.index_type_str(env.index_dtype)
            tid0 = backend.cast_expr("cute.arch.thread_idx()[0]", index_type)
            tid1 = backend.cast_expr("cute.arch.thread_idx()[1]", index_type)
            tid2 = backend.cast_expr("cute.arch.thread_idx()[2]", index_type)
            bdim0 = backend.cast_expr("cute.arch.block_dim()[0]", index_type)
            bdim1 = backend.cast_expr("cute.arch.block_dim()[1]", index_type)
            lane_expr = (
                f"{tid0} + ({tid1}) * ({bdim0}) + ({tid2}) * ({bdim0}) * ({bdim1})"
            )
            # ⭐⭐ ONE GROUP COUNT, DERIVED FROM THE LAUNCH GEOMETRY (FIXLIST item 5).
            #
            # There used to be TWO numbers here and they were genuinely different
            # questions -- a SMEM-keying CAPACITY that deliberately over-counted
            # (``ceil(MAX_THREADS_PER_BLOCK / group_span)``) and a cluster ``expect_tx``
            # PROMISE that used the branch-LOCAL ``num_threads``.  Conflating them either
            # way is a bug in one direction or the other:
            #
            #   * over-promising HANGS.  MEASURED at N=1024 cluster_n=2: the launch is
            #     ``block=(512,1,1)`` with ``group_span=512``, so the live count is 1 while
            #     the keying count is ``1024//512 == 2``.  ``expect_tx`` named
            #     ``2*2*4 = 16`` bytes against ``1*2*4 = 8`` actually stored, and the probe
            #     spun at 100% GPU with zero output until killed -- the signature
            #     ``_cute_cluster_exchange``'s own ``elect_one`` note records.
            #   * under-counting the CAPACITY races.  ``axis_sizes`` is branch-LOCAL: a
            #     sibling control-flow branch can widen ``blockDim`` (e.g. a free
            #     ``hl.arange`` another mutually-exclusive branch maps onto thread axis
            #     1/2), and the redundant rows that then re-run this reduction must key
            #     their partials on distinct slots.
            #
            # ⇒ THE FIX IS ONE NUMBER THAT IS BOTH: the group count of the LAUNCH that
            # actually happens.  ``self._codegen.max_thread_block_dims`` is the running
            # per-axis maximum of that launch, and crucially it **already folds in the
            # synthetic ``hl.arange`` axes** that ``tile_strategy.thread_block_dims()``
            # cannot see (``generate_ast._current_active_thread_axis_sizes`` merges
            # ``cute_synthetic_arange_axis_sizes`` explicitly).  So the elementwise
            # ``max(recorded, planned)`` covers BOTH gaps:
            #     ``recorded`` alone misses a strategy that has not entered its loop yet;
            #     ``planned``  alone misses the synthetic arange axes.
            # Neither is an upper bound; their max is the tightest one available here.
            #
            # ⭐ THIS IS NOT A NEW IDIOM IN THIS FILE.  ``BlockReductionStrategy`` already
            # does exactly this arithmetic for exactly this hazard class -- ``max(recorded,
            # planned)`` per axis, product, then a soundness decline whose comment reads
            # "the strided thread-reduction path assumes every participating lane is backed
            # by a live thread, so using it here would read unwritten SMEM partials".  The
            # merge applies that established shape to this site.
            #
            # ⚠ IT CANNOT RESURRECT THE HANG, because it is bounded on both sides:
            # ``max_thread_block_dims`` only ever grows, and the launcher hard-caps the
            # product (``check_thread_limit`` raises above ``MAX_THREADS_PER_BLOCK`` ==
            # 1024).  So
            #     live_count  <=  merged_count  <=  ceil(1024 / group_span)
            # i.e. the merged value is sandwiched between the old promise and the old
            # capacity -- which is precisely what lets one number serve both consumers.
            #
            # ⛔⛔ WHAT THIS DOES **NOT** FIX, and it is a real open bug -- see
            # ``_cute_cluster_exchange`` in ``cute/reduce_helpers.py``.  The store there is
            # predicated on ``lane_in_group < cluster_n`` with **no ``group_id <
            # group_count`` conjunct**, while ``group_id`` is a RUNTIME value
            # (``lane_var // group_span`` where ``lane_var`` is built from
            # ``cute.arch.block_dim()``) and ``group_count`` sizes the exchange buffer.  So
            # a group whose runtime id exceeds ANY compile-time count computes a
            # ``crd2idx`` past the allocation and stores into a peer CTA's SMEM.  No
            # compile-time count -- this one, either old one, or any future one -- can bound
            # a runtime index that nothing predicates against.  The sound fix is either a
            # ``group_id`` bound on the store or a runtime ``expect_tx`` (the DSL types it
            # ``bytes: Int``, and helion already issues a runtime-predicated arrival in
            # ``program_id.py``).  Structurally reachable, unwitnessed in-tree
            # (``cute_cluster_n`` appears only as ``[1]`` in the suite).  Recorded, not
            # fixed here.
            launch_threads = 1
            # ⚠ ``state.codegen``, NOT ``self._codegen``.  ``state.codegen`` IS the
            # ``GenerateAST`` and is already dereferenced ~20 lines above
            # (``state.codegen.current_grid_state``), so it is guaranteed present here;
            # ``self._codegen`` is set on some strategies and not others, and reaching for it
            # would reintroduce exactly the "ask the class, not the capability" coupling this
            # branch exists to remove.
            recorded_dims = state.codegen.max_thread_block_dims
            planned_dims = self.fn.tile_strategy.thread_block_dims()
            for recorded, planned in zip(recorded_dims, planned_dims, strict=True):
                launch_threads *= max(int(recorded), int(planned), 1)
            group_count = max(1, launch_threads // group_span)
            cluster_group_count = group_count
        else:
            # The two-stage shared-memory reduction assumes its ``lane_var`` is
            # the linear thread index across ALL of the launch block's threads.
            # If ``axis_sizes`` only covers a subset of the planned block dims
            # (e.g. an inner reduction strategy contributes another thread axis
            # that hasn't been entered yet), the emitted reduction would race
            # across the missing axis. Bail out and fall back to the warp-level
            # path in that case.
            planned_dims = self._planned_thread_dims()
            planned_block_threads = planned_dims[0] * planned_dims[1] * planned_dims[2]
            if num_threads != planned_block_threads:
                return None
            lane_expr = backend.thread_linear_index_expr(axis_sizes)
            if lane_expr is None:
                return None
            group_count = num_threads // group_span
            # Already the live count on this branch (it is derived from
            # ``num_threads``, which this branch has just checked equals the planned
            # block threads), so the two coincide.  Named anyway so the cluster leg
            # reads ONE variable rather than choosing per branch.
            cluster_group_count = max(1, group_count)

        lane_var = self.fn.new_var("persistent_reduce_lane", dce=True)
        lane_in_group_var = self.fn.new_var("persistent_reduce_lane_in_group", dce=True)
        lane_mod_pre_var = self.fn.new_var("persistent_reduce_lane_mod_pre", dce=True)
        result_var = self.fn.new_var("persistent_reduce_result", dce=True)
        state.add_statement(f"{lane_var} = {lane_expr}")
        state.add_statement(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
        state.add_statement(f"{lane_mod_pre_var} = ({lane_in_group_var}) % 1")
        state.add_statement(
            f"{result_var} = _cute_grouped_reduce_shared_two_stage("
            f"{input_expr}, {reduction_type!r}, {identity_expr}, "
            f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
            f"acc_dtype={_dtype_str(dtype)}, "
            f"pre=1, group_span={group_span}, group_count={group_count})"
        )
        # ── CAPABILITY ②: THE CLUSTER LEG, across CTAs (PORT_SPEC §9) ───────────────
        #
        # ⭐ THIS IS THE CALL SITE THAT DID NOT EXIST.  ``_cute_cluster_emitted_n`` is the
        # only setter of ``fn.cute_state.cluster_shape`` (which the host emits as
        # ``_helion_cute_cluster_shape``), and it was called ONLY from
        # ``LoopedReductionStrategy``; persistent inherited the method and never asked, so
        # the knob was inert here.  Asking it here -- at the point the intra-CTA result
        # exists -- is what makes the launch geometry, the exchange and the barrier appear
        # together, which is the property the acceptance test's three legs check.
        #
        # Structurally identical to the looped leg (``LoopedReductionStrategy``'s own
        # ``_cute_cross_warp_reduction_expr``), and deliberately so: the exchange composes
        # ON TOP of the intra-CTA ``_cute_grouped_reduce_shared_two_stage`` result rather
        # than being fused into it, so the only difference between the two paths is WHERE
        # the statements are appended -- persistent has no ``device_loop.outer_suffix``, so
        # they go straight into the body via ``state.add_statement``.
        #
        # The ORDERING is quack's (``rmsnorm.py:316`` passes ``cluster_wait`` as
        # ``row_reduce``'s ``hook_fn``): local combine, then ``cluster_wait``, then the
        # exchange.  Waiting EARLIER is correct but slower; waiting LATER races the peers'
        # barrier initialisation.
        cluster_n = self._cute_cluster_emitted_n()
        if cluster_n > 1:
            mbar_var = self._cute_cluster_mbar_var(state)
            cluster_var = self.fn.new_var("persistent_reduce_cluster", dce=True)
            group_id_var = self.fn.new_var("persistent_reduce_group", dce=True)
            state.add_statement(f"{group_id_var} = ({lane_var}) // {group_span}")
            state.add_statement("cute.arch.cluster_wait()")
            state.add_statement(
                f"{cluster_var} = _cute_cluster_reduce("
                f"{result_var}, {reduction_type!r}, "
                f"{group_id_var}, {lane_in_group_var}, {mbar_var}, "
                f"acc_dtype={_dtype_str(dtype)}, "
                # ⚠ ``cluster_group_count``, NOT ``group_count`` -- see the ⚠⚠ note where
                # it is derived.  The exchange's ``expect_tx`` byte count must match the
                # stores that actually happen or the barrier never flips.
                f"group_count={cluster_group_count}, cluster_n={cluster_n})"
            )
            return cluster_var
        return result_var

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        env = CompileEnvironment.current()
        backend = env.backend
        # Record (for the CuTe backend) the branch path under which this reduction
        # claims its thread axis, so a free ``hl.arange`` in a mutually-exclusive
        # sibling grid branch reuses this axis instead of claiming a fresh one that
        # would widen the launch block and race this single-axis reduction. No-op
        # outside a dynamic ``_if`` branch.
        if backend.name == "cute":
            state.codegen.record_cute_strategy_axis_branch_path(self._get_thread_axis())
        numel = env.block_sizes[self.block_index].numel
        if isinstance(numel, sympy.Integer) and numel == 0:
            default = ir.Reduction.default_accumulator(reduction_type, fake_input.dtype)
            assert isinstance(default, (float, int, bool))
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_output.size()])
            return expr_from_string(
                backend.full_expr(shape_dims, constant_repr(default), fake_output.dtype)
            )
        acc_dtype = reduction_acc_dtype(reduction_type, fake_input.dtype)
        default = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
        if (
            self._synthetic_cute_lane_var is not None
            and isinstance(default, (float, int, bool))
            and not self._lane_reduce_marker_unsupported(state)
            and (threads := self._lane_reduce_threads_in_group()) is not None
        ):
            # The reduction axis is split across a synthetic per-thread lane
            # loop: the single warp reduction only covers one lane's worth of
            # elements. Emit a marker so the ``split_lane_loop_reductions``
            # post-pass produces the two-pass (accumulate across lanes ->
            # warp-combine across ``threads`` -> consume) structure.
            from .tile_strategy import _lane_reduce_marker_expr

            identity_expr = backend.cast_expr(
                constant_repr(default), _dtype_str(acc_dtype)
            )
            group_params = self._reshape_merged_reduction_group_params()
            group_pre, group_span, group_lane_expr = group_params or (1, 0, "")
            if backend.is_indexed_reduction(reduction_type):
                # An indexed reduction has no single-accumulator marker: lower it
                # into a value marker + a dependent index marker (see
                # ``_indexed_lane_reduce_expr``) and let the layered split emit
                # one accumulate/finalize pass per dependency layer.
                expr = self._indexed_lane_reduce_expr(
                    state,
                    input_name,
                    reduction_type,
                    dim,
                    fake_input,
                    fake_output,
                    threads,
                    group_pre=group_pre,
                    group_span=group_span,
                    group_lane_expr=group_lane_expr,
                )
            else:
                # ⭐⭐ G1: TRY TO EMIT BOTH COMBINES HERE, INLINE, and fall back to the
                # marker only where the mechanism cannot express it (see
                # ``_emit_inline_lane_reduce`` for the four decline conditions, each of
                # which is a requirement rather than a preference).  When it succeeds the
                # reduction is COMPLETE at this point: no marker is created, so nothing
                # downstream can discharge its obligation by deleting the fold.
                inline = self._emit_inline_lane_reduce(
                    state,
                    input_name,
                    reduction_type,
                    identity_expr,
                    threads,
                    acc_dtype_str=_dtype_str(acc_dtype),
                    group_pre=group_pre,
                    group_span=group_span,
                    group_lane_expr=group_lane_expr,
                    result_hint=(
                        state.fx_node.name
                        if state.fx_node is not None
                        else reduction_type
                    ),
                )
                expr = inline
                if expr is None:
                    expr = _lane_reduce_marker_expr(
                        input_name,
                        reduction_type,
                        identity_expr,
                        threads,
                        acc_dtype_str=_dtype_str(acc_dtype),
                        group_pre=group_pre,
                        group_span=group_span,
                        group_lane_expr=group_lane_expr,
                    )
            return expr_from_string(
                self.maybe_reshape(expr, dim, fake_input, fake_output)
            )
        if isinstance(default, (float, int, bool)):
            cross_warp = self._cute_cross_warp_reduction_expr(
                state, input_name, reduction_type, default, acc_dtype
            )
        else:
            cross_warp = None
        if cross_warp is not None:
            expr = cross_warp
        else:
            expr = self.call_reduction_function(
                input_name,
                reduction_type,
                dim,
                fake_input,
                fake_output,
            )
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))


class LoopedReductionStrategy(ReductionStrategy):
    def __init__(
        self,
        fn: DeviceFunction,
        block_index: int,
        block_size: int,
    ) -> None:
        env = CompileEnvironment.current()
        if block_size <= 1:
            raise exc.InvalidConfig(
                f"LoopedReductionStrategy requires block_size > 1, got {block_size}"
            )
        # Compute thread count for warp-level reductions
        max_threads = env.backend.max_reduction_threads()
        if max_threads is not None:
            # CuTe argreduce uses cute.arch.warp_reduction which is only
            # correct for threads_in_group<=32. Cap to warp size whenever
            # the rolled reduction will fold an indexed reduction over this
            # block.
            if env.backend.name == "cute" and _block_has_indexed_reduction(
                fn, block_index
            ):
                max_threads = min(max_threads, _CUTE_WARP_REDUCTION_THREADS)
            thread_count = next_power_of_2(min(block_size, max_threads))
        else:
            thread_count = 0
        tile_dispatch = getattr(fn, "tile_strategy", None)
        if tile_dispatch is not None:
            thread_count = env.backend.adjust_reduction_thread_count(
                thread_count, tile_dispatch.strategies
            )
        # ``cute_threads_per_row``: LOWER the row's thread count so the chunk can be
        # covered by a WIDER copy.  Until now this knob was registered in five places
        # (``config_spec.py``, ``device_ir.py:1080``) and read by NOTHING, and
        # ``02_PERF.md`` §4 item 2 concluded that wiring it "will NOT help" because it is
        # capped by the same 1024 that forces the chunk loop.  MEASURED, that is exactly
        # backwards -- the cap is *why* the copy is scalar:
        #
        #   ``chunk_plan`` (``cute/tv_layout.py:880``) narrows ``vec`` while
        #   ``chunk % (threads_per_row * vec)``, and ``ChunkTVPlan.__post_init__``
        #   requires that identity.  So the widest legal copy is
        #   ``vec <= chunk // threads_per_row``.  With ``threads_per_row`` pinned to
        #   ``next_power_of_2(min(reduction_loops, 1024))``, ``rl=1024`` gives
        #   ``vec <= 1`` -- a 16-bit SCALAR load, whatever ``cute_vector_widths`` asks
        #   for.  And the ``block_size > thread_count`` guard below is then FALSE, so the
        #   TV block does not even run.  Both of run 1's 400-config sweeps lived at
        #   ``rl <= 1024`` and therefore never emitted a vector load at all, which is why
        #   the time was identical at vw=2/4/8 (LEDGER E012).
        #
        # Lowering ``threads_per_row`` is the direct lever on the access width: at
        # ``tpr=256`` a ``vec=4`` copy is legal at ``rl=1024``, and ``vec=8`` at
        # ``rl=2048``.  The lane loop makes up the difference -- the row is still fully
        # covered, because ``lane_extent = chunk // (tpr * vec)`` is re-derived from the
        # same plan (that is the coverage identity ``ChunkTVPlan`` asserts), so this
        # cannot under-read.
        #
        # Only ever a REDUCTION, never an increase: raising the count past what
        # ``adjust_reduction_thread_count`` allowed would overflow the block dim, and
        # raising it past ``block_size`` would give threads with no elements to fold.
        #
        # ⚠ ONLY WHEN A WIDER COPY IS ACTUALLY ON THE TABLE (LEDGER E067).  The whole
        # justification above is "lowering ``threads_per_row`` is the direct lever on the
        # ACCESS WIDTH" -- so at ``vec_width == 1`` there is no wider copy to enable and
        # the narrowing is pure loss: it shrinks the reduction thread axis, costs lanes,
        # and buys nothing.  It also fires UNASKED, because ``config_spec.normalize()``
        # fills an omitted ``cute_threads_per_row`` from the quack ladder
        # (``_fill_missing`` -> ``threads_per_row_for(size_hint)``), so a config that
        # never mentioned the knob still had its thread geometry rewritten.
        #
        # MEASURED both ways.  ``test_looped_reduction_uses_per_thread_lanes`` asks for
        # ``reduction_loop=2048`` with NO TV knobs, normalizes to ``tpr=[64]`` /
        # ``cute_vector_widths=[1]``, and requires ``group_span=1024`` /
        # ``block=(1024, 1, 1)`` -- it got 64.  And the alternative fix (make
        # ``_fill_missing`` return an "auto" sentinel so an omitted key stays unset) was
        # measured WORSE than this one: layer_norm 32768x1024 went 1.048 -> 0.949, because
        # the frozen bench configs omit the key too and DO benefit from the ladder at
        # ``vec > 1``.  Gating on ``vec_width`` keeps the win where the lever is real and
        # restores the documented geometry where it is not.
        if env.backend.name == "cute" and thread_count > 1:
            requested_tpr = env.config_spec.cute_threads_per_row.config_get(
                cast(
                    "list[int]", fn.config.config.get("cute_threads_per_row", []) or []
                ),
                block_index,
                None,
            )
            requested_vec = env.config_spec.cute_vector_widths.config_get(
                cast("list[int]", fn.config.config.get("cute_vector_widths", []) or []),
                block_index,
                1,
            )
            if (
                isinstance(requested_tpr, int)
                and 0 < requested_tpr < thread_count
                and isinstance(requested_vec, int)
                and requested_vec > 1
            ):
                # A power of two, so the lane/vec arithmetic stays exact.  The spec's
                # ``_normalize`` already enforces membership in ``THREADS_PER_ROW_CHOICES``
                # (all powers of two), so this is belt-and-braces for a hand-written
                # config that bypassed normalization.
                if requested_tpr & (requested_tpr - 1) == 0:
                    thread_count = requested_tpr
        self._thread_count = thread_count
        self.block_size = block_size
        self._loop_block_size = block_size
        self._cute_reduction_lane_var: str | None = None
        self._cute_reduction_lane_extent = 1
        self._cute_reduction_vec_width = 1
        # ``"unroll"`` (bf16/fp16 per-element bitcast) -- controls how the lane body
        # emits each per-iter load.  ⛔ The third value, ``"vec"`` (one explicit
        # ``cute.arch.load(ptr, V x elem)`` with its mask DEFERRED to a post-fold
        # scalar), was deleted: measured with the TV arm forced OFF -- so the legacy
        # modes were visible rather than merely shadowed -- it fired at ZERO sites, and
        # 0 of the 40 frozen cells reached it.
        self._cute_reduction_vec_mode = "unroll"
        # The TV layout that owns this reduction's access width, or None when
        # the reduction is not on the TV path.  See ``_build_cute_tv_plan``.
        self._cute_tv_plan: ChunkTVPlan | None = None
        # (atom_var, thr_var) for THE one layout, once emitted.  One tuple per
        # strategy is the structural guarantee that both legs share a slice.
        self._cute_tv_shared: tuple[str, str] | None = None
        # (tensor_name, "S"|"D") -> fragment var, reset per chunk body.
        self._cute_tv_partitions: dict[tuple[str, str], str] = {}
        # ⭐ B3: alias keys whose store-then-load RAW is resolved by FORWARDING the
        # store's fragment to the load, rather than by declining the plan.  Filled
        # by ``_build_cute_tv_plan`` from ``_cute_tv_forwardable_raw_keys`` (the ONE
        # place that classification happens) and read at the emission site by
        # ``memory_ops._cute_tv_forwards_store_fragment``.  Empty means "no RAW";
        # the plan is declined outright when some RAW cannot forward, so a
        # non-empty set here always means every RAW in the IR forwards.
        self._cute_tv_forwarded_raw_keys: frozenset[object] = frozenset()
        # The chunk body list (its last element is the lane loop), where
        # per-chunk ``local_tile``/``partition_*``/fragment declarations go.
        self._cute_tv_chunk_prefix: list[ast.AST] | None = None
        # The emitted ``for vi in range_constexpr(vec)`` node.  Held by
        # reference rather than found via ``lane_body[-1]`` because the store
        # leg appends its flush copies AFTER it, so the tail moves.
        self._cute_tv_constexpr_loop: ast.For | None = None
        # The chunk's tile coordinate along N (``roffset // chunk``).
        self._cute_tv_chunk_index_var: str | None = None
        # The CTA-LOCAL chunk coordinate, for the per-CTA staging buffer.  Equal to
        # ``_cute_tv_chunk_index_var`` without a cluster; offset by the CTA's cluster
        # rank with one.  See the note where it is emitted.
        self._cute_tv_stage_chunk_index_var: str | None = None
        # ``cute_reduction_reload``: where the SECOND read of the row comes from.
        # ``None`` = wherever it comes from today; ``"smem"`` = a per-CTA staged
        # tile written during the first sweep.  Set below, next to the plan,
        # because it is only meaningful when the TV layout owns the access.
        self._cute_tv_reload_from: str | None = None
        # ⭐ ``cute_row_residency``: the REQUESTED residency, resolved ONCE by
        # ``_cute_row_residency_config`` and read by every arm downstream.  Seeded to
        # ``gmem`` -- the no-mechanism baseline -- so a strategy whose TV plan is
        # declined below never looks like it asked for a cache it cannot have.
        self._cute_row_residency_requested: str = ROW_RESIDENCY_GMEM
        # Per-chunk SMEM staging state, all reset per chunk body by
        # ``codegen_device_loop``:
        #   * the ``sX`` tensor var (one per kernel, hoisted to the preamble);
        #   * tensor_name -> the staging partition var already emitted in THIS
        #     chunk body, so the writer and the reader share one tile.
        self._cute_tv_stage_smem_var: str | None = None
        self._cute_tv_stage_partitions: dict[tuple[str, str], str] = {}
        # Which tensors have had their row staged into SMEM at least once.  The
        # reader consults this rather than re-deriving eligibility, so a read can
        # never be emitted for a tensor whose write was declined.
        self._cute_tv_staged_tensors: set[str] = set()
        # ⭐ The ``registers`` analogue: ``tile id -> fragment var already holding it``.
        # Minted per instance for the same reason as every dict above -- a class-level
        # mutable default would be shared by every strategy in the process, and the emitted
        # fragment names repeat across kernels.  See the declaration on the class for why
        # this is keyed on the tile id and why it must NOT be reset per sweep.
        self._cute_tv_rmem_frag_by_tile: dict[tuple[object, ...], str] = {}
        # Memo for the device-IR walk that decides which tensors are read more
        # than once (``_cute_tv_multi_read_tensors``).
        self._cute_tv_multi_read_cache: frozenset[str] | None = None
        # ``cute_cluster_n``: CTAs of a launch cluster that split ONE row and combine
        # through DSMEM (quack ``rmsnorm_config.py:58-82``).  Split request/decision
        # exactly like ``reload_from`` above and for the same reason (E017 trap 4):
        # ``thread_block_dims()`` is not final here, so only knob-and-shape facts are
        # checked now and the geometry-dependent half is ``cute_cluster_feasible()``.
        #
        # ⚠ Requested UNCONDITIONALLY, not only on the TV path.  The cluster splits the
        # reduction EXTENT across CTAs, which is a property of the loop nest, not of the
        # copy width -- so it applies whether or not a ``ChunkTVPlan`` was built.  That
        # matters concretely: MEASURED, the N=32768 cell's best config lands on
        # ``vec=4`` where the plan exists, but the N>=65536 cells' configs also want the
        # cluster and gating on the plan would have coupled two independent decisions.
        self._cute_cluster_n_requested = 1
        # Set by codegen once the launch geometry is committed, so the emitter and the
        # launch cannot disagree about whether a cluster is in play.
        self._cute_cluster_n_emitted = 1
        # ``_cute_cluster_mbar_names`` (fx-node id -> mbarrier var) and
        # ``_cute_cluster_y_name`` are NOT initialised here any more: they are base-class
        # fields whose emitters (``_cute_cluster_mbar_var`` / ``_cute_cluster_y_var``)
        # moved to ``ReductionStrategy`` so PERSISTENT can reach them too.  The mbar dict
        # is materialised lazily by that emitter, on the instance, rather than as a
        # mutable class default -- see its docstring.
        if (
            env.backend.name == "cute"
            and thread_count > 0
            and block_size > thread_count
        ):
            self._cute_reduction_lane_extent = (
                block_size + thread_count - 1
            ) // thread_count
            self._loop_block_size = thread_count * self._cute_reduction_lane_extent
            self._cute_reduction_lane_var = fn.new_var(
                f"reduction_lane_{block_index}",
                dce=False,
            )
            # Read autotuner-selected vector width and partition lane extent
            # into outer × inner = lane_extent/V × V.  When V==1 (default)
            # this preserves the original scalar codegen.
            cute_vector_widths = cast(
                "list[int]",
                fn.config.config.get("cute_vector_widths", []) or [],
            )
            vec_width = env.config_spec.cute_vector_widths.config_get(
                cute_vector_widths,
                block_index,
                1,
            )
            # ── THE TV LAYOUT OWNS V (PORT_SPEC_layout.md §7) ──────────────
            #
            # A ``ChunkTVPlan`` is built first and asked what width is legal;
            # ``vec_width`` from the config is only a CAP.  ``lane_extent`` is
            # then read back OFF the plan.  Contrast the pre-rework code, which
            # divided ``_cute_reduction_lane_extent`` in place by a config
            # value and left every later decline unable to undo it -- that
            # asymmetry (stride committed early, width decided per load site)
            # IS class 1.  Here a decline is ``plan.with_vec(smaller)``, which
            # re-derives the trip count and the atom width together, so index
            # and access width cannot disagree.
            # ``chunk=self._loop_block_size``: the extent one ``for roffset``
            # iteration covers.  It is the ONE loop-specific input the base-class
            # plan builder takes, and it is read AFTER the round-up above.
            plan = self._build_cute_tv_plan(
                fn,
                chunk=self._loop_block_size,
                state_free=True,
                vec_cap=vec_width,
                block_index=block_index,
            )
            if plan is not None and plan.vec > 1:
                mode = _cute_vec_kernel_mode()
                if mode in ("vec", "unroll"):
                    self._cute_tv_plan = plan
                    self._cute_reduction_vec_width = plan.vec
                    self._cute_reduction_vec_mode = mode
                    # Read back off the plan; do NOT divide in place.
                    self._cute_reduction_lane_extent = plan.lane_extent
                    assert plan.covers_chunk(), (
                        f"TV plan does not cover its chunk: {plan.describe()}"
                    )
                    # ⚠ E013 trap 6: this runs BEFORE ``super().__init__``, so
                    # ``self.fn`` / ``self.block_index`` do not exist yet.  Both
                    # are passed explicitly for exactly that reason -- do not
                    # "simplify" them away.
                    self._cute_tv_reload_from = self._cute_reload_from_config(
                        fn, plan, block_index
                    )
        if env.backend.name == "cute" and self._cute_tv_plan is None:
            # ⚠ NO TV PLAN, so no mechanism can serve the second read -- but the config
            # may still have ASKED for one, and that request must not vanish silently.
            # MEASURED before this branch existed: ``cute_vector_widths=[1]`` with
            # ``cute_row_residency=["smem"]`` emitted NO marker at all while the config
            # read ``['smem']`` -- i.e. the loudest decline was the only invisible one,
            # which is the exact defect the marker exists to remove.
            #
            # ⚠ AND IT SITS OUTSIDE THE ``block_size > thread_count`` GUARD ABOVE, which
            # is where I first put it and where it was DEAD: that guard is false whenever
            # the block exactly fills the thread count (MEASURED: bs=1024, tc=1024 at
            # ``vw=1``), which is precisely the no-plan shape this branch is for.
            self._cute_row_residency_requested = self._cute_row_residency_config(
                fn, block_index
            )
            if self._cute_row_residency_requested != ROW_RESIDENCY_GMEM:
                self._cute_row_residency_decline = (
                    "no TV plan was built for this reduction (the copy width "
                    "collapsed to 1), so there is no partitioned row to cache"
                )
        # AFTER the block above, because the request is a function of
        # ``self._loop_block_size`` -- which that block rounds up from ``block_size``.
        # Reading the knob earlier would cap the cluster against the wrong chunk.
        #
        # ``rounded=`` is the plan's own answer: a TV plan predicates its tail, so the
        # cluster may use the rounded extent; without a plan the exact-division rule
        # applies.  Passing ``self._cute_tv_plan is not None`` rather than re-deriving
        # it is what keeps the tiler (the loop bound), the staging size and the cluster
        # reading ONE decision -- E014's four consequences of a single decline are
        # exactly what happens when they disagree.
        #
        # ``_cute_tv_chunk`` is the base class's name for "the extent one chunk
        # covers".  Assigned HERE, after the round-up, so the codegen-time
        # ``cute_tv_rounded_extent`` reads the same number ``_build_cute_tv_plan``
        # and ``_cute_cluster_n_config`` were handed above.
        self._cute_tv_chunk = self._loop_block_size
        if env.backend.name == "cute" and thread_count > 0:
            self._cute_cluster_n_requested = self._cute_cluster_n_config(
                fn,
                block_index,
                chunk=self._loop_block_size,
                rounded=self._cute_tv_plan is not None,
            )
        # ── THE MASK MUST EXIST WHENEVER THE TILE CAN EXCEED ``numel`` ─────────
        #
        # 🔴 THIS LINE IS INVARIANT I4's PRECONDITION, and getting it wrong is a
        # SILENT WRONG ANSWER.  I4 says the op identity is already supplied at the
        # combine by ``_mask_to`` -- but ``_mask_to`` only fires when this
        # ``mask_var`` exists, so the granularity tested here must be the same one
        # the round-up uses, namely ``chunk * cluster_n``.
        #
        # MEASURED, by the ``n_wrong`` A/B (verification step 4), when it was
        # ``known_multiple(numel, chunk)`` alone: at N=12288, chunk=4096,
        # cluster_n=2 the extent IS an exact multiple of the chunk, so no mask was
        # created -- yet the tile granularity is 8192, so the loop still rounded up
        # to 16384 and swept 4096 phantom columns with NO identity gate on the
        # accumulate.  18 of 1440 configs wrong, relerr 0.14 (rms_norm) and 6.4
        # (layer_norm).  Note the *copies* were correctly guarded throughout; what
        # leaked was the stale fragment content reaching the combine, which is
        # precisely the failure mode E005 item 4 warned about and which the
        # ``:1692`` comment misdiagnosed as needing a fill AT THE LOAD.
        #
        # The cluster read here is the REQUESTED one, which is >= the one codegen
        # finally emits (``cute_cluster_feasible`` can only decline).  So this can
        # only ever create a mask the round-up turns out not to need -- a redundant
        # compare, never a missing one.  Fail-closed in the correct direction.
        mask_granularity = self._loop_block_size
        if self._cute_tv_plan is not None:
            mask_granularity = tile_granularity(
                self._loop_block_size, self._cute_cluster_n_requested
            )
        if env.known_multiple(env.block_sizes[block_index].numel, mask_granularity):
            mask_var: str | None = None
        else:
            mask_var = fn.new_var(f"mask_{block_index}", dce=True)
        super().__init__(
            fn=fn,
            block_index=block_index,
            mask_var=mask_var,
            block_size_var=fn.new_var(f"_REDUCTION_BLOCK_{block_index}"),
        )
        self.offset_vars[block_index] = fn.new_var(f"roffset_{block_index}", dce=True)
        self.index_vars[block_index] = fn.new_var(f"rindex_{block_index}", dce=True)

    def _reduction_thread_count(self) -> int:
        return self._thread_count

    def _active_thread_axis_sizes(
        self, state: CodegenState, device_loop: DeviceLoopState
    ) -> dict[int, int]:
        axis_sizes: dict[int, int] = {}
        seen: set[int] = set()
        for loops in state.codegen.active_device_loops.values():
            for loop_state in loops:
                if not isinstance(loop_state, (DeviceLoopState, DeviceGridState)):
                    continue
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                for axis, size in loop_state.thread_axis_sizes.items():
                    axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        current_grid = state.codegen.current_grid_state
        if isinstance(current_grid, DeviceGridState):
            for axis, size in current_grid.thread_axis_sizes.items():
                axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        for axis, size in device_loop.thread_axis_sizes.items():
            axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
        return axis_sizes

    def _cute_cross_warp_reduction_expr(
        self,
        state: CodegenState,
        device_loop: DeviceLoopState,
        input_name: str,
        reduction_type: str,
        default_value: float | bool,
        dtype: torch.dtype,
    ) -> str | None:
        env = CompileEnvironment.current()
        backend = env.backend
        if (
            backend.name != "cute"
            or self._thread_count <= 32
            or backend.is_indexed_reduction(reduction_type)
        ):
            return None

        axis_sizes = self._active_thread_axis_sizes(state, device_loop)
        reduction_axis = self._get_thread_axis()
        axis_sizes[reduction_axis] = max(
            axis_sizes.get(reduction_axis, 1), self._thread_count
        )
        num_threads = 1
        for size in axis_sizes.values():
            num_threads *= size
        group_span = self._thread_count
        if num_threads % group_span != 0:
            return None
        # The two-stage shared-memory reduction assumes its ``lane_var`` is
        # the linear thread index across ALL of the launch block's threads.
        # If ``axis_sizes`` only covers a subset of the planned block dims
        # (e.g. an inner reduction strategy contributes another thread axis
        # that hasn't been entered yet), the emitted reduction would race
        # across the missing axis. Bail out and fall back to the warp-level
        # path in that case.
        planned_dims = self._planned_thread_dims()
        planned_block_threads = planned_dims[0] * planned_dims[1] * planned_dims[2]
        if num_threads != planned_block_threads:
            return None
        lane_expr = backend.thread_linear_index_expr(axis_sizes)
        if lane_expr is None:
            return None

        identity_expr = backend.cast_expr(
            constant_repr(default_value), _dtype_str(dtype)
        )
        group_count = num_threads // group_span
        lane_var = self.fn.new_var("looped_reduce_lane", dce=True)
        lane_in_group_var = self.fn.new_var("looped_reduce_lane_in_group", dce=True)
        lane_mod_pre_var = self.fn.new_var("looped_reduce_lane_mod_pre", dce=True)
        result_var = self.fn.new_var("looped_reduce_result", dce=True)
        device_loop.outer_suffix.append(
            statement_from_string(f"{lane_var} = {lane_expr}")
        )
        device_loop.outer_suffix.append(
            statement_from_string(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
        )
        device_loop.outer_suffix.append(
            statement_from_string(f"{lane_mod_pre_var} = ({lane_in_group_var}) % 1")
        )
        device_loop.outer_suffix.append(
            statement_from_string(
                f"{result_var} = _cute_grouped_reduce_shared_two_stage("
                f"{input_name}, {reduction_type!r}, {identity_expr}, "
                f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
                f"acc_dtype={_dtype_str(dtype)}, "
                f"pre=1, group_span={group_span}, group_count={group_count})"
            )
        )
        # ── the CLUSTER leg: one more combine, across CTAs (PORT_SPEC §9) ──────
        #
        # Composed on TOP of the intra-CTA result rather than fused into it -- see
        # the section header in ``cute/reduce_helpers.py`` for why.  The ordering
        # here is quack's: local combine, then ``cluster_wait``, then the exchange
        # (quack passes ``cluster_wait`` as ``row_reduce``'s ``hook_fn``,
        # ``rmsnorm.py:316``, which places it after the shuffle and before the
        # cross-CTA step).  Emitting the wait EARLIER would be correct but slower;
        # emitting it LATER would race the peers' barrier initialisation.
        cluster_n = self._cute_cluster_emitted_n()
        if cluster_n > 1:
            mbar_var = self._cute_cluster_mbar_var(state)
            cluster_var = self.fn.new_var("looped_reduce_cluster", dce=True)
            group_id_var = self.fn.new_var("looped_reduce_group", dce=True)
            # ⭐ TIER 2: ``group_count`` MUST COUNT ROW GROUPS, NOT THREADS.
            #
            # ``group_count = num_threads // group_span`` above is a THREAD count.  When
            # ``block_sizes[0] > num_threads[0]`` (more rows per CTA than threads on the
            # row axis) the extra rows are covered by a SERIAL loop that the tile strategy
            # mints (``for lane_0 in range(bs0 // nt0)``), and this exchange sits INSIDE
            # it.  So the thread count stops being the row count: measured
            # ``bs=2 nt=1 cn=2`` -> ``group_count=1`` with ``lane_loops=[('lane_0', 2)]``,
            # i.e. one thread arrives TWICE against a barrier armed for one round and both
            # rows index the same ``(group_count, cluster_n)`` slot.  Result: a hard
            # ``unspecified launch failure`` that POISONS THE CUDA CONTEXT.
            #
            # The serial trip count is available as data -- ``grid.lane_loops`` is the
            # list the tile strategy registered, reachable here without asking the sibling
            # strategy for its private dicts (measured ``[('lane_0', 2)]`` at the emission
            # site).  Multiply it in, so the buffer, the barrier's expected-transaction
            # count and the number of stores actually issued are all sized by the same
            # number again.
            grid_state = state.codegen.current_grid_state
            serial_rows = 1
            for _lane_name, lane_extent in (
                getattr(grid_state, "lane_loops", None) or []
            ):
                serial_rows *= max(1, int(lane_extent))
            # ⛔⛔ SIZING THE BUFFER FOR ALL ROWS AT ONCE DEADLOCKS -- MEASURED.
            #
            # The obvious reading of "make group_count count rows" is
            # ``group_count * serial_rows``, arming the barrier for every row's stores.
            # That HANGS (measured: kernel never returns, GPU idle at 0%), and the reason
            # is that the barrier is a per-rendezvous object, not an accumulator:
            # ``expect_tx`` is issued INSIDE the serial loop, and the
            # ``mbarrier_wait(mbar, 0)`` at the end of the SAME iteration blocks until the
            # full byte count arrives -- but the remaining rows' stores are only issued by
            # LATER iterations of the very loop the wait is blocking.  It waits for bytes
            # that cannot exist yet.
            #
            # ⇒ each serial iteration is its own complete exchange: ``cluster_n`` stores
            # in, ``cluster_n`` expected.  The BUFFER, however, must still be per-row so
            # the iterations do not overwrite each other's slots -- those are two different
            # numbers, and the current helper derives both from the single
            # ``group_count`` parameter.  That coupling is the real Tier-2 blocker; see the
            # scoping note.  Keeping the arming per-iteration is what the phase-0 barrier
            # can express today.
            cluster_group_count = group_count
            # ⛔ AND ``group_id`` MUST DISTINGUISH ROW 0 FROM ROW 1 OF THE SAME THREAD.
            # Resizing the buffer alone is not enough: ``group_id`` is derived from the
            # thread index, so every serial iteration would still write the SAME slot and
            # the last row would win.  The serial loop variable is the row index within
            # this thread, so it is exactly the missing coordinate.  ``group_count`` is the
            # per-iteration stride (thread-groups per row), which keeps the existing
            # thread-derived component contiguous within a row.
            device_loop.outer_suffix.append(
                statement_from_string(f"{group_id_var} = ({lane_var}) // {group_span}")
            )
            # ⭐ ``row_slot`` = which BUFFER ROW this serial iteration owns, and ``phase`` =
            # which barrier parity this iteration waits on.  Both are the serial row index;
            # they are separate parameters because they answer different questions
            # (storage vs rendezvous) and only one of them is a parity.
            # ⚠ EMITTED ONLY WHEN THEY DO SOMETHING.  A non-serial reduction (the common
            # case, and 13 of the 40 frozen cells) leaves ``serial_kwargs`` EMPTY rather
            # than spelling out ``row_slot=Int32(0), phase=Int32(0), buf_rows=<group_count>``
            # -- which are exactly the callee's defaults, so the two spellings are
            # semantically identical.  Omitting them keeps those cells BYTE-identical, which
            # turns "this diff is a provably inert default" from an argument I have to make
            # into one the hasher makes for me.  (Per APPROVED_CHANGES.md's "READ ONCE":
            # explicit-default noise is an *acceptable* diff, but a free byte-identical is
            # strictly better than an acceptable diff.)
            serial_kwargs = ""
            if serial_rows > 1:
                lane_names = [n for n, _e in grid_state.lane_loops]
                # One serial row axis only: with two nested row lane loops the row index is
                # a mixed-radix combination and a single stride would alias.  Assert rather
                # than emit a plausible wrong answer.
                assert len(lane_names) == 1, (
                    "cluster + multiple serial row lane loops is not expressible: "
                    f"lane_loops={grid_state.lane_loops}"
                )
                row_var = lane_names[0]
                serial_kwargs = (
                    f", row_slot=({group_id_var}) * {serial_rows}"
                    f" + cutlass.Int32({row_var})"
                    f", phase=cutlass.Int32({row_var}) % 2"
                    f", buf_rows={group_count * serial_rows}"
                )
            device_loop.outer_suffix.append(
                statement_from_string("cute.arch.cluster_wait()")
            )
            device_loop.outer_suffix.append(
                statement_from_string(
                    f"{cluster_var} = _cute_cluster_reduce("
                    f"{result_var}, {reduction_type!r}, "
                    f"{group_id_var}, {lane_in_group_var}, {mbar_var}, "
                    f"acc_dtype={_dtype_str(dtype)}, "
                    f"group_count={cluster_group_count}, cluster_n={cluster_n}"
                    f"{serial_kwargs})"
                )
            )
            return cluster_var
        return result_var

    def codegen_device_loop(self, state: CodegenState) -> DeviceLoopState:
        env = CompileEnvironment.current()
        block_index = self.block_index
        numel = env.block_sizes[block_index].numel
        offset_var = self.offset_var(block_index)
        index_var = self.index_var(block_index)
        block_size_var = self.block_size_var(block_index)
        assert block_size_var is not None
        if state.device_function.constexpr_arg(block_size_var):
            state.codegen.host_statements.append(
                statement_from_string(f"{block_size_var} = {self._loop_block_size!r}")
            )
        inner_body: list[ast.AST] = [
            statement_from_string(
                f"{index_var} = {offset_var} + {self._index_init_expr(f'({block_size_var})', env.index_type(), block_index)}"
            ),
        ]
        reduction_lane_var = self._cute_reduction_lane_var
        vec = self._cute_reduction_vec_width
        # Detect whether the upcoming graph contains a reduction op so we
        # can choose between the reduce-sweep shape (single vec load +
        # V-fold) and the consume-sweep shape (V scalar elementwise ops in
        # an inner constexpr loop).
        active_graph_info = getattr(state.codegen, "_cute_active_graph_info", None)
        graph_has_reduction = True
        if vec > 1 and active_graph_info is not None:
            graph = getattr(active_graph_info, "graph", None)
            if graph is not None:
                graph_has_reduction = any(
                    isinstance(n.meta.get("lowering"), ReductionLowering)
                    for n in graph.nodes
                )
        # Unroll the lane body via a constexpr V-loop when the consume sweep
        # mixes scalars (no reduction in graph), OR when the reduce sweep is
        # in ``"unroll"`` mode (bf16/fp16 inputs that the CuTe DSL can't
        # safely subscript as a vector), OR -- ALWAYS -- when the TV layout is
        # driving this reduction.
        #
        # The TV clause is ``PORT_SPEC_layout.md`` §8b Rule 1 turned into code.
        # The layout "is" the vectorization, so a per-element loop looks like
        # pure overhead; keeping it anyway is what makes under-reading
        # inexpressible rather than merely absent.  Both the loop's trip count
        # and the copy's width now come from ``plan.vec``, so they cannot
        # disagree -- whereas eliding the loop would make coverage depend on
        # the copy alone, which is the bf16 trap.
        #
        # It is also what fixes class 1 on the ``"vec"``-mode kernels
        # (cross_entropy): those took the ``else`` branch below, which emits
        # ``rindex = roffset + tid*V + lane*TC*V`` -- a V-scaled stride with a
        # single scalar read per iteration, i.e. 1 of every V elements.
        consume_unroll = vec > 1 and (
            not graph_has_reduction
            or self._cute_reduction_vec_mode == "unroll"
            or self._cute_tv_plan is not None
        )
        # Map from (tensor_name, base_expr) -> (hoist_var, dtype) so the
        # dispatcher can reuse one hoist per (tensor, base) pair instead of
        # emitting a fresh vec load on every dispatcher call.
        self._cute_lane_vec_loads: dict[tuple[str, str], tuple[str, torch.dtype]] = {}
        # Variable name holding the per-lane-iter base index for vec hoists
        # in ``unroll`` mode — the dispatcher uses this to compute the vec
        # pointer offset once.
        self._cute_lane_base_index_var: str | None = None
        vec_lane_var: str | None = None
        base_expr: str = ""
        if reduction_lane_var is not None:
            if vec > 1:
                # base = offset + thread_idx*V + lane*(THREADS*V)
                #
                # ⭐ THE PLAN OWNS THE STRIDES AND THE TERM ORDER
                # (``ChunkTVPlan.emit_lane_base``).  This site was already correct, but
                # the same formula hand-written at three sites is how the transposed
                # stride shipped a wrong answer on the tile path, so the private copy
                # is retired here too.
                #
                # ⚠ This branch also runs with NO TV plan (the legacy ``unroll`` mode),
                # which needs the identical expression -- ``unroll`` reads the elements
                # one at a time from the same interleaved partition a copy would fetch
                # in one go.  So rather than keep a second spelling for that arm, both
                # go through the plan and the no-plan arm builds the plan-shaped object
                # from the numbers it already has: ``vec`` and ``_thread_count`` are
                # exactly what a plan here would carry (asserted below when one does).
                thread_term = self._index_init_expr(
                    f"({block_size_var})", env.index_type(), block_index
                )
                if (plan := self._cute_tv_plan) is not None:
                    assert (
                        plan.vec == vec and plan.threads_per_row == self._thread_count
                    ), (
                        f"TV plan geometry disagrees with the emitted lane loop: "
                        f"plan.vec={plan.vec} vec={vec} "
                        f"plan.tpr={plan.threads_per_row} tc={self._thread_count}"
                    )
                base_expr = emit_lane_base_for(
                    threads_per_row=self._thread_count,
                    vec=vec,
                    offset_expr=offset_var,
                    lane_var=reduction_lane_var,
                    thread_expr=thread_term,
                )
                if consume_unroll:
                    vec_lane_var = self.fn.new_var(
                        f"reduction_vec_lane_{block_index}",
                        dce=False,
                    )
                    self._cute_lane_base_index_var = self.fn.new_var(
                        f"reduction_lane_base_{block_index}",
                        dce=False,
                    )
                    # index_var = base + vi  (used inside the constexpr loop)
                    inner_body[0] = statement_from_string(
                        f"{index_var} = {self._cute_lane_base_index_var} + cutlass.Int32({vec_lane_var})"
                    )
                else:
                    inner_body[0] = statement_from_string(f"{index_var} = {base_expr}")
            else:
                inner_body[0] = statement_from_string(
                    f"{index_var} = {offset_var} + {self._index_init_expr(f'({block_size_var})', env.index_type(), block_index)} + cutlass.Int32({reduction_lane_var}) * {self._thread_count}"
                )
        if (mask_var := self._mask_var) is not None:
            inner_body.append(
                statement_from_string(
                    f"{mask_var} = {index_var} < {state.sympy_expr(numel)}"
                )
            )
        body = inner_body
        if reduction_lane_var is not None:
            from .tile_strategy import _create_lane_loop

            if consume_unroll and vec_lane_var is not None:
                # for vi in cutlass.range_constexpr(V): ...
                #
                # ⚠ This loop is NOT elided on the TV path either
                # (``PORT_SPEC_layout.md`` §8b Rule 1).  On the TV path it is
                # genuinely redundant with the copy's width, but keeping it
                # means the ONLY way to under-read is for the copy width and
                # the loop bound to disagree -- and both now come from
                # ``plan.vec``.  Eliding it would make correctness depend on
                # the copy alone, which is exactly the bf16 trap.
                vec_for = cast(
                    "ast.For",
                    ast.parse(
                        f"for {vec_lane_var} in cutlass.range_constexpr({vec}):\n"
                        f"    pass"
                    ).body[0],
                )
                vec_for.body = inner_body  # type: ignore[assignment]
                # The lane-loop body holds the per-lane base index, then any
                # dispatcher-requested vec hoists, then the constexpr loop.
                base_stmt = statement_from_string(
                    f"{self._cute_lane_base_index_var} = {base_expr}"
                )
                lane_body: list[ast.AST] = [
                    base_stmt,
                    vec_for,
                ]
                self._cute_tv_constexpr_loop = vec_for
                body = [
                    _create_lane_loop(
                        reduction_lane_var,
                        self._cute_reduction_lane_extent,
                        lane_body,
                    )
                ]
                # Stash the lane body list so the dispatcher can splice
                # hoists in (BETWEEN base_stmt and vec_for) as it runs.
                self._cute_lane_body = lane_body
                # Per-chunk TV declarations live above the lane loop, so one
                # ``local_tile``/``partition_*`` serves every lane iteration.
                #
                # NOTE the mutate-in-place idiom, matching ``_cute_lane_body``:
                # ``body`` is the SAME list object that becomes the outer
                # ``for roffset`` loop's body, and the dispatcher inserts into
                # it while load/store sites are codegen'd.  Rebuilding it with
                # ``[*prefix, *body]`` would snapshot an empty prefix.
                if self._cute_tv_plan is not None:
                    self._cute_tv_partitions = {}
                    # Per-chunk, like ``_cute_tv_partitions``: the staging
                    # partitions are ``local_tile``s of THIS chunk body, so a
                    # partition from the previous sweep must not be reused (it
                    # would be out of scope).  ``_cute_tv_staged_tensors`` is
                    # deliberately NOT reset -- it records that a tensor was
                    # staged in the FIRST sweep, which is what makes the SECOND
                    # sweep's read legal.
                    self._cute_tv_stage_partitions = {}
                    chunk_index_var = self.fn.new_var(
                        f"_tv_chunk_{block_index}", dce=False
                    )
                    self._cute_tv_chunk_index_var = chunk_index_var
                    # ⭐ AND THE EXPRESSION IT HOLDS, for the tile-id channel (task 4).
                    #
                    # ⛔ THE VARIABLE NAME IS NOT AN IDENTITY.  Each sweep of one row mints
                    # its OWN ``_tv_chunk_N``, and MEASURED on the two-moment norm they all
                    # hold the SAME value:
                    #     _tv_chunk_1 = roffset_1 // _REDUCTION_BLOCK_1
                    #     _tv_chunk_2 = roffset_1 // _REDUCTION_BLOCK_1
                    #     _tv_chunk_3 = roffset_1 // _REDUCTION_BLOCK_1
                    # so a tile id keyed on the NAME reports three different tiles where
                    # there is one, and can never identify a re-read -- which is precisely
                    # the defect ``fuse_tv_copy_sweeps`` works around by unparsing and
                    # inlining single-assignment temporaries before comparing text.
                    # Recording the EXPRESSION is what makes the id an identity.
                    self._cute_tv_chunk_index_expr = f"{offset_var} // {block_size_var}"
                    body.insert(
                        0,
                        statement_from_string(
                            f"{chunk_index_var} = {offset_var} // {block_size_var}"
                        ),
                    )
                    # ⚠ THE STAGING TILE NEEDS A **CTA-LOCAL** CHUNK INDEX.
                    #
                    # ``chunk_index_var`` is a GLOBAL chunk number, which is what the
                    # gmem ``local_tile`` wants (it tiles the whole row).  But the
                    # staging buffer only holds this CTA's share -- ``num_chunks =
                    # N / cluster_n / chunk`` entries -- because that is exactly what
                    # makes its footprint flat in N.  Indexing it globally would run
                    # off the end for every CTA of rank > 0.
                    #
                    # Without a cluster the two indices are identical, and the same var
                    # is reused so the no-cluster path stays byte-identical.
                    cluster_n_here = self._cute_cluster_emitted_n()
                    if cluster_n_here > 1:
                        per_cta = self._cute_cluster_per_cta_extent()
                        assert per_cta is not None
                        # Named with the ``_tv_chunk`` prefix on purpose: the gate test
                        # ``reload_smem_is_chunk_indexed`` asserts the staged
                        # ``local_tile``'s column coordinate is a ``_tv_chunk*`` var
                        # (a constant there means every chunk aliases one slot --
                        # MEASURED relerr 261.6, and FASTER than correct).  Keeping the
                        # prefix means that guard covers the clustered form too instead
                        # of being silently bypassed by a new variable name.
                        stage_chunk_var = self.fn.new_var(
                            f"_tv_chunklocal_{block_index}", dce=False
                        )
                        chunks_per_cta = per_cta // self._loop_block_size
                        body.insert(
                            1,
                            statement_from_string(
                                f"{stage_chunk_var} = {chunk_index_var} - "
                                f"{self._cute_cluster_y_var(state)} * {chunks_per_cta}"
                            ),
                        )
                        self._cute_tv_stage_chunk_index_var = stage_chunk_var
                    else:
                        self._cute_tv_stage_chunk_index_var = chunk_index_var
                    self._cute_tv_chunk_prefix = body
                    self._cute_tv_lane_loop = body[-1]
            else:
                body = [
                    _create_lane_loop(
                        reduction_lane_var,
                        self._cute_reduction_lane_extent,
                        inner_body,
                    )
                ]

        # ── TRAP 1: ``cluster_n`` GOES IN THE TILER, not just the launch ────────
        #
        # quack ``reduction_base.py:47`` puts ``cluster_n`` in ``num_blocks_N``'s
        # DENOMINATOR, so ``tiler_mn[1]`` *shrinks* as the cluster grows and the
        # per-CTA SMEM footprint stays FLAT as N rises (32 KB across an 8x range of N,
        # PORT_SPEC §2b).  helion has no ``tiler_n`` variable -- its analogue is the
        # extent of this ``for roffset`` loop -- so the tiler edit IS this loop bound:
        # CTA ``cy`` of the cluster covers ``[cy * per_cta, (cy+1) * per_cta)`` instead
        # of the whole row.
        #
        # The one-line diagnostic PORT_SPEC §9c asks for: if the staged SMEM footprint
        # grows with N, ``cluster_n`` is missing from the tiler.  It cannot be, here:
        # ``_cute_stage_num_chunks`` divides by the same ``cluster_n``, so the two move
        # together by construction rather than by agreement.
        loop_begin = "0"
        loop_end = state.sympy_expr(numel)
        # ── AND THE RAGGED TAIL IS ALSO A TILER EDIT ────────────────────────────
        #
        # quack's ``num_blocks_N`` is a ``ceil_div``, so its tile covers ``N' >= N``.
        # helion's analogue of ``tiler_n`` is this loop's extent, so the round-up IS
        # this bound.  ``cute_tv_rounded_extent`` returns ``None`` whenever the tile
        # already equals ``N``, which is what keeps every divisible cell's emitted
        # text byte-identical.  The out-of-range iterations it admits are made
        # harmless by the per-lane guard on each ``cute.copy`` (invariant I3) and
        # the op identity at each combine (I4).
        rounded = self.cute_tv_rounded_extent()
        if rounded is not None:
            loop_end = str(rounded)
        cluster_n = self._cute_cluster_emitted_n()
        if cluster_n > 1:
            per_cta = self._cute_cluster_per_cta_extent()
            assert per_cta is not None, (
                "cluster requested but the per-CTA extent is not static; "
                "cute_cluster_feasible should have declined"
            )
            cy_var = self._cute_cluster_y_var(state)
            loop_begin = f"{cy_var} * {per_cta}"
            loop_end = f"{cy_var} * {per_cta} + {per_cta}"
        for_node = create(
            ast.For,
            target=create(ast.Name, id=offset_var, ctx=ast.Store()),
            iter=expr_from_string(
                self.get_range_call_str(
                    state.config,
                    [self.block_index],
                    begin=loop_begin,
                    end=loop_end,
                    step=block_size_var,
                ),
            ),
            body=body,
            orelse=[],
            type_comment=None,
        )
        # Extract end_var_name from the actual numel expression used in the range()
        from .tile_strategy import LoopDimInfo

        end_var_name = state.sympy_expr(numel)
        block_id_to_info = {
            block_index: LoopDimInfo(end_var_name=end_var_name, end_expr=numel)
        }
        tracker = ThreadAxisTracker()
        if self._thread_count > 0:
            tracker.record(block_index, self._get_thread_axis(), self._thread_count)
        return DeviceLoopState(
            self,
            for_node=for_node,
            inner_statements=inner_body,
            block_id_to_info=block_id_to_info,
            thread_axis_sizes=tracker.sizes,
            block_thread_axes=tracker.block_axes,
        )

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        _log_cute_reduction_layout(state)
        # See ``PersistentReductionStrategy.codegen_reduction``: record the branch
        # path of this reduction's thread axis so a mutually-exclusive sibling
        # branch's free ``hl.arange`` can reuse the axis (CuTe backend only).
        if CompileEnvironment.current().backend.name == "cute":
            state.codegen.record_cute_strategy_axis_branch_path(self._get_thread_axis())
        with install_inductor_kernel_handlers(state.codegen, {}):
            env = CompileEnvironment.current()
            backend = env.backend
            device_loop = state.codegen.active_device_loops[self.block_index][-1]
            assert isinstance(device_loop, DeviceLoopState)
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_input.size()])
            # Promotes fp16/bf16 -> fp32 and integer sum/prod -> int64.
            acc_dtype = reduction_acc_dtype(reduction_type, fake_input.dtype)
            default = ir.Reduction.default_accumulator(reduction_type, acc_dtype)
            assert isinstance(default, (float, int, bool))
            assert state.fx_node is not None
            acc = self.fn.new_var(f"{state.fx_node.name}_acc", dce=True)
            acc_full = backend.full_expr(shape_dims, constant_repr(default), acc_dtype)
            device_loop.outer_prefix.append(
                statement_from_string(f"{acc} = {acc_full}")
            )
            result = self.fn.new_var(state.fx_node.name, dce=True)
            if not backend.is_indexed_reduction(reduction_type):
                # ⛔ THE V-FOLD BLOCK IS GONE WITH THE ``"vec"`` MODE.
                #
                # It folded a length-V vector to a scalar and gated it by the masks
                # the vec load had DEFERRED (``_cute_pending_vec_masks``).  Both were
                # reachable only from ``vec_mode == "vec"``: that arm was the sole
                # writer of ``_cute_emitted_vec_load`` and the sole producer of a
                # pending mask.  With the mode deleted this block could never fire, so
                # keeping it would be dead code that still *reads* as a live mask path.
                #
                # ⭐ EVERY SURVIVING MODE KEEPS ITS MASK WHERE IT BELONGS, per element:
                # the TV path's ``rindex = base + vi`` inside ``range_constexpr(vec)``
                # IS the fragment slot ``vi`` holds, so the caller's ``x if mask else
                # <identity>`` still sits between the load and the combine; the
                # ``unroll`` / ``tile_unroll`` modes return a per-element scalar for the
                # same reason.  Deferral existed only because ``vec`` handed the
                # combine a whole vector.
                vec_input = input_name
                combine_expr = backend.reduction_combine_expr(
                    reduction_type, acc, vec_input, acc_dtype
                )
                state.add_statement(f"{acc} = {combine_expr}")
                expr = self._cute_cross_warp_reduction_expr(
                    state,
                    device_loop,
                    acc,
                    reduction_type,
                    default,
                    acc_dtype,
                ) or self.call_reduction_function(
                    acc,
                    reduction_type,
                    dim,
                    fake_input,
                    fake_output,
                )
            else:
                acc_index = self.fn.new_var(f"{state.fx_node.name}_acc_index", dce=True)
                index_dtype = env.index_dtype
                device_loop.outer_prefix.append(
                    statement_from_string(
                        f"{acc_index} = {backend.reduction_index_init_expr(shape_dims, index_dtype)}"
                    )
                )
                index = self.broadcast_str(
                    self.index_var(self.block_index), fake_input, dim
                )
                for stmt in backend.argreduce_loop_update_statements(
                    reduction_type=reduction_type,
                    acc=acc,
                    acc_index=acc_index,
                    value=input_name,
                    index=index,
                ):
                    state.add_statement(stmt)
                expr = self.call_indexed_reduction(
                    acc,
                    acc_index,
                    reduction_type,
                    dim,
                    fake_output,
                )
            # Ensure the final reduction result matches torch.* dtype semantics
            expr = self.maybe_reshape(expr, dim, fake_input, fake_output)
            expr = backend.cast_expr(expr, _dtype_str(fake_output.dtype))
            device_loop.outer_suffix.append(statement_from_string(f"{result} = {expr}"))

            # Optional: emit a dtype static assert right after the assignment when enabled
            if env.settings.debug_dtype_asserts:
                device_loop.outer_suffix.append(
                    statement_from_string(
                        f"tl.static_assert({result}.dtype == {_dtype_str(fake_output.dtype)})"
                    )
                )
            return expr_from_string(result)


class BlockReductionStrategy(ReductionStrategy):
    """This is used when we are reducing over a tile rather than an entire tensor."""

    def __init__(
        self,
        state: CodegenState,
        block_index: int,
    ) -> None:
        super().__init__(
            fn=state.device_function,
            block_index=block_index,
            mask_var=state.codegen.mask_var(block_index),
            block_size_var=None,
        )
        self.offset_vars[block_index] = "0"
        # Store reference to codegen to access existing index variables
        self._codegen = state.codegen

    def index_var(self, block_idx: int) -> str:
        # Use the existing index variable from the active device loop
        # instead of the newly created one from TileStrategy.__init__
        return self._codegen.index_var(block_idx)

    def _reduction_thread_count(self) -> int:
        """Return the live thread extent of the reduced tile block.

        Unlike a real reduction axis, a tile block reduced over its inner
        (tiled) dim is mapped to a normal tile thread axis (plus, when the
        block is wider than its thread extent, a runtime lane loop). When that
        block has a live thread axis the partials must be combined ACROSS those
        threads, so report the thread extent (0 when the block has no live
        thread axis, e.g. a pure lane loop / serial dim — the base behavior).
        """
        extent = self.fn.tile_strategy.thread_extent_for_block_id(self.block_index)
        return extent if extent is not None and extent > 0 else 0

    def _lane_loop_cross_warp_group_params(
        self,
    ) -> tuple[int, int, int, str] | None:
        """Return ``(pre, group_span, group_count, lane_expr)`` for a tile block
        that is reduced over its inner (tiled) dim AND carries a runtime lane
        loop, or ``None`` when no cross-warp de-interleaving is required.

        When the reduced block is mapped to a thread axis ABOVE a sibling tile
        axis (e.g. ``hl.tile([o, d])`` where ``d`` is reduced, ``d`` on
        ``thread_idx[1]`` and ``o`` on ``thread_idx[0]``), the threads that
        share a row are strided by the sibling axis extent (stride 32 here), so
        the reduce group is spread across warps. A plain
        ``cute.arch.warp_reduction_*`` would fold together CONSECUTIVE lanes
        (different rows). Compute the grouped/strided parameters that the
        cross-warp ``_cute_grouped_reduce_shared_two_stage`` helper needs:

        * ``pre`` — product of live thread extents on axes *below* the reduce
          axis (the sibling rows that must stay distinct);
        * ``group_span`` — ``pre`` times the reduce-axis extent (the lanes that
          form one reduction);
        * ``group_count`` — the number of independent groups in the CTA;
        * ``lane_expr`` — the linear thread index across all live thread axes.

        Returns ``None`` (so the caller keeps the plain warp-reduce / no-op
        finalize) unless the reduce group is genuinely cross-warp
        (``pre > 1`` and ``group_span`` a multiple of 32 greater than 32).
        """
        env = CompileEnvironment.current()
        backend = env.backend
        if backend.name != "cute":
            return None
        block_axes, axis_sizes = self._active_thread_layout()
        reduce_axis = block_axes.get(self.block_index)
        if reduce_axis is None:
            reduce_axis = self._aliased_active_thread_axis(block_axes)
        if reduce_axis is None:
            return None
        # Live thread extents per axis (sibling axes included) so the linear
        # lane index strides are computed correctly.
        logical_axis_sizes = {
            axis: size for axis, size in axis_sizes.items() if size > 1
        }
        if reduce_axis not in logical_axis_sizes:
            return None
        pre = 1
        for axis in range(reduce_axis):
            pre *= logical_axis_sizes.get(axis, 1)
        if pre <= 1:
            # The reduce axis is already at the bottom of the linear lane
            # index: consecutive warp lanes belong to the reduction, so the
            # plain warp reduce is correct (no de-interleaving needed).
            return None
        reduce_extent = logical_axis_sizes[reduce_axis]
        group_span = pre * reduce_extent
        # ⛔⛔ A SINGLE-WARP GROUP STILL NEEDS DE-INTERLEAVING -- DECLINING IT WAS A
        # SILENT WRONG ANSWER.  This guard used to read
        # ``group_span <= 32 or group_span % 32 != 0 -> return None``, justified as
        # "single-warp groups are not handled by the cross-warp two-stage path".  The
        # premise is true and the conclusion does not follow: ``pre > 1`` means the
        # reduce group is STRIDED over the linear lane index, and a strided group is
        # mis-folded by a consecutive-lane ``warp_reduction_*`` whether or not it fits
        # in one warp.  Returning ``None`` here hands the caller exactly that plain
        # warp reduce (``_finalize_lane_reduce_marker``'s fallback), which folds
        # DIFFERENT ROWS together.
        #
        # ⭐ AND THE SINGLE-WARP FORM ALREADY EXISTS -- only this producer refused to
        # ask for it.  ``_finalize_lane_reduce_marker`` (``tile_strategy.py``) documents
        # a three-way dispatch and implements it: ``group_span > 32`` (a multiple of 32)
        # goes to the cross-warp ``_cute_grouped_reduce_shared_two_stage``, and
        # ``1 < group_span <= 32`` with ``pre > 1`` goes to
        # ``_cute_grouped_reduce_warp``, which masks by ``lane % pre`` and reduces over
        # ``threads_in_group=group_span`` -- precisely this case.  So the fix is to stop
        # declining, not to write a new helper.
        #
        # MEASURED (``{-1,+1}`` integer data, so a correct kernel MUST be bit-exact),
        # on ``out[tm,tn] = v - sum(v,-1)`` at ``block_sizes=[32,32]``:
        # ``num_threads=[4,8]`` and ``[8,4]`` (``group_span == 32``) were WRONG by
        # maxabs 24 / 20 and are now bit-exact; ``[8,8]``/``[16,8]``/``[8,16]``/
        # ``[16,16]`` were already correct and are byte-identical.  On a 3-D tile the
        # same change fixes ``group_span`` 8 and 16 (``num_threads=[2,2,2]``,
        # ``[2,2,4]``, ``[2,4,4]``: all wrong -> all bit-exact), so the defect was never
        # specific to span 32 -- it was every ``pre > 1`` group that is not a >32
        # multiple of 32.  ⚠ Wrong on ``origin/main`` too, identically, so this fixes an
        # inherited bug rather than branch damage.
        #
        # ⚠ ``pre <= 1`` is still declined ABOVE: with the reduce axis at the bottom of
        # the linear index consecutive lanes DO belong to the reduction, so the plain
        # warp reduce is correct there and cheaper (one shuffle instead of ``pre``).
        # What remains declined here is only the genuinely unhandled geometry: a group
        # wider than a warp that is not warp-aligned, which neither helper can fold.
        if group_span > 32 and group_span % 32 != 0:
            return None
        num_threads = 1
        for size in logical_axis_sizes.values():
            num_threads *= size
        if num_threads % group_span != 0:
            return None
        lane_expr = backend.thread_linear_index_expr(logical_axis_sizes)
        if lane_expr is None:
            return None
        return pre, group_span, num_threads // group_span, lane_expr

    def _active_thread_layout(self) -> tuple[dict[int, int], dict[int, int]]:
        axis_sizes: dict[int, int] = {}
        block_axes: dict[int, int] = {}
        seen: set[int] = set()
        for loops in self._codegen.active_device_loops.values():
            for loop_state in loops:
                if not isinstance(loop_state, (DeviceLoopState, DeviceGridState)):
                    continue
                key = id(loop_state)
                if key in seen:
                    continue
                seen.add(key)
                for axis, size in loop_state.thread_axis_sizes.items():
                    axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
                block_axes.update(loop_state.block_thread_axes)
        current_grid = getattr(self._codegen, "current_grid_state", None)
        if isinstance(current_grid, DeviceGridState):
            for axis, size in current_grid.thread_axis_sizes.items():
                axis_sizes[axis] = max(axis_sizes.get(axis, 1), size)
            block_axes.update(current_grid.block_thread_axes)
        return block_axes, axis_sizes

    def _aliased_active_thread_axis(self, block_axes: dict[int, int]) -> int | None:
        env = CompileEnvironment.current()
        target_block = self.block_index
        for candidate_block_id, axis in block_axes.items():
            if candidate_block_id == target_block:
                return axis
            source = env.block_sizes[candidate_block_id].block_size_source
            value = getattr(source, "value", None)
            if isinstance(value, torch.SymInt):
                if env.get_block_id(value) == target_block:
                    return axis
            elif isinstance(value, int):
                target_size = env.block_sizes[target_block].size
                if isinstance(target_size, (int, torch.SymInt)) and env.known_equal(
                    target_size, value
                ):
                    return axis
        return None

    def _aliased_strategy_block_id(self) -> int | None:
        env = CompileEnvironment.current()
        target_block = self.block_index
        for strategy in self.fn.tile_strategy.strategies:
            for candidate_block_id in strategy.block_ids:
                if candidate_block_id == target_block:
                    return candidate_block_id
                source = env.block_sizes[candidate_block_id].block_size_source
                value = getattr(source, "value", None)
                if isinstance(value, torch.SymInt):
                    if env.get_block_id(value) == target_block:
                        return candidate_block_id
                elif isinstance(value, int):
                    target_size = env.block_sizes[target_block].size
                    if isinstance(target_size, (int, torch.SymInt)) and env.known_equal(
                        target_size, value
                    ):
                        return candidate_block_id
        return None

    def _strided_thread_reduction_expr(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        default_value: float | bool,
    ) -> str | None:
        env = CompileEnvironment.current()
        backend = env.backend
        current_grid = getattr(self._codegen, "current_grid_state", None)
        allow_lane_axis_fallback = (
            isinstance(current_grid, DeviceGridState) and current_grid.has_lane_loops()
        )
        normalized_dim = dim if dim >= 0 else fake_input.ndim + dim

        def debug(*parts: object) -> None:
            return None

        def block_thread_extent_hint(block_id: int) -> int | None:
            extent = self.fn.tile_strategy.thread_extent_for_block_id(block_id)
            if extent is not None:
                return extent
            configured_threads = env.config_spec.num_threads.config_get(
                self.fn.config.num_threads, block_id, 0
            )
            if configured_threads > 0:
                return configured_threads
            configured_block_size = self.fn.resolved_block_size(block_id)
            return (
                configured_block_size
                if isinstance(configured_block_size, int)
                else None
            )

        def active_loop_states() -> list[DeviceLoopState | DeviceGridState]:
            loop_states: list[DeviceLoopState | DeviceGridState] = []
            seen: set[int] = set()
            for loops in self._codegen.active_device_loops.values():
                for loop_state in loops:
                    if not isinstance(loop_state, (DeviceLoopState, DeviceGridState)):
                        continue
                    key = id(loop_state)
                    if key in seen:
                        continue
                    seen.add(key)
                    loop_states.append(loop_state)
            return loop_states

        loop_states = active_loop_states()
        info_by_block: dict[int, LoopDimInfo] = {}
        if isinstance(current_grid, DeviceGridState):
            info_by_block.update(current_grid.block_id_to_info)
        for loop_state in loop_states:
            for block_id, info in loop_state.block_id_to_info.items():
                info_by_block.setdefault(block_id, info)
        planned_dims = self._planned_thread_dims()
        active_thread_blocks: list[tuple[int, int, int, LoopDimInfo]] = []
        seen_thread_blocks: set[int] = set()
        active_block_axes, active_axis_sizes = self._active_thread_layout()
        active_block_ids = set(info_by_block) | set(active_block_axes)
        for block_id in active_block_ids:
            if block_id in seen_thread_blocks:
                continue
            axis = active_block_axes.get(block_id)
            if axis is None:
                continue
            live_extent = active_axis_sizes.get(axis, 1)
            if live_extent <= 1:
                continue
            extent = block_thread_extent_hint(block_id)
            if extent is None:
                extent = live_extent
            else:
                extent = min(extent, live_extent)
            if extent <= 1:
                continue
            if extent > live_extent:
                continue
            info = info_by_block.get(block_id)
            if info is None:
                size = env.block_sizes[block_id].size
                if not isinstance(size, (int, torch.SymInt)):
                    size = extent
                end_expr = _to_sympy(size)
                info = LoopDimInfo(
                    end_var_name=state.sympy_expr(end_expr),
                    end_expr=end_expr,
                )
            active_thread_blocks.append((block_id, axis, extent, info))
            seen_thread_blocks.add(block_id)
        active_thread_blocks.sort(key=operator.itemgetter(1, 0))
        active_block_axes = {
            block_id: axis for block_id, axis, _, _ in active_thread_blocks
        }
        active_axis_sizes: dict[int, int] = {}
        for _, axis, extent, _ in active_thread_blocks:
            active_axis_sizes[axis] = max(active_axis_sizes.get(axis, 1), extent)

        def resolve_tensor_dim_mapping() -> dict[int, tuple[int, int, int]]:
            mapping: dict[int, tuple[int, int, int]] = {}
            used_block_ids: set[int] = set()
            used_axes: set[int] = set()
            for dim_idx in range(fake_input.ndim):
                dim_size = fake_input.size(dim_idx)
                candidates: dict[tuple[int, int, int], int] = {}
                block_id = env.resolve_block_id(dim_size)
                if block_id is not None and block_id in active_block_axes:
                    axis = active_block_axes[block_id]
                    extent = block_thread_extent_hint(block_id)
                    if extent is not None:
                        candidates[(block_id, axis, extent)] = 0
                for candidate_block_id, axis, extent, info in active_thread_blocks:
                    matches_end = isinstance(
                        dim_size, (int, torch.SymInt)
                    ) and info.is_end_matching(dim_size)
                    matches_thread_extent = isinstance(
                        dim_size, (int, torch.SymInt)
                    ) and env.known_equal(dim_size, extent)
                    candidate_source = getattr(
                        env.block_sizes[candidate_block_id].block_size_source,
                        "value",
                        None,
                    )
                    matches_source_value = (
                        isinstance(dim_size, torch.SymInt)
                        and isinstance(candidate_source, torch.SymInt)
                        and candidate_source._sympy_() == dim_size._sympy_()
                    )
                    if (
                        not matches_end
                        and not matches_thread_extent
                        and not matches_source_value
                    ):
                        continue
                    priority = 3
                    if matches_source_value:
                        priority = 1
                    elif matches_end:
                        priority = 2
                    candidate = (
                        candidate_block_id,
                        axis,
                        extent,
                    )
                    previous = candidates.get(candidate)
                    if previous is None or priority < previous:
                        candidates[candidate] = priority
                chosen: tuple[int, int, int] | None = None
                ordered_candidates = sorted(
                    candidates.items(),
                    key=lambda item: (item[1], item[0][1], item[0][0]),
                )
                for candidate, _priority in ordered_candidates:
                    block_id, axis, _ = candidate
                    if block_id in used_block_ids or axis in used_axes:
                        continue
                    chosen = candidate
                    break
                if (
                    chosen is None
                    and allow_lane_axis_fallback
                    and dim_idx != normalized_dim
                ):
                    for candidate_block_id, axis, extent, _info in sorted(
                        active_thread_blocks, key=operator.itemgetter(1, 0)
                    ):
                        if candidate_block_id in used_block_ids or axis in used_axes:
                            continue
                        chosen = (candidate_block_id, axis, extent)
                        break
                if chosen is None and ordered_candidates:
                    chosen = ordered_candidates[0][0]
                if chosen is None:
                    continue
                mapping[dim_idx] = chosen
                used_block_ids.add(chosen[0])
                used_axes.add(chosen[1])
            return mapping

        if backend.name != "cute":
            debug("skip backend", backend.name)
            return None
        if backend.is_indexed_reduction(reduction_type):
            debug("skip indexed", reduction_type)
            return None
        if self._reduction_block_is_serial():
            debug("skip serial", self.block_index)
            return None
        if state.fx_node is not None:
            for arg in state.fx_node.args:
                if not isinstance(arg, torch.fx.Node):
                    continue
                target_name = getattr(arg.target, "__name__", "")
                if any(
                    name in target_name
                    for name in ("sum", "prod", "mean", "amax", "amin")
                ):
                    debug("skip nested reduction arg", target_name)
                    return None
        if self._reduction_block_has_lane_loops():
            # Lane loops serialize part of the logical tile in Python rather
            # than mapping it to actual threads. Thread-reduction fast paths
            # assume every participating axis is backed by a live thread, so
            # they are invalid under active lane loops.
            debug("skip lane loops")
            return None

        tensor_dim_mapping = resolve_tensor_dim_mapping()
        mapped_block_ids = {block_id for block_id, _, _ in tensor_dim_mapping.values()}
        logical_axes = {
            axis for _, axis, _ in tensor_dim_mapping.values() if axis is not None
        }
        reduce_axis: int | None = None
        reduce_thread_extent: int | None = None
        if 0 <= normalized_dim < fake_input.ndim:
            mapping = tensor_dim_mapping.get(normalized_dim)
            if mapping is not None:
                _, reduce_axis, reduce_thread_extent = mapping

        block_axes = dict(active_block_axes)
        axis_sizes = dict(active_axis_sizes)
        if reduce_axis is not None and reduce_thread_extent is not None:
            axis_sizes[reduce_axis] = max(
                axis_sizes.get(reduce_axis, 1), reduce_thread_extent
            )
            if 0 <= reduce_axis < len(self._codegen.max_thread_block_dims):
                self._codegen.max_thread_block_dims[reduce_axis] = max(
                    self._codegen.max_thread_block_dims[reduce_axis],
                    reduce_thread_extent,
                )
        if reduce_axis is None:
            reduce_axis = self._aliased_active_thread_axis(block_axes)
        if reduce_axis is None:
            aliased_block_id = self._aliased_strategy_block_id()
            # Only treat the reduce dim as a strided thread reduction when the
            # aliased block is actually backed by a *live* thread axis. A block
            # with ``block_size == 1`` (a grid/serial dim such as a size-1
            # contributor axis) reports no live thread extent; its
            # ``_thread_axis_map`` entry still records a phantom local axis that
            # collides with an unrelated sibling block's real thread axis (e.g.
            # the M tile on CuTe). Using that phantom axis would fold the
            # reduction across the sibling's tile instead of squeezing the
            # size-1 dim, so bail to the loop-carried / passthrough path.
            if (
                aliased_block_id is not None
                and self.fn.tile_strategy.thread_extent_for_block_id(aliased_block_id)
                is None
            ):
                aliased_block_id = None
            if aliased_block_id is not None:
                reduce_axis = self.fn.tile_strategy.thread_axis_for_block_id(
                    aliased_block_id
                )
                reduce_thread_extent = block_thread_extent_hint(aliased_block_id)
                if reduce_axis is not None and reduce_thread_extent is not None:
                    if (
                        reduce_axis >= len(planned_dims)
                        or planned_dims[reduce_axis] <= 1
                        or reduce_thread_extent > planned_dims[reduce_axis]
                    ):
                        reduce_axis = None
                        reduce_thread_extent = None
                    else:
                        axis_sizes[reduce_axis] = max(
                            axis_sizes.get(reduce_axis, 1), reduce_thread_extent
                        )
                        if 0 <= reduce_axis < len(self._codegen.max_thread_block_dims):
                            self._codegen.max_thread_block_dims[reduce_axis] = max(
                                self._codegen.max_thread_block_dims[reduce_axis],
                                reduce_thread_extent,
                            )
        if reduce_axis is None:
            strategy = self.fn.tile_strategy.block_id_to_strategy.get(
                (self.block_index,)
            )
            if strategy is not None:
                reduce_axis = self.fn.tile_strategy.thread_axis_for_strategy(strategy)
            if reduce_axis is not None:
                hint = _reduction_threads_from_annotation(state)
                if hint is None:
                    hint = backend.reduction_threads_hint(
                        self.block_size_var(self.block_index)
                    )
                if (
                    hint is not None
                    and reduce_axis < len(planned_dims)
                    and planned_dims[reduce_axis] > 1
                    and hint <= planned_dims[reduce_axis]
                ):
                    axis_sizes[reduce_axis] = max(axis_sizes.get(reduce_axis, 1), hint)
                else:
                    reduce_axis = None
        if reduce_axis is None:
            debug("skip no reduce axis", tuple(fake_input.size()), dim)
            return None
        logical_axes.add(reduce_axis)
        logical_axis_sizes: dict[int, int] = {}
        for block_id, axis, extent, _info in active_thread_blocks:
            if block_id in mapped_block_ids or block_id >= self.block_index:
                logical_axis_sizes[axis] = max(
                    logical_axis_sizes.get(axis, 1),
                    extent,
                )
        if reduce_axis not in logical_axis_sizes and 0 <= reduce_axis < len(
            self._codegen.max_thread_block_dims
        ):
            reduce_size = axis_sizes.get(reduce_axis, 1)
            if reduce_thread_extent is None:
                reduce_size = max(
                    reduce_size,
                    self._codegen.max_thread_block_dims[reduce_axis],
                )
            logical_axis_sizes[reduce_axis] = reduce_size
        if not logical_axis_sizes:
            debug("skip no logical axis sizes", tuple(fake_input.size()), dim)
            return None
        for axis, size in logical_axis_sizes.items():
            if 0 <= axis < len(self._codegen.max_thread_block_dims):
                self._codegen.max_thread_block_dims[axis] = max(
                    self._codegen.max_thread_block_dims[axis], size
                )
        if reduce_thread_extent is None and 0 <= reduce_axis < len(
            self._codegen.max_thread_block_dims
        ):
            logical_axis_sizes[reduce_axis] = max(
                logical_axis_sizes.get(reduce_axis, 1),
                self._codegen.max_thread_block_dims[reduce_axis],
            )

        pre = 1
        for axis in range(reduce_axis):
            pre *= logical_axis_sizes.get(axis, 1)
        reduce_extent = logical_axis_sizes.get(reduce_axis, 1)
        group_span = pre * reduce_extent
        lane_expr = backend.thread_linear_index_expr(logical_axis_sizes)
        if lane_expr is None:
            debug("skip no lane expr", tuple(fake_input.size()), dim)
            return None

        # The accumulator dtype is chosen here and passed EXPLICITLY to every
        # grouped-combine helper (and to the SMEM budget check below); it is never
        # inferred from the identity.  A block reduction folds the tile values as
        # they arrive from the graph, so the base is ``fake_input.dtype`` (float
        # widening is already handled upstream by the graph's own casts) with the
        # integer sum/prod widening applied on top so an int32 sum cannot wrap.
        acc_dtype = widen_integer_acc_dtype(reduction_type, fake_input.dtype)
        dtype = _dtype_str(acc_dtype)
        identity_expr = backend.cast_expr(constant_repr(default_value), dtype)
        num_threads = 1
        for size in logical_axis_sizes.values():
            num_threads *= size
        tensor_thread_axes: set[int] = set()
        tensor_thread_footprint = 1
        for _block_id, axis, extent in tensor_dim_mapping.values():
            if axis is None or extent is None or axis in tensor_thread_axes:
                continue
            tensor_thread_axes.add(axis)
            tensor_thread_footprint *= extent
        if (
            reduce_axis is not None
            and reduce_thread_extent is not None
            and reduce_axis not in tensor_thread_axes
        ):
            tensor_thread_axes.add(reduce_axis)
            tensor_thread_footprint *= reduce_thread_extent
        actual_threads = 1
        planned_dims = self.fn.tile_strategy.thread_block_dims()
        for axis, (recorded, planned) in enumerate(
            zip(self._codegen.max_thread_block_dims, planned_dims, strict=True)
        ):
            if axis not in logical_axis_sizes:
                continue
            size = max(recorded, planned)
            actual_threads *= max(size, 1)
        if num_threads > actual_threads:
            # Some logical axes are being serialized (for example via lane loops)
            # rather than mapped to actual threads. The strided thread-reduction
            # path assumes every participating lane is backed by a live thread, so
            # using it here would read unwritten SMEM partials.
            debug(
                "skip actual threads",
                tuple(fake_input.size()),
                dim,
                num_threads,
                actual_threads,
                logical_axis_sizes,
            )
            return None
        # Skip to the direct ``cute.arch.warp_reduction_*`` path when the
        # entire CTA is a single warp (num_threads == group_span <= 32):
        # the standard ``call_reduction_function`` can emit a one-shot
        # warp_reduction with ``threads_in_group=group_span``.
        #
        # When ``num_threads > group_span`` (e.g. warp-per-row layouts
        # with multiple warps per CTA, each owning one row), keep the
        # ``_cute_grouped_reduce_warp`` path at the bottom — it picks
        # the right per-warp reduce even when other thread axes coexist
        # within the CTA.  The "skip" shortcut would route through
        # ``_needs_loop_carried_accumulator``, which returns True when
        # the reduction block is no longer in ``active_device_loops``
        # (e.g. ``cute_dynamic_row_sum``'s ``acc.sum(-1)`` after the
        # inner ``hl.tile`` exits) and would silently drop the reduce.
        if pre <= 1 and group_span <= 32 and num_threads == group_span:
            debug(
                "skip small direct",
                tuple(fake_input.size()),
                dim,
                "block",
                self.block_index,
                "reduce_axis",
                reduce_axis,
                "pre",
                pre,
                "group_span",
                group_span,
                "mapping",
                tensor_dim_mapping,
                "active_thread_blocks",
                active_thread_blocks,
                "logical_axis_sizes",
                logical_axis_sizes,
            )
            return None
        debug(
            "use strided",
            tuple(fake_input.size()),
            dim,
            "block",
            self.block_index,
            "reduce_axis",
            reduce_axis,
            "pre",
            pre,
            "group_span",
            group_span,
            "mapping",
            tensor_dim_mapping,
            "active_thread_blocks",
            active_thread_blocks,
            "logical_axis_sizes",
            logical_axis_sizes,
        )
        if group_span > 32:
            assert num_threads % group_span == 0, (
                f"num_threads ({num_threads}) must be divisible by "
                f"group_span ({group_span})"
            )
            smem_budget_bytes = _cute_shared_memory_budget_bytes()
            group_count = num_threads // group_span
            lane_var = self.fn.new_var("strided_lane", dce=True)
            lane_in_group_var = self.fn.new_var("strided_lane_in_group", dce=True)
            lane_mod_pre_var = self.fn.new_var("strided_lane_mod_pre", dce=True)
            state.add_statement(f"{lane_var} = {lane_expr}")
            state.add_statement(f"{lane_in_group_var} = ({lane_var}) % {group_span}")
            state.add_statement(f"{lane_mod_pre_var} = ({lane_in_group_var}) % {pre}")
            if group_span % 32 == 0:
                warps_per_group = group_span // 32
                partials_size = group_count * pre * warps_per_group
                results_size = group_count * pre
                # The staging SMEM is sized from the ACCUMULATOR dtype, which is
                # what the helper allocates -- not from the input dtype.
                if (
                    _cute_reduction_smem_bytes(partials_size + results_size, acc_dtype)
                    > smem_budget_bytes
                ):
                    return None
                return self._strided_thread_reduction_expr_shared_two_stage(
                    state=state,
                    input_name=input_name,
                    reduction_type=reduction_type,
                    acc_dtype=acc_dtype,
                    identity_expr=identity_expr,
                    lane_var=lane_var,
                    lane_in_group_var=lane_in_group_var,
                    lane_mod_pre_var=lane_mod_pre_var,
                    pre=pre,
                    group_span=group_span,
                    group_count=group_count,
                )
            if (
                _cute_reduction_smem_bytes(num_threads + group_count * pre, acc_dtype)
                > smem_budget_bytes
            ):
                return None
            return self._strided_thread_reduction_expr_shared_tree(
                state=state,
                input_name=input_name,
                reduction_type=reduction_type,
                acc_dtype=acc_dtype,
                identity_expr=identity_expr,
                lane_var=lane_var,
                lane_in_group_var=lane_in_group_var,
                lane_mod_pre_var=lane_mod_pre_var,
                pre=pre,
                group_span=group_span,
                num_threads=num_threads,
                group_count=group_count,
            )

        return (
            "_cute_grouped_reduce_warp("
            f"{input_name}, {reduction_type!r}, {identity_expr}, {lane_expr}, "
            f"acc_dtype={dtype}, pre={pre}, group_span={group_span})"
        )

    def _strided_thread_reduction_expr_shared_two_stage(
        self,
        *,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        acc_dtype: torch.dtype,
        identity_expr: str,
        lane_var: str,
        lane_in_group_var: str,
        lane_mod_pre_var: str,
        pre: int,
        group_span: int,
        group_count: int,
    ) -> str:
        result_var = self.fn.new_var("strided_reduce_result", dce=True)
        state.add_statement(
            f"{result_var} = _cute_grouped_reduce_shared_two_stage("
            f"{input_name}, {reduction_type!r}, {identity_expr}, "
            f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
            f"acc_dtype={_dtype_str(acc_dtype)}, "
            f"pre={pre}, group_span={group_span}, group_count={group_count})"
        )
        return result_var

    def _strided_thread_reduction_expr_shared_tree(
        self,
        *,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        acc_dtype: torch.dtype,
        identity_expr: str,
        lane_var: str,
        lane_in_group_var: str,
        lane_mod_pre_var: str,
        pre: int,
        group_span: int,
        num_threads: int,
        group_count: int,
    ) -> str:
        result_var = self.fn.new_var("strided_reduce_result", dce=True)
        state.add_statement(
            f"{result_var} = _cute_grouped_reduce_shared_tree("
            f"{input_name}, {reduction_type!r}, {identity_expr}, "
            f"{lane_var}, {lane_in_group_var}, {lane_mod_pre_var}, "
            f"acc_dtype={_dtype_str(acc_dtype)}, "
            f"pre={pre}, group_span={group_span}, "
            f"num_threads={num_threads}, group_count={group_count})"
        )
        return result_var

    def codegen_reduction(
        self,
        state: CodegenState,
        input_name: str,
        reduction_type: str,
        dim: int,
        fake_input: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> ast.AST:
        _log_cute_reduction_layout(state)
        default = ir.Reduction.default_accumulator(reduction_type, fake_input.dtype)
        assert isinstance(default, (float, int, bool))
        env = CompileEnvironment.current()
        dim_size = fake_input.size(dim)
        is_zero_dim = False
        if (
            isinstance(dim_size, int)
            and dim_size == 0
            or isinstance(dim_size, torch.SymInt)
            and env.known_equal(dim_size, 0)
        ):
            is_zero_dim = True
        if is_zero_dim:
            shape_dims = self.fn.tile_strategy.shape_dims([*fake_output.size()])
            return expr_from_string(
                env.backend.full_expr(
                    shape_dims, constant_repr(default), fake_output.dtype
                )
            )
        if (
            strided_expr := self._strided_thread_reduction_expr(
                state, input_name, reduction_type, dim, fake_input, default
            )
        ) is not None:
            expr = strided_expr
        elif self._needs_loop_carried_accumulator():
            # The reduction block is not backed by a live thread axis in the
            # active loop nest (it is iterated either by a serial device loop,
            # by a lane loop, or has no thread axis at all).
            if (
                (
                    self._reduction_block_has_lane_loops()
                    or self._reduction_block_in_device_lane_loop()
                )
                and not self._lane_reduce_marker_unsupported(state)
                and (threads := self._lane_reduce_threads_in_group()) is not None
            ):
                # The block is split across a per-thread lane loop. The
                # single-pass lane loop can only produce a per-lane partial,
                # but every consumer needs the full reduction. Emit a marker
                # that the ``split_lane_loop_reductions`` post-pass rewrites
                # into a two-pass (accumulate across lanes -> combine across
                # ``threads`` -> consume) lane structure.
                from .tile_strategy import _lane_reduce_marker_expr

                acc_dtype = reduction_acc_dtype(reduction_type, fake_input.dtype)
                identity_expr = env.backend.cast_expr(
                    constant_repr(default), _dtype_str(acc_dtype)
                )
                group_params = self._lane_loop_cross_warp_group_params()
                # The reduce group may be spread across warps (the reduced tile
                # dim sits ABOVE a sibling tile axis on the linear thread index).
                # Carry the strided/grouped params so the post-pass finalize uses
                # the cross-warp two-stage shared reduction instead of a
                # (row-cross-contaminating) consecutive-lane warp reduce.
                if group_params is None:
                    group_params = (1, 0, 1, "")
                group_pre, group_span, group_count, group_lane_expr = group_params
                if env.backend.is_indexed_reduction(reduction_type):
                    # No single-accumulator marker exists for an indexed
                    # reduction; lower it into a value marker plus a dependent
                    # index marker (see ``_indexed_lane_reduce_expr``).
                    expr = self._indexed_lane_reduce_expr(
                        state,
                        input_name,
                        reduction_type,
                        dim,
                        fake_input,
                        fake_output,
                        threads,
                        group_pre=group_pre,
                        group_span=group_span,
                        group_lane_expr=group_lane_expr,
                        group_count=group_count,
                    )
                else:
                    # ⭐⭐ G1 AT THE SECOND SITE.  Try the inline fold+combine here too; the
                    # marker below stays as the fallback for every shape the mechanism
                    # cannot express.
                    #
                    # ⚠⚠ THIS IS THE SITE ATTENTION USES, i.e. the hardest placement in the
                    # set and the one that has broken twice.  ``acc = baddbmm(acc, p, v)`` is
                    # a matmul over the lane-distributed axis with an accumulator carried
                    # across the enclosing device loop, sitting next to elementwise-LOOKING
                    # statements (``l_i = l_i * alpha + l_ij``) that are genuinely
                    # lane-invariant.  MEASURED, both failure directions: hoisting the fold
                    # out while leaving the recurrence inside a lane loop gives relerr 1.0
                    # (the completed sum added 64 times), and raising unconditionally at a
                    # decline site here broke ALL 8 attention examples, twice.
                    #
                    # ⇒ what makes it safe to even try is that the inline emitter DECLINES
                    # on an unduplicatable producer (``_contains_unduplicatable_op``, which
                    # names matmuls and cross-thread collectives), so attention's
                    # matmul-carrying body routes to the marker path and its measured-correct
                    # per-lane restore -- unchanged.  The decline is the mechanism, not a
                    # missing feature.
                    inline = self._emit_inline_lane_reduce(
                        state,
                        input_name,
                        reduction_type,
                        identity_expr,
                        threads,
                        acc_dtype_str=_dtype_str(acc_dtype),
                        group_pre=group_pre,
                        group_span=group_span,
                        group_lane_expr=group_lane_expr,
                        group_count=group_count,
                        result_hint=(
                            state.fx_node.name
                            if state.fx_node is not None
                            else reduction_type
                        ),
                    )
                    expr = inline
                    if expr is None:
                        expr = _lane_reduce_marker_expr(
                            input_name,
                            reduction_type,
                            identity_expr,
                            threads,
                            acc_dtype_str=_dtype_str(acc_dtype),
                            group_pre=group_pre,
                            group_span=group_span,
                            group_lane_expr=group_lane_expr,
                            group_count=group_count,
                        )
            else:
                # A serial device loop (or no thread axis at all). A warp-level
                # reduction would fold together unrelated tensor elements, so
                # each iteration contributes only its current scalar value and
                # the surrounding loop-carried accumulator performs the real
                # reduction.
                expr = input_name
        else:
            expr = self.call_reduction_function(
                input_name,
                reduction_type,
                dim,
                fake_input,
                fake_output,
            )
        return expr_from_string(self.maybe_reshape(expr, dim, fake_input, fake_output))
