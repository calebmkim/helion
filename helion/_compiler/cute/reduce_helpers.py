# pyrefly: ignore-errors
"""Device-side combine helpers for CuTe reductions.

ABI note (see ``_redfix/01_BUGS.md`` class 5): every grouped-combine helper here
takes its working dtype as an **explicit** ``acc_dtype`` constexpr argument.  The
accumulator is not a memory object -- it must not inherit a copy atom's dtype,
and it must not be inferred from ``type(identity)`` either.  Inference was the
channel that silently pinned an ``int32`` sum's accumulator to ``int32`` (so a
large sum wrapped), because the identity for an integer reduction is an integer
of the *input* width.  ``acc_dtype`` is required, positional-only-by-keyword, and
is the single source of truth for

* the dtype the values are combined in (both the incoming value and the identity
  are coerced to it, which also keeps the CUTLASS DSL's strict ternary type
  check happy when a caller hands in a narrower value), and
* the dtype (hence the byte size) of the staging shared memory.

Callers pick the accumulator dtype with
``helion._compiler.reduction_strategy.reduction_acc_dtype``.
"""

from __future__ import annotations

import operator

import cutlass
import cutlass.cute as cute

from .cluster_helpers import store_shared_remote

# ``cutlass.Int32(0)`` as a module-level singleton so the cluster helpers below can default
# ``row_slot``/``phase`` without calling a constructor in an argument default (ruff B008: the
# call would run once at import and be shared, which is fine for an immutable zero but is
# exactly the pattern the rule exists to stop people relying on).  Reading it from here is
# both lint-clean and cheaper -- one object instead of one per call site.
#
# SEMANTICS: ``row_slot=0``/``phase=0`` mean "no serial row loop", i.e. a single-row-per-thread
# reduction whose one exchange owns buffer row 0 and waits on barrier phase 0 -- which is
# precisely today's non-serial behaviour, so every existing caller is unaffected.
_CLUSTER_NO_SERIAL_ROW = cutlass.Int32(0)


@cute.jit
def _warp_reduce_sum(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction_sum(value, threads_in_group=threads_in_group)


@cute.jit
def _warp_reduce_max(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction_max(value, threads_in_group=threads_in_group)


@cute.jit
def _warp_reduce_min(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    """Cross-lane ``min``.  ⚠ NOTE THE OPERAND ORDER IN THE LAMBDA -- IT IS NOT A TYPO.

    ⭐ WHY THIS ONE IS NOT AN INTRINSIC LIKE ITS THREE SIBLINGS.  ``sum`` and ``max`` call
    ``cute.arch.warp_reduction_sum`` / ``_max``; there is **no ``warp_reduction_min``** in the
    DSL (VERIFIED against the installed cutlass: ``hasattr(cute.arch,
    "warp_reduction_max")`` is True, ``..._min`` is False).  So ``min`` -- like ``prod`` --
    must go through the generic ``cute.arch.warp_reduction(val, op)``.  That asymmetry is the
    DSL's, not ours.

    ⭐⭐ WHY THE OPERANDS ARE SWAPPED, which is the thing a reader stops on.  The generic
    reduction's loop body is, verbatim from the DSL source::

        val = op(val, shuffle_sync_bfly(val, offset=offset, ...))

    so the accumulator is ALWAYS the first argument and the shuffled peer the second.  Now
    ``min(a, b)`` is Python's builtin, and its contract is *"return ``a`` unless ``b < a``"* --
    it is not symmetric in which operand it RETURNS, only in which value.  Inside a
    ``cute.jit`` trace ``b < a`` is ``Numeric.__lt__``, which traces to an IR comparison
    (``_binary_op(operator.lt)``), so the two spellings emit the comparison with its operands
    the other way round:

        lambda a, b: min(a, b)   ->  select(peer < acc,  peer, acc)   # prefers the ACC on a tie
        lambda a, b: min(b, a)   ->  select(acc  < peer, acc,  peer)  # prefers the PEER on a tie

    For every totally-ordered value those agree, which is why the kernel is correct either
    way.  They differ only where ``<`` is not a total order -- IEEE ties (``+0.0`` vs
    ``-0.0``) and NaN, where a comparison is false in BOTH directions and ``min`` therefore
    returns whichever operand it was given first.  ⇒ the swap chooses **which lane wins a
    non-ordered comparison**, and with the accumulator second the peer wins, so a NaN
    entering from a peer lane propagates instead of being swallowed by the accumulator.

    ⚠ WHETHER THAT WAS THE AUTHOR'S INTENT IS UNRECORDED.  It arrived in
    ``a137a7614`` "[cutedsl] Refactor reductions to use helper methods (#2008)" with no
    comment, alongside ``prod``'s ``operator.mul`` (symmetric, so the order there is
    genuinely arbitrary).  ⇒ treat this as **load-bearing-by-accident**: it is the NaN
    polarity the shipped kernels have been tested at, so a refactor must PRESERVE the order
    rather than tidy it -- and that is exactly why the FIXLIST asks for this to be explained
    before the four bodies are abstracted behind one ``combine`` callable.  An abstraction
    that normalises the operand order is a silent change to NaN behaviour in a cross-lane
    reduction, which is the hardest class of thing to notice.
    """
    return cute.arch.warp_reduction(
        value,
        lambda a, b: min(b, a),
        threads_in_group=threads_in_group,
    )


@cute.jit
def _warp_reduce_prod(value: cute.Numeric, *, threads_in_group: int) -> cute.Numeric:
    return cute.arch.warp_reduction(
        value,
        operator.mul,
        threads_in_group=threads_in_group,
    )


@cute.jit
def _cute_grouped_reduce_warp_sum(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    lane_in_group = lane_expr % group_span
    lane_mod_pre = lane_in_group % pre
    selected = _warp_reduce_sum(
        value if lane_mod_pre == 0 else ident,
        threads_in_group=group_span,
    )
    for p in cutlass.range_constexpr(1, pre):
        reduced = _warp_reduce_sum(
            value if lane_mod_pre == p else ident,
            threads_in_group=group_span,
        )
        selected = reduced if lane_mod_pre == p else selected
    return selected


@cute.jit
def _cute_grouped_reduce_warp_max(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    lane_in_group = lane_expr % group_span
    lane_mod_pre = lane_in_group % pre
    selected = _warp_reduce_max(
        value if lane_mod_pre == 0 else ident,
        threads_in_group=group_span,
    )
    for p in cutlass.range_constexpr(1, pre):
        reduced = _warp_reduce_max(
            value if lane_mod_pre == p else ident,
            threads_in_group=group_span,
        )
        selected = reduced if lane_mod_pre == p else selected
    return selected


@cute.jit
def _cute_grouped_reduce_warp_min(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    lane_in_group = lane_expr % group_span
    lane_mod_pre = lane_in_group % pre
    selected = _warp_reduce_min(
        value if lane_mod_pre == 0 else ident,
        threads_in_group=group_span,
    )
    for p in cutlass.range_constexpr(1, pre):
        reduced = _warp_reduce_min(
            value if lane_mod_pre == p else ident,
            threads_in_group=group_span,
        )
        selected = reduced if lane_mod_pre == p else selected
    return selected


@cute.jit
def _cute_grouped_reduce_warp_prod(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    lane_in_group = lane_expr % group_span
    lane_mod_pre = lane_in_group % pre
    selected = _warp_reduce_prod(
        value if lane_mod_pre == 0 else ident,
        threads_in_group=group_span,
    )
    for p in cutlass.range_constexpr(1, pre):
        reduced = _warp_reduce_prod(
            value if lane_mod_pre == p else ident,
            threads_in_group=group_span,
        )
        selected = reduced if lane_mod_pre == p else selected
    return selected


_WARP_DISPATCH = {
    "sum": _cute_grouped_reduce_warp_sum,
    "max": _cute_grouped_reduce_warp_max,
    "min": _cute_grouped_reduce_warp_min,
    "prod": _cute_grouped_reduce_warp_prod,
}


def _cute_grouped_reduce_warp(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_expr: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: int,
    group_span: int,
) -> cute.Numeric:
    impl = _WARP_DISPATCH.get(reduction_type)
    if impl is None:
        raise ValueError(f"unsupported CuTe reduction type: {reduction_type!r}")
    return impl(
        input_value,
        identity,
        lane_expr,
        acc_dtype=acc_dtype,
        pre=pre,
        group_span=group_span,
    )


# ======================================================================================
# The cluster / DSMEM combine (``PORT_SPEC_layout.md`` §9, quack ``reduce.py:31-82``).
#
# A clustered reduction splits ONE row across ``cluster_n`` CTAs of a launch cluster.
# Each CTA reduces its own N-tile down to a per-row total with the ordinary intra-CTA
# combine above, then every CTA broadcasts that total to every peer with
# ``st.async.shared::cluster`` and waits on one mbarrier whose expected-transaction
# count is exactly the number of incoming stores.  After the wait every CTA holds all
# ``cluster_n`` partials for its row and folds them locally, so **all peers
# independently compute the same total** -- which is what lets each peer go on to write
# its own slice of the output with no further communication.
#
# ⚠ HOW THIS DEVIATES FROM QUACK, AND WHY.  quack's ``cluster_reduce``
# (``reduce.py:31-66``) fuses the cross-WARP and cross-CTA combines into one exchange
# over a ``(rows_per_block, (warps_per_row, cluster_n))`` buffer, then finishes with a
# warp shuffle.  helion already has a *working, tested* cross-warp combine
# (``_cute_grouped_reduce_shared_two_stage``) whose group geometry is
# ``(group_count, group_span)`` over the LINEAR thread index -- not warps -- and which
# already broadcasts its result to every thread of the group.  Composing on top of it
# means the cluster step exchanges ONE scalar per (row group, CTA) instead of one per
# (warp, CTA): fewer DSMEM stores, no dependence on ``warps_per_row``, and -- the real
# reason -- the warp-vs-group geometries never have to be reconciled.  Reconciling them
# is where an aliasing bug would live, and an aliasing bug here is SILENT (see below).
# The cost is one extra barrier's worth of latency, which is what the measurement in
# ``_redfix/LEDGER.md`` E018 prices.
#
# ⚠ ONE ENTRY POINT, op chosen by dispatch (quack's ``block_or_cluster_reduce``,
# ``reduce.py:69-82``).  The block/cluster choice is a function of ``cluster_n``, a
# compile-time constant, and the two combines differ only in their tail.  Keeping one
# entry point means the reduction BUFFER LAYOUT is written once, so a block/cluster skew
# in the indexing is not expressible.  ``cluster_n == 1`` never reaches here at all --
# the emitter simply does not request a cluster -- so the pre-existing block path stays
# byte-identical.
#
# ⚠ THE ALIASING HAZARD, and why every index is derived here rather than passed in.
# Each CTA writes its partial at column ``cta_rank`` -- an address determined by the
# **SENDER's** rank -- into EVERY peer's buffer.  So peer p's slot for sender s is the
# same slot in every CTA, and no two senders collide.  Getting this backwards is not a
# crash and not a hang: the mbarrier's byte count is still satisfied exactly, so the
# kernel completes and returns a plausible wrong number.  MEASURED while building this:
# indexing the column by the *destination* lane instead gives relerr 7.6e-1 at
# cluster_n=4 (about 1/cluster_n of the truth) with zero diagnostics.  That is the same
# failure mode ``reload_smem_is_chunk_indexed`` exists to pin on the staging buffer,
# where a chunk-indexed alias measured 1.033x -- the fastest number of the session --
# at relerr 261.6.  Hence: this helper takes ``group_count`` / ``cluster_n`` as
# constexprs and computes every address itself; the emitter cannot hand it a
# pre-flattened offset.
# ======================================================================================


@cute.jit
def _cute_cluster_mbar_alloc() -> cute.Pointer:
    """One mbarrier for one clustered reduction.

    Allocated separately from the exchange buffer because ``alloc_smem`` is a
    trace-order bump allocator whose result must be identical in every CTA of the
    cluster, and because the barrier needs its own 8-byte-aligned ``Int64`` slot.

    ⚠ ONE PER REDUCTION, never shared between two.  ``mbarrier_wait(mbar, 0)``
    waits for phase 0; a second reduction reusing the same barrier would be waiting
    on phase 1 and would either hang or fall straight through.  quack has the same
    constraint and solves it the same way, by allocating ``stage`` barriers and
    passing ``mbar_ptr + 1`` to the second reduction (``rmsnorm.py:302``).
    """
    return cute.arch.alloc_smem(cutlass.Int64, 1, 8)


@cute.jit
def _cute_cluster_mbar_init(mbar_ptr: cute.Pointer, tidx: cutlass.Int32) -> None:
    """quack ``reduction_base.py:80-97`` ``_initialize_cluster``.

    Arrival count 1 because it is the ``expect_tx`` BYTE count, not an arrival count,
    that gates the wait.  ``mbarrier_init_fence`` then ``cluster_arrive_relaxed``
    publishes the initialised barrier to the peers before any peer can store into it.
    The matching ``cluster_wait`` is emitted at the reduce site rather than here --
    quack passes it as ``row_reduce``'s ``hook_fn`` so it lands after the local
    combine and before the exchange (``rmsnorm.py:316``), which is the latest point
    that is still correct and therefore the cheapest.
    """
    if tidx < 1:
        cute.arch.mbarrier_init(mbar_ptr, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()


@cute.jit
def _cute_cluster_exchange(
    value: cute.Numeric,
    buf: cute.Tensor,
    mbar_ptr: cute.Pointer,
    group_id: cutlass.Int32,
    lane_in_group: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    group_count: cutlass.Constexpr[int],
    cluster_n: cutlass.Constexpr[int],
    row_slot: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    phase: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
) -> None:
    """Publish this CTA's per-row partial to all ``cluster_n`` peers and wait.

    quack ``reduce.py:45-59``.  One thread names the byte count of ALL incoming
    stores (``group_count * cluster_n * sizeof(acc)``); the first ``cluster_n`` lanes
    of each row group then each send that group's partial to one peer.  The store is
    itself the release -- it completes a transaction on the peer's barrier -- so the
    ``mbarrier_wait`` below is what orders every read after every incoming write, and
    no separate fence is needed or would suffice.

    ⚠ NO ``cute.arch.elect_one()`` HERE, and its absence is load-bearing.  quack
    wraps the ``expect_tx`` in ``with cute.arch.elect_one():`` (``reduce.py:46``)
    because *its* enclosing predicate is ``warp_idx == 0``, i.e. a whole warp, and
    one arrival must be issued rather than 32.  helion's enclosing predicate here
    already selects EXACTLY ONE thread (``lane_in_group == 0 and group_id == 0``), so
    ``elect_one`` would be electing from a single-thread mask -- and MEASURED, that
    HANGS: the kernel spins at 100% GPU with the barrier never flipping.  ``elect``
    is a warp-convergent vote, so entering it already divergent down to one lane does
    not terminate.  If this predicate is ever widened to a warp, ``elect_one`` must
    come back with it; the two are a matched pair, not independent.
    """
    if lane_in_group == 0 and group_id == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(
            mbar_ptr, group_count * cluster_n * (acc_dtype.width // 8)
        )
    cta_rank = cute.arch.block_idx_in_cluster()
    if lane_in_group < cluster_n:
        # Column = the SENDER's rank; peer = ``lane_in_group``.  See the hazard note
        # at the top of this section -- swapping these is silently wrong.
        store_shared_remote(
            acc_dtype(value),
            buf.iterator + cute.crd2idx((row_slot, cta_rank), buf.layout),
            mbar_ptr,
            lane_in_group,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase)


def _cute_cluster_buffer(
    acc_dtype: cutlass.Constexpr,
    group_count: int,
    cluster_n: int,
) -> cute.Tensor:
    """The ``(group_count, cluster_n)`` exchange buffer.

    ``order=(1, 0)`` puts the cluster rank fastest-varying, so one row group's
    ``cluster_n`` partials are contiguous and the fold below reads them in sequence.
    """
    return cute.make_tensor(
        cute.arch.alloc_smem(acc_dtype, group_count * cluster_n, 8),
        cute.make_ordered_layout((group_count, cluster_n), order=(1, 0)),
    )


@cute.jit
def _cute_cluster_reduce_sum(
    input_value: cute.Numeric,
    group_id: cutlass.Int32,
    lane_in_group: cutlass.Int32,
    mbar_ptr: cute.Pointer,
    *,
    acc_dtype: cutlass.Constexpr,
    group_count: cutlass.Constexpr[int],
    cluster_n: cutlass.Constexpr[int],
    row_slot: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    phase: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    buf_rows: cutlass.Constexpr[int] = 0,
) -> cute.Numeric:
    buf = _cute_cluster_buffer(acc_dtype, buf_rows or group_count, cluster_n)
    _cute_cluster_exchange(
        acc_dtype(input_value),
        buf,
        mbar_ptr,
        group_id,
        lane_in_group,
        acc_dtype=acc_dtype,
        group_count=group_count,
        cluster_n=cluster_n,
        row_slot=row_slot,
        phase=phase,
    )
    # Every thread folds all ``cluster_n`` partials of its own row group.  Redundant
    # across threads on purpose: it makes the result already broadcast, matching what
    # the intra-CTA combine returns, so the caller needs no further shuffle.
    acc = buf[row_slot, 0]
    for i in cutlass.range_constexpr(1, cluster_n):
        acc = acc + buf[row_slot, i]
    return acc


@cute.jit
def _cute_cluster_reduce_max(
    input_value: cute.Numeric,
    group_id: cutlass.Int32,
    lane_in_group: cutlass.Int32,
    mbar_ptr: cute.Pointer,
    *,
    acc_dtype: cutlass.Constexpr,
    group_count: cutlass.Constexpr[int],
    cluster_n: cutlass.Constexpr[int],
    row_slot: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    phase: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    buf_rows: cutlass.Constexpr[int] = 0,
) -> cute.Numeric:
    buf = _cute_cluster_buffer(acc_dtype, buf_rows or group_count, cluster_n)
    _cute_cluster_exchange(
        acc_dtype(input_value),
        buf,
        mbar_ptr,
        group_id,
        lane_in_group,
        acc_dtype=acc_dtype,
        group_count=group_count,
        cluster_n=cluster_n,
        row_slot=row_slot,
        phase=phase,
    )
    acc = buf[row_slot, 0]
    for i in cutlass.range_constexpr(1, cluster_n):
        candidate = buf[row_slot, i]
        acc = max(acc, candidate)
    return acc


@cute.jit
def _cute_cluster_reduce_min(
    input_value: cute.Numeric,
    group_id: cutlass.Int32,
    lane_in_group: cutlass.Int32,
    mbar_ptr: cute.Pointer,
    *,
    acc_dtype: cutlass.Constexpr,
    group_count: cutlass.Constexpr[int],
    cluster_n: cutlass.Constexpr[int],
    row_slot: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    phase: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    buf_rows: cutlass.Constexpr[int] = 0,
) -> cute.Numeric:
    buf = _cute_cluster_buffer(acc_dtype, buf_rows or group_count, cluster_n)
    _cute_cluster_exchange(
        acc_dtype(input_value),
        buf,
        mbar_ptr,
        group_id,
        lane_in_group,
        acc_dtype=acc_dtype,
        group_count=group_count,
        cluster_n=cluster_n,
        row_slot=row_slot,
        phase=phase,
    )
    acc = buf[row_slot, 0]
    for i in cutlass.range_constexpr(1, cluster_n):
        candidate = buf[row_slot, i]
        acc = min(acc, candidate)
    return acc


@cute.jit
def _cute_cluster_reduce_prod(
    input_value: cute.Numeric,
    group_id: cutlass.Int32,
    lane_in_group: cutlass.Int32,
    mbar_ptr: cute.Pointer,
    *,
    acc_dtype: cutlass.Constexpr,
    group_count: cutlass.Constexpr[int],
    cluster_n: cutlass.Constexpr[int],
    row_slot: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    phase: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    buf_rows: cutlass.Constexpr[int] = 0,
) -> cute.Numeric:
    buf = _cute_cluster_buffer(acc_dtype, buf_rows or group_count, cluster_n)
    _cute_cluster_exchange(
        acc_dtype(input_value),
        buf,
        mbar_ptr,
        group_id,
        lane_in_group,
        acc_dtype=acc_dtype,
        group_count=group_count,
        cluster_n=cluster_n,
        row_slot=row_slot,
        phase=phase,
    )
    acc = buf[row_slot, 0]
    for i in cutlass.range_constexpr(1, cluster_n):
        acc = acc * buf[row_slot, i]
    return acc


_CLUSTER_DISPATCH = {
    "sum": _cute_cluster_reduce_sum,
    "max": _cute_cluster_reduce_max,
    "min": _cute_cluster_reduce_min,
    "prod": _cute_cluster_reduce_prod,
}


# ``store_shared_remote``'s inline asm has 32-bit and 64-bit forms only
# (``cluster_helpers.py:25`` ``_ASYNC_STORE_SUFFIX`` = f32/s32/s64), so a NARROWER
# accumulator cannot cross a cluster.  Widen it for the exchange.
#
# WHY THIS IS NEEDED AT ALL, and why the obvious place is the wrong one (LEDGER E039).
# ``reduction_acc_dtype`` widens integer accumulators to int64 for ``sum``/``prod``
# (the class-5 ABI) but deliberately leaves ``max``/``min`` at the input width,
# because a selecting reduction's running value is always one of its inputs and so
# cannot leave the input's range.  That is correct *semantics* -- and it means an
# int8/int16 ``amax``/``amin`` arrives here with an 8/16-bit accumulator and trips
# ``store_shared_remote``'s assert.  ``cluster_helpers.py:20-24`` states the
# assumption that broke: "integer accumulators widen to int64", true for sum, false
# for max/min.
#
# Fixed HERE rather than in ``reduction_acc_dtype`` because the constraint is about
# the TRANSPORT, not the arithmetic: widening the semantic accumulator would change
# the reduction's result dtype and defeat the reason max/min are exempt. Widening
# only the exchange is value-preserving for a selecting reduction by construction
# (every value is one of the inputs, so it fits the narrow range and round-trips),
# and the caller keeps its own narrow ``dtype`` for everything downstream.
#
# MEASURED (adversary half 1): without this, ``torch.amax``/``amin`` on an int8 or
# int16 input raises ``AssertionError: store_shared_remote val must be Float32,
# Int32 or Int64, got Int16`` at every N>=32768 -- reachable from a plain
# ``kernel(x)`` call with NO user config, because ``default_config()`` supplies
# ``cute_cluster_n`` from quack's ladder and LEDGER E026 lowered the cluster's
# threshold to N>=32768. 8 of 16 dtype x op combinations crashed on the branch;
# 16/16 were fine on upstream base (where the knob does not exist).
_CLUSTER_EXCHANGE_MIN_BITS = 32


def _cluster_exchange_dtype(acc_dtype: cutlass.Constexpr) -> cutlass.Constexpr:
    width = getattr(acc_dtype, "width", None)
    if width is not None and width < _CLUSTER_EXCHANGE_MIN_BITS:
        return cutlass.Int32 if not acc_dtype.is_float else cutlass.Float32
    return acc_dtype


def _cute_cluster_reduce(
    input_value: cute.Numeric,
    reduction_type: str,
    group_id: cutlass.Int32,
    lane_in_group: cutlass.Int32,
    mbar_ptr: cute.Pointer,
    *,
    acc_dtype: cutlass.Constexpr,
    group_count: int,
    cluster_n: int,
    row_slot: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    phase: cutlass.Int32 = _CLUSTER_NO_SERIAL_ROW,
    buf_rows: int = 0,
) -> cute.Numeric:
    """The cross-CTA row combine.  One entry point; see the section header."""
    impl = _CLUSTER_DISPATCH.get(reduction_type)
    if impl is None:
        raise ValueError(f"unsupported CuTe cluster reduction type: {reduction_type!r}")
    return impl(
        input_value,
        group_id,
        lane_in_group,
        mbar_ptr,
        acc_dtype=_cluster_exchange_dtype(acc_dtype),
        group_count=group_count,
        cluster_n=cluster_n,
        row_slot=row_slot,
        phase=phase,
        buf_rows=buf_rows,
    )


@cute.jit
def _cute_grouped_reduce_shared_two_stage_sum(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    warps_per_group = group_span // 32
    partials_size = group_count * pre * warps_per_group
    results_size = group_count * pre
    smem_size = partials_size + results_size
    smem_ptr = cute.arch.alloc_smem(acc_dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_id = lane_var // group_span
    lane_in_warp = lane_var % 32
    warp_in_group = lane_in_group_var // 32
    partials_base = group_id * (pre * warps_per_group)
    results_base = partials_size + group_id * pre

    for p in cutlass.range_constexpr(pre):
        masked_input = value if lane_mod_pre_var == p else ident
        warp_partial = _warp_reduce_sum(masked_input, threads_in_group=32)
        partial_idx = partials_base + p * warps_per_group + warp_in_group
        if lane_in_warp == 0:
            smem[partial_idx] = warp_partial
        cute.arch.sync_threads()

        if warp_in_group == 0:
            stage2_input = (
                smem[partials_base + p * warps_per_group + lane_in_warp]
                if lane_in_warp < warps_per_group
                else ident
            )
            group_result = _warp_reduce_sum(stage2_input, threads_in_group=32)
            if lane_in_warp == 0:
                smem[results_base + p] = group_result
        cute.arch.sync_threads()

    return smem[results_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_two_stage_max(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    warps_per_group = group_span // 32
    partials_size = group_count * pre * warps_per_group
    results_size = group_count * pre
    smem_size = partials_size + results_size
    smem_ptr = cute.arch.alloc_smem(acc_dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_id = lane_var // group_span
    lane_in_warp = lane_var % 32
    warp_in_group = lane_in_group_var // 32
    partials_base = group_id * (pre * warps_per_group)
    results_base = partials_size + group_id * pre

    for p in cutlass.range_constexpr(pre):
        masked_input = value if lane_mod_pre_var == p else ident
        warp_partial = _warp_reduce_max(masked_input, threads_in_group=32)
        partial_idx = partials_base + p * warps_per_group + warp_in_group
        if lane_in_warp == 0:
            smem[partial_idx] = warp_partial
        cute.arch.sync_threads()

        if warp_in_group == 0:
            stage2_input = (
                smem[partials_base + p * warps_per_group + lane_in_warp]
                if lane_in_warp < warps_per_group
                else ident
            )
            group_result = _warp_reduce_max(stage2_input, threads_in_group=32)
            if lane_in_warp == 0:
                smem[results_base + p] = group_result
        cute.arch.sync_threads()

    return smem[results_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_two_stage_min(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    warps_per_group = group_span // 32
    partials_size = group_count * pre * warps_per_group
    results_size = group_count * pre
    smem_size = partials_size + results_size
    smem_ptr = cute.arch.alloc_smem(acc_dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_id = lane_var // group_span
    lane_in_warp = lane_var % 32
    warp_in_group = lane_in_group_var // 32
    partials_base = group_id * (pre * warps_per_group)
    results_base = partials_size + group_id * pre

    for p in cutlass.range_constexpr(pre):
        masked_input = value if lane_mod_pre_var == p else ident
        warp_partial = _warp_reduce_min(masked_input, threads_in_group=32)
        partial_idx = partials_base + p * warps_per_group + warp_in_group
        if lane_in_warp == 0:
            smem[partial_idx] = warp_partial
        cute.arch.sync_threads()

        if warp_in_group == 0:
            stage2_input = (
                smem[partials_base + p * warps_per_group + lane_in_warp]
                if lane_in_warp < warps_per_group
                else ident
            )
            group_result = _warp_reduce_min(stage2_input, threads_in_group=32)
            if lane_in_warp == 0:
                smem[results_base + p] = group_result
        cute.arch.sync_threads()

    return smem[results_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_two_stage_prod(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    warps_per_group = group_span // 32
    partials_size = group_count * pre * warps_per_group
    results_size = group_count * pre
    smem_size = partials_size + results_size
    smem_ptr = cute.arch.alloc_smem(acc_dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_id = lane_var // group_span
    lane_in_warp = lane_var % 32
    warp_in_group = lane_in_group_var // 32
    partials_base = group_id * (pre * warps_per_group)
    results_base = partials_size + group_id * pre

    for p in cutlass.range_constexpr(pre):
        masked_input = value if lane_mod_pre_var == p else ident
        warp_partial = _warp_reduce_prod(masked_input, threads_in_group=32)
        partial_idx = partials_base + p * warps_per_group + warp_in_group
        if lane_in_warp == 0:
            smem[partial_idx] = warp_partial
        cute.arch.sync_threads()

        if warp_in_group == 0:
            stage2_input = (
                smem[partials_base + p * warps_per_group + lane_in_warp]
                if lane_in_warp < warps_per_group
                else ident
            )
            group_result = _warp_reduce_prod(stage2_input, threads_in_group=32)
            if lane_in_warp == 0:
                smem[results_base + p] = group_result
        cute.arch.sync_threads()

    return smem[results_base + lane_mod_pre_var]


_TWO_STAGE_DISPATCH = {
    "sum": _cute_grouped_reduce_shared_two_stage_sum,
    "max": _cute_grouped_reduce_shared_two_stage_max,
    "min": _cute_grouped_reduce_shared_two_stage_min,
    "prod": _cute_grouped_reduce_shared_two_stage_prod,
}


def _cute_grouped_reduce_shared_two_stage(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: int,
    group_span: int,
    group_count: int,
) -> cute.Numeric:
    impl = _TWO_STAGE_DISPATCH.get(reduction_type)
    if impl is None:
        raise ValueError(f"unsupported CuTe reduction type: {reduction_type!r}")
    return impl(
        input_value,
        identity,
        lane_var,
        lane_in_group_var,
        lane_mod_pre_var,
        acc_dtype=acc_dtype,
        pre=pre,
        group_span=group_span,
        group_count=group_count,
    )


@cute.jit
def _cute_grouped_reduce_shared_tree_sum(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    smem_size = num_threads + group_count * pre
    smem_ptr = cute.arch.alloc_smem(acc_dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_base = lane_var - lane_in_group_var
    group_id = lane_var // group_span
    result_base = num_threads + group_id * pre

    for p in cutlass.range_constexpr(pre):
        smem[lane_var] = value if lane_mod_pre_var == p else ident
        cute.arch.sync_threads()
        stride = 1
        while stride < group_span:
            if (
                lane_in_group_var % (stride * 2) == 0
                and lane_in_group_var + stride < group_span
            ):
                smem[lane_var] = (
                    smem[lane_var] + smem[group_base + lane_in_group_var + stride]
                )
            cute.arch.sync_threads()
            stride *= 2

        if lane_in_group_var == 0:
            smem[result_base + p] = smem[lane_var]
        cute.arch.sync_threads()

    return smem[result_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_tree_max(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    smem_size = num_threads + group_count * pre
    smem_ptr = cute.arch.alloc_smem(acc_dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_base = lane_var - lane_in_group_var
    group_id = lane_var // group_span
    result_base = num_threads + group_id * pre

    for p in cutlass.range_constexpr(pre):
        smem[lane_var] = value if lane_mod_pre_var == p else ident
        cute.arch.sync_threads()
        stride = 1
        while stride < group_span:
            if (
                lane_in_group_var % (stride * 2) == 0
                and lane_in_group_var + stride < group_span
            ):
                lhs = smem[lane_var]
                rhs = smem[group_base + lane_in_group_var + stride]
                smem[lane_var] = max(rhs, lhs)
            cute.arch.sync_threads()
            stride *= 2

        if lane_in_group_var == 0:
            smem[result_base + p] = smem[lane_var]
        cute.arch.sync_threads()

    return smem[result_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_tree_min(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    smem_size = num_threads + group_count * pre
    smem_ptr = cute.arch.alloc_smem(acc_dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_base = lane_var - lane_in_group_var
    group_id = lane_var // group_span
    result_base = num_threads + group_id * pre

    for p in cutlass.range_constexpr(pre):
        smem[lane_var] = value if lane_mod_pre_var == p else ident
        cute.arch.sync_threads()
        stride = 1
        while stride < group_span:
            if (
                lane_in_group_var % (stride * 2) == 0
                and lane_in_group_var + stride < group_span
            ):
                lhs = smem[lane_var]
                rhs = smem[group_base + lane_in_group_var + stride]
                smem[lane_var] = min(rhs, lhs)
            cute.arch.sync_threads()
            stride *= 2

        if lane_in_group_var == 0:
            smem[result_base + p] = smem[lane_var]
        cute.arch.sync_threads()

    return smem[result_base + lane_mod_pre_var]


@cute.jit
def _cute_grouped_reduce_shared_tree_prod(
    input_value: cute.Numeric,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: cutlass.Constexpr[int],
    group_span: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    group_count: cutlass.Constexpr[int],
) -> cute.Numeric:
    value = acc_dtype(input_value)
    ident = acc_dtype(identity)
    smem_size = num_threads + group_count * pre
    smem_ptr = cute.arch.alloc_smem(acc_dtype, smem_size)
    smem = cute.make_tensor(smem_ptr, (smem_size,))
    group_base = lane_var - lane_in_group_var
    group_id = lane_var // group_span
    result_base = num_threads + group_id * pre

    for p in cutlass.range_constexpr(pre):
        smem[lane_var] = value if lane_mod_pre_var == p else ident
        cute.arch.sync_threads()
        stride = 1
        while stride < group_span:
            if (
                lane_in_group_var % (stride * 2) == 0
                and lane_in_group_var + stride < group_span
            ):
                smem[lane_var] = (
                    smem[lane_var] * smem[group_base + lane_in_group_var + stride]
                )
            cute.arch.sync_threads()
            stride *= 2

        if lane_in_group_var == 0:
            smem[result_base + p] = smem[lane_var]
        cute.arch.sync_threads()

    return smem[result_base + lane_mod_pre_var]


_TREE_DISPATCH = {
    "sum": _cute_grouped_reduce_shared_tree_sum,
    "max": _cute_grouped_reduce_shared_tree_max,
    "min": _cute_grouped_reduce_shared_tree_min,
    "prod": _cute_grouped_reduce_shared_tree_prod,
}


def _cute_grouped_reduce_shared_tree(
    input_value: cute.Numeric,
    reduction_type: str,
    identity: cute.Numeric,
    lane_var: cutlass.Int32,
    lane_in_group_var: cutlass.Int32,
    lane_mod_pre_var: cutlass.Int32,
    *,
    acc_dtype: cutlass.Constexpr,
    pre: int,
    group_span: int,
    num_threads: int,
    group_count: int,
) -> cute.Numeric:
    impl = _TREE_DISPATCH.get(reduction_type)
    if impl is None:
        raise ValueError(f"unsupported CuTe reduction type: {reduction_type!r}")
    return impl(
        input_value,
        identity,
        lane_var,
        lane_in_group_var,
        lane_mod_pre_var,
        acc_dtype=acc_dtype,
        pre=pre,
        group_span=group_span,
        num_threads=num_threads,
        group_count=group_count,
    )


@cute.jit
def _cute_argmax_index_impl(
    smem: cute.Tensor,
    valid_smem: cute.Tensor,
    start_idx: cutlass.Int32,
    stride: cutlass.Int32,
    *,
    extent: cutlass.Constexpr[int],
) -> cutlass.Int64:
    best_index = cutlass.Int64(0)
    best_value = smem[start_idx]
    best_valid = valid_smem[start_idx]
    for candidate_index in cutlass.range_constexpr(1, extent):
        candidate_offset = start_idx + stride * candidate_index
        candidate = smem[candidate_offset]
        candidate_valid = valid_smem[candidate_offset]
        better = candidate_valid != cutlass.Int32(0) and (
            best_valid == cutlass.Int32(0)
            or (
                best_valid != cutlass.Int32(0)
                and (
                    candidate > best_value
                    or (
                        candidate == best_value
                        and cutlass.Int64(candidate_index) < best_index
                    )
                )
            )
        )
        if better:
            best_value = candidate
            best_valid = candidate_valid
            best_index = cutlass.Int64(candidate_index)
    return best_index


@cute.jit
def _cute_argmin_index_impl(
    smem: cute.Tensor,
    valid_smem: cute.Tensor,
    start_idx: cutlass.Int32,
    stride: cutlass.Int32,
    *,
    extent: cutlass.Constexpr[int],
) -> cutlass.Int64:
    best_index = cutlass.Int64(0)
    best_value = smem[start_idx]
    best_valid = valid_smem[start_idx]
    for candidate_index in cutlass.range_constexpr(1, extent):
        candidate_offset = start_idx + stride * candidate_index
        candidate = smem[candidate_offset]
        candidate_valid = valid_smem[candidate_offset]
        better = candidate_valid != cutlass.Int32(0) and (
            best_valid == cutlass.Int32(0)
            or (
                best_valid != cutlass.Int32(0)
                and (
                    candidate < best_value
                    or (
                        candidate == best_value
                        and cutlass.Int64(candidate_index) < best_index
                    )
                )
            )
        )
        if better:
            best_value = candidate
            best_valid = candidate_valid
            best_index = cutlass.Int64(candidate_index)
    return best_index


_ARGREDUCE_DISPATCH = {
    "argmax": _cute_argmax_index_impl,
    "argmin": _cute_argmin_index_impl,
}


def _cute_argreduce_index(
    smem: cute.Tensor,
    valid_smem: cute.Tensor,
    start_idx: cutlass.Int32,
    stride: cutlass.Int32,
    *,
    extent: int,
    reduction_type: str,
) -> cutlass.Int64:
    impl = _ARGREDUCE_DISPATCH.get(reduction_type)
    if impl is None:
        raise ValueError(f"unsupported CuTe argreduce type: {reduction_type!r}")
    return impl(smem, valid_smem, start_idx, stride, extent=extent)
