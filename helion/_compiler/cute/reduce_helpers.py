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
