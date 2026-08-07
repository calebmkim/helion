from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from cutlass import Float32
from cutlass import Int32
from cutlass import Int64
from cutlass._mlir.dialects import llvm
import cutlass.cute as cute
from cutlass.cutlass_dsl import T
from cutlass.cutlass_dsl import dsl_user_op

if TYPE_CHECKING:
    from cutlass._mlir import ir

# ``st.async.shared::cluster`` PTX suffix and operand constraint per scalar type.
# quack ``utils.py:65-66``.  ``Int64`` is included deliberately: class 5 made
# integer accumulators widen to int64 (``acc_dtype`` is an explicit required ABI
# argument on every combine entry point), so an int64 accumulator reaching a
# cluster reduce is reachable, and the packed ``(max, sum)`` form the online
# softmax path uses is an Int64 as well.  The x4 helper below handles only
# Float32/Int32 because a v4 store of two int64s would be 32 bytes.
_ASYNC_STORE_SUFFIX: dict[object, str] = {Float32: "f32", Int32: "s32", Int64: "s64"}
_ASYNC_STORE_CONSTRAINT: dict[object, str] = {Float32: "f", Int32: "r", Int64: "l"}


@dsl_user_op
def _set_block_rank(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> Int32:
    """Map an SMEM pointer to the address at another CTA rank in the cluster."""

    smem_ptr_i32 = cast("Any", smem_ptr).toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: Float32 | Int32 | Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Store ONE scalar into a peer CTA's SMEM and complete its async tx.

    Port of quack ``utils.py:47-75``.  This is the primitive
    ``cluster_reduce`` (``quack/reduce.py:52-58``) needs and the reason
    :func:`store_shared_remote_x4` is **not** substitutable for it: the x4 form
    writes four values to ONE ``smem_ptr`` as a packed ``.v4`` store, whereas a
    cluster reduce writes one value to ``cluster_n`` DIFFERENT peers at an
    address that is a function of the *sender's* rank, with the peer varying by
    lane.  There is no lane-varying-peer dimension in the packed form.

    The store is a *release* on the peer's mbarrier: it decrements the peer's
    expected-transaction count, so the peer's :func:`mbarrier_wait` is what
    orders the read against every incoming store.  No extra fence is required
    (and none would be sufficient -- the completion is what carries the data).
    """

    remote_smem_ptr_i32 = _set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = _set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    assert isinstance(val, (Float32, Int32, Int64)), (
        f"store_shared_remote val must be Float32, Int32 or Int64, got {type(val)}"
    )
    dtype = type(val)
    suffix = _ASYNC_STORE_SUFFIX[dtype]
    constraint = _ASYNC_STORE_CONSTRAINT[dtype]
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            cast("Any", val).ir_value(loc=loc, ip=ip),
            remote_mbar_ptr_i32,
        ],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} "
        "[$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
    )


@dsl_user_op
def store_shared_remote_x4(
    val0: Float32 | Int32,
    val1: Float32 | Int32,
    val2: Float32 | Int32,
    val3: Float32 | Int32,
    *,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    loc: ir.Location | None = None,
    ip: ir.InsertionPoint | None = None,
) -> None:
    """Store four scalars into another CTA's SMEM and complete the async tx."""

    remote_smem_ptr_i32 = _set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = _set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    assert isinstance(val0, (Float32, Int32)), "val must be Float32 or Int32"
    dtype = Float32 if isinstance(val0, Float32) else Int32
    suffix = {Float32: "f32", Int32: "s32"}[dtype]
    constraint = {Float32: "f", Int32: "r"}[dtype]
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            remote_mbar_ptr_i32,
            dtype(val0).ir_value(loc=loc, ip=ip),
            dtype(val1).ir_value(loc=loc, ip=ip),
            dtype(val2).ir_value(loc=loc, ip=ip),
            dtype(val3).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        f".reg .v4 .{suffix} abcd;\n\t"
        f"mov.{suffix} abcd.x, $2;\n\t"
        f"mov.{suffix} abcd.y, $3;\n\t"
        f"mov.{suffix} abcd.z, $4;\n\t"
        f"mov.{suffix} abcd.w, $5;\n\t"
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.{suffix} [$0], abcd, [$1];\n\t"
        "}\n",
        f"r,r,{constraint},{constraint},{constraint},{constraint}",
        has_side_effects=True,
        is_align_stack=False,
    )
