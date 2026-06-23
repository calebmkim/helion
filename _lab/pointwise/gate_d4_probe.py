"""Gate D (4th pass) faithfulness probe for PointwiseElementwiseFact.bytes_per_elem.

bytes_per_elem = SUM over memory_op_facts with accessed_numel >= total_numel of dtype.itemsize.
The property it claims to model: per-element HBM byte traffic (reads + writes) of the kernel.

We probe REALISTIC elementwise patterns where accessed_numel >= total_numel could DIVERGE
from true per-element HBM traffic and mis-size the seed tile. Compile-only (bind + walk facts),
no GPU execution. Fake tensors on cuda; only tracing runs.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl
from helion._compiler.autotuner_heuristics import compiler_seed_configs

assert helion.__file__.startswith(
    "/home/calebkim/helion-new-heuristics/helion-pointwise/"
), helion.__file__


# ---- Probe 1: same tensor read MULTIPLE times (relu_squared x*x). True HBM traffic = 1 read
# of x + 1 store. If the walker emits 2 loads of x, bytes_per_elem double-counts x. ----
@helion.kernel()
def relu_squared(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        v = torch.relu(x[t])
        out[t] = v * v  # one tile load of x reused; squares the loaded value
    return out


@helion.kernel()
def x_times_x(x: torch.Tensor) -> torch.Tensor:
    """Reads x[t] TWICE syntactically (x[t] * x[t]) -> does codegen emit 2 loads?"""
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        out[t] = x[t] * x[t]
    return out


# ---- Probe 2: in-place residual add (read x, read y, write x). True traffic = 2 reads + 1
# write = 3 ops of x/y/x. ----
@helion.kernel()
def residual_add_inplace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    for t in hl.tile(x.size()):
        x[t] = x[t] + y[t]
    return x


# ---- Probe 3: oversized operand where args[0] is a HOST-SLICED narrow view. Kernel tiles the
# narrow [M,N]; the underlying buffer is [M,2N]. Does the fake see the full buffer (M*2N,
# counted via >=) or the narrowed shape (M*N, still counted)? Either way it should count. ----
@helion.kernel()
def narrowed_add(x: torch.Tensor, ynarrow: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        out[t] = x[t] + ynarrow[t]
    return out


# ---- Probe 4: dtype-mixed up-cast. x is bf16, scale is fp32 FULL-extent. True traffic =
# 2 (bf16) + 4 (fp32 read) + 2 (bf16 store) = 8. ----
@helion.kernel()
def mixed_dtype_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        out[t] = (x[t].to(torch.float32) * scale[t]).to(x.dtype)
    return out


# ---- Probe 5: per-row broadcast [M,1] (full-rank broadcast). accessed_numel = M (stride-0 on
# dim1). total_numel = M*N. Must be EXCLUDED (amortized). ----
@helion.kernel()
def per_row_scale(x: torch.Tensor, rowscale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * rowscale[tm, 0]
    return out


def dump(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    pf = spec.pointwise_facts
    mfs = spec.memory_op_facts
    print("=== %s ===" % name)
    if not pf:
        print("  NO pointwise fact (routed elsewhere: red=%d mm=%d acc=%d)" % (
            len(spec.reduction_facts), len(spec.matmul_facts), len(spec.accumulator_facts)))
        return
    f = pf[0]
    print("  total_numel=%d bytes/elem=%d n_load=%d n_store=%d n_block=%d hints=%s" % (
        f.total_numel, f.bytes_per_elem, f.n_load, f.n_store, f.n_block_dims, f.block_size_hints))
    for m in mfs:
        full = m.dtype is not None and m.accessed_numel >= f.total_numel
        print("    %-5s %-10s dtype=%s acc_numel=%s full=%s itemsize=%s" % (
            m.kind, m.tensor_name, m.dtype, m.accessed_numel, full,
            m.dtype.itemsize if m.dtype is not None else None))
    s = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    print("  seed_block_sizes=%s" % (dict(s[0].config).get("block_sizes") if s else None))


def main():
    M, N = 4096, 4096
    x2 = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    x2b = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    y2 = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(M, N, device="cuda", dtype=torch.float32)
    rowscale = torch.randn(M, 1, device="cuda", dtype=torch.bfloat16)
    big = torch.randn(M, 2 * N, device="cuda", dtype=torch.bfloat16)
    ynarrow = big[:, :N]  # host-sliced narrow view -> contiguous? strided? shape [M,N]

    dump("relu_squared (x reused)", relu_squared, (x2,))
    dump("x_times_x (x[t]*x[t])", x_times_x, (x2b,))
    dump("residual_add_inplace", residual_add_inplace, (x2, y2))
    dump("narrowed_add (y=big[:, :N])", narrowed_add, (x2, ynarrow))
    dump("mixed_dtype_scale (bf16*fp32)", mixed_dtype_scale, (x2, scale))
    dump("per_row_scale ([M,1] bcast)", per_row_scale, (x2, rowscale))


if __name__ == "__main__":
    main()
