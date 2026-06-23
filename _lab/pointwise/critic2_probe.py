"""Completeness-critic 2nd pass: probe HIGH-value gaps NOT covered by prior Gate-D passes.
Compile-only (bind + walk facts + seed). No GPU timing. Pin one GPU for fake-tensor context.
Fact is now 2-field (total_numel, bytes_per_elem) post refactor-critic trim.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl
from helion._compiler.autotuner_heuristics import compiler_seed_configs

assert helion.__file__.startswith(
    "/home/calebkim/helion-new-heuristics/helion-pointwise/"
), helion.__file__


# G1: double-load idiom x[t]*x[t] -- common. Does walker emit 2 loads of x -> double-count?
@helion.kernel()
def x_times_x(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        out[t] = x[t] * x[t]
    return out


# G2: many-input fused elementwise (5 distinct full-extent loads + 1 store). traffic-6.
@helion.kernel()
def add5(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    for t in hl.tile(a.size()):
        out[t] = a[t] + b[t] + c[t] + d[t] + e[t]
    return out


# G3: rank-3 pointwise (3 block dims). Does the distributor produce [1,1,inner]?
@helion.kernel()
def add3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i, j, k in hl.tile(x.size()):
        out[i, j, k] = x[i, j, k] + y[i, j, k]
    return out


# G4: multi-output pointwise (1 load, 2 distinct stores) -- e.g. sign+abs. traffic = 1+2 = 3.
@helion.kernel()
def two_out(x: torch.Tensor):
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    for t in hl.tile(x.size()):
        v = x[t]
        a[t] = torch.relu(v)
        b[t] = -torch.relu(-v)
    return a, b


def dump(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    pf = spec.pointwise_facts
    mfs = spec.memory_op_facts
    print("=== %s ===" % name)
    if not pf:
        print("  NO pointwise fact (red=%d mm=%d acc=%d)" % (
            len(spec.reduction_facts), len(spec.matmul_facts), len(spec.accumulator_facts)))
        return
    f = pf[0]
    print("  total_numel=%d bytes/elem=%d  n_block=%d" % (
        f.total_numel, f.bytes_per_elem, len(spec.block_sizes)))
    for m in mfs:
        full = m.dtype is not None and m.accessed_numel >= f.total_numel
        print("    %-5s name=%-6s dtype=%s acc_numel=%s full=%s isize=%s" % (
            m.kind, m.tensor_name, m.dtype, m.accessed_numel, full,
            m.dtype.itemsize if m.dtype is not None else None))
    s = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    bs = dict(s[0].config).get("block_sizes") if s else None
    print("  seed block_sizes=%s  heuristic=%s" % (bs, spec.autotuner_heuristics))


def main():
    M, N = 4096, 4096
    bf = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, N, **bf)
    a, b, c, d, e = (torch.randn(M, N, **bf) for _ in range(5))
    x3 = torch.randn(64, 256, 1024, **bf)
    y3 = torch.randn(64, 256, 1024, **bf)

    dump("G1 x_times_x (x[t]*x[t])", x_times_x, (x,))
    dump("G2 add5 (5 loads+1 store)", add5, (a, b, c, d, e))
    dump("G3 add3d (rank-3)", add3d, (x3, y3))
    dump("G4 two_out (1 load, 2 stores)", two_out, (x,))


if __name__ == "__main__":
    main()
