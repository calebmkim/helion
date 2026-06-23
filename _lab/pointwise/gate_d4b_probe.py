"""Gate D 4th-pass: does a REAL activation that references its input multiple times in-expression
(SiLU x*sigmoid(x), GELU, x*x relu_squared written inline) emit multiple HBM-load facts and thus
double-count bytes_per_elem vs the true 1-read traffic? Compile-only."""
from __future__ import annotations
import torch
import helion
import helion.language as hl
from helion._compiler.autotuner_heuristics import compiler_seed_configs

assert helion.__file__.startswith(
    "/home/calebkim/helion-new-heuristics/helion-pointwise/"
), helion.__file__


@helion.kernel()
def silu_inline(x: torch.Tensor) -> torch.Tensor:
    """SiLU/swish: out = x * sigmoid(x). x referenced twice in the SAME expression."""
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        out[t] = x[t] * torch.sigmoid(x[t])
    return out


@helion.kernel()
def silu_bound(x: torch.Tensor) -> torch.Tensor:
    """SiLU written the idiomatic way: bind v = x[t] once, reuse."""
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        v = x[t]
        out[t] = v * torch.sigmoid(v)
    return out


@helion.kernel()
def gelu_inline(x: torch.Tensor) -> torch.Tensor:
    """tanh-approx GELU: 0.5*x*(1+tanh(0.797885*(x+0.044715*x^3))). x referenced many times."""
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        out[t] = 0.5 * x[t] * (1.0 + torch.tanh(0.7978845608 * (x[t] + 0.044715 * x[t] * x[t] * x[t])))
    return out


def dump(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    pf = spec.pointwise_facts
    mfs = spec.memory_op_facts
    print("=== %s ===" % name)
    f = pf[0]
    nload_x = sum(1 for m in mfs if m.kind == "load")
    print("  bytes/elem=%d n_load=%d n_store=%d  (true HBM: 1 read + 1 write = 4)" % (
        f.bytes_per_elem, f.n_load, f.n_store))
    for m in mfs:
        full = m.dtype is not None and m.accessed_numel >= f.total_numel
        print("    %-5s %-8s acc=%s full=%s" % (m.kind, m.tensor_name, m.accessed_numel, full))
    s = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    print("  seed=%s" % (dict(s[0].config).get("block_sizes") if s else None))


def main():
    M, N = 4096, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dump("silu_inline x*sigmoid(x)", silu_inline, (x,))
    dump("silu_bound v*sigmoid(v)", silu_bound, (x,))
    dump("gelu_inline (x used 4x)", gelu_inline, (x,))


if __name__ == "__main__":
    main()
