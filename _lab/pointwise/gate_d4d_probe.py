"""Gate D 4th-pass: the DANGEROUS direction - does the fact ever UNDER-count (exclude a truly
full-extent op -> too-large tile -> spill)? Test (a) transposed/non-contiguous FULL operand
(must be counted), (b) .expand() broadcast (must be excluded), (c) 1-D flatten of a 2-D tensor.
Compile-only."""
from __future__ import annotations
import torch
import helion
import helion.language as hl
from helion._compiler.autotuner_heuristics import compiler_seed_configs

assert helion.__file__.startswith(
    "/home/calebkim/helion-new-heuristics/helion-pointwise/"
), helion.__file__


@helion.kernel()
def transposed_add(x: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
    """yt is a transposed (non-contiguous, full M*N) view. Real traffic = M*N reads (strided)."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + yt[tm, tn]
    return out


@helion.kernel()
def expand_add(x: torch.Tensor, colvec: torch.Tensor) -> torch.Tensor:
    """colvec[1,N].expand(M,N): full SHAPE [M,N] but stride-0 dim0 -> only N distinct elems.
    Must be EXCLUDED (amortized)."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + colvec[tm, tn]
    return out


def dump(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    f = spec.pointwise_facts[0]
    s = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    print("=== %s ===  bytes/elem=%d seed=%s" % (
        name, f.bytes_per_elem, dict(s[0].config).get("block_sizes") if s else None))
    for m in spec.memory_op_facts:
        full = m.dtype is not None and m.accessed_numel >= f.total_numel
        print("    %-5s %-8s acc=%-10s full=%s" % (m.kind, m.tensor_name, m.accessed_numel, full))


def main():
    M, N = 4096, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(N, M, device="cuda", dtype=torch.bfloat16)
    yt = y.t()  # transposed view: shape [M,N], non-contig, strides both nonzero -> full M*N
    colvec = torch.randn(1, N, device="cuda", dtype=torch.bfloat16).expand(M, N)  # stride-0 dim0
    dump("transposed_add (yt full)", transposed_add, (x, yt))
    dump("expand_add ([1,N].expand)", expand_add, (x, colvec))


if __name__ == "__main__":
    main()
