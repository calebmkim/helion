"""Gate D fix verification: full-rank broadcasts ([M,1], [1,N]) must be EXCLUDED from
bytes_per_elem (accessed_numel == total_numel test), while corpus traffic is unchanged."""

from __future__ import annotations

import torch

import helion
import helion.language as hl

from helion._compiler.autotuner_heuristics import compiler_seed_configs
from examples.geglu import _geglu
from examples.swiglu import _swiglu_fwd

import sys
sys.path.insert(0, "/home/calebkim/helion-new-heuristics/helion-pointwise/_lab/pointwise")
import ptw_kernels as PK


@helion.kernel()
def rowbias_2d(x: torch.Tensor, rbias: torch.Tensor) -> torch.Tensor:
    """x[M,N] + rbias[M,1] (per-row bias broadcast over N) — a FULL-RANK broadcast."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(torch.float32) + rbias[tm, 0].to(torch.float32)).to(x.dtype)
    return out


@helion.kernel()
def colbias_2d(x: torch.Tensor, cbias: torch.Tensor) -> torch.Tensor:
    """x[M,N] + cbias[1,N] (per-col bias broadcast over M) — a FULL-RANK broadcast."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(torch.float32) + cbias[0, tn].to(torch.float32)).to(x.dtype)
    return out


def info(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    pf = spec.pointwise_facts
    be = pf[0].bytes_per_elem if pf else None
    sc = dict(seeds[0].config) if seeds else None
    code = bound.to_triton_code(seeds[0]) if seeds else ""
    bs = [ln.strip() for ln in code.splitlines() if "_BLOCK_SIZE_0 = tl.constexpr" in ln]
    nw = [ln.strip() for ln in code.splitlines() if "num_warps" in ln]
    print(f"{name:20} pw={len(pf)} bytes/elem={be} seed={sc}")
    if bs:
        print(f"   codegen: {bs[0]}  ;  {(nw[0] if nw else '')[:90]}")


@helion.kernel()
def expand_add(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """x[M,N] + e[M,N] where e is passed as a stride-0 .expand() view — the Gate D stride
    divergence: full SIZE shape but one underlying row/col of HBM traffic."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(torch.float32) + e[tm, tn].to(torch.float32)).to(x.dtype)
    return out


def main():
    from examples.add import add

    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    rb = torch.randn(8192, 1, device="cuda", dtype=torch.bfloat16)
    cb = torch.randn(1, 8192, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(8192, device="cuda", dtype=torch.bfloat16)
    info("swiglu", _swiglu_fwd, (x, b))
    info("geglu", _geglu, (x, b))
    info("relu_squared", PK.relu_squared, (x,))
    info("bias_gelu rank1[N]", PK.bias_gelu, (x, bias))
    info("rowbias [M,1] FIX", rowbias_2d, (x, rb))
    info("colbias [1,N] FIX", colbias_2d, (x, cb))
    # stride-0 expand cases (the NEW Gate D divergence)
    info("expand [M,1]->MN", expand_add, (x, rb.expand_as(x)))
    info("expand [1,N]->MN", expand_add, (x, cb.expand_as(x)))
    info("add(x,bias[N]) bcast", add, (x, bias))


if __name__ == "__main__":
    main()
