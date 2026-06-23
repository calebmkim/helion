"""Gate D oversized-operand fix: a kernel tiling [M,N] but reading a WIDER buffer big[M,2N]
sub-indexed must count big as full-extent (touches M*N under the tile) -> bytes_per_elem=4, not 2."""
from __future__ import annotations
import torch
import helion
import helion.language as hl
from helion._compiler.autotuner_heuristics import compiler_seed_configs


@helion.kernel()
def sub_tile_add(x: torch.Tensor, big: torch.Tensor) -> torch.Tensor:
    """out[M,N] = x[M,N] + big[M,N-window of a wider [M,2N] buffer]. Oversized operand (big)."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = (x[tm, tn].to(torch.float32) + big[tm, tn].to(torch.float32)).to(x.dtype)
    return out


def info(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    s = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    pf = spec.pointwise_facts
    print("%-22s pw=%d bytes/elem=%s seed=%s" % (
        name, len(pf), pf[0].bytes_per_elem if pf else None,
        dict(s[0].config) if s else None))


def main():
    M, N = 4096, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    big_full = torch.randn(M, 2 * N, device="cuda", dtype=torch.bfloat16)  # UNSLICED [M,2N]
    # kernel tiles x.size()=[M,N] and indexes big_full[tm,tn] (tn in [0,N)) -> reads the [M,N]
    # sub-region of a [M,2N] operand; big_full.accessed_numel = M*2N > total_numel = M*N.
    info("sub_tile oversized[M,2N]", sub_tile_add, (x, big_full))


if __name__ == "__main__":
    main()
