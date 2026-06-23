"""Completeness-critic 2nd pass, part C: tile-distribution faithfulness when the INNERMOST
block dim is NOT the contiguous one (column-major / transposed output), and scan-like state.
The seed ALWAYS loads the budget into block[n-1] (last dim) and forces outer->1. If the last
*block* dim is the STRIDED one, [1, budget] would coalesce poorly. Is that a real mis-handle?
Compile-only. Pin one GPU.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl
from helion._compiler.autotuner_heuristics import compiler_seed_configs

assert helion.__file__.startswith(
    "/home/calebkim/helion-new-heuristics/helion-pointwise/"
), helion.__file__


# C1: transposed write -- out is column-major (out.T contiguous). The last ITERATION dim (tile_n)
# maps to the contiguous dim of x but the STRIDED dim of out. Where does the budget land?
@helion.kernel()
def transpose_copy(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty((n, m), device=x.device, dtype=x.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_n, tile_m] = x[tile_m, tile_n] * 2.0
    return out


# C2: SHORT inner dim, LONG outer dim. [M=1048576, N=8]. Inner extent 8 -> tile [1,8] would
# give only 8-wide tiles and 1 outer -> 8 elems/program, terrible. Does outer->1 starve it?
@helion.kernel()
def tall_skinny(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


# C3: cumsum (scan) -- carries state along a dim. Is it reduction/accum/pointwise?
@helion.kernel()
def row_cumsum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        out[tile_m, :] = torch.cumsum(x[tile_m, :], dim=-1)
    return out


def dump(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    pf = spec.pointwise_facts
    print("=== %s ===" % name)
    print("  red=%d mm=%d acc=%d pointwise=%d  heuristic=%s" % (
        len(spec.reduction_facts), len(spec.matmul_facts), len(spec.accumulator_facts),
        len(pf), spec.autotuner_heuristics))
    bsz = [bs.size_hint for bs in spec.block_sizes]
    print("  block_size hints (per dim, in tile order)=%s" % bsz)
    if pf:
        f = pf[0]
        s = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        bs = dict(s[0].config).get("block_sizes") if s else None
        print("  total_numel=%d bytes/elem=%d seed=%s" % (f.total_numel, f.bytes_per_elem, bs))


def main():
    bf = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(4096, 4096, **bf)
    xts = torch.randn(1048576, 8, **bf); yts = torch.randn(1048576, 8, **bf)
    xc = torch.randn(2048, 4096, **bf)

    dump("C1 transpose_copy (strided out)", transpose_copy, (x,))
    dump("C2 tall_skinny (M=1048576,N=8)", tall_skinny, (xts, yts))
    dump("C3 row_cumsum (scan)", row_cumsum, (xc,))


if __name__ == "__main__":
    main()
