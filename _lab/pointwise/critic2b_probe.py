"""Completeness-critic 2nd pass, part B: disjointness FALSE-EXCLUSION + dynamic-shape + skinny.
A pointwise kernel WRONGLY excluded (accum/reduction false-positive) is a coverage hole.
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


# H1: manual 2-level tiling of a PURE elementwise (outer tile_m, inner tile_n loop, NO carry).
# A user hand-writes the inner loop for a pure map. Must still be pointwise (no carry tensor).
@helion.kernel()
def manual_nested_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


# H2: pointwise with a SCALAR python-float accumulation across the inner loop that is NOT a
# tensor carry (just reads/writes). Idiomatic? Mostly checks accum builder doesn't false-fire.
@helion.kernel()
def gelu_unary(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        v = x[t]
        out[t] = v * 0.5 * (1.0 + torch.erf(v * 0.7071067811865476))
    return out


# H3: dynamic-shape pointwise (no specialization). Does total_numel via size_hint give a sane
# tile, and does the occupancy cap behave?
@helion.kernel()
def dyn_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        out[t] = x[t] + y[t]
    return out


# H4: TINY/skinny problem -> occupancy cap may drive tile to 1. Check seed >= 1, no crash,
# and it is at least the BLOCK_FLOOR or capped sanely.
@helion.kernel()
def tiny_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for t in hl.tile(x.size()):
        out[t] = x[t] + y[t]
    return out


def dump(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    pf = spec.pointwise_facts
    print("=== %s ===" % name)
    print("  red=%d mm=%d acc=%d pointwise=%d  heuristic=%s" % (
        len(spec.reduction_facts), len(spec.matmul_facts), len(spec.accumulator_facts),
        len(pf), spec.autotuner_heuristics))
    if pf:
        f = pf[0]
        s = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        bs = dict(s[0].config).get("block_sizes") if s else None
        print("  total_numel=%d bytes/elem=%d seed=%s" % (f.total_numel, f.bytes_per_elem, bs))


def main():
    bf = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(4096, 4096, **bf); y = torch.randn(4096, 4096, **bf)
    xs = torch.randn(8, 128, **bf); ys = torch.randn(8, 128, **bf)        # tiny
    xd = torch.randn(2048, 3000, **bf); yd = torch.randn(2048, 3000, **bf) # non-pow2

    dump("H1 manual_nested_add (inner loop, no carry)", manual_nested_add, (x, y))
    dump("H2 gelu_unary", gelu_unary, (x,))
    dump("H3 dyn_add (non-pow2 N=3000)", dyn_add, (xd, yd))
    dump("H4 tiny_add (8x128)", tiny_add, (xs, ys))


if __name__ == "__main__":
    main()
