"""Confirm the tall-skinny regression: seed [1,8] vs compiler DEFAULT for [M=1048576, N=8].
If default > seed in elems/program, the seed REGRESSES below default on this realistic class
(e.g. RGBA images, [tokens, head_dim=8], small-feature elementwise). Compile-only.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl
from helion._compiler.autotuner_heuristics import compiler_seed_configs

assert helion.__file__.startswith(
    "/home/calebkim/helion-new-heuristics/helion-pointwise/"
), helion.__file__


@helion.kernel()
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out


def report(M, N):
    bf = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, N, **bf); y = torch.randn(M, N, **bf)
    bound = add.bind((x, y))
    spec = bound.config_spec
    # compiler default config
    dflt = spec.default_config()
    dflt_bs = dict(dflt).get("block_sizes")
    # seed
    s = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    seed_bs = dict(s[0].config).get("block_sizes") if s else None
    def prod(b):
        p = 1
        for v in b: p *= v
        return p
    print("M=%-9d N=%-5d  DEFAULT block_sizes=%-12s (%d elems)  SEED=%-12s (%d elems)  seed/default=%.3f" % (
        M, N, dflt_bs, prod(dflt_bs), seed_bs, prod(seed_bs), prod(seed_bs)/prod(dflt_bs)))


def main():
    report(1048576, 8)     # tall-skinny: head_dim/RGBA-like
    report(2097152, 4)     # RGBA image pixels
    report(524288, 16)
    report(131072, 64)
    report(16384, 11008)   # the canonical wide case (sanity: seed should beat default here)
    report(4096, 4096)


if __name__ == "__main__":
    main()
