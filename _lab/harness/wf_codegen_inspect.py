"""Inspect welford codegen for default vs persistent-both configs, at pow2 and
non-pow2 N. Specifically: how is `Tn = chunk.size(-1)` realized in Triton? Is it
the tile constexpr (breaks under masking) or the valid count?
"""
from __future__ import annotations

import sys
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.welford import welford  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 1e-5


def args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def show(label, m, n, cfg):
    a = args(m, n)
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    code = b.to_triton_code(helion.Config(**dict(b._config)))
    print(f"\n############ {label}  ({m},{n})  cfg={cfg} ############")
    print(code)


def main():
    print(f"helion={helion.__file__}")
    # default config for a non-pow2 shape
    a = args(4096, 1536)
    bd = welford.bind(a)
    cfg_d = dict(bd.config_spec.default_config())
    print("DEFAULT cfg (4096,1536):", cfg_d)
    show("DEFAULT", 4096, 1536, cfg_d)
    # persistent-both at non-pow2: red=normalize=next_pow2(N)=2048, masked
    show("PERSIST-BOTH-pow2tile", 4096, 1536,
         {"block_sizes": [1, np2(1536), np2(1536)], "num_warps": 8, "num_stages": 1})
    # persistent-both with EXACT N tile (no mask) -- but N not pow2, may be illegal
    show("PERSIST-BOTH-exactN", 4096, 1536,
         {"block_sizes": [1, 1536, 1536], "num_warps": 8, "num_stages": 1})
    # looped DIVISOR chunk: 1536 = 512*3, so chunk=512 divides exactly
    show("LOOPED-DIV512", 4096, 1536,
         {"block_sizes": [1, 512, 512], "num_warps": 8, "num_stages": 1})


if __name__ == "__main__":
    main()
