"""Map welford's seed block_sizes positions to block_ids and roles."""
from __future__ import annotations

import sys
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.welford import welford  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion.autotuner.config_spec import BlockSizeSpec  # noqa: E402

EPS = 1e-5


def show(m, n):
    args = (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)
    b = welford.bind(args)
    spec = b.env.config_spec
    dev_ir = b.host_function.device_ir
    grid_ids = {x for bids in dev_ir.grid_block_ids for x in bids}
    fact = spec.reduction_facts[0]
    seeds = compiler_seed_configs(b.env, dev_ir)
    sd = dict(seeds[0])
    print(f"\n(m={m}, n={n}): seed block_sizes={sd['block_sizes']} warps={sd['num_warps']}")
    print(f"  red(combine) block_id={fact.block_id}  apply_block_ids={fact.apply_block_ids}")
    print(f"  np2(n)={1 << (n-1).bit_length()}  largest_pow2_div={n & (-n)}")
    for i in range(len(spec.block_sizes)):
        bs = spec.block_sizes[i]
        bid = bs.block_id
        role = "grid/M" if bid in grid_ids else ("combine(red)" if bid == fact.block_id
               else ("apply" if bid in fact.apply_block_ids else "other"))
        print(f"  idx={i} block_id={bid} role={role:14s} seed_val={sd['block_sizes'][i]}")


if __name__ == "__main__":
    print(f"helion={helion.__file__}")
    for (m, n) in [(262144, 1536), (262144, 1024), (262144, 2048), (65536, 1500), (262144, 1543)]:
        show(m, n)
