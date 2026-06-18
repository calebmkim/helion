"""AUDITOR: is the softmax win the rnumel WARPS RAMP (generalizable lever) or a
softmax hack? Matched A/B: persistent R_BLOCK (full-N) at the seed's ramp warps
vs a fixed w32 baseline, M_BLOCK floor, same block_sizes. If the win = ramp, then
at small N (ramp picks w4/w8) seed beats w32, and at large N (ramp = w32) they tie.
fp32.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402

N_RUNS = 7


def med(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def ramp_warps(rnumel):
    if rnumel > 16384:
        return 32
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    return 16


def build(R, w, args):
    # softmax_two_pass block_sizes = [M_BLOCK, R_BLOCK]; red axis is index 1
    seed = helion.Config(block_sizes=[1, R], num_warps=w, num_stages=1)
    k = helion.kernel(softmax_two_pass.fn, configs=[seed])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b


def main():
    shapes = [(4096, 256), (4096, 512), (4096, 1024), (4096, 4096),
              (32768, 256), (8192, 16384), (8192, 32768), (4096, 65536)]
    ratios = []
    print(f"{'shape':>16} {'rnumel':>8} {'rampW':>6} {'seed(ramp)':>11} {'w32':>9} {'seed/w32':>9}")
    for (m, n) in shapes:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        R = np2(n)
        w = ramp_warps(n)
        b_seed = build(R, w, (x,))
        t_seed = med(lambda: b_seed(x)) * 1000
        b_w32 = build(R, 32, (x,))
        t_w32 = med(lambda: b_w32(x)) * 1000
        ratio = t_w32 / t_seed  # >1 => seed faster
        ratios.append(ratio)
        print(f"{f'({m},{n})':>16} {n:>8} {w:>6} {t_seed:>11.1f} {t_w32:>9.1f} {ratio:>9.2f}x")
    geo = 1.0
    for r in ratios:
        geo *= r
    geo = geo ** (1.0 / len(ratios))
    print(f"\nGEOMEAN seed/w32 (>1 => ramp wins) = {geo:.3f}x")


if __name__ == "__main__":
    main()
