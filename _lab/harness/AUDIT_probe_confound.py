"""Show the worker's grid_occupancy_probe confound: it attributes the win to
LOOP STRUCTURE but it is really NUM_WARPS.

Worker probe compared persistent/warps16 vs looped/warps32 -> concluded "looped
wins at small M". This adds the missing controls: persistent/warps32 and
looped/warps16. If persistent/warps32 ~= looped/warps32, the loop flip is a
no-op and the win is the warps.

Replicates the worker's exact rnumels (32768, 65536) and M sweep.
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.sum import sum_kernel  # noqa: E402

N_RUNS = 5
MS = [1, 2, 4, 8, 16, 32, 64, 132, 256]
RNUMELS = [32768, 65536]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def run(m, n, reduction_loops, warps):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    ref = x.sum(-1)
    cfg = helion.Config(block_sizes=[1], reduction_loops=reduction_loops,
                        num_warps=warps, num_stages=1)
    k = helion.kernel(sum_kernel.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3)
    return med(lambda: b(x)) * 1000


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  helion={helion.__file__}\n")
    for n in RNUMELS:
        print(f"=== rnumel={n} ({n*4//1024}KiB) ===")
        print(f"  {'M':>5} {'P/16':>7} {'P/32':>7} {'L/16':>7} {'L/32':>7} "
              f"{'worker(P16/L32)':>16} {'fair(P32/L32)':>14}")
        for m in MS:
            p16 = run(m, n, [None], 16)
            p32 = run(m, n, [None], 32)
            l16 = run(m, n, [16384], 16)
            l32 = run(m, n, [16384], 32)
            worker_ratio = p16 / l32   # what the worker reported as "pers/loop"
            fair_ratio = p32 / l32     # same warps: pure loop-structure effect
            print(f"  {m:>5} {p16:>7.1f} {p32:>7.1f} {l16:>7.1f} {l32:>7.1f} "
                  f"{worker_ratio:>16.3f} {fair_ratio:>14.3f}")
        print()


if __name__ == "__main__":
    main()
