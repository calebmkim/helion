"""ADVERSARIAL-AUDITOR independent occupancy sweep.

Goal: decide whether the grid-occupancy branch (m_extent<64 AND rnumel>=128KiB
-> looped/warps32) is keyed on GENERALIZABLE occupancy physics or is a disguised
"if kernel==long_sum".

Unlike the worker's grid_occupancy_probe (which compares persistent/16 vs
looped/32 -- conflating loop-structure AND warp count), this sweep ISOLATES the
levers. For a smooth M sweep at fixed huge rnumel and a smaller rnumel, at
several num_warps, we time:
  - persistent (reduction_loops=[None])
  - looped       (reduction_loops=[16384])
each at warps in {4,8,16,32}. So we can read:
  * does looped beat persistent at the SAME warps? (pure loop-structure effect)
  * does warps32 beat warps16/8/4 regardless of loop structure?
  * does any benefit track m_extent SMOOTHLY (occupancy) or step exactly at the
    long_sum M values / the GRID_OCCUPANCY_MIN=64 fence?

We use sum_kernel (num_load=1) -- the same workload class as long_sum -- so the
sweep is kernel-agnostic: if the win tracks m_extent here it generalizes.
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
MS = [1, 4, 16, 32, 48, 64, 96, 128, 256]
WARPS = [4, 8, 16, 32]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def time_cfg(x, ref, reduction_loops, warps):
    cfg = helion.Config(block_sizes=[1], reduction_loops=reduction_loops,
                        num_warps=warps, num_stages=1)
    k = helion.kernel(sum_kernel.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3), "correctness"
    return med(lambda: b(x)) * 1000


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  N_RUNS={N_RUNS}\n")
    for n in [131072, 32768]:  # 512KiB (above ceiling), 128KiB (LOOPED_MIN exactly)
        print(f"========== rnumel={n} ({n*4//1024}KiB) ==========")
        hdr = f"{'M':>5} | " + " | ".join(f"P/w{w}" for w in WARPS) + " || " + \
              " | ".join(f"L/w{w}" for w in WARPS) + " || best_P best_L  L_vs_P"
        print(hdr)
        for m in MS:
            x = torch.randn(m, n, device="cuda", dtype=torch.float32)
            ref = x.sum(-1)
            pers = {w: time_cfg(x, ref, [None], w) for w in WARPS}
            loop = {w: time_cfg(x, ref, [16384], w) for w in WARPS}
            best_p = min(pers.values())
            best_l = min(loop.values())
            row = f"{m:>5} | " + " | ".join(f"{pers[w]:5.1f}" for w in WARPS) + \
                  " || " + " | ".join(f"{loop[w]:5.1f}" for w in WARPS) + \
                  f" || {best_p:6.1f} {best_l:6.1f}  {best_p/best_l:5.2f}"
            print(row)
        print()


if __name__ == "__main__":
    main()
