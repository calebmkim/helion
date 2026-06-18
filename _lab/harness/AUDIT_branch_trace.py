"""Trace which heuristic branch fires for held-out shapes across kernels.

For each (kernel, shape) print the seed config the heuristic emits + which
branch it takes (persistent / looped-by-byte-ceiling / looped-by-grid-occupancy),
plus the raw m_extent and rnumel_bytes so we can verify the AND-condition.
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.long_sum import longsum  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion._compiler.autotuner_heuristics.triton import TritonReductionHeuristic as H  # noqa: E402

EPS = 1e-5


def build(kernel, m, n):
    if kernel == "rms_norm":
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        return rms_norm_fwd, (x, w, EPS)
    if kernel == "sum":
        return sum_kernel, (torch.randn(m, n, device="cuda", dtype=torch.float32),)
    return longsum, (torch.randn(m, n, device="cuda", dtype=torch.float32),)


CASES = [
    # held-out rms_norm: small M but small rnumel -> AND-condition must NOT fire
    ("rms_norm", 16, 4096), ("rms_norm", 128, 4096),
    ("rms_norm", 1024, 32768),  # held-out big row, occupied
    # held-out sum
    ("sum", 16, 4096), ("sum", 1, 4096),
    # held-out long_sum: (32,65536) branch must FIRE; others
    ("long_sum", 32, 65536), ("long_sum", 1, 100000),
    ("long_sum", 4, 262143), ("long_sum", 1, 1048576),
]


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  helion={helion.__file__}")
    print(f"PERSIST_MAX_BYTES={H.PERSIST_MAX_BYTES} GRID_OCCUPANCY_MIN={H.GRID_OCCUPANCY_MIN} "
          f"LOOPED_MIN_BYTES={H.LOOPED_MIN_BYTES}\n")
    hdr = f"{'kernel':>10} {'shape':>14} {'m_extent':>9} {'rnumelB':>9} {'grid_starved':>12} {'seed':>45} {'branch':>22}"
    print(hdr)
    print("-" * len(hdr))
    for kernel, m, n in CASES:
        fn, args = build(kernel, m, n)
        bound = fn.bind(args)
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        if not seeds:
            print(f"{kernel:>10} {str((m,n)):>14}  NO SEED")
            continue
        fact = bound.env.config_spec.reduction_facts[0]
        rnumel_bytes = fact.size_hint * fact.itemsize
        m_extent = H._m_extent(bound.env, fact)
        grid_starved = m_extent < H.GRID_OCCUPANCY_MIN and rnumel_bytes >= H.LOOPED_MIN_BYTES
        seed = dict(seeds[0])
        rl = seed.get("reduction_loops")
        if rl == [None]:
            branch = "persistent"
        elif rnumel_bytes > H.PERSIST_MAX_BYTES:
            branch = "looped(byte-ceiling)"
        else:
            branch = "looped(grid-occupancy)"
        seedstr = f"rl={rl},w{seed['num_warps']},bs={seed['block_sizes']}"
        print(f"{kernel:>10} {str((m,n)):>14} {m_extent:>9} {rnumel_bytes:>9} "
              f"{str(grid_starved):>12} {seedstr:>45} {branch:>22}")


if __name__ == "__main__":
    main()
