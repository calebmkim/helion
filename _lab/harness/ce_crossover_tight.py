"""Tighten the persistent->looped crossover for num_load>=2 around 64K-96K elems,
across M (grid-occupancy), and confirm num_load=1 (sum) stays persistent-tie at
huge rnumel (no regression risk if we add a multi-load persist byte-cap).

Reports bestPersist/bestLoop (>1 => looped wins) per (kernel, M, rnumel).
Run with the canonical invocation (SETUP.md).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

LONG = torch.int64
N_RUNS = 5
EPS = 1e-5


def median_do_bench(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def time_cfg(fn, args, cfg):
    try:
        k = helion.kernel(fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        b(*args)
        return median_do_bench(lambda: b(*args)) * 1000
    except Exception:  # noqa: BLE001
        return None


def bpl(fn, args, v, mblock=1):
    bp = None
    for w in (8, 16, 32):
        x = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [None],
                                "num_warps": w, "num_stages": 1})
        if x is not None:
            bp = x if bp is None else min(bp, x)
    bl = None
    for ch in (8192, 16384, 32768):
        if ch >= v:
            continue
        for w in (16, 32):
            x = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [ch],
                                    "num_warps": w, "num_stages": 1})
            if x is not None:
                bl = x if bl is None else min(bl, x)
    return bp, bl


def sum_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def rms_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), EPS)


def ce_args(n, v):
    return (torch.randn(n, v, device="cuda", dtype=torch.float32),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG))


def row(label, fn, args, v, mblock=1):
    bp, bl = bpl(fn, args, v, mblock)
    r = (bp / bl) if (bp and bl) else None
    kib = v * 4 // 1024
    print(f"  {label:<22} {kib:>5}KiB  P={bp and round(bp,1):>8}  "
          f"L={bl and round(bl,1):>8}  P/L={r and round(r,2)}")


def main():
    print(f"helion={helion.__file__}\n")
    print("=== num_load>=2 crossover, M=1024 (grid-occupied) ===")
    for v in (65536, 73728, 81920, 90112, 98304):
        row(f"rms_norm nl=2", rms_norm_fwd.fn, rms_args(1024, v), v)
        row(f"cross_entropy nl=3", cross_entropy.fn, ce_args(1024, v), v)
    print("\n=== num_load>=2 crossover, M=8 (grid-STARVED) ===")
    for v in (65536, 98304, 131072):
        row(f"rms_norm nl=2", rms_norm_fwd.fn, rms_args(8, v), v)
        row(f"cross_entropy nl=3", cross_entropy.fn, ce_args(8, v), v)
    print("\n=== num_load=1 (sum) must stay persistent-tie at huge rnumel ===")
    for v in (65536, 98304, 131072, 262144):
        row(f"sum nl=1 M=1024", sum_kernel.fn, sum_args(1024, v), v)
        row(f"sum nl=1 M=8", sum_kernel.fn, sum_args(8, v), v)


if __name__ == "__main__":
    main()
