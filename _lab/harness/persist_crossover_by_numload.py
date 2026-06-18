"""Find the persistent->looped crossover as a function of rnumel, SEPARATELY for
num_load=1 (sum) and num_load>=2 (rms_norm nl=2, layer_norm nl=3, cross_entropy nl=3).

The champion's "persistent to the 2^20 structural cap" was validated ONLY on
num_load=1 (sum, v3_crossover_sweep). This sweep tests whether multi-load wide
rows want looped EARLIER. Holds warps EQUAL within each persist/loop best (sweeps
warps for each). Reports bestPersist/bestLoop (>1 => looped wins) per rnumel.

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
from examples.layer_norm import layer_norm_fwd  # noqa: E402
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
    except Exception as e:  # noqa: BLE001
        return None


def best_persist_loop(fn, args, v, mblock=1):
    bp = None
    for w in (8, 16, 32):
        x = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [None],
                                "num_warps": w, "num_stages": 1})
        if x is not None:
            bp = x if bp is None else min(bp, x)
    bl = None
    for ch in (4096, 8192, 16384, 32768):
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
    bp, bl = best_persist_loop(fn, args, v, mblock)
    r = (bp / bl) if (bp and bl) else None
    print(f"  {label:<26} P={bp and round(bp,1):>8}  L={bl and round(bl,1):>8}  "
          f"P/L={r and round(r,2)}")


def main():
    print(f"helion={helion.__file__}\n")
    # M fixed at 1024 (grid-occupied) to isolate rnumel; sweep rnumel across the
    # crossover region. The in-sample rms_norm/layer_norm max rnumel is ~16384.
    for v in (16384, 32768, 49152, 65536, 98304, 131072):
        print(f"### rnumel = {v} (M=1024) ###")
        row(f"sum nl=1", sum_kernel.fn, sum_args(1024, v), v)
        row(f"rms_norm nl=2", rms_norm_fwd.fn, rms_args(1024, v), v)
        row(f"cross_entropy nl=3", cross_entropy.fn, ce_args(1024, v), v)
        print()


if __name__ == "__main__":
    main()
