"""AUDITOR independent A/B: persistent vs looped across num_load, reporting BOTH
warps-held-equal (per warp) AND best-of-each, to separate a num_warps confound
(the v3 lesson) from a genuine num_load-keyed perf crossover.

For each (kernel, M, rnumel) prints:
  - per-warp P/L at w in {16,32} (warps HELD EQUAL: persistent_w vs looped_w)
  - best-of-each P/L (min over warps for each path) -- what a seed would pick
A num_load=1 control (sum) must stay tied; num_load>=2 (rms_norm nl=2,
cross_entropy nl=3) is the test class.

Run with the canonical invocation (SETUP.md / GPU1).
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
WARPS = (8, 16, 32)
CHUNKS = (4096, 8192, 16384, 32768)


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


def persist_times(fn, args, mblock=1):
    """{warp: us} for persistent."""
    out = {}
    for w in WARPS:
        x = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [None],
                                "num_warps": w, "num_stages": 1})
        if x is not None:
            out[w] = x
    return out


def loop_times(fn, args, v, mblock=1):
    """{warp: best-us-over-chunks} for looped."""
    out = {}
    for w in WARPS:
        best = None
        for ch in CHUNKS:
            if ch >= v:
                continue
            x = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [ch],
                                    "num_warps": w, "num_stages": 1})
            if x is not None:
                best = x if best is None else min(best, x)
        if best is not None:
            out[w] = best
    return out


def sum_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def rms_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), EPS)


def ce_args(n, v):
    return (torch.randn(n, v, device="cuda", dtype=torch.float32),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG))


def row(label, fn, args, v, mblock=1):
    P = persist_times(fn, args, mblock)
    L = loop_times(fn, args, v, mblock)
    kib = v * 4 // 1024
    # per-warp held-equal P/L
    eq = []
    for w in (16, 32):
        if w in P and w in L:
            eq.append(f"w{w}:{P[w]/L[w]:.2f}")
    bp = min(P.values()) if P else None
    bl = min(L.values()) if L else None
    boe = (bp / bl) if (bp and bl) else None
    bestPw = min(P, key=P.get) if P else None
    bestLw = min(L, key=L.get) if L else None
    print(f"  {label:<20} {kib:>5}KiB | held-equal P/L {' '.join(eq):<22}"
          f"| bestP={bp and round(bp,1)}(w{bestPw}) bestL={bl and round(bl,1)}(w{bestLw}) "
          f"best-of-each P/L={boe and round(boe,3)}")


def main():
    print(f"helion={helion.__file__}\n")
    print("P/L > 1 => looped wins. held-equal isolates num_warps confound.\n")
    for M in (1024, 8):
        print(f"########## M = {M} ##########")
        for v in (16384, 32768, 49152, 65536, 98304, 131072, 262144):
            print(f"--- rnumel={v} ({v*4//1024}KiB) ---")
            row("sum nl=1", sum_kernel.fn, sum_args(M, v), v, mblock=1)
            row("rms_norm nl=2", rms_norm_fwd.fn, rms_args(M, v), v, mblock=1)
            row("cross_entropy nl=3", cross_entropy.fn, ce_args(M, v), v, mblock=1)
        print()


if __name__ == "__main__":
    main()
