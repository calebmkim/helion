"""Tight threshold sweep in the (128KiB, 256KiB] region the v6 cap at 128KiB
sends to looped. Question: does persistent still WIN in this band for nl>=2
(=> 128KiB cap is harmful), or does looped already win (=> 128KiB fine)?

Reports best-of-each P/L (what a seed picks) at fine rnumel steps, at BOTH
M=1024 (grid-occupied, the in-sample-style) and M=8 (grid-starved). 3 reps to
show spread. The v6 cap LOOPS everything > 32768 elems (128KiB) for nl>=2.
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

LONG = torch.int64
EPS = 1e-5
WARPS = (8, 16, 32)
CHUNKS = (4096, 8192, 16384, 32768)


def med(fn, reps=5):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def time_cfg(fn, args, cfg):
    try:
        k = helion.kernel(fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        b(*args)
        return med(lambda: b(*args)) * 1000
    except Exception:  # noqa: BLE001
        return None


def best_p(fn, args, mblock=1):
    vals = [time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [None],
            "num_warps": w, "num_stages": 1}) for w in WARPS]
    vals = [x for x in vals if x is not None]
    return min(vals) if vals else None


def best_l(fn, args, v, mblock=1):
    best = None
    for ch in CHUNKS:
        if ch >= v:
            continue
        for w in WARPS:
            x = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [ch],
                                    "num_warps": w, "num_stages": 1})
            if x is not None:
                best = x if best is None else min(best, x)
    return best


def rms_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), EPS)


def ln_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32), [n],
            torch.randn(n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), EPS)


def ce_args(n, v):
    return (torch.randn(n, v, device="cuda", dtype=torch.float32),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG))


def row(label, fn, args, v, mblock=1):
    ratios = []
    for _ in range(3):
        bp = best_p(fn, args, mblock)
        bl = best_l(fn, args, v, mblock)
        ratios.append((bp / bl) if (bp and bl) else None)
    rr = [r for r in ratios if r is not None]
    kib = v * 4 // 1024
    verdict = "PERSIST wins" if (rr and max(rr) < 1.0) else (
        "LOOPED wins" if (rr and min(rr) > 1.0) else "mixed/tie")
    print(f"  {label:<18} {kib:>5}KiB | P/L reps={[round(r,3) for r in rr]} -> {verdict}")


def main():
    print(f"helion={helion.__file__}")
    print("v6 cap=128KiB: LOOPS everything > 32768 elems for nl>=2.")
    print("If PERSIST wins in (128,256]KiB the cap is HARMFUL there.\n")
    for M in (1024, 8):
        print(f"########## M={M} ##########")
        for v in (40960, 49152, 57344, 65536):  # 160,192,224,256 KiB
            row("rms_norm nl=2", rms_norm_fwd.fn, rms_args(M, v), v, 1)
            row("layer_norm nl=3", layer_norm_fwd.fn, ln_args(M, v), v, 1)
            row("cross_entropy nl=3", cross_entropy.fn, ce_args(M, v), v, 1)
            print()


if __name__ == "__main__":
    main()
