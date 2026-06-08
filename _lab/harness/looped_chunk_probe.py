"""Probe the looped-branch chunk/warps/stages in the looped-winning region.

For the rnumel where looped wins (>=98304) and a couple of representative M
(tiny-M long_sum-like + occupied), break down looped latency by chunk so we can
pick LOOPED_CHUNK by evidence. Also report num_warps/num_stages sensitivity.
"""

from __future__ import annotations

import itertools
import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402

EPS = 1e-5
N_RUNS = 5
# Looped region + the boundary; tiny-M (long_sum) and occupied.
CASES = [(8, 98304), (8, 131072), (8, 262144), (4, 130000), (16, 262144), (1024, 131072)]
CHUNKS = [2048, 4096, 8192, 16384]
WARPS = [8, 16, 32]
STAGES = [1, 2]


def build_args(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, EPS)


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def run_cfg(args, ref, chunk, nw, ns):
    try:
        cfg = helion.Config(block_sizes=[1], reduction_loops=[chunk],
                            num_warps=nw, num_stages=ns)
        k = helion.kernel(rms_norm_fwd.fn, configs=[cfg])
        b = k.bind(args)
        b.ensure_config_exists(args)
        out = b(*args)
        out = out[0] if isinstance(out, tuple) else out
        if not torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-4):
            return None
        return med(lambda: b(*args)) * 1000
    except Exception:  # noqa: BLE001
        return None


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  helion={helion.__file__}\n")
    for m, n in CASES:
        args = build_args(m, n)
        x, w, e = args
        ref = rms_norm_pytorch(x, w, e)
        print(f"=== M={m} rnumel={n} ({n*4//1024}KiB) ===")
        # per-chunk best (over warps/stages)
        print(f"  {'chunk':>6} {'best_us':>8} {'(w,s)':>8}")
        for chunk in CHUNKS:
            if chunk >= n:
                continue
            best = (float('inf'), None)
            for nw, ns in itertools.product(WARPS, STAGES):
                lat = run_cfg(args, ref, chunk, nw, ns)
                if lat is not None and lat < best[0]:
                    best = (lat, (nw, ns))
            print(f"  {chunk:>6} {best[0]:>8.1f} {str(best[1]):>8}")
        print()


if __name__ == "__main__":
    main()
