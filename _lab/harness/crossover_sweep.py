"""Synthetic persistent-vs-looped crossover sweep (workload characterization).

Goal (Step 2b): find, by EVIDENCE, the rnumel at which a persistent single-pass
reduction STOPS beating a looped-chunk reduction on H100/fp32, so we can set
``PERSIST_MAX_BYTES`` at that crossover (expressed in bytes via itemsize), and
find the best looped-branch params (chunk / num_warps / num_stages) for the rows
above the crossover.

This is IN-SAMPLE-legitimate: we are characterizing a WORKLOAD PROPERTY (how a
contiguous fp32 row reduction behaves as rnumel grows) on rms_norm_fwd — NOT
tuning on any held-out shape. We sweep rnumel over a grid spanning the regime and
a few M (row-count) values to confirm the crossover is an rnumel property, not an
M property.

For each (M, rnumel) we bare-seed rms_norm_fwd with:
  - persistent: reduction_loops=[None]
  - looped: reduction_loops=[chunk] for chunk in CHUNKS
and for each we also sweep num_warps / num_stages a little so the looped branch is
given a fair shot (the comparison is persistent-best vs looped-best). All fp32,
correctness-gated, median-of-N do_bench. We report per-(M,rnumel) the best
persistent latency, the best looped latency + its params, and the ratio
persist/looped (>1 => looped wins).
"""

from __future__ import annotations

import itertools
import math
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

# rnumel grid spanning the regime where the crossover is expected (16384 =
# current PERSIST_MAX = 64KiB fp32; up to 262144 = 1MiB fp32).
RNUMELS = [16384, 32768, 49152, 65536, 98304, 131072, 196608, 262144]
# A few M values: small-M (long_sum-like grid-starved), medium, large.
MS = [8, 1024, 4096]
# Looped chunk candidates (power-of-2-ish; the default uses 4096).
CHUNKS = [2048, 4096, 8192]
# Warp / stage candidates to give each branch a fair shot.
WARPS = [4, 8, 16]
STAGES = [1, 2, 3]


def build_args(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, EPS)


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def run_cfg(args, ref, reduction_loops, num_warps, num_stages):
    """Bare-seed a single config; return latency_us or None if incorrect/failed."""
    try:
        cfg = helion.Config(
            block_sizes=[1] if args[0].shape[0] > 0 else [1],
            reduction_loops=reduction_loops,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        k = helion.kernel(rms_norm_fwd.fn, configs=[cfg])
        b = k.bind(args)
        b.ensure_config_exists(args)
        out = b(*args)
        out = out[0] if isinstance(out, tuple) else out
        if not torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-4):
            return None
        return med(lambda: b(*args)) * 1000
    except Exception as e:  # noqa: BLE001
        return None


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}")
    print(f"RNUMELS={RNUMELS}  MS={MS}  CHUNKS={CHUNKS}  WARPS={WARPS}  STAGES={STAGES}\n")

    hdr = (f"{'M':>6} {'rnumel':>8} {'KiB':>6} | "
           f"{'pers_us':>8} {'pers(w,s)':>10} | "
           f"{'loop_us':>8} {'loop(c,w,s)':>16} | {'pers/loop':>9} {'winner':>7}")
    print(hdr)
    print("-" * len(hdr))

    # accumulate crossover evidence
    for m in MS:
        for rnumel in RNUMELS:
            args = build_args(m, rnumel)
            x, w, e = args
            ref = rms_norm_pytorch(x, w, e)

            # best persistent (sweep warps/stages)
            best_p = (math.inf, None)
            for nw, ns in itertools.product(WARPS, STAGES):
                lat = run_cfg(args, ref, [None], nw, ns)
                if lat is not None and lat < best_p[0]:
                    best_p = (lat, (nw, ns))

            # best looped (sweep chunk/warps/stages)
            best_l = (math.inf, None)
            for chunk, nw, ns in itertools.product(CHUNKS, WARPS, STAGES):
                if chunk >= rnumel:
                    continue  # chunk>=rnumel collapses to persistent
                lat = run_cfg(args, ref, [chunk], nw, ns)
                if lat is not None and lat < best_l[0]:
                    best_l = (lat, (chunk, nw, ns))

            ratio = (best_p[0] / best_l[0]) if best_l[0] not in (0, math.inf) else float("nan")
            winner = "loop" if ratio > 1.0 else "pers"
            print(
                f"{m:>6} {rnumel:>8} {rnumel*4//1024:>6} | "
                f"{best_p[0]:>8.1f} {str(best_p[1]):>10} | "
                f"{best_l[0]:>8.1f} {str(best_l[1]):>16} | {ratio:>9.3f} {winner:>7}"
            )
        print()


if __name__ == "__main__":
    main()
