"""AUDITOR A3+A4: the v4-vs-v6 consistency + fence check.

v4 (ACCEPTED) made nl>=2 large-rnumel rms_norm/layer_norm PERSISTENT/w32 (the
OOS multi-load recovery). v6 now LOOPS them (>128KiB). For the EXACT v4-recovery
shapes + held-out cross_entropy + held-out rms/layer_norm large-rnumel, directly
A/B the two SEEDS:
  v6_seed  = looped chunk=16384, w32     (what v6 emits for nl>=2 >128KiB)
  v4_seed  = persistent (reduction_loops=[None]), w32  (what v4 emitted)
plus the best-of-each persistent (sweep warps) as a sanity ceiling.

v6 supersedes v4 ONLY if v6_seed (looped) is FASTER than v4_seed (persist/w32)
on these shapes. If slower -> v6 regressed the accepted OOS recovery -> FAIL.

The fence question: does looped help nl>=2 GENERALLY (rms_norm + layer_norm +
cross_entropy), or only cross_entropy?  Run all three.
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
    except Exception as e:  # noqa: BLE001
        return f"ERR:{type(e).__name__}"


def best_persist(fn, args, mblock=1):
    vals = []
    for w in (8, 16, 32):
        x = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [None],
                                "num_warps": w, "num_stages": 1})
        if isinstance(x, float):
            vals.append(x)
    return min(vals) if vals else None


def row(label, fn, args, mblock=1):
    # v6 seed: looped chunk 16384 w32
    v6 = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [16384],
                             "num_warps": 32, "num_stages": 1})
    # v4 seed: persistent w32
    v4 = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [None],
                             "num_warps": 32, "num_stages": 1})
    bp = best_persist(fn, args, mblock)
    if isinstance(v6, float) and isinstance(v4, float):
        speedup = v4 / v6  # >1 => v6 looped faster => v6 supersedes (good)
        verdict = "v6 FASTER (supersedes v4)" if speedup > 1.02 else (
            "v6 SLOWER (REGRESSES v4)" if speedup < 0.98 else "~tie")
    else:
        speedup = None
        verdict = f"v6={v6} v4={v4}"
    print(f"  {label:<30} v6_loop/w32={v6 if not isinstance(v6,float) else round(v6,1):>8}  "
          f"v4_persist/w32={v4 if not isinstance(v4,float) else round(v4,1):>8}  "
          f"bestPersist={bp and round(bp,1)}  v4/v6={speedup and round(speedup,3)} -> {verdict}")


def rms_args(m, n):
    return (torch.randn(m, n, device="cuda"), torch.randn(n, device="cuda"), EPS)


def ln_args(m, n):
    return (torch.randn(m, n, device="cuda"), [n], torch.randn(n, device="cuda"),
            torch.randn(n, device="cuda"), EPS)


def ce_args(n, v):
    return (torch.randn(n, v, device="cuda"),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG))


def main():
    print(f"helion={helion.__file__}")
    print("v4/v6 > 1 => v6 looped faster (v6 supersedes v4). < 1 => v6 regressed v4.\n")

    print("=== A3: EXACT v4-accepted OOS recovery shapes (rms_norm/layer_norm) ===")
    row("rms_norm (1,131072) nl=2", rms_norm_fwd.fn, rms_args(1, 131072), 1)
    row("rms_norm (16,131072) nl=2", rms_norm_fwd.fn, rms_args(16, 131072), 1)
    row("rms_norm (16,262144) nl=2", rms_norm_fwd.fn, rms_args(16, 262144), 1)
    row("layer_norm (1,131072) nl=3", layer_norm_fwd.fn, ln_args(1, 131072), 1)
    row("layer_norm (16,131072) nl=3", layer_norm_fwd.fn, ln_args(16, 131072), 1)
    row("layer_norm (16,262144) nl=3", layer_norm_fwd.fn, ln_args(16, 262144), 1)

    print("\n=== A4: held-out cross_entropy (the widening target) ===")
    row("cross_entropy (2048,32000)", cross_entropy.fn, ce_args(2048, 32000), 1)
    row("cross_entropy (4096,32000)", cross_entropy.fn, ce_args(4096, 32000), 1)
    row("cross_entropy (8192,128000)", cross_entropy.fn, ce_args(8192, 128000), 1)
    row("cross_entropy (2048,128256)", cross_entropy.fn, ce_args(2048, 128256), 1)

    print("\n=== A4: held-out rms_norm/layer_norm large-rnumel (generality) ===")
    row("rms_norm (2048,65536) nl=2", rms_norm_fwd.fn, rms_args(2048, 65536), 1)
    row("rms_norm (4096,131072) nl=2", rms_norm_fwd.fn, rms_args(4096, 131072), 1)
    row("layer_norm (2048,65536) nl=3", layer_norm_fwd.fn, ln_args(2048, 65536), 1)


if __name__ == "__main__":
    main()
