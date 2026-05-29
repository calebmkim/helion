"""STEP A safety check: rms_norm (num_load=2) persistent num_warps A/B.

The sum/long_sum sweep (num_load=1) says w32 is best from rnumel~16384 up. But
rms_norm has num_load=2 (re-streams x for the normalize pass) -- a DIFFERENT
arith-intensity. Before changing ANY warps breakpoint that touches rms_norm we
must A/B rms_norm itself at its in-sample shapes. Mandate: rms_norm must not
regress; if a breakpoint would change it, A/B explicitly and keep no-regression.

For each rms_norm in-sample shape, persistent path, sweep num_warps {4,8,16,32}
and report best + per-warp us + G vs torch.compile. We then know whether bumping
the ramp to w32 at rnumel>=16384 helps or hurts rms_norm.
"""

from __future__ import annotations

import os
import sys

import torch
import torch._dynamo

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402

EPS = 1e-5
N_RUNS = 7
WARPS = [4, 8, 16, 32]
# in-sample rms_norm shapes spanning the rnumel where the ramp breakpoints sit
SHAPES = [
    (2048, 1024), (2048, 2048), (2048, 4096), (2048, 8192), (2048, 16384),
    (4096, 5120), (8192, 8192), (32768, 256), (32768, 1024),
]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def time_cfg(args, ref, warps):
    x, w, eps = args
    cfg = helion.Config(block_sizes=[1], reduction_loops=[None],
                        num_warps=warps, num_stages=1)
    k = helion.kernel(rms_norm_fwd.fn, configs=[cfg])
    b = k.bind(args)
    b.ensure_config_exists(args)
    out = b(*args)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-4), "correctness"
    return med(lambda: b(*args)) * 1000


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  N_RUNS={N_RUNS}  rms_norm PERSISTENT path\n")
    # current ramp: rnumel<=1024->4, <=4096->8, else 16
    print(f"{'shape':>14} {'rnumel':>7} {'KiB':>4} | " + " ".join(f"{'w'+str(w):>7}" for w in WARPS)
          + f" | {'tc':>7} | best curr->new")
    for shape in SHAPES:
        m, n = shape
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        args = (x, w, EPS)
        ref = rms_norm_pytorch(x, w, EPS)
        torch._dynamo.reset()
        tc = torch.compile(rms_norm_pytorch)
        _ = tc(x, w, EPS)
        t_tc = med(lambda: tc(x, w, EPS)) * 1000
        t = {wp: time_cfg(args, ref, wp) for wp in WARPS}
        best_w = min(t, key=t.get)
        curr = 4 if n <= 1024 else (8 if n <= 4096 else 16)
        print(f"{str(shape):>14} {n:>7} {n*4//1024:>4} | "
              + " ".join(f"{t[wp]:>7.1f}" for wp in WARPS)
              + f" | {t_tc:>7.1f} | w{best_w} curr=w{curr}")


if __name__ == "__main__":
    main()
