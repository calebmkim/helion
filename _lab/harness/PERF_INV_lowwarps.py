"""Perf-investigator: is the small-N gap a num_warps-too-high artifact?

The full-autotune ORACLE for rms_norm (32768,256) picked block_sizes=[2] (= the
seed FLOOR, NOT a big M-block) with num_warps=1 -- so the oracle's win is NOT a
larger M-block; it is FEWER warps (the seed ramp gives w4 at rnumel<=1024). A
256-elem reduction over-subscribes 4 warps (128 lanes) on a 256-wide row.

A/B: hold the seed config, vary ONLY num_warps in {1,2,4} at the FLOOR M-block.
If w1/w2 recovers the gap, the lever is "tinier warps for tiny rnumel" -- a
generalizable rnumel-keyed knob the seed already owns. Compare against the
M-block route (mblock big, w4) measured separately. Both at floor warps.

Run across the headroom shapes AND large-rnumel controls (no-regression: w1/w2
must NOT help / must hurt at large rnumel where w16/w32 is right).
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

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7
LONG = torch.int64


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def build_rms(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), 1e-5)


def build_x(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def build_ce(shape):
    n, v = shape
    return (torch.randn(n, v, device="cuda", dtype=torch.float32),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG))


REG = {
    "rms_norm": (rms_norm_fwd, build_rms, lambda a: rms_norm_pytorch(*a),
                 lambda: torch.compile(rms_norm_pytorch)),
    "sum": (sum_kernel, build_x, lambda a: torch.sum(a[0], -1),
            lambda: torch.compile(lambda x: torch.sum(x, -1))),
    "softmax_two_pass": (softmax_two_pass, build_x,
                         lambda a: torch.nn.functional.softmax(a[0], 1),
                         lambda: torch.compile(lambda x: torch.nn.functional.softmax(x, 1))),
    "cross_entropy": (cross_entropy, build_ce,
                      lambda a: torch.nn.functional.cross_entropy(*a),
                      lambda: torch.compile(torch.nn.functional.cross_entropy)),
}

CASES = [
    ("rms_norm", (32768, 256)),
    ("sum", (8192, 256)),
    ("sum", (32768, 256)),
    ("softmax_two_pass", (32768, 256)),
    ("cross_entropy", (4096, 4096)),
    # large-rnumel controls (low warps must NOT win)
    ("rms_norm", (2048, 16384)),
    ("rms_norm", (8192, 8192)),
]
WARPS = [1, 2, 4, 8, 16, 32]


def correct(out, ref):
    o = out[0] if isinstance(out, tuple) else out
    o = o.float(); r = ref.float()
    return bool(o.shape == r.shape and torch.allclose(o, r, rtol=1e-3, atol=1e-3))


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  (floor M-block, vary num_warps)\n")
    for name, shape in CASES:
        fn, build, reffn, tcfn = REG[name]
        args = build(shape)
        ref = reffn(args)
        torch._dynamo.reset()
        tc = tcfn(); _ = tc(*args)
        tc_lat = med(lambda: tc(*args))
        bound0 = fn.bind(args)
        seed = dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])
        seedW = seed["num_warps"]
        print(f"=== {name} {shape} (rnumel={shape[1]}) tc={tc_lat*1000:.2f}us "
              f"seed_warps={seedW} mblock={seed['block_sizes'][0]} ===")
        row = "   "
        best = (None, 0.0)
        cells = []
        for w in WARPS:
            cfg = dict(seed); cfg["num_warps"] = w
            try:
                k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
                b = k.bind(args); b.ensure_config_exists(args)
                out = b(*args); ok = correct(out, ref)
                lat = med(lambda: b(*args)); g = tc_lat / lat
                tag = "*" if w == seedW else " "
                cells.append(f"w{w}{tag}={g:.3f}")
                if ok and g > best[1]:
                    best = (w, g)
            except Exception:
                cells.append(f"w{w}=ERR")
        print("   " + "  ".join(cells))
        print(f"   BEST warps={best[0]} G={best[1]:.3f}   (* = seed warps)\n")


if __name__ == "__main__":
    main()
