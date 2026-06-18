"""AUDITOR (v4): confirm the rnumel-only gate creates NO new cliff and the
small-rnumel guard holds.

(1) Small-rnumel guard: (32768,256) rnumel=256 must stay w4 (NOT w32). The
    tiny-row w32 catastrophe is excluded by `>16384`, not by num_load.
(2) Sane warps across the ramp + breakpoint: print the LIVE v4 seed num_warps
    for rnumel 256 .. 32768 and confirm the ramp is monotone 4->8->16->32 with
    the only step at >16384 (no double cliff, no skipped rung).
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
# (kernel, m, n, expected_warps) spanning every ramp rung + the guard case
CASES = [
    ("sum", 32768, 256, 4),     # small-rnumel guard: must stay w4, NOT w32
    ("rms_norm", 32768, 256, 4),  # multi-load small row also stays w4
    ("sum", 2048, 1024, 4),
    ("sum", 2048, 2048, 8),
    ("sum", 2048, 4096, 8),
    ("sum", 2048, 8192, 16),
    ("sum", 2048, 16384, 16),   # exactly at breakpoint -> still w16 (strict >)
    ("sum", 2048, 16385, 32),   # one past -> w32
    ("sum", 1, 32768, 32),
]


def build(kernel, m, n):
    if kernel == "rms_norm":
        return rms_norm_fwd, (torch.randn(m, n, device="cuda", dtype=torch.float32),
                              torch.randn(n, device="cuda", dtype=torch.float32), EPS)
    return sum_kernel, (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}")
    print("LIVE v4 seed num_warps across the ramp + the small-rnumel guard.\n")
    fails = 0
    for kernel, m, n, exp in CASES:
        fn, args = build(kernel, m, n)
        b = fn.bind(args)
        f = b.env.config_spec.reduction_facts[0]
        s = dict(compiler_seed_configs(b.env, b.host_function.device_ir)[0])
        w = s["num_warps"]
        ok = (w == exp)
        if not ok:
            fails += 1
        tag = "OK" if ok else f"FAIL exp w{exp}"
        print(f"  {kernel:>9}{(m, n)} nl={f.num_load} rn={f.size_hint:>6} "
              f"-> w{w:<2}  [{tag}]")
    print(f"\n=> " + ("PASS: ramp monotone, single step at >16384, "
                      "small-rnumel guard holds (no new fence)."
                      if fails == 0 else f"FAIL: {fails} mismatches."))


if __name__ == "__main__":
    main()
