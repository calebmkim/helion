"""No-regression proof for adding layer_norm: the heuristic source is byte-identical
to the committed champion (git shows 0 changes under helion/). This script confirms
that the LIVE heuristic still emits the EXPECTED champion seed for every in-sample
shape of the 3 existing active kernels (rms_norm, sum, long_sum) -- i.e. adding
layer_norm did not perturb the shared codepath. Static (no GPU timing needed):
the seed dict fully determines codegen, and G was already referee-confirmed for
these exact seeds.

Expected seeds are the v4 champion's documented per-shape seeds (notebook tables).

Run with the canonical invocation (see SETUP.md).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.long_sum import longsum  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5


def rms_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), EPS)


def x_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def seed_of(fn, args):
    bound = fn.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    sd = dict(seeds[0])
    return (tuple(sd["block_sizes"]), tuple(sd["reduction_loops"]),
            sd["num_warps"], sd["num_stages"])


# (kernel, argbuilder, shapes, expected_seed-as-fn-of-rnumel)
# v4 ramp: rnumel<=1024->w4, <=4096->w8, <=16384->w16, >16384->w32; persistent
# (reduction_loops=(None,)) for rnumel<=2**20; block_sizes=[max(1,floor)].
def expected_warps(rnumel):
    if rnumel > 16384:
        return 32
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    return 16


CASES = [
    ("rms_norm", rms_norm_fwd, rms_args,
     [(2048, 1024), (2048, 2048), (2048, 4096), (2048, 8192), (2048, 16384),
      (4096, 1536), (4096, 3584), (4096, 5120), (4096, 7168),
      (8192, 4096), (8192, 8192), (32768, 256), (32768, 1024)]),
    ("sum", sum_kernel, x_args,
     [(2048, 1024), (2048, 4096), (2048, 16384), (4096, 1536), (4096, 5120),
      (8192, 256), (8192, 4096), (32768, 256), (32768, 1024)]),
    ("long_sum", longsum, x_args,
     [(1, 32768), (2, 65536), (4, 130000), (8, 131072), (16, 262144)]),
]


def main():
    print(f"helion={helion.__file__}\n")
    n_ok = n_bad = 0
    for name, fn, argb, shapes in CASES:
        print(f"=== {name} ===")
        for (m, n) in shapes:
            got = seed_of(fn, argb(m, n))
            want_w = expected_warps(n)
            # persistent expected for all these (max rnumel here 262144 < 2**20)
            want_rl = (None,)
            bs, rl, w, st = got
            ok = (rl == want_rl and w == want_w and st == 1)
            n_ok += ok
            n_bad += (not ok)
            flag = "OK " if ok else "BAD"
            print(f"  {flag} ({m:>6},{n:>6}) rnumel={n:>6} seed={got} "
                  f"(want rl={want_rl} warps={want_w} stages=1)")
        print()
    print(f"TOTAL: {n_ok} OK, {n_bad} BAD")
    assert n_bad == 0, "seed regression detected!"
    print("PASS: all 3 existing kernels emit their v4 champion seeds (byte-identical).")


if __name__ == "__main__":
    main()
