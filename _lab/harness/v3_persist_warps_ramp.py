"""STEP A part (b): best num_warps for the PERSISTENT path, across rnumel.

Establishes the persistent num_warps ramp breakpoints. Critical safety check:
the warps=32 breakpoint must sit ABOVE rms_norm/sum's max in-sample rnumel
(16384 elems = 64KiB) so widening the ramp to 32 does NOT change them.

Sweep rnumel from the rms_norm/sum regime (1024..16384) up through the huge
long_sum regime (32768..1048576), at a few M, persistent path only, warps
{4,8,16,32}. Print best warp + the per-warp times so breakpoints are visible.
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

from examples.sum import sum_kernel  # noqa: E402

N_RUNS = 9
RNUMELS = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
MS = [1, 16, 256]
WARPS = [4, 8, 16, 32]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def time_cfg(x, ref, warps):
    cfg = helion.Config(block_sizes=[1], reduction_loops=[None],
                        num_warps=warps, num_stages=1)
    k = helion.kernel(sum_kernel.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3), "correctness"
    return med(lambda: b(x)) * 1000


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  N_RUNS={N_RUNS}  PERSISTENT path only\n")
    print(f"{'rnumel':>8} {'KiB':>5} {'M':>4} | " + " ".join(f"{'w'+str(w):>7}" for w in WARPS) + " | best")
    for n in RNUMELS:
        kib = n * 4 // 1024
        for m in MS:
            x = torch.randn(m, n, device="cuda", dtype=torch.float32)
            ref = x.sum(-1)
            t = {w: time_cfg(x, ref, w) for w in WARPS}
            best_w = min(t, key=t.get)
            print(f"{n:>8} {kib:>5} {m:>4} | " + " ".join(f"{t[w]:>7.1f}" for w in WARPS)
                  + f" | w{best_w}")
        print()


if __name__ == "__main__":
    main()
