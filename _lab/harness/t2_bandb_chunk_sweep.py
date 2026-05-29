"""Band-B sub-cap R_BLOCK chunk sweep for kl_div / jsd at the wide rows where
persistent spills. Find the best looped R_BLOCK chunk (+ warps) so we can pick a
single generalizable Band-B chunk. M_BLOCK pinned at floor. fp32.
"""

from __future__ import annotations

import argparse
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

N_RUNS = 5


def median_do_bench(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def setup(kernel):
    if kernel == "kl_div":
        from examples.kl_div import kl_div_forward as fn

        def inputs(BT, V):
            yp = torch.randn(BT, V, device="cuda").log_softmax(-1)
            yt = torch.randn(BT, V, device="cuda").softmax(-1)
            return (yp, yt, False, "batchmean", 1e-10)
        red_idx = 0  # block_sizes=[V, M]
        return fn, inputs, red_idx, [(4096, 65536), (4096, 131072)]
    else:
        from examples.jsd import jsd_forward as fn

        def inputs(BT, V):
            lq = torch.randn(BT, V, device="cuda").log_softmax(-1)
            lp = torch.randn(BT, V, device="cuda").log_softmax(-1)
            return (lq, lp, None, 0.5, -100)
        red_idx = 0
        return fn, inputs, red_idx, [(8192, 32768), (8192, 65536)]


def build(fn, R, w, red_idx, args):
    bs = [1, 1]
    bs[red_idx] = R
    bs[1 - red_idx] = 1
    seed = helion.Config(block_sizes=bs, num_warps=w, num_stages=1)
    k = helion.kernel(fn.fn, configs=[seed]); b = k.bind(args); b.ensure_config_exists(args)
    return b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kernel", required=True)
    a = ap.parse_args()
    fn, inputs, red_idx, shapes = setup(a.kernel)
    for (BT, V) in shapes:
        args = inputs(BT, V)
        print(f"\n=== {a.kernel} ({BT},{V}) V*4={V*4//1024}KiB ===")
        for R in [4096, 8192, 16384, 32768]:
            if R > V:
                continue
            for w in [16, 32]:
                try:
                    b = build(fn, R, w, red_idx, args)
                    lat = median_do_bench(lambda: b(*args))
                    print(f"   R={R:>6} (R*4={R*4//1024}KiB) w={w:>2}: {lat*1000:>9.1f} us")
                except Exception as e:  # noqa: BLE001
                    print(f"   R={R:>6} w={w}: FAIL {type(e).__name__}")


if __name__ == "__main__":
    main()
