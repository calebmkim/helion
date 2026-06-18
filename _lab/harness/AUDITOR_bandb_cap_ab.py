"""AUDITOR matched A/B + cap sweep for the Band-B R_BLOCK cap.

For kl_div / jsd at wide in-sample rows: directly construct configs (bypass the
heuristic) with R_BLOCK = full-N persistent (next_pow2(V)) and caps
{4096, 8192, 16384, 32768, 65536}, M_BLOCK at floor=1, warps HELD EQUAL.
Reports latency and the ratio to the 16KiB cap (4096 fp32 elems) so we can see if
16KiB is near-optimal or cherry-picked, and whether full-N genuinely spills.
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
from helion._utils import next_power_of_2 as np2  # noqa: E402

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
        return fn, inputs, 0
    else:
        from examples.jsd import jsd_forward as fn

        def inputs(BT, V):
            lq = torch.randn(BT, V, device="cuda").log_softmax(-1)
            lp = torch.randn(BT, V, device="cuda").log_softmax(-1)
            return (lq, lp, None, 0.5, -100)
        return fn, inputs, 0


def build(fn, R, w, red_idx, args):
    bs = [1, 1]
    bs[red_idx] = R
    bs[1 - red_idx] = 1
    seed = helion.Config(block_sizes=bs, num_warps=w, num_stages=1)
    k = helion.kernel(fn.fn, configs=[seed])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--shapes", required=True, help="semicolon list BT,V")
    ap.add_argument("--warps", type=int, default=32)
    a = ap.parse_args()
    fn, inputs, red_idx = setup(a.kernel)
    shapes = []
    for tok in a.shapes.split(";"):
        bt, v = tok.split(",")
        shapes.append((int(bt), int(v)))

    for (BT, V) in shapes:
        args = inputs(BT, V)
        fullN = np2(V)
        caps = [4096, 8192, 16384, 32768, 65536, fullN]
        caps = sorted(set(c for c in caps if c <= fullN))
        print(f"\n=== {a.kernel} ({BT},{V}) V*4={V*4//1024}KiB  fullN_pow2={fullN} w={a.warps} ===")
        results = {}
        for R in caps:
            try:
                b = build(fn, R, a.warps, red_idx, args)
                lat = median_do_bench(lambda: b(*args)) * 1000
                results[R] = lat
                tag = " [16KiB cap]" if R == 4096 else (" [full-N persistent]" if R == fullN else "")
                print(f"   R={R:>7} (R*4={R*4//1024:>6}KiB) w={a.warps}: {lat:>10.1f} us{tag}")
            except Exception as e:  # noqa: BLE001
                print(f"   R={R:>7} w={a.warps}: FAIL {type(e).__name__}: {str(e)[:80]}")
        if 4096 in results and results:
            base = results[4096]
            best = min(results.values())
            best_R = [k for k, v in results.items() if v == best][0]
            print(f"   -- 16KiB-cap={base:.1f}us  best={best:.1f}us @R={best_R} "
                  f"(16KiB/best={base/best:.3f})  fullN/16KiB="
                  f"{results.get(fullN, float('nan'))/base:.2f}x")


if __name__ == "__main__":
    main()
