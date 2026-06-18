"""Check that a Band-B sub-cap R_BLOCK=4096 does NOT regress the NARROW Band-B
rows (where the full-N persistent seed was best/tied). Compare persistent (full-N
R_BLOCK) vs capped R_BLOCK=4096, both at the rnumel-ramp warps, M_BLOCK floor.
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
from helion._utils import next_power_of_2  # noqa: E402

N_RUNS = 7


def median_do_bench(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def warps(rnumel):
    return 32 if rnumel > 16384 else (4 if rnumel <= 1024 else (8 if rnumel <= 4096 else 16))


def setup(kernel):
    if kernel == "kl_div":
        from examples.kl_div import kl_div_forward as fn

        def inputs(BT, V):
            return (torch.randn(BT, V, device="cuda").log_softmax(-1),
                    torch.randn(BT, V, device="cuda").softmax(-1),
                    False, "batchmean", 1e-10)
        return fn, inputs, 0, [(4096, 4096), (4096, 8192), (4096, 16384), (4096, 32768)]
    else:
        from examples.jsd import jsd_forward as fn

        def inputs(BT, V):
            return (torch.randn(BT, V, device="cuda").log_softmax(-1),
                    torch.randn(BT, V, device="cuda").log_softmax(-1),
                    None, 0.5, -100)
        return fn, inputs, 0, [(8192, 4096), (8192, 8192), (8192, 16384)]


def build(fn, R, w, red_idx, args):
    bs = [1, 1]; bs[red_idx] = R; bs[1 - red_idx] = 1
    seed = helion.Config(block_sizes=bs, num_warps=w, num_stages=1)
    k = helion.kernel(fn.fn, configs=[seed]); b = k.bind(args); b.ensure_config_exists(args)
    return b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kernel", required=True)
    a = ap.parse_args()
    fn, inputs, red_idx, shapes = setup(a.kernel)
    print(f"{a.kernel}: narrow Band-B rows, persistent(full-N) vs capped R=4096")
    for (BT, V) in shapes:
        args = inputs(BT, V)
        w = warps(V)
        full_R = next_power_of_2(V)
        bp = build(fn, full_R, w, red_idx, args)
        lat_p = median_do_bench(lambda: bp(*args))
        cap_R = min(full_R, 4096)
        bc = build(fn, cap_R, w, red_idx, args)
        lat_c = median_do_bench(lambda: bc(*args))
        ratio = lat_p / lat_c  # >1 => capped faster
        print(f"  ({BT},{V}) w={w}: persist(R={full_R})={lat_p*1000:>8.1f}us  "
              f"cap(R={cap_R})={lat_c*1000:>8.1f}us  persist/cap={ratio:.3f}")


if __name__ == "__main__":
    main()
