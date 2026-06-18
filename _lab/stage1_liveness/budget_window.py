"""Find the LIVE_BUDGET window: persistent iff peak_live * m * sh * itemsize <= LIVE_BUDGET.
Constraints:
  - every CURRENTLY-persistent standard curriculum cell must STAY persistent (no regression):
        max over those of (peak_live * sh * itemsize)  <=  LIVE_BUDGET
  - flj narrow-V must FLIP to looped:
        LIVE_BUDGET  <  min over flj-narrow-V targets of (peak_live * sh * itemsize)
peak_live (graph-structure constant, from probe_live_tiles MAX_PEAK_LIVE) and itemsize
(ReductionFact.itemsize: 4 for fp32-promoted norm/softmax/loss family incl flj; streamed
sum/long_sum/cross_entropy use input itemsize = 2 bf16 / 4 fp32).
"""
from __future__ import annotations

import os
import sys

_WT = "/home/calebkim/helion-new-heuristics/helion-3stage"
sys.path.insert(0, os.path.join(_WT, "_lab", "prompts"))
import shapes_v3_draft as SH  # noqa: E402

CAP = 245760
FW = 81920


def npow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


# standard-track kernels: (peak_live, full_width, itemsize_bf16, itemsize_fp32)
STD = {
    "rms_norm": (3, True, 4, 4),
    "layer_norm": (3, True, 4, 4),
    "sum": (2, False, 2, 4),
    "long_sum": (2, False, 2, 4),
    "cross_entropy": (2, False, 2, 4),
}


def cur_persist(sh, itemsize, fw):
    return (sh * itemsize <= CAP) and (not fw or sh <= FW)


def main():
    # max footprint over currently-persistent standard cells (these must stay persistent)
    worst = []
    for kn, (peak, fw, ib, ifp) in STD.items():
        kmax = 0
        kmax_cell = None
        for split in ("train", "val", "test", "robustness"):
            for (m, n) in SH.SHAPES.get(kn, {}).get(split, []):
                for dt, isz in (("bf16", ib), ("fp32", ifp)):
                    if cur_persist(n, isz, fw):
                        fp = peak * n * isz  # m_block=1 for all curriculum
                        if fp > kmax:
                            kmax, kmax_cell = fp, (split, dt, m, n)
        worst.append((kn, kmax, kmax_cell))
        print(f"{kn:14} peak={peak} max currently-persistent footprint = {kmax:>9}  at {kmax_cell}")
    keep_min = max(w[1] for w in worst)
    print(f"\n=> LIVE_BUDGET must be >= {keep_min} to keep all currently-persistent standard cells")

    # flj narrow-V targets (peak=7, itemsize=4 both dtypes): must FLIP (footprint > BUDGET)
    print("\nflj narrow-V footprints (peak=7, itemsize=4), must exceed BUDGET to flip:")
    flj_fps = []
    for (m, v) in [(4096, 32000), (8192, 32000), (4096, 50257), (2048, 50257)]:
        fp = 7 * v * 4
        flj_fps.append(fp)
        print(f"   flj({m},{v}): {fp}")
    flip_max = min(flj_fps)
    print(f"=> LIVE_BUDGET must be < {flip_max} to flip ALL flj narrow-V targets")

    print(f"\n*** WINDOW: [{keep_min}, {flip_max}) -> valid={keep_min < flip_max} ***")
    for cand in (393216, 458752, 524288, 655360, 786432):
        ok = keep_min <= cand < flip_max
        print(f"   candidate LIVE_BUDGET={cand}: keeps_all={cand>=keep_min} flips_flj={cand<flip_max} -> {'OK' if ok else 'no'}")


if __name__ == "__main__":
    main()
