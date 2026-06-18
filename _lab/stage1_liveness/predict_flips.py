"""Predict which standard-track curriculum cells flip persistent<->looped under
footprint_factor = peak_live_tiles, using the cap formulas from triton.py. peak_live is a
per-kernel graph-structure constant (shape-independent; confirmed by probe_live_tiles).

This de-risks Part B BEFORE building the fact: it tells us exactly which cells the
config-recorder diff will flag, so we know the Gate-R sweep scope up front.
"""

from __future__ import annotations

import os
import sys

_WT = "/home/calebkim/helion-new-heuristics/helion-3stage"
sys.path.insert(0, os.path.join(_WT, "_lab", "prompts"))
import shapes_v3_draft as SH  # noqa: E402

ROW_PERSIST_MAX_BYTES = 245760
FULL_WIDTH_PERSIST_MAX_ELEMS = 81920
LOOPED_CHUNK = 16384
ELEMENT_CAP = 1048576  # max_tensor_numel (approx; sh well under it for all curric)


def npow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


def ppow2(x):
    p = 1
    while p * 2 <= x:
        p <<= 1
    return p


# Standard-track kernels only (Part B touches standard footprint_factor). peak_live from
# probe_live_tiles.py (MAX_PEAK_LIVE). full_width + itemsize from the ReductionFact dump.
# itemsize is the fp32-promoted reduction-input size: 4 for the norm/softmax family, 2 for
# the bf16-streamed sum/long_sum/cross_entropy. We sweep both bf16/fp32 input dtype; the
# fp32-promoted families keep itemsize=4 at both, the streamed ones use the input itemsize.
STD = {
    # kernel: (peak_live, full_width, itemsize_bf16, itemsize_fp32)
    "rms_norm": (3, True, 4, 4),
    "layer_norm": (3, True, 4, 4),
    "sum": (2, False, 2, 4),
    "long_sum": (2, False, 2, 4),
    "cross_entropy": (2, False, 2, 4),
}


def decide(sh, m, itemsize, full_width, ff):
    rdim = npow2(sh)
    can_persist = (
        sh <= ELEMENT_CAP
        and m * sh * itemsize * ff <= ROW_PERSIST_MAX_BYTES
        and (not full_width or m * sh * ff <= FULL_WIDTH_PERSIST_MAX_ELEMS)
    )
    if can_persist:
        return ("persist", rdim)
    budget = ROW_PERSIST_MAX_BYTES // (m * itemsize * ff)
    return ("loop", max(1, min(LOOPED_CHUNK, ppow2(budget))))


def main():
    for kn, (peak, fw, isz_bf16, isz_fp32) in STD.items():
        flips = []
        for split in ("train", "val", "test", "robustness"):
            for (m, n) in SH.SHAPES.get(kn, {}).get(split, []):
                for dt, isz in (("bf16", isz_bf16), ("fp32", isz_fp32)):
                    before = decide(n, 1, isz, fw, 1)
                    after = decide(n, 1, isz, fw, peak)
                    if before != after:
                        flips.append((split, dt, m, n, before, after))
        print(f"=== {kn} peak_live={peak} fw={fw}: {len(flips)} cells flip ===")
        for split, dt, m, n, b, a in flips:
            print(f"   {split:11} {dt:4} ({m},{n}): {b} -> {a}")


if __name__ == "__main__":
    main()
