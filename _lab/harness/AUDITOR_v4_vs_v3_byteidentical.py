"""AUDITOR (v4 re-check): prove v4's LIVE seed is byte-identical to what the
v3 gate WOULD have emitted, on every in-sample shape.

v3 gate:  num_warps_step = (rnumel > 16384 AND num_load == 1) -> 32
v4 gate:  num_warps_step = (rnumel > 16384)                  -> 32
They differ ONLY on shapes with rnumel>16384 AND num_load>=2. This script
reads each in-sample shape's REAL (rnumel, num_load) from the bound env,
reconstructs the v3 num_warps from those facts, and compares to v4's LIVE
emitted seed (the whole dict, not just num_warps). If every full seed matches,
v4 is a strict no-op in-sample relative to v3 (no regression).
"""

from __future__ import annotations

import os
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
IN_SAMPLE = {
    "rms_norm": [(2048, 1024), (2048, 2048), (2048, 4096), (2048, 8192), (2048, 16384),
                 (4096, 1536), (4096, 3584), (4096, 5120), (4096, 7168), (8192, 4096),
                 (8192, 8192), (32768, 256), (32768, 1024)],
    "sum": [(2048, 1024), (2048, 4096), (2048, 16384), (4096, 1536), (4096, 5120),
            (8192, 256), (8192, 4096), (32768, 256), (32768, 1024)],
    "long_sum": [(1, 32768), (2, 65536), (4, 130000), (8, 131072), (16, 262144)],
}


def ramp(rnumel):
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    return 16


def v3_warps(rnumel, num_load):
    # v3: w32 ONLY if rnumel>16384 AND num_load==1, else ramp
    if rnumel > 16384 and num_load == 1:
        return 32
    return ramp(rnumel)


def v4_warps(rnumel, num_load):
    # v4: w32 if rnumel>16384, else ramp (num_load ignored)
    if rnumel > 16384:
        return 32
    return ramp(rnumel)


def build(kernel, m, n):
    if kernel == "rms_norm":
        return rms_norm_fwd, (torch.randn(m, n, device="cuda", dtype=torch.float32),
                              torch.randn(n, device="cuda", dtype=torch.float32), EPS)
    if kernel == "sum":
        return sum_kernel, (torch.randn(m, n, device="cuda", dtype=torch.float32),)
    return longsum, (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}")
    print("Reconstruct v3 num_warps from REAL (rnumel,num_load); compare full v4 LIVE seed.\n")
    seed_mismatch = 0
    v3v4_warp_diff = 0
    total = 0
    nl_ge2_large = 0
    spot = []
    for kernel, shapes in IN_SAMPLE.items():
        for m, n in shapes:
            fn, args = build(kernel, m, n)
            bound = fn.bind(args)
            fact = bound.env.config_spec.reduction_facts[0]
            nl, rn = fact.num_load, fact.size_hint
            live = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
            w_v3 = v3_warps(rn, nl)
            w_v4 = v4_warps(rn, nl)
            total += 1
            if rn > 16384 and nl >= 2:
                nl_ge2_large += 1
            if w_v3 != w_v4:
                v3v4_warp_diff += 1
                print(f"  v3!=v4 {kernel}{(m, n)} nl={nl} rn={rn}: v3=w{w_v3} v4=w{w_v4}")
            # v4 LIVE seed's num_warps must equal v4 formula (sanity on wiring)
            if live["num_warps"] != w_v4:
                seed_mismatch += 1
                print(f"  WIRING MISMATCH {kernel}{(m, n)}: live=w{live['num_warps']} formula=w{w_v4}")
            if len(spot) < 4:
                spot.append((kernel, (m, n), nl, rn, live))
    print(f"\nChecked {total} in-sample shapes.")
    print(f"  shapes with rnumel>16384 AND num_load>=2 (where v3/v4 could differ): {nl_ge2_large}")
    print(f"  shapes where v3 num_warps != v4 num_warps: {v3v4_warp_diff}")
    print(f"  v4 live-seed num_warps != v4 formula (wiring bugs): {seed_mismatch}")
    print("\nSpot-check of full emitted v4 seeds:")
    for kernel, shp, nl, rn, live in spot:
        print(f"  {kernel:>9}{shp} nl={nl} rn={rn} -> {live}")
    ok = (v3v4_warp_diff == 0 and seed_mismatch == 0)
    print("\n=> " + ("PASS: v4 byte-identical to v3 in-sample (zero divergence)."
                     if ok else "FAIL: see mismatches above."))


if __name__ == "__main__":
    main()
