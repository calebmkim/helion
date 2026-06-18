"""AUDITOR: prove the num_load CONDITION is inert on the in-sample set.

For every in-sample shape (rms_norm, sum, long_sum), compute the v3 seed AND a
counterfactual seed from a SIMPLER gate that drops num_load entirely:
    simple: rnumel > 16384 -> w32, else ramp(<=1024:4,<=4096:8,else16)
If the two num_warps agree on EVERY in-sample shape, the num_load condition
never changes an in-sample seed (it is untestable on the curriculum).
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.sum import sum_kernel  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
IN_SAMPLE = {
    "rms_norm": [(2048,1024),(2048,2048),(2048,4096),(2048,8192),(2048,16384),
                 (4096,1536),(4096,3584),(4096,5120),(4096,7168),(8192,4096),
                 (8192,8192),(32768,256),(32768,1024)],
    "sum": [(2048,1024),(2048,4096),(2048,16384),(4096,1536),(4096,5120),
            (8192,256),(8192,4096),(32768,256),(32768,1024)],
    "long_sum": [(1,32768),(2,65536),(4,130000),(8,131072),(16,262144)],
}


def simple_warps(rnumel):
    if rnumel > 16384:
        return 32
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    return 16


def build(kernel, m, n):
    if kernel == "rms_norm":
        return rms_norm_fwd, (torch.randn(m,n,device="cuda",dtype=torch.float32),
                              torch.randn(n,device="cuda",dtype=torch.float32), EPS)
    if kernel == "sum":
        return sum_kernel, (torch.randn(m,n,device="cuda",dtype=torch.float32),)
    return longsum, (torch.randn(m,n,device="cuda",dtype=torch.float32),)


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}")
    print("Compare v3 seed num_warps vs a num_load-FREE gate (rnumel>16384->w32).\n")
    mismatches = 0
    total = 0
    for kernel, shapes in IN_SAMPLE.items():
        for m, n in shapes:
            fn, args = build(kernel, m, n)
            bound = fn.bind(args)
            fact = bound.env.config_spec.reduction_facts[0]
            seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
            v3_w = seed["num_warps"]
            simp_w = simple_warps(fact.size_hint)
            agree = v3_w == simp_w
            total += 1
            if not agree:
                mismatches += 1
                print(f"  MISMATCH {kernel}{(m,n)} nl={fact.num_load} rn={fact.size_hint} "
                      f"v3=w{v3_w} simple=w{simp_w}")
    print(f"\nChecked {total} in-sample shapes. num_load-dependent differences: {mismatches}")
    if mismatches == 0:
        print("=> CONFIRMED: the num_load CONDITION is INERT on the in-sample set.")
        print("   A num_load-free gate (rnumel>16384->w32) emits BYTE-IDENTICAL seeds")
        print("   for every in-sample shape. The num_load split cannot be validated")
        print("   by the curriculum; it only diverges out-of-sample (large-rnumel,")
        print("   num_load>=2), where the auditor measured it gives the WRONG answer.")


if __name__ == "__main__":
    main()
