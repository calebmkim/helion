"""v3 branch trace: which branch fires for in-sample + held-out shapes.

v3 has ONLY two branches:
  - persistent  (rnumel <= backend.max_tensor_numel) -- the workhorse
  - looped      (rnumel >  backend.max_tensor_numel) -- structural tail only
The grid-occupancy branch and the byte-fence are DELETED.

Prints the seed config + branch + rnumel(elems)/bytes + m_extent + the
persistent num_warps regime (num_load gate) so the auditor can verify:
  * every in-sample shape is PERSISTENT,
  * num_load==1 huge rows get w32, num_load==2 (rms_norm) never does,
  * looped only fires above the structural cap (no in-sample coverage).
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
from helion._compiler.autotuner_heuristics.triton import TritonReductionHeuristic as H  # noqa: E402

EPS = 1e-5


def build(kernel, m, n):
    if kernel == "rms_norm":
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        return rms_norm_fwd, (x, w, EPS)
    if kernel == "sum":
        return sum_kernel, (torch.randn(m, n, device="cuda", dtype=torch.float32),)
    return longsum, (torch.randn(m, n, device="cuda", dtype=torch.float32),)


CASES = [
    # in-sample rms_norm (max rnumel 16384) -- all persistent, num_load=2 ramp
    ("rms_norm", 2048, 16384), ("rms_norm", 32768, 256), ("rms_norm", 8192, 8192),
    # in-sample sum (max rnumel 16384) -- persistent
    ("sum", 2048, 16384), ("sum", 8192, 256),
    # in-sample long_sum (rnumel 32768..262144, all <= 2**20) -- ALL persistent/w32
    ("long_sum", 1, 32768), ("long_sum", 2, 65536), ("long_sum", 4, 130000),
    ("long_sum", 8, 131072), ("long_sum", 16, 262144),
    # held-out long_sum: at/below cap -> persistent; above cap -> looped tail
    ("long_sum", 4, 262143), ("long_sum", 1, 1048576),  # 1048576 elems = cap -> persist
    ("long_sum", 1, 2097152),  # > cap -> looped tail (no in-sample coverage)
]


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  helion={helion.__file__}")
    hdr = (f"{'kernel':>10} {'shape':>15} {'rnumel':>9} {'rnumelB':>9} {'cap':>9} "
           f"{'numload':>7} {'m_ext':>7} {'seed':>34} {'branch':>12}")
    print(hdr)
    print("-" * len(hdr))
    for kernel, m, n in CASES:
        fn, args = build(kernel, m, n)
        bound = fn.bind(args)
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        if not seeds:
            print(f"{kernel:>10} {str((m,n)):>15}  NO SEED")
            continue
        fact = bound.env.config_spec.reduction_facts[0]
        cap = bound.env.backend.max_tensor_numel
        m_extent = H._m_extent(bound.env, fact)
        seed = dict(seeds[0])
        rl = seed.get("reduction_loops")
        branch = "persistent" if rl == [None] else "looped(cap)"
        seedstr = f"rl={rl},w{seed['num_warps']},bs={seed['block_sizes']}"
        print(f"{kernel:>10} {str((m,n)):>15} {fact.size_hint:>9} "
              f"{fact.size_hint*fact.itemsize:>9} {str(cap):>9} {fact.num_load:>7} "
              f"{m_extent:>7} {seedstr:>34} {branch:>12}")


if __name__ == "__main__":
    main()
