"""SPOT-CHECK the one sub-1.0 regime (2048,16384) G_seed~0.917 for max/min.
Is it a real heuristic gap (a reachable Helion config beats the seed) or
at-ceiling (torch.compile just wins; no Helion lever recovers it)?

Sweep the reachable levers the heuristic could pick at this shape:
  - num_warps in {4,8,16,32} on the persistent seed (the seed picks w16)
  - looped fallback (reduction_loops=[chunk]) at a few chunks
and compare to torch.compile. Also run sum_kernel at the same shape as the
IN-SAMPLE control: if sum is also ~0.92 here, the regime is at-ceiling for the
heuristic's lever set, not a new-op gap.

Run with the canonical invocation (SETUP.md).
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

from _lab.harness.fixture_maxmin import max_kernel  # noqa: E402
from _lab.harness.fixture_maxmin import min_kernel  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

SHAPE = (2048, 16384)
N_RUNS = 7


def med(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2] * 1000  # us


def build(kern, cfg, args):
    k = helion.kernel(kern.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b


def seed_of(kern, args):
    bound = kern.bind(args)
    return dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])


def run_one(name, kern, ref_fn):
    m, n = SHAPE
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    args = (x,)
    ref = ref_fn(x)
    sd = seed_of(kern, args)
    print(f"### {name}  seed={ {k: sd[k] for k in ('block_sizes','reduction_loops','num_warps')} }")

    torch._dynamo.reset()
    tc = torch.compile(ref_fn)
    tc(x)
    tc_us = med(lambda: tc(x))
    print(f"   torch.compile = {tc_us:8.1f} us")

    # seed
    bs = build(kern, sd, args)
    seed_us = med(lambda: bs(x))
    print(f"   SEED          = {seed_us:8.1f} us  G={tc_us/seed_us:.3f}")

    # warp sweep on the persistent config
    for w in (4, 8, 16, 32):
        cfg = dict(sd); cfg["num_warps"] = w
        try:
            b = build(kern, cfg, args)
            us = med(lambda: b(x))
            tag = " <- seed" if w == sd["num_warps"] else ""
            print(f"   persist w={w:<2}    = {us:8.1f} us  G={tc_us/us:.3f}{tag}")
        except Exception as e:  # noqa: BLE001
            print(f"   persist w={w:<2}    FAILED: {type(e).__name__}")

    # looped fallback at a couple chunks
    for chunk in (2048, 4096, 8192):
        cfg = dict(sd); cfg["reduction_loops"] = [chunk]; cfg["num_warps"] = 32
        try:
            b = build(kern, cfg, args)
            us = med(lambda: b(x))
            print(f"   loop {chunk:<5} w32 = {us:8.1f} us  G={tc_us/us:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"   loop {chunk:<5} w32 FAILED: {type(e).__name__}")
    print()


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  helion={helion.__file__}")
    print(f"SHAPE={SHAPE} (the sub-1.0 regime), fp32\n")
    run_one("max", max_kernel, lambda x: torch.amax(x, dim=-1))
    run_one("min", min_kernel, lambda x: torch.amin(x, dim=-1))
    run_one("sum (IN-SAMPLE control)", sum_kernel, lambda x: x.sum(-1))


if __name__ == "__main__":
    main()
