"""Perf-investigator probe: dump seed config + M-block floor for each small-N shape.

For each (kernel, shape) report:
  - the heuristic seed config
  - the autotuner M-block floor (what _m_block_size returns)
  - the M grid extent
  - whether the M-block could legally be raised (block size spec bounds)

This is the cheap structural probe before the expensive benchmark sweep.
"""

from __future__ import annotations

import sys
from typing import cast

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

LONG = torch.int64


def build_rms(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), 1e-5)


def build_sum(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def build_softmax(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def build_ce(shape):
    n, v = shape
    return (torch.randn(n, v, device="cuda", dtype=torch.float32),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG))


CASES = [
    ("rms_norm", rms_norm_fwd, build_rms, (32768, 256)),
    ("sum", sum_kernel, build_sum, (8192, 256)),
    ("sum", sum_kernel, build_sum, (32768, 256)),
    ("softmax_two_pass", softmax_two_pass, build_softmax, (32768, 256)),
    ("cross_entropy", cross_entropy, build_ce, (4096, 4096)),
    # large-N controls
    ("rms_norm", rms_norm_fwd, build_rms, (2048, 16384)),
    ("long_sum", None, None, (256, 131072)),  # placeholder
]


def main():
    from examples.long_sum import longsum
    for name, fn, build, shape in CASES:
        if name == "long_sum":
            fn = longsum
            build = lambda s: (torch.randn(*s, device="cuda", dtype=torch.float32),)
        args = build(shape)
        bound = fn.bind(args)
        spec = bound.env.config_spec
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        seed = dict(seeds[0]) if seeds else None
        fact = spec.reduction_facts[0] if spec.reduction_facts else None
        # M-block floor info per block_sizes entry
        bs_info = []
        for i in range(len(spec.block_sizes)):
            bsp = cast("object", spec.block_sizes[i])
            bs_info.append({
                "idx": i,
                "size_hint": bsp.size_hint,
                "min_size": bsp.min_size,
                "autotuner_min": bsp.autotuner_min,
                "floor": max(1, bsp.min_size, bsp.autotuner_min),
            })
        print(f"=== {name} {shape} ===")
        print(f"  seed = {seed}")
        if fact is not None:
            is_t1 = fact.block_id in spec.reduction_loops.valid_block_ids()
            print(f"  rnumel(size_hint)={fact.size_hint} num_load={fact.num_load} "
                  f"num_tiled_accumulators={getattr(fact,'num_tiled_accumulators','?')} "
                  f"itemsize={fact.itemsize} block_id={fact.block_id} track={'T1' if is_t1 else 'T2'}")
        for b in bs_info:
            print(f"  block_sizes[{b['idx']}]: size_hint={b['size_hint']} "
                  f"min_size={b['min_size']} autotuner_min={b['autotuner_min']} floor={b['floor']}")
        print()


if __name__ == "__main__":
    main()
