"""Probe: for each (kernel, shape) print the LIVE spec lengths the seed-time code
would see — spec.indexing.length, spec.load_eviction_policies.length,
spec.store_indices, fact.num_load/num_store — so the matched-lever A/B builds the
indexing/eviction lists at EXACTLY the right length (codegen_knob_map.md discipline).
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

LONG = torch.int64


def build_rms(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), 1e-5)


def build_x(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def build_ce(shape):
    n, v = shape
    return (torch.randn(n, v, device="cuda", dtype=torch.float32),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG))


CASES = [
    ("sum", (8192, 256), sum_kernel, build_x),
    ("rms_norm", (32768, 256), rms_norm_fwd, build_rms),
    ("softmax_two_pass", (32768, 256), softmax_two_pass, build_x),
    ("cross_entropy", (4096, 4096), cross_entropy, build_ce),
    ("rms_norm", (2048, 16384), rms_norm_fwd, build_rms),
    ("long_sum", (256, 131072), longsum, build_x),
]


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}\n")
    for name, shape, fn, build in CASES:
        args = build(shape)
        bound = fn.bind(args)
        spec = bound.env.config_spec
        fact = spec.reduction_facts[0]
        idx_choices = spec.indexing.choices if hasattr(spec.indexing, "choices") else "?"
        ev_choices = (spec.load_eviction_policies.choices
                      if hasattr(spec.load_eviction_policies, "choices") else "?")
        print(f"=== {name} {shape} ===")
        print(f"  indexing.length            = {spec.indexing.length}")
        print(f"  load_eviction_policies.len = {spec.load_eviction_policies.length}")
        print(f"  store_indices              = {spec.store_indices}")
        print(f"  fact.num_load/num_store    = {fact.num_load}/{fact.num_store}")
        print(f"  fact.size_hint(rnumel)     = {fact.size_hint}")
        print(f"  indexing default           = {spec.indexing.default()}")
        print(f"  eviction default           = {spec.load_eviction_policies.default()}")
        print(f"  indexing choices           = {idx_choices}")
        print(f"  eviction choices           = {ev_choices}")
        print()


if __name__ == "__main__":
    main()
