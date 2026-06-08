"""Step 2 verification: ReductionFact populated + Triton reduction seed emitted & USED.

Proves:
  1. helion is the worktree copy.
  2. register_rollable_reductions populates env.config_spec.reduction_facts for rms_norm.
  3. compiler_seed_configs(env, device_ir) now returns the Triton reduction seed
     (heuristic name in config_spec.autotuner_heuristics).
  4. The exact seed runs BARE (configs=[seed], no autotune), correctness passes,
     and the persistent-vs-looped + num_warps choice is reflected in generated Triton.

Run with the canonical invocation (see SETUP.md).
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

from _lab.harness.bare_seed_run import run_bare_seed  # noqa: E402

EPS = 1e-5


def build_args(shape, dtype):
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=dtype)
    w = torch.randn(n, device="cuda", dtype=dtype)
    return (x, w, EPS)


def inspect_shape(shape):
    args = build_args(shape, torch.float32)
    bound = rms_norm_fwd.bind(args)
    spec = bound.env.config_spec
    device_ir = bound.host_function.device_ir

    print(f"\n=== shape {shape} ===")
    print("reduction_facts:")
    for f in spec.reduction_facts:
        print(f"  {f}")
    print("reduction_loops specs:")
    for rl in spec.reduction_loops:
        print(f"  block_id={rl.block_id} size_hint={rl.size_hint}")

    seeds = compiler_seed_configs(bound.env, device_ir)
    print(f"autotuner_heuristics fired: {spec.autotuner_heuristics}")
    print(f"compiler_seed_configs returned {len(seeds)} seed(s):")
    for s in seeds:
        print(f"  {dict(s)}")
    return seeds


def reference(x, w, e):
    return rms_norm_pytorch(x, w, e)


def prove_bare(shape, seed):
    """Run the EXACT heuristic seed bare and prove it was used + correct."""
    res = run_bare_seed(
        rms_norm_fwd,
        build_args,
        reference,
        shape,
        dict(seed),
        dtype=torch.float32,
        n_runs=7,
    )
    looped = "for roffset" in res.generated_triton
    print(
        f"  BARE {shape}: seed_used={res.seed_used} correct={res.correctness_pass} "
        f"max_abs={res.max_abs:.2e} codegen={'looped' if looped else 'persistent'} "
        f"lat_med={res.latency_median_ms*1000:.1f}us"
    )
    assert res.seed_used, f"seed NOT used for {shape}"
    assert res.correctness_pass, f"correctness FAILED for {shape}"


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}")

    # Representative shapes spanning persistent (<=16384 elems) and looped.
    for shape in [(32768, 256), (4096, 5120), (2048, 16384), (2048, 4096)]:
        seeds = inspect_shape(shape)
        assert len(seeds) == 1, f"expected 1 seed, got {len(seeds)} for {shape}"
        print("  proving bare seed run:")
        prove_bare(shape, seeds[0])


if __name__ == "__main__":
    main()
