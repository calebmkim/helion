"""GENERALITY STRESS-TEST: confirm the DECLINE path for jagged reductions.

jagged_mean + jagged_softmax classify with 0 reduction_facts / 0 seeds (the v5
dynamic-size guard declines a jagged_tile axis whose extent is data-dependent).
This is out-of-scope BY DESIGN, not a heuristic gap. Confirm:
  (1) v7 emits 0 seeds (heuristics_used=[]) -> falls back to default config;
  (2) the default-config kernel is CORRECT vs the PyTorch reference.

Run with the canonical invocation (SETUP.md).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.jagged_mean import jagged_mean_kernel  # noqa: E402
from examples.jagged_mean import reference_jagged_mean_kernel_pytorch  # noqa: E402
from examples.jagged_softmax import jagged_softmax_kernel  # noqa: E402
from examples.jagged_softmax import reference_jagged_softmax_pytorch  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402


def jagged_offsets(num_rows, max_cols):
    lengths = torch.randint(1, max_cols + 1, (num_rows,), device="cuda")
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device="cuda"), torch.cumsum(lengths, dim=0)]
    )
    return x_offsets, int(x_offsets[-1])


def _default_run(kern, args):
    """Run with the deterministic default_config (configs=[cfg], NO autotune) --
    the runtime 'default fallback' for a declined (0-seed) kernel."""
    b0 = kern.bind(args)
    cfg = b0.config_spec.default_config()
    k = helion.kernel(kern.fn, configs=[cfg])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b(*args)


def check_jagged_mean():
    num_rows, max_cols, M = 512, 64, 8
    x_offsets, nnz = jagged_offsets(num_rows, max_cols)
    x_data = torch.randn(nnz, M, dtype=torch.float32, device="cuda")
    fc = torch.randint(1, M + 1, (num_rows,), dtype=torch.int32, device="cuda")
    args = (x_data, x_offsets, fc, M)
    bound = jagged_mean_kernel.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    out = _default_run(jagged_mean_kernel, args)
    ref = reference_jagged_mean_kernel_pytorch(*args)
    ok = bool(torch.allclose(out, ref, rtol=1e-3, atol=1e-3))
    print(f"jagged_mean: n_seeds={len(seeds)} (expect 0)  correct(default)={ok}  "
          f"maxabs={float((out-ref).abs().max()):.2e}")


def check_jagged_softmax():
    num_rows, max_cols, M = 512, 64, 128
    x_offsets, nnz = jagged_offsets(num_rows, max_cols)
    x_data = torch.randn(nnz, M, dtype=torch.float32, device="cuda")
    args = (x_data, x_offsets)
    bound = jagged_softmax_kernel.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    out = _default_run(jagged_softmax_kernel, args)
    ref = reference_jagged_softmax_pytorch(x_data, x_offsets)
    ok = bool(torch.allclose(out, ref, rtol=1e-3, atol=1e-3))
    print(f"jagged_softmax: n_seeds={len(seeds)} (expect 0)  correct(default)={ok}  "
          f"maxabs={float((out-ref).abs().max()):.2e}")


def main():
    print(f"helion={helion.__file__}\n")
    check_jagged_mean()
    check_jagged_softmax()


if __name__ == "__main__":
    main()
