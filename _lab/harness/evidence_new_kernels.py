"""Emit Step-0-style EVIDENCE BLOCKs (seed-used + correctness) for sum + long_sum.

Uses the canonical run_bare_seed harness with the HEURISTIC's seed (from
compiler_seed_configs) for a representative shape of each new kernel, proving:
no autotune ran, the seed was actually used (codegen persistent-vs-looped +
num_warps match), correctness vs torch.sum(x,-1), stable latency.
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
from examples.sum import sum_kernel  # noqa: E402

from _lab.harness.bare_seed_run import run_bare_seed  # noqa: E402
from _lab.harness.evidence_block import EvidenceFields  # noqa: E402
from _lab.harness.evidence_block import format_evidence_block  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402


def build_args(shape, dtype):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=dtype),)


def reference(x):
    return torch.sum(x, dim=-1)


def get_seed(fn, shape):
    args = build_args(shape, torch.float32)
    bound = fn.bind(args)
    return dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])


CASES = [
    ("sum", sum_kernel, (2048, 16384)),       # persistent, mid row
    ("sum", sum_kernel, (32768, 256)),        # persistent, large-M small-N
    ("long_sum", longsum, (1, 32768)),        # grid-starved -> looped/32
    ("long_sum", longsum, (8, 131072)),       # huge rnumel -> looped/32
]


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    for name, fn, shape in CASES:
        seed = get_seed(fn, shape)
        res = run_bare_seed(
            fn, build_args, reference, shape, seed,
            dtype=torch.float32, n_runs=7, rtol=1e-3, atol=1e-3,
        )
        cmd = ("cd /home/calebkim/helion-new-heuristics/wt-reduction && "
               "CUDA_VISIBLE_DEVICES=2 "
               "PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction "
               "/home/calebkim/.conda/envs/helion/bin/python "
               "_lab/harness/evidence_new_kernels.py")
        block = format_evidence_block(EvidenceFields(
            kernel_shape=f"{name} {shape} fp32",
            exact_command=cmd,
            seed_raw=res.seed_raw,
            seed_normalized=res.seed_normalized,
            seed_used=res.seed_used,
            seed_used_how=res.seed_used_how,
            autotune_ran=res.autotune_ran,
            autotune_evidence=res.autotune_evidence,
            correctness_pass=res.correctness_pass,
            max_abs=res.max_abs,
            max_rel=res.max_rel,
            rtol=res.rtol,
            atol=res.atol,
            tol_justification=("fp32 sum reduction-order drift; max_rel can be "
                               "large only on near-zero row sums (random normals), "
                               "covered by atol=1e-3"),
            latency_median_ms=res.latency_median_ms,
            latency_min_ms=res.latency_min_ms,
            latency_max_ms=res.latency_max_ms,
            latency_stddev_ms=res.latency_stddev_ms,
            n_runs=res.n_runs,
            gpu_index=gpu,
            accept_reject_rule=("heuristic seed runs with no autotune, used as-is "
                                "(codegen matches), correct, stably timed"),
        ))
        print(block)
        print()


if __name__ == "__main__":
    main()
