"""Dump the emitted seed dict for all 8 active kernels (the v7-side dump). Writes a
deterministic repr to a file so we can diff v7-HEAD vs v6-parent (run twice across
a git checkout). This proves BYTE-IDENTICAL seeds, not just sc=False.
"""
from __future__ import annotations

import json
import sys
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward  # noqa: E402
from examples.cross_entropy import cross_entropy  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5


def rms_args(m, n):
    return (torch.randn(m, n, device="cuda"), torch.randn(n, device="cuda"), EPS)


def sum_args(m, n):
    return (torch.randn(m, n, device="cuda"),)


def ln_args(m, n):
    return (torch.randn(m, n, device="cuda"), [n], torch.randn(n, device="cuda"),
            torch.randn(n, device="cuda"), EPS)


def kl_args(m, v):
    return (torch.log_softmax(torch.randn(m, v, device="cuda"), -1),
            torch.softmax(torch.randn(m, v, device="cuda"), -1))


def jsd_args(m, v):
    return (torch.log_softmax(torch.randn(m, v, device="cuda"), -1),
            torch.log_softmax(torch.randn(m, v, device="cuda"), -1))


def ce_args(m, v):
    return (torch.randn(m, v, device="cuda"),
            torch.randint(0, v, (m,), device="cuda", dtype=torch.int64))


CASES = [
    ("rms_norm", rms_norm_fwd, rms_args, [(2048, 1024), (2048, 16384), (8192, 8192)]),
    ("sum", sum_kernel, sum_args, [(2048, 1024), (2048, 16384), (32768, 256)]),
    ("long_sum", longsum, sum_args, [(1, 32768), (16, 262144)]),
    ("layer_norm", layer_norm_fwd, ln_args, [(4096, 1024), (4096, 15872)]),
    ("softmax", softmax_two_pass, sum_args, [(4096, 256), (4096, 16384)]),
    ("kl_div", kl_div_forward, kl_args, [(4096, 4096), (4096, 131072)]),
    ("jsd", jsd_forward, jsd_args, [(8192, 4096), (8192, 131072)]),
    ("cross_entropy", cross_entropy, ce_args, [(4096, 16384), (8192, 131072)]),
]


def main():
    out = {}
    for name, kern, argfn, shapes in CASES:
        for (m, n) in shapes:
            bound = kern.bind(argfn(m, n))
            seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
            out[f"{name}:{m}x{n}"] = [
                dict(sorted(dict(s).items())) for s in seeds
            ]
    print(json.dumps(out, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
