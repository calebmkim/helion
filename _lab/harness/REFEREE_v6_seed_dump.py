"""Referee: dump the heuristic seed for every in-sample shape of the 7 existing
active kernels, as JSON, so v5 vs v6 source can be diffed byte-for-byte."""

from __future__ import annotations

import json
import sys

import torch

import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WT), helion.__file__
sys.path.insert(0, WT)

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

LONG = torch.int64
EPS = 1e-5


def seed_for(kern, args):
    bound = kern.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    if len(seeds) != 1:
        return {"n_seeds": len(seeds)}
    return dict(seeds[0])


def rms_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), EPS)


def sum_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def ln_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32), [n],
            torch.randn(n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), EPS)


def sm_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def kl_args(m, v):
    lp = torch.log_softmax(torch.randn(m, v, device="cuda", dtype=torch.float32), -1)
    t = torch.softmax(torch.randn(m, v, device="cuda", dtype=torch.float32), -1)
    return (lp, t)


def jsd_args(m, v):
    lq = torch.log_softmax(torch.randn(m, v, device="cuda", dtype=torch.float32), -1)
    lp = torch.log_softmax(torch.randn(m, v, device="cuda", dtype=torch.float32), -1)
    return (lq, lp)


CASES = [
    ("rms_norm", rms_norm_fwd, rms_args, [(2048, 1024), (2048, 16384), (4096, 7168),
        (8192, 8192), (32768, 256), (32768, 1024)]),
    ("sum", sum_kernel, sum_args, [(2048, 1024), (2048, 16384), (8192, 4096),
        (32768, 256)]),
    ("long_sum", longsum, sum_args, [(1, 32768), (8, 131072), (16, 262144)]),
    ("layer_norm", layer_norm_fwd, ln_args, [(4096, 1024), (4096, 8192),
        (4096, 15872), (8192, 7168)]),
    ("softmax", softmax_two_pass, sm_args, [(4096, 256), (4096, 4096),
        (4096, 16384), (32768, 1024)]),
    ("kl_div", kl_div_forward, kl_args, [(4096, 4096), (4096, 65536),
        (4096, 131072)]),
    ("jsd", jsd_forward, jsd_args, [(8192, 4096), (8192, 65536), (8192, 131072)]),
]


def main():
    torch.manual_seed(0)
    out = {}
    for name, kern, argfn, shapes in CASES:
        for (a, b) in shapes:
            key = f"{name}:{a}x{b}"
            try:
                out[key] = seed_for(kern, argfn(a, b))
            except Exception as e:  # noqa: BLE001
                out[key] = {"ERR": f"{type(e).__name__}: {str(e)[:60]}"}
    print(json.dumps(out, sort_keys=True, indent=1))


if __name__ == "__main__":
    main()
