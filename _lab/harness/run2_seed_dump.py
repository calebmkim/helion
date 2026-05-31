"""Dump the live compiler seed(s) for all 9 kernels on representative shapes.
Path-AGNOSTIC: prints helion.__file__ and resolves whatever PYTHONPATH points at,
so it can be run against wt-reduction-2 (new) AND wt-reduction (v8) to diff.
Usage: ... python run2_seed_dump.py [out.json]
"""
from __future__ import annotations
import json, sys
import torch
import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

EPS = 1e-5
print(f"# helion={helion.__file__}", file=sys.stderr)

from examples.rms_norm import rms_norm_fwd
from examples.sum import sum_kernel
from examples.long_sum import longsum
from examples.layer_norm import layer_norm_fwd
from examples.softmax import softmax_two_pass
from examples.kl_div import kl_div_forward
from examples.jsd import jsd_forward
from examples.cross_entropy import cross_entropy
from examples.welford import welford


def rms_args(m, n): return (torch.randn(m, n, device="cuda"), torch.randn(n, device="cuda"), EPS)
def sum_args(m, n): return (torch.randn(m, n, device="cuda"),)
def ln_args(m, n): return (torch.randn(m, n, device="cuda"), [n], torch.randn(n, device="cuda"), torch.randn(n, device="cuda"), EPS)
def kl_args(m, v): return (torch.log_softmax(torch.randn(m, v, device="cuda"), -1), torch.softmax(torch.randn(m, v, device="cuda"), -1))
def jsd_args(m, v): return (torch.log_softmax(torch.randn(m, v, device="cuda"), -1), torch.log_softmax(torch.randn(m, v, device="cuda"), -1))
def ce_args(m, v): return (torch.randn(m, v, device="cuda"), torch.randint(0, v, (m,), device="cuda", dtype=torch.int64))
def wf_args(m, n): return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"), torch.rand(m, n, device="cuda"), EPS)

CASES = [
    ("rms_norm", rms_norm_fwd, rms_args, [(2048, 1024), (2048, 16384), (8192, 8192), (32768, 256)]),
    ("sum", sum_kernel, sum_args, [(2048, 1024), (2048, 16384), (32768, 256)]),
    ("long_sum", longsum, sum_args, [(1, 32768), (16, 262144)]),
    ("layer_norm", layer_norm_fwd, ln_args, [(4096, 1024), (4096, 15872), (8192, 4096)]),
    ("softmax", softmax_two_pass, sum_args, [(4096, 256), (4096, 16384)]),
    ("kl_div", kl_div_forward, kl_args, [(4096, 4096), (4096, 131072)]),
    ("jsd", jsd_forward, jsd_args, [(8192, 4096), (8192, 131072)]),
    ("cross_entropy", cross_entropy, ce_args, [(4096, 16384), (8192, 131072)]),
    ("welford", welford, wf_args, [(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096),
                                    (262144, 2560), (262144, 5120), (8192, 4096)]),
]


def seed_for(kern, args):
    bound = kern.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    return [dict(s) for s in seeds]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    dump = {}
    for name, kern, af, shapes in CASES:
        for (m, n) in shapes:
            key = f"{name}({m},{n})"
            try:
                dump[key] = seed_for(kern, af(m, n))
            except Exception as e:  # noqa: BLE001
                dump[key] = {"error": f"{type(e).__name__}: {e}"}
    txt = json.dumps(dump, sort_keys=True, indent=2, default=str)
    print(txt)
    if out:
        open(out, "w").write(txt)


if __name__ == "__main__":
    main()
