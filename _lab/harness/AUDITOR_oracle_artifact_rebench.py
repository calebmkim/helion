"""AUDITOR: fair fresh-process re-bench of the perf-investigator's VERBATIM oracle
configs vs tc-default and vs the v6 seed.

The perf-investigator (PERF_INV_oracle.py) ran on GPU=2 and benched the oracle in
the SAME process immediately after bound.autotune() — warm autotuner cache + any
co-tenant on GPU 2. The claimed G's (sum 8192x256 G=1.58, rms_norm 32768x256
G=1.108, softmax 32768x256 G=1.061, ce 4096x4096 G=1.095) are suspect.

Here we take the EXACT verbatim oracle Config dicts the perf-investigator reported
(logs/perf_inv_oracle.out) and:
  - build a FRESH kernel from each (no autotune), correctness-check vs fp32 ref
  - fair do_bench (median-of-N, N runs, repeated, sorted-median) on idle GPU 1
  - bench tc-default in the same process for the G ratio
  - bench the v6 seed for context (G_v6 and oracle/v6)
Report fair G_oracle and compare to the claimed value. Artifact if it collapses.
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

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
import helion.runtime as rt  # noqa: E402

N_RUNS = 9
LONG = torch.int64
NUM_SM = rt.get_num_sm(torch.device("cuda"))


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


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


# VERBATIM oracle configs from logs/perf_inv_oracle.out (full oracle: lines).
CASES = [
    ("sum", (8192, 256), sum_kernel, build_x,
     lambda a: torch.sum(a[0], -1),
     lambda: torch.compile(lambda x: torch.sum(x, -1)),
     1.580,
     {'block_sizes': [32], 'reduction_loops': [32], 'num_warps': 8, 'num_stages': 3,
      'indexing': ['pointer', 'pointer', 'tensor_descriptor'],
      'load_eviction_policies': ['last', 'first'],
      'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 2, 'maxnreg': 128}),
    ("rms_norm", (32768, 256), rms_norm_fwd, build_rms,
     lambda a: rms_norm_pytorch(*a),
     lambda: torch.compile(rms_norm_pytorch),
     1.108,
     {'block_sizes': [8], 'reduction_loops': [128], 'num_warps': 16, 'num_stages': 2,
      'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer',
                   'tensor_descriptor', 'tensor_descriptor'],
      'load_eviction_policies': ['', 'last', 'last', '', 'last'], 'pid_type': 'flat'}),
    ("softmax_two_pass", (32768, 256), softmax_two_pass, build_x,
     lambda a: torch.nn.functional.softmax(a[0], 1),
     lambda: torch.compile(lambda x: torch.nn.functional.softmax(x, 1)),
     1.061,
     {'block_sizes': [2, 256], 'num_warps': 4, 'num_stages': 2,
      'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'],
      'load_eviction_policies': ['', 'last'], 'pid_type': 'flat'}),
    ("cross_entropy", (4096, 4096), cross_entropy, build_ce,
     lambda a: torch.nn.functional.cross_entropy(*a),
     lambda: torch.compile(torch.nn.functional.cross_entropy),
     1.095,
     {'block_sizes': [1], 'reduction_loops': [None], 'num_warps': 16, 'num_stages': 8,
      'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
      'load_eviction_policies': ['', '', 'first', 'first', 'last'], 'pid_type': 'flat'}),
]


def correct(out, ref):
    o = out[0] if isinstance(out, tuple) else out
    o = o.float(); r = ref.float()
    return bool(o.shape == r.shape and torch.allclose(o, r, rtol=1e-3, atol=1e-3))


def bench_cfg(fn, args, cfg, ref):
    k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args); b.ensure_config_exists(args)
    out = b(*args)
    ok = correct(out, ref)
    lat = med(lambda: b(*args)) * 1000
    return lat, ok


def v6_seed(fn, args):
    bound0 = fn.bind(args)
    return dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} NUM_SM={NUM_SM} N_RUNS={N_RUNS} helion={helion.__file__}\n")
    hdr = (f"{'kernel/shape':>24} {'tc_us':>8} {'v6_us':>8} {'orac_us':>8} "
           f"{'G_v6':>6} {'G_orac':>7} {'claimed':>8} {'orac/v6':>8} {'ok':>3}")
    print(hdr); print("-" * len(hdr))
    for name, shape, fn, build, reffn, tcfn, claimed, oracle in CASES:
        args = build(shape); ref = reffn(args)
        # tc default
        torch._dynamo.reset()
        tc = tcfn(); _ = tc(*args)
        tc_lat = med(lambda: tc(*args)) * 1000
        # v6 seed
        v6 = v6_seed(fn, args)
        v6_lat, v6_ok = bench_cfg(fn, args, v6, ref)
        # verbatim oracle
        or_lat, or_ok = bench_cfg(fn, args, oracle, ref)
        print(f"{f'{name} {shape}':>24} {tc_lat:>8.2f} {v6_lat:>8.2f} {or_lat:>8.2f} "
              f"{tc_lat/v6_lat:>6.3f} {tc_lat/or_lat:>7.3f} {claimed:>8.3f} "
              f"{or_lat/v6_lat:>8.3f} {str(or_ok and v6_ok):>3}")


if __name__ == "__main__":
    main()
