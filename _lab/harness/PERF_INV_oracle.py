"""Perf-investigator: full-autotune ORACLE per shape + verbatim re-bench + field-diff.

For each (kernel, shape): run Helion effort=full, capture bound._config (the FULL
verbatim winner), re-bench it VERBATIM (lever-isolation guard: assert the resolved
config == emitted winner), report G_oracle and the seed-vs-oracle field diff.

NO fabricated single-lever configs. fp32.
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
from examples.long_sum import longsum  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7
LONG = torch.int64
LEVERS = ["block_sizes", "reduction_loops", "num_warps", "num_stages", "indexing",
          "load_eviction_policies"]


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


REG = {
    "rms_norm": (rms_norm_fwd, build_rms,
                 lambda a: rms_norm_pytorch(*a),
                 lambda: torch.compile(rms_norm_pytorch)),
    "sum": (sum_kernel, build_x,
            lambda a: torch.sum(a[0], -1),
            lambda: torch.compile(lambda x: torch.sum(x, -1))),
    "long_sum": (longsum, build_x,
                 lambda a: torch.sum(a[0], -1),
                 lambda: torch.compile(lambda x: torch.sum(x, -1))),
    "softmax_two_pass": (softmax_two_pass, build_x,
                         lambda a: torch.nn.functional.softmax(a[0], 1),
                         lambda: torch.compile(lambda x: torch.nn.functional.softmax(x, 1))),
    "cross_entropy": (cross_entropy, build_ce,
                      lambda a: torch.nn.functional.cross_entropy(*a),
                      lambda: torch.compile(torch.nn.functional.cross_entropy)),
}

CASES = [
    ("rms_norm", (32768, 256)),
    ("sum", (8192, 256)),
    ("softmax_two_pass", (32768, 256)),
    ("cross_entropy", (4096, 4096)),
    ("rms_norm", (2048, 16384)),
    ("long_sum", (256, 131072)),
]


def assert_verbatim(bound, oracle_cfg):
    resolved = dict(bound._config)
    coupled = ["block_sizes", "reduction_loops", "num_warps", "num_stages"]
    mism = {l: (oracle_cfg.get(l), resolved.get(l)) for l in coupled
            if oracle_cfg.get(l) != resolved.get(l)}
    assert not mism, f"NOT verbatim: {mism}"


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}\n")
    for name, shape in CASES:
        fn, build, reffn, tcfn = REG[name]
        args = build(shape)
        ref = reffn(args)

        bound0 = fn.bind(args)
        seed = dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])

        torch._dynamo.reset()
        tc = tcfn()
        _ = tc(*args)
        tc_lat = med(lambda: tc(*args))

        os.environ["HELION_AUTOTUNE_EFFORT"] = "full"
        k = helion.kernel(fn.fn)
        bound = k.bind(args)
        bound.autotune(args)
        oracle = dict(bound._config)
        assert_verbatim(bound, oracle)
        out = bound(*args)
        o = out[0] if isinstance(out, tuple) else out
        o = o.float(); r = ref.float()
        ok = bool(o.shape == r.shape and torch.allclose(o, r, rtol=1e-3, atol=1e-3))
        oracle_lat = med(lambda: bound(*args))

        print(f"=== {name} {shape} (rnumel={shape[1]}) correct={ok} ===")
        print(f"  tc={tc_lat*1000:.2f}us  oracle={oracle_lat*1000:.2f}us  "
              f"G_oracle={tc_lat/oracle_lat:.3f}")
        print(f"  {'lever':>22} {'seed':>20} {'oracle':>20}")
        for lev in LEVERS:
            sv, ov = seed.get(lev), oracle.get(lev)
            mark = "" if sv == ov else "  <-- DIFF"
            print(f"  {lev:>22} {str(sv):>20} {str(ov):>20}{mark}")
        print(f"  full oracle: {oracle}\n")


if __name__ == "__main__":
    main()
