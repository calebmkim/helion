"""STAGE-1 breakpoint sweep: pid_type {flat, persistent_interleaved} x num_sm_multiplier {1,2,4}.

The MATCHED-LEVER A/B that sets the grid-bound gate by EVIDENCE (mirrors
STREAM_WARPS32_MIN_ELEMS discipline). For each (kernel-structure, M, N) we build
the EXACT v6 seed (block_sizes/reduction_loops/num_warps/num_stages held EQUAL)
and vary ONLY the pid_type + num_sm_multiplier lever:

  - flat            : the v6 seed verbatim (default flat pid)
  - pi_mult1/2/4    : pid_type=persistent_interleaved + num_sm_multiplier in {1,2,4}

Metric per cell = flat_lat / variant_lat  (>1 => the persistent variant is FASTER).

Swept across BOTH regimes the lever could affect:
  - grid-bound small-N / large-M  (m_extent >> num_sms AND small rnumel)
  - large-rnumel                  (long_sum / wide rms_norm — the v3 fence regime)
and across num_load=1 (sum) AND num_load>=2 (rms_norm) structures, because the
persistent-vs-looped story is num_load-dependent and pid could be too.

Correctness checked vs the fp32 reference for every config. do_bench median-of-N,
fresh inputs per shape. Run with the canonical invocation (SETUP.md).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.sum import sum_kernel  # noqa: E402
from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch  # noqa: E402
from examples.long_sum import longsum  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
import helion.runtime as rt  # noqa: E402

N_RUNS = 5
NUM_SM = rt.get_num_sm(torch.device("cuda"))


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


# ---- kernel adapters (build_args, reference, run) ----
def sum_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def sum_ref(args):
    return torch.sum(args[0], dim=-1)


def rms_args(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, 1e-5)


def rms_ref(args):
    return rms_norm_pytorch(args[0], args[1], args[2])


KERNELS = {
    "sum": {"fn": sum_kernel, "args": sum_args, "ref": sum_ref, "num_load": 1},
    "long_sum": {"fn": longsum, "args": sum_args, "ref": sum_ref, "num_load": 1},
    "rms_norm": {"fn": rms_norm_fwd, "args": rms_args, "ref": rms_ref, "num_load": 2},
}

VARIANTS = [
    ("flat", {}),
    ("pi_m1", {"pid_type": "persistent_interleaved", "num_sm_multiplier": 1}),
    ("pi_m2", {"pid_type": "persistent_interleaved", "num_sm_multiplier": 2}),
    ("pi_m4", {"pid_type": "persistent_interleaved", "num_sm_multiplier": 4}),
]


def get_v6_seed(fn, args):
    bound = fn.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    return dict(seeds[0])


def check_correct(out, ref):
    out = out[0] if isinstance(out, tuple) else out
    o = out.to(torch.float32)
    r = ref.to(torch.float32)
    ok = bool(torch.allclose(o, r, rtol=1e-3, atol=1e-3))
    max_abs = float((o - r).abs().max())
    return ok, max_abs


def bench_variant(fn, args, base_seed, override, ref):
    cfg = helion.Config(**{**base_seed, **override})
    k = helion.kernel(fn.fn, configs=[cfg])
    b = k.bind(args)
    b.ensure_config_exists(args)
    norm = dict(b._config)
    out = b(*args)
    ok, max_abs = check_correct(out, ref)
    if not ok:
        raise AssertionError(f"correctness FAIL max_abs={max_abs}")
    lat = median_do_bench(lambda: b(*args))
    return lat, norm.get("pid_type"), norm.get("num_sm_multiplier"), max_abs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=list(KERNELS))
    ap.add_argument("--shapes", required=True,
                    help="semicolon list MxN e.g. 8192x256;32768x256")
    a = ap.parse_args()
    spec = KERNELS[a.kernel]
    fn = spec["fn"]
    shapes = [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(";")]

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} NUM_SM={NUM_SM} kernel={a.kernel} num_load={spec['num_load']}")
    hdr = (f"{'shape':>14} {'m_ext':>7} {'rnumel':>7} {'codegen':>9} "
           f"{'flat_us':>8} {'piM1':>7} {'piM2':>7} {'piM4':>7} "
           f"{'r_m1':>6} {'r_m2':>6} {'r_m4':>6} {'best':>6}")
    print(hdr)
    print("-" * len(hdr))
    for (m, n) in shapes:
        args = spec["args"](m, n)
        ref = spec["ref"](args)
        base = get_v6_seed(fn, args)
        rl = base.get("reduction_loops", [None])
        codegen = "looped" if (rl and rl[0]) else "persist"
        m_ext = m
        rnumel = n
        lats = {}
        for name, override in VARIANTS:
            lat, pt, nm, _ = bench_variant(fn, args, base, override, ref)
            lats[name] = lat * 1000
        f = lats["flat"]
        r1 = f / lats["pi_m1"]
        r2 = f / lats["pi_m2"]
        r4 = f / lats["pi_m4"]
        best = max([("flat", 1.0), ("piM1", r1), ("piM2", r2), ("piM4", r4)],
                   key=lambda kv: kv[1])
        print(f"{f'{m}x{n}':>14} {m_ext:>7} {rnumel:>7} "
              f"{codegen:>9} {f:>8.1f} {lats['pi_m1']:>7.1f} {lats['pi_m2']:>7.1f} "
              f"{lats['pi_m4']:>7.1f} {r1:>6.2f} {r2:>6.2f} {r4:>6.2f} "
              f"{best[0]}({best[1]:.2f})")


if __name__ == "__main__":
    main()
