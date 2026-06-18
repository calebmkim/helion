"""Is pid_type=persistent_interleaved a REAL lever or a PASSENGER in the oracle bundle?

The brief's premise: sum/rms_norm (32768,256) oracle G>1 uses pid=persistent_interleaved
+ num_sm_multiplier + (block_sizes, reduction_loops, num_warps, num_stages all different
from the v6 seed). My matched A/B vs the v6 seed shows pid ALONE regresses. This isolates
pid WITHIN the oracle's full lever bundle: take the verbatim oracle config and flip ONLY
pid_type (flat <-> persistent_interleaved, and sweep num_sm_multiplier). If the win
survives the pid flip while the rest is held EQUAL, pid is real-but-coupled; if flat ties
or beats persistent within the bundle, pid is a passenger (the win is the other levers).

Documented (32768,256) verbatim oracle winner (rms_norm, ledger oracle_cache):
  block_sizes=[4], reduction_loops=[128], num_warps=1, num_stages=5, pid=persistent_interleaved

Run with the canonical invocation.
"""

from __future__ import annotations

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

import helion.runtime as rt  # noqa: E402

N_RUNS = 7
NUM_SM = rt.get_num_sm(torch.device("cuda"))


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


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


def check(out, ref):
    out = out[0] if isinstance(out, tuple) else out
    o = out.to(torch.float32); r = ref.to(torch.float32)
    return bool(torch.allclose(o, r, rtol=1e-3, atol=1e-3)), float((o - r).abs().max())


def bench(fn, args, cfg, ref):
    k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    out = b(*args)
    ok, ma = check(out, ref)
    if not ok:
        return None, ma  # incorrect -> skip
    return median_do_bench(lambda: b(*args)) * 1000, ma


# The verbatim oracle bundle (other levers) — the bundle MINUS pid_type.
# Sweep pid in {flat, persistent_interleaved x mult{1,2,4}} holding it EQUAL.
CASES = [
    # (label, fn, args_fn, ref_fn, M, N, oracle_bundle_without_pid)
    ("rms_norm 32768x256 oracle-bundle", rms_norm_fwd, rms_args, rms_ref, 32768, 256,
     {"block_sizes": [4], "reduction_loops": [128], "num_warps": 1, "num_stages": 5}),
    ("sum 32768x256 oracle-bundle", sum_kernel, sum_args, sum_ref, 32768, 256,
     {"block_sizes": [4], "reduction_loops": [128], "num_warps": 1, "num_stages": 5}),
    ("sum 8192x256 oracle-bundle", sum_kernel, sum_args, sum_ref, 8192, 256,
     {"block_sizes": [4], "reduction_loops": [128], "num_warps": 1, "num_stages": 5}),
]


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} NUM_SM={NUM_SM}\n")
    for label, fn, args_fn, ref_fn, m, n, bundle in CASES:
        args = args_fn(m, n)
        ref = ref_fn(args)
        print(f"=== {label} (bundle={bundle}) ===")
        variants = [
            ("flat", {}),
            ("pi_m1", {"pid_type": "persistent_interleaved", "num_sm_multiplier": 1}),
            ("pi_m2", {"pid_type": "persistent_interleaved", "num_sm_multiplier": 2}),
            ("pi_m4", {"pid_type": "persistent_interleaved", "num_sm_multiplier": 4}),
        ]
        lats = {}
        for name, ov in variants:
            cfg = {**bundle, **ov}
            lat, ma = bench(fn, args, cfg, ref)
            lats[name] = lat
            print(f"   {name:>7}: {lat if lat else 'INCORRECT':>9}{'' if lat is None else ' us'}  maxabs={ma:.2e}")
        if lats.get("flat"):
            f = lats["flat"]
            for name in ("pi_m1", "pi_m2", "pi_m4"):
                if lats.get(name):
                    print(f"      flat/{name} = {f/lats[name]:.3f}  (>1 => {name} faster)")
        print()


if __name__ == "__main__":
    main()
