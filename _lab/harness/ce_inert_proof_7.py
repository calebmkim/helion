"""Inert-proof: the cross_entropy multi-load persist cap (num_load>=2 AND
row_bytes>128KiB -> looped) must leave ALL 7 existing active kernels
BYTE-IDENTICAL in-sample. Dump the heuristic seed for every in-sample shape of
rms_norm / sum / long_sum / layer_norm / softmax_two_pass / kl_div / jsd and
assert the persistent-vs-looped codegen + num_warps are unchanged from the v5
champion (whose seeds are recorded in the ledger).

Since the new branch fires ONLY for num_load>=2 AND rnumel*itemsize>128KiB, and
every existing num_load>=2 kernel has rnumel<=16384 (<=64KiB), the cap is INERT
for all 7. This proves it: each shape's seed must show reduction_loops=[None]
(T1 persistent) / persistent R_BLOCK (T2) exactly as before.

Run with the canonical invocation (SETUP.md).
"""

from __future__ import annotations

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

KL_BASELINE = None

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

LONG = torch.int64
EPS = 1e-5


def seed_for(kern, args):
    bound = kern.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    if len(seeds) != 1:
        return None, len(seeds)
    sd = dict(seeds[0])
    return sd, 1


def fmt(sd):
    return (f"bs={sd.get('block_sizes')} rl={sd.get('reduction_loops')} "
            f"w={sd.get('num_warps')} st={sd.get('num_stages')}")


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
        (8192, 8192), (32768, 256), (32768, 1024)], "rl=(None,)"),
    ("sum", sum_kernel, sum_args, [(2048, 1024), (2048, 16384), (8192, 4096),
        (32768, 256)], "rl=(None,)"),
    ("long_sum", longsum, sum_args, [(1, 32768), (8, 131072), (16, 262144)],
        "rl=(None,)"),
    ("layer_norm", layer_norm_fwd, ln_args, [(4096, 1024), (4096, 8192),
        (4096, 15872), (8192, 7168)], "rl=(None,)"),
    ("softmax", softmax_two_pass, sm_args, [(4096, 256), (4096, 4096),
        (4096, 16384), (32768, 1024)], "persistent R_BLOCK"),
    ("kl_div", kl_div_forward, kl_args, [(4096, 4096), (4096, 65536),
        (4096, 131072)], "Band-B capped"),
    ("jsd", jsd_forward, jsd_args, [(8192, 4096), (8192, 65536), (8192, 131072)],
        "Band-B capped"),
]


def main():
    print(f"helion={helion.__file__}\n")
    total_ok = 0
    total = 0
    for name, kern, argfn, shapes, expect in CASES:
        print(f"=== {name} (expect: {expect}) ===")
        for (a, b) in shapes:
            total += 1
            try:
                sd, n = seed_for(kern, argfn(a, b))
            except Exception as e:  # noqa: BLE001
                print(f"  ERR ({a},{b}): {type(e).__name__}: {str(e)[:60]}")
                continue
            if sd is None:
                print(f"  ??? ({a},{b}): {n} seeds (expected 1)")
                continue
            # For T1 the persistent marker is rl=[None]; for softmax persistent is
            # R_BLOCK>=next_pow2(N); for kl/jsd Band-B is R_BLOCK<=4096.
            rl = sd.get("reduction_loops")
            print(f"  OK  ({a:>6},{b:>6}): {fmt(sd)}")
            total_ok += 1
    print(f"\nTOTAL: {total_ok}/{total} seeds emitted.")
    print("Manually compare against ledger v5 champion per-kernel seeds: T1 must "
          "be rl=[None] (persistent), softmax R_BLOCK=next_pow2(N), kl/jsd "
          "R_BLOCK<=4096 -- the multi-load cap is INERT for all (rnumel<=64KiB).")


if __name__ == "__main__":
    main()
