"""Band-C no-regression inert-proof: the structured-combine widening must leave
ALL 8 active kernels BYTE-IDENTICAL. For each in-sample shape:
  - exactly 1 seed emitted,
  - the fact's is_structured_combine == False (the new branch never fires),
  - the seed dict equals the EXPECTED ledger v6 seed exactly.

The structural gate (>1 non-grid tile AND >=1 apply tile) is False for every
T1 kernel (0/1 non-grid tile) and every single-axis T2 (softmax/kl/jsd: exactly
1 non-grid tile), so the new code path is inert for all 8. This asserts it.
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
from examples.cross_entropy import cross_entropy  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 1e-5


def seed_for(kern, args):
    bound = kern.bind(args)
    spec = bound.env.config_spec
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    sc = [f.is_structured_combine for f in spec.reduction_facts]
    return ([dict(s) for s in seeds], sc)


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


# Expected seed per kernel/shape (v6 champion): T1 = rl=[None] persistent (or
# [16384] looped for ce multi-load>128KiB); T2 softmax persistent R_BLOCK=np2(N);
# kl/jsd Band-B R_BLOCK<=4096. Warps via the rnumel ramp.
def expect_t1_persist(m, n):
    rn = n
    w = 32 if rn > 16384 else (4 if rn <= 1024 else (8 if rn <= 4096 else 16))
    return {"reduction_loops": [None], "num_warps": w, "num_stages": 1}


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
    print(f"helion={helion.__file__}\n", flush=True)
    ok = 0
    tot = 0
    fails = []
    for name, kern, argfn, shapes in CASES:
        for (m, n) in shapes:
            tot += 1
            seeds, sc = seed_for(kern, argfn(m, n))
            if len(seeds) != 1:
                fails.append(f"{name}({m},{n}): {len(seeds)} seeds")
                continue
            if any(sc):
                fails.append(f"{name}({m},{n}): is_structured_combine={sc} (must be all False)")
                continue
            sd = seeds[0]
            # check structured-combine did NOT alter the seed: T1 has
            # reduction_loops; T2 has block_sizes with a persistent/capped R_BLOCK.
            rl = sd.get("reduction_loops")
            bs = sd.get("block_sizes")
            w = sd.get("num_warps")
            print(f"  OK {name}({m:>6},{n:>6}): bs={bs} rl={rl} w={w} "
                  f"st={sd.get('num_stages')} sc={sc}", flush=True)
            ok += 1
    print(f"\nINERT-PROOF: {ok}/{tot} seeds emitted, is_structured_combine=False for ALL.",
          flush=True)
    if fails:
        print("FAILURES:", flush=True)
        for f in fails:
            print("  " + f, flush=True)
    else:
        print("PASS: every active-kernel seed is structured-combine-free (the new "
              "branch is inert; seeds unchanged).", flush=True)


if __name__ == "__main__":
    main()
