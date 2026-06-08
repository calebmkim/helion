"""TASK 3 correctness (NON-NEGOTIABLE): the v8 looped/capped apply tile must be CORRECT
at non-pow2 N AND prime N. Drives the REAL heuristic (compiler_seed_configs) and checks
allclose vs F.layer_norm fp32 at:
  - in-sample non-pow2 canary (262144,1536)
  - prime N: (262144,1543), (131072,3079)  [prime -> lpd=1 -> combine=1, apply capped]
  - N just above the apply cap with non-pow2: (262144,3072)=2^10*3 (lpd=1024), (262144,6144)=2^11*3
The apply pass is a masked element-wise write -> looping it must NOT change correctness.
"""
from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.welford import eager_layer_norm  # noqa: E402
from examples.welford import welford  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 1e-5
SHAPES = [
    (262144, 1536),   # in-sample non-pow2 canary
    (262144, 1543),   # PRIME
    (131072, 3079),   # PRIME, > apply cap
    (262144, 3072),   # 2^10*3, lpd=1024, np2=4096 -> apply LOOPED, non-pow2
    (131072, 6144),   # 2^11*3, lpd=2048, np2=8192 -> apply LOOPED, non-pow2
]


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def args(m, n):
    return (
        torch.rand(n, device="cuda"),
        torch.rand(n, device="cuda"),
        torch.rand(m, n, device="cuda"),
        EPS,
    )


def med(fn, reps=3):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    allok = True
    for (m, n) in SHAPES:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        bound = welford.bind(a)
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        if not seeds:
            print(f"({m},{n}) NO SEED (declined)", flush=True)
            continue
        seed = dict(seeds[0])
        k = helion.kernel(welford.fn, configs=[helion.Config(**seed)])
        b = k.bind(a)
        b.ensure_config_exists(a)
        code = b.to_triton_code(helion.Config(**dict(b._config)))
        nloops = code.count("for offset")
        out = b(*a)
        out = out[0] if isinstance(out, tuple) else out
        ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
        maxabs = float((out.float() - ref.float()).abs().max())
        denom = ref.float().abs().clamp_min(1e-4)
        maxrel = float(((out.float() - ref.float()).abs() / denom).max())
        bs = seed.get("block_sizes")
        apply_tile = bs[2]
        apply_looped = apply_tile < np2(n)
        flag = "" if ok else "  <<< WRONG!!!"
        allok = allok and ok
        print(
            f"({m},{n}) prime={is_prime(n)} lpd={n & (-n)} np2={np2(n)} "
            f"seed={bs} apply={'LOOPED' if apply_looped else 'persist'} "
            f"for-loops={nloops} ok={ok} maxabs={maxabs:.2e} maxrel={maxrel:.2e}{flag}",
            flush=True,
        )
    print(f"\nALL CORRECT: {allok}", flush=True)


if __name__ == "__main__":
    main()
