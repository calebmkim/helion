"""AUDITOR independent correctness sweep for welford v7.

For each N (pow2, several non-pow2, a prime, and a no-good-pow2-divisor N), take
the seed the heuristic emits, compile+run it, and compare to F.layer_norm fp32.
Report combine_block (= largest_pow2_div capped), Tn behaviour, max_abs/max_rel,
and whether the seed config is actually USED (codegen for-loop count over the
combine tile).

The CRITICAL question: does the divisor-chunk fix hold for ALL non-pow2 N, incl
a prime N where the only pow2 divisor is 1 (combine_block=1)? A combine_block=1
chunk has Tn=1 per chunk == true count -> should be CORRECT (just slow). Verify.
"""
from __future__ import annotations

import sys
import torch
import torch.nn.functional as F
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.welford import welford  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5

# (M, N, label). M kept modest for prime/tiny-divisor N so the slow combine still
# finishes; correctness does not depend on M.
CASES = [
    (262144, 1024, "pow2"),
    (262144, 2048, "pow2"),
    (262144, 1536, "non-pow2 canary (=512*3, div=512)"),
    (65536, 1500, "non-pow2 (=4*375, div=4 tiny)"),
    (262144, 2560, "non-pow2 (=512*5, div=512)"),
    (262144, 3072, "non-pow2 (=1024*3, div=1024)"),
    (8192, 1543, "PRIME (div=1 -> combine_block=1)"),
    (8192, 1999, "PRIME (div=1)"),
    (8192, 768, "non-pow2 (=256*3, div=256)"),
    (4096, 4096, "pow2 4096 floor"),
]


def eager_ln(weight, bias, x, eps):
    return F.layer_norm(x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=eps)


def args(m, n):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (torch.rand(n, device="cuda", generator=g),
            torch.rand(n, device="cuda", generator=g),
            torch.rand(m, n, device="cuda", generator=g), EPS)


def main():
    print(f"helion={helion.__file__}", flush=True)
    print(f"dev={torch.cuda.get_device_name(0)} torch={torch.__version__}\n", flush=True)
    print(f"{'N':>6} {'div':>6} {'combine':>8} {'apply':>6} {'used':>5} {'ok':>4} "
          f"{'max_abs':>10} {'max_rel':>10}  label", flush=True)
    all_ok = True
    for (m, n, label) in CASES:
        a = args(m, n)
        ref = eager_ln(*a).float()
        bound = welford.bind(a)
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        if not seeds:
            print(f"{n:>6} {'-':>6} {'DECLINED':>8} -- a seed was NOT emitted", flush=True)
            continue
        seed = dict(seeds[0])
        bs = seed["block_sizes"]
        fact = bound.env.config_spec.reduction_facts[0]
        red_idx = bound.env.config_spec.block_sizes.block_id_to_index(fact.block_id)
        apply_idx = [bound.env.config_spec.block_sizes.block_id_to_index(b)
                     for b in fact.apply_block_ids]
        combine = bs[red_idx]
        applyv = bs[apply_idx[0]] if apply_idx else None
        div = n & (-n)

        k = helion.kernel(welford.fn, configs=[helion.Config(**seed)])
        b = k.bind(a)
        b.ensure_config_exists(a)
        code = b.to_triton_code(helion.Config(**dict(b._config)))
        # number of inner for-loops in codegen; combine_block < N -> the combine
        # pass is a real loop (not persistent).
        nloops = code.count("for ")
        out = b(*a)
        out = (out[0] if isinstance(out, tuple) else out).float()
        maxabs = float((out - ref).abs().max())
        denom = ref.abs().clamp_min(1e-4)
        maxrel = float(((out - ref).abs() / denom).max())
        ok = maxabs <= 2e-3 and maxrel <= 2e-3
        all_ok = all_ok and ok
        print(f"{n:>6} {div:>6} {combine:>8} {applyv!s:>6} {nloops:>5} "
              f"{str(ok):>4} {maxabs:>10.2e} {maxrel:>10.2e}  {label}", flush=True)
        del a, ref, out, b, k
        torch.cuda.empty_cache()
    print(f"\nALL CORRECT: {all_ok}", flush=True)


if __name__ == "__main__":
    main()
