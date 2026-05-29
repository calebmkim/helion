"""AUDITOR correctness v2 (pinned configs only -- NO autotuning).

For each N compare to F.layer_norm fp32 using the proper combined-tolerance
criterion (torch.allclose, |out-ref| <= atol + rtol*|ref|):
  (A) SEEDED welford   : combine = pow2 divisor of N (the seed under test)
  (C) OLDBUG welford   : combine = next_pow2(N) MASKED (the prior bug) -- shows
      the masked combine is WRONG at non-pow2 N (Tn counts padding).
Both are PINNED configs (helion.kernel(..., configs=[cfg])) so neither autotunes.

A correct fix => (A) passes at every N; (C) FAILS (large error) at non-pow2 N
and passes only at pow2 N. Report max_abs and a 1e-2/1e-1 allclose.
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

CASES = [
    (8192, 1024, "pow2"),
    (8192, 1536, "non-pow2 (=512*3)"),
    (8192, 1500, "non-pow2 (=4*375, tiny div=4)"),
    (8192, 2560, "non-pow2 (=512*5)"),
    (8192, 3072, "non-pow2 (=1024*3)"),
    (8192, 1543, "PRIME (div=1)"),
    (8192, 1999, "PRIME (div=1)"),
    (8192, 768, "non-pow2 (=256*3)"),
]


def eager_ln(weight, bias, x, eps):
    return F.layer_norm(x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=eps)


def args(m, n):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (torch.rand(n, device="cuda", generator=g),
            torch.rand(n, device="cuda", generator=g),
            torch.rand(m, n, device="cuda", generator=g), EPS)


def err(out, ref):
    out = out.float(); ref = ref.float()
    maxabs = float((out - ref).abs().max())
    ac = bool(torch.allclose(out, ref, rtol=1e-2, atol=1e-1))
    return maxabs, ac


def run_cfg(a, cfg):
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    return (out[0] if isinstance(out, tuple) else out).float()


def main():
    print(f"helion={helion.__file__}", flush=True)
    print(f"dev={torch.cuda.get_device_name(0)}\n", flush=True)
    print(f"{'N':>6} {'combineA':>9} {'maxabsA':>10} {'acA':>6}   "
          f"{'combineC':>9} {'maxabsC':>10} {'acC':>6}  label", flush=True)
    a_all_ok = True
    c_bug_shown = True  # OLDBUG must FAIL at every non-pow2
    for (m, n, label) in CASES:
        a = args(m, n)
        ref = eager_ln(*a)
        bound = welford.bind(a)
        seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
        fact = bound.env.config_spec.reduction_facts[0]
        red_idx = bound.env.config_spec.block_sizes.block_id_to_index(fact.block_id)
        np2n = 1 << (n - 1).bit_length()
        is_pow2 = (n & (n - 1)) == 0

        oA = run_cfg(a, seed); eA = err(oA, ref)
        a_all_ok = a_all_ok and eA[1]

        broken = dict(seed); broken["block_sizes"] = list(seed["block_sizes"])
        broken["block_sizes"][red_idx] = np2n
        oC = run_cfg(a, broken); eC = err(oC, ref)
        if not is_pow2 and eC[1]:
            c_bug_shown = False  # bug NOT shown (oldbug passed at non-pow2)

        print(f"{n:>6} {seed['block_sizes'][red_idx]:>9} {eA[0]:>10.2e} {str(eA[1]):>6}   "
              f"{np2n:>9} {eC[0]:>10.2e} {str(eC[1]):>6}  {label}", flush=True)
        del a, ref, oA, oC
        torch.cuda.empty_cache()
    print(f"\nSEED correct at ALL N (allclose 1e-2/1e-1): {a_all_ok}", flush=True)
    print(f"OLDBUG masked-combine WRONG at every non-pow2 N: {c_bug_shown}", flush=True)


if __name__ == "__main__":
    main()
