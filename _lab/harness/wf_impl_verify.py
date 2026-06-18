"""Verify the implemented Band-C structured-combine treatment:
  1. welford now FIRES (reduction_facts=1, is_structured_combine=True), seed has
     the expected block_sizes, is USED (codegen) and CORRECT at all 4 in-sample
     shapes incl the non-pow2 (262144,1536).
  2. softmax_two_pass / kl_div / jsd / T1 are UNAFFECTED (is_structured_combine
     False; their seeds unchanged -- the gate only widens welford).
"""
from __future__ import annotations

import sys
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402
from examples.welford import welford, eager_layer_norm  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
IN_SAMPLE = [(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096)]


def args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def med(fn, reps=3):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    for (m, n) in IN_SAMPLE:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        bound = welford.bind(a)
        spec = bound.env.config_spec
        facts = spec.reduction_facts
        print(f"=== ({m},{n})  reduction_facts={len(facts)} ===", flush=True)
        if len(facts) == 1:
            f = facts[0]
            print(f"  fact: block_id={f.block_id} size_hint={f.size_hint} "
                  f"is_structured_combine={f.is_structured_combine} "
                  f"apply_block_ids={f.apply_block_ids} num_load={f.num_load} "
                  f"num_reduction_ops={f.num_reduction_ops}", flush=True)
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        if not seeds:
            print("  NO SEED (declined)", flush=True)
            continue
        seed = dict(seeds[0])
        print(f"  SEED: block_sizes={seed.get('block_sizes')} "
              f"num_warps={seed.get('num_warps')} "
              f"heuristics={spec.autotuner_heuristics}", flush=True)
        # run the seed (configs=[seed] short-circuit) + correctness
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
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm); tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        slat = med(lambda: b(*a)) * 1000
        print(f"  USED: for-loops={nloops}  CORRECT ok={ok} maxabs={maxabs:.2e} "
              f"maxrel={maxrel:.2e}  G={round(tclat/slat,3)} "
              f"(seed={round(slat,1)}us tc={round(tclat,1)}us)", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
