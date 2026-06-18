"""Fresh-process confirmation: shipped SEED (looped/w32) vs persistent/w32.

One shape per invocation (--m --n) to rule out compile-order / cache artifacts.
Times the EXACT shipped seed and persistent/warps32, plus torch.compile default
reference, median-of-9 fresh do_bench. Prints both absolute us and G vs tc.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch._dynamo

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.long_sum import longsum  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 9


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def time_cfg(x, ref, cfg):
    k = helion.kernel(longsum.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    abs_err = float((out.float() - ref.float()).abs().max())
    rel_err = float(((out.float() - ref.float()).abs() / (ref.float().abs() + 1e-12)).max())
    ok = torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3)
    return med(lambda: b(x)) * 1000, ok, abs_err, rel_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    m, n = a.m, a.n
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    ref = x.sum(-1)

    bound = longsum.bind((x,))
    seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])

    torch._dynamo.reset()

    def _sumref(t):
        return torch.sum(t, dim=-1)
    tc = torch.compile(_sumref)
    _ = tc(x)
    t_tc = med(lambda: tc(x)) * 1000

    t_seed, ok_s, abs_s, rel_s = time_cfg(x, ref, helion.Config(**seed))
    t_p32, ok_p, abs_p, rel_p = time_cfg(
        x, ref, helion.Config(block_sizes=[1], reduction_loops=[None], num_warps=32, num_stages=1))

    print(f"GPU={gpu} shape=({m},{n}) rnumelKiB={n*4//1024}")
    print(f"  seedcfg=rl={seed['reduction_loops']},w{seed['num_warps']}")
    print(f"  tc_default={t_tc:.2f}us")
    print(f"  SEED(looped/w32) ={t_seed:7.2f}us  G={t_tc/t_seed:.3f}  ok={ok_s} maxabs={abs_s:.2e} maxrel={rel_s:.2e}")
    print(f"  persistent/w32   ={t_p32:7.2f}us  G={t_tc/t_p32:.3f}  ok={ok_p} maxabs={abs_p:.2e} maxrel={rel_p:.2e}")
    print(f"  P32 vs SEED speedup = {t_seed/t_p32:.3f}x")


if __name__ == "__main__":
    main()
