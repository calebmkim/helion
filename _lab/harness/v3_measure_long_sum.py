"""v3 long_sum measurement, ONE shape per process (tiny latencies -> isolate).

Reports per-shape, all median-of-9 fresh do_bench, fp32:
  - v3 SEED (what the heuristic emits)   -> G_seed = tc/seed
  - un-seeded DEFAULT_config             -> G_default = tc/default
  - persistent/w32 reference             -> the decisive A/B (seed should TIE it)
Also confirms seed-used (codegen has NO `for roffset` when persistent) + corr.

Usage: --m M --n N
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


def run_cfg(x, ref, cfg):
    k = helion.kernel(longsum.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    tcode = b.to_triton_code(helion.Config(**dict(b._config)))
    looped_codegen = "for roffset" in tcode
    want_looped = bool(dict(b._config).get("reduction_loops", [None])[0])
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    ok = torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3)
    abs_e = float((out.float() - ref.float()).abs().max())
    t = med(lambda: b(x)) * 1000
    return t, ok, abs_e, (want_looped == looped_codegen), want_looped


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
    tc = torch.compile(lambda t: torch.sum(t, dim=-1))
    _ = tc(x)
    t_tc = med(lambda: tc(x)) * 1000

    # un-seeded default_config
    cfg_d = longsum.bind((x,)).config_spec.default_config()
    t_d, ok_d, _, _, _ = run_cfg(x, ref, helion.Config(**dict(cfg_d)))

    t_s, ok_s, abs_s, used_s, looped_s = run_cfg(x, ref, helion.Config(**seed))
    t_p, ok_p, _, _, _ = run_cfg(
        x, ref, helion.Config(block_sizes=[1], reduction_loops=[None], num_warps=32, num_stages=1))

    print(f"GPU={gpu} shape=({m},{n}) rnumelKiB={n*4//1024}")
    print(f"  v3_seed = rl={seed['reduction_loops']},w{seed['num_warps']},bs={seed['block_sizes']}  "
          f"codegen={'looped' if looped_s else 'persistent'}  seed_used={used_s}  ok={ok_s} maxabs={abs_s:.2e}")
    print(f"  tc_default      ={t_tc:7.2f}us")
    print(f"  v3 SEED         ={t_s:7.2f}us  G_seed   ={t_tc/t_s:.3f}")
    print(f"  default_config  ={t_d:7.2f}us  G_default={t_tc/t_d:.3f}  (ok={ok_d})")
    print(f"  persistent/w32  ={t_p:7.2f}us  G_p32    ={t_tc/t_p:.3f}  (ok={ok_p})")
    print(f"  DECISIVE: seed/p32 = {t_s/t_p:.3f}x  (==1.0 => v3 seed IS persistent/w32)")


if __name__ == "__main__":
    main()
