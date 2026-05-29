"""v3 looped-tail soundness check: for a row ABOVE the persistent cap (where a
persistent reduction literally cannot compile), confirm the v3 SEED goes LOOPED,
is correct, and beats the un-seeded default. Also confirm persistent FAILS (the
structural reason the looped branch exists). No in-sample shape reaches here.
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

N_RUNS = 7


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def run_cfg(x, ref, cfg):
    k = helion.kernel(longsum.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    tcode = b.to_triton_code(helion.Config(**dict(b._config)))
    looped_codegen = "for roffset" in tcode
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    ok = torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3)
    abs_e = float((out.float() - ref.float()).abs().max())
    return med(lambda: b(x)) * 1000, ok, abs_e, looped_codegen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--n", type=int, default=2097152)
    a = ap.parse_args()
    m, n = a.m, a.n
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    ref = x.sum(-1)

    bound = longsum.bind((x,))
    cap = bound.env.backend.max_tensor_numel
    seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])

    torch._dynamo.reset()
    tc = torch.compile(lambda t: torch.sum(t, dim=-1))
    _ = tc(x)
    t_tc = med(lambda: tc(x)) * 1000

    cfg_d = longsum.bind((x,)).config_spec.default_config()
    t_d, ok_d, _, _ = run_cfg(x, ref, helion.Config(**dict(cfg_d)))
    t_s, ok_s, abs_s, looped_s = run_cfg(x, ref, helion.Config(**seed))

    # persistent must FAIL above the cap -- prove it
    pers_fails = False
    try:
        run_cfg(x, ref, helion.Config(block_sizes=[1], reduction_loops=[None],
                                      num_warps=32, num_stages=1))
    except Exception as e:
        pers_fails = "exceeds triton maximum tensor numel" in str(e)

    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} shape=({m},{n}) "
          f"rnumel={n} cap={cap}  (rnumel {'>' if n>cap else '<='} cap)")
    print(f"  v3_seed = rl={seed['reduction_loops']},w{seed['num_warps']}  "
          f"codegen={'looped' if looped_s else 'persistent'}  ok={ok_s} maxabs={abs_s:.2e}")
    print(f"  v3 SEED        ={t_s:8.2f}us  G_seed={t_tc/t_s:.3f}")
    print(f"  default_config ={t_d:8.2f}us  G_dflt={t_tc/t_d:.3f} (ok={ok_d})")
    print(f"  tc_default     ={t_tc:8.2f}us")
    print(f"  persistent/w32 FAILS to compile above cap: {pers_fails}  "
          f"(=> looped branch is structurally REQUIRED here)")


if __name__ == "__main__":
    main()
