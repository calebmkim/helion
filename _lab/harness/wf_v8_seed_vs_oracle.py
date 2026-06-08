"""TASK 5: v8 seed vs FRESH quick-autotune ORACLE at the recovery shape (262144,4096)
and the non-pow2 canary (262144,1536). Uses the REAL heuristic-emitted seed
(compiler_seed_configs). Confirms seed/oracle ~ 1.0 (the v8 seed is now at the
deterministic-seed ceiling). Fair re-bench of the verbatim oracle winner (all levers).
"""
from __future__ import annotations

import os
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

EPS = 1e-5
SHAPES = [(262144, 4096), (262144, 1536)]


def args(m, n):
    return (
        torch.rand(n, device="cuda"),
        torch.rand(n, device="cuda"),
        torch.rand(m, n, device="cuda"),
        EPS,
    )


def med(fn, reps=5):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def rebench(a, cfg, ref):
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
    mx = float((out.float() - ref.float()).abs().max())
    return med(lambda: b(*a)) * 1000, ok, mx


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    for (m, n) in SHAPES:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm)
        tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        # REAL v8 heuristic seed
        bound = welford.bind(a)
        seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
        seed_c = {k: seed[k] for k in ("block_sizes", "num_warps", "num_stages")}
        slat, sok, smx = rebench(a, seed_c, ref)
        # fresh quick-autotune oracle
        os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"
        k = helion.kernel(welford.fn)
        b = k.bind(a)
        b.autotune(a)
        win = dict(b._config)
        wlat, wok, wmx = rebench(a, win, ref)
        print(f"=== ({m},{n}) tc={round(tclat,1)}us ===", flush=True)
        print(f"  v8 SEED  {seed_c['block_sizes']} w{seed_c['num_warps']}: "
              f"{round(slat,1)}us ok={sok} maxabs={smx:.1e} G={round(tclat/slat,3)}", flush=True)
        print(f"  ORACLE   bs={win.get('block_sizes')} w{win.get('num_warps')} "
              f"s{win.get('num_stages')} pid={win.get('pid_type')}: "
              f"{round(wlat,1)}us ok={wok} maxabs={wmx:.1e} G={round(tclat/wlat,3)}", flush=True)
        print(f"  --> seed/oracle = {round(slat/wlat,3)}\n", flush=True)


if __name__ == "__main__":
    main()
