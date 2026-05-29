"""AUDITOR independent win re-measurement for welford v7.

For each shape compute:
  - G_default  = tc / helion-DEFAULT-config latency   (the claimed 0.526 baseline)
  - G_seed     = tc / heuristic-SEED latency          (the claimed 0.894 win)
  - G_oracle   = tc / quick-autotune-winner latency   (the ceiling) [only at 4096]
where tc = torch.compile(F.layer_norm) fp32. do_bench median, repeated.
All configs PINNED except the oracle (quick autotune). Correctness re-checked.
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn.functional as F
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402
from examples.welford import welford  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
RUN_ORACLE = os.environ.get("RUN_ORACLE", "0") == "1"
SHAPES = [(262144, 1536), (262144, 2048), (262144, 4096)]


def eager_ln(weight, bias, x, eps):
    return F.layer_norm(x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=eps)


def args(m, n):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (torch.rand(n, device="cuda", generator=g),
            torch.rand(n, device="cuda", generator=g),
            torch.rand(m, n, device="cuda", generator=g), EPS)


def med(fn, reps=5):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def lat_for(a, cfg, ref):
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = (out[0] if isinstance(out, tuple) else out).float()
    ok = bool(torch.allclose(out, ref.float(), rtol=1e-2, atol=1e-1))
    maxabs = float((out - ref.float()).abs().max())
    return med(lambda: b(*a)) * 1000, ok, maxabs


def main():
    print(f"helion={helion.__file__}", flush=True)
    print(f"dev={torch.cuda.get_device_name(0)}  RUN_ORACLE={RUN_ORACLE}\n", flush=True)
    for (m, n) in SHAPES:
        a = args(m, n)
        ref = eager_ln(*a)
        bound = welford.bind(a)
        spec = bound.env.config_spec
        seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
        default = dict(spec.default_config().config)
        torch._dynamo.reset()
        tc = torch.compile(eager_ln); tc(*a)
        tclat = med(lambda: tc(*a)) * 1000

        dlat, dok, dmax = lat_for(a, default, ref)
        slat, sok, smax = lat_for(a, seed, ref)
        print(f"=== ({m},{n}) tc={tclat:.1f}us ===", flush=True)
        print(f"  DEFAULT bs={default.get('block_sizes')} w={default.get('num_warps')}: "
              f"{dlat:.1f}us ok={dok} maxabs={dmax:.1e}  G_default={tclat/dlat:.3f}", flush=True)
        print(f"  SEED    bs={seed.get('block_sizes')} w={seed.get('num_warps')}: "
              f"{slat:.1f}us ok={sok} maxabs={smax:.1e}  G_seed={tclat/slat:.3f}", flush=True)
        print(f"  seed/default speedup = {dlat/slat:.2f}x", flush=True)

        if RUN_ORACLE and n == 4096:
            os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"
            k = helion.kernel(welford.fn)
            b = k.bind(a)
            b.autotune(a)
            win = dict(b._config)
            wlat, wok, wmax = lat_for(a, win, ref)
            print(f"  ORACLE  bs={win.get('block_sizes')} w={win.get('num_warps')}: "
                  f"{wlat:.1f}us ok={wok}  G_oracle={tclat/wlat:.3f} "
                  f"(seed/oracle={slat/wlat:.3f})", flush=True)
        print(flush=True)
        del a, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
