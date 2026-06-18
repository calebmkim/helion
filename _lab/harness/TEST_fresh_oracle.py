"""Fresh-oracle re-validation (ledger-keeper terminal step).

For a small mix of IN-SAMPLE + TEST shapes across several kernels: run a FRESH
Helion oracle (autotune, full effort + tc baseline), re-bench the FULL VERBATIM
winning config fairly (all levers together, our own do_bench), and compare to the
FROZEN v7 heuristic seed. Confirms (a) the oracle hasn't drifted, (b) the v7 seed
is still near the oracle ceiling (seed/oracle ratio), (c) the champion still holds.

Effort is configurable; default 'full' per brief, with a generation cap to bound
wall-clock. Reports tc_lat, seed_lat, oracle_lat, G_seed, G_oracle, seed/oracle.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
N_RUNS = 7
LEVERS = ["block_sizes", "reduction_loops", "num_warps", "num_stages"]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def geomean(xs):
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


# ---- specs: fn, args, ref, out-unwrap, correct ----
def _unwrap(o):
    return o[0] if isinstance(o, tuple) else o


def spec(kernel):
    if kernel == "rms_norm":
        from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch
        def args(s):
            m, n = s
            return (torch.randn(m, n, device="cuda", dtype=torch.float32),
                    torch.randn(n, device="cuda", dtype=torch.float32), EPS)
        return dict(fn=rms_norm_fwd, args=args, ref=lambda a: rms_norm_pytorch(*a),
                    out=_unwrap, tol=(1e-3, 1e-4))
    if kernel == "softmax":
        from examples.softmax import softmax_two_pass
        def args(s):
            m, n = s
            x = torch.randn(m, n, device="cuda", dtype=torch.float32)
            assert x.dtype == torch.float32
            return (x,)
        return dict(fn=softmax_two_pass, args=args,
                    ref=lambda a: torch.nn.functional.softmax(a[0], dim=1),
                    out=_unwrap, tol=(1e-3, 1e-4))
    if kernel == "kl_div":
        from examples.kl_div import kl_div_forward
        def args(s):
            bt, v = s
            return (torch.randn(bt, v, device="cuda").log_softmax(-1),
                    torch.randn(bt, v, device="cuda").softmax(-1), False, "batchmean", 1e-10)
        return dict(fn=kl_div_forward, args=args,
                    ref=lambda a: torch.nn.KLDivLoss(reduction="batchmean").to("cuda")(a[0], a[1]),
                    out=_unwrap, tol=("scalar", None))
    if kernel == "welford":
        from examples.welford import welford, eager_layer_norm
        def args(s):
            m, n = s
            return (torch.rand(n, device="cuda", dtype=torch.float32),
                    torch.rand(n, device="cuda", dtype=torch.float32),
                    torch.rand(m, n, device="cuda", dtype=torch.float32), EPS)
        return dict(fn=welford, args=args, ref=lambda a: eager_layer_norm(*a),
                    out=_unwrap, tol=(1e-3, 1e-3))
    raise ValueError(kernel)


def correct(out, ref, tol):
    if tol[0] == "scalar":
        return abs(float(out) - float(ref)) / (abs(float(ref)) + 1e-12) < 1e-3
    o = out.to(torch.float32); r = ref.to(torch.float32)
    return bool(torch.allclose(o, r, rtol=tol[0], atol=tol[1]))


# (kernel, shape, "in_sample" | "test")
CASES = [
    ("rms_norm", (8192, 8192), "in_sample"),
    ("rms_norm", (4096, 10240), "test"),
    ("softmax", (8192, 8192), "test"),
    ("kl_div", (4096, 24576), "test"),
    ("welford", (131072, 2048), "test"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effort", default="full")
    a = ap.parse_args()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} helion={helion.__file__} effort={a.effort} (fresh-oracle re-validation)\n",
          flush=True)
    rows = []
    for kernel, shape, origin in CASES:
        sp = spec(kernel)
        args = sp["args"](shape)
        ref = sp["ref"](args)

        bound = sp["fn"].bind(args)
        seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
        seeded = helion.kernel(sp["fn"].fn, configs=[helion.Config(**seed)])
        bs = seeded.bind(args); bs.ensure_config_exists(args)
        seed_ok = correct(sp["out"](bs(*args)), ref, sp["tol"])
        seed_lat = med(lambda: bs(*args))

        torch._dynamo.reset()
        tc = torch.compile(lambda *aa: sp["ref"](aa))
        tc(*args)
        tc_lat = med(lambda: tc(*args))

        os.environ["HELION_AUTOTUNE_EFFORT"] = a.effort
        os.environ["HELION_FORCE_AUTOTUNE"] = "1"
        k = helion.kernel(sp["fn"].fn)
        bound_o = k.bind(args)
        bound_o.autotune(args)
        oracle_cfg = dict(bound_o._config)
        oracle_ok = correct(sp["out"](bound_o(*args)), ref, sp["tol"])
        oracle_lat = med(lambda: bound_o(*args))

        g_seed = tc_lat / seed_lat
        g_oracle = tc_lat / oracle_lat
        s_over_o = oracle_lat / seed_lat  # >1 => seed slower than oracle
        rows.append((kernel, shape, origin, g_seed, g_oracle, s_over_o, seed_ok, oracle_ok))
        print(f"=== {kernel}{shape} [{origin}]  seed_ok={seed_ok} oracle_ok={oracle_ok} ===", flush=True)
        print(f"  tc={tc_lat*1000:.1f}us seed={seed_lat*1000:.1f}us(G={g_seed:.3f}) "
              f"oracle={oracle_lat*1000:.1f}us(G={g_oracle:.3f})  seed/oracle={1/s_over_o:.3f}", flush=True)
        for lever in LEVERS:
            sv, ov = seed.get(lever), oracle_cfg.get(lever)
            mark = "" if sv == ov else "  <-- DIFF"
            print(f"    {lever:>16} seed={str(sv):>16} oracle={str(ov):>20}{mark}", flush=True)
        print(flush=True)

    print("=" * 70, flush=True)
    print(f"{'kernel/shape':>26} {'origin':>10} {'G_seed':>8} {'G_oracle':>9} {'seed/oracle':>12}",
          flush=True)
    for kernel, shape, origin, gs, go, soo, *_ in rows:
        print(f"{kernel+str(shape):>26} {origin:>10} {gs:>8.3f} {go:>9.3f} {1/soo:>12.3f}",
              flush=True)
    print(f"\nGEOMEAN seed/oracle = {geomean([1/r[5] for r in rows]):.3f} "
          f"(1.0 = seed AT oracle ceiling; <1 = seed below oracle)", flush=True)


if __name__ == "__main__":
    main()
