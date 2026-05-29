"""Referee-OWN per-shape G measurement for the 3 new T2 kernels.
ONE shape per process invocation (fresh process => no cross-shape cache/JIT
interference). Reports seed_lat, default_lat, tc_default_lat (median-of-N
do_bench), G_seed, G_default, and correctness max_abs + max_rel + tol.

Usage:
  python REFEREE_v5_measure_g.py --kernel softmax|kl_div|jsd --m M --n N

Refs:
  softmax: F.softmax(x, dim=1) fp32
  kl_div : torch.nn.KLDivLoss(batchmean, log_target=False)
  jsd    : examples.jsd.TorchJSDBaseline(beta=0.5, ignore_index=-100) loss
"""
from __future__ import annotations
import argparse, json, os, sys
import torch
import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WT), helion.__file__
sys.path.insert(0, WT)

from triton.testing import do_bench
from helion._compiler.autotuner_heuristics import compiler_seed_configs

N_RUNS = 9


def median_bench(fn):
    torch.cuda.synchronize()
    xs = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    xs.sort()
    return xs[len(xs) // 2], xs[0], xs[-1]


def build(fn, cfg_dict, args):
    k = helion.kernel(fn.fn, configs=[helion.Config(**cfg_dict)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b


def get_seed(fn, args):
    b = fn.bind(args)
    seeds = compiler_seed_configs(b.env, b.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    return dict(seeds[0]), b


def errs(out, ref):
    o = out.to(torch.float32).flatten()
    r = ref.to(torch.float32).flatten()
    max_abs = float((o - r).abs().max())
    denom = r.abs().clamp_min(1e-12)
    max_rel = float(((o - r).abs() / denom).max())
    return max_abs, max_rel


def setup(kernel, m, n):
    if kernel == "softmax":
        from examples.softmax import softmax_two_pass as fn
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        args = (x,)
        ref = torch.nn.functional.softmax(x, dim=1)
        getout = lambda o: o
        tc_callable = lambda: torch.nn.functional.softmax(x, dim=1)
        return fn, args, ref, getout, tc_callable
    if kernel == "kl_div":
        from examples.kl_div import kl_div_forward as fn
        yp = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1)
        yt = torch.randn(m, n, device="cuda", dtype=torch.float32).softmax(-1)
        args = (yp, yt, False, "batchmean", 1e-10)
        ref = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to("cuda")(yp, yt)
        getout = lambda o: o
        tc_callable = lambda: torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to("cuda")(yp, yt)
        return fn, args, ref, getout, tc_callable
    if kernel == "jsd":
        from examples.jsd import jsd_forward as fn
        from examples.jsd import TorchJSDBaseline
        lq = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1)
        lp = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1)
        args = (lq, lp, None, 0.5, -100)
        baseline = TorchJSDBaseline(beta=0.5, ignore_index=-100).to("cuda")
        ref = baseline(lq, lp)
        getout = lambda o: (o[0] if isinstance(o, tuple) else o)
        tc_callable = lambda: baseline(lq, lp)
        return fn, args, ref, getout, tc_callable
    raise ValueError(kernel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=["softmax", "kl_div", "jsd"])
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    torch.manual_seed(0)

    fn, args, ref, getout, tc_callable = setup(a.kernel, a.m, a.n)
    ref_loss = getout(ref) if a.kernel != "softmax" else ref

    # --- seed ---
    seed, bound0 = get_seed(fn, args)
    bs = build(fn, seed, args)
    out_s = getout(bs(*args))
    ms_abs, ms_rel = errs(out_s, ref_loss)
    seed_lat, seed_lo, seed_hi = median_bench(lambda: bs(*args))

    # --- default (un-seeded) ---
    cfgd = dict(bound0.config_spec.default_config())
    bd = build(fn, cfgd, args)
    out_d = getout(bd(*args))
    md_abs, md_rel = errs(out_d, ref_loss)
    default_lat, _, _ = median_bench(lambda: bd(*args))

    # --- torch.compile default-mode of the reference ---
    torch._dynamo.reset()
    tc = torch.compile(tc_callable)
    _ = tc()
    tc_lat, tc_lo, tc_hi = median_bench(tc)

    spread = (seed_hi - seed_lo) / seed_lat * 100.0
    result = {
        "kernel": a.kernel, "shape": [a.m, a.n], "seed": seed,
        "seed_lat_us": seed_lat * 1000, "seed_spread_pct": spread,
        "default_lat_us": default_lat * 1000, "tc_lat_us": tc_lat * 1000,
        "g_seed": tc_lat / seed_lat, "g_default": tc_lat / default_lat,
        "seed_max_abs": ms_abs, "seed_max_rel": ms_rel,
        "default_max_abs": md_abs, "default_max_rel": md_rel,
    }
    print("REFEREE_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main()
