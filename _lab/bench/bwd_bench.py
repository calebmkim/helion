"""WS2 backward bench workhorse — bare backward-kernel seed/cfg/oracle/tc, same-process.

Drives the Helion BACKWARD kernels (rms_norm_bwd / layer_norm_bwd / softmax_bwd) DIRECTLY
(configs=[cfg], no autograd wrapper) — this times exactly the kernel the seed controls, which
is what the autotuner oracle searches. tritonbench `-bwd` (run_seeded) stays the bankable
headline G; this is the design/attribution + oracle tool (method §4: hand-rolled = cross-check).

All §4 backward footguns handled:
  * saved tensors (rsqrt / mean,rstd / softmax_output) are the REAL forward values, so the
    accuracy gate vs the autograd reference is valid (random saved tensors give wrong grads).
  * accuracy gate FIRST, vs autograd reference at the SAME dtype (upcast fp32 for allclose;
    max_abs, not max_rel, to dodge near-zero traps). A failing cell is never reported as a win.
  * same-process do_bench, median-of-N; dtype forced + asserted.
  * records the NORMALIZED running config (proves configs=[cfg] ran, no autotune).
  * the autotuner oracle is `bound.autotune(args, force=True)` (run3_oracle pattern).

Usage (cwd=/tmp, PYTHONPATH=<worktree>):
  python bwd_bench.py --mode seedtc --dtype fp32 --shapes 4096,4096 rms_norm_bwd
  python bwd_bench.py --mode oracle --dtype fp32 --shapes 4096,4096 rms_norm_bwd
  python bwd_bench.py --mode cfg --cfg '{"block_sizes":[...],...}' --shapes 4096,4096 rms_norm_bwd
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import torch
from triton.testing import do_bench

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

_WT = os.environ.get(
    "HELION_WORKTREE",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
if _WT not in sys.path:
    sys.path.insert(0, _WT)
sys.path.insert(0, os.path.join(_WT, "_lab", "prompts"))

import shapes_v3_draft as SH  # noqa: E402

from examples.layer_norm import layer_norm_bwd  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_bwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402
from examples.softmax import softmax_bwd  # noqa: E402

assert os.path.realpath(helion.__file__).startswith(os.path.realpath(_WT) + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT}); set PYTHONPATH=<worktree>."
)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
EPS = 1e-5
N_RUNS = 11
TOL = {"fp32": 2e-3, "bf16": 3e-2, "fp16": 2e-2}


def _med(fn) -> float:
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[
        N_RUNS // 2
    ] * 1000.0  # us


# ---- build fns: return (fn, args, ref_grads, out_names, tc_bwd_fn) ----
# args = the bare backward-kernel inputs; ref_grads = autograd-reference grads (tuple,
# aligned to the kernel's returned grads); tc_bwd_fn = a bare backward-only callable for tc.

def build_rms_bwd(m, n, dt):
    x = torch.randn(m, n, device="cuda", dtype=dt)
    weight = torch.randn(n, device="cuda", dtype=dt)
    grad_out = torch.randn(m, n, device="cuda", dtype=dt)
    # REAL inv_rms from the forward (so grads are correct).
    inv_rms = torch.rsqrt(x.float().pow(2).mean(-1) + EPS).reshape(m, 1)
    args = (grad_out, x, weight, inv_rms)

    # autograd reference grads (grad_x, grad_weight)
    xr = x.detach().clone().requires_grad_(True)
    wr = weight.detach().clone().requires_grad_(True)
    y = rms_norm_pytorch(xr, wr, EPS)
    gx, gw = torch.autograd.grad(y, [xr, wr], grad_out)
    ref = (gx, gw)

    # bare tc backward-only
    xt = x.detach().clone().requires_grad_(True)
    wt = weight.detach().clone().requires_grad_(True)
    fwd = torch.compile(lambda a, b: rms_norm_pytorch(a, b, EPS))
    yt = fwd(xt, wt)

    def tc_bwd():
        return torch.autograd.grad(yt, [xt, wt], grad_out, retain_graph=True)

    return rms_norm_bwd, args, ref, ("grad_x", "grad_weight"), tc_bwd


def build_ln_bwd(m, n, dt):
    x = torch.randn(m, n, device="cuda", dtype=dt)
    weight = torch.randn(n, device="cuda", dtype=dt)
    bias = torch.randn(n, device="cuda", dtype=dt)
    grad_out = torch.randn(m, n, device="cuda", dtype=dt)
    mean = x.float().mean(-1)
    var = x.float().var(-1, unbiased=False)
    rstd = torch.rsqrt(var + EPS)
    args = (grad_out, x, mean, rstd, weight)  # compute_bias_grad defaults True

    xr = x.detach().clone().requires_grad_(True)
    wr = weight.detach().clone().requires_grad_(True)
    br = bias.detach().clone().requires_grad_(True)
    y = torch.nn.functional.layer_norm(xr, [n], wr, br, EPS)
    gx, gw, gb = torch.autograd.grad(y, [xr, wr, br], grad_out)
    ref = (gx, gw, gb)

    xt = x.detach().clone().requires_grad_(True)
    wt = weight.detach().clone().requires_grad_(True)
    bt = bias.detach().clone().requires_grad_(True)
    fwd = torch.compile(lambda a, b, c: torch.nn.functional.layer_norm(a, [n], b, c, EPS))
    yt = fwd(xt, wt, bt)

    def tc_bwd():
        return torch.autograd.grad(yt, [xt, wt, bt], grad_out, retain_graph=True)

    return layer_norm_bwd, args, ref, ("grad_x", "grad_weight", "grad_bias"), tc_bwd


def build_softmax_bwd(m, n, dt):
    x = torch.randn(m, n, device="cuda", dtype=dt)
    so = torch.softmax(x, dim=1)
    grad_out = torch.randn(m, n, device="cuda", dtype=dt)
    args = (grad_out, so)

    xr = x.detach().clone().requires_grad_(True)
    y = torch.softmax(xr, dim=1)
    (gx,) = torch.autograd.grad(y, [xr], grad_out)
    ref = (gx,)

    xt = x.detach().clone().requires_grad_(True)
    fwd = torch.compile(lambda a: torch.softmax(a, dim=1))
    yt = fwd(xt)

    def tc_bwd():
        return torch.autograd.grad(yt, [xt], grad_out, retain_graph=True)

    return softmax_bwd, args, ref, ("grad_x",), tc_bwd


BUILDERS = {
    "rms_norm_bwd": build_rms_bwd,
    "layer_norm_bwd": build_ln_bwd,
    "softmax_bwd": build_softmax_bwd,
}

# backward kernel -> the forward curriculum key whose shapes apply (user-confirmed: the
# backward kernels reuse the rms_norm / layer_norm / softmax forward shape suites).
CURRICULUM = {
    "rms_norm_bwd": "rms_norm",
    "layer_norm_bwd": "layer_norm",
    "softmax_bwd": "softmax",
}


def _as_tuple(o):
    return o if isinstance(o, tuple) else (o,)


def _accuracy(out, ref, tol):
    out = _as_tuple(out)
    ref = _as_tuple(ref)
    max_abs = 0.0
    ok = True
    for o, r in zip(out, ref):
        if o is None or r is None:
            continue
        d = (o.float() - r.float()).abs().max().item()
        max_abs = max(max_abs, d)
        if not torch.allclose(o.float(), r.float(), rtol=tol, atol=tol):
            ok = False
    return ok, round(max_abs, 6)


def _bind_cfg(fn, args, cfg):
    k = helion.kernel(fn.fn, config=cfg, static_shapes=True)
    out = k(*args)
    norm = str(k.bind(args)._config)
    return k, out, norm


def run_shape(name, m, n, dt_name, mode, cfg_override=None):
    dt = DTYPES[dt_name]
    fn, args, ref, out_names, tc_bwd = BUILDERS[name](m, n, dt)
    tol = TOL[dt_name]
    row = {"kernel": name, "dtype": dt_name, "shape": [m, n], "mode": mode}

    # seed config
    bound0 = fn.bind(args)
    seeds = compiler_seed_configs(bound0.env, bound0.host_function.device_ir)
    seed = seeds[0] if seeds else bound0.config_spec.default_config()
    row["n_seeds"] = len(seeds)
    row["seed_cfg"] = str(dict(seed))

    if mode in ("seedtc", "cfg"):
        cfg = helion.Config(**json.loads(cfg_override)) if (mode == "cfg") else seed
        k, out, norm = _bind_cfg(fn, args, cfg)
        ok, max_abs = _accuracy(out, ref, tol)
        row.update({"ran_cfg": norm, "acc": ok, "max_abs": max_abs, "tol": tol})
        if ok:
            row["seed_us"] = round(_med(lambda: k(*args)), 3)
        # tc bare backward (always, for the floor)
        row["tc_us"] = round(_med(tc_bwd), 3)
        if row.get("seed_us"):
            row["G"] = round(row["tc_us"] / row["seed_us"], 4)

    elif mode == "oracle":
        t0 = time.time()
        k_at = helion.kernel(fn.fn)
        bk = k_at.bind(args)
        oracle_cfg = bk.autotune(args, force=True)
        row["autotune_s"] = round(time.time() - t0, 1)
        ocfg = dict(oracle_cfg)
        row["oracle_cfg"] = str(ocfg)
        # re-bench oracle bare + seed bare + tc bare, same process
        ko, out_o, onorm = _bind_cfg(fn, args, helion.Config(**ocfg))
        ok_o, ma_o = _accuracy(out_o, ref, tol)
        row["oracle_ran"] = onorm
        row["oracle_acc"] = ok_o
        row["oracle_max_abs"] = ma_o
        ks, out_s, snorm = _bind_cfg(fn, args, seed)
        ok_s, ma_s = _accuracy(out_s, ref, tol)
        row["seed_acc"] = ok_s
        row["oracle_us"] = round(_med(lambda: ko(*args)), 3) if ok_o else None
        row["seed_us"] = round(_med(lambda: ks(*args)), 3) if ok_s else None
        row["tc_us"] = round(_med(tc_bwd), 3)
        if row.get("oracle_us"):
            row["oracle_vs_tc"] = round(row["tc_us"] / row["oracle_us"], 4)
        if row.get("seed_us"):
            row["G_seed"] = round(row["tc_us"] / row["seed_us"], 4)
        if row.get("oracle_us") and row.get("seed_us"):
            row["seed_over_oracle"] = round(row["seed_us"] / row["oracle_us"], 4)
        # field diff seed -> oracle
        diff = {}
        sd = dict(seed)
        for kk in sorted(set(sd) | set(ocfg)):
            if sd.get(kk) != ocfg.get(kk):
                diff[kk] = {"seed": sd.get(kk), "oracle": ocfg.get(kk)}
        row["field_diff"] = diff

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="seedtc", choices=["seedtc", "cfg", "oracle"])
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("--shapes", default="4096,4096", help="M,N;M,N (ignored if --split set)")
    ap.add_argument("--split", default=None,
                    help="pull shapes from the forward curriculum split (train/val/test/robustness)")
    ap.add_argument("--cfg", default=None, help="JSON config for --mode cfg")
    ap.add_argument("--out", default="/tmp/ws2_bwd_bench.json")
    ap.add_argument("kernels", nargs="*", default=list(BUILDERS))
    a = ap.parse_args()
    print(f"helion={helion.__file__} mode={a.mode} dtype={a.dtype} split={a.split}", flush=True)
    explicit = [tuple(int(v) for v in p.split(",")) for p in a.shapes.split(";")]
    out = []
    for name in (a.kernels or list(BUILDERS)):
        shapes = SH.SHAPES[CURRICULUM[name]][a.split] if a.split else explicit
        for (m, n) in shapes:
            torch._dynamo.reset()
            r = run_shape(name, m, n, a.dtype, a.mode, a.cfg)
            out.append(r)
            print("ROW " + json.dumps(r), flush=True)
            json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
