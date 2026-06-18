"""WS2 backward bench for the FOUR NEW m-reduction curriculum kernels
(bias_grad, dyt, group_norm, instance_norm) — bare backward-kernel seed/cfg/oracle/tc,
same-process. Mirrors `_lab/bench/bwd_bench.py` (validated bare==tritonbench) for kernels
that are NOT tritonbench operators: the bare backward kernel IS the bench path + accuracy
gate the task requires.

All method §4 backward footguns handled:
  * saved tensors (mean/rstd) are the REAL forward values (from the torch reference), so the
    accuracy gate vs the autograd reference is valid.
  * accuracy gate FIRST, vs autograd reference at the SAME dtype (upcast fp32; max_abs).
  * same-process do_bench, median-of-N; dtype forced. dynamo reset per shape.
  * records the NORMALIZED running config (proves configs=[cfg] ran, no autotune).
  * oracle = bound.autotune(args, force=True).

Usage (cwd=/tmp, PYTHONPATH=<worktree>):
  python mr_bench.py --mode seedtc --dtype fp32 --split train group_norm
  python mr_bench.py --mode cfg --cfg '{"block_sizes":[4,1],...}' --shapes "512,128,64,32" group_norm
  python mr_bench.py --mode oracle --dtype fp32 --shapes "512,64,128" instance_norm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from triton.testing import do_bench

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

_HERE = os.path.dirname(os.path.abspath(__file__))
_WT = os.environ.get("HELION_WORKTREE", os.path.abspath(os.path.join(_HERE, "..", "..")))
if _WT not in sys.path:
    sys.path.insert(0, _WT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mreduction_shapes as MShapes  # noqa: E402
import mreduction_styles as MS  # noqa: E402

assert os.path.realpath(helion.__file__).startswith(os.path.realpath(_WT) + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT}); set PYTHONPATH=<worktree>."
)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
EPS = 1e-5
N_RUNS = 11
TOL = {"fp32": 2e-3, "bf16": 3e-2, "fp16": 3e-2}
ALPHA = 0.7


def _med(fn) -> float:
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[
        N_RUNS // 2
    ] * 1000.0  # us


# ---- builders: return (fn, args, ref_grads, out_names, tc_bwd_fn) ----

def build_bias_grad(shape, dt):
    m, n = shape
    go = torch.randn(m, n, device="cuda", dtype=dt)
    args = (go,)
    ref = (go.float().sum(0).to(dt),)
    x = torch.randn(m, n, device="cuda", dtype=dt)
    bias = torch.randn(n, device="cuda", dtype=dt, requires_grad=True)
    fwd = torch.compile(lambda a, b: a + b)
    yt = fwd(x, bias)

    def tc_bwd():
        return torch.autograd.grad(yt, [bias], go, retain_graph=True)

    return MS.bias_grad_bwd, args, ref, ("grad_bias",), tc_bwd


def build_dyt(shape, dt):
    m, n = shape
    go = torch.randn(m, n, device="cuda", dtype=dt)
    x = torch.randn(m, n, device="cuda", dtype=dt)
    w = torch.randn(n, device="cuda", dtype=dt)
    bvec = torch.randn(n, device="cuda", dtype=dt)
    args = (go, x, w, ALPHA)
    rgx, rgw, rgb = MS.dyt_ref(go, x, w, bvec, ALPHA)
    ref = (rgx, rgw, rgb)
    xt = x.detach().clone().requires_grad_(True)
    wt = w.detach().clone().requires_grad_(True)
    bt = bvec.detach().clone().requires_grad_(True)
    fwd = torch.compile(lambda a, b, c: b[None, :] * torch.tanh(ALPHA * a) + c[None, :])
    yt = fwd(xt, wt, bt)

    def tc_bwd():
        return torch.autograd.grad(yt, [xt, wt, bt], go, retain_graph=True)

    return MS.dyt_bwd, args, ref, ("grad_x", "grad_weight", "grad_bias"), tc_bwd


def build_group_norm(shape, dt):
    nn, c, s, g = shape
    x = torch.randn(nn, c, s, device="cuda", dtype=dt)
    go = torch.randn(nn, c, s, device="cuda", dtype=dt)
    w = torch.randn(c, device="cuda", dtype=dt)
    bvec = torch.randn(c, device="cuda", dtype=dt)
    rgx, rgw, rgb, mean, rstd = MS.group_norm_ref(go, x, w, bvec, g, EPS)
    args = (go, x, mean, rstd, w, g)
    ref = (rgx, rgw, rgb)
    xt = x.detach().clone().requires_grad_(True)
    wt = w.detach().clone().requires_grad_(True)
    bt = bvec.detach().clone().requires_grad_(True)
    fwd = torch.compile(lambda a, b, c: torch.nn.functional.group_norm(a, g, b, c, EPS))
    yt = fwd(xt, wt, bt)

    def tc_bwd():
        return torch.autograd.grad(yt, [xt, wt, bt], go, retain_graph=True)

    return MS.group_norm_bwd, args, ref, ("grad_x", "grad_weight", "grad_bias"), tc_bwd


def build_instance_norm(shape, dt):
    bb, c, s = shape
    x = torch.randn(bb, c, s, device="cuda", dtype=dt)
    go = torch.randn(bb, c, s, device="cuda", dtype=dt)
    w = torch.randn(c, device="cuda", dtype=dt)
    bvec = torch.randn(c, device="cuda", dtype=dt)
    rgx, rgw, rgb, mean, rstd = MS.instance_norm_ref(go, x, w, bvec, EPS)
    args = (go, x, mean, rstd, w)
    ref = (rgx, rgw, rgb)
    xt = x.detach().clone().requires_grad_(True)
    wt = w.detach().clone().requires_grad_(True)
    bt = bvec.detach().clone().requires_grad_(True)
    fwd = torch.compile(
        lambda a, b, c: torch.nn.functional.instance_norm(a, weight=b, bias=c, eps=EPS)
    )
    yt = fwd(xt, wt, bt)

    def tc_bwd():
        return torch.autograd.grad(yt, [xt, wt, bt], go, retain_graph=True)

    return MS.instance_norm_bwd, args, ref, ("grad_x", "grad_weight", "grad_bias"), tc_bwd


BUILDERS = {
    "bias_grad": build_bias_grad,
    "dyt": build_dyt,
    "group_norm": build_group_norm,
    "instance_norm": build_instance_norm,
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


def run_shape(name, shape, dt_name, mode, cfg_override=None):
    dt = DTYPES[dt_name]
    fn, args, ref, out_names, tc_bwd = BUILDERS[name](shape, dt)
    tol = TOL[dt_name]
    row = {"kernel": name, "dtype": dt_name, "shape": list(shape), "mode": mode}

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
        ko, out_o, onorm = _bind_cfg(fn, args, helion.Config(**ocfg))
        ok_o, ma_o = _accuracy(out_o, ref, tol)
        row.update({"oracle_ran": onorm, "oracle_acc": ok_o, "oracle_max_abs": ma_o})
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
    ap.add_argument("--shapes", default=None, help='"a,b,c;d,e,f" explicit shapes')
    ap.add_argument("--split", default=None, help="train/val/test/robustness")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--out", default="/tmp/mr_bench.json")
    ap.add_argument("kernels", nargs="*", default=list(BUILDERS))
    a = ap.parse_args()
    print(f"helion={helion.__file__} mode={a.mode} dtype={a.dtype} split={a.split}", flush=True)
    explicit = None
    if a.shapes:
        explicit = [tuple(int(v) for v in p.split(",")) for p in a.shapes.split(";")]
    out = []
    for name in (a.kernels or list(BUILDERS)):
        shapes = MShapes.SHAPES[name][a.split] if a.split else explicit
        for shape in shapes:
            torch._dynamo.reset()
            r = run_shape(name, tuple(shape), a.dtype, a.mode, a.cfg)
            out.append(r)
            print("ROW " + json.dumps(r), flush=True)
            json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
