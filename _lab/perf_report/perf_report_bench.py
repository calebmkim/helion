"""Unified 3-arm perf-report harness for the reduction-seed heuristic.

Measurement + reporting ONLY (not a hill-climb). Per (corpus, kernel, shape, dtype)
times up to 3 arms IN ONE PROCESS on the SAME input tensors (footgun #4), by explicit
config replay (footgun #7 — the env flags are process-global, so we extract+replay
configs instead of flipping flags mid-process):

  seed    = helion.kernel(fn.fn, config=compiler_seed_configs(...)[0])   [the heuristic]
  default = helion.kernel(fn.fn, config=spec._base_default_config())     [unseeded base]
            NB: NOT default_config() — the reduction heuristic sets promote_seed_to_default,
            so default_config() would return the SEED. _base_default_config() is the true base.
  tc      = torch.compile(reference)  DEFAULT mode (real corpora only; footgun #3)

Metric: cold-L2 median-of-9 triton do_bench (this triton build's do_bench clears L2 before
every rep — verified). Rerun >5% spread rows at median-of-15 (footgun #13). Accuracy-gate
before timing vs an eager reference at the SAME dtype, upcast to fp32 (footgun #6). Forward
only, requires_grad=False (footgun #1). dynamo-reset per shape (footgun #2). Fresh process
per kernel via the CLI (footgun #11).

Arms:
  real corpora (curriculum, transfer, mreduction, vllm): seed / default / tc  -> G_tc, G_def
  synthetic + adversarial:                               seed / default       -> G_def only

Run (fresh process per kernel), from /tmp:
  HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-redesign \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/perf_report/perf_report_bench.py \
    --corpus curriculum --kernel rms_norm --out-dir <RESULTS>
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import traceback

# --------------------------------------------------------------------------- #
#  Path wiring (portable; discover prompts-lab as a sibling of the worktree).
# --------------------------------------------------------------------------- #
_THIS = os.path.abspath(__file__)
_WT_ROOT = os.path.abspath(os.path.join(os.path.dirname(_THIS), "..", ".."))
_LOCAL_ROOT = os.path.abspath(os.path.join(_WT_ROOT, ".."))
_PL = os.path.join(_LOCAL_ROOT, "prompts-lab")
_PERF = os.path.join(_PL, "perf-report")
_KSRC = os.path.join(_PERF, "kernel_sources")

for _d in (
    _WT_ROOT,
    os.path.join(_WT_ROOT, "examples"),
    os.path.join(_WT_ROOT, "_lab", "prompts"),
    os.path.join(_WT_ROOT, "_lab", "harness"),
    os.path.join(_PL, "transfer"),
    os.path.join(_PL, "vllm-bench"),
    os.path.join(_KSRC, "vllm"),  # for `import kut...` and `refs`
):
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402
from triton.testing import do_bench  # noqa: E402

import helion  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
# NB: `helion.runtime.kernel` re-exports the kernel() FUNCTION, shadowing the submodule; import
# the Kernel CLASS explicitly via importlib to dodge the shadow (see probe_fire.py lesson).
from helion.runtime.kernel import Kernel as _HelionKernel  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT_ROOT}); set PYTHONPATH."
)

EPS = 1e-5
DEV = "cuda"
_DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------- #
#  Timing primitives (cold-L2 do_bench; median-of-medians; noise-aware rerun).
# --------------------------------------------------------------------------- #
def _med(fn, n: int) -> tuple[float, float]:
    """Return (median_us, spread_frac) of n independent do_bench-median samples."""
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[n // 2] * 1e3  # ms -> us
    spread = (s[-1] - s[0]) / s[n // 2] if s[n // 2] > 0 else 0.0
    return med, spread


def timed(fn) -> dict:
    """median-of-9, escalate to median-of-15 on >5% spread (footgun #13)."""
    med, spread = _med(fn, 9)
    if spread > 0.05:
        med, spread = _med(fn, 15)
    return {"us": round(med, 3), "spread": round(spread, 4)}


def _geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


# --------------------------------------------------------------------------- #
#  Config extraction (seed + unseeded base), faithful replay.
# --------------------------------------------------------------------------- #
def _extract_configs(kfn, args) -> tuple[object, object, list]:
    """Return (seed_cfg_or_None, base_default_cfg, fired_names). Bind once."""
    bound = kfn.bind(args)
    spec = bound.env.config_spec
    seeds = list(spec.compiler_seed_configs)  # persisted during bind (inside env ctx)
    fired = list(spec.autotuner_heuristics)
    with bound.env:
        base_default = spec._base_default_config()
    seed = seeds[0] if seeds else None
    del bound
    return seed, base_default, fired


def _replay(kfn, cfg):
    """Rewrap the kernel fn at a fixed config, preserving authored settings."""
    if cfg is None:
        return None
    s = kfn.settings
    return helion.kernel(
        kfn.fn,
        config=cfg,
        static_shapes=s.static_shapes,
        ignore_warnings=list(s.ignore_warnings or []),
    )


def _cfg_dict(cfg) -> dict | None:
    if cfg is None:
        return None
    try:
        return dict(cfg.config)
    except Exception:  # noqa: BLE001
        return {"repr": repr(cfg)}


# --------------------------------------------------------------------------- #
#  Cell spec: each corpus builds a per-cell descriptor.
#    kfn, args, ref (eager, same-dtype), acc(out, ref)->(ok, detail),
#    tc_ref (zero-arg callable to compile) or None, clone_for_timing (bool)
# --------------------------------------------------------------------------- #
def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def _acc_close(rtol, atol, extract=_first):
    def acc(out, ref):
        o = extract(out).to(torch.float32)
        r = (_first(ref) if isinstance(ref, (tuple, list)) else ref).to(torch.float32)
        ok = bool(torch.allclose(o, r, rtol=rtol, atol=atol))
        maxabs = float((o - r).abs().max())
        return ok, f"maxabs={maxabs:.3e}"
    return acc


# ---- CURRICULUM (dtype-parameterized rebuild of run2_measure_g builders) ----
def _cur_build(kernel, m, n, dt):
    import examples.rms_norm as RN
    import examples.layer_norm as LN
    import examples.softmax as SM
    import examples.welford as WF
    import examples.cross_entropy as CE
    import examples.kl_div as KL
    import examples.jsd as JSD
    import examples.sum as SUM
    import examples.long_sum as LS

    def rn(*s):
        return torch.randn(*s, device=DEV, dtype=dt)

    tol = (2e-2, 2e-2) if dt != torch.float32 else (1e-3, 1e-4)
    if kernel == "rms_norm":
        x, w = rn(m, n), rn(n)
        args = (x, w, EPS)
        ref = RN.rms_norm_pytorch(x, w, EPS)
        return RN.rms_norm_fwd, args, ref, _acc_close(*tol), lambda: RN.rms_norm_pytorch(x, w, EPS)
    if kernel == "layer_norm":
        x, w, b = rn(m, n), rn(n), rn(n)
        args = (x, [n], w, b, EPS)
        ref = torch.nn.functional.layer_norm(x, [n], w, b, EPS)
        return (LN.layer_norm_fwd, args, ref, _acc_close(*tol),
                lambda: torch.nn.functional.layer_norm(x, [n], w, b, EPS))
    if kernel == "welford":
        w, b, x = torch.rand(n, device=DEV, dtype=dt), torch.rand(n, device=DEV, dtype=dt), torch.rand(m, n, device=DEV, dtype=dt)
        args = (w, b, x, EPS)
        ref = WF.eager_layer_norm(*args)
        return WF.welford, args, ref, _acc_close(*tol), lambda: WF.eager_layer_norm(*args)
    if kernel == "softmax":
        x = rn(m, n)
        args = (x,)
        ref = torch.nn.functional.softmax(x, dim=1)
        return SM.softmax_two_pass, args, ref, _acc_close(*tol), lambda: torch.nn.functional.softmax(x, dim=1)
    if kernel == "cross_entropy":
        lg = rn(m, n)
        lb = torch.randint(0, n, (m,), device=DEV, dtype=torch.int64)
        args = (lg, lb)
        ref = torch.nn.functional.cross_entropy(lg, lb)
        # CE loss can be near-zero-ish; use loose tol at bf16
        ctol = (2e-2, 2e-2) if dt != torch.float32 else (1e-3, 1e-3)
        return CE.cross_entropy, args, ref, _acc_close(*ctol), lambda: torch.nn.functional.cross_entropy(lg, lb)
    if kernel == "kl_div":
        yp = rn(m, n).log_softmax(-1)
        yt = rn(m, n).softmax(-1)
        args = (yp, yt, False, "batchmean", 1e-10)
        ref = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(DEV)(yp, yt)
        return (KL.kl_div_forward, args, ref, _acc_close(2e-2, 2e-2),
                lambda: torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(DEV)(yp, yt))
    if kernel == "jsd":
        # jsd_forward returns (loss, dX); compare loss only (footgun #6c). tc computes loss only
        # -> Helion times an EXTRA dX output tc doesn't -> G_tc is CONSERVATIVE (biased against seed).
        lq = rn(m, n).log_softmax(-1)
        lp = rn(m, n).log_softmax(-1)
        args = (lq, lp, None, 0.5, -100)
        baseline = JSD.TorchJSDBaseline(beta=0.5, ignore_index=-100)
        ref = baseline(lq, lp)
        return (JSD.jsd_forward, args, ref, _acc_close(2e-2, 2e-2), lambda: baseline(lq, lp))
    if kernel == "sum":
        x = rn(m, n)
        args = (x,)
        ref = torch.sum(x, dim=-1)
        return SUM.sum_kernel, args, ref, _acc_close(*tol), lambda: torch.sum(x, dim=-1)
    if kernel == "long_sum":
        x = rn(m, n)
        args = (x,)
        ref = torch.sum(x, dim=-1)
        return LS.longsum, args, ref, _acc_close(*tol), lambda: torch.sum(x, dim=-1)
    raise KeyError(kernel)


# ---- TRANSFER (ab_three_arm_transfer adapters, already dtype-param) ----------
def _transfer_build(kernel, shape, dt):
    import ab_three_arm_transfer as AB

    build = AB._make(kernel)
    kfn, args, ref_callable, chk = build(tuple(shape), dt)
    ref_out = ref_callable()

    def acc(out, ref):
        try:
            chk(out, ref)
            return True, "ok"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}:{str(e)[:60]}"

    return kfn, args, ref_out, acc, ref_callable


# ---- MREDUCTION (dtype-param builders + forward-only closed-form tc refs) -----
def _mred_build(kernel, shape, dt):
    import mreduction_styles_view_only as MR
    import examples.rms_norm as RN
    import examples.layer_norm as LN

    def rn(*s):
        return torch.randn(*s, device=DEV, dtype=dt)

    tol = (3e-2, 3e-2)

    if kernel == "bias_grad_bwd":
        (M, N) = shape
        go = rn(M, N)
        args = (go,)
        ref = MR.bias_grad_ref(go)   # autograd-free already
        tc_ref = lambda: go.to(torch.float32).sum(0).to(dt)
        return MR.bias_grad_bwd, args, ref, _acc_close(*tol), tc_ref
    if kernel == "dyt_bwd":
        (M, N) = shape
        x, w, bvec, go = rn(M, N), rn(N), rn(N), rn(M, N)
        alpha = 0.7
        args = (go, x, w, alpha)
        rgx, rgw, rgb = MR.dyt_ref(go, x, w, bvec, alpha)
        ref = (rgx, rgw, rgb)

        def tc_ref():
            xf, dyf, wf = x.float(), go.float(), w.float()
            t = torch.tanh(alpha * xf)
            gx = dyf * wf[None, :] * alpha * (1 - t * t)
            gw = (dyf * t).sum(0)
            gb = dyf.sum(0)
            return gx.to(dt), gw.to(dt), gb.to(dt)

        return MR.dyt_bwd, args, ref, _acc_tuple(*tol), tc_ref
    if kernel == "rms_norm_bwd":
        (M, N) = shape
        xr, wr, gor = rn(M, N), rn(N), rn(M, N)
        rms_val = torch.rsqrt((xr.float() ** 2).mean(-1, keepdim=True) + EPS).to(dt)
        args = (gor, xr, wr, rms_val)
        ref = _rms_bwd_ref(gor, xr, wr, N)  # autograd forward-only closed form

        def tc_ref():
            return _rms_bwd_ref(gor, xr, wr, N)

        return RN.rms_norm_bwd, args, ref, _acc_tuple(*tol), tc_ref
    if kernel == "layer_norm_bwd":
        (M, N) = shape
        xl, wl, gol = rn(M, N), rn(N), rn(M, N)
        mean_l = xl.float().mean(-1)
        rstd_l = torch.rsqrt(xl.float().var(-1, unbiased=False) + EPS)
        args = (gol, xl, mean_l, rstd_l, wl)
        ref = _ln_bwd_ref(gol, xl, mean_l, rstd_l, wl, N)

        def tc_ref():
            return _ln_bwd_ref(gol, xl, mean_l, rstd_l, wl, N)

        return LN.layer_norm_bwd, args, ref, _acc_tuple(*tol), tc_ref
    if kernel == "group_norm_bwd":
        (Nn, C, S, G) = shape
        xg, wg, bg, gog = rn(Nn, C, S), rn(C), rn(C), rn(Nn, C, S)
        _, _, _, mean_g, rstd_g = MR.group_norm_ref(gog, xg, wg, bg, G)
        args = (gog, xg, mean_g, rstd_g, wg, G)
        rgx, rgw, rgb, _, _ = MR.group_norm_ref(gog, xg, wg, bg, G)
        ref = (rgx, rgw, rgb)

        def tc_ref():
            return _gn_bwd_ref(gog, xg, mean_g, rstd_g, wg, G, dt)

        return MR.group_norm_bwd, args, ref, _acc_tuple(*tol), tc_ref
    if kernel == "instance_norm_bwd":
        (Bb, C, S) = shape
        xi, wi, bi, goi = rn(Bb, C, S), rn(C), rn(C), rn(Bb, C, S)
        _, _, _, mean_i, rstd_i = MR.instance_norm_ref(goi, xi, wi, bi)
        args = (goi, xi, mean_i, rstd_i, wi)
        rgx, rgw, rgb, _, _ = MR.instance_norm_ref(goi, xi, wi, bi)
        ref = (rgx, rgw, rgb)

        def tc_ref():
            return _in_bwd_ref(goi, xi, mean_i, rstd_i, wi, dt)

        return MR.instance_norm_bwd, args, ref, _acc_tuple(*tol), tc_ref
    raise KeyError(kernel)


def _acc_tuple(rtol, atol):
    """Accuracy over a tuple of gradient tensors (compare all present)."""
    def acc(out, ref):
        outs = out if isinstance(out, (tuple, list)) else (out,)
        refs = ref if isinstance(ref, (tuple, list)) else (ref,)
        worst = 0.0
        ok_all = True
        for o, r in zip(outs, refs):
            of, rf = o.detach().to(torch.float32), r.detach().to(torch.float32)
            ok = bool(torch.allclose(of, rf, rtol=rtol, atol=atol))
            ok_all = ok_all and ok
            worst = max(worst, float((of - rf).abs().max()))
        return ok_all, f"maxabs={worst:.3e}"
    return acc


# forward-only closed-form gradient refs (port of the kernel math; NO autograd) --
def _rms_bwd_ref(grad_out, x, weight, n):
    xf, dyf = x.float(), grad_out.float()
    rsqrt = torch.rsqrt((xf ** 2).mean(-1, keepdim=True) + EPS)
    wf = weight.float()[None, :]
    gw = (xf * dyf * rsqrt).sum(0)
    gx = wf * dyf * rsqrt - xf * rsqrt ** 3 * (wf * dyf * xf).mean(-1, keepdim=True)
    return gx.to(x.dtype), gw.to(weight.dtype)


def _ln_bwd_ref(grad_out, x, mean, rstd, weight, n):
    xf, dyf = x.float(), grad_out.float()
    wf = weight.float()[None, :]
    x_hat = (xf - mean.float()[:, None]) * rstd.float()[:, None]
    gw = (dyf * x_hat).sum(0)
    gb = dyf.sum(0)
    wdy = wf * dyf
    c1 = (x_hat * wdy).sum(-1, keepdim=True) / n
    c2 = wdy.sum(-1, keepdim=True) / n
    gx = (wdy - (x_hat * c1 + c2)) * rstd.float()[:, None]
    return gx.to(x.dtype), gw.to(weight.dtype), gb.to(weight.dtype)


def _gn_bwd_ref(grad_out, x, mean, rstd, weight, num_groups, dt):
    Nn, C, S = x.shape
    G = num_groups
    Cg = C // G
    cnt = Cg * S
    xf = x.float().reshape(Nn, G, Cg, S)
    dyf = grad_out.float().reshape(Nn, G, Cg, S)
    mn = mean.float().reshape(Nn, G, 1, 1)
    rs = rstd.float().reshape(Nn, G, 1, 1)
    wg = weight.float().reshape(1, G, Cg, 1)
    x_hat = (xf - mn) * rs
    gw = (dyf * x_hat).sum(3).sum(0).reshape(C)
    gb = dyf.sum(3).sum(0).reshape(C)
    wdy = dyf * wg
    c1 = (x_hat * wdy).sum(3, keepdim=True).sum(2, keepdim=True) / cnt
    c2 = wdy.sum(3, keepdim=True).sum(2, keepdim=True) / cnt
    dx = (wdy - (x_hat * c1 + c2)) * rs
    return dx.reshape(Nn, C, S).to(dt), gw.to(dt), gb.to(dt)


def _in_bwd_ref(grad_out, x, mean, rstd, weight, dt):
    Bb, C, S = x.shape
    xf, dyf = x.float(), grad_out.float()
    mn = mean.float()[:, :, None]
    rs = rstd.float()[:, :, None]
    wf = weight.float().reshape(1, C, 1)
    x_hat = (xf - mn) * rs
    gw = (dyf * x_hat).sum(-1).sum(0)
    gb = dyf.sum(-1).sum(0)
    wdy = wf * dyf
    c1 = (x_hat * wdy).sum(-1, keepdim=True) / S
    c2 = wdy.sum(-1, keepdim=True) / S
    dx = (wdy - (x_hat * c1 + c2)) * rs
    return dx.to(dt), gw.to(dt), gb.to(dt)


# ---- VLLM (bench_arms builders + refs; native dtype; in-place) ---------------
def _vllm_build(kernel, shape):
    import bench_arms as B

    mod_name, kern_attr, builder, _sub, _key = B.SPECS[kernel]
    mod = importlib.import_module(mod_name)
    kfn = getattr(mod, kern_attr)
    tok, hidden, group = shape
    built = builder(tok, hidden, group)
    args, ref_fn, out_idx, returns = built[0], built[1], built[2], built[3]

    # eager reference on cloned args (for acc gate)
    def _clone(a):
        return tuple(t.clone() if torch.is_tensor(t) else t for t in a)

    def acc(out, _ref_unused):
        ak = _clone(args)
        ar = _clone(args)
        if returns:
            ok_k = kfn_replay_for_acc(ak)
            out_r = ref_fn(*ar)
            return B.cmp_outputs(ak, ar, out_idx, returns, ok_k, out_r)
        else:
            kfn_replay_for_acc(ak)
            ref_fn(*ar)
            return B.cmp_outputs(ak, ar, out_idx, returns)

    # acc uses a closure kfn set later; return metadata for the runner to handle in-place
    return kfn, args, ref_fn, out_idx, returns


# --------------------------------------------------------------------------- #
#  Corpus dispatch: build a cell, run its arms, return a row dict.
# --------------------------------------------------------------------------- #
def _clone_args(args):
    return tuple(t.clone() if torch.is_tensor(t) else t for t in args)


def run_real_cell(corpus, kernel, shape, dtype):
    """3-arm (seed/default/tc) cell for curriculum/transfer/mreduction/vllm."""
    dt = _DT[dtype] if dtype in _DT else None
    row = {"corpus": corpus, "kernel": kernel, "shape": list(shape), "dtype": dtype}

    # ---- build ----
    if corpus == "curriculum":
        kfn, args, ref, acc, tc_ref = _cur_build(kernel, shape[0], shape[1], dt)
        inplace = False
    elif corpus == "transfer":
        kfn, args, ref, acc, tc_ref = _transfer_build(kernel, shape, dt)
        inplace = False
    elif corpus == "mreduction":
        kfn, args, ref, acc, tc_ref = _mred_build(kernel, tuple(shape), dt)
        inplace = False
    elif corpus == "vllm":
        return run_vllm_cell(kernel, shape)
    else:
        raise KeyError(corpus)

    torch._dynamo.reset()
    seed_cfg, base_cfg, fired = _extract_configs(kfn, args)
    row["fired_heuristics"] = fired
    row["seed_config"] = _cfg_dict(seed_cfg)
    row["base_default_config"] = _cfg_dict(base_cfg)
    row["configs_differ"] = _cfg_dict(seed_cfg) != _cfg_dict(base_cfg)

    arms = {}
    # ---- seed + default (Helion) ----
    for name, cfg in (("seed", seed_cfg), ("default", base_cfg)):
        a = {"config_present": cfg is not None}
        k = _replay(kfn, cfg)
        if k is None:
            a["status"] = "no-config"
            arms[name] = a
            continue
        try:
            out = k(*_clone_args(args)) if inplace else k(*args)
            ok, detail = acc(out, ref)
            a["acc"] = ok
            a["acc_detail"] = detail
            if ok:
                a.update(timed((lambda kk=k: kk(*args))))
                a["status"] = "ok"
            else:
                a["status"] = "acc-fail"
        except Exception as e:  # noqa: BLE001
            a["status"] = f"compile-fail:{type(e).__name__}"
            a["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        arms[name] = a

    # ---- tc ----
    a = {}
    if tc_ref is None:
        a["status"] = "n/a-no-tc"
    else:
        try:
            torch._dynamo.reset()
            tc = torch.compile(tc_ref)
            out_tc = tc()
            ok, detail = acc(out_tc, ref)
            a["acc"] = ok
            a["acc_detail"] = detail
            if ok:
                a.update(timed(tc))
                a["status"] = "ok"
            else:
                a["status"] = "acc-fail"
        except Exception as e:  # noqa: BLE001
            a["status"] = f"compile-fail:{type(e).__name__}"
            a["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    arms["tc"] = a
    row["arms"] = arms
    _ratios(row)
    _cleanup()
    return row


def run_vllm_cell(kernel, shape):
    """vLLM native-dtype cell. In-place kernels: clone args per timed call (symmetric)."""
    import bench_arms as B

    # shapes.json gives [tok, hidden] for 2-D quant kernels, [tok, hidden, group] for grouped.
    tok, hidden = shape[0], shape[1]
    group = shape[2] if len(shape) > 2 else None
    row = {"corpus": "vllm", "kernel": kernel, "shape": list(shape), "dtype": "native"}
    mod_name, kern_attr, builder, sub, key_fn = B.SPECS[kernel]
    mod = importlib.import_module(mod_name)
    kfn = getattr(mod, kern_attr)
    built = builder(tok, hidden, group)
    args, ref_fn, out_idx, returns = built[0], built[1], built[2], built[3]

    torch._dynamo.reset()
    seed_cfg, base_cfg, fired = _extract_configs(kfn, args)
    row["fired_heuristics"] = fired
    row["seed_config"] = _cfg_dict(seed_cfg)
    row["base_default_config"] = _cfg_dict(base_cfg)
    row["configs_differ"] = _cfg_dict(seed_cfg) != _cfg_dict(base_cfg)

    def acc_of(cfg):
        k = _replay(kfn, cfg)
        if k is None:
            return None, "no-config", None
        ak, ar = _clone_args(args), _clone_args(args)
        try:
            if returns:
                ok_k = k(*ak)
                out_r = ref_fn(*ar)
                ok, detail = B.cmp_outputs(ak, ar, out_idx, returns, ok_k, out_r)
            else:
                k(*ak)
                ref_fn(*ar)
                ok, detail = B.cmp_outputs(ak, ar, out_idx, returns)
            return k, ("ok" if ok else "acc-fail"), (ok, detail)
        except Exception as e:  # noqa: BLE001
            return None, f"compile-fail:{type(e).__name__}", (False, f"{type(e).__name__}: {str(e)[:200]}")

    arms = {}
    for name, cfg in (("seed", seed_cfg), ("default", base_cfg)):
        a = {"config_present": cfg is not None}
        k, status, accpair = acc_of(cfg)
        a["status"] = status
        if accpair is not None:
            a["acc"], a["acc_detail"] = accpair
        if status == "ok":
            a.update(timed((lambda kk=k: kk(*_clone_args(args)))))
        arms[name] = a

    # tc arm: compile the torch reference (fresh cloned args, non-in-place semantics).
    a = {}
    try:
        torch._dynamo.reset()
        ar = _clone_args(args)
        tc = torch.compile(ref_fn)
        tc(*_clone_args(ar))  # warm
        a["status"] = "ok"
        a.update(timed((lambda: tc(*_clone_args(ar)))))
        a["acc"] = True
        a["acc_detail"] = "tc==ref by construction"
    except Exception as e:  # noqa: BLE001
        a["status"] = f"compile-fail:{type(e).__name__}"
        a["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    arms["tc"] = a
    row["arms"] = arms
    _ratios(row)
    _cleanup()
    return row


def run_synth_cell(corpus, kernel):
    """2-arm (seed/default) cell for synthetic probes / adversarial synth."""
    row = {"corpus": corpus, "kernel": kernel, "dtype": "native"}
    subdir = "synthetic_probes" if corpus == "synthetic_probes" else "adversarial_synth"
    kfn, args, shape = _load_synth(subdir, kernel)
    row["shape"] = list(shape) if shape is not None else None

    torch._dynamo.reset()
    seed_cfg, base_cfg, fired = _extract_configs(kfn, args)
    row["fired_heuristics"] = fired
    row["seed_config"] = _cfg_dict(seed_cfg)
    row["base_default_config"] = _cfg_dict(base_cfg)
    row["configs_differ"] = _cfg_dict(seed_cfg) != _cfg_dict(base_cfg)

    arms = {}
    # reference = seed arm's own output (they must agree; both are the same kernel).
    ref_out = None
    for name, cfg in (("seed", seed_cfg), ("default", base_cfg)):
        a = {"config_present": cfg is not None}
        k = _replay(kfn, cfg)
        if k is None:
            a["status"] = "no-config"
            arms[name] = a
            continue
        try:
            out = k(*args)
            if ref_out is None and name == "seed":
                ref_out = out
            # accuracy: seed vs default agree (self-consistency)
            if ref_out is not None:
                ok, detail = _synth_selfcheck(out, ref_out)
                a["acc"] = ok
                a["acc_detail"] = detail
            a.update(timed((lambda kk=k: kk(*args))))
            a["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            a["status"] = f"compile-fail:{type(e).__name__}"
            a["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        arms[name] = a
    arms["tc"] = {"status": "n/a-no-tc"}
    row["arms"] = arms
    _ratios(row)
    _cleanup()
    return row


def _synth_selfcheck(out, ref):
    def flat(o):
        return o if torch.is_tensor(o) else (o[0] if isinstance(o, (tuple, list)) else o)
    o, r = flat(out), flat(ref)
    if not torch.is_tensor(o) or not torch.is_tensor(r):
        return True, "non-tensor-out"
    of, rf = o.detach().to(torch.float32), r.detach().to(torch.float32)
    ok = bool(torch.allclose(of, rf, rtol=1e-2, atol=1e-2))
    return ok, f"maxabs={float((of - rf).abs().max()):.3e}"


def _load_synth(subdir, kernel):
    """Import kernel.py for a probe; return (kfn, args, shape)."""
    base = os.path.join(_KSRC, subdir)
    if subdir == "synthetic_probes":
        path = os.path.join(base, kernel, "kernel.py")
    else:
        path = os.path.join(base, f"{kernel}.py")
    spec = importlib.util.spec_from_file_location(f"probe_{kernel.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "get_kernel"):
        kfn = mod.get_kernel()
    elif hasattr(mod, "kernel"):
        kfn = mod.kernel
    else:
        # find the single @helion.kernel Kernel object at module level
        cands = [v for v in vars(mod).values() if isinstance(v, _HelionKernel)]
        kfn = cands[0]
    args = mod.make_args()
    shape = None
    for a in args:
        if torch.is_tensor(a):
            shape = tuple(a.shape)
            break
    return kfn, args, shape


# --------------------------------------------------------------------------- #
#  Ratios + cleanup
# --------------------------------------------------------------------------- #
def _ratios(row):
    arms = row["arms"]

    def us(name):
        a = arms.get(name, {})
        return a.get("us") if a.get("status") == "ok" else None

    s, d, t = us("seed"), us("default"), us("tc")
    row["G_tc"] = round(t / s, 4) if (s and t) else None    # >1 => seed beats tc
    row["G_def"] = round(d / s, 4) if (s and d) else None   # >1 => seed beats default
    row["us"] = {"seed": s, "default": d, "tc": t}


def _cleanup():
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  Driver: one (corpus, kernel) per process; iterate its shapes x dtypes.
# --------------------------------------------------------------------------- #
def _load_shapes():
    with open(os.path.join(_PERF, "shapes.json")) as f:
        return json.load(f)


def _iter_kernel_cells(SH, corpus, kernel):
    """Yield (shape, dtype) for a real-corpus kernel, honoring required_splits."""
    c = SH["corpora"][corpus]
    kdef = c["kernels"][kernel]
    dtypes = c["dtypes"]
    splits = c["required_splits"]
    for split in splits:
        for shape in kdef["shapes"][split]:
            for dtype in dtypes:
                yield tuple(shape), dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    SH = _load_shapes()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.corpus}__{args.kernel}.json")
    print(f"helion={helion.__file__}", flush=True)
    print(f"corpus={args.corpus} kernel={args.kernel} -> {out_path}", flush=True)

    rows = []
    if args.corpus in ("synthetic_probes", "adversarial_synth"):
        try:
            row = run_synth_cell(args.corpus, args.kernel)
        except Exception as e:  # noqa: BLE001
            row = {"corpus": args.corpus, "kernel": args.kernel,
                   "error": f"{type(e).__name__}: {str(e)[:300]}",
                   "trace": traceback.format_exc()}
        rows.append(row)
        _log_row(row)
        json.dump({"rows": rows}, open(out_path, "w"), indent=1)
    else:
        for shape, dtype in _iter_kernel_cells(SH, args.corpus, args.kernel):
            tag = f"{args.corpus}/{args.kernel}/{shape}/{dtype}"
            try:
                row = run_real_cell(args.corpus, args.kernel, shape, dtype)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = {"corpus": args.corpus, "kernel": args.kernel, "shape": list(shape),
                       "dtype": dtype, "error": "OOM"}
            except Exception as e:  # noqa: BLE001
                row = {"corpus": args.corpus, "kernel": args.kernel, "shape": list(shape),
                       "dtype": dtype, "error": f"{type(e).__name__}: {str(e)[:300]}",
                       "trace": traceback.format_exc()}
            rows.append(row)
            _log_row(row)
            json.dump({"rows": rows}, open(out_path, "w"), indent=1)  # checkpoint per cell
    json.dump({"rows": rows}, open(out_path, "w"), indent=1)
    print(f"\n=== DONE {args.corpus}/{args.kernel}: {len(rows)} cells -> {out_path} ===", flush=True)


def _log_row(row):
    if "error" in row:
        print(f"[ERR ] {row.get('shape')}/{row.get('dtype')}: {row['error'][:120]}", flush=True)
        return
    u = row.get("us", {})
    print(f"[cell] {str(row.get('shape')):22s} {row.get('dtype','?'):6s} "
          f"seed={u.get('seed')} def={u.get('default')} tc={u.get('tc')} "
          f"G_tc={row.get('G_tc')} G_def={row.get('G_def')} "
          f"differ={row.get('configs_differ')}", flush=True)


if __name__ == "__main__":
    main()
