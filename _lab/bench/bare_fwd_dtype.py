"""Dtype-aware bare-forward seeded-Helion vs torch.compile-default (+ optional unseeded
default arm). The climb workhorse for the bf16/fp16 dtype run.

Threads a --dtype axis through the prior bare_fwd_seed_vs_tc.py method (forward-only,
single-process, dynamo-reset per shape, median-of-N). All §4 footguns handled:
  * inputs built requires_grad=False at the chosen dtype; bare forward timed (no autograd
    wrapper) for BOTH arms.
  * torch._dynamo.reset() before each shape so tc isn't penalised by dynamic-shape recompiles.
  * single process, same input tensors, median-of-N (N=15) over do_bench medians.
  * accuracy gate vs the eager reference built at the SAME dtype, both upcast to fp32 before
    allclose; per-(kernel,dtype) tolerance (measured floor, logged — never silently widened).
  * records the NORMALIZED running config to prove the seed (configs=[seed], no autotune) ran.
  * JSON-checkpoints every row so a kill loses nothing.

Usage (cwd=/tmp, PYTHONPATH=<worktree>):
  python bare_fwd_dtype.py --dtype bf16 --split test rms_norm softmax cross_entropy
  python bare_fwd_dtype.py --dtype bf16 --split test --arms seed,tc,default --shapes 8192,50257 cross_entropy
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys

import torch
from triton.testing import do_bench

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

_WT = os.environ.get(
    "HELION_WORKTREE",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.join(_WT, "_lab", "prompts"))
import shapes_v3_draft as SH  # noqa: E402

from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.welford import welford, eager_layer_norm  # noqa: E402
from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402
from examples.long_sum import longsum  # noqa: E402

EPS = 1e-5
LONG = torch.int64
N_RUNS = 15
# CUDA-graph headline timing on by default (ITER2: removes host-overhead artifact). Disable
# with HELION_LAB_NO_CG=1 for kernels that aren't graph-safe.
USE_CG = os.environ.get("HELION_LAB_NO_CG", "0") != "1"

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# fp16 is intentionally NOT benchmarked for these loss/divergence kernels. At realistic
# vocab sizes (V >= ~30k) the softmax / exp(log_softmax) probabilities (~1/V) underflow
# fp16's 5-bit exponent (min normal ~6e-5) to 0, so log(0) = -inf and 0 * -inf = NaN ->
# the whole row is NaN. bf16 shares fp32's 8-bit exponent and is safe. Real mixed-
# precision training keeps loss kernels in bf16/fp32 (autocast runs softmax/log/loss in
# fp32) for exactly this reason, so fp16-wide-V is not a real workload. (Empirically,
# 2026-06-08: kl_div NaNs at every test vocab >= 32768; jsd NaNs at the widest V >= ~115k.
# torch's own KLDivLoss avoids NaN via fp32/masking; the Helion examples don't guard it
# because the regime is out of scope.) These kernels ARE benchmarked at bf16 and fp32.
FP16_UNSUPPORTED = {"kl_div", "jsd"}

# Per-dtype accuracy tolerance (rtol=atol). fp32 tight; half-precision floors are MEASURED
# per kernel (see _lab/dtype_notebook.md) — these are the justified defaults; a kernel that
# needs looser is recorded explicitly. Multi-pass losses (kl_div/jsd/welford) compound.
TOL = {
    "fp32": {"_default": 1e-3, "kl_div": 2e-2, "jsd": 2e-2, "welford": 2e-2},
    "bf16": {"_default": 2e-2, "cross_entropy": 3e-2, "kl_div": 4e-2, "jsd": 4e-2, "welford": 4e-2},
    "fp16": {"_default": 1e-2, "cross_entropy": 2e-2, "kl_div": 3e-2, "jsd": 3e-2, "welford": 3e-2},
}


def _tol(kn: str, dt: str) -> float:
    t = TOL[dt]
    return t.get(kn, t["_default"])


def _first(o):
    return o[0] if isinstance(o, tuple) else o


def _foreign_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10).stdout
    except Exception:  # noqa: BLE001
        return 0
    me = os.getpid(); m = 0
    for line in out.splitlines():
        p = [c.strip() for c in line.split(",")]
        if len(p) == 2 and p[0].isdigit() and int(p[0]) != me:
            m = max(m, int(p[1]) if p[1].isdigit() else 0)
    return m


def _med(fn) -> float:
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2] * 1000.0


def _cudagraph_us(fn, warmup=5, iters=50, reps=15) -> float:
    """Per-call DEVICE time via CUDA-graph replay. Removes Python host-enqueue overhead
    from BOTH arms identically (fair), the artifact that makes plain do_bench mis-attribute
    host dispatch cost to low-M bandwidth-bound kernels (ITER2 finding). This is the HEADLINE
    metric for bandwidth-bound seed-vs-tc; do_bench is kept as a recorded secondary.
    Graphs only kernels that are CUDA-graph-safe (no host sync / dynamic alloc mid-call)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record(); g.replay(); en.record(); torch.cuda.synchronize()
        times.append(st.elapsed_time(en) / iters * 1000.0)
    return sorted(times)[len(times) // 2]


# build fns return (args, eager_ref_tensor, out_extract). dt = torch dtype.
def b_rms(m, n, dt):
    x = torch.randn(m, n, device="cuda", dtype=dt); w = torch.randn(n, device="cuda", dtype=dt)
    return (x, w, EPS), rms_norm_pytorch(x, w, EPS), _first


def b_ln(m, n, dt):
    x = torch.randn(m, n, device="cuda", dtype=dt); w = torch.randn(n, device="cuda", dtype=dt); b = torch.randn(n, device="cuda", dtype=dt)
    return (x, [n], w, b, EPS), torch.nn.functional.layer_norm(x, [n], w, b, EPS), _first


def b_welford(m, n, dt):
    w = torch.rand(n, device="cuda", dtype=dt); b = torch.rand(n, device="cuda", dtype=dt); x = torch.rand(m, n, device="cuda", dtype=dt)
    a = (w, b, x, EPS); return a, eager_layer_norm(*a), _first


def b_softmax(m, n, dt):
    x = torch.randn(m, n, device="cuda", dtype=dt); return (x,), torch.nn.functional.softmax(x, dim=1), _first


def b_ce(m, n, dt):
    lg = torch.randn(m, n, device="cuda", dtype=dt); lb = torch.randint(0, n, (m,), device="cuda", dtype=LONG)
    return (lg, lb), torch.nn.functional.cross_entropy(lg, lb), _first


def b_kl(m, n, dt):
    yp = torch.randn(m, n, device="cuda", dtype=dt).log_softmax(-1); yt = torch.randn(m, n, device="cuda", dtype=dt).softmax(-1)
    return (yp, yt, False, "batchmean", 1e-10), torch.nn.KLDivLoss(reduction="batchmean").to("cuda")(yp, yt), _first


def _jsd_ref(lq, lp):
    p, q = lq.exp(), lp.exp()
    mm = 0.5 * (p + q)
    return (0.5 * (p * (lq - mm.log())).sum(-1) + 0.5 * (q * (lp - mm.log())).sum(-1)).mean()


def b_jsd(m, n, dt):
    lq = torch.randn(m, n, device="cuda", dtype=dt).log_softmax(-1); lp = torch.randn(m, n, device="cuda", dtype=dt).log_softmax(-1)
    return (lq, lp, None, 0.5, -100), _jsd_ref(lq, lp), _first


def b_sum(m, n, dt):
    x = torch.randn(m, n, device="cuda", dtype=dt); return (x,), torch.sum(x, dim=-1), _first


def b_longsum(m, n, dt):
    x = torch.randn(m, n, device="cuda", dtype=dt); return (x,), torch.sum(x, dim=-1), _first


KERNELS = {
    "rms_norm": (rms_norm_fwd, b_rms, lambda a: rms_norm_pytorch(*a)),
    "layer_norm": (layer_norm_fwd, b_ln, lambda a: torch.nn.functional.layer_norm(a[0], a[1], a[2], a[3], a[4])),
    "softmax": (softmax_two_pass, b_softmax, lambda a: torch.nn.functional.softmax(a[0], dim=1)),
    "sum": (sum_kernel, b_sum, lambda a: torch.sum(a[0], dim=-1)),
    "cross_entropy": (cross_entropy, b_ce, lambda a: torch.nn.functional.cross_entropy(a[0], a[1])),
    "long_sum": (longsum, b_longsum, lambda a: torch.sum(a[0], dim=-1)),
    "welford": (welford, b_welford, lambda a: eager_layer_norm(*a)),
    "kl_div": (kl_div_forward, b_kl, lambda a: torch.nn.KLDivLoss(reduction="batchmean").to("cuda")(a[0], a[1])),
    "jsd": (jsd_forward, b_jsd, lambda a: _jsd_ref(a[0], a[1])),
}


def bench(kn: str, dt_name: str, split: str, arms: list[str], explicit_shapes=None) -> dict:
    fn, build, tc_ref = KERNELS[kn]
    dt = DTYPES[dt_name]
    if dt_name == "fp16" and kn in FP16_UNSUPPORTED:
        return {"kernel": kn, "dtype": dt_name, "split": split, "rows": [],
                "skipped": "fp16 unsupported for this loss kernel: at realistic vocab the "
                           "softmax/exp probabilities (~1/V) underflow fp16's 5-bit exponent "
                           "-> log(0)=-inf -> NaN. Real training keeps loss kernels in "
                           "bf16/fp32 (see FP16_UNSUPPORTED note). Benchmarked at bf16/fp32."}
    shapes = explicit_shapes if explicit_shapes else SH.SHAPES[kn][split]
    rows = []
    for (m, n) in shapes:
        torch._dynamo.reset()
        args, ref, extract = build(m, n, dt)
        row = {"shape": [m, n], "dtype": dt_name}
        # seed arm
        bound0 = fn.bind(args)
        seeds = compiler_seed_configs(bound0.env, bound0.host_function.device_ir)
        seed = seeds[0] if seeds else bound0.config_spec.default_config()
        k_seed = helion.kernel(fn.fn, config=seed, static_shapes=True)
        try:
            out = extract(k_seed(*args))
            # footgun #7: record the NORMALIZED running config (proves configs=[seed] ran, no autotune)
            norm_cfg = str(k_seed.bind(args)._config)
            tol = _tol(kn, dt_name)
            diff_t = (out.float() - ref.float()).abs()
            max_abs = float(diff_t.max())
            denom = ref.float().abs().clamp_min(1e-12)
            max_rel = float((diff_t / denom).max())
            acc = bool(torch.allclose(out.float(), ref.float(), rtol=tol, atol=tol))
            row.update({"seed_cfg": norm_cfg, "acc": acc, "tol": tol,
                        "max_abs": round(max_abs, 6), "max_rel": round(max_rel, 6)})
            if "seed" in arms:
                row["seed_us"] = round(_med(lambda: k_seed(*args)), 3)
                if USE_CG:
                    try:
                        row["seed_cg"] = round(_cudagraph_us(lambda: k_seed(*args)), 3)
                    except Exception as e:  # noqa: BLE001
                        row["seed_cg_err"] = type(e).__name__
        except Exception as e:  # noqa: BLE001
            row["seed_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            rows.append(row); print("ROW " + json.dumps(row), file=sys.stderr); continue
        # tc arm
        if "tc" in arms:
            tcfn = torch.compile(lambda: tc_ref(args))
            tcfn()
            row["tc_us"] = round(_med(tcfn), 3)
            if "seed_us" in row and row["seed_us"]:
                row["G_seed"] = round(row["tc_us"] / row["seed_us"], 4)
            if USE_CG:
                try:
                    row["tc_cg"] = round(_cudagraph_us(tcfn), 3)
                    if row.get("seed_cg"):
                        row["G_cg"] = round(row["tc_cg"] / row["seed_cg"], 4)
                except Exception as e:  # noqa: BLE001
                    row["tc_cg_err"] = type(e).__name__
        # unseeded default arm
        if "default" in arms:
            torch._dynamo.reset()
            d_args, _, d_extract = build(m, n, dt)
            d_bound = fn.bind(d_args)
            d_cfg = d_bound.config_spec.default_config()
            k_def = helion.kernel(fn.fn, config=d_cfg, static_shapes=True)
            k_def(*d_args)
            row["default_us"] = round(_med(lambda: k_def(*d_args)), 3)
            if "seed_us" in row and row["seed_us"]:
                row["lift_vs_default"] = round(row["default_us"] / row["seed_us"], 4)
        row["foreign_mib"] = _foreign_mib()
        rows.append(row)
        print("ROW " + json.dumps(row), file=sys.stderr)
    # Headline G uses CUDA-graph (G_cg) where available, else do_bench G_seed.
    def _g(r):
        return r.get("G_cg") or r.get("G_seed")
    # ACCURACY GATE: only fold shapes that PASS the accuracy check into the headline G.
    # A kernel that returns wrong/NaN output (input-width half accumulator, fp16
    # underflow, ...) is not a valid perf comparison vs torch, so its timing must NOT
    # count as a win/loss. Failing shapes are still timed and kept per-row, but are
    # excluded from geo/median/min/max and the loser list, and surfaced in
    # acc_fail_excluded so they can never be silently reported as wins.
    valid = [r for r in rows if r.get("acc") is True]
    gs = [_g(r) for r in valid if _g(r)]
    return {"kernel": kn, "dtype": dt_name, "split": split, "rows": rows,
            "median_G": round(st.median(gs), 4) if gs else None,
            "geo_G": round(st.geometric_mean(gs), 4) if gs else None,
            "min_G": round(min(gs), 4) if gs else None,
            "max_G": round(max(gs), 4) if gs else None,
            "losers_vs_tc": [r["shape"] for r in valid if _g(r) and _g(r) < 1 / 1.03],
            "n_valid": len(gs), "n_total": len(rows),
            "acc_fail_excluded": [r["shape"] for r in rows if r.get("acc") is not True],
            "any_acc_fail": any(r.get("acc") is False for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--split", default="test")
    ap.add_argument("--arms", default="seed,tc")
    ap.add_argument("--shapes", default=None, help="explicit M,N;M,N override")
    ap.add_argument("--out", default="/tmp/barefwd_dtype_out.json")
    ap.add_argument("kernels", nargs="*")
    a = ap.parse_args()
    assert os.path.realpath(helion.__file__).startswith(os.path.realpath(_WT)), helion.__file__
    arms = a.arms.split(",")
    explicit = None
    if a.shapes:
        explicit = [tuple(int(v) for v in pair.split(",")) for pair in a.shapes.split(";")]
    out = []
    for kn in (a.kernels or list(KERNELS)):
        sys.stderr.write(f"\n===== {kn} [{a.dtype}/{a.split}] =====\n")
        try:
            r = bench(kn, a.dtype, a.split, arms, explicit)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"  {kn} FAILED: {e}\n"); out.append({"kernel": kn, "error": str(e)}); continue
        if r.get("skipped"):
            sys.stderr.write(f"  SKIPPED: {r['skipped']}\n")
        else:
            sys.stderr.write(f"  median_G={r['median_G']} geo={r['geo_G']} "
                             f"({r['min_G']}-{r['max_G']}) losers={r['losers_vs_tc']} "
                             f"n_valid={r['n_valid']}/{r['n_total']} "
                             f"acc_excluded={r['acc_fail_excluded']}\n")
        out.append(r)
        json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
