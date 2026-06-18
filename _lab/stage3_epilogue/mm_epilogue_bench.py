"""Forward-only benchmark harness for the Stage-3 matmul + reduction-epilogue corpus.

Cloned from `_lab/transfer/ab_three_arm_transfer.py`. Single-process, same inputs across
arms, median-of-9 cold-L2 do_bench, accuracy-gate BEFORE timing, geomean over acc-passing
rows. Honors hillclimb-method §4 footguns (forward-only, dynamo-reset per shape,
empty_cache between shapes, cold-L2 metric, exclude acc-fails, flag sub-25us).

Arms (per kernel, shape, dtype):
  (a) helion_default = config_spec.default_config()           (heuristics off)
  (b) helion_seeded  = compiler_seed_configs(...)[0] or default (== default until the
                       Stage-3 heuristic exists -- that's expected)
  (c) helion_best    = best over a small forced-config GRID    (well-tuned Helion ref)
  (d) tc_default     = torch.compile(ref) DEFAULT mode         } both gated on the skinny
      tc_max         = torch.compile(ref) mode='max-autotune'  } predicate M >= 8*max(K,N)

Report: seeded_vs_default, best_vs_default, best_vs_tc_default, best_vs_tc_max; flag
noisy (sub-25us) / acc-fail rows.

CLI:
  python mm_epilogue_bench.py <kernel> --shapes "M,K,N;..." --dtypes bf16
  python mm_epilogue_bench.py matmul_rms_norm                 # default curriculum, bf16
  python mm_epilogue_bench.py all --split train --dtypes bf16,fp16,fp32

Run (from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-3stage \
    /home/calebkim/.conda/envs/helion/bin/python \
    /home/calebkim/helion-new-heuristics/helion-3stage/_lab/stage3_epilogue/mm_epilogue_bench.py \
    matmul_rms_norm --shapes "131072,256,256" --dtypes bf16
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

import helion
from helion import Config
from helion._compiler.autotuner_heuristics import compiler_seed_configs

# import the sibling corpus regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matmul_epilogue_kernels as MK  # noqa: E402

DEV = "cuda"
N_RUNS = 9
NOISE_FLOOR_US = 25.0
HBM_PEAK_TBps = 3.35  # H100 SXM ~3.35 TB/s; cold-L2 BW above this => measurement bug
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# ---- shape curriculum (M, K, N) --------------------------------------------------
SPLITS = {
    "train": [(131072, 256, 256), (131072, 256, 512), (65536, 512, 512),
              (262144, 128, 256)],
    "val": [(131072, 256, 1024), (98304, 384, 384)],
    "test": [(196608, 256, 768), (131072, 512, 1024)],
    "robustness": [(1024, 256, 256), (131072, 256, 2048)],  # last: expect no valid cfg
}

def _med(fn) -> float:
    """median-of-9 cold-L2 do_bench, returned in MICROSECONDS (this Triton build flushes
    L2 between reps; §4 #9).  do_bench returns milliseconds -> * 1e3 for us."""
    from triton.testing import do_bench

    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2] * 1e3


def _bytes_moved(kernel, m, k, n, dtype):
    """resident HBM traffic for an effective-BW sanity check (cold-L2 BW must be < HBM peak)."""
    es = torch.finfo(dtype).bits // 8
    # x[M,K] + y[K,N] read; output is [M,N] for full-width kernels, [M,1] for scalar.
    scalar = kernel in ("matmul_sum", "matmul_logsumexp", "matmul_max")
    out_elems = m * (1 if scalar else n)
    return (m * k + k * n) * es + out_elems * es


def _build(kernel_name, m, k, n, dtype):
    """-> (kernel_fn, args, ref_callable)."""
    knl, ref, extra = MK.KERNELS[kernel_name]
    x = torch.randn(m, k, device=DEV, dtype=dtype)
    y = torch.randn(k, n, device=DEV, dtype=dtype)
    extra_args = extra(n, dtype, DEV)
    args = (x, y, *extra_args)
    return knl, args, (lambda: ref(*args))


def _bind(kernel_fn, args, cfg):
    """forced-config bare-forward (§4 #1): no autograd wrapper, configs=[cfg], static."""
    if cfg is None:
        return None
    k = helion.kernel(kernel_fn.fn, config=cfg, static_shapes=True)
    return lambda: k(*args)


def _try_run(call):
    """run a forced-config arm; return (output, None) or (None, error-string)."""
    try:
        out = call()
        torch.cuda.synchronize()
        return out, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:80]}"


def _acc_ok(kernel_name, out, ref_out, dt_name):
    """measured, per-kernel accuracy gate (§4 #6b): max_abs for the bounded softmax
    output, max_abs/output-RMS for the magnitude-scaling reductions.  Shared with the
    corpus so the harness and the corpus correctness check agree."""
    return MK.acc_ok(kernel_name, out, ref_out, dt_name)


def _seed_call(kernel_fn, args):
    """helion_seeded arm: compiler_seed_configs(...)[0] if any else default."""
    bound = kernel_fn.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    cfg = seeds[0] if seeds else bound.config_spec.default_config()
    fired = bool(seeds)
    return _bind(kernel_fn, args, cfg), fired, cfg


# small forced-config GRID for helion_best (the well-tuned Helion reference).
GRID_TILE_M = [16, 32, 64, 128, 256]
GRID_TILE_K = [16, 32, 64]
GRID_WARPS = [4, 8]


def _grid_best(kernel_name, kernel_fn, args, ref_out, dt_name):
    """sweep the forced-config grid; return (best_call, best_lat_us, best_cfg, n_valid)."""
    best = (None, float("inf"), None)
    n_valid = 0
    for tm in GRID_TILE_M:
        for tk in GRID_TILE_K:
            for w in GRID_WARPS:
                cfg = Config(block_sizes=[tm, tk], num_warps=w, num_stages=3)
                call = _bind(kernel_fn, args, cfg)
                out, err = _try_run(call)
                if err is not None:
                    continue
                ok, _ = _acc_ok(kernel_name, out, ref_out, dt_name)
                if not ok:
                    continue
                n_valid += 1
                lat = _med(call)
                if lat < best[1]:
                    best = (call, lat, cfg)
    return best[0], best[1], best[2], n_valid


def _skinny(m, k, n):
    return m >= 8 * max(k, n)


def _bench_one(kernel_name, shape, dt_name):
    m, k, n = shape
    dtype = _DTYPES[dt_name]
    # Footgun #2/#11: reset dynamo per shape so tc recompiles for THIS shape.
    torch._dynamo.reset()
    knl, args, ref = _build(kernel_name, m, k, n, dtype)
    ref_out = ref()

    bound = knl.bind(args)
    default_cfg = bound.config_spec.default_config()
    d_call = _bind(knl, args, default_cfg)
    s_call, seed_fired, seed_cfg = _seed_call(knl, args)

    # ---- accuracy gate BEFORE timing (footgun #6); acc-fail rows excluded from geomean.
    acc = {}
    for nm, call in (("default", d_call), ("seeded", s_call)):
        out, err = _try_run(call)
        if err is not None:
            acc[nm] = f"ERR:{err}"
            continue
        ok, why = _acc_ok(kernel_name, out, ref_out, dt_name)
        acc[nm] = True if ok else f"FAIL:{why}"

    # ---- helion_best grid (acc-gated inside) ----
    b_call, t_best, best_cfg, n_valid = _grid_best(kernel_name, knl, args, ref_out, dt_name)
    acc["best"] = True if b_call is not None else "no-valid-cfg"

    # ---- torch.compile arms, gated on skinny predicate ----
    tc_d_lat = tc_m_lat = float("nan")
    acc["tc_default"] = acc["tc_max"] = "skipped(not-skinny)"
    if _skinny(m, k, n):
        torch._dynamo.reset()
        tc_d = torch.compile(ref)
        out, err = _try_run(lambda: tc_d())
        if err is None:
            ok, why = _acc_ok(kernel_name, out, ref_out, dt_name)
            acc["tc_default"] = True if ok else f"FAIL:{why}"
            tc_d_lat = _med(tc_d)
        else:
            acc["tc_default"] = f"ERR:{err}"
        torch._dynamo.reset()
        tc_m = torch.compile(ref, mode="max-autotune")
        out, err = _try_run(lambda: tc_m())
        if err is None:
            ok, why = _acc_ok(kernel_name, out, ref_out, dt_name)
            acc["tc_max"] = True if ok else f"FAIL:{why}"
            tc_m_lat = _med(tc_m)
        else:
            acc["tc_max"] = f"ERR:{err}"

    # ---- timings for helion arms (only if acc passed) ----
    td = _med(d_call) if (d_call and acc["default"] is True) else float("nan")
    ts = _med(s_call) if (s_call and acc["seeded"] is True) else float("nan")
    tb = t_best if b_call is not None else float("nan")

    # cold-L2 effective-BW sanity check on helion_best (must be < HBM peak; §4 #9).
    # tb is in us -> seconds = tb * 1e-6.
    bw_tbps = (_bytes_moved(kernel_name, m, k, n, dtype) / (tb * 1e-6) / 1e12) if tb == tb else None
    bw_warn = bw_tbps is not None and bw_tbps > HBM_PEAK_TBps

    def _r(a, b):
        return round(a / b, 4) if (a == a and b == b and b) else None

    row = {
        "kernel": kernel_name, "shape": list(shape), "dtype": dt_name,
        "lat_us": {
            "default": round(td, 2) if td == td else None,
            "seeded": round(ts, 2) if ts == ts else None,
            "best": round(tb, 2) if tb == tb else None,
            "tc_default": round(tc_d_lat, 2) if tc_d_lat == tc_d_lat else None,
            "tc_max": round(tc_m_lat, 2) if tc_m_lat == tc_m_lat else None,
        },
        "seeded_vs_default": _r(td, ts),
        "best_vs_default": _r(td, tb),
        "best_vs_tc_default": _r(tc_d_lat, tb),
        "best_vs_tc_max": _r(tc_m_lat, tb),
        "acc": acc,
        "seed_fired": seed_fired,
        "best_cfg": dict(best_cfg.config) if best_cfg else None,
        "grid_valid_cfgs": n_valid,
        "skinny": _skinny(m, k, n),
        "noisy_sub25us": (
            min([t for t in (td, ts, tb, tc_d_lat, tc_m_lat) if t == t] or [1e9])
            < NOISE_FLOOR_US
        ),
        "best_bw_tbps": round(bw_tbps, 3) if bw_tbps is not None else None,
        "bw_exceeds_hbm_WARN": bw_warn,
    }
    # Footgun #8: free big tensors between (multi-GB) shapes.
    del knl, args, ref, ref_out, d_call, s_call, b_call
    torch.cuda.empty_cache()
    return row


def bench(kernels, shapes, dtypes):
    print(f"helion: {helion.__file__}", file=sys.stderr)
    all_rows = []
    for kernel_name in kernels:
        rows = []
        for dt_name in dtypes:
            for shape in shapes:
                row = _bench_one(kernel_name, shape, dt_name)
                rows.append(row)
                all_rows.append(row)
                print("ROW " + json.dumps(row), file=sys.stderr)
        # geomean over acc-passing best rows (footgun #6a).
        ok = [r for r in rows if r["acc"].get("best") is True]
        def _gm(key):
            vals = [r[key] for r in ok if r[key]]
            return round(statistics.geometric_mean(vals), 4) if vals else None
        summary = {
            "kernel": kernel_name,
            "n_acc_pass_best": len(ok), "n_total": len(rows),
            "geomean_seeded_vs_default": _gm("seeded_vs_default"),
            "geomean_best_vs_default": _gm("best_vs_default"),
            "geomean_best_vs_tc_default": _gm("best_vs_tc_default"),
            "geomean_best_vs_tc_max": _gm("best_vs_tc_max"),
            "noisy_rows": [r["shape"] for r in rows if r["noisy_sub25us"]],
            "acc_fail_rows": [
                {"shape": r["shape"], "dtype": r["dtype"], "acc": r["acc"]}
                for r in rows
                if not all(
                    v is True or (isinstance(v, str) and v.startswith("skip"))
                    for v in r["acc"].values()
                )
            ],
            "bw_warn_rows": [r["shape"] for r in rows if r["bw_exceeds_hbm_WARN"]],
        }
        print("SUMMARY " + json.dumps(summary))
    return all_rows


def _parse_shapes(s):
    out = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        m, k, n = (int(v) for v in tok.split(","))
        out.append((m, k, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", help="corpus kernel name, or 'all' / 'fit' / 'held_out'")
    ap.add_argument("--shapes", default=None, help='"M,K,N;M,K,N;..." (overrides --split)')
    ap.add_argument("--split", default="train", choices=list(SPLITS),
                    help="curriculum split if --shapes not given")
    ap.add_argument("--dtypes", default="bf16")
    a = ap.parse_args()

    if a.kernel == "all":
        kernels = list(MK.KERNELS)
    elif a.kernel == "fit":
        kernels = list(MK.FIT)
    elif a.kernel == "held_out":
        kernels = list(MK.HELD_OUT)
    else:
        kernels = [a.kernel]
    shapes = _parse_shapes(a.shapes) if a.shapes else SPLITS[a.split]
    dtypes = a.dtypes.split(",")
    bench(kernels, shapes, dtypes)


if __name__ == "__main__":
    main()
