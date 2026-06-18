"""Dtype-aware autotune ORACLE + seed/oracle field-diff (the Step-2/3 answer key).

For one (kernel, M, N, dtype): run the Helion autotuner FRESH (force=True), fair-re-bench
its winning config with do_bench, measure the LIVE seed in the SAME process (noise-robust
ratio), gate BOTH for correctness vs the at-dtype eager reference, field-diff seed-vs-oracle
(the differing fields ARE the worklist), and cache to _lab/logs/dtype/oracle_cache.json.

Reuses bare_fwd_dtype.py's build fns / tolerances (single source of truth for shapes+refs).
Budget via HELION_AUTOTUNE_EFFORT (quick to iterate, full to confirm). Foreground, one shape
at a time, JSON-checkpointed. fp32/bf16/fp16 via --dtype.

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=quick \
    PYTHONPATH=/home/dev/local/helion-pr-with-lab /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-pr-with-lab/_lab/harness/oracle_dtype.py \
    --kernel welford --dtype bf16 --M 16384 --N 896
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

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(_WT, "_lab", "bench"))
import bare_fwd_dtype as BF  # noqa: E402

EPS = 0.05
LOG_DIR = os.path.join(_WT, "_lab", "logs", "dtype")
CACHE_PATH = os.path.join(LOG_DIR, "oracle_cache.json")
N_RUNS = 11


def _bench(fn):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2] * 1000.0  # us


def field_diff(seed_cfg: dict, oracle_cfg: dict):
    diff = {}
    for k in sorted(set(seed_cfg) | set(oracle_cfg)):
        sv, ov = seed_cfg.get(k), oracle_cfg.get(k)
        if sv != ov:
            diff[k] = {"seed": sv, "oracle": ov}
    return diff


def run_one(kernel: str, M: int, N: int, dt_name: str, effort: str):
    fn, build, tc_ref = BF.KERNELS[kernel]
    dt = BF.DTYPES[dt_name]
    args, ref, extract = build(M, N, dt)
    tol = BF._tol(kernel, dt_name)

    def acc(out):
        return bool(torch.allclose(out.float(), ref.float(), rtol=tol, atol=tol))

    # live seed (normalized)
    from helion._compiler.autotuner_heuristics import compiler_seed_configs
    b0 = fn.bind(args)
    seeds = compiler_seed_configs(b0.env, b0.host_function.device_ir)
    seed_cfg_obj = seeds[0] if seeds else b0.config_spec.default_config()
    k_seed = helion.kernel(fn.fn, config=seed_cfg_obj, static_shapes=True)
    bs = k_seed.bind(args); bs.ensure_config_exists(args)
    seed_norm = dict(bs._config)
    out_s = extract(k_seed(*args)); seed_ok = acc(out_s)

    # oracle: fresh autotune
    t0 = time.time()
    k_at = helion.kernel(fn.fn, static_shapes=True)
    bk = k_at.bind(args)
    oracle_obj = bk.autotune(args, force=True)
    autotune_s = time.time() - t0
    oracle_cfg = dict(oracle_obj)
    k_or = helion.kernel(fn.fn, config=helion.Config(**oracle_cfg), static_shapes=True)
    bo = k_or.bind(args); bo.ensure_config_exists(args)
    oracle_norm = dict(bo._config)
    out_o = extract(k_or(*args)); oracle_ok = acc(out_o)

    # tc default
    torch._dynamo.reset()
    tc = torch.compile(lambda: tc_ref(args)); tc()

    seed_us = _bench(lambda: k_seed(*args)) if seed_ok else None
    oracle_us = _bench(lambda: k_or(*args)) if oracle_ok else None
    tc_us = _bench(tc)

    entry = {
        "kernel": kernel, "shape": [M, N], "dtype": dt_name, "effort": effort,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "autotune_s": round(autotune_s, 1),
        "seed_cfg": seed_norm, "seed_correct": seed_ok,
        "oracle_cfg": oracle_norm, "oracle_correct": oracle_ok,
        "seed_us": round(seed_us, 2) if seed_us else None,
        "oracle_us": round(oracle_us, 2) if oracle_us else None,
        "tc_us": round(tc_us, 2),
        "seed_oracle": round(seed_us / oracle_us, 4) if (seed_us and oracle_us) else None,
        "G_floor": round(tc_us / seed_us, 4) if seed_us else None,
        "oracle_vs_tc": round(tc_us / oracle_us, 4) if oracle_us else None,
        "field_diff_seed_vs_oracle": field_diff(seed_norm, oracle_norm),
    }
    return entry


def _save(entry):
    os.makedirs(LOG_DIR, exist_ok=True)
    cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {"entries": {}}
    key = f"{entry['kernel']}:{entry['dtype']}:{entry['shape'][0]}x{entry['shape'][1]}"
    cache["entries"][key] = entry
    cache["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(cache, open(CACHE_PATH, "w"), indent=2, default=str)


def _print(e):
    so = f"{e['seed_oracle']}" if e["seed_oracle"] is not None else "None"
    verdict = "VICTORY" if (e["seed_oracle"] and e["seed_oracle"] <= 1 + EPS) else "GAP"
    print(f"\n=== {e['kernel']}({e['shape'][0]},{e['shape'][1]}) {e['dtype']} [{verdict}] "
          f"effort={e['effort']} autotune={e['autotune_s']}s ===", flush=True)
    print(f"  seed/oracle={so}  G_floor(tc/seed)={e['G_floor']}  oracle/tc={e['oracle_vs_tc']}", flush=True)
    print(f"  seed_us={e['seed_us']} oracle_us={e['oracle_us']} tc_us={e['tc_us']}", flush=True)
    print(f"  seed_correct={e['seed_correct']} oracle_correct={e['oracle_correct']}", flush=True)
    print("  FIELD DIFF seed->oracle (the worklist):", flush=True)
    if e["field_diff_seed_vs_oracle"]:
        for k, v in e["field_diff_seed_vs_oracle"].items():
            print(f"    {k}: seed={v['seed']!r}  oracle={v['oracle']!r}", flush=True)
    else:
        print("    (seed config == oracle config)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=list(BF.KERNELS))
    ap.add_argument("--dtype", default="bf16", choices=list(BF.DTYPES))
    ap.add_argument("--M", type=int)
    ap.add_argument("--N", type=int)
    ap.add_argument("--batch", help="JSON list of {kernel,M,N,dtype?}")
    a = ap.parse_args()
    assert os.path.realpath(helion.__file__).startswith(os.path.realpath(_WT)), helion.__file__
    effort = os.environ.get("HELION_AUTOTUNE_EFFORT", "quick")
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__} effort={effort}", flush=True)
    if a.batch:
        shapes = json.load(open(a.batch))
    else:
        assert a.kernel and a.M and a.N, "need --kernel/--M/--N or --batch"
        shapes = [{"kernel": a.kernel, "M": a.M, "N": a.N, "dtype": a.dtype}]
    for sp in shapes:
        kn, M, N = sp["kernel"], int(sp["M"]), int(sp["N"])
        dtn = sp.get("dtype", a.dtype)
        tag = f"{kn}({M},{N}){dtn}"
        try:
            e = run_one(kn, M, N, dtn, effort)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"[OOM ] {tag}", flush=True); continue
        except Exception as ex:  # noqa: BLE001
            print(f"[ERR ] {tag}: {type(ex).__name__}: {ex}"[:300], flush=True); continue
        _save(e); _print(e)
    print(f"\n[cache -> {CACHE_PATH}]", flush=True)


if __name__ == "__main__":
    main()
