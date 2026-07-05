"""Multi-shape climb workhorse: probe + cold-L2 co-bench a list of jobs in ONE
process (serial GPU, the §6 invariant). Reuses bench.py building blocks.

jobs JSON = list of:
  {"kernel": "matmul", "shape": [M,K,N], "dtype": "bf16",
   "configs": ["default","seed", {"block_sizes":[...],...}], "tc": true,
   "reps": 5, "probe_only": false, "static_shapes": "auto"}

Usage: python sweep.py --jobs jobs.json --out out.json
Prints one compact line per job; full detail in --out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B


def run_job(job: dict) -> dict:
    kernel = job["kernel"]
    shape = job["shape"]
    dtype = job["dtype"]
    reps = job.get("reps", 5)
    ss_opt = job.get("static_shapes", "auto")
    ss = B.STATIC_SHAPES[kernel] if ss_opt == "auto" else (ss_opt == "true")
    pool = job.get("configs", ["default", "seed"])
    want_tc = job.get("tc", False)
    probe_only = job.get("probe_only", False)

    info: dict = {"kernel": kernel, "shape": shape, "dtype": dtype,
                  "width_bits": B.WIDTH_BITS[dtype], "static_shapes": ss}
    try:
        torch._dynamo.reset()
        fn = B._kernel_fn(kernel)
        args, ref, tc_fn, meta = B.make_inputs(kernel, shape, dtype)
        info["meta"] = meta
        kp = B._build(kernel, fn, None, ss)
        bound = kp.bind(kp.normalize_args(*args))
        spec = bound.env.config_spec
        seeds = [dict(c) for c in spec.compiler_seed_configs]
        default_cfg = dict(spec.default_config())
        info["matmul_facts"] = [
            {"static_m": f.static_m, "static_n": f.static_n, "static_k": f.static_k,
             "ids": (f.m_block_id, f.n_block_id, f.k_block_id),
             "lhs_dtype": str(f.lhs_dtype)} for f in spec.matmul_facts
        ]
        info["fired"] = list(spec.autotuner_heuristics)
        info["seed_configs"] = seeds
        info["default_config"] = default_cfg
        if probe_only:
            info["ok"] = True
            info["results"] = []
            return info

        resolved: list[tuple[str, dict | None]] = []
        for item in pool:
            if item == "default":
                resolved.append(("default", default_cfg))
            elif item == "seed":
                resolved.append(("seed0", seeds[0] if seeds else None))
            elif isinstance(item, str) and item.startswith("seed"):
                i = int(item[4:])
                resolved.append((item, seeds[i] if i < len(seeds) else None))
            elif isinstance(item, dict):
                resolved.append(("m:" + json.dumps(item, sort_keys=True), item))
            else:
                resolved.append((str(item), None))

        results = []
        cache: dict[str, dict] = {}
        if want_tc and tc_fn is not None:
            rec = {"label": "tc", "kind": "tc"}
            try:
                out = tc_fn()
                ok, ma = B.accuracy_ok(out, ref)
                rec["accuracy_ok"] = ok
                rec["max_abs"] = ma
                rec["perf_ms"] = B._median_of(tc_fn, reps) if ok else None
            except Exception as e:
                rec.update({"accuracy_ok": False, "perf_ms": None,
                            "error": f"{type(e).__name__}: {e}"})
            results.append(rec)

        for label, cfg in resolved:
            rec = {"label": label, "config": cfg, "kind": "helion"}
            if cfg is None:
                rec.update({"accuracy_ok": False, "perf_ms": None, "error": "no config"})
                results.append(rec)
                continue
            ck = json.dumps(cfg, sort_keys=True, default=str)
            if ck in cache:
                p = cache[ck]
                rec.update({"accuracy_ok": p["accuracy_ok"], "perf_ms": p["perf_ms"],
                            "max_abs": p.get("max_abs"), "dup_of": p["label"]})
                results.append(rec)
                continue
            try:
                k = B._build(kernel, fn, cfg, ss)
                out = k(*args)
                ok, ma = B.accuracy_ok(out, ref)
                rec["accuracy_ok"] = ok
                rec["max_abs"] = ma
                rec["perf_ms"] = B._median_of(lambda: k(*args), reps) if ok else None
            except Exception as e:
                rec.update({"accuracy_ok": False, "perf_ms": None,
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-800:]})
            cache[ck] = rec
            results.append(rec)

        d = next((r for r in results if r["label"] == "default"), None)
        tc = next((r for r in results if r["label"] == "tc"), None)
        for r in results:
            if r.get("perf_ms"):
                if d and d.get("perf_ms"):
                    r["xD"] = round(d["perf_ms"] / r["perf_ms"], 3)
                if tc and tc.get("perf_ms"):
                    r["G"] = round(tc["perf_ms"] / r["perf_ms"], 3)
        info["ok"] = True
        info["results"] = results
    except Exception as e:
        info.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()[-1500:]})
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    out = []
    for i, job in enumerate(jobs):
        info = run_job(job)
        out.append(info)
        tag = f"{job['kernel']}/{job['shape']}/{job['dtype']}"
        if not info.get("ok"):
            print(f"[{i}] {tag} ERROR {info.get('error')}")
            continue
        if job.get("probe_only"):
            s0 = info["seed_configs"][0] if info["seed_configs"] else None
            bs = s0.get("block_sizes") if s0 else None
            nw = s0.get("num_warps") if s0 else None
            ns = s0.get("num_stages") if s0 else None
            print(f"[{i}] {tag} fired={info['fired']} seed0 bs={bs} w={nw} s={ns}")
        else:
            parts = []
            for r in info["results"]:
                ms = r.get("perf_ms")
                ms_s = f"{ms * 1000:.1f}us" if ms else ("FAIL" if not r.get("accuracy_ok") else "None")
                extra = []
                if r.get("xD"):
                    extra.append(f"xD={r['xD']}")
                if r.get("G"):
                    extra.append(f"G={r['G']}")
                parts.append(f"{r['label'][:22]}={ms_s}" + ("(" + ",".join(extra) + ")" if extra else ""))
            print(f"[{i}] {tag} | " + " | ".join(parts))
        json.dump(out, open(a.out, "w"), indent=2, default=str)
    json.dump(out, open(a.out, "w"), indent=2, default=str)
    print(f"WROTE {a.out} ({len(out)} jobs)")


if __name__ == "__main__":
    main()
