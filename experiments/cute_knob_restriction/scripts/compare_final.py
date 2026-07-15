#!/usr/bin/env python
"""Final comparison probe: one full max-autotune search on a matmul shape, recording
every config tried, generations, distinct configs, wall, and per-config time.

Backend + subprocess + fused-flag are driven by env (caller sets them). For cute we
use fp8 scaled_mm; for triton (apples-to-oranges baseline) we use the plain bf16 matmul
example (triton has no tcgen05 fp8 path). Records the tally CSV verbatim + a summary JSON.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time

WORKTREE = "/home/dev/local/helion-rank0"
sys.path.insert(0, str(Path(WORKTREE) / "benchmarks" / "cute"))
sys.path.insert(0, WORKTREE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument(
        "--kind", choices=("cute_fp8", "triton_fp8", "triton_bf16"), required=True
    )
    ap.add_argument("--log-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    out = {
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "seed": args.seed,
        "kind": args.kind,
        "env": {
            k: os.environ.get(k)
            for k in (
                "HELION_BACKEND",
                "HELION_AUTOTUNE_RANDOM_SEED",
                "HELION_FORCE_AUTOTUNE",
                "HELION_AUTOTUNE_EFFORT",
                "HELION_AUTOTUNE_BUDGET_SECONDS",
                "HELION_AUTOTUNE_BENCHMARK_SUBPROCESS",
                "HELION_FUSED_ACCURACY_CHECK",
                "HELION_AUTOTUNE_ACCURACY_CHECK",
                "HELION_CACHE_DIR",
                "HELION_KEEP_CACHE",
            )
        },
        "status": "started",
    }
    Path(args.out_json).write_text(json.dumps(out, default=str, indent=2))

    try:
        import helion, torch  # noqa

        assert WORKTREE in helion.__file__, f"WRONG HELION: {helion.__file__}"
        out["helion_file"] = helion.__file__
        import compare_matmul_backends as M

        if args.kind in ("cute_fp8", "triton_fp8"):
            # Same fp8 e4m3 scaled_mm kernel for both backends (apples-to-apples);
            # the only difference is HELION_BACKEND (set by the caller). The kernel
            # body is backend-agnostic (hl.dot/hl.tile), so Triton lowers its own
            # fp8 dot -- only the tcgen05 fused epilogue is CuTe-specific.
            ns = argparse.Namespace(
                m=args.m,
                n=args.n,
                k=args.k,
                epilogue="scaled_mm",
                dtype="float8_e4m3fn",
                seed=0,
            )
            dtype, a, b, bias, residual = M._make_matmul_problem(ns)
            kernel_args = M._helion_matmul_args(ns, a, b, bias, residual)
            from examples.fp8_matmul import fp8_matmul as matmul
        else:  # triton_bf16 (apples-to-oranges baseline; plain bf16 matmul)
            ns = argparse.Namespace(
                m=args.m, n=args.n, k=args.k, epilogue="none", dtype="bfloat16", seed=0
            )
            dtype, a, b, bias, residual = M._make_matmul_problem(ns)
            kernel_args = M._helion_matmul_args(ns, a, b, bias, residual)
            from examples.matmul import matmul

        t0 = time.time()
        bound = matmul.bind(kernel_args)
        winner = bound.autotune(kernel_args, force=True)
        out["autotune_wall_s"] = round(time.time() - t0, 2)
        out["winner_config"] = dict(winner.config)
        out["status"] = "ok"
    except Exception as e:
        import traceback

        out["status"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()

    # parse tally: distinct configs, generations, per-config durations
    csv_path = Path(args.log_csv).with_suffix(".csv")
    start = {}
    durs = []
    gens = Counter()  # rows per generation (all statuses)
    gen_distinct = {}  # generation -> set of config_ids
    n_started = n_ok = 0
    ids_all = set()
    ids_ok = set()
    if csv_path.exists():
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                cid = r["config_id"]
                g = r.get("generation", "")
                ids_all.add(cid)
                gen_distinct.setdefault(g, set()).add(cid)
                gens[g] += 1
                if r["status"] == "started":
                    n_started += 1
                    start[cid] = float(r["timestamp_s"]) if r["timestamp_s"] else None
                elif r["status"] == "ok":
                    n_ok += 1
                    ids_ok.add(cid)
                    if start.get(cid) is not None and r["timestamp_s"]:
                        durs.append(float(r["timestamp_s"]) - start[cid])

    out["configs_started"] = n_started
    out["configs_ok"] = n_ok
    out["distinct_config_ids"] = len(ids_all)
    out["distinct_ok_config_ids"] = len(ids_ok)
    out["n_generations"] = len([g for g in gen_distinct if g != ""])
    out["distinct_configs_per_generation"] = {
        g: len(s)
        for g, s in sorted(gen_distinct.items(), key=lambda kv: (kv[0] == "", kv[0]))
    }
    if durs:
        out["per_config_median_s"] = round(statistics.median(durs), 3)
        out["per_config_mean_s"] = round(statistics.mean(durs), 3)
        out["per_config_min_s"] = round(min(durs), 3)
        out["per_config_max_s"] = round(max(durs), 3)
        out["n_timed_configs"] = len(durs)

    # Did the search exhaust the wall-clock budget (returned best-so-far, NEVER
    # reached the final-verification rebench), or did it finish with budget to
    # spare and run the final verification tail?
    log_path = Path(args.log_csv).with_suffix(".log")
    out["budget_exceeded"] = None
    out["ran_final_verification"] = None
    if log_path.exists():
        log_text = log_path.read_text(errors="replace")
        out["budget_exceeded"] = "budget" in log_text and "exceeded" in log_text
        out["ran_final_verification"] = "Final verification" in log_text

    Path(args.out_json).write_text(json.dumps(out, default=str, indent=2))
    print(
        "DONE "
        + json.dumps(
            {
                k: out.get(k)
                for k in (
                    "kind",
                    "status",
                    "distinct_ok_config_ids",
                    "n_generations",
                    "per_config_median_s",
                    "autotune_wall_s",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
