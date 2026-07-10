#!/usr/bin/env python
"""Turnkey audit driver for the H100 matmul-seed climb.

Runs bench.py (cold-L2 + accuracy-gated) once per shape in an ISOLATED subprocess -- the
correct methodology (no cross-kernel autotune/compile-cache contamination) -- over a split
from shapes.py, then prints per-shape default/seed timings + ratios and the geomean.

Usage (one H100, pinned; run from the worktree):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<worktree> \
      python _lab/matmul-h100/run_audit.py --kernel matmul --split train --tc --out /tmp/audit.json

--tc adds the cuBLAS (torch.compile) roofline arm -> G_vs_tc (slower; omit for a quick
default-vs-seed pass -> x_over_default only). --split all runs train+val+test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shapes as S  # the curriculum (shapes.py, this dir)


def shape_str(kernel: str, tup: tuple) -> tuple[str, str]:
    """(shape-tuple) -> (comma-int --shape string, dtype) matching bench.py's parsing."""
    if kernel == "mamba2_chunk_state":
        b, seq, nh, chunk, hd, ds, dt = tup
        return f"{b},{seq},{nh},{chunk},{hd},{ds}", dt
    m, k, n, dt = tup  # matmul / fp8_gemm: (M, K, N, dtype)
    return f"{m},{k},{n}", dt


def geomean(xs: list) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=list(S.SPLITS))
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--tc", action="store_true", help="add cuBLAS roofline arm (-> G_vs_tc)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="cap #shapes (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    splits = ["train", "val", "test"] if a.split == "all" else [a.split]
    tuples = [t for sp in splits for t in S.SPLITS[a.kernel][sp]]
    if a.limit:
        tuples = tuples[: a.limit]

    bench = os.path.join(HERE, "bench.py")
    rows = []
    for tup in tuples:
        sh, dt = shape_str(a.kernel, tup)
        outp = tempfile.NamedTemporaryFile("r", suffix=".json", delete=False).name
        cmd = [sys.executable, bench, "--kernel", a.kernel, "--shape", sh, "--dtype", dt,
               "--configs", '["default","seed"]', "--reps", str(a.reps), "--out", outp]
        if a.tc:
            cmd.append("--tc")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            info = json.load(open(outp))
        except Exception:
            info = {"ok": False, "error": (proc.stderr or "")[-500:]}
        res = {r["label"]: r for r in info.get("results", [])}
        seed = res.get("seed0", {})
        row = {
            "shape": list(tup), "dtype": dt, "ok": info.get("ok"),
            "default_ms": res.get("default", {}).get("perf_ms"),
            "seed_ms": seed.get("perf_ms"),
            "tc_ms": res.get("tc", {}).get("perf_ms"),
            "x_over_default": seed.get("x_over_default"),
            "G_vs_tc": seed.get("G_vs_tc"),
            "seed_config": seed.get("config"),
            "seed_acc_ok": seed.get("accuracy_ok"),
        }
        rows.append(row)
        print(f"{a.kernel} {tup}: seed_ms={row['seed_ms']} "
              f"xD={row['x_over_default']} G={row['G_vs_tc']} acc={row['seed_acc_ok']}")

    summary = {
        "kernel": a.kernel, "split": a.split, "n_shapes": len(rows),
        "geomean_G_vs_tc": geomean([r["G_vs_tc"] for r in rows]),
        "geomean_x_over_default": geomean([r["x_over_default"] for r in rows]),
        "n_accuracy_fail": sum(1 for r in rows if r["seed_acc_ok"] is False),
    }
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=2, default=str)
    print("SUMMARY " + json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
