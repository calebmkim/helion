"""Autotune ONE matmul/fp8 shape (answer-key oracle for a stuck shape). In-process
benchmark (HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0 set by caller) avoids the matmul
lambda-epilogue pickle / spawn-worker. Prints the winning config; re-bench it
separately via bench.py for a fair cold-L2 number (never trust the autotuner's own).

Usage: HELION_AUTOTUNE_EFFORT=full HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0 \
  python autotune_one.py --kernel matmul --shape 16384,8192,512 --dtype bf16 --out o.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as B

import helion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shape = [int(s) for s in a.shape.split(",")]
    fn = B._kernel_fn(a.kernel)
    ss = B.STATIC_SHAPES[a.kernel]
    args, ref, tc_fn, meta = B.make_inputs(a.kernel, shape, a.dtype)
    k = helion.kernel(fn, static_shapes=ss)
    info = {"kernel": a.kernel, "shape": shape, "dtype": a.dtype, "meta": meta,
            "effort": os.environ.get("HELION_AUTOTUNE_EFFORT"),
            "subproc": os.environ.get("HELION_AUTOTUNE_BENCHMARK_SUBPROCESS")}
    t0 = time.time()
    try:
        best = k.autotune(args, force=True)
        info["ok"] = True
        info["winner_config"] = dict(best)
    except Exception as e:
        info.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()[-2000:]})
    info["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(info, open(a.out, "w"), indent=2, default=str)
    print("AUTOTUNE " + json.dumps({"ok": info.get("ok"), "s": info.get("wall_clock_s"),
                                    "winner": info.get("winner_config")}, default=str))


if __name__ == "__main__":
    main()
