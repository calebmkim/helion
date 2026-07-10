"""Overnight driver: run every (corpus, kernel) as its OWN process (fresh dynamo/CUDA state,
one hang/OOM can't poison the rest), sequentially (GPU discipline: never concurrent). Each
process is bounded by a wall-clock timeout; on timeout/crash we log and move on.

Modes:
  --smoke   : one cheapest cell per (corpus,kernel) — the pre-launch gate (proves every path
              builds -> times -> writes). ~35 cells, minutes.
  (default) : the full matrix (455 cells). Resumable: skips a (corpus,kernel) whose output JSON
              already has all its cells (so a re-launch continues where it stopped).

Usage:
  PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES=0 python perf-repro/run_all.py --smoke --out-dir <D>
  PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES=0 python perf-repro/run_all.py --out-dir <D>
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

_PERF = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.join(_PERF, "perf_report_bench.py")
_WT = os.path.abspath(os.path.join(_PERF, ".."))
_PY = os.environ.get("HELION_PY", sys.executable)


def _pairs(shapes):
    out = []
    for corpus, cd in shapes["corpora"].items():
        for kernel, kd in cd["kernels"].items():
            split = cd["required_splits"][0]
            shps = kd["shapes"][split]
            dtypes = cd["dtypes"]
            out.append((corpus, kernel, shps, dtypes))
    return out


def _cheapest(shps):
    """The smallest shape (min product) — cheapest to build+time for the smoke gate."""
    def prod(s):
        p = 1
        for x in s:
            p *= x
        return p
    return min(shps, key=prod)


def _only_str(shape, dtype):
    return "x".join(str(x) for x in shape) + ":" + dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true", help="one cheapest cell per (corpus,kernel)")
    ap.add_argument("--per-proc-timeout", type=int, default=int(os.environ.get("RUN_PROC_TIMEOUT", "3600")),
                    help="wall-clock seconds per (corpus,kernel) process")
    ap.add_argument("--only-corpus", default=None, help="restrict to one corpus")
    args = ap.parse_args()

    shapes = json.load(open(os.path.join(_PERF, "shapes.json")))
    os.makedirs(args.out_dir, exist_ok=True)
    logdir = os.path.join(args.out_dir, "logs")
    os.makedirs(logdir, exist_ok=True)

    pairs = _pairs(shapes)
    if args.only_corpus:
        pairs = [p for p in pairs if p[0] == args.only_corpus]

    env = dict(os.environ)
    env["PYTHONPATH"] = _WT
    env["HELION_AUTOTUNE_EFFORT"] = "none"

    def _expected_cells(shps, dtypes):
        return len(shps) * len(dtypes)

    summary = []
    t_start = time.time()
    for i, (corpus, kernel, shps, dtypes) in enumerate(pairs, 1):
        outp = os.path.join(args.out_dir, f"{corpus}__{kernel}.json")
        tag0 = f"[{i}/{len(pairs)}] {corpus}/{kernel}"
        # KERNEL-LEVEL SKIP: if this (corpus,kernel) already has all its cells recorded, skip it
        # (full-run resume after a driver restart). Smoke mode never skips.
        if not args.smoke and os.path.exists(outp):
            try:
                have = len(json.load(open(outp)).get("rows", []))
                if have >= _expected_cells(shps, dtypes):
                    print(f"{tag0} SKIP (already {have} cells)", flush=True)
                    summary.append({"corpus": corpus, "kernel": kernel, "status": "skip",
                                    "secs": 0.0, "cells": have})
                    continue
            except Exception:  # noqa: BLE001
                pass
        cmd = [_PY, "-u", _BENCH, "--corpus", corpus, "--kernel", kernel, "--out-dir", args.out_dir]
        if args.smoke:
            s = _cheapest(shps)
            cmd += ["--only-shapes", _only_str(s, dtypes[0])]
        else:
            cmd += ["--resume"]  # cell-level resume within the kernel (continues a crashed process)
        # fresh cache dir per process so a corrupt compile can't poison the next kernel
        env["HELION_CACHE_DIR"] = os.path.join(args.out_dir, f".cache_{corpus}_{kernel}")
        logf = os.path.join(logdir, f"{corpus}__{kernel}.log")
        tag = f"[{i}/{len(pairs)}] {corpus}/{kernel}"
        print(f"{tag} -> {logf}", flush=True)
        t0 = time.time()
        status = "ok"
        try:
            with open(logf, "w") as lf:
                r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                   timeout=args.per_proc_timeout)
            if r.returncode != 0:
                status = f"exit{r.returncode}"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            # reap any orphaned ptxas/compile workers from the killed process
            subprocess.run(["pkill", "-9", "-f", "compile_worker"], check=False)
            subprocess.run(["pkill", "-9", "-f", "ptxas"], check=False)
        dt = time.time() - t0
        # did it actually write cells?
        outp = os.path.join(args.out_dir, f"{corpus}__{kernel}.json")
        ncells = len(json.load(open(outp)).get("rows", [])) if os.path.exists(outp) else 0
        print(f"    {status}  {dt:.0f}s  {ncells} cell(s)  (see {logf})", flush=True)
        summary.append({"corpus": corpus, "kernel": kernel, "status": status,
                        "secs": round(dt, 1), "cells": ncells})

    # write a run manifest
    man = {"mode": "smoke" if args.smoke else "full", "total_secs": round(time.time() - t_start, 1),
           "pairs": summary}
    json.dump(man, open(os.path.join(args.out_dir, "run_manifest.json"), "w"), indent=1)
    bad = [p for p in summary if p["status"] != "ok" or p["cells"] == 0]
    print(f"\n=== {'SMOKE' if args.smoke else 'FULL'} DONE in {man['total_secs']:.0f}s: "
          f"{len(summary)-len(bad)}/{len(summary)} clean ===", flush=True)
    if bad:
        print("PROBLEMS:", flush=True)
        for p in bad:
            print(f"  {p['corpus']}/{p['kernel']}: status={p['status']} cells={p['cells']}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
