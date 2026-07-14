"""Top-level sweep orchestrator (Phase A compile-probe -> Phase B timing) per cell.

Iterates shapes_matmul_perf.json (NOT the markdown). For each (kernel, shape):
  Phase A: for each arm, run compile_probe in an isolated setsid process with a
           120 s timeout; killpg on expiry -> status=timeout. Records survivors.
  Phase B: run time_cell in a fresh process, timing only the surviving arms;
           interleaved M1+M2, per-round arrays. Emits 2 JSONL records (one/method).
Checkpoints results.jsonl after every method record. One process per (kernel,shape)
in Phase B (spec's "one fresh process per kernel"); arms measured together in it.

Foreground only, one job at a time. NEVER backgrounds a GPU job.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

ROOT = "/home/dev/helion-matmul-b200/_lab/matmul-perf-char"
PKG_PARENT = ROOT  # so `-m mmperf.x` resolves
WORKTREE = "/home/dev/helion-matmul-b200"
PY = "/home/dev/.venvs/helion/bin/python"


def child_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONPATH"] = f"{WORKTREE}:{PKG_PARENT}"
    env["TRITON_CACHE_DIR"] = f"{ROOT}/cache/triton"
    env["TORCHINDUCTOR_CACHE_DIR"] = f"{ROOT}/cache/inductor"
    # keep autotune logs quiet-ish; do NOT set HELION_AUTOTUNE_EFFORT (would change paths)
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def run_with_timeout(cmd: list[str], timeout_s: float) -> tuple[int, str, str, bool]:
    """Run cmd in its own process group; killpg on timeout. Returns
    (returncode, stdout, stderr, timed_out)."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env(),
        cwd=ROOT,
        preexec_fn=os.setsid,
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            out, err = "", "killpg cleanup timed out"
    return proc.returncode, out or "", err or "", timed_out


def phase_a_compile(kernel: str, arm: str, shape_json: str, fast_accum: int,
                    timeout_s: float) -> dict:
    cmd = [PY, "-m", "mmperf.compile_probe", kernel, arm, shape_json,
           "--fast-accum", str(fast_accum)]
    rc, out, err, timed_out = run_with_timeout(cmd, timeout_s)
    if timed_out:
        return {"status": "timeout", "error": f"compile > {timeout_s}s (killpg)"}
    # parse last json line
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"status": "compile_fail",
            "error": f"no json (rc={rc}); stderr tail: {err[-300:]}"}


def phase_b_time(kernel: str, shape_json: str, arms: list[str], reps: int,
                 fast_accum: int, flush_mib: int, timeout_s: float) -> tuple[list[dict], str]:
    cmd = [PY, "-m", "mmperf.time_cell", kernel, shape_json,
           "--arms", ",".join(arms), "--reps", str(reps),
           "--fast-accum", str(fast_accum), "--flush-mib", str(flush_mib)]
    rc, out, err, timed_out = run_with_timeout(cmd, timeout_s)
    recs = []
    for line in out.splitlines():
        if line.startswith("@@REC@@"):
            try:
                recs.append(json.loads(line[len("@@REC@@"):]))
            except json.JSONDecodeError:
                pass
    note = ""
    if timed_out:
        note = f"phase_b timeout > {timeout_s}s (killpg)"
    elif not recs:
        note = f"phase_b no records (rc={rc}); stderr tail: {err[-400:]}"
    return recs, note


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum",
                    default="/home/dev/prompts-lab/tasks/shapes_matmul_perf.json")
    ap.add_argument("--out", default=f"{ROOT}/results/results.jsonl")
    ap.add_argument("--kernels", default="matmul,fp8_gemm,bmm,mamba2_chunk_state")
    ap.add_argument("--only-shape", default="",
                    help="limit to shapes whose tuple matches, e.g. '2048,2048,2048' (debug)")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--flush-mib", type=int, default=512)
    ap.add_argument("--compile-timeout", type=float, default=120.0)
    ap.add_argument("--time-timeout", type=float, default=240.0)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="append mode; skip (kernel,shape) cells already in --out")
    args = ap.parse_args()

    already_done: set = set()
    if args.resume and os.path.exists(args.out):
        args.append = True
        with open(args.out) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kernel") and r.get("shape") and r.get("method") != "none":
                    already_done.add((r["kernel"], tuple(r["shape"])))

    with open(args.curriculum) as fh:
        cur = json.load(fh)
    kernels_want = [k for k in args.kernels.split(",") if k]

    from mmperf import common  # local import (needs PYTHONPATH); only for device hdr
    import torch  # noqa: F401
    dev = common.device_props()
    flush_mib = args.flush_mib
    header = {
        "_run_header": True,
        "device": dev["name"],
        "sm_tag": dev["sm_tag"],
        "cc": dev["cc"],
        "sm_count": dev["sm_count"],
        "l2_mib": dev["l2_mib"],
        "flush_mib": flush_mib,
        "peak_tflops": common.PEAK_TFLOPS,
        "clocks_start": common.clocks_temp(),
    }

    mode = "a" if args.append else "w"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    outfh = open(args.out, mode, buffering=1)  # line-buffered checkpoint
    if not args.append:
        outfh.write(json.dumps(header) + "\n")

    log = open(f"{ROOT}/results/sweep.log", mode, buffering=1)

    def emit(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")

    emit(f"device={dev['name']} {dev['sm_tag']} L2={dev['l2_mib']}MiB flush={flush_mib}MiB")

    total_cells = 0
    for kernel in kernels_want:
        kdef = cur["kernels"][kernel]
        shape_key = "mkn" if "mkn" in kdef["shapes"][0] else (
            "bmkn" if "bmkn" in kdef["shapes"][0] else "s")
        for spec in kdef["shapes"]:
            shape_json = json.dumps(spec)
            shp = spec[shape_key]
            if args.only_shape:
                want = tuple(int(x) for x in args.only_shape.split(","))
                if tuple(shp) != want:
                    continue
            if (kernel, tuple(shp)) in already_done:
                emit(f"--- SKIP (resume) {kernel} {shp} already done ---")
                continue
            total_cells += 1
            emit(f"=== {kernel} {shp} ({spec.get('type')}) ===")

            fast_accum = 1
            # Phase A: compile each arm, isolated + timeout
            survivors = []
            arm_a_status = {}
            for arm in ["seed", "helion_default", "tc_max_autotune"]:
                t0 = time.time()
                res = phase_a_compile(kernel, arm, shape_json, fast_accum,
                                      args.compile_timeout)
                arm_a_status[arm] = res
                dt = time.time() - t0
                emit(f"  A/{arm}: {res.get('status')} ({dt:.1f}s)"
                     + (f" cfg={res.get('cfg')}" if res.get('cfg') else "")
                     + (f" src={res.get('default_source')}" if res.get('default_source') else "")
                     + (f" ERR={res.get('error')[:120]}" if res.get('error') else ""))
                if res.get("status") == "ok":
                    survivors.append(arm)

            if not survivors:
                emit("  !! no arm survived Phase A; recording failure stub")
                stub = {
                    "kernel": kernel, "shape": list(shp), "type": spec.get("type"),
                    "phaseA": arm_a_status, "method": "none",
                    "note": "no arm compiled",
                }
                outfh.write(json.dumps(stub) + "\n")
                continue

            # Phase B: time survivors together
            recs, note = phase_b_time(kernel, shape_json, survivors, args.reps,
                                      fast_accum, flush_mib, args.time_timeout)
            if not recs:
                emit(f"  !! Phase B produced no records: {note}")
                stub = {
                    "kernel": kernel, "shape": list(shp), "type": spec.get("type"),
                    "phaseA": arm_a_status, "method": "none", "note": note,
                }
                outfh.write(json.dumps(stub) + "\n")
                continue

            for rec in recs:
                # fold Phase-A statuses into arms that didn't make it to Phase B
                rec["phaseA_status"] = {a: arm_a_status[a].get("status")
                                        for a in arm_a_status}
                for arm in ["seed", "helion_default", "tc_max_autotune"]:
                    if arm not in rec.get("arms", {}):
                        rec.setdefault("arms", {})[arm] = {
                            "status": arm_a_status.get(arm, {}).get("status", "compile_fail"),
                            "error": arm_a_status.get(arm, {}).get("error"),
                        }
                outfh.write(json.dumps(rec) + "\n")
                r = rec["ratios"]
                g = r.get("G_vs_tc", {}).get("median")
                xd = r.get("xD_vs_default", {}).get("median")
                g_s = f"{g:.3f}" if isinstance(g, (int, float)) else "n/a"
                xd_s = f"{xd:.2f}" if isinstance(xd, (int, float)) else "n/a"
                emit(f"  B/{rec['method']}: R={rec.get('R')} "
                     f"G_vs_tc={g_s} xD_vs_default={xd_s}")
            emit(f"  done {kernel} {shp}  (src={recs[0].get('default_source')})")

    footer = {"_run_footer": True, "clocks_end": common.clocks_temp(),
              "total_cells": total_cells}
    outfh.write(json.dumps(footer) + "\n")
    outfh.close()
    emit(f"SWEEP COMPLETE: {total_cells} cells -> {args.out}")


if __name__ == "__main__":
    main()
