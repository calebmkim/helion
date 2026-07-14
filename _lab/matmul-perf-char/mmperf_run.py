"""Orchestrator for the H100 matmul-family perf characterization.

Owns the shape loop + per-arm compile TIMEOUT (subprocess + killpg) + JSONL
checkpoint after every (kernel,shape,method). Foreground, ONE GPU subprocess at a
time (never backgrounded). Per (kernel,shape):
  1. prepare each of the 3 arms in its OWN subprocess (ptxas-hang / OOM containment):
     a hung [16,16,16] default is SIGKILLed via killpg and recorded status=timeout,
     never taking down the seed/tc timings.
  2. time the arms that passed prepare (status ok), M1+M2 interleaved, R rounds.
  3. derive median/CI/TFLOP-s/ratios FROM the raw per-round arrays; write one JSONL
     record per (kernel,shape,method).

Usage:
  CUDA_VISIBLE_DEVICES=0 MMPERF_WORKTREE=<wt> python mmperf_run.py \
      --shapes shapes_matmul_perf.json --out results.jsonl [--only matmul] \
      [--limit N] [--step0]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
MMPERF = os.path.join(HERE, "mmperf.py")
_DEFAULT_WT = os.path.dirname(os.path.dirname(HERE))
WORKTREE = os.environ.get("MMPERF_WORKTREE", _DEFAULT_WT)
COMPILE_TIMEOUT = 120  # §1: per-config compile timeout
TIME_TIMEOUT = 600     # generous ceiling for the timing subprocess


def run_sub(args_list, timeout_s, tmp_out):
    """Run mmperf.py subcommand in its OWN process group; killpg on timeout.
    Returns (parsed_json | None, status_hint, stderr_tail)."""
    env = dict(os.environ)
    env["MMPERF_WORKTREE"] = WORKTREE
    env.setdefault("PYTHONPATH", WORKTREE)
    t0 = time.time()
    proc = subprocess.Popen([PY, MMPERF] + args_list, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=30)
        except Exception:
            out, err = "", ""
        return None, "timeout", (err or "")[-800:] + f" [elapsed {time.time()-t0:.0f}s]"
    data = None
    if os.path.exists(tmp_out):
        try:
            data = json.load(open(tmp_out))
        except Exception:
            data = None
    hint = "ok" if rc == 0 else "compile_fail"
    return data, hint, (err or "")[-800:]


def prepare_arm(kernel, shape_str, dtype, arm, tmp_dir):
    tmp = os.path.join(tmp_dir, f"prep_{arm}.json")
    if os.path.exists(tmp):
        os.remove(tmp)
    data, hint, err = run_sub(
        ["prepare", "--kernel", kernel, "--shape", shape_str, "--dtype", dtype,
         "--arm", arm, "--out", tmp], COMPILE_TIMEOUT, tmp)
    if data is None:
        # killed (timeout) or crashed before writing
        return {"arm": arm, "status": hint if hint != "ok" else "timeout",
                "error": err or "subprocess died before writing", "config": None}
    return data


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def ci95(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = statistics.mean(xs)
    sd = statistics.stdev(xs)
    h = 1.96 * sd / math.sqrt(len(xs))
    return [round(m - h, 4), round(m + h, 4)]


def per_round_ratios(base_arr, seed_arr):
    """Per-round base/seed on the overlapping rounds."""
    n = min(len(base_arr), len(seed_arr))
    out = []
    for i in range(n):
        if seed_arr[i] and seed_arr[i] > 0 and base_arr[i]:
            out.append(round(base_arr[i] / seed_arr[i], 4))
    return out


def build_record(kernel, shape, dtype, method, prep, tinfo, di, meta_facts):
    """One JSONL record per (kernel,shape,method). Derives all summaries from raw t_us[]."""
    methods = tinfo.get("methods", {}) if tinfo else {}
    marms = methods.get(method, {})
    seed_prep = prep["seed"]
    rec = {
        "device": di["cc"], "gpu": di["name"], "l2_mib": di["l2_mib"],
        "flush_mib": di["flush_mib"], "kernel": kernel, "shape": shape, "dtype": dtype,
        "method": method, "type": meta_facts["type"],
        "seed_fired": seed_prep.get("seed_fired"),
        "heuristics_fired": seed_prep.get("heuristics_fired"),
        "default_source": "n_a",  # H100: B200 table is sm100-gated, never fires
        "seed_cfg": seed_prep.get("seed_cfg"),
        "default_cfg": seed_prep.get("default_cfg"),
        "n_seeds": seed_prep.get("n_seeds"),
        "matmul_facts": seed_prep.get("matmul_facts"),
        "R": (tinfo.get("R_eff") or tinfo.get("R")) if tinfo else None,
        "seed_est_us": tinfo.get("seed_est_us") if tinfo else None,
    }
    arms_out = {}
    for arm in ["seed", "helion_default", "tc_max_autotune"]:
        p = prep[arm]
        a = {"prep_status": p.get("status"), "acc_pass": p.get("acc_pass"),
             "max_abs": p.get("max_abs")}
        if p.get("status") == "ok" and arm in marms:
            m = marms[arm]
            a["status"] = m.get("status")
            a["t_us"] = m.get("t_us", [])
            a["median"] = median(m.get("t_us", []))
            if arm != "tc_max_autotune":
                a["config"] = p.get("config")
            if "inner" in m:
                a["inner"] = m["inner"]
            if m.get("error"):
                a["error"] = m["error"]
        else:
            a["status"] = p.get("status")  # prep failed => no timing
            a["t_us"] = []
            a["median"] = None
            if p.get("error"):
                a["error"] = p.get("error")[:300]
            if arm != "tc_max_autotune":
                a["config"] = p.get("config")
        arms_out[arm] = a
    rec["arms"] = arms_out

    # ratios (only when both arms ok with data)
    seed_arr = arms_out["seed"].get("t_us", []) if arms_out["seed"].get("status") == "ok" else []
    ratios = {}
    tc = arms_out["tc_max_autotune"]
    if seed_arr and tc.get("status") == "ok" and tc.get("t_us"):
        pr = per_round_ratios(tc["t_us"], seed_arr)  # tc/seed
        if pr:
            ratios["G_vs_tc"] = {"per_round": pr, "median": round(median(pr), 4), "ci": ci95(pr)}
    dfl = arms_out["helion_default"]
    if seed_arr and dfl.get("status") == "ok" and dfl.get("t_us"):
        pr = per_round_ratios(dfl["t_us"], seed_arr)  # default/seed
        if pr:
            ratios["xD_vs_default"] = {"per_round": pr, "median": round(median(pr), 4), "ci": ci95(pr)}
    rec["ratios"] = ratios

    # TFLOP/s + %-peak + implied BW from the SEED median (canonical arm)
    smed = arms_out["seed"].get("median")
    if smed and smed > 0:
        fl = meta_facts["flops"]
        tflops = fl / (smed * 1e-6) / 1e12
        rec["tflops_seed"] = round(tflops, 2)
        peak = di["peak_tflops"].get(dtype, 989.5)
        rec["pct_peak_seed"] = round(100.0 * tflops / peak, 2)
        bts = meta_facts["bytes"]
        rec["bw_tbps"] = round(bts / (smed * 1e-6) / 1e12, 4)
    rec["max_abs"] = seed_prep.get("max_abs")
    rec["acc_pass"] = arms_out["seed"].get("acc_pass")
    # noise floor flag: seed median < threshold
    thr = 25.0
    rec["noise_floor"] = bool(smed is not None and smed < thr)
    return rec


def iter_shapes(spec, only=None):
    ks = spec["kernels"]
    for kernel, kd in ks.items():
        if only and kernel != only:
            continue
        dtype = kd["dtype"].replace("_e4m3", "").replace("fp8_e4m3", "fp8")
        if kernel == "fp8_gemm":
            dtype = "fp8"
        for sh in kd["shapes"]:
            if "mkn" in sh:
                shape = sh["mkn"]
                shape_str = ",".join(str(x) for x in shape)
            elif "bmkn" in sh:
                shape = sh["bmkn"]
                shape_str = ",".join(str(x) for x in shape)
            elif "s" in sh:
                shape = sh["s"]
                shape_str = ",".join(str(x) for x in shape)
            else:
                continue
            yield kernel, dtype, shape, shape_str, sh["type"], sh.get("prov", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="single kernel")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--step0", action="store_true", help="only matmul 4096^3 bf16")
    ap.add_argument("--tmp", default="/tmp/mmperf_tmp")
    ap.add_argument("--resume", action="store_true", help="skip (kernel,shape) already in out")
    a = ap.parse_args()

    os.makedirs(a.tmp, exist_ok=True)
    spec = json.load(open(a.shapes))
    R_default = spec["_meta"].get("R_rounds_default", 7)
    R_small = spec["_meta"].get("R_rounds_small_shapes", 15)
    small_thr = spec["_meta"].get("small_shape_us_threshold", 25)

    # device info once (via subcommand so it's in the GPU process)
    dtmp = os.path.join(a.tmp, "devinfo.json")
    di_data, _, _ = run_sub_devinfo(dtmp)
    di = di_data

    # flops/bytes replicated inline (importing mmperf would init CUDA in the orchestrator).
    def flops(kernel, shape):
        if kernel in ("matmul", "fp8_gemm"):
            m, k, n = shape
            return 2.0 * m * n * k
        if kernel == "bmm":
            b, m, k, n = shape
            return 2.0 * b * m * n * k
        if kernel == "mamba2_chunk_state":
            b, seq, nh, chunk, hd, ds = shape
            nchunks = (seq + chunk - 1) // chunk
            return 2.0 * (b * nchunks * nh) * hd * ds * chunk
        return 0.0

    def bytes_moved(kernel, shape, dtype):
        isz = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}[dtype]
        osz = 2
        if kernel in ("matmul", "fp8_gemm"):
            m, k, n = shape
            return isz * (m * k + k * n) + osz * (m * n)
        if kernel == "bmm":
            b, m, k, n = shape
            return isz * b * (m * k + k * n) + osz * b * (m * n)
        if kernel == "mamba2_chunk_state":
            b, seq, nh, chunk, hd, ds = shape
            return isz * (b * seq * ds + b * seq * nh * hd) + osz * (b * (seq // chunk) * nh * hd * ds)
        return 0.0

    done = set()
    if a.resume and os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["kernel"], tuple(r["shape"]), r["method"]))
            except Exception:
                pass

    shapes = list(iter_shapes(spec, only=a.only))
    if a.step0:
        shapes = [s for s in shapes if s[0] == "matmul" and s[2] == [4096, 4096, 4096]]
    if a.limit:
        shapes = shapes[:a.limit]

    outfh = open(a.out, "a")
    n = len(shapes)
    for idx, (kernel, dtype, shape, shape_str, typ, prov) in enumerate(shapes):
        tag = f"[{idx+1}/{n}] {kernel} {shape} {dtype} ({typ})"
        # skip the ENTIRE cell (prepare+time) when both methods already recorded
        if a.resume and all((kernel, tuple(shape), m) in done
                            for m in ["M1_cudagraph_coldL2", "M2_do_bench_coldL2"]):
            print(f"\n=== {tag} === SKIP (already in {os.path.basename(a.out)})", flush=True)
            continue
        print(f"\n=== {tag} ===", flush=True)
        clocks_before = run_sub_devinfo(os.path.join(a.tmp, "clk.json"))[0].get("clocks")

        # 1. prepare all 3 arms (isolated)
        prep = {}
        for arm in ["seed", "helion_default", "tc_max_autotune"]:
            p = prepare_arm(kernel, shape_str, dtype, arm, a.tmp)
            prep[arm] = p
            print(f"  prepare {arm:16s}: {p.get('status'):12s} acc={p.get('acc_pass')} "
                  f"ma={p.get('max_abs')}", flush=True)

        ok_arms = {arm: {"status": prep[arm].get("status"), "config": prep[arm].get("config")}
                   for arm in prep if prep[arm].get("status") == "ok"}

        # R is self-calibrated INSIDE the timing subprocess from the seed's measured
        # cold-L2 device time (sub-small_thr us -> bump to R_small); we pass both.
        R = R_default

        # 2. time the ok arms
        tinfo = {}
        if ok_arms:
            ttmp = os.path.join(a.tmp, "time.json")
            if os.path.exists(ttmp):
                os.remove(ttmp)
            data, hint, err = run_sub(
                ["time", "--kernel", kernel, "--shape", shape_str, "--dtype", dtype,
                 "--arms", json.dumps(ok_arms), "--rounds", str(R_default),
                 "--small-us", str(small_thr), "--small-rounds", str(R_small),
                 "--out", ttmp],
                TIME_TIMEOUT, ttmp)
            if data is None:
                print(f"  TIME subprocess failed/timeout: {err[:200]}", flush=True)
                tinfo = {"methods": {}, "R": R, "time_error": err[:400]}
            else:
                tinfo = data
                print(f"  timed arms: {data.get('arms_timed')}", flush=True)
        else:
            print("  no ok arms to time", flush=True)
            tinfo = {"methods": {}, "R": R}

        clocks_after = run_sub_devinfo(os.path.join(a.tmp, "clk.json"))[0].get("clocks")

        # 3. one record per method
        meta_facts = {"type": typ, "prov": prov,
                      "flops": flops(kernel, shape), "bytes": bytes_moved(kernel, shape, dtype)}
        for method in ["M1_cudagraph_coldL2", "M2_do_bench_coldL2"]:
            if a.resume and (kernel, tuple(shape), method) in done:
                continue
            rec = build_record(kernel, shape, dtype, method, prep, tinfo, di, meta_facts)
            rec["clocks"] = {"before": clocks_before, "after": clocks_after}
            rec["prov"] = prov
            outfh.write(json.dumps(rec, default=str) + "\n")
            outfh.flush()
            os.fsync(outfh.fileno())
        # concise progress line
        def med(arm, method):
            m = tinfo.get("methods", {}).get(method, {}).get(arm, {})
            return median(m.get("t_us", []))
        s1 = med("seed", "M1_cudagraph_coldL2")
        d1 = med("helion_default", "M1_cudagraph_coldL2")
        t1 = med("tc_max_autotune", "M1_cudagraph_coldL2")
        print(f"  M1 medians us: seed={s1} default={d1} tc={t1} "
              f"| G_vs_tc={round(t1/s1,3) if (s1 and t1) else None} "
              f"xD={round(d1/s1,3) if (s1 and d1) else None}", flush=True)

    outfh.close()
    print(f"\nDONE: {n} shapes -> {a.out}", flush=True)


def run_sub_devinfo(tmp):
    env = dict(os.environ)
    env["MMPERF_WORKTREE"] = WORKTREE
    env.setdefault("PYTHONPATH", WORKTREE)
    proc = subprocess.run([PY, MMPERF, "devinfo"], env=env, capture_output=True, text=True, timeout=60)
    try:
        return json.loads(proc.stdout), "ok", ""
    except Exception:
        return {}, "fail", proc.stderr[-400:]


if __name__ == "__main__":
    main()
