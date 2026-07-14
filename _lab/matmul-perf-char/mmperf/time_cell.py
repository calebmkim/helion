"""Phase-B timing process for ONE (kernel, shape) cell.

Runs in a fresh subprocess (fresh dynamo state; cache-warm from Phase A so NO ptxas
re-invocation -> cannot hang). Only times the arms passed in --arms (those that
survived Phase A compile). Builds each arm's callable, runs the fp32 accuracy gate,
then interleaved M1 (cudagraph-diff) + M2 (do_bench-style) cold-L2 timing with full
per-round arrays retained. Emits the two per-method records (as JSON) to stdout.

Usage:
    python -m mmperf.time_cell <kernel> <shape_json> --arms seed,helion_default,tc_max_autotune
        [--reps 7] [--fast-accum 1] [--flush-mib 512]
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from mmperf import common
from mmperf import kernels


def build_helion_arm(bound, cfg):
    compiled = bound.compile_config(cfg)
    return compiled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("shape_json")
    ap.add_argument("--arms", default="seed,helion_default,tc_max_autotune")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--reps-small", type=int, default=15)
    ap.add_argument("--small-us", type=float, default=25.0)
    ap.add_argument("--fast-accum", type=int, default=1)
    ap.add_argument("--flush-mib", type=int, default=512)
    args = ap.parse_args()

    common.set_fairness_locks()
    spec = json.loads(args.shape_json)
    kdef = kernels.KERNELS[args.kernel]
    want_arms = [a for a in args.arms.split(",") if a]

    arg_tuple, ref, meta = kdef["make_inputs"](spec)
    bound = kdef["bind"](arg_tuple)
    cfgs = common.extract_configs(bound)

    dtype = kdef["dtype"]
    shape = list(kernels.shape_tuple(args.kernel, spec))

    record_common = {
        "kernel": args.kernel,
        "shape": shape,
        "type": spec.get("type"),
        "prov": spec.get("prov"),
        "dtype": dtype,
        "seed_fired": cfgs["seed_fired"],
        "default_source": cfgs["default_source"],
        "promoted_is_formula": cfgs["promoted_is_formula"],
        "seed_cfg": common.cfg_summary(cfgs["seed_cfg"]),
        "default_cfg": common.cfg_summary(cfgs["default_cfg"]),
        "meta": meta,
    }

    # ---- build arm callables + accuracy gate ----
    arm_fns: dict = {}
    arm_status: dict = {}
    arm_acc: dict = {}
    arm_winner: dict = {}

    def make_call(compiled, arg_tuple):
        # bind compiled+args NOW (default-arg capture) so we never close over a
        # loop/rebound variable -- the late-binding-closure footgun.
        return lambda: compiled(*arg_tuple)

    def gate(name, fn):
        try:
            out = fn()
            torch.cuda.synchronize()
            acc = common.accuracy(out, ref)
            arm_acc[name] = acc
            arm_status[name] = "ok" if acc["acc_ok"] else "acc_fail"
            if acc["acc_ok"]:
                arm_fns[name] = fn
        except torch.cuda.OutOfMemoryError as e:
            arm_status[name] = "oom"
            arm_acc[name] = {"error": str(e)[:200]}
        except Exception as e:  # noqa: BLE001
            arm_status[name] = "compile_fail"
            arm_acc[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    if "seed" in want_arms and cfgs["seed_cfg"] is not None:
        c_seed = build_helion_arm(bound, cfgs["seed_cfg"])
        gate("seed", make_call(c_seed, arg_tuple))
    if "helion_default" in want_arms and cfgs["default_cfg"] is not None:
        c_def = build_helion_arm(bound, cfgs["default_cfg"])
        gate("helion_default", make_call(c_def, arg_tuple))
    if "tc_max_autotune" in want_arms:
        if args.kernel == "fp8_gemm":
            tcfn = kdef["tc"](arg_tuple, fast_accum=bool(args.fast_accum))
        else:
            tcfn = kdef["tc"](arg_tuple)
        gate("tc_max_autotune", tcfn)
        # NOTE: this is a STATIC expectation, not a per-cell verification. Profiling
        # (results/tc_backend_probe.md) shows it holds for all GEMM cells EXCEPT
        # matmul M=1 decode, where Inductor picks a Triton GEMV. Treat as a hint;
        # the profiled backend map is authoritative.
        arm_winner["tc_max_autotune"] = (
            "cuBLAS/cuBLASLt (expected; profiler-verified except matmul M=1 decode=Triton)"
            if kdef["tc_is_cublas"] else "Triton (no cuBLAS analog)"
        )

    # ---- decide R: noise-floor bump for tiny shapes, budget cap for slow arms ----
    # Probe each surviving arm's rough single-call cost (M2, few reps) so we can
    # (a) bump R to 15 when the seed is < small_us (noise floor), and (b) CAP R so a
    # catastrophic helion_default (seconds/call on a huge GEMM) can't blow wall-time.
    timer = common.ColdL2Timer(args.flush_mib)
    reps = args.reps
    probe_med: dict = {}
    if arm_fns:
        pr = timer.m2_do_bench(arm_fns, reps=4, warmup=2)
        probe_med = {k: common.median(v) for k, v in pr.items()}
    seed_med = probe_med.get("seed")
    max_med = max(probe_med.values()) if probe_med else 0.0
    if seed_med is not None and seed_med < args.small_us:
        reps = args.reps_small
    # M1 batches `inner` graph replays per event window to lift a tiny kernel out of
    # event/flush noise; a multi-second helion_default would make inner=10 explode, so
    # scale it down once the slowest arm is comfortably above event resolution.
    if max_med >= 2000.0:
        inner = 1
    elif max_med >= 300.0:
        inner = 3
    else:
        inner = 10
    # budget cap: keep total device time under ~90 s. M1 replays each arm `inner`x per
    # round (+ an inner baseline); M2 once per arm per round. Cost/round ~
    # (inner+1)*max_med(M1) + sum(M2). Cap reps so 2 methods stay bounded.
    total_us = sum(probe_med.values()) if probe_med else 0.0
    m1_per_round = (inner + 1) * total_us
    m2_per_round = total_us
    per_round = m1_per_round + m2_per_round
    if per_round > 0:
        budget_us = 90.0 * 1e6
        max_reps = int(budget_us / per_round)
        if max_reps < reps:
            reps = max(3, max_reps)
    record_common["R"] = reps
    record_common["m1_inner"] = inner
    record_common["probe_med_us"] = probe_med

    # ---- interleaved timing (both methods) ----
    # fewer warmups when an arm is very slow (a multi-second default * 5 warmups would
    # blow the phase-B timeout); keep 5 for normal kernels for stable clocks.
    warm = 2 if max_med >= 2000.0 else 5
    m1 = timer.m1_cudagraph(arm_fns, reps=reps, inner=inner, warmup=warm) if arm_fns else {}
    m2 = timer.m2_do_bench(arm_fns, reps=reps, warmup=warm) if arm_fns else {}

    flops = kdef["flops"](spec)

    def make_method_record(method_name, per_arm):
        arms_out = {}
        for name in want_arms:
            st = arm_status.get(name, "compile_fail")
            entry = {"status": st}
            if name in arm_acc and "acc_ok" in arm_acc[name]:
                entry["acc_ok"] = arm_acc[name]["acc_ok"]
                entry["rel"] = arm_acc[name]["rel"]
            elif name in arm_acc:
                entry["error"] = arm_acc[name].get("error")
            if name in per_arm and per_arm[name]:
                t = per_arm[name]
                entry["t_us"] = t
                entry["median"] = common.median(t)
                entry["ci"] = common.ci95(t)
            if name in arm_winner:
                entry["winner"] = arm_winner[name]
            arms_out[name] = entry

        rec = dict(record_common)
        rec["method"] = method_name
        rec["arms"] = arms_out

        # ratios (only when both arms ok this method)
        def med(name):
            a = arms_out.get(name, {})
            return a.get("median") if a.get("status") == "ok" and "median" in a else None

        seed_m = med("seed")
        ratios = {}
        if seed_m and seed_m > 0:
            tc_m = med("tc_max_autotune")
            def_m = med("helion_default")
            # per-round ratios (align by round index; require both arrays)
            def per_round(base_name):
                b = per_arm.get(base_name)
                s = per_arm.get("seed")
                if not b or not s:
                    return None
                n = min(len(b), len(s))
                return [b[i] / s[i] for i in range(n) if s[i] > 0]
            if tc_m:
                pr = per_round("tc_max_autotune")
                ratios["G_vs_tc"] = {
                    "median": common.median(pr) if pr else tc_m / seed_m,
                    "ci": common.ci95(pr) if pr else [None, None],
                    "per_round": pr,
                }
            if def_m:
                pr = per_round("helion_default")
                ratios["xD_vs_default"] = {
                    "median": common.median(pr) if pr else def_m / seed_m,
                    "ci": common.ci95(pr) if pr else [None, None],
                    "per_round": pr,
                }
            # seed perf metrics
            rec["tflops_seed"] = flops / (seed_m * 1e-6) / 1e12
            peak = common.PEAK_TFLOPS.get(dtype)
            if peak:
                rec["pct_peak_seed"] = 100.0 * rec["tflops_seed"] / peak
        rec["ratios"] = ratios
        if "seed" in arm_acc and "rel" in arm_acc["seed"]:
            rec["acc_pass"] = arm_acc["seed"]["acc_ok"]
            rec["max_rel"] = arm_acc["seed"]["rel"]
        rec["noise_floor"] = reps >= args.reps_small
        return rec

    for method_name, per_arm in [
        ("M1_cudagraph_coldL2", m1),
        ("M2_do_bench_coldL2", m2),
    ]:
        sys.stdout.write("@@REC@@" + json.dumps(make_method_record(method_name, per_arm)) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
