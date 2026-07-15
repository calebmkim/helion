#!/usr/bin/env python
"""CUDA-graph TRUE-GPU-TIME co-bench — controls for CPU launch overhead.

Motivation: for L2-resident shapes the helion-cute launch path costs ~112us of
HOST dispatch while the GPU work is ~12us. Non-cudagraph timers (stock do_bench,
wall-clock) are dominated by / fragile to whether launches pipeline and hide that
CPU cost — which manufactured the fake 4.5x "seed bimodality". CUDA-graph capture
replays the kernel with per-launch CPU dispatch REMOVED -> true device time.

Per arm reports, all single-process under one thermal warmup:
  - cg_cold : CUDA-graph replay, L2 flushed before each replay (true GPU cold-L2)
  - cg_warm : CUDA-graph replay, no flush (true GPU warm-L2)
  - ev_cold : event-timed NON-graph, L2 flush/rep (bridge to stored cobench; carries
              the launch-overhead confound on purpose, to show the artifact)
Matmul (mm/fp8) and attention both supported. aten/sdpa included for grounding.

Input  : --spec <cobench_spec.json>   (same specs used by mm_cobench/attn_cobench)
Output : --out-json <file>
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import statistics
import sys
import traceback

WORKTREE = "/home/dev/local/helion-rank0"
sys.path.insert(0, str(Path(WORKTREE) / "benchmarks" / "cute"))


@contextlib.contextmanager
def _scrubbed_argv():
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        yield
    finally:
        sys.argv = saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--family", choices=["matmul", "attention"], required=True)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--thermal-ms", type=int, default=8000)
    a = ap.parse_args()

    spec = json.loads(Path(a.spec).read_text())
    out = dict(spec)
    out["kind"] = "cudagraph_cobench"

    import os

    os.environ["HELION_BACKEND"] = "cute"
    import torch

    import helion

    assert WORKTREE in helion.__file__, f"WRONG HELION: {helion.__file__}"
    out["helion_file"] = helion.__file__

    from triton import runtime as TR

    active = TR.driver.active
    l2 = active.get_empty_cache_for_benchmark()
    di = active.get_device_interface()

    def _clear():
        active.clear_cache(l2)

    # ---- build family-specific problem + kernel factory ----
    if a.family == "matmul":
        import compare_matmul_backends as M

        sh = spec["shape"]
        m, n, k = sh["m"], sh["n"], sh["k"]
        epilogue = spec.get("epilogue", "none")
        dtype_name = spec.get("dtype", "bfloat16")
        ns = argparse.Namespace(
            m=m, n=n, k=k, epilogue=epilogue, dtype=dtype_name, seed=0
        )
        dtype, mat_a, mat_b, bias, residual = M._make_matmul_problem(ns)
        expected = M._matmul_expected(ns, mat_a, mat_b, bias, residual, dtype)
        kargs = M._helion_matmul_args(ns, mat_a, mat_b, bias, residual)
        if epilogue == "scaled_mm":
            from examples.fp8_matmul import fp8_matmul as kernel
        else:
            from examples.matmul import matmul as kernel
        flop = 2.0 * m * n * k

        def make_helion(arm):
            cfg = helion.Config(**arm["config"])
            bound = kernel.bind(kargs)
            if any(key in arm["config"] for key in M._TCGEN05_CONFIG_KEYS):
                bound.env.config_spec.cute_tcgen05_search_enabled = True
            bound.set_config(cfg)
            codegen = None
            try:
                codegen = M._helion_codegen_markers(bound.to_triton_code(cfg))
            except Exception as e:
                codegen = {"error": f"{type(e).__name__}: {e}"}
            return (lambda b=bound: b(*kargs)), codegen

        def make_ref(arm):
            if epilogue == "scaled_mm":
                sa, sb = M._scaled_mm_scales(ns)
                return lambda: torch._scaled_mm(
                    mat_a, mat_b, sa, sb, use_fast_accum=False, out_dtype=torch.bfloat16
                )
            return lambda: M._apply_epilogue(ns, mat_a @ mat_b, bias, residual, dtype)

        def accuracy(actualfn):
            try:
                M._check_close(actualfn(), expected, dtype)
                return "PASS"
            except Exception as e:
                return f"FAIL:{type(e).__name__}"

        ref_kinds = {"aten"}
    else:
        import compare_attention_backends as A

        sh = spec["shape"]
        variant = spec["variant"]
        ns = argparse.Namespace(
            z=sh["z"],
            h=sh["h"],
            seq_len=sh["seq_len"],
            head_dim=sh["head_dim"],
            dtype=spec["dtype"],
            causal=1 if variant == "causal" else 0,
            biased=1 if variant == "biased" else 0,
            seed=0,
            helion_return_lse=0,
            helion_cute_benchmark_timer="wall",
        )
        dtype = A._dtype_from_name(spec["dtype"])
        q, kk, vv = A._make_inputs(ns, dtype)
        bias = A._make_bias(ns, dtype)
        causal = bool(ns.causal)
        flop = A._attention_flops(ns)
        from examples.attention import attention_output
        from examples.attention import biased_attention_output
        from examples.attention import causal_attention_output

        if bias is not None:
            kernel, kargs = biased_attention_output, (q, kk, vv, bias)
        elif causal:
            kernel, kargs = causal_attention_output, (q, kk, vv)
        else:
            kernel, kargs = attention_output, (q, kk, vv)
        expected = A._sdpa_reference(q, kk, vv, causal=causal, bias=bias)

        def make_helion(arm):
            with _scrubbed_argv():
                cfg = helion.Config(**arm["config"])
                bound = kernel.bind(kargs)
                bound.set_config(cfg)
                codegen = None
                try:
                    codegen = A._helion_codegen_markers(bound.to_triton_code(cfg))
                except Exception as e:
                    codegen = {"error": f"{type(e).__name__}: {e}"}
                return (lambda b=bound: b(*kargs)), codegen

        def make_ref(arm):
            return lambda: A._sdpa_reference(q, kk, vv, causal=causal, bias=bias)

        def accuracy(actualfn):
            try:
                return "PASS" if A._check_close(actualfn(), expected, dtype) else "FAIL"
            except Exception as e:
                return f"FAIL:{type(e).__name__}"

        ref_kinds = {"sdpa"}

    # ---- thermal warmup ONCE ----
    M_warm = __import__("compare_matmul_backends") if a.family == "attention" else M
    M_warm._gpu_warmup(a.thermal_ms)

    def event_time(callfn, *, flush, reps):
        ss = [di.Event(enable_timing=True) for _ in range(reps)]
        ee = [di.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            if flush:
                _clear()
            ss[i].record()
            callfn()
            ee[i].record()
        di.synchronize()
        ts = [s.elapsed_time(e) for s, e in zip(ss, ee, strict=True)]
        return statistics.median(ts), min(ts)

    def bench_arm(callfn):
        rec = {}
        # warm up launch path
        for _ in range(20):
            callfn()
        di.synchronize()
        # (1) non-graph event-timed cold — bridge to stored cobench (carries the confound)
        ev_med, ev_min = event_time(callfn, flush=True, reps=min(a.reps, 800))
        rec["ev_cold_median_ms"], rec["ev_cold_min_ms"] = ev_med, ev_min
        # (2) CUDA graph capture of ONE call
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    callfn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                callfn()
            for _ in range(20):
                g.replay()
            di.synchronize()
            cg_cold_med, cg_cold_min = event_time(g.replay, flush=True, reps=a.reps)
            cg_warm_med, cg_warm_min = event_time(g.replay, flush=False, reps=a.reps)
            rec["cg_cold_median_ms"], rec["cg_cold_min_ms"] = cg_cold_med, cg_cold_min
            rec["cg_warm_median_ms"], rec["cg_warm_min_ms"] = cg_warm_med, cg_warm_min
            rec["cudagraph"] = "ok"
        except Exception as e:
            rec["cudagraph"] = f"failed:{type(e).__name__}:{e}"
        return rec

    results = []
    for arm in spec["arms"]:
        rec = dict(arm)
        try:
            kind = arm["kind"]
            if kind == "helion":
                callfn, codegen = make_helion(arm)
                rec["codegen"] = codegen
                rec["accuracy"] = accuracy(callfn)
                rec.update(bench_arm(callfn))
            elif kind in ref_kinds:
                callfn = make_ref(arm)
                rec["accuracy"] = "PASS"
                rec["codegen"] = None
                rec.update(bench_arm(callfn))
            else:
                rec["status"] = "skip"
                rec["skipped_reason"] = f"kind {kind} (e.g. fa4/quack unavailable)"
                results.append(rec)
                out["results"] = results
                Path(a.out_json).write_text(
                    json.dumps(out, default=str, indent=2) + "\n"
                )
                continue
            for key in ("cg_cold_median_ms", "cg_warm_median_ms", "ev_cold_median_ms"):
                if rec.get(key):
                    rec[key.replace("median_ms", "tflops")] = (
                        flop / (rec[key] * 1e-3)
                    ) / 1e12
            rec.setdefault("status", "ok")
        except BaseException as e:
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()
        results.append(rec)
        out["results"] = results
        Path(a.out_json).write_text(json.dumps(out, default=str, indent=2) + "\n")
        print(
            f"ARM_DONE {arm.get('name'):30s} st={rec.get('status')} "
            f"cg_cold={rec.get('cg_cold_median_ms')} cg_warm={rec.get('cg_warm_median_ms')} "
            f"ev_cold={rec.get('ev_cold_median_ms')} cg={rec.get('cudagraph', '?')[:20]} "
            f"acc={rec.get('accuracy')}",
            flush=True,
        )

    out["results"] = results
    Path(a.out_json).write_text(json.dumps(out, default=str, indent=2) + "\n")
    print("CUDAGRAPH_COBENCH_DONE " + a.out_json, flush=True)


if __name__ == "__main__":
    main()
