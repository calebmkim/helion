"""H100 matmul-family PERF CHARACTERIZATION harness — 3 arms, 2 cold-L2 methods.

MEASUREMENT + REPORTING only (no heuristic edits). Seed under test = the H100
budget-FORMULA `TritonH100MatmulHeuristic` (rank-0 seed on branch matmul-h100-lab,
promote_seed_to_default=False -> seed is the autotuner-seed candidate, extracted via
bind(...).config_spec.compiler_seed_configs[0]).

Two subcommands (both run in ISOLATED subprocesses, foreground, one GPU job at a time):

  prepare  --kernel K --shape S --dtype D --arm ARM --out J
     Build ONE arm, accuracy-gate it vs an fp32 reference, emit status/config/max_abs.
     Run once per arm so a ptxas hang / OOM on the [16,16,16] default arm kills only
     THAT subprocess (killpg from the orchestrator), never the seed/tc timings. Also
     warms Helion's on-disk compile cache so the timing pass is a cache hit.

  time     --kernel K --shape S --dtype D --arms JSON --out J
     Given the arms that PASSED prepare (status ok + their configs), time them with
     BOTH methods INTERLEAVED, R rounds, arms measured ADJACENTLY within a round
     (common-mode thermal/clock drift cancels in the per-round ratio). Retains the
     FULL per-round t_us[] array for every (method,arm). Nothing summarized in place.

M1 = CUDA-graph device time, cold-L2 (graph-diff): time [flush+kernel] graph minus
     [flush] graph, replay-averaged. Canonical.
M2 = triton do_bench, cold-L2 (its own L2 flush). Includes host launch overhead.

The orchestrator (mmperf_run.py) owns the shape loop, killpg timeouts, and JSONL
checkpointing after every (kernel,shape,method).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
# When committed in-tree at <repo>/_lab/matmul-perf-char/, the worktree is two levels up.
# Override with MMPERF_WORKTREE to point at a different Helion checkout.
_DEFAULT_WT = os.path.dirname(os.path.dirname(_HERE))
WORKTREE = os.environ.get("MMPERF_WORKTREE", _DEFAULT_WT)
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

import torch  # noqa: E402

import helion  # noqa: E402

# Silent-wrong-helion guard (the footgun that has burned this project).
assert os.path.realpath(helion.__file__).startswith(
    os.path.realpath(WORKTREE) + os.sep
), f"WRONG HELION: {helion.__file__} not under {WORKTREE}; set MMPERF_WORKTREE / PYTHONPATH"

# --- fairness locks (§4) ---
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

DEV = torch.device("cuda")
DT = {"bf16": torch.bfloat16, "fp16": torch.float16,
      "fp32": torch.float32, "fp8": torch.float8_e4m3fn}
WIDTH_BITS = {"bf16": 16, "fp16": 16, "fp32": 32, "fp8": 8}
STATIC_SHAPES = {"matmul": True, "fp8_gemm": True, "mamba2_chunk_state": True, "bmm": True}


# ----------------------------------------------------------------------------
# device + flush sizing (§0: flush = max(256 MiB, 4 x device L2))
# ----------------------------------------------------------------------------
def device_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    l2 = int(p.L2_cache_size)
    flush = max(256 * 2**20, 4 * l2)
    cc = f"sm{p.major}{p.minor}"  # sm90 (H100), sm100 (B200), sm80 (A100)
    # H100 SXM dense: bf16/fp16 989.5, fp8 1978.9 TFLOP/s. B200: 2250 / 4500.
    is_b200 = p.major == 10
    peak = {"bf16": 2250.0, "fp16": 2250.0, "fp8": 4500.0, "fp32": 1125.0} if is_b200 \
        else {"bf16": 989.5, "fp16": 989.5, "fp8": 1978.9, "fp32": 66.9}
    return {"name": p.name, "cc": cc, "major": p.major, "minor": p.minor,
            "sms": p.multi_processor_count, "l2_bytes": l2,
            "l2_mib": round(l2 / 2**20, 1), "flush_bytes": flush,
            "flush_mib": round(flush / 2**20, 1), "total_mem_gib": round(p.total_memory / 2**30, 1),
            "peak_tflops": peak}


def clocks() -> dict:
    try:
        import subprocess
        q = "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=15).stdout.strip()
        return {"raw": out}
    except Exception as e:
        return {"err": str(e)}


# ----------------------------------------------------------------------------
# kernel fns + inputs (mirrors the proven _lab/matmul-h100/bench.py make_inputs)
# ----------------------------------------------------------------------------
def _kernel_fn(kernel: str):
    if kernel == "matmul":
        from examples.matmul import matmul
        return matmul.fn
    if kernel == "fp8_gemm":
        from examples.fp8_gemm import fp8_gemm
        return fp8_gemm.fn
    if kernel == "mamba2_chunk_state":
        from examples.mamba2_chunk_state import helion_mamba2_chunk_state_kernel
        return helion_mamba2_chunk_state_kernel.fn
    if kernel == "bmm":
        from examples.bmm import bmm
        return bmm.fn
    raise ValueError(kernel)


def flops(kernel: str, shape: list[int]) -> float:
    """2*M*N*K (times batch for bmm/mamba grid). For the headline TFLOP/s."""
    if kernel in ("matmul", "fp8_gemm"):
        m, k, n = shape
        return 2.0 * m * n * k
    if kernel == "bmm":
        b, m, k, n = shape
        return 2.0 * b * m * n * k
    if kernel == "mamba2_chunk_state":
        b, seq, nh, chunk, hd, ds = shape
        nchunks = (seq + chunk - 1) // chunk
        # inner dot [hd,chunk]@[chunk,ds] batched over b*nchunks*nh
        return 2.0 * (b * nchunks * nh) * hd * ds * chunk
    return 0.0


def bytes_moved(kernel: str, shape: list[int], dtype: str) -> float:
    """Rough operand+output bytes for an implied-BW sanity check (cold-L2 => below HBM peak)."""
    isz = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}[dtype]
    osz = 2  # helion matmul/fp8 output is 16-bit (bf16/half)
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


def make_inputs(kernel: str, shape: list[int], dtype: str):
    """(args, ref_fp32, tc_fn|None, meta). tc_fn = the external library yardstick."""
    dt = DT[dtype]
    if kernel == "matmul":
        m, k, n = shape
        x = torch.randn(m, k, device=DEV, dtype=dt)
        y = torch.randn(k, n, device=DEV, dtype=dt)
        ref = torch.matmul(x.float(), y.float())
        _tc = torch.compile(torch.matmul, mode="max-autotune-no-cudagraphs")
        _tc(x, y)
        return (x, y), ref, (lambda: _tc(x, y)), {"mnk": (m, n, k)}

    if kernel == "fp8_gemm":
        m, k, n = shape
        from helion._testing import HALF_DTYPE
        xf = torch.randn(m, k, device=DEV, dtype=torch.float32)
        yf = torch.randn(k, n, device=DEV, dtype=torch.float32)
        x_fp8 = xf.to(torch.float8_e4m3fn)
        y_fp8 = yf.to(torch.float8_e4m3fn).T.contiguous().T  # col-major
        ref = (x_fp8.to(torch.float32)) @ (y_fp8.to(torch.float32))
        sa = torch.tensor(1.0, device=DEV)
        sb = torch.tensor(1.0, device=DEV)
        # production fp8 baseline = _scaled_mm fast_accum=True (cuBLASLt); compile+warm once
        _tc = torch.compile(
            lambda a, b: torch._scaled_mm(a, b, sa, sb, use_fast_accum=True, out_dtype=HALF_DTYPE),
            mode="max-autotune-no-cudagraphs")
        _tc(x_fp8, y_fp8)
        return (x_fp8, y_fp8), ref, (lambda: _tc(x_fp8, y_fp8)), {"mnk": (m, n, k)}

    if kernel == "bmm":
        bsz, m, k, n = shape
        a = torch.randn(bsz, m, k, device=DEV, dtype=dt)
        bmat = torch.randn(bsz, k, n, device=DEV, dtype=dt)
        ref = torch.bmm(a.float(), bmat.float())
        _tc = torch.compile(torch.bmm, mode="max-autotune-no-cudagraphs")
        _tc(a, bmat)
        return (a, bmat), ref, (lambda: _tc(a, bmat)), {"mnk": (m, n, k), "batch": bsz}

    if kernel == "mamba2_chunk_state":
        b, seq, nh, chunk, hd, ds = shape
        ng = 1
        nchunks = (seq + chunk - 1) // chunk
        B = torch.rand(b, seq, ng, ds, device=DEV, dtype=dt)
        x = torch.rand(b, seq, nh, hd, device=DEV, dtype=dt)
        dt_t = torch.rand(b, nh, nchunks, chunk, device=DEV, dtype=dt)
        dA = torch.rand(b, nh, nchunks, chunk, device=DEV, dtype=dt)
        from examples.mamba2_chunk_state import ref_chunk_state
        ref = ref_chunk_state(B, x, dt_t, dA)  # eager ref; mamba's "tc" arm

        # tc arm for mamba = torch.compile of the eager ref (max-autotune picks a Triton kernel)
        _tc = torch.compile(ref_chunk_state, mode="max-autotune-no-cudagraphs")
        _tc(B, x, dt_t, dA)
        return (B, x, dt_t, dA), ref, (lambda: _tc(B, x, dt_t, dA)), \
            {"mnk": (hd, ds, chunk), "grid": b * nchunks * nh}

    raise ValueError(kernel)


def _build(kernel: str, fn, cfg, ss: bool):
    if cfg is None:
        return helion.kernel(fn, static_shapes=ss)
    c = helion.Config.from_dict(cfg) if isinstance(cfg, dict) else cfg
    return helion.kernel(fn, config=c, static_shapes=ss)


def _as_tensor_list(x):
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for t in x if isinstance(t, torch.Tensor)]
    raise TypeError(f"cannot extract tensors from {type(x)}")


def accuracy_ok(out, ref, rtol=2e-2, atol=2e-2, rel_floor=5e-2):
    outs, refs = _as_tensor_list(out), _as_tensor_list(ref)
    if len(outs) != len(refs):
        return False, float("inf")
    worst, ok_all = 0.0, True
    for o, r in zip(outs, refs):
        o32, r32 = o.detach().float(), r.detach().float()
        if o32.shape != r32.shape:
            if o32.numel() == r32.numel():
                o32 = o32.reshape(r32.shape)
            else:
                return False, float("inf")
        if torch.isnan(o32).any() or torch.isinf(o32).any():
            return False, float("inf")
        max_abs = (o32 - r32).abs().max().item()
        worst = max(worst, max_abs)
        denom = max(r32.abs().max().item(), 1e-6)
        ok = bool(torch.allclose(o32, r32, rtol=rtol, atol=atol)) or (max_abs / denom) < rel_floor
        ok_all = ok_all and ok
    return ok_all, worst


# ----------------------------------------------------------------------------
# seed / default / tc resolution
# ----------------------------------------------------------------------------
def probe(kernel, fn, args, ss):
    """Return (seed_cfg, default_cfg, fired, facts, n_seeds)."""
    kp = _build(kernel, fn, None, ss)
    bound = kp.bind(kp.normalize_args(*args))
    spec = bound.env.config_spec
    seeds = [dict(c) for c in spec.compiler_seed_configs]
    default_cfg = dict(spec.default_config())
    facts = [{"lhs_ndim": f.lhs_ndim, "rhs_ndim": f.rhs_ndim,
              "static_m": f.static_m, "static_n": f.static_n, "static_k": f.static_k,
              "lhs_dtype": str(f.lhs_dtype), "rhs_dtype": str(f.rhs_dtype)}
             for f in spec.matmul_facts]
    fired = list(spec.autotuner_heuristics)
    return (seeds[0] if seeds else None), default_cfg, fired, facts, len(seeds)


# ----------------------------------------------------------------------------
# cold-L2 flush buffer (shared by M1 graphs)
# ----------------------------------------------------------------------------
def make_flush_buf():
    n = device_info()["flush_bytes"]
    return torch.empty(int(n), dtype=torch.int8, device=DEV)


# ----------------------------------------------------------------------------
# M1: CUDA-graph device time, cold-L2 (graph-diff)
# ----------------------------------------------------------------------------
def capture_graph(thunk, flush_buf, warmup=5):
    """Capture a [flush + thunk()] graph. Warmup on a side stream first (required)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            flush_buf.zero_()
            thunk()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        flush_buf.zero_()
        out = thunk()
    torch.cuda.synchronize()
    return g, out


def capture_flush(flush_buf, warmup=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            flush_buf.zero_()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        flush_buf.zero_()
    torch.cuda.synchronize()
    return g


def time_graph_ms(g, inner: int) -> float:
    """Replay g `inner` times, return total elapsed ms (device time)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(inner):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def pick_inner(g_full, g_flush, target_ms=30.0, cap=200) -> int:
    """Adaptive replay count: total ~target_ms so short kernels get many replays,
    slow [16,16,16] defaults get few. Based on one timed full replay."""
    t1 = time_graph_ms(g_full, 1)
    if t1 <= 0:
        return 10
    inner = int(max(3, min(cap, math.ceil(target_ms / t1))))
    return inner


# ----------------------------------------------------------------------------
# M2: triton do_bench, cold-L2
# ----------------------------------------------------------------------------
def do_bench_ms(thunk) -> float:
    import triton.testing
    return triton.testing.do_bench(thunk, warmup=25, rep=100, return_mode="median")


# ============================================================================
# subcommand: prepare (one arm, isolated -> ptxas-hang containment)
# ============================================================================
def cmd_prepare(a):
    shape = [int(s) for s in a.shape.split(",")]
    ss = STATIC_SHAPES[a.kernel]
    fn = _kernel_fn(a.kernel)
    info = {"kernel": a.kernel, "shape": shape, "dtype": a.dtype, "arm": a.arm,
            "helion_file": helion.__file__}
    try:
        args, ref, tc_fn, meta = make_inputs(a.kernel, shape, a.dtype)
        info["meta"] = meta
        seed_cfg, default_cfg, fired, facts, n_seeds = probe(a.kernel, fn, args, ss)
        info.update({"seed_fired": bool(fired), "heuristics_fired": fired,
                     "matmul_facts": facts, "n_seeds": n_seeds,
                     "seed_cfg": seed_cfg, "default_cfg": default_cfg})
        if a.arm == "tc_max_autotune":
            out = tc_fn() if tc_fn is not None else None
            if out is None:
                info.update({"status": "skip", "error": "no tc arm (mamba handled via compiled ref)"})
            else:
                ok, ma = accuracy_ok(out, ref)
                info.update({"status": "ok" if ok else "acc_fail", "max_abs": ma, "acc_pass": ok})
        else:
            cfg = seed_cfg if a.arm == "seed" else default_cfg
            info["config"] = cfg
            if cfg is None:
                info.update({"status": "compile_fail", "error": "no config (seed empty)"})
            else:
                k = _build(a.kernel, fn, cfg, ss)
                out = k(*args)
                ok, ma = accuracy_ok(out, ref)
                info.update({"status": "ok" if ok else "acc_fail", "max_abs": ma, "acc_pass": ok})
    except torch.cuda.OutOfMemoryError as e:
        info.update({"status": "oom", "error": f"{type(e).__name__}: {str(e)[:300]}"})
    except Exception as e:
        info.update({"status": "compile_fail", "error": f"{type(e).__name__}: {str(e)[:300]}",
                     "traceback": traceback.format_exc()[-1500:]})
    with open(a.out, "w") as fh:
        json.dump(info, fh, default=str)
    print("PREPARE " + json.dumps({"arm": a.arm, "status": info.get("status"),
                                   "acc": info.get("acc_pass"), "ma": info.get("max_abs")}, default=str))


# ============================================================================
# subcommand: time (ok arms, M1+M2 interleaved, R rounds, RAW arrays)
# ============================================================================
def cmd_time(a):
    shape = [int(s) for s in a.shape.split(",")]
    ss = STATIC_SHAPES[a.kernel]
    fn = _kernel_fn(a.kernel)
    arms_in = json.loads(a.arms)  # {arm: {"status":..,"config":..}}  (only ok arms passed)
    R = a.rounds
    di = device_info()
    info = {"kernel": a.kernel, "shape": shape, "dtype": a.dtype, "device": di["cc"],
            "l2_mib": di["l2_mib"], "flush_mib": di["flush_mib"], "R": R,
            "helion_file": helion.__file__, "default_source": "n_a"}
    out = {"M1_cudagraph_coldL2": {}, "M2_do_bench_coldL2": {}}
    try:
        args, ref, tc_fn, meta = make_inputs(a.kernel, shape, a.dtype)
        info["meta"] = meta
        flush_buf = make_flush_buf()

        # Build thunks for each ok arm (helion arms rebuild from cached compile => fast).
        thunks = {}
        for arm, spec in arms_in.items():
            if spec.get("status") != "ok":
                continue
            if arm == "tc_max_autotune":
                thunks[arm] = tc_fn
            else:
                k = _build(a.kernel, fn, spec["config"], ss)
                thunks[arm] = (lambda kk: (lambda: kk(*args)))(k)

        arm_order = [x for x in ["seed", "helion_default", "tc_max_autotune"] if x in thunks]

        # ---- M1: capture one graph per arm + one shared flush graph ----
        m1 = {arm: {"status": "ok", "t_us": []} for arm in arm_order}
        graphs, inner = {}, {}
        g_flush = None
        try:
            g_flush = capture_flush(flush_buf)
        except Exception as e:
            info["m1_flush_capture_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        for arm in arm_order:
            if g_flush is None:
                m1[arm] = {"status": "graph_fail", "t_us": [], "error": "flush capture failed"}
                continue
            try:
                g, _ = capture_graph(thunks[arm], flush_buf)
                graphs[arm] = g
                inner[arm] = pick_inner(g, g_flush)
            except Exception as e:
                m1[arm] = {"status": "graph_fail", "t_us": [],
                           "error": f"{type(e).__name__}: {str(e)[:200]}"}

        # Adaptive R: self-calibrate on the SEED kernel's cold-L2 device time. Sub-25us
        # kernels swing (noise floor) -> bump rounds to R_small. Recorded so it's transparent.
        R_eff = R
        if "seed" in graphs and g_flush is not None:
            try:
                probe_inner = inner["seed"]
                t_full = time_graph_ms(graphs["seed"], probe_inner)
                t_flush = time_graph_ms(g_flush, probe_inner)
                seed_est_us = max(0.0, (t_full - t_flush) / probe_inner * 1000.0)
                info["seed_est_us"] = round(seed_est_us, 3)
                if seed_est_us < a.small_us:
                    R_eff = a.small_rounds
            except Exception:
                pass
        info["R_eff"] = R_eff
        R = R_eff
        # R rounds; per round time shared flush once + each arm's full graph (adjacent)
        for _r in range(R):
            if g_flush is not None:
                # a per-round flush timing, averaged over max inner used this round
                pass
            for arm in arm_order:
                if arm not in graphs:
                    continue
                ii = inner[arm]
                t_full = time_graph_ms(graphs[arm], ii)
                t_flush = time_graph_ms(g_flush, ii)
                t_us = max(0.0, (t_full - t_flush) / ii * 1000.0)
                m1[arm]["t_us"].append(round(t_us, 4))
        for arm in arm_order:
            m1[arm]["inner"] = inner.get(arm)
        out["M1_cudagraph_coldL2"] = m1
        # free graphs before M2
        del graphs
        if g_flush is not None:
            del g_flush
        torch.cuda.synchronize()

        # ---- M2: do_bench per arm, R rounds, adjacent ----
        m2 = {arm: {"status": "ok", "t_us": []} for arm in arm_order}
        for _r in range(R):
            for arm in arm_order:
                try:
                    ms = do_bench_ms(thunks[arm])
                    m2[arm]["t_us"].append(round(ms * 1000.0, 4))
                except Exception as e:
                    m2[arm]["status"] = "bench_fail"
                    m2[arm]["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["M2_do_bench_coldL2"] = m2

        info["ok"] = True
        info["arms_timed"] = arm_order
    except Exception as e:
        info.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                     "traceback": traceback.format_exc()[-1500:]})
    info["methods"] = out
    with open(a.out, "w") as fh:
        json.dump(info, fh, default=str)
    print("TIME " + json.dumps({"ok": info.get("ok"), "arms": info.get("arms_timed"),
                                 "err": info.get("error")}, default=str))


# ============================================================================
# subcommand: backend — capture the torch.compile(max-autotune) SELECTED kernel
# (§4/§5(d)/§7: log the tc_max_autotune winner per shape). Parses the inductor
# select_algorithm autotune log: the choice printed at 100.0% is the winner. A name
# of `mm`/`bmm`/`addmm`/`_scaled_mm` => the aten/cuBLAS(-Lt) path won; `triton_*` =>
# a Triton template won.
# ============================================================================
def cmd_backend(a):
    # The inductor autotune table is written directly to sys.stderr (not logging) by
    # AlgorithmSelectorCache.log_results, gated on max_autotune + PRINT_AUTOTUNE, and
    # printed ONCE per compile (cached after). So: fresh inductor cache + capture stderr.
    import contextlib
    import io
    import re
    shape = [int(s) for s in a.shape.split(",")]
    info = {"kernel": a.kernel, "shape": shape, "dtype": a.dtype, "helion_file": helion.__file__}
    buf = io.StringIO()
    try:
        if a.kernel == "mamba2_chunk_state":
            pass  # mamba HAS a tc arm (compiled eager ref) -> we DO capture it
        # First compile in this fresh process autotunes and writes the AUTOTUNE table to
        # sys.stderr -> capture it. No prior warm (we never call make_inputs here).
        with contextlib.redirect_stderr(buf):
            _recompile_tc(a.kernel, shape, a.dtype, buf)
        log = buf.getvalue()
        if True:
            winners = []
            cur_op = None
            for line in log.splitlines():
                m = re.search(r"AUTOTUNE\s+(\w+)\(", line)
                if m:
                    cur_op = m.group(1)
                    continue
                mw = re.match(r"\s+(\S+)\s+[\d.]+\s+ms\s+100\.0%", line)
                if mw:
                    winners.append((cur_op, mw.group(1).strip()))
            info["autotune_ops"] = winners
            names = [w[1] for w in winners]

            def classify(nm):
                return "Triton" if nm.startswith("triton") else "aten/cuBLAS"
            if names:
                info["winner"] = "+".join(dict.fromkeys(names))
                info["winner_kind"] = "+".join(sorted({classify(n) for n in names}))
            else:
                info["winner"] = "unknown (no 100.0% line parsed)"
            info["raw"] = log[-3000:]
        info["ok"] = True
    except Exception as e:
        info.update({"ok": False, "winner": "error",
                     "error": f"{type(e).__name__}: {str(e)[:300]}",
                     "traceback": traceback.format_exc()[-800:]})
    with open(a.out, "w") as fh:
        json.dump(info, fh, default=str)
    print("BACKEND " + json.dumps({"winner": info.get("winner"),
                                   "kind": info.get("winner_kind")}, default=str))


def _recompile_tc(kernel, shape, dtype, buf):
    """Rebuild ONLY the tc callable with a fresh compile so the AUTOTUNE table prints
    into the redirected stderr. Uses the same op wiring as make_inputs' tc arm.
    Requires: a FRESH inductor cache (else cache-hit skips autotune -> no print) and
    autotune_in_subproc=False (else the table goes to a subprocess's stderr). The
    orchestrator sets TORCHINDUCTOR_CACHE_DIR per probe; we force subproc off here."""
    dt = DT[dtype]
    import torch._inductor.config as ic
    ic.max_autotune = True
    ic.autotune_in_subproc = False
    if kernel == "matmul":
        m, k, n = shape
        x = torch.randn(m, k, device=DEV, dtype=dt); y = torch.randn(k, n, device=DEV, dtype=dt)
        f = torch.compile(torch.matmul, mode="max-autotune-no-cudagraphs")
        f(x, y)
    elif kernel == "fp8_gemm":
        m, k, n = shape
        from helion._testing import HALF_DTYPE
        xf = torch.randn(m, k, device=DEV, dtype=torch.float32)
        yf = torch.randn(k, n, device=DEV, dtype=torch.float32)
        x8 = xf.to(torch.float8_e4m3fn); y8 = yf.to(torch.float8_e4m3fn).T.contiguous().T
        sa = torch.tensor(1.0, device=DEV); sb = torch.tensor(1.0, device=DEV)
        f = torch.compile(lambda a2, b2: torch._scaled_mm(a2, b2, sa, sb, use_fast_accum=True,
                                                          out_dtype=HALF_DTYPE),
                          mode="max-autotune-no-cudagraphs")
        f(x8, y8)
    elif kernel == "bmm":
        bsz, m, k, n = shape
        a2 = torch.randn(bsz, m, k, device=DEV, dtype=dt); b2 = torch.randn(bsz, k, n, device=DEV, dtype=dt)
        f = torch.compile(torch.bmm, mode="max-autotune-no-cudagraphs")
        f(a2, b2)
    elif kernel == "mamba2_chunk_state":
        b, seq, nh, chunk, hd, ds = shape
        ng = 1; nchunks = (seq + chunk - 1) // chunk
        B = torch.rand(b, seq, ng, ds, device=DEV, dtype=dt)
        x = torch.rand(b, seq, nh, hd, device=DEV, dtype=dt)
        dt_t = torch.rand(b, nh, nchunks, chunk, device=DEV, dtype=dt)
        dA = torch.rand(b, nh, nchunks, chunk, device=DEV, dtype=dt)
        from examples.mamba2_chunk_state import ref_chunk_state
        f = torch.compile(ref_chunk_state, mode="max-autotune-no-cudagraphs")
        f(B, x, dt_t, dA)
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    bp = sub.add_parser("backend")
    bp.add_argument("--kernel", required=True)
    bp.add_argument("--shape", required=True)
    bp.add_argument("--dtype", required=True)
    bp.add_argument("--out", required=True)
    bp.set_defaults(func=cmd_backend)
    pp = sub.add_parser("prepare")
    pp.add_argument("--kernel", required=True)
    pp.add_argument("--shape", required=True)
    pp.add_argument("--dtype", required=True)
    pp.add_argument("--arm", required=True, choices=["seed", "helion_default", "tc_max_autotune"])
    pp.add_argument("--out", required=True)
    pp.set_defaults(func=cmd_prepare)
    tp = sub.add_parser("time")
    tp.add_argument("--kernel", required=True)
    tp.add_argument("--shape", required=True)
    tp.add_argument("--dtype", required=True)
    tp.add_argument("--arms", required=True, help="JSON {arm:{status,config}}")
    tp.add_argument("--rounds", type=int, default=7)
    tp.add_argument("--small-us", type=float, default=25.0)
    tp.add_argument("--small-rounds", type=int, default=15)
    tp.add_argument("--out", required=True)
    tp.set_defaults(func=cmd_time)
    ip = sub.add_parser("devinfo")
    ip.set_defaults(func=lambda a: print(json.dumps({**device_info(), "clocks": clocks()}, indent=2)))
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
