"""Self-contained perf-report harness for the reduction-seed heuristic (PR #2996).

Reproduces the /examples + vLLM headline tables. Per (corpus, kernel, shape, dtype) it
times up to 4 arms IN ONE PROCESS on the SAME input tensors, by explicit config replay
(the env flags are process-global, so we extract+replay configs rather than flipping flags
mid-process):

  seed         = helion.kernel(fn.fn, config=compiler_seed_configs(...)[0])   [the heuristic]
  default      = helion.kernel(fn.fn, config=spec._base_default_config())     [unseeded base]
                 NB: NOT default_config() — a promote_seed_to_default heuristic would make
                 default_config() return the SEED. _base_default_config() is the true base.
  tc           = torch.compile(reference)  DEFAULT mode (real corpora only)
  vllm_shipped = vLLM's nvidia_h100.json config, nearest-shape lookup (vLLM corpus only)

TIMING (the part vetted with the manager):
  * INTERLEAVED round-robin across arms (Helion's own interleaved_bench mechanism): each
    round clears L2, times seed, clears L2, times default, ... so slow drift (thermal/clock)
    hits every arm within microseconds and CANCELS in the ratio. This is the gold standard
    for A/B ratio claims and is literally the primitive Helion's autotuner rebench uses.
  * COLD-L2: a 256MB memset zeroes L2 before every timed call (deployment regime; also the
    regime the autotuner selects in). Verified: with the flush, cold-L2 do_bench == cold-L2
    cudagraph replay, i.e. CPU launch overhead (~40us for Helion) is HIDDEN behind the ~86us
    memset — so these numbers are pure GPU device time. See notes/LAUNCH_OVERHEAD_NOTE.md.
  * COLDGRAPH CROSS-CHECK: per cell we also capture cold-L2 cudagraph device time and the
    aggregator flags any arm where do_bench and coldgraph diverge (the finite-margin canary).
  * median-of-9 interleaved rounds, escalate to 15 on >5% spread. Reps INSIDE each round are
    scaled to a ~25ms wall-clock budget (fast kernels -> many reps, slow -> few, clamped
    [5,500]) like do_bench / the autotuner; set PERF_REPS_PER_ROUND to force a fixed count.
  * NO CUDA graphs in the headline number; forward only; requires_grad=False; dynamo-reset
    per shape; fresh process per kernel via the CLI.

Raw data (Level 2): every cell records, per arm, {us, coldgraph_us, per-round samples,
spread, config, fired-heuristic, acc pass/fail + maxabs}. Ratios are DERIVED from this by
aggregate_report.py — never hand-entered.

Run (fresh process per kernel):
  HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=<this-worktree> \
    /home/dev/helion/.venv/bin/python perf-repro/perf_report_bench.py \
    --corpus curriculum --kernel rms_norm --out-dir perf-repro/results
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback

# --------------------------------------------------------------------------- #
#  Path wiring — SELF-CONTAINED. Everything the harness needs is either in the
#  helion worktree (examples/, helion/) or vendored under perf-repro/deps/.
#  The ONLY external reference is vLLM's tuned-config JSON (read live via
#  VLLM_CONFIG_DIR inside deps/bench_arms.py — it belongs to vLLM, not us).
# --------------------------------------------------------------------------- #
_THIS = os.path.abspath(__file__)
_PERF_DIR = os.path.dirname(_THIS)  # .../perf-repro
_WT_ROOT = os.path.abspath(os.path.join(_PERF_DIR, ".."))  # helion worktree root
_DEPS = os.path.join(_PERF_DIR, "deps")

for _d in (
    _WT_ROOT,  # `import helion`, `import examples.*`
    os.path.join(_WT_ROOT, "examples"),
    _DEPS,  # vendored builders: ab_three_arm_transfer, mreduction_styles_view_only, bench_arms, refs
    os.path.join(_DEPS, "kut"),  # vLLM kernel sources imported as `kut.*` and bench_arms' `import refs`
):
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402
from triton import runtime as _tr  # noqa: E402  (L2-clear + device interface)
from triton.testing import do_bench  # noqa: E402

import helion  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402  (kept for API symmetry)
# `helion.runtime.kernel` re-exports the kernel() FUNCTION, shadowing the submodule; import the
# Kernel CLASS via importlib to dodge the shadow.
_KRT = importlib.import_module("helion.runtime.kernel")
_HelionKernel = _KRT.Kernel

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT_ROOT}); set PYTHONPATH."
)

EPS = 1e-5
DEV = "cuda"
_DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# median-of-N interleaved rounds; escalate on noisy cells.
_ROUNDS = int(os.environ.get("PERF_ROUNDS", "9"))
_ROUNDS_HI = int(os.environ.get("PERF_ROUNDS_HI", "15"))
_SPREAD_GATE = float(os.environ.get("PERF_SPREAD_GATE", "0.05"))
# Reps INSIDE one interleaved round (each rep = one cold-L2 timed launch per arm). By default
# the count is SCALED to a per-round wall-clock budget (like triton do_bench and Helion's
# autotuner _repeat_for_target_ms): fast kernels get MANY reps (tight median where per-launch
# jitter is worst), slow kernels get FEW (kernel dominates noise; over-sampling just wastes time),
# clamped to [floor, cap]. Set PERF_REPS_PER_ROUND to a fixed int to force a constant rep count.
# Budget matches triton do_bench / tritonbench defaults: rep=100ms target, warmup=25ms.
_REPS_FIXED = os.environ.get("PERF_REPS_PER_ROUND")  # None => budget-scaled
_ROUND_BUDGET_MS = float(os.environ.get("PERF_ROUND_BUDGET_MS", "100"))  # tritonbench do_bench rep=100ms
_WARMUP_MS = float(os.environ.get("PERF_WARMUP_MS", "25"))  # tritonbench do_bench warmup=25ms
_REPS_FLOOR = int(os.environ.get("PERF_REPS_FLOOR", "5"))
_REPS_CAP = int(os.environ.get("PERF_REPS_CAP", "1000"))
# do we also capture cold-L2 cudagraph device time per arm (the launch-overhead cross-check)?
_COLDGRAPH = os.environ.get("PERF_COLDGRAPH", "1") == "1"


# --------------------------------------------------------------------------- #
#  Timing primitives.
#
#  We reimplement Helion's interleaved_bench loop here (verbatim mechanism) rather
#  than call it, for two reasons: (1) our arms are plain thunks, not Helion
#  CompiledConfig objects; (2) we need the PER-ROUND samples for the Level-2 raw
#  data, which interleaved_bench medians away. The mechanism — warmup each arm
#  once, then for each round clear L2 and event-bracket each arm in turn — is
#  identical to helion/autotuner/benchmarking.py::interleaved_bench.
# --------------------------------------------------------------------------- #
_DI = _tr.driver.active.get_device_interface()


def _l2_clearer():
    """A callable that zeroes a 256MB scratch buffer -> evicts L2 (cold-L2). Same primitive
    triton do_bench and helion interleaved_bench use."""
    cache = _tr.driver.active.get_empty_cache_for_benchmark()
    return lambda: _tr.driver.active.clear_cache(cache)


def interleaved_round(fns: list, reps: int, clear_l2) -> list[list[float]]:
    """One interleaved measurement pass. Returns per-fn list of `reps` per-launch times (us).

    Round-robin: for each rep index i, and each fn j, clear L2 then event-bracket fn[j]().
    Timing every arm microseconds apart within a rep makes slow drift common-mode -> it
    cancels in the ratio. Events are recorded on the default stream (where these kernels
    launch), so the start->end bracket encloses exactly the device work (the pre-flush
    memset having given the CPU time to enqueue the launch)."""
    n = len(fns)
    starts = [[_DI.Event(enable_timing=True) for _ in range(reps)] for _ in range(n)]
    ends = [[_DI.Event(enable_timing=True) for _ in range(reps)] for _ in range(n)]
    _DI.synchronize()
    for i in range(reps):
        for j in range(n):
            clear_l2()
            starts[j][i].record()
            fns[j]()
            ends[j][i].record()
    _DI.synchronize()
    # ms -> us
    return [
        [starts[j][i].elapsed_time(ends[j][i]) * 1e3 for i in range(reps)]
        for j in range(n)
    ]


def _round_median_us(per_launch_times_us: list[float]) -> float:
    return statistics.median(per_launch_times_us)


def _estimate_reps(fns: list, clear_l2) -> int:
    """Choose reps/round so one round ~= _ROUND_BUDGET_MS of wall clock, clamped to
    [_REPS_FLOOR, _REPS_CAP]. Mirrors do_bench / autotuner rep-scaling: probe the cost of
    ONE full interleaved rep (all arms + their L2 flushes, the real unit of a round), then
    reps = budget / probe. Fixed count if PERF_REPS_PER_ROUND is set. Interleaved-specific:
    we base it on the WHOLE-round cost so every arm stays lock-step within the budget."""
    if _REPS_FIXED is not None:
        return int(_REPS_FIXED)
    # time a few full interleaved reps (flush + each arm) to get the per-rep wall cost.
    probe = 5
    _DI.synchronize()
    t0 = time.perf_counter()
    for _ in range(probe):
        for f in fns:
            clear_l2()
            f()
    _DI.synchronize()
    per_rep_ms = (time.perf_counter() - t0) * 1e3 / probe
    if per_rep_ms <= 0:
        return _REPS_CAP
    reps = int(_ROUND_BUDGET_MS / per_rep_ms)
    return max(_REPS_FLOOR, min(_REPS_CAP, reps))


def timed_interleaved(named_fns: dict) -> dict:
    """Interleaved median-of-9 (->15 on >5% spread) for a dict {arm_name: thunk}.

    Returns {arm_name: {us, spread, round_medians:[...]}}. All arms warmed up once each
    before any timing, then measured together round-by-round so drift is common-mode."""
    names = list(named_fns.keys())
    fns = [named_fns[k] for k in names]
    clear_l2 = _l2_clearer()

    # warmup to ~_WARMUP_MS per arm (ramps clocks, fills icache) — matches do_bench's 25ms warmup.
    # One priming call each, time it, then repeat to fill the warmup budget.
    for f in fns:
        f()
    _DI.synchronize()
    t0 = time.perf_counter()
    for f in fns:
        f()
    _DI.synchronize()
    one_pass_ms = (time.perf_counter() - t0) * 1e3
    n_warm = max(1, int(_WARMUP_MS / one_pass_ms)) if one_pass_ms > 0 else 1
    for _ in range(n_warm):
        for f in fns:
            f()
    _DI.synchronize()

    # reps/round: budget-scaled (fast->many, slow->few) unless PERF_REPS_PER_ROUND forces fixed.
    reps = _estimate_reps(fns, clear_l2)

    def collect(rounds):
        # per arm: a list of `rounds` round-medians (each round-median over `reps` launches)
        acc = {k: [] for k in names}
        for _r in range(rounds):
            per_arm = interleaved_round(fns, reps, clear_l2)
            for k, times in zip(names, per_arm):
                acc[k].append(_round_median_us(times))
        return acc

    acc = collect(_ROUNDS)
    # spread across the round-medians; escalate if any arm is noisy.
    def spread_of(vals):
        s = sorted(vals)
        med = s[len(s) // 2]
        return (s[-1] - s[0]) / med if med > 0 else 0.0

    if any(spread_of(v) > _SPREAD_GATE for v in acc.values()):
        acc = collect(_ROUNDS_HI)

    out = {}
    for k in names:
        vals = acc[k]
        med = statistics.median(vals)
        out[k] = {
            "us": round(med, 3),
            "spread": round(spread_of(vals), 4),
            "round_medians": [round(v, 3) for v in vals],
            "reps_per_round": reps,
        }
    return out


def coldgraph_us(fn) -> float | None:
    """Cold-L2 pure-GPU device time via CUDA-graph replay (the launch-overhead cross-check).

    Captures ONE fn() into a graph, then times budget-scaled replays each preceded by an L2
    clear, event-bracketed, one sync. Reps are scaled to the SAME per-round budget as the main
    measurement (via a probe of one L2-clear+replay), so a fast kernel gets many replays and a
    slow one few — apples-to-apples with the interleaved number. A graph replay is a single
    stream-ordered submission with no per-call Python dispatch, so this is pure device time at
    cold-L2. If interleaved do_bench-style ~= this, launch overhead is hidden and the headline
    is GPU-side truth. Returns None if capture fails (some kernels can't be captured)."""
    if not _COLDGRAPH:
        return None
    clear_l2 = _l2_clearer()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(3):
            g.replay()
        _DI.synchronize()
        # scale reps to the round budget (probe one clear+replay), clamped like the main path.
        if _REPS_FIXED is not None:
            reps = int(_REPS_FIXED)
        else:
            _DI.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                clear_l2(); g.replay()
            _DI.synchronize()
            per_ms = (time.perf_counter() - t0) * 1e3 / 5
            reps = _REPS_CAP if per_ms <= 0 else max(_REPS_FLOOR, min(_REPS_CAP, int(_ROUND_BUDGET_MS / per_ms)))
        st = [_DI.Event(enable_timing=True) for _ in range(reps)]
        en = [_DI.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            clear_l2()
            st[i].record()
            g.replay()
            en[i].record()
        _DI.synchronize()
        return round(statistics.median([st[i].elapsed_time(en[i]) * 1e3 for i in range(reps)]), 3)
    except Exception:  # noqa: BLE001  (capture failure -> skip cross-check for this cell)
        return None


def _geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


# --------------------------------------------------------------------------- #
#  Compile watchdog: some backward kernels drive ptxas into a pathological
#  multi-minute register-allocation on their largest shape. Bound each arm's
#  compile+first-call with a wall-clock alarm; on timeout, reap the orphaned
#  ptxas/compile-worker children and mark the cell compile-fail-timeout.
# --------------------------------------------------------------------------- #
COMPILE_TIMEOUT_S = int(os.environ.get("PERF_COMPILE_TIMEOUT_S", "150"))


class _CompileTimeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise _CompileTimeout()


def _reap_compile_children():
    """Kill any ptxas / inductor compile-worker children of THIS process. Best-effort."""
    try:
        me = os.getpid()
        out = subprocess.run(["ps", "-eo", "pid,ppid,cmd"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return
    for line in out.splitlines():
        p = line.split(None, 2)
        if len(p) < 3 or not p[0].isdigit():
            continue
        pid, cmd = int(p[0]), p[2]
        if pid == me:
            continue
        if ("ptxas" in cmd or "compile_worker" in cmd) and str(me) in line:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass


def with_compile_timeout(fn, seconds=COMPILE_TIMEOUT_S):
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# --------------------------------------------------------------------------- #
#  Config extraction (seed + unseeded base), faithful replay.
# --------------------------------------------------------------------------- #
def _extract_configs(kfn, args) -> tuple[object, object, list]:
    """Return (seed_cfg_or_None, base_default_cfg, fired_names). Bind once.

    NB: kfn.reset() clears the in-memory _bound_kernels cache BEFORE binding. Required for
    correctness on static_shapes=False kernels (e.g. per_token_group_fp8_quant): their bind
    cache key buckets tensor dims to a constant and does not hl.specialize num_tokens, so two
    shapes with the same (groups_per_row, hidden, group) but different num_tokens collide on one
    cache entry. Without reset, a later shape inherits an earlier shape's compiler_seed_configs
    (e.g. tok=1's block_sizes=[1]) — the wrong, ~4x-slower seed. reset() forces a fresh bind per
    cell so each shape gets its own num_tokens-dependent seed. Harmless for clean kernels."""
    kfn.reset()
    bound = kfn.bind(args)
    spec = bound.env.config_spec
    seeds = list(spec.compiler_seed_configs)  # persisted during bind (inside env ctx)
    fired = list(spec.autotuner_heuristics)
    with bound.env:
        base_default = spec._base_default_config()
    seed = seeds[0] if seeds else None
    del bound
    return seed, base_default, fired


def _replay(kfn, cfg):
    """Rewrap the kernel fn at a fixed config, preserving authored settings."""
    if cfg is None:
        return None
    s = kfn.settings
    return helion.kernel(
        kfn.fn,
        config=cfg,
        static_shapes=s.static_shapes,
        ignore_warnings=list(s.ignore_warnings or []),
    )


def _cfg_dict(cfg) -> dict | None:
    if cfg is None:
        return None
    try:
        return dict(cfg.config)
    except Exception:  # noqa: BLE001
        return {"repr": repr(cfg)}


# --------------------------------------------------------------------------- #
#  Accuracy helpers.
# --------------------------------------------------------------------------- #
def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def _acc_close(rtol, atol, extract=_first):
    def acc(out, ref):
        o = extract(out).to(torch.float32)
        r = (_first(ref) if isinstance(ref, (tuple, list)) else ref).to(torch.float32)
        ok = bool(torch.allclose(o, r, rtol=rtol, atol=atol))
        maxabs = float((o - r).abs().max())
        return ok, f"maxabs={maxabs:.3e}"
    return acc


def _acc_tuple(rtol, atol):
    """Accuracy over a tuple of gradient tensors (compare all present)."""
    def acc(out, ref):
        outs = out if isinstance(out, (tuple, list)) else (out,)
        refs = ref if isinstance(ref, (tuple, list)) else (ref,)
        worst = 0.0
        ok_all = True
        for o, r in zip(outs, refs):
            of, rf = o.detach().to(torch.float32), r.detach().to(torch.float32)
            ok = bool(torch.allclose(of, rf, rtol=rtol, atol=atol))
            ok_all = ok_all and ok
            worst = max(worst, float((of - rf).abs().max()))
        return ok_all, f"maxabs={worst:.3e}"
    return acc


# ---- POINTWISE (PR #2866): flat family + 3 lever kernels --------------------
def _pointwise_build(kernel, shape, dt):
    import pointwise_builders as PB

    kfn, args, ref_out, rtol, atol, tc_ref, acc_kind = PB.build(kernel, shape, dt)
    acc = _acc_tuple(rtol, atol) if acc_kind == "tuple" else _acc_close(rtol, atol)
    return kfn, args, ref_out, acc, tc_ref


# ---- CURRICULUM (9 headline example kernels) --------------------------------
def _cur_build(kernel, m, n, dt):
    import examples.rms_norm as RN
    import examples.layer_norm as LN
    import examples.softmax as SM
    import examples.welford as WF
    import examples.cross_entropy as CE
    import examples.kl_div as KL
    import examples.jsd as JSD
    import examples.sum as SUM
    import examples.long_sum as LS

    def rn(*s):
        return torch.randn(*s, device=DEV, dtype=dt)

    tol = (2e-2, 2e-2) if dt != torch.float32 else (1e-3, 1e-4)
    if kernel == "rms_norm":
        x, w = rn(m, n), rn(n)
        args = (x, w, EPS)
        ref = RN.rms_norm_pytorch(x, w, EPS)
        return RN.rms_norm_fwd, args, ref, _acc_close(*tol), lambda: RN.rms_norm_pytorch(x, w, EPS)
    if kernel == "layer_norm":
        x, w, b = rn(m, n), rn(n), rn(n)
        args = (x, [n], w, b, EPS)
        ref = torch.nn.functional.layer_norm(x, [n], w, b, EPS)
        return (LN.layer_norm_fwd, args, ref, _acc_close(*tol),
                lambda: torch.nn.functional.layer_norm(x, [n], w, b, EPS))
    if kernel == "welford":
        w, b, x = torch.rand(n, device=DEV, dtype=dt), torch.rand(n, device=DEV, dtype=dt), torch.rand(m, n, device=DEV, dtype=dt)
        args = (w, b, x, EPS)
        ref = WF.eager_layer_norm(*args)
        return WF.welford, args, ref, _acc_close(*tol), lambda: WF.eager_layer_norm(*args)
    if kernel == "softmax":
        x = rn(m, n)
        args = (x,)
        ref = torch.nn.functional.softmax(x, dim=1)
        return SM.softmax_two_pass, args, ref, _acc_close(*tol), lambda: torch.nn.functional.softmax(x, dim=1)
    if kernel == "cross_entropy":
        lg = rn(m, n)
        lb = torch.randint(0, n, (m,), device=DEV, dtype=torch.int64)
        args = (lg, lb)
        ref = torch.nn.functional.cross_entropy(lg, lb)
        ctol = (2e-2, 2e-2) if dt != torch.float32 else (1e-3, 1e-3)
        return CE.cross_entropy, args, ref, _acc_close(*ctol), lambda: torch.nn.functional.cross_entropy(lg, lb)
    if kernel == "kl_div":
        yp = rn(m, n).log_softmax(-1)
        yt = rn(m, n).softmax(-1)
        args = (yp, yt, False, "batchmean", 1e-10)
        ref = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(DEV)(yp, yt)
        return (KL.kl_div_forward, args, ref, _acc_close(2e-2, 2e-2),
                lambda: torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(DEV)(yp, yt))
    if kernel == "jsd":
        # jsd_forward returns (loss, dX); compare loss only. tc computes loss only -> Helion times
        # an EXTRA dX output tc doesn't -> G_tc is CONSERVATIVE (biased against seed).
        lq = rn(m, n).log_softmax(-1)
        lp = rn(m, n).log_softmax(-1)
        args = (lq, lp, None, 0.5, -100)
        baseline = JSD.TorchJSDBaseline(beta=0.5, ignore_index=-100)
        ref = baseline(lq, lp)
        return (JSD.jsd_forward, args, ref, _acc_close(2e-2, 2e-2), lambda: baseline(lq, lp))
    if kernel == "sum":
        x = rn(m, n)
        args = (x,)
        ref = torch.sum(x, dim=-1)
        return SUM.sum_kernel, args, ref, _acc_close(*tol), lambda: torch.sum(x, dim=-1)
    if kernel == "long_sum":
        x = rn(m, n)
        args = (x,)
        ref = torch.sum(x, dim=-1)
        return LS.longsum, args, ref, _acc_close(*tol), lambda: torch.sum(x, dim=-1)
    raise KeyError(kernel)


# ---- TRANSFER-hosted HEADLINE kernels (fused_linear_jsd, grpo) ---------------
def _transfer_build(kernel, shape, dt):
    import ab_three_arm_transfer as AB

    build = AB._make(kernel)
    kfn, args, ref_callable, chk = build(tuple(shape), dt)
    ref_out = ref_callable()

    def acc(out, ref):
        try:
            chk(out, ref)
            return True, "ok"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}:{str(e)[:60]}"

    return kfn, args, ref_out, acc, ref_callable


# ---- MREDUCTION HEADLINE kernels (rms_norm_bwd, layer_norm_bwd) --------------
def _mred_build(kernel, shape, dt):
    import examples.rms_norm as RN
    import examples.layer_norm as LN

    def rn(*s):
        return torch.randn(*s, device=DEV, dtype=dt)

    tol = (3e-2, 3e-2)
    if kernel == "rms_norm_bwd":
        (M, N) = shape
        xr, wr, gor = rn(M, N), rn(N), rn(M, N)
        rms_val = torch.rsqrt((xr.float() ** 2).mean(-1, keepdim=True) + EPS).to(dt)
        args = (gor, xr, wr, rms_val)
        # Ref must use the SAME rsqrt the kernel receives (rms_val, bf16 at bf16), not recompute it
        # in fp32 — otherwise the kernel is graded against a different-precision rsqrt (a harness
        # bug, not a kernel error). torch.compile of tc_ref gets the same rms_val too -> all fair.
        ref = _rms_bwd_ref(gor, xr, wr, N, rms_val)

        def tc_ref():
            return _rms_bwd_ref(gor, xr, wr, N, rms_val)

        return RN.rms_norm_bwd, args, ref, _acc_tuple(*tol), tc_ref
    if kernel == "layer_norm_bwd":
        (M, N) = shape
        xl, wl, gol = rn(M, N), rn(N), rn(M, N)
        mean_l = xl.float().mean(-1)
        rstd_l = torch.rsqrt(xl.float().var(-1, unbiased=False) + EPS)
        args = (gol, xl, mean_l, rstd_l, wl)
        ref = _ln_bwd_ref(gol, xl, mean_l, rstd_l, wl, N)

        def tc_ref():
            return _ln_bwd_ref(gol, xl, mean_l, rstd_l, wl, N)

        return LN.layer_norm_bwd, args, ref, _acc_tuple(*tol), tc_ref
    raise KeyError(kernel)


def _rms_bwd_ref(grad_out, x, weight, n, rsqrt_in=None):
    # rsqrt_in: the SAME rsqrt tensor handed to the kernel (so the ref grades against the kernel's
    # actual rsqrt precision). Fall back to fp32-recompute only if not provided.
    xf, dyf = x.float(), grad_out.float()
    rsqrt = rsqrt_in.float() if rsqrt_in is not None else torch.rsqrt((xf ** 2).mean(-1, keepdim=True) + EPS)
    wf = weight.float()[None, :]
    gw = (xf * dyf * rsqrt).sum(0)
    gx = wf * dyf * rsqrt - xf * rsqrt ** 3 * (wf * dyf * xf).mean(-1, keepdim=True)
    return gx.to(x.dtype), gw.to(weight.dtype)


def _ln_bwd_ref(grad_out, x, mean, rstd, weight, n):
    xf, dyf = x.float(), grad_out.float()
    wf = weight.float()[None, :]
    x_hat = (xf - mean.float()[:, None]) * rstd.float()[:, None]
    gw = (dyf * x_hat).sum(0)
    gb = dyf.sum(0)
    wdy = wf * dyf
    c1 = (x_hat * wdy).sum(-1, keepdim=True) / n
    c2 = wdy.sum(-1, keepdim=True) / n
    gx = (wdy - (x_hat * c1 + c2)) * rstd.float()[:, None]
    return gx.to(x.dtype), gw.to(weight.dtype), gb.to(weight.dtype)


# --------------------------------------------------------------------------- #
#  Corpus dispatch.
# --------------------------------------------------------------------------- #
def _clone_args(args):
    # Used ONLY for accuracy checks (pristine input per config) — NOT in the timing path. The
    # timing thunks call kernels in place with no clone (matches production; see run_vllm_cell).
    return tuple(t.clone() if torch.is_tensor(t) else t for t in args)


def run_real_cell(corpus, kernel, shape, dtype):
    """seed/default/tc cell. A `*_gen` corpus (held-out generalization shapes) uses the SAME
    builder as its reproduction parent — only the shape LIST differs (unseen shapes)."""
    dt = _DT[dtype] if dtype in _DT else None
    row = {"corpus": corpus, "kernel": kernel, "shape": list(shape), "dtype": dtype}
    # generalization corpora route to their reproduction parent's builder.
    family = corpus[:-4] if corpus.endswith("_gen") else corpus

    if family == "pointwise":
        kfn, args, ref, acc, tc_ref = _pointwise_build(kernel, shape, dt)
    elif family == "curriculum":
        kfn, args, ref, acc, tc_ref = _cur_build(kernel, shape[0], shape[1], dt)
    elif family == "transfer":
        kfn, args, ref, acc, tc_ref = _transfer_build(kernel, shape, dt)
    elif family == "mreduction":
        kfn, args, ref, acc, tc_ref = _mred_build(kernel, tuple(shape), dt)
    elif family == "vllm":
        return run_vllm_cell(kernel, shape, corpus=corpus)
    elif family == "qk_norm_rope":
        return run_qk_norm_rope_cell(tuple(shape), corpus=corpus)
    else:
        raise KeyError(corpus)

    torch._dynamo.reset()
    seed_cfg, base_cfg, fired = _extract_configs(kfn, args)
    row["fired_heuristics"] = fired
    row["seed_config"] = _cfg_dict(seed_cfg)
    row["base_default_config"] = _cfg_dict(base_cfg)
    row["configs_differ"] = _cfg_dict(seed_cfg) != _cfg_dict(base_cfg)

    # ---- compile each arm + accuracy-gate (records status/acc; builds timing thunks) ----
    arms = {}
    thunks = {}  # arm_name -> zero-arg thunk to time (only for arms that compiled+ran)
    for name, cfg in (("seed", seed_cfg), ("default", base_cfg)):
        a = {"config_present": cfg is not None}
        k = _replay(kfn, cfg)
        if k is None:
            a["status"] = "no-config"
            arms[name] = a
            continue
        try:
            out = with_compile_timeout(lambda kk=k: kk(*args))
            ok, detail = acc(out, ref)
            a["acc"] = ok
            a["acc_detail"] = detail
            a["status"] = "ok" if ok else "acc-fail"
            thunks[name] = (lambda kk=k: kk(*args))
        except _CompileTimeout:
            _reap_compile_children()
            a["status"] = "compile-fail:timeout"
            a["error"] = f"compile > {COMPILE_TIMEOUT_S}s (pathological ptxas)"
        except Exception as e:  # noqa: BLE001
            a["status"] = f"compile-fail:{type(e).__name__}"
            a["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        arms[name] = a

    # ---- tc ----
    a = {}
    if tc_ref is None:
        a["status"] = "n/a-no-tc"
    else:
        try:
            torch._dynamo.reset()
            tc = torch.compile(tc_ref)
            out_tc = with_compile_timeout(tc)
            ok, detail = acc(out_tc, ref)
            a["acc"] = ok
            a["acc_detail"] = detail
            a["status"] = "ok" if ok else "acc-fail"
            thunks["tc"] = tc
        except _CompileTimeout:
            _reap_compile_children()
            a["status"] = "compile-fail:timeout"
            a["error"] = f"compile > {COMPILE_TIMEOUT_S}s"
        except Exception as e:  # noqa: BLE001
            a["status"] = f"compile-fail:{type(e).__name__}"
            a["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    arms["tc"] = a

    _time_arms(arms, thunks)
    row["arms"] = arms
    _ratios(row)
    _cleanup()
    return row


def run_vllm_cell(kernel, shape, corpus="vllm"):
    """vLLM native-dtype cell: seed/default/tc/vllm_shipped. In-place kernels: clone per timed call.
    corpus is 'vllm' (reproduction, original 4 shapes) or 'vllm_gen' (tuned-grid sample)."""
    import bench_arms as B

    tok, hidden = shape[0], shape[1]
    group = shape[2] if len(shape) > 2 else None
    row = {"corpus": corpus, "kernel": kernel, "shape": list(shape), "dtype": "native"}
    mod_name, kern_attr, builder, sub, key_fn = B.SPECS[kernel]
    mod = importlib.import_module(mod_name)
    kfn = getattr(mod, kern_attr)
    built = builder(tok, hidden, group)
    args, ref_fn, out_idx, returns = built[0], built[1], built[2], built[3]

    torch._dynamo.reset()
    seed_cfg, base_cfg, fired = _extract_configs(kfn, args)
    row["fired_heuristics"] = fired
    row["seed_config"] = _cfg_dict(seed_cfg)
    row["base_default_config"] = _cfg_dict(base_cfg)
    row["configs_differ"] = _cfg_dict(seed_cfg) != _cfg_dict(base_cfg)

    # vLLM shipped config (nearest-shape lookup into the tuned JSON), same mechanism vLLM uses.
    vllm_cfg = vllm_chosen = vllm_exact = None
    try:
        entries = B.load_json_configs(sub)
        if entries is not None:
            want_key = key_fn(tok, hidden, group)
            vllm_cfg, vllm_chosen, vllm_exact = B.nearest_vllm_config(entries, want_key)
    except Exception as e:  # noqa: BLE001
        row["vllm_lookup_error"] = f"{type(e).__name__}: {e}"
    row["vllm_shipped_config"] = _cfg_dict(vllm_cfg)
    row["vllm_chosen_key"] = vllm_chosen
    row["vllm_exact_dims"] = vllm_exact
    row["vllm_config_dir"] = B.VLLM_CONFIG_DIR  # provenance: exactly which configs we compared to

    def compile_and_check(cfg):
        """Compile+run once on cloned args, accuracy-check vs eager ref. Returns (k, status, accpair)."""
        k = _replay(kfn, cfg)
        if k is None:
            return None, "no-config", None
        ak, ar = _clone_args(args), _clone_args(args)
        try:
            if returns:
                ok_k = with_compile_timeout(lambda: k(*ak))
                out_r = ref_fn(*ar)
                ok, detail = B.cmp_outputs(ak, ar, out_idx, returns, ok_k, out_r)
            else:
                with_compile_timeout(lambda: k(*ak))
                ref_fn(*ar)
                ok, detail = B.cmp_outputs(ak, ar, out_idx, returns)
            return k, ("ok" if ok else "acc-fail"), (ok, detail)
        except _CompileTimeout:
            _reap_compile_children()
            return None, "compile-fail:timeout", (False, f"compile > {COMPILE_TIMEOUT_S}s")
        except Exception as e:  # noqa: BLE001
            return None, f"compile-fail:{type(e).__name__}", (False, f"{type(e).__name__}: {str(e)[:200]}")

    arms = {}
    thunks = {}
    for name, cfg in (("seed", seed_cfg), ("default", base_cfg), ("vllm_shipped", vllm_cfg)):
        a = {"config_present": cfg is not None}
        if cfg is None:
            a["status"] = "no-config"
            arms[name] = a
            continue
        k, status, accpair = compile_and_check(cfg)
        a["status"] = status
        if accpair is not None:
            a["acc"], a["acc_detail"] = accpair
        if k is not None and status in ("ok", "acc-fail"):
            # NO CLONE in the timing thunk: production (vllm registers mutates_args and calls the op
            # directly on pre-allocated buffers) never clones — it overwrites the output buffer in
            # place. These kernels have data-independent tiling, so timing on the evolving buffer
            # measures identical work; verified stable (0.4% spread) + bounded (no blowup). Cloning
            # even the mutated arg adds 2.9-5.2x overhead on small shapes that production never pays.
            # (Accuracy was already checked on full clones in compile_and_check — that's the safe place.)
            thunks[name] = (lambda kk=k: kk(*args))
        arms[name] = a

    # tc arm: compile the torch reference. tc mutates its args in place too -> also NO clone in the
    # timed thunk (warm on a throwaway clone once, then time on `ar` in place).
    a = {}
    try:
        torch._dynamo.reset()
        ar = _clone_args(args)
        tc = torch.compile(ref_fn)
        with_compile_timeout(lambda: tc(*_clone_args(ar)))  # warm on a throwaway clone
        a["status"] = "ok"
        a["acc"] = True
        a["acc_detail"] = "tc==ref by construction"
        thunks["tc"] = (lambda: tc(*ar))
    except _CompileTimeout:
        _reap_compile_children()
        a["status"] = "compile-fail:timeout"
        a["error"] = f"compile > {COMPILE_TIMEOUT_S}s"
    except Exception as e:  # noqa: BLE001
        a["status"] = f"compile-fail:{type(e).__name__}"
        a["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    arms["tc"] = a

    _time_arms(arms, thunks)
    row["arms"] = arms
    _ratios(row)
    _cleanup()
    return row


# --------------------------------------------------------------------------- #
#  qk_norm_rope: an OUT-OF-SAMPLE KERNEL (not just shape) — 3D grid + inner RMS reduction
#  + RoPE epilogue, structurally unlike the 4 quant kernels. Own arg tuple + config key
#  {q_heads, kv_heads, num_tokens}. In-place on qkv. Tuned JSON vendored (our vllm-src predates it).
# --------------------------------------------------------------------------- #
_QK_HEAD_DIM = 128
_QK_EPS = 1e-6


def _qk_build(q_heads, kv_heads, num_tokens):
    """Build (kfn, args, ref_fn) for fused_qk_norm_rope at (q_heads, kv_heads, num_tokens)."""
    import kut.fused_qk_norm_rope as QK
    import refs

    hd, dev, dt = _QK_HEAD_DIM, DEV, torch.bfloat16
    total = (q_heads + 2 * kv_heads) * hd
    qkv = torch.empty(num_tokens, total, device=dev, dtype=dt).uniform_(-0.1, 0.1)
    positions = torch.arange(num_tokens, device=dev, dtype=torch.long)
    q_weight = torch.empty(hd, device=dev, dtype=dt).uniform_(0.8, 1.2)
    k_weight = torch.empty(hd, device=dev, dtype=dt).uniform_(0.8, 1.2)
    cos_sin = QK._compute_cos_sin_cache(40960, hd).to(dt)  # rotary_ratio=1.0 -> rotary_dim=head_dim
    args = (qkv, q_heads, kv_heads, kv_heads, hd, _QK_EPS, q_weight, k_weight,
            cos_sin, True, positions)
    ref_fn = refs.ref_fused_qk_norm_rope
    return QK.fused_qk_norm_rope, args, ref_fn


def run_qk_norm_rope_cell(shape, corpus="qk_norm_rope_gen"):
    """seed/default/tc/vllm_shipped cell for fused_qk_norm_rope. In-place; shape=(q_heads,kv_heads,num_tokens)."""
    import bench_arms as B

    q_heads, kv_heads, num_tokens = shape
    row = {"corpus": corpus, "kernel": "fused_qk_norm_rope", "shape": list(shape), "dtype": "native"}
    kfn, args, ref_fn = _qk_build(q_heads, kv_heads, num_tokens)

    torch._dynamo.reset()
    seed_cfg, base_cfg, fired = _extract_configs(kfn, args)
    row["fired_heuristics"] = fired
    row["seed_config"] = _cfg_dict(seed_cfg)
    row["base_default_config"] = _cfg_dict(base_cfg)
    row["configs_differ"] = _cfg_dict(seed_cfg) != _cfg_dict(base_cfg)

    # vLLM tuned config: key = {q_heads, kv_heads, num_tokens} — use qk_norm_rope's OWN nearest
    # lookup (heads-based, not hidden_size-based). Prefer live VLLM_CONFIG_DIR, fall back to the
    # vendored snapshot (our vllm-src predates this kernel).
    vllm_cfg = vllm_chosen = vllm_exact = None
    cfg_src = None
    platform = getattr(B, "PLATFORM", "nvidia_h100")
    for cfg_dir in (B.VLLM_CONFIG_DIR, os.path.join(_DEPS, "vllm_configs")):
        path = os.path.join(cfg_dir, "fused_qk_norm_rope", f"{platform}.json")
        if os.path.exists(path):
            entries = json.load(open(path))
            keys = [e["key"] for e in entries if e.get("key")]
            pool = [k for k in keys if k["q_heads"] == q_heads and k["kv_heads"] == kv_heads]
            if pool:
                toks = sorted(k["num_tokens"] for k in pool)
                chosen_tok = next((n for n in toks if n >= num_tokens), toks[-1])
                chosen = {"q_heads": q_heads, "kv_heads": kv_heads, "num_tokens": chosen_tok}
                cfg_body = next(e["config"] for e in entries if e["key"] == chosen)
                vllm_cfg = helion.Config(**cfg_body)
                vllm_chosen = chosen
                vllm_exact = (chosen_tok == num_tokens)
            cfg_src = cfg_dir
            break
    row["vllm_shipped_config"] = _cfg_dict(vllm_cfg)
    row["vllm_chosen_key"] = vllm_chosen
    row["vllm_exact_dims"] = vllm_exact
    row["vllm_config_dir"] = cfg_src

    def acc_qkv(mutated_k, mutated_ref):
        # compare only the q|k region that both mutate (v is untouched); upcast fp32.
        qk = (q_heads + kv_heads) * _QK_HEAD_DIM
        ok_o = mutated_k[:, :qk].to(torch.float32)
        rf = mutated_ref[:, :qk].to(torch.float32)
        ok = bool(torch.allclose(ok_o, rf, atol=5e-2, rtol=5e-2))  # upstream's tuned tol
        return ok, f"maxabs={float((ok_o - rf).abs().max()):.3e}"

    def compile_and_check(cfg):
        k = _replay(kfn, cfg)
        if k is None:
            return None, "no-config", None
        ak = _clone_args(args)
        ar = _clone_args(args)
        try:
            with_compile_timeout(lambda: k(*ak))
            ref_fn(*ar)
            ok, detail = acc_qkv(ak[0], ar[0])
            return k, ("ok" if ok else "acc-fail"), (ok, detail)
        except _CompileTimeout:
            _reap_compile_children()
            return None, "compile-fail:timeout", (False, f"compile > {COMPILE_TIMEOUT_S}s")
        except Exception as e:  # noqa: BLE001
            return None, f"compile-fail:{type(e).__name__}", (False, f"{type(e).__name__}: {str(e)[:200]}")

    arms = {}
    thunks = {}
    for name, cfg in (("seed", seed_cfg), ("default", base_cfg), ("vllm_shipped", vllm_cfg)):
        a = {"config_present": cfg is not None}
        if cfg is None:
            a["status"] = "no-config"
            arms[name] = a
            continue
        k, status, accpair = compile_and_check(cfg)
        a["status"] = status
        if accpair is not None:
            a["acc"], a["acc_detail"] = accpair
        if k is not None and status in ("ok", "acc-fail"):
            # NO CLONE in timing (matches production; see run_vllm_cell). qkv is overwritten in
            # place — data-independent tiling + bounded RoPE rotation, verified stable/finite.
            thunks[name] = (lambda kk=k: kk(*args))
        arms[name] = a

    # tc arm: torch.compile the reference. Warm on a throwaway clone, then time in place (no clone).
    a = {}
    try:
        torch._dynamo.reset()
        ar = _clone_args(args)
        tc = torch.compile(ref_fn)
        tc(*_clone_args(ar))  # warm on a throwaway clone
        a["status"] = "ok"
        a["acc"] = True
        a["acc_detail"] = "tc==ref by construction"
        thunks["tc"] = (lambda: tc(*ar))
    except _CompileTimeout:
        _reap_compile_children()
        a["status"] = "compile-fail:timeout"
        a["error"] = f"compile > {COMPILE_TIMEOUT_S}s"
    except Exception as e:  # noqa: BLE001
        a["status"] = f"compile-fail:{type(e).__name__}"
        a["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    arms["tc"] = a

    _time_arms(arms, thunks)
    row["arms"] = arms
    _ratios(row)
    _cleanup()
    return row


def _time_arms(arms: dict, thunks: dict):
    """Interleave-time all arms that compiled+ran (status ok OR acc-fail), on the SAME rounds,
    then attach per-arm us (eager cold-L2) / spread / round_medians AND coldgraph device time.
    Both metrics are stored per arm; the aggregator picks which to headline per corpus. Timing is
    recorded even for acc-fail arms (perf is always visible); ratios gate on acc separately."""
    timeable = {name: thunks[name] for name in thunks if arms[name].get("status") in ("ok", "acc-fail")}
    if not timeable:
        return
    result = timed_interleaved(timeable)
    for name, t in result.items():
        arms[name].update(t)  # us (eager cold-L2), spread, round_medians
    # coldgraph device time per arm (pure GPU, launch amortized). Both metrics are kept per arm;
    # the aggregator chooses which to headline per corpus (device time for vLLM-family — deployed
    # under CUDA graphs; eager for the functional reduction kernels).
    for name in timeable:
        cg = coldgraph_us(timeable[name])
        if cg is not None:
            arms[name]["coldgraph_us"] = cg


# --------------------------------------------------------------------------- #
#  Ratios (derived; aggregator recomputes geomeans from raw us).
# --------------------------------------------------------------------------- #
def _ratios(row):
    arms = row["arms"]

    def raw_us(name):
        a = arms.get(name, {})
        return a.get("us") if a.get("status") in ("ok", "acc-fail") else None

    def ok_us(name):
        a = arms.get(name, {})
        return a.get("us") if a.get("status") == "ok" else None

    s_ok, d_ok, t_ok, v_ok = ok_us("seed"), ok_us("default"), ok_us("tc"), ok_us("vllm_shipped")
    row["G_tc"] = round(t_ok / s_ok, 4) if (s_ok and t_ok) else None    # >1 => seed beats tc
    row["G_def"] = round(d_ok / s_ok, 4) if (s_ok and d_ok) else None   # >1 => seed beats default
    row["G_vllm"] = round(v_ok / s_ok, 4) if (s_ok and v_ok) else None  # >1 => seed beats vllm

    us = {"seed": raw_us("seed"), "default": raw_us("default"), "tc": raw_us("tc")}
    if arms.get("vllm_shipped") is not None:
        us["vllm_shipped"] = raw_us("vllm_shipped")
    row["us"] = us
    # coldgraph device-time companion (launch-overhead cross-check)
    cg = {n: arms.get(n, {}).get("coldgraph_us") for n in ("seed", "default", "tc", "vllm_shipped")}
    row["coldgraph_us"] = {k: v for k, v in cg.items() if v is not None}

    # perf ratios IGNORING accuracy (for acc-fail cells where seed==default fail identically —
    # the perf comparison is still meaningful; clearly separate from the acc-gated G above).
    s_r, d_r, t_r, v_r = raw_us("seed"), raw_us("default"), raw_us("tc"), raw_us("vllm_shipped")
    row["perf_only_tc"] = round(t_r / s_r, 4) if (s_r and t_r) else None
    row["perf_only_def"] = round(d_r / s_r, 4) if (s_r and d_r) else None
    row["perf_only_vllm"] = round(v_r / s_r, 4) if (s_r and v_r) else None


def _cleanup():
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  Driver: one (corpus, kernel) per process; iterate its shapes x dtypes.
# --------------------------------------------------------------------------- #
def _load_shapes():
    with open(os.path.join(_PERF_DIR, "shapes.json")) as f:
        return json.load(f)


def _iter_kernel_cells(SH, corpus, kernel):
    c = SH["corpora"][corpus]
    kdef = c["kernels"][kernel]
    dtypes = c["dtypes"]
    splits = c["required_splits"]
    for split in splits:
        for shape in kdef["shapes"][split]:
            for dtype in dtypes:
                yield tuple(shape), dtype


def _parse_only(only: str) -> set:
    out = set()
    for tok in only.split(","):
        tok = tok.strip()
        if not tok:
            continue
        shp, _, dt = tok.partition(":")
        shape = tuple(int(x) for x in shp.split("x"))
        out.add((shape, dt))
    return out


def _log_row(row):
    if "error" in row:
        print(f"[ERR ] {row.get('shape')}/{row.get('dtype')}: {row['error'][:120]}", flush=True)
        return
    u = row.get("us", {})
    cg = row.get("coldgraph_us", {})
    print(f"[cell] {str(row.get('shape')):22s} {row.get('dtype','?'):7s} "
          f"seed={u.get('seed')} def={u.get('default')} tc={u.get('tc')} "
          f"vllm={u.get('vllm_shipped')} | G_tc={row.get('G_tc')} G_def={row.get('G_def')} "
          f"G_vllm={row.get('G_vllm')} | differ={row.get('configs_differ')} "
          f"seed_cg={cg.get('seed')}", flush=True)


def _merge_rows(out_path, new_rows):
    # If no prior full run exists, --only-shapes just writes the new rows (nothing to merge into).
    if not os.path.exists(out_path):
        json.dump({"rows": new_rows}, open(out_path, "w"), indent=1)
        print(f"WROTE {len(new_rows)} row(s) to new {out_path}", flush=True)
        return
    existing = json.load(open(out_path))["rows"]

    def key(r):
        return (tuple(r["shape"]) if r.get("shape") is not None else None, r.get("dtype"))

    new_by_key = {key(r): r for r in new_rows}
    merged, replaced = [], 0
    for r in existing:
        k = key(r)
        if k in new_by_key:
            merged.append(new_by_key.pop(k)); replaced += 1
        else:
            merged.append(r)
    merged.extend(new_by_key.values())
    json.dump({"rows": merged}, open(out_path, "w"), indent=1)
    print(f"MERGED: replaced {replaced}, added {len(new_by_key)}, total {len(merged)} in {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only-shapes", default="",
                    help="comma list MxN:dtype (e.g. 16384x1024:bf16); merges into existing JSON")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already recorded (us or error) in the output JSON")
    args = ap.parse_args()

    SH = _load_shapes()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.corpus}__{args.kernel}.json")
    print(f"helion={helion.__file__}", flush=True)
    _reps_desc = _REPS_FIXED if _REPS_FIXED is not None else f"budget~{_ROUND_BUDGET_MS}ms[{_REPS_FLOOR}..{_REPS_CAP}]"
    print(f"corpus={args.corpus} kernel={args.kernel} rounds={_ROUNDS} reps/round={_reps_desc} "
          f"coldgraph={_COLDGRAPH} -> {out_path}", flush=True)

    only = _parse_only(args.only_shapes) if args.only_shapes else None

    # CELL-LEVEL RESUME: if a prior (crashed) run left a checkpoint, load its rows and skip cells
    # already recorded — so a restart continues from where it stopped, not from scratch. A cell is
    # "done" if it has a us dict OR an error (errors count as done so we don't loop forever on a
    # reproducibly-crashing cell; the SEGFAULT/OOM case that kills the PROCESS mid-cell leaves no
    # row, so it's naturally retried).
    rows = []
    done = set()
    if only is None and args.resume and os.path.exists(out_path):
        try:
            rows = json.load(open(out_path)).get("rows", [])
            for r in rows:
                k = (tuple(r["shape"]) if r.get("shape") is not None else None, r.get("dtype"))
                if "us" in r or "error" in r:
                    done.add(k)
            print(f"RESUME: {len(done)} cell(s) already done in {out_path}", flush=True)
        except Exception:  # noqa: BLE001
            rows, done = [], set()

    for shape, dtype in _iter_kernel_cells(SH, args.corpus, args.kernel):
        if only is not None and (shape, dtype) not in only:
            continue
        if (shape, dtype) in done:
            continue
        try:
            row = run_real_cell(args.corpus, args.kernel, shape, dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {"corpus": args.corpus, "kernel": args.kernel, "shape": list(shape),
                   "dtype": dtype, "error": "OOM"}
        except Exception as e:  # noqa: BLE001
            row = {"corpus": args.corpus, "kernel": args.kernel, "shape": list(shape),
                   "dtype": dtype, "error": f"{type(e).__name__}: {str(e)[:300]}",
                   "trace": traceback.format_exc()}
        rows.append(row)
        _log_row(row)
        if only is None:
            json.dump({"rows": rows}, open(out_path, "w"), indent=1)  # checkpoint per cell
    if only is not None:
        _merge_rows(out_path, rows)
    else:
        json.dump({"rows": rows}, open(out_path, "w"), indent=1)
    print(f"\n=== DONE {args.corpus}/{args.kernel} -> {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
