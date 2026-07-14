"""Shared harness for the matmul-family perf characterization (B200 / sm100).

Faithful arm extraction (see DECISIONS.md D1):
  seed            = TritonB200FormulaMatmulHeuristic.get_seed_config  (== promoted
                    compiler_default_config on this branch — the FORMULA under test)
  helion_default  = TritonB200MatmulHeuristic.get_seed_config if it fires
                    (default_source=table) ELSE config_spec._base_default_config()
                    (the ~[16,16,16] base, default_source=base) -- the incumbent being beaten
  tc_max_autotune = torch.compile(op, mode="max-autotune-no-cudagraphs")

Timing: 512 MiB cold-L2 flush (4x the 126.5 MiB B200 L2 -- triton's 256 MiB do_bench
would NOT evict). M1 = cudagraph graph-diff device time; M2 = manual flush+launch+event.
"""

from __future__ import annotations

import os
import statistics
from typing import Any
from typing import Callable

import torch

import helion

_HERE = "/home/dev/helion-matmul-b200/"
assert helion.__file__.startswith(_HERE), (
    f"WRONG HELION: {helion.__file__} not under {_HERE}"
)

from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    TritonB200FormulaMatmulHeuristic as _FORMULA,
)
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    TritonB200MatmulHeuristic as _TABLE,
)

DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8_e4m3": torch.float8_e4m3fn,
}

# B200 dense peaks (TFLOP/s), verified per SKU in DECISIONS.md D8.
PEAK_TFLOPS = {"bf16": 2250.0, "fp16": 2250.0, "fp8_e4m3": 4500.0}


# ---------------------------------------------------------------------------
# device + fairness locks
# ---------------------------------------------------------------------------
def device_props() -> dict[str, Any]:
    p = torch.cuda.get_device_properties(0)
    l2 = getattr(p, "L2_cache_size", None) or getattr(p, "l2_cache_size", 0)
    return {
        "name": p.name,
        "cc": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "l2_bytes": int(l2),
        "l2_mib": round(l2 / 1024 / 1024, 1),
        "total_mem_gib": round(p.total_memory / 1024**3, 1),
        "sm_tag": "sm100" if p.major == 10 else f"sm{p.major}{p.minor}",
    }


def flush_mib(l2_bytes: int) -> int:
    """L2 flush buffer size: max(256 MiB, 4x device L2). DECISIONS.md D2."""
    return max(256, int(round(4 * l2_bytes / 1024 / 1024)))


def set_fairness_locks() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def clocks_temp() -> dict[str, Any]:
    """SM clock (MHz), temp (C), throttle reasons via nvidia-smi (best-effort)."""
    import subprocess

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
                "-i",
                os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        parts = out.stdout.strip().split(",")
        return {
            "sm_clock_mhz": parts[0].strip() if parts else "n/a",
            "temp_c": parts[1].strip() if len(parts) > 1 else "n/a",
            "throttle": parts[2].strip() if len(parts) > 2 else "n/a",
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:100]}


# ---------------------------------------------------------------------------
# faithful config extraction
# ---------------------------------------------------------------------------
def _fire(heuristic: Any, bound: Any) -> Any:
    env = bound.env
    dir_ = bound.host_function.device_ir
    with env:
        if not heuristic.is_eligible(env, dir_):
            return None
        return heuristic.get_seed_config(env, dir_)


def extract_configs(bound: Any) -> dict[str, Any]:
    """Return the seed / default / base configs + default_source for a bound kernel."""
    formula_cfg = _fire(_FORMULA, bound)
    table_cfg = _fire(_TABLE, bound)
    with bound.env:
        base_cfg = bound.env.config_spec._base_default_config()
        promoted = bound.env.config_spec.compiler_default_config
    # sanity: on this branch the promoted default must equal the formula seed
    seed_fired = formula_cfg is not None
    cd_is_formula = (
        promoted is not None
        and formula_cfg is not None
        and dict(promoted) == dict(formula_cfg)
    )
    if table_cfg is not None:
        default_cfg = table_cfg
        default_source = "table"
    else:
        default_cfg = base_cfg
        default_source = "base"
    return {
        "seed_cfg": formula_cfg,
        "default_cfg": default_cfg,
        "default_source": default_source,
        "base_cfg": base_cfg,
        "table_cfg": table_cfg,
        "seed_fired": seed_fired,
        "promoted_is_formula": cd_is_formula,
    }


def cfg_summary(cfg: Any) -> dict[str, Any] | None:
    if cfg is None:
        return None
    d = dict(cfg)
    return {
        "block_sizes": d.get("block_sizes"),
        "num_warps": d.get("num_warps"),
        "num_stages": d.get("num_stages"),
    }


# ---------------------------------------------------------------------------
# accuracy gate
# ---------------------------------------------------------------------------
def accuracy(out: torch.Tensor, ref: torch.Tensor, rtol: float = 5e-2) -> dict[str, Any]:
    o = out.float()
    r = ref.float()
    if not torch.isfinite(o).all():
        return {"acc_ok": False, "rel": float("inf")}
    denom = r.abs().amax().clamp_min(1e-6)
    rel = float((o - r).abs().amax() / denom)
    return {"acc_ok": bool(rel < rtol), "rel": rel}


# ---------------------------------------------------------------------------
# cold-L2 timing: M1 (cudagraph diff) + M2 (do_bench-style), INTERLEAVED
#
# Interleaving (spec §2): within each round the arms are measured ADJACENTLY so
# thermal/clock/allocator drift is common-mode and cancels in the per-round ratio
# G_r = base[r]/seed[r]. Returns {arm: [t_us per round]} for R rounds.
# ---------------------------------------------------------------------------
def _event_time(fn: Callable[[], Any]) -> float:
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    st.record()
    fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000.0  # ms -> us


class ColdL2Timer:
    def __init__(self, flush_mib_val: int):
        self.flush = torch.empty(
            flush_mib_val * 1024 * 1024 // 4, dtype=torch.int32, device="cuda"
        )

    def m1_cudagraph(
        self,
        arm_fns: dict[str, Callable[[], Any]],
        reps: int,
        warmup: int = 5,
        inner: int = 10,
    ) -> dict[str, list[float]]:
        """M1: cold-L2 device time via graph-diff, interleaved across arms.

        Captures a [flush+kernel] graph per arm + a shared [flush]-only graph. Per
        round: measure the flush-only baseline as the per-replay time of `inner`
        batched replays (one event window -> amortizes event/launch overhead, the
        spec's "replay-averaged"), then each arm's [flush+kernel] per-replay time the
        same way. The arm's device time for that round = t(flush+kernel) - t(flush).
        Batching `inner` replays inside one event pair is what makes the ~15 us
        signal riding on a ~64 us flush measurable (single-replay events are too
        noisy)."""
        flush = self.flush
        # warmup every arm on a side stream (required before capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                flush.zero_()
                for fn in arm_fns.values():
                    fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        gB = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gB):
            flush.zero_()
        graphs: dict[str, torch.cuda.CUDAGraph] = {}
        for name, fn in arm_fns.items():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                flush.zero_()
                fn()
            graphs[name] = g

        def per_replay(g: torch.cuda.CUDAGraph) -> float:
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            st.record()
            for _ in range(inner):
                g.replay()
            en.record()
            torch.cuda.synchronize()
            return st.elapsed_time(en) * 1000.0 / inner  # us per replay

        out: dict[str, list[float]] = {name: [] for name in arm_fns}
        for _ in range(reps):
            base = per_replay(gB)  # this round's flush-only per-replay baseline
            for name, g in graphs.items():
                out[name].append(per_replay(g) - base)
        return out

    def m2_do_bench(
        self, arm_fns: dict[str, Callable[[], Any]], reps: int, warmup: int = 5
    ) -> dict[str, list[float]]:
        """M2: cold-L2, no cudagraph — the tritonbench do_bench method.

        Faithful to triton.testing.do_bench: enqueue [flush, start, fn, end] for
        every (round, arm) and let the HOST RUN AHEAD (a single synchronize per
        round, NOT per launch), then read the per-launch event deltas. This is the
        key vs a naive inner-sync loop: syncing every call forces host-in-the-loop
        serialization that over-states host overhead; running ahead reflects what
        do_bench actually reports (host overhead only surfaces when dispatch can't
        keep up with a short kernel). Arms are adjacent within a round so
        thermal/clock drift is common-mode in the ratio. Flush (512 MiB) before each
        launch => cold L2 per arm; the start event sits AFTER the flush so the flush
        itself is excluded from the timed window."""
        flush = self.flush
        for _ in range(warmup):
            flush.zero_()
            for fn in arm_fns.values():
                fn()
        torch.cuda.synchronize()
        out: dict[str, list[float]] = {name: [] for name in arm_fns}
        names = list(arm_fns)
        for _ in range(reps):
            evs: dict[str, tuple] = {}
            for name in names:  # enqueue whole round, no inner sync
                flush.zero_()
                st = torch.cuda.Event(enable_timing=True)
                en = torch.cuda.Event(enable_timing=True)
                st.record()
                arm_fns[name]()
                en.record()
                evs[name] = (st, en)
            torch.cuda.synchronize()  # one sync per round -> host ran ahead
            for name in names:
                st, en = evs[name]
                out[name].append(st.elapsed_time(en) * 1000.0)
        return out


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def ci95(xs: list[float]) -> list[float]:
    """Simple percentile CI (2.5/97.5) — robust, no distributional assumption."""
    if len(xs) < 2:
        return [float("nan"), float("nan")]
    s = sorted(xs)
    lo = s[max(0, int(round(0.025 * (len(s) - 1))))]
    hi = s[min(len(s) - 1, int(round(0.975 * (len(s) - 1))))]
    return [lo, hi]


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and x > 0 and x == x]
    return statistics.geometric_mean(xs) if xs else float("nan")
