#!/usr/bin/env python
"""Clean single-process verification of a claimed SELECTION gap.

Takes the exact configs, and in ONE process times each with BOTH:
  (a) helion's ACTUAL in-search timer (autotuner.benchmarking.interleaved_bench,
      event-timed L2-flushed non-cudagraph — what LFBO ranks on), and
  (b) the cudagraph cold-L2 ruler (true GPU time).
Then reports which config each timer picks as fastest. A selection gap is CONFIRMED
only if the in-search timer's winner DIFFERS from the cudagraph winner (i.e. the
timer mis-ranks the truly-faster config). Same operands, same process, interleaved.

Usage: rank0_verify_selection.py --shape m64_k4096_n24576 [--repeat 200] [--trials 5]
Configs are read from rank0/verify_configs.json: {"shape": {...}, "configs": [ {name, config}... ]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

WORKTREE = "/home/dev/local/helion-rank0"
sys.path.insert(0, str(Path(WORKTREE) / "benchmarks" / "cute"))
LAB = Path("/home/dev/local/helion-rank0/_lab/matmul-autotune")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument(
        "--repeat",
        type=int,
        default=200,
        help="interleaved_bench repeat (in-search timer)",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=5,
        help="repeat the whole interleaved measurement N times (each is one "
        "independent 'autotuner ranking'); we check winner stability",
    )
    ap.add_argument("--cg-reps", type=int, default=1000)
    ap.add_argument(
        "--cg-rounds",
        type=int,
        default=12,
        help="cudagraph interleaved rounds (reliable-estimator N)",
    )
    ap.add_argument(
        "--cg-bestofk",
        type=int,
        default=3,
        help="cudagraph best-of-K windows per round (kills slow-mode)",
    )
    a = ap.parse_args()

    spec = json.loads(Path(a.configs_json).read_text())
    sh = spec["shape"]
    m, n, k = sh["m"], sh["n"], sh["k"]

    import os

    os.environ["HELION_BACKEND"] = "cute"
    # branch-neutral for fixed configs; lets any bk256 config compile
    os.environ.setdefault("HELION_RANK0_BK256", "1")
    os.environ.setdefault("HELION_RANK0_AB_CEIL", "1")
    import torch

    import helion

    assert WORKTREE in helion.__file__
    import compare_matmul_backends as M
    from examples.fp8_matmul import fp8_matmul as kernel
    from triton import runtime as TR

    from helion.autotuner.benchmarking import interleaved_bench

    ns = argparse.Namespace(
        m=m, n=n, k=k, epilogue="scaled_mm", dtype="float8_e4m3fn", seed=0
    )
    dtype, mat_a, mat_b, bias, residual = M._make_matmul_problem(ns)
    expected = M._matmul_expected(ns, mat_a, mat_b, bias, residual, dtype)
    kargs = M._helion_matmul_args(ns, mat_a, mat_b, bias, residual)

    names, fns, bounds = [], [], []
    for c in spec["configs"]:
        cfg = helion.Config(**c["config"])
        bound = kernel.bind(kargs)
        if any(key in c["config"] for key in M._TCGEN05_CONFIG_KEYS):
            bound.env.config_spec.cute_tcgen05_search_enabled = True
        bound.set_config(cfg)
        # accuracy gate
        try:
            M._check_close(bound(*kargs), expected, dtype)
            acc = "PASS"
        except Exception as e:
            acc = f"FAIL:{type(e).__name__}"
        names.append(c["name"])
        fns.append(lambda b=bound: b(*kargs))
        bounds.append((c["name"], acc))

    M._gpu_warmup(8000)

    # (a) helion's ACTUAL in-search timer, repeated `trials` times (each trial =
    # one independent autotuner-style ranking). Two variants:
    #   stock cute timer = interleaved_bench_generic (WALL-CLOCK, folds launch)
    #   PR#2994 timer     = interleaved_bench(current_stream=True) (event, launch-free)
    # We time BOTH so we can see if #2994 flips the ranking to match cudagraph.
    from helion.autotuner.benchmarking import interleaved_bench_generic

    insearch_trials = []  # stock wall-clock
    insearch_cs_trials = []  # PR#2994 current-stream events
    for _ in range(a.trials):
        insearch_trials.append(interleaved_bench_generic(fns, repeat=a.repeat))
        insearch_cs_trials.append(
            interleaved_bench(fns, repeat=a.repeat, current_stream=True)
        )

    # (b) cudagraph cold-L2 ruler, same process
    active = TR.driver.active
    l2 = active.get_empty_cache_for_benchmark()
    di = active.get_device_interface()

    # cudagraph GROUND TRUTH via the sweep's RELIABLE estimator (cobench_reliable):
    # best-of-K windows (min kills the transient slow-mode) x N interleaved rounds,
    # report median-of-rounds. A single window is NOT reliable (catches slow mode).
    def make_graph(fn):
        for _ in range(20):
            fn()
        di.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(20):
            g.replay()
        di.synchronize()
        return g

    def window(g, reps):
        ss = [di.Event(enable_timing=True) for _ in range(reps)]
        ee = [di.Event(enable_timing=True) for _ in range(reps)]
        for i in range(reps):
            active.clear_cache(l2)
            ss[i].record()
            g.replay()
            ee[i].record()
        di.synchronize()
        return statistics.median(
            [s.elapsed_time(e) for s, e in zip(ss, ee, strict=True)]
        )

    graphs = [make_graph(fn) for fn in fns]
    cg_rounds = {j: [] for j in range(len(fns))}
    for _ in range(a.cg_rounds):
        for j, g in enumerate(graphs):
            cg_rounds[j].append(min(window(g, a.cg_reps) for _ in range(a.cg_bestofk)))
    cg = [statistics.median(cg_rounds[j]) for j in range(len(fns))]
    cg_spread = [max(cg_rounds[j]) / min(cg_rounds[j]) for j in range(len(fns))]

    # analysis
    def med(trials, j):
        return statistics.median([t[j] for t in trials])

    wall_median = [med(insearch_trials, j) for j in range(len(fns))]
    cs_median = [med(insearch_cs_trials, j) for j in range(len(fns))]
    wall_winner = names[min(range(len(fns)), key=lambda j: wall_median[j])]
    cs_winner = names[min(range(len(fns)), key=lambda j: cs_median[j])]
    cg_winner = names[min(range(len(fns)), key=lambda j: cg[j])]

    out = {
        "shape": sh,
        "names": names,
        "accuracy": dict(bounds),
        "stock_wallclock_median_ms": dict(zip(names, wall_median)),
        "pr2994_currentstream_median_ms": dict(zip(names, cs_median)),
        "cudagraph_ms": dict(zip(names, cg)),
        "cudagraph_spread": dict(zip(names, cg_spread)),
        "stock_wallclock_winner": wall_winner,
        "pr2994_currentstream_winner": cs_winner,
        "cudagraph_winner (ground truth)": cg_winner,
        "STOCK_HAS_SELECTION_GAP": wall_winner != cg_winner,
        "PR2994_FIXES_SELECTION_GAP": (
            wall_winner != cg_winner and cs_winner == cg_winner
        ),
    }
    Path(a.out_json).write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "stock_wallclock_median_ms",
                    "pr2994_currentstream_median_ms",
                    "cudagraph_ms",
                    "cudagraph_spread",
                    "stock_wallclock_winner",
                    "pr2994_currentstream_winner",
                    "cudagraph_winner (ground truth)",
                    "STOCK_HAS_SELECTION_GAP",
                    "PR2994_FIXES_SELECTION_GAP",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
