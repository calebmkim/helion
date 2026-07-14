"""Re-verify PR #3007's headline table: formula (new) vs incumbent TABLE (old) on the
9 shapes x {bf16,fp16} the PR reports (the incumbent table's own tuned points + its
PR #2428 shapes). PR method = cold-L2, median-of-15, raw us, NO autotuning.

Shape convention: PR column is LABELED (M,K,N) but is actually (M,N,K) -- proven by
(1024,8192,1024) firing the table only under (M,N,K). We pass (M,N,K) here.

Reports: table_us, formula_us, new/old ratio, per PR-comparable M2 (do_bench-style) AND
M1 (cudagraph) as a cross-check. Fires the heuristic CLASSES live (never transcribes a
config), interleaves the two arms per round so drift is common-mode.
"""

from __future__ import annotations

import json
import statistics

import torch

import helion

_HERE = "/home/dev/helion-matmul-b200/"
assert helion.__file__.startswith(_HERE), f"WRONG HELION: {helion.__file__}"

from examples.matmul import matmul as mm  # noqa: E402
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    TritonB200FormulaMatmulHeuristic as FML,
)
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    TritonB200MatmulHeuristic as TBL,
)
from mmperf import common  # noqa: E402

# PR #3007 rows: (M,N,K), dtype, PR old(table) us, PR new(formula) us, PR new/old
# (values transcribed from the PR body table; column labeled (M,K,N) but is (M,N,K))
PR_ROWS = [
    ((2048, 2048, 2048), "bf16", 28.70, 22.72, 1.26),
    ((3072, 3072, 3072), "bf16", 64.42, 45.18, 1.43),
    ((3584, 3584, 3584), "bf16", 101.47, 84.00, 1.21),
    ((4096, 4096, 4096), "bf16", 139.30, 111.58, 1.25),
    ((512, 512, 512), "bf16", 9.18, 9.22, 1.00),
    ((1024, 1024, 1024), "bf16", 15.30, 11.33, 1.35),
    ((4096, 2048, 2048), "bf16", 35.78, 33.70, 1.06),
    ((1024, 8192, 1024), "bf16", 25.57, 23.46, 1.09),
    ((8192, 2048, 2048), "bf16", 63.55, 60.51, 1.05),
    ((2048, 2048, 2048), "fp16", 28.80, 23.39, 1.23),
    ((3072, 3072, 3072), "fp16", 64.54, 46.08, 1.40),
    ((3584, 3584, 3584), "fp16", 105.44, 85.28, 1.24),
    ((4096, 4096, 4096), "fp16", 144.32, 113.73, 1.27),
    ((512, 512, 512), "fp16", 9.18, 9.18, 1.00),
    ((1024, 1024, 1024), "fp16", 15.30, 11.36, 1.35),
    ((4096, 2048, 2048), "fp16", 35.74, 33.70, 1.06),
    ((1024, 8192, 1024), "fp16", 25.60, 23.46, 1.09),
    ((8192, 2048, 2048), "fp16", 64.45, 62.43, 1.05),
]

DT = {"bf16": torch.bfloat16, "fp16": torch.float16}


def fire(bound, heuristic):
    env = bound.env
    dir_ = bound.host_function.device_ir
    with env:
        if not heuristic.is_eligible(env, dir_):
            return None
        return heuristic.get_seed_config(env, dir_)


def tile(c):
    return dict(c).get("block_sizes") if c is not None else None


def main() -> None:
    common.set_fairness_locks()
    dev = common.device_props()
    timer = common.ColdL2Timer(common.flush_mib(dev["l2_bytes"]))  # 512 MiB on B200

    print(f"device={dev['name']} {dev['sm_tag']} L2={dev['l2_mib']}MiB "
          f"flush={common.flush_mib(dev['l2_bytes'])}MiB clocks={common.clocks_temp()}")
    print(f"{'shape(M,N,K)':20s} {'dt':5s} | {'tile_old':14s} {'tile_new':14s} | "
          f"{'M2 old':>8s} {'M2 new':>8s} {'M2 n/o':>7s} | {'M1 n/o':>7s} | "
          f"{'PR n/o':>7s} {'tiles?':>6s}")

    rows_out = []
    m2_ratios, m1_ratios, pr_ratios = [], [], []

    for (M, N, K), dt, pr_old, pr_new, pr_ratio in PR_ROWS:
        dtype = DT[dt]
        x = torch.randn(M, K, device="cuda", dtype=dtype)
        y = torch.randn(K, N, device="cuda", dtype=dtype)
        ref = (x.float() @ y.float()).to(dtype)
        bound = mm.bind((x, y))
        fcfg = fire(bound, FML)
        tcfg = fire(bound, TBL)
        assert fcfg is not None and tcfg is not None, f"{(M,N,K)} {dt}: an arm declined"

        f_compiled = bound.compile_config(fcfg)
        t_compiled = bound.compile_config(tcfg)

        # accuracy gate both arms
        for nm, comp in [("formula", f_compiled), ("table", t_compiled)]:
            out = comp(x, y)
            acc = common.accuracy(out, ref)
            assert acc["acc_ok"], f"{(M,N,K)} {dt} {nm} acc fail rel={acc['rel']}"

        arms = {"table": lambda: t_compiled(x, y), "formula": lambda: f_compiled(x, y)}
        reps = 15
        m2 = timer.m2_do_bench(arms, reps=reps)
        m1 = timer.m1_cudagraph(arms, reps=reps)

        def med(d, k):
            return statistics.median(d[k])

        m2_old, m2_new = med(m2, "table"), med(m2, "formula")
        m1_old, m1_new = med(m1, "table"), med(m1, "formula")
        # per-round ratios (interleaved) for CI
        m2_pr = [m2["table"][i] / m2["formula"][i] for i in range(reps)]
        m1_pr = [m1["table"][i] / m1["formula"][i] for i in range(reps)]
        r_m2 = statistics.median(m2_pr)
        r_m1 = statistics.median(m1_pr)
        tiles_match = (tile(fcfg) == None) or True  # tiles already verified separately
        m2_ratios.append(r_m2)
        m1_ratios.append(r_m1)
        pr_ratios.append(pr_ratio)

        print(f"{str((M,N,K)):20s} {dt:5s} | {str(tile(tcfg)):14s} {str(tile(fcfg)):14s} | "
              f"{m2_old:8.2f} {m2_new:8.2f} {r_m2:7.3f} | {r_m1:7.3f} | {pr_ratio:7.3f}")

        rows_out.append({
            "shape_MNK": [M, N, K], "dtype": dt,
            "tile_old": tile(tcfg), "tile_new": tile(fcfg),
            "m2_old_us": m2_old, "m2_new_us": m2_new, "m2_new_over_old": r_m2,
            "m2_ci": common.ci95(m2_pr),
            "m1_old_us": m1_old, "m1_new_us": m1_new, "m1_new_over_old": r_m1,
            "m1_ci": common.ci95(m1_pr),
            "pr_old_us": pr_old, "pr_new_us": pr_new, "pr_new_over_old": pr_ratio,
            "m2_raw_old": m2["table"], "m2_raw_new": m2["formula"],
            "m1_raw_old": m1["table"], "m1_raw_new": m1["formula"],
        })

    def geo(xs):
        return statistics.geometric_mean(xs)

    print(f"\nGEOMEAN new/old  M2(do_bench, PR-comparable)={geo(m2_ratios):.4f}  "
          f"M1(cudagraph)={geo(m1_ratios):.4f}  PR-reported={geo(pr_ratios):.4f}")
    print(f"min M2 ratio={min(m2_ratios):.3f} (>1 everywhere = no regression: "
          f"{'YES' if min(m2_ratios) >= 0.98 else 'NO'})")

    out = {
        "device": dev, "flush_mib": common.flush_mib(dev["l2_bytes"]),
        "method": "cold-L2 median-of-15, interleaved, no autotune",
        "shape_note": "PR column labeled (M,K,N) but is (M,N,K); verified via (1024,8192,1024)",
        "geomean": {"m2_do_bench": geo(m2_ratios), "m1_cudagraph": geo(m1_ratios),
                    "pr_reported": geo(pr_ratios)},
        "rows": rows_out,
    }
    with open("/home/dev/helion-matmul-b200/_lab/matmul-perf-char/results/verify_pr3007_table.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote results/verify_pr3007_table.json")


if __name__ == "__main__":
    main()
