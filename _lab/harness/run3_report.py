"""Merge Run-1 (C1/C4/C7) + Run-2 (C2/C3/C5/C6/C8) into one report.

Writes a markdown report to _lab/logs/run3/RUN3_REPORT.md:
  - per (kernel, shape): the 8-config latency row (us) + speedup-vs-tc-default,
  - per (kernel, shape): the 4 Helion autotune arms' best-so-far by GENERATION,
  - per-kernel + overall geomeans for each config.

Reads:  run1_<kernel>.json (rows: default_lat_us, seed_lat_us, tc_lat_us),
        run2_<kernel>_<MxN>.json (arms[*].rebench_lat_us_median + per_gen_best_ms,
        c8_tc_max_us).  Missing/OOM cells render as "OOM/—".

Usage: python run3_report.py [--out _lab/logs/run3]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

from run3_run1_matrix import TEST_SHAPES

ARMS = [("unseeded_quick", "C2"), ("seeded_quick", "C5"),
        ("unseeded_full", "C3"), ("seeded_full", "C6")]


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else None


def f(x, d=1):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "—"


def load_run1(out_dir):
    """{kernel: {(m,n): {c1,c4,c7}}}"""
    r = {}
    for p in glob.glob(os.path.join(out_dir, "run1_*.json")):
        d = json.load(open(p))
        k = d["kernel"]
        r[k] = {}
        for row in d["rows"]:
            if "error" in row:
                continue
            r[k][tuple(row["shape"])] = {
                "c1": row.get("default_lat_us"), "c4": row.get("seed_lat_us"),
                "c7": row.get("tc_lat_us"),
            }
    return r


def load_cell(out_dir, kernel, m, n):
    p = os.path.join(out_dir, f"run2_{kernel}_{m}x{n}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "logs", "run3"))
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out)
    run1 = load_run1(out_dir)
    lines = ["# Run-3 reduction benchmark report (TEST shapes, fp32, H100)",
             "",
             "Latencies in microseconds (do_bench median-of-7). Configs: "
             "C1 Helion-default · C4 Helion-seed · C7 tc-default · "
             "C2 unseeded-quick · C5 seeded-quick · C3 unseeded-full · "
             "C6 seeded-full · C8 tc-max-autotune. "
             "Autotune arms (C2/C3/C5/C6) = winner fair-re-benched.", ""]
    # per-config geomean accumulators (of G = c7/lat, so higher=better vs tc-default)
    overall = {c: [] for c in ["c1", "c4", "c7", "c2", "c5", "c3", "c6", "c8"]}
    per_kernel_g = {}

    for kernel in TEST_SHAPES:
        lines.append(f"\n## {kernel}\n")
        lines.append("| shape | C1 dflt | C4 seed | C7 tc | C2 u-q | C5 s-q "
                     "| C3 u-f | C6 s-f | C8 tc-max |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        kg = {c: [] for c in overall}
        for (m, n) in TEST_SHAPES[kernel]:
            r1 = run1.get(kernel, {}).get((m, n), {})
            cell = load_cell(out_dir, kernel, m, n)
            row = {"c1": r1.get("c1"), "c4": r1.get("c4"), "c7": r1.get("c7")}
            if cell:
                arms = cell.get("arms", {})
                row["c2"] = arms.get("unseeded_quick", {}).get("rebench_lat_us_median")
                row["c5"] = arms.get("seeded_quick", {}).get("rebench_lat_us_median")
                row["c3"] = arms.get("unseeded_full", {}).get("rebench_lat_us_median")
                row["c6"] = arms.get("seeded_full", {}).get("rebench_lat_us_median")
                row["c8"] = cell.get("c8_tc_max_us")
            lines.append(
                f"| ({m},{n}) | " + " | ".join(
                    f(row.get(c)) for c in
                    ["c1", "c4", "c7", "c2", "c5", "c3", "c6", "c8"]) + " |")
            c7 = row.get("c7")
            for c in overall:
                lat = row.get(c)
                if c7 and lat:
                    kg[c].append(c7 / lat)  # G vs tc-default
            # per-generation table for the 4 arms
            if cell:
                _gen_table(lines, cell)
        per_kernel_g[kernel] = {c: geomean(kg[c]) for c in kg}
        for c in overall:
            overall[c].extend(kg[c])

    # summary
    lines.append("\n## Per-kernel geomean G (tc_default / latency; >1 beats tc-default)\n")
    cols = ["c1", "c4", "c7", "c2", "c5", "c3", "c6", "c8"]
    lines.append("| kernel | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for kernel in TEST_SHAPES:
        g = per_kernel_g[kernel]
        lines.append(f"| {kernel} | " + " | ".join(
            (f"{g[c]:.3f}" if g[c] else "—") for c in cols) + " |")
    lines.append("| **OVERALL** | " + " | ".join(
        (f"{geomean(overall[c]):.3f}" if geomean(overall[c]) else "—")
        for c in cols) + " |")

    rep = os.path.join(out_dir, "RUN3_REPORT.md")
    with open(rep, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {rep}")
    print("\nOVERALL geomean G (vs tc-default):")
    for c in cols:
        gm = geomean(overall[c])
        print(f"  {c}: {gm:.4f}" if gm else f"  {c}: —")


def _gen_table(lines, cell):
    arms = cell.get("arms", {})
    pg = {a: arms.get(a, {}).get("per_gen_best_ms", {}) for a, _ in ARMS}
    allg = sorted({int(g) for d in pg.values() for g in d})
    if not allg:
        return
    lines.append("")
    lines.append("<details><summary>per-generation best-so-far (ms)</summary>\n")
    lines.append("| gen | C2 u-q | C5 s-q | C3 u-f | C6 s-f |")
    lines.append("|---|---|---|---|---|")
    for g in allg:
        cells = []
        for a, _ in ARMS:
            v = pg[a].get(str(g)) or pg[a].get(g)
            cells.append(f(v, 4) if v else "—")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines.append("\n</details>")


if __name__ == "__main__":
    main()
