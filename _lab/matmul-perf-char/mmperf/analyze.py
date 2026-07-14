"""Analyze results.jsonl -> structured aggregates for the summary. Pure reader:
every number is derived from the raw per-round arrays, nothing stored-in-place.

Emits analysis.json with: per-cell rows, per-type geomeans, all-in geomean,
aligned-vs-adversarial split, default_source (table|base) split, M1-vs-M2 deltas,
mamba (separate), and the default-failure surface.
"""

from __future__ import annotations

import json
import statistics
import sys

RESULTS = "/home/dev/helion-matmul-b200/_lab/matmul-perf-char/results/results.jsonl"
OUT = "/home/dev/helion-matmul-b200/_lab/matmul-perf-char/results/analysis.json"

# adversarial (honesty / hard) shape types vs aligned-friendly
ADVERSARIAL_TYPES = {
    "non_pow2", "non_aligned", "tile_tail", "prime", "deep_K", "small_K",
    "tall_skinny", "tall_skinny_extreme", "wide", "decode",
}


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0 and x == x]
    return statistics.geometric_mean(xs) if xs else None


def load():
    hdr = None
    recs = []
    for l in open(RESULTS):
        r = json.loads(l)
        if r.get("_run_header"):
            hdr = r
            continue
        if r.get("_run_footer"):
            continue
        if r.get("method", "").startswith("M"):
            recs.append(r)
    return hdr, recs


def arm_med(rec, arm):
    a = rec.get("arms", {}).get(arm, {})
    if a.get("status") == "ok" and "median" in a:
        return a["median"]
    return None


def main():
    hdr, recs = load()
    by_method = {"M1_cudagraph_coldL2": [], "M2_do_bench_coldL2": []}
    for r in recs:
        by_method[r["method"]].append(r)

    analysis = {"header": hdr, "cells": [], "by_method": {}}

    # per-cell rows (keyed on M1 canonical; carry M2 medians for delta)
    m1 = {(_r["kernel"], tuple(_r["shape"])): _r for _r in by_method["M1_cudagraph_coldL2"]}
    m2 = {(_r["kernel"], tuple(_r["shape"])): _r for _r in by_method["M2_do_bench_coldL2"]}

    for key, r in m1.items():
        r2 = m2.get(key, {})
        row = {
            "kernel": r["kernel"],
            "shape": r["shape"],
            "type": r["type"],
            "prov": r.get("prov"),
            "dtype": r["dtype"],
            "default_source": r["default_source"],
            "seed_fired": r["seed_fired"],
            "seed_cfg": r["seed_cfg"],
            "default_cfg": r["default_cfg"],
            "acc_pass": r.get("acc_pass"),
            "max_rel": r.get("max_rel"),
            "tflops_seed_M1": r.get("tflops_seed"),
            "pct_peak_seed_M1": r.get("pct_peak_seed"),
            "R": r.get("R"),
            "noise_floor": r.get("noise_floor"),
            "m1": {
                "seed_us": arm_med(r, "seed"),
                "default_us": arm_med(r, "helion_default"),
                "tc_us": arm_med(r, "tc_max_autotune"),
                "G_vs_tc": r["ratios"].get("G_vs_tc", {}).get("median"),
                "G_vs_tc_ci": r["ratios"].get("G_vs_tc", {}).get("ci"),
                "xD_vs_default": r["ratios"].get("xD_vs_default", {}).get("median"),
            },
            "m2": {
                "seed_us": arm_med(r2, "seed"),
                "default_us": arm_med(r2, "helion_default"),
                "tc_us": arm_med(r2, "tc_max_autotune"),
                "G_vs_tc": r2.get("ratios", {}).get("G_vs_tc", {}).get("median"),
                "xD_vs_default": r2.get("ratios", {}).get("xD_vs_default", {}).get("median"),
            },
            "tc_winner": r.get("arms", {}).get("tc_max_autotune", {}).get("winner"),
        }
        analysis["cells"].append(row)

    cells = analysis["cells"]

    def agg(rows, method="m1"):
        gvt = [c[method]["G_vs_tc"] for c in rows]
        xd = [c[method]["xD_vs_default"] for c in rows]
        return {
            "n": len(rows),
            "geo_G_vs_tc": geomean(gvt),
            "min_G_vs_tc": min([x for x in gvt if x], default=None),
            "max_G_vs_tc": max([x for x in gvt if x], default=None),
            "geo_xD_vs_default": geomean(xd),
        }

    # split helpers
    gemm_cells = [c for c in cells if c["kernel"] in ("matmul", "fp8_gemm", "bmm")]
    matmul_cells = [c for c in cells if c["kernel"] == "matmul"]
    fp8_cells = [c for c in cells if c["kernel"] == "fp8_gemm"]
    bmm_cells = [c for c in cells if c["kernel"] == "bmm"]
    mamba_cells = [c for c in cells if c["kernel"] == "mamba2_chunk_state"]

    summary = {}
    # per-kernel
    for name, rows in [("matmul", matmul_cells), ("fp8_gemm", fp8_cells),
                       ("bmm", bmm_cells), ("mamba2_chunk_state", mamba_cells)]:
        summary[name] = {"m1": agg(rows, "m1"), "m2": agg(rows, "m2")}

    # all-in GEMM (matmul+fp8+bmm; mamba separate — no cuBLAS analog)
    summary["all_gemm"] = {"m1": agg(gemm_cells, "m1"), "m2": agg(gemm_cells, "m2")}
    summary["mamba_only"] = {"m1": agg(mamba_cells, "m1"), "m2": agg(mamba_cells, "m2")}

    # per-type geomean (GEMM only, over acc-passing cells)
    by_type = {}
    for c in gemm_cells:
        if c["acc_pass"] is False:
            continue
        by_type.setdefault(c["type"], []).append(c)
    summary["by_type_gemm"] = {t: agg(rows, "m1") for t, rows in sorted(by_type.items())}

    # aligned vs adversarial (GEMM)
    aligned = [c for c in gemm_cells if c["type"] not in ADVERSARIAL_TYPES]
    adversarial = [c for c in gemm_cells if c["type"] in ADVERSARIAL_TYPES]
    summary["aligned_gemm"] = agg(aligned, "m1")
    summary["adversarial_gemm"] = agg(adversarial, "m1")

    # default_source split (xD_vs_default): table (head-to-head) vs base (coverage)
    table_cells = [c for c in cells if c["default_source"] == "table"]
    base_cells = [c for c in cells if c["default_source"] == "base"]
    summary["default_source_split"] = {
        "table_headtohead": {
            "n": len(table_cells),
            "geo_xD_vs_default": geomean([c["m1"]["xD_vs_default"] for c in table_cells]),
            "min_xD": min([c["m1"]["xD_vs_default"] for c in table_cells if c["m1"]["xD_vs_default"]], default=None),
            "max_xD": max([c["m1"]["xD_vs_default"] for c in table_cells if c["m1"]["xD_vs_default"]], default=None),
            "cells": [{"kernel": c["kernel"], "shape": c["shape"], "type": c["type"],
                       "xD": c["m1"]["xD_vs_default"], "G_vs_tc": c["m1"]["G_vs_tc"]}
                      for c in table_cells],
        },
        "base_coverage": {
            "n": len(base_cells),
            "geo_xD_vs_default": geomean([c["m1"]["xD_vs_default"] for c in base_cells]),
            "min_xD": min([c["m1"]["xD_vs_default"] for c in base_cells if c["m1"]["xD_vs_default"]], default=None),
            "max_xD": max([c["m1"]["xD_vs_default"] for c in base_cells if c["m1"]["xD_vs_default"]], default=None),
        },
    }

    # M1 vs M2 divergence (per cell |G_M2/G_M1 - 1|), flag short kernels
    div = []
    for c in cells:
        g1, g2 = c["m1"]["G_vs_tc"], c["m2"]["G_vs_tc"]
        s1 = c["m1"]["seed_us"]
        if g1 and g2:
            div.append({"kernel": c["kernel"], "shape": c["shape"], "seed_us": s1,
                        "G_M1": g1, "G_M2": g2, "rel_delta": g2 / g1 - 1.0,
                        "noise_floor": c["noise_floor"]})
    div.sort(key=lambda d: -abs(d["rel_delta"]))
    summary["m1_vs_m2_divergence_top"] = div[:12]
    summary["m1_vs_m2_median_abs_delta"] = statistics.median([abs(d["rel_delta"]) for d in div]) if div else None

    # default failure surface (none expected on B200, but report)
    fails = []
    for r in by_method["M1_cudagraph_coldL2"]:
        for arm in ["seed", "helion_default", "tc_max_autotune"]:
            a = r.get("arms", {}).get(arm, {})
            if a.get("status") not in ("ok", None):
                fails.append({"kernel": r["kernel"], "shape": r["shape"], "arm": arm,
                              "status": a.get("status"), "error": a.get("error")})
    summary["arm_failures"] = fails

    analysis["summary"] = summary

    with open(OUT, "w") as fh:
        json.dump(analysis, fh, indent=2)

    # console digest
    def fmt(x, p=3):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "n/a"

    print("=== DEVICE ===", hdr["device"], hdr["sm_tag"], "L2", hdr["l2_mib"], "flush", hdr["flush_mib"])
    print("\n=== per-kernel geomean (M1) ===")
    for k in ["matmul", "fp8_gemm", "bmm", "mamba2_chunk_state"]:
        s = summary[k]["m1"]
        print(f"  {k:20s} n={s['n']:2d} G_vs_tc geo={fmt(s['geo_G_vs_tc'])} "
              f"[{fmt(s['min_G_vs_tc'])},{fmt(s['max_G_vs_tc'])}] xD_vs_default geo={fmt(s['geo_xD_vs_default'],2)}")
    print(f"\n  ALL-GEMM (mm+fp8+bmm) G_vs_tc geo={fmt(summary['all_gemm']['m1']['geo_G_vs_tc'])}")
    print(f"  MAMBA (vs Triton tc)  G_vs_tc geo={fmt(summary['mamba_only']['m1']['geo_G_vs_tc'])}")
    print("\n=== default_source split (xD_vs_default, M1) ===")
    t = summary["default_source_split"]["table_headtohead"]
    b = summary["default_source_split"]["base_coverage"]
    print(f"  TABLE head-to-head:  n={t['n']} geo_xD={fmt(t['geo_xD_vs_default'],3)} [{fmt(t['min_xD'],2)},{fmt(t['max_xD'],2)}]")
    print(f"  BASE  coverage win:  n={b['n']} geo_xD={fmt(b['geo_xD_vs_default'],2)} [{fmt(b['min_xD'],2)},{fmt(b['max_xD'],2)}]")
    print("\n=== aligned vs adversarial (GEMM, M1 G_vs_tc) ===")
    print(f"  aligned:     n={summary['aligned_gemm']['n']} geo={fmt(summary['aligned_gemm']['geo_G_vs_tc'])}")
    print(f"  adversarial: n={summary['adversarial_gemm']['n']} geo={fmt(summary['adversarial_gemm']['geo_G_vs_tc'])}")
    print("\n=== M1 vs M2 ===")
    print(f"  median |G_M2/G_M1 - 1| = {fmt(summary['m1_vs_m2_median_abs_delta'])}")
    print("  top divergences:")
    for d in summary["m1_vs_m2_divergence_top"][:6]:
        print(f"    {d['kernel']} {d['shape']} seed={fmt(d['seed_us'],1)}us "
              f"G_M1={fmt(d['G_M1'])} G_M2={fmt(d['G_M2'])} delta={fmt(d['rel_delta']*100,1)}% nf={d['noise_floor']}")
    print("\n=== arm failures:", summary["arm_failures"] if summary["arm_failures"] else "NONE")
    print(f"\nwrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
