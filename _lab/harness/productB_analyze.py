"""Product B analyzer: read convergence CSVs, compute Slice 1 + Slice 2.

Reads logs/productB/<kernel>_<M>_<N>_<mode>_s<seed>.csv and the matching
.driver.log (for the seed-injection verification). Computes, off the SAME curve:

  Curve construction (per run, parsing traps handled):
    - keep rows with status=='ok' AND a finite, parseable perf_ms
    - skip 'started' rows (empty perf_ms) and 'error' rows
    - missing fields are EMPTY STRINGS, parsed defensively
    - best-vs-generation = groupby(generation).min(perf_ms), then cumulative-min
      across generations (monotone best-so-far)
    - best-vs-wallclock = cumulative-min of perf_ms ordered by timestamp_s
    - per-generation cumulative wall-clock = max timestamp_s seen up to & incl gen

  Slice 1 (same-budget perf): seeded vs unseeded best-perf-so-far at
    gen<=1 (small budget), gen<=2, and gen<=5 (full budget = no-regression
    guardrail). Lower perf_ms is better. Report seeded/unseeded speedup ratio
    (>1 => seeded better) and median+spread across random seeds.

  Slice 2 (time-to-target, HEADLINE): target = X% of the UNSEEDED full-budget
    best perf (median across seeds). Wall-clock for SEEDED to first reach the
    target, vs UNSEEDED. Report median+spread; the time win = seeded reaches
    target sooner.

Usage: productB_analyze.py [--target-pct 95] [--dir logs/productB]
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import statistics
from collections import defaultdict

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
DEFAULT_DIR = os.path.join(WT, "logs", "productB")
FULL_GEN = 5  # quick profile max_generations


def parse_float(s):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def parse_int(s):
    if s is None:
        return None
    s = s.strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def load_run(csv_path):
    """Return list of dicts {gen, ts, perf_ms, config} for status=='ok' rows."""
    pts = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status", "").strip() != "ok":
                continue
            perf = parse_float(r.get("perf_ms"))
            if perf is None:
                continue
            gen = parse_int(r.get("generation"))
            ts = parse_float(r.get("timestamp_s"))
            pts.append({"gen": gen if gen is not None else 0,
                        "ts": ts if ts is not None else 0.0,
                        "perf": perf, "config": r.get("config", "")})
    return pts


def best_by_gen_cummin(pts):
    """best-so-far perf at each generation index (cumulative min over gens)."""
    by_gen = defaultdict(list)
    for p in pts:
        by_gen[p["gen"]].append(p["perf"])
    gens = sorted(by_gen)
    out = {}
    run_best = math.inf
    for g in gens:
        run_best = min(run_best, min(by_gen[g]))
        out[g] = run_best
    return out  # {gen: best_so_far_perf}


def best_at_budget(pts, max_gen):
    """best perf among all ok configs with generation <= max_gen."""
    vals = [p["perf"] for p in pts if p["gen"] <= max_gen]
    return min(vals) if vals else math.inf


def wallclock_to_perf(pts, target_perf):
    """earliest timestamp_s at which best-so-far (by wallclock) <= target_perf."""
    ordered = sorted(pts, key=lambda p: p["ts"])
    run_best = math.inf
    for p in ordered:
        run_best = min(run_best, p["perf"])
        if run_best <= target_perf:
            return p["ts"]
    return None  # never reached


def total_wallclock(pts):
    return max((p["ts"] for p in pts), default=0.0)


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def med_spread(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (None, None, None)
    return (statistics.median(xs), min(xs), max(xs))


def shape_key(fname):
    # <kernel>_<M>_<N>_<mode>_s<seed>.csv
    m = re.match(r"(.+)_(\d+)_(\d+)_(seeded|unseeded)_s(\d+)\.csv$",
                 os.path.basename(fname))
    if not m:
        return None
    return {"kernel": m.group(1), "M": int(m.group(2)), "N": int(m.group(3)),
            "mode": m.group(4), "seed": int(m.group(5))}


def seed_verify(driver_log):
    """Pull the seed-injection verification facts from the .driver.log."""
    info = {}
    if not os.path.exists(driver_log):
        return info
    txt = open(driver_log).read()
    for line in txt.splitlines():
        if "config_spec.compiler_seed_configs (n=" in line:
            info["n_seeds"] = parse_int(re.search(r"n=(\d+)", line).group(1))
        if "SEED-FLAT-ENCODE" in line:
            info["flat_encode"] = line.split("SEED-FLAT-ENCODE:")[1].strip()
        if "autotuner_heuristics:" in line:
            info["heuristics"] = line.split("autotuner_heuristics:")[1].strip()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--target-pct", type=float, default=95.0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.csv")))
    # group by (kernel,M,N)
    shapes = defaultdict(lambda: {"seeded": {}, "unseeded": {}})
    for fn in files:
        k = shape_key(fn)
        if k is None:
            continue
        shapes[(k["kernel"], k["M"], k["N"])][k["mode"]][k["seed"]] = fn

    print(f"# Product B analysis  target={a.target_pct}% of unseeded full-budget")
    print(f"# dir={a.dir}  full_gen={FULL_GEN}\n")

    summary = []
    for (kern, M, N), modes in sorted(shapes.items()):
        print(f"================ {kern} ({M},{N})  rnumel={N} ================")
        # ---- seed-injection verification (from one seeded run's driver log) ----
        any_seeded = next(iter(modes["seeded"].values()), None)
        if any_seeded:
            v = seed_verify(any_seeded.replace(".csv", ".driver.log"))
            print(f"  [verify] seeded n_seeds={v.get('n_seeds')} "
                  f"heuristics={v.get('heuristics')}")
            print(f"  [verify] {v.get('flat_encode','(no flat-encode line)')}")
        any_uns = next(iter(modes["unseeded"].values()), None)
        if any_uns:
            vu = seed_verify(any_uns.replace(".csv", ".driver.log"))
            print(f"  [verify] unseeded n_seeds={vu.get('n_seeds')} "
                  f"heuristics={vu.get('heuristics')}")

        # ---- load runs ----
        runs = {"seeded": {}, "unseeded": {}}
        for mode in ("seeded", "unseeded"):
            for s, fn in sorted(modes[mode].items()):
                runs[mode][s] = load_run(fn)

        # ---- Slice 1: same-budget best perf (ms) ----
        budgets = [1, 2, FULL_GEN]
        print(f"\n  -- Slice 1: best perf_ms at budget (median[min,max] across "
              f"seeds); ratio = unseeded/seeded (>1 => seeded faster) --")
        s1 = {}
        for b in budgets:
            seeded_vals = [best_at_budget(runs["seeded"][s], b)
                           for s in runs["seeded"]]
            uns_vals = [best_at_budget(runs["unseeded"][s], b)
                        for s in runs["unseeded"]]
            sm = med_spread(seeded_vals)
            um = med_spread(uns_vals)
            ratio = (um[0] / sm[0]) if (sm[0] and um[0]
                                        and math.isfinite(sm[0])
                                        and math.isfinite(um[0])) else None
            s1[b] = {"seeded": sm, "unseeded": um, "ratio": ratio}
            rstr = f"{ratio:.3f}x" if ratio else "n/a"
            print(f"     gen<= {b}: seeded {sm[0]:.4f}[{sm[1]:.4f},{sm[2]:.4f}]  "
                  f"unseeded {um[0]:.4f}[{um[1]:.4f},{um[2]:.4f}]  "
                  f"seeded-advantage={rstr}")

        # ---- Slice 2: time-to-target (wall-clock) ----
        # target = target_pct of unseeded full-budget best (median across seeds)
        uns_full = [best_at_budget(runs["unseeded"][s], FULL_GEN)
                    for s in runs["unseeded"]]
        uns_full_med = med_spread(uns_full)[0]
        target = uns_full_med / (a.target_pct / 100.0)  # perf_ms threshold
        print(f"\n  -- Slice 2 (HEADLINE): time-to-target. "
              f"unseeded full-budget median={uns_full_med:.4f}ms; "
              f"target(reach {a.target_pct}% of it) = perf<= {target:.4f}ms --")
        t_seeded, t_uns = [], []
        for s in runs["seeded"]:
            t = wallclock_to_perf(runs["seeded"][s], target)
            t_seeded.append(t)
        for s in runs["unseeded"]:
            t = wallclock_to_perf(runs["unseeded"][s], target)
            t_uns.append(t)
        sm = med_spread([t for t in t_seeded if t is not None])
        um = med_spread([t for t in t_uns if t is not None])
        n_s_reach = sum(1 for t in t_seeded if t is not None)
        n_u_reach = sum(1 for t in t_uns if t is not None)
        print(f"     seeded   reached {n_s_reach}/{len(t_seeded)}: "
              f"wall-clock median={fmt(sm[0])}s [{fmt(sm[1])},{fmt(sm[2])}]")
        print(f"     unseeded reached {n_u_reach}/{len(t_uns)}: "
              f"wall-clock median={fmt(um[0])}s [{fmt(um[1])},{fmt(um[2])}]")
        speedup = (um[0] / sm[0]) if (sm[0] and um[0]) else None
        print(f"     time-to-target speedup (unseeded_t/seeded_t) = "
              f"{fmt(speedup)}x  "
              f"(>1 => seeded reaches target sooner)")
        # also report total wall-clock per run (full search cost)
        tot_s = med_spread([total_wallclock(runs["seeded"][s])
                            for s in runs["seeded"]])
        tot_u = med_spread([total_wallclock(runs["unseeded"][s])
                            for s in runs["unseeded"]])
        print(f"     [context] total full-search wall-clock: "
              f"seeded median={fmt(tot_s[0])}s, unseeded median={fmt(tot_u[0])}s")

        summary.append({
            "shape": f"{kern}({M},{N})",
            "s1": s1, "target_ms": target, "uns_full_med": uns_full_med,
            "t_seeded": sm, "t_uns": um, "tt_speedup": speedup,
            "n_s_reach": (n_s_reach, len(t_seeded)),
            "n_u_reach": (n_u_reach, len(t_uns)),
            "tot_s": tot_s[0], "tot_u": tot_u[0],
        })
        print()

    # ---- compact roll-up ----
    print("\n================ ROLL-UP ================")
    print(f"{'shape':22} {'S1 gen1':>9} {'S1 gen2':>9} {'S1 full':>9} "
          f"{'S2 t-to-tgt':>22}")
    for r in summary:
        g1 = r["s1"][1]["ratio"]; g2 = r["s1"][2]["ratio"]; gf = r["s1"][FULL_GEN]["ratio"]
        print(f"{r['shape']:22} {fmt(g1):>8}x {fmt(g2):>8}x {fmt(gf):>8}x "
              f"  seeded {fmt(r['t_seeded'][0])}s vs uns {fmt(r['t_uns'][0])}s "
              f"({fmt(r['tt_speedup'])}x)")


def fmt(x):
    if x is None:
        return "n/a"
    return f"{x:.3f}"


if __name__ == "__main__":
    main()
