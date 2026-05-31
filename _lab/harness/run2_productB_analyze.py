"""Run-2 Product-B analyzer. Reads the matrix CSVs (HELION_AUTOTUNE_LOG output) for
ONE shape-dir and computes, per (kernel,shape):
  - per-run best-perf-so-far vs GENERATION and vs cumulative WALL-CLOCK (2 axes).
  - median-over-reps final best for {seeded,unseeded} x {quick,full}.
  - 3a(a) BUDGET REDUCTION: does seeded-QUICK match unseeded-FULL's optimum? gap%.
  - 3a(b) CONVERGENCE: gens & wall-clock for seeded-FULL vs unseeded-FULL to reach
    95%/99% of the unseeded-FULL final optimum (lower perf_ms = better).
CSV traps handled: 2 rows/config (started w/ empty perf_ms, then ok/error); empty
strings not NaN; keep status=='ok' & finite perf_ms.
Usage: python run2_productB_analyze.py /tmp/pbpilot KERNEL M N
"""
from __future__ import annotations
import csv, sys, glob, math, os
from statistics import median


def parse_csv(path):
    """Return list of (timestamp_s, generation, perf_ms) for ok rows, time-ordered."""
    rows = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                p = r.get("perf_ms", "")
                if p in ("", None):
                    continue
                try:
                    perf = float(p); ts = float(r.get("timestamp_s") or 0)
                    gen = int(r.get("generation") or 0)
                except (ValueError, TypeError):
                    continue
                if math.isfinite(perf) and perf > 0:
                    rows.append((ts, gen, perf))
    except FileNotFoundError:
        return []
    rows.sort()
    return rows


def best_vs_wallclock(rows):
    """cumulative-min perf vs wall-clock; returns (final_best, [(ts, best)], t0)."""
    if not rows:
        return None, [], 0
    t0 = rows[0][0]
    best = math.inf; curve = []
    for ts, _g, perf in rows:
        best = min(best, perf); curve.append((ts - t0, best))
    return best, curve, t0


def best_vs_gen(rows):
    by = {}
    for _ts, g, perf in rows:
        by[g] = min(by.get(g, math.inf), perf)
    out = {}; run = math.inf
    for g in sorted(by):
        run = min(run, by[g]); out[g] = run
    return out


def time_to_target(curve, target):
    """wall-clock (s) at which cumulative-min first <= target; None if never."""
    for t, b in curve:
        if b <= target:
            return t
    return None


def gen_to_target(genbest, target):
    for g in sorted(genbest):
        if genbest[g] <= target:
            return g
    return None


def collect(outdir, kernel, M, N):
    tag = f"{kernel}_{M}x{N}"
    runs = {}  # (mode,effort) -> list of (final_best, curve, genbest, total_wall)
    for path in sorted(glob.glob(f"{outdir}/{tag}_*.csv")):
        base = os.path.basename(path)[:-4]
        parts = base.split("_")
        # ..._<mode>_<effort>_s<seed>
        seed = parts[-1]; effort = parts[-2]; mode = parts[-3]
        rows = parse_csv(path)
        fb, curve, _ = best_vs_wallclock(rows)
        if fb is None:
            continue
        gb = best_vs_gen(rows)
        total_wall = curve[-1][0] if curve else 0
        runs.setdefault((mode, effort), []).append((fb, curve, gb, total_wall))
    return tag, runs


def main():
    outdir, kernel, M, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    tag, runs = collect(outdir, kernel, M, N)
    print(f"=== Product B: {tag} ===")

    def med_final(key):
        rs = runs.get(key, [])
        return median([r[0] for r in rs]) if rs else None

    sf = med_final(("seeded", "full")); uf = med_final(("unseeded", "full"))
    sq = med_final(("seeded", "quick")); uq = med_final(("unseeded", "quick"))
    print(f"median final best_perf_ms: seeded-quick={sq} unseeded-quick={uq} "
          f"seeded-full={sf} unseeded-full={uf}")

    # 3a(a) budget reduction: seeded-QUICK vs unseeded-FULL optimum
    if sq and uf:
        gap = (sq / uf - 1) * 100
        print(f"[3a-a BUDGET] seeded-QUICK / unseeded-FULL = {sq/uf:.3f} "
              f"(gap {gap:+.1f}%); seeded-quick {'MATCHES/BEATS' if sq<=uf*1.02 else 'below'} "
              f"unseeded-full optimum -> budget saved = full->quick")

    # 3a(b) convergence to 95/99% of unseeded-FULL final
    if uf:
        for pct in (0.95, 0.99):
            target = uf / pct  # perf_ms target (higher perf_ms allowed = pct of speed)
            # NOTE perf_ms lower is better; "95% of optimum speed" = perf_ms <= uf/0.95
            print(f"  -- reach {int(pct*100)}% of unseeded-full optimum (perf<= {target:.4f}ms):")
            for mode in ("seeded", "unseeded"):
                rs = runs.get((mode, "full"), [])
                if not rs:
                    continue
                tts = [time_to_target(r[1], target) for r in rs]
                gts = [gen_to_target(r[2], target) for r in rs]
                tts = [t for t in tts if t is not None]; gts = [g for g in gts if g is not None]
                mt = median(tts) if tts else None; mg = median(gts) if gts else None
                print(f"     {mode}-full: median wall={mt and round(mt,1)}s  median gen={mg}  "
                      f"(n={len(rs)})")
    # gen0 seed advantage (full)
    for mode in ("seeded", "unseeded"):
        rs = runs.get((mode, "full"), [])
        g0 = [r[2].get(0) for r in rs if r[2].get(0)]
        if g0:
            print(f"  gen0 median best ({mode}-full) = {median(g0):.4f}ms (n={len(g0)})")


if __name__ == "__main__":
    main()
