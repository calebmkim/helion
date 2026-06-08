"""Run-2 Goal-3b analyzer: BEAT-MAX-EFFORT. For ONE shape, compares the
seeded-PORTFOLIO max-effort arm vs the unseeded max-effort arm over N runs.
Per arm: per-run final best perf_ms (cumulative min of ok rows), best-of-N,
median-of-N, worst, spread. Probabilistic beat: fraction of runs within Tpct of
the best-overall optimum (P(arm run reaches ~optimum)). Lower perf_ms = better.
Usage: python run2_productB_3b_analyze.py /tmp/pb3b PREFIX  (e.g. wf4096 or sum16384)
"""
from __future__ import annotations
import csv, glob, math, re, sys
from statistics import median


def final_best(path):
    best = math.inf
    cfg = None
    try:
        for r in csv.DictReader(open(path)):
            if r.get("status") != "ok":
                continue
            p = r.get("perf_ms", "")
            if p in ("", None):
                continue
            try:
                v = float(p)
            except ValueError:
                continue
            if math.isfinite(v) and 0 < v < best:
                best = v
                cfg = r.get("config", "")
    except FileNotFoundError:
        return None, None
    return (best if best < math.inf else None), cfg


def arm(outdir, prefix, mode):
    res = []
    for f in sorted(glob.glob(f"{outdir}/{prefix}_{mode}_full_s*.csv")):
        fb, cfg = final_best(f)
        if fb:
            res.append((f, fb, cfg))
    return res


def brief_cfg(cfg):
    if not cfg:
        return "?"
    bs = re.search(r"block_sizes=(\[[^]]*\])", cfg)
    nw = re.search(r"num_warps=(\d+)", cfg)
    ev = re.search(r"load_eviction_policies=(\[[^]]*\])", cfg)
    return f"{bs.group(1) if bs else '?'} w{nw.group(1) if nw else '?'} ev={ev.group(1) if ev else '?'}"


def main():
    outdir, prefix = sys.argv[1], sys.argv[2]
    Tpct = float(sys.argv[3]) if len(sys.argv) > 3 else 1.05  # within 5% of optimum
    arms = {m: arm(outdir, prefix, m) for m in ("unseeded", "seededPF")}
    all_best = [v for rs in arms.values() for _, v, _ in rs]
    if not all_best:
        print(f"no data for {prefix}")
        return
    opt = min(all_best)
    thresh = opt * Tpct
    print(f"=== Goal-3b {prefix}: optimum(best over all runs)={opt:.4f}ms; "
          f"'reaches optimum' = perf<= {thresh:.4f}ms ({(Tpct-1)*100:.0f}% of opt) ===")
    summary = {}
    for m in ("unseeded", "seededPF"):
        rs = arms[m]
        vals = [v for _, v, _ in rs]
        if not vals:
            print(f"  {m}: (none)")
            continue
        reach = sum(1 for v in vals if v <= thresh)
        bestc = brief_cfg([c for _, v, c in rs if v == min(vals)][0])
        summary[m] = {"n": len(vals), "best": min(vals), "median": median(vals),
                      "worst": max(vals), "spread_pct": (max(vals) / min(vals) - 1) * 100,
                      "P_reach": reach / len(vals)}
        print(f"  {m:>9} (n={len(vals)}): best={min(vals):.4f} median={median(vals):.4f} "
              f"worst={max(vals):.4f} spread={ (max(vals)/min(vals)-1)*100:.1f}% "
              f"P(reach optimum)={reach}/{len(vals)}={reach/len(vals):.2f}")
        print(f"            per-run: {[round(v,4) for v in vals]}  best-cfg: {bestc}")
    u, s = summary.get("unseeded"), summary.get("seededPF")
    if u and s:
        beat = s["P_reach"] > u["P_reach"]
        print(f"  >>> BEAT? P(seeded reaches optimum)={s['P_reach']:.2f} vs "
              f"P(unseeded)={u['P_reach']:.2f}  -> {'BEAT (seeded more reliable)' if beat else 'NO BEAT (tie/both reliable)'}")


if __name__ == "__main__":
    main()
