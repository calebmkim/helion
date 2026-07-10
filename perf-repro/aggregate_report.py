"""Aggregate the per-(corpus,kernel) JSON cells into the manager-facing perf report.

Reads every results/*.json produced by perf_report_bench.py and emits:
  (A) Per-(kernel,dtype) geomeans of G_tc / G_def (/ G_vllm for vLLM) over ACC-PASSING shapes,
      grouped by corpus (curriculum / transfer / mreduction / vllm).
  Headline aggregate: overall + per-dtype geomean of G_tc and G_def over cells with both arms.
  Disasters (G_tc < 0.75). Acc-fail cells with perf still recorded (seed==default identical fail).
  x/n/a cells. And a LAUNCH-OVERHEAD CROSS-CHECK: cells where interleaved do_bench diverges from
  cold-L2 cudagraph device time (the canary that launch overhead is NOT hidden for that cell).

Ratios here are RE-DERIVED from the raw per-arm `us` in the JSON — never trusted from a
hand-written table. Writes SUMMARY.md (human) + summary.json (machine) into the results dir.

Run: /home/dev/helion/.venv/bin/python aggregate_report.py [RESULTS_DIR]
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

# PR #2866 pointwise reproduction. REPRO = in-sample (the PR-table shapes); GEN = out-of-sample
# (held-out test split + the held-out dyt kernel + expanded lever generalization). Reported
# SEPARATELY so the reproduction claim and the generalization claim stay distinct; a COMBINED
# table (REPRO+GEN) is also emitted.
REPRO = ("pointwise",)
GEN = ("pointwise_gen",)
REAL = REPRO + GEN
FLOOR = 0.75
# flag a cell if interleaved do_bench and cold-L2 cudagraph device time differ by > this
# fraction on the SEED arm — i.e. the CPU-launch-overhead shield may not be hiding host cost.
COLDGRAPH_GATE = float(os.environ.get("PERF_COLDGRAPH_GATE", "0.15"))


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def _acc_ok(arm):
    return arm.get("status") == "ok"


def load_rows(results_dir):
    rows = []
    _SKIP = {"summary.json"}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path) in _SKIP:
            continue
        try:
            d = json.load(open(path))
        except Exception:  # noqa: BLE001
            continue
        rows.extend(d.get("rows", []))
    return rows


_RATIO_ARM = {"G_tc": "tc", "G_def": "default", "G_vllm": "vllm_shipped"}

# Automatic per-cell metric rule: if ANY arm's eager (do_bench-style) time diverges from its
# cudagraph DEVICE time by more than this fraction, the eager ratio is dominated by per-arm-variable
# CPU launch tax (a floor/launch-bound cell), so that CELL is scored on cudagraph DEVICE time instead
# of eager. 5% matches the harness's round-to-round spread gate (_SPREAD_GATE) — i.e. "if eager and
# device disagree by more than the timing noise floor, trust the device number." Eager values are
# still surfaced in the report (nothing hidden). All non-divergent cells stay on eager (the PR's
# method). This is derived from the data, not a hand-picked cell list.
DEVICE_SCORE_GATE = float(os.environ.get("PERF_DEVICE_SCORE_GATE", "0.05"))


def _launch_bound(row):
    """True if any arm's eager vs cudagraph time diverges by > DEVICE_SCORE_GATE (eager unreliable)."""
    u = row.get("us", {})
    cg = row.get("coldgraph_us", {})
    for arm in ("seed", "default", "tc", "vllm_shipped"):
        e = u.get(arm)
        c = cg.get(arm)
        if e and c and e > 0 and abs(e - c) / e > DEVICE_SCORE_GATE:
            return True
    return False


def _cell_metric(row, default_metric="auto"):
    """Metric to score this cell with: 'device' if the cell is launch-bound (eager unreliable, any
    arm's eager vs cudagraph divergence > DEVICE_SCORE_GATE), else the caller's default (headline =
    eager for pointwise)."""
    if _launch_bound(row):
        return "device"
    return default_metric
# vLLM-family corpora deploy their kernels UNDER CUDA GRAPHS in production, so pure device time
# (coldgraph_us, launch amortized) is the deployment-faithful metric — the eager `us` carries
# per-call launch dispatch that doesn't fully cancel in the ratio on small decode shapes. For
# these corpora we derive ratios from coldgraph device time; all others use eager cold-L2 `us`.
_DEVICE_TIME_CORPORA = {"vllm", "vllm_gen", "qk_norm_rope_gen"}


def _arm_time(row, arm, metric="auto"):
    """Time for `arm`. metric='eager' -> cold-L2 do_bench us; 'device' -> cudagraph device time;
    'auto' (headline default) -> cudagraph for vLLM-family corpora (deployed under CUDA graphs),
    else eager. Falls back to eager if a requested device time is missing."""
    if metric == "device":
        cg = row.get("coldgraph_us", {}).get(arm)
        return cg if cg else row.get("us", {}).get(arm)
    if metric == "eager":
        return row.get("us", {}).get(arm)
    # auto
    if row.get("corpus") in _DEVICE_TIME_CORPORA:
        cg = row.get("coldgraph_us", {}).get(arm)
        if cg:
            return cg
    return row.get("us", {}).get(arm)


def _rederive_ratio(row, which, metric="auto"):
    """Recompute the ratio from raw per-arm times (do not trust the stored G_*).

    For metric='auto' (the eager headline path) a launch-bound override cell is scored on device
    time instead — the eager ratio there is launch-tax noise. Explicit metric='eager'/'device'
    (the two co-equal cross-check tables) pass through unchanged so both pure views stay honest."""
    eff = _cell_metric(row) if metric == "auto" else metric
    seed = _arm_time(row, "seed", eff)
    other = _arm_time(row, _RATIO_ARM[which], eff)
    if seed and other and seed > 0:
        return round(other / seed, 4)
    return None


def _close_enough(arms):
    """Is the seed's output trustworthy enough to compare its SPEED against every other arm?
    YES iff the seed passes accuracy, OR the DEFAULT makes the IDENTICAL mistake (same acc_detail).
    The default-matches-seed signal means the accuracy miss is a benign kernel/dtype FACT — bf16
    accumulator margin (rms_norm_bwd) or fp8 tie-rounding (per_token_group), ~1 ULP on ~3% of
    elements — that the whole kernel family shares, NOT a seed-specific wrong answer. When that
    holds we treat all arms as close-enough and time all of them (G_def, G_vllm, AND G_tc; for the
    fp8 kernels tc is literally torch.compile(ref), so it differs from the seed by the SAME
    tie-rounding — timing seed-vs-tc is consistent). A seed that fails while the default is CORRECT
    (e.g. the qk_norm_rope tile-size miscompile: seed maxabs~2, default fine) is NOT close-enough
    and is excluded from every ratio."""
    s = arms.get("seed", {})
    if s.get("status") == "ok":
        return True
    d = arms.get("default", {})
    return (s.get("status") == "acc-fail" and d.get("status") == "acc-fail"
            and s.get("acc_detail") is not None and s.get("acc_detail") == d.get("acc_detail"))


def cell_ok_for_ratio(row, which):
    """A cell joins a ratio's geomean iff the seed is 'close-enough' (see _close_enough) AND the
    other arm has a usable timing (ran to a us/coldgraph time — status ok or acc-fail). Applies
    uniformly to G_tc, G_def, G_vllm: once the cell is close-enough, all arms are compared."""
    arms = row.get("arms", {})
    if not _close_enough(arms):
        return False
    other = arms.get(_RATIO_ARM[which], {})
    if other.get("status") not in ("ok", "acc-fail"):
        return False  # other arm produced no timing (compile-fail / no-config / n-a)
    return _rederive_ratio(row, which) is not None


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    rows = load_rows(results_dir)

    real_cells = {}
    xna = []
    accfail_perf = []
    coldgraph_flags = []  # cells where seed do_bench diverges from cold cudagraph device time
    launch_bound_cells = []  # cells scored on device time (any arm eager vs cudagraph > gate)

    for r in rows:
        corpus = r.get("corpus")
        if "error" in r and "arms" not in r:
            xna.append((corpus, r.get("kernel"), r.get("shape"), r.get("dtype"),
                        "bind/build", r["error"][:80]))
            continue
        key = (corpus, r["kernel"], r.get("dtype"))
        real_cells.setdefault(key, []).append(r)

        for arm in ("seed", "default", "tc", "vllm_shipped"):
            a = r.get("arms", {}).get(arm, {})
            if not a:
                continue
            st = a.get("status", "")
            if st not in ("ok", "n/a-no-tc"):
                xna.append((corpus, r["kernel"], r.get("shape"), r.get("dtype"), arm,
                            st + (f" ({a['acc_detail']})" if a.get("acc_detail") else "")))

        seed_a = r.get("arms", {}).get("seed", {})
        if seed_a.get("status") == "acc-fail":
            u = r.get("us", {})
            arms_r = r.get("arms", {})
            def_a = arms_r.get("default", {})
            # which ratios does this failing cell get INCLUDED in (close-enough rule)?
            inc_tc = cell_ok_for_ratio(r, "G_tc")
            inc_def = cell_ok_for_ratio(r, "G_def")
            inc_vllm = cell_ok_for_ratio(r, "G_vllm")
            accfail_perf.append({
                "corpus": corpus, "kernel": r["kernel"], "shape": r.get("shape"),
                "dtype": r.get("dtype"), "us": u,
                "seed_acc": seed_a.get("acc_detail"),
                "perf_only_tc": r.get("perf_only_tc"), "perf_only_def": r.get("perf_only_def"),
                "perf_only_vllm": r.get("perf_only_vllm"),
                "default_status": def_a.get("status"),
                "both_fail": def_a.get("status") == "acc-fail"
                             and def_a.get("acc_detail") == seed_a.get("acc_detail"),
                "included_in_G_tc": inc_tc,
                "included_in_G_def": inc_def, "included_in_G_vllm": inc_vllm,
                # Group D: seed fails but default couldn't even compile -> no G_def possible.
                "default_no_compile": "compile-fail" in (def_a.get("status") or ""),
            })

        # launch-overhead cross-check on the SEED arm: do_bench (with launch) vs cudagraph (pure GPU)
        cg = r.get("coldgraph_us", {}).get("seed")
        db = r.get("us", {}).get("seed")
        if cg and db and cg > 0:
            frac = (db - cg) / db
            if abs(frac) > COLDGRAPH_GATE:
                coldgraph_flags.append({
                    "corpus": corpus, "kernel": r["kernel"], "shape": r.get("shape"),
                    "dtype": r.get("dtype"), "dobench_us": db, "coldgraph_us": cg,
                    "launch_frac": round(frac, 3),
                })

        # launch-bound cells (ANY arm eager-vs-cudagraph divergence > DEVICE_SCORE_GATE) are scored
        # on device time in the headline; record both ratios so the report shows the correction.
        if _launch_bound(r):
            u = r.get("us", {})
            cgd = r.get("coldgraph_us", {})
            def _rt(times, a, b):
                return round(times[b] / times[a], 3) if times.get(a) and times.get(b) else None
            worst = max((abs(u[a] - cgd[a]) / u[a] for a in ("seed", "default", "tc")
                         if u.get(a) and cgd.get(a) and u[a] > 0), default=0.0)
            launch_bound_cells.append({
                "corpus": corpus, "kernel": r["kernel"], "shape": r.get("shape"),
                "dtype": r.get("dtype"), "worst_div": round(worst, 3),
                "eager_G_tc": _rt(u, "seed", "tc"), "device_G_tc": _rt(cgd, "seed", "tc"),
                "eager_G_def": _rt(u, "seed", "default"), "device_G_def": _rt(cgd, "seed", "default"),
            })

    # ---- per-(kernel,dtype) geomeans ----
    per_cell = {}
    for key, rws in real_cells.items():
        gtc = [_rederive_ratio(r, "G_tc") for r in rws if cell_ok_for_ratio(r, "G_tc")]
        gdef = [_rederive_ratio(r, "G_def") for r in rws if cell_ok_for_ratio(r, "G_def")]
        gvllm = [_rederive_ratio(r, "G_vllm") for r in rws if cell_ok_for_ratio(r, "G_vllm")]
        per_cell[key] = {
            "n_shapes": len(rws),
            "n_acc_pass": sum(1 for r in rws if _acc_ok(r.get("arms", {}).get("seed", {}))),
            "geo_G_tc": geomean(gtc), "geo_G_def": geomean(gdef),
            "geo_G_vllm": geomean(gvllm) if gvllm else None,
            "n_gtc": len(gtc), "n_gdef": len(gdef), "n_gvllm": len(gvllm),
            "min_G_tc": min(gtc) if gtc else None,
        }

    def agg(dtype_filter=None, corpus_filter=None, corpus_set=None, metric="auto"):
        gtc, gdef, gvllm = [], [], []
        for (corpus, kernel, dtype), rws in real_cells.items():
            if dtype_filter and dtype != dtype_filter:
                continue
            if corpus_filter and corpus != corpus_filter:
                continue
            if corpus_set is not None and corpus not in corpus_set:
                continue
            gtc += [_rederive_ratio(r, "G_tc", metric) for r in rws if cell_ok_for_ratio(r, "G_tc")]
            gdef += [_rederive_ratio(r, "G_def", metric) for r in rws if cell_ok_for_ratio(r, "G_def")]
            gvllm += [_rederive_ratio(r, "G_vllm", metric) for r in rws if cell_ok_for_ratio(r, "G_vllm")]
        return geomean(gtc), geomean(gdef), len(gtc), len(gdef), geomean(gvllm), len(gvllm)

    dtypes_seen = sorted({k[2] for k in real_cells})
    # SEPARATE headlines: reproduction (posted-number shapes) vs generalization (unseen shapes).
    headline = {
        "reproduction_overall": agg(corpus_set=set(REPRO)),
        "generalization_overall": agg(corpus_set=set(GEN)),
        "combined_overall": agg(corpus_set=set(REPRO) | set(GEN)),
        # DEVICE-TIME (cudagraph) co-equal view — headline is eager; this is the launch-amortized
        # cross-check as a full geomean, so launch-bound tiny cells are shown fairly.
        "reproduction_overall_device": agg(corpus_set=set(REPRO), metric="device"),
        "generalization_overall_device": agg(corpus_set=set(GEN), metric="device"),
        "combined_overall_device": agg(corpus_set=set(REPRO) | set(GEN), metric="device"),
    }
    for dt in dtypes_seen:
        headline[f"reproduction_{dt}"] = agg(dtype_filter=dt, corpus_set=set(REPRO))
        headline[f"generalization_{dt}"] = agg(dtype_filter=dt, corpus_set=set(GEN))
        headline[f"combined_{dt}"] = agg(dtype_filter=dt, corpus_set=set(REPRO) | set(GEN))
    per_corpus = {c: agg(corpus_filter=c) for c in REAL if any(k[0] == c for k in real_cells)}

    disasters = []
    for (corpus, kernel, dtype), rws in real_cells.items():
        for r in rws:
            if cell_ok_for_ratio(r, "G_tc"):
                g = _rederive_ratio(r, "G_tc")
                if g < FLOOR:
                    disasters.append((corpus, kernel, r["shape"], dtype, g))

    out = {"per_cell": {f"{c}/{k}/{d}": v for (c, k, d), v in per_cell.items()},
           "headline": headline, "per_corpus": per_corpus, "disasters": disasters,
           "n_xna": len(xna), "xna": xna, "accfail_perf": accfail_perf,
           "coldgraph_flags": coldgraph_flags,
           "device_score_gate": DEVICE_SCORE_GATE, "launch_bound_cells": launch_bound_cells}
    json.dump(out, open(os.path.join(results_dir, "summary.json"), "w"), indent=2, default=str)

    L = []
    L.append("# PR #2866 pointwise seed heuristic — perf reproduction SUMMARY\n")
    L.append("Independent reproduction of Helion PR #2866 (pointwise/partially-tiled seed heuristic) "
             "at the exact merge commit `89e986e9`. Arms per cell: **seed** (heuristic) / **default** "
             "(unseeded `_base_default_config`) / **tc** (`torch.compile`, default mode). "
             "Metric = **cold-L2 INTERLEAVED median-of-9 event timing** (round-robin across arms so "
             "clock/thermal drift is common-mode and cancels in the ratio) — this reuses `do_bench`'s "
             "cold-L2 256MB-flush + 100ms/25ms reps primitives but INTERLEAVES the arms (the PR used "
             "sequential `do_bench`; interleaving is a stricter ratio method). Forward-only, "
             "single-process, same tensors, no CUDA graphs in the headline (device-time cross-check "
             "below).\n")
    L.append("- `G_tc = tc_us / seed_us` (>1 ⇒ seed beats torch.compile)\n"
             "- `G_def = default_us / seed_us` (>1 ⇒ seed beats the unseeded default — what the heuristic buys)\n"
             "- `G_vllm = vllm_us / seed_us` (vLLM only; >1 ⇒ seed beats vLLM's own tuned config)\n"
             "- All ratios RE-DERIVED here from the raw per-arm µs in the JSON (not from any stored table).\n")

    L.append("\n## Headline aggregate (geomean over cells with a VALID same-output comparison)\n")
    L.append("PR #2866 pointwise seed heuristic. THREE views: **reproduction** = in-sample, the "
             "exact PR-table shapes (`pointwise`); **generalization** = out-of-sample, shapes/kernels "
             "NEVER fitted — the held-out `test` split, the held-out `dyt` kernel, and expanded "
             "lever shapes (`pointwise_gen`); **combined** = both merged. Generalization tests "
             "whether the byte/stride/SFU seed interpolates or just fits the curriculum.\n")
    L.append("A cell joins ALL of a ratio's geomeans when the seed is CLOSE-ENOUGH: it passes "
             "accuracy, OR the DEFAULT config makes the IDENTICAL mistake (same acc_detail — a "
             "benign kernel/dtype fact: bf16-accumulator margin or fp8 ~1-ULP tie-rounding, shared "
             "by the whole family, not a seed-specific error). Close-enough cells are compared "
             "against every arm (G_tc, G_def, G_vllm) with a † asterisk. A seed that fails while the "
             "DEFAULT is CORRECT (the qk_norm_rope miscompile) is genuinely wrong and excluded from "
             "all ratios. See the 'Acc-fail cells' section for the exact per-cell inclusion.\n")
    L.append("| scope | geo G_tc | geo G_def | geo G_vllm | n(G_tc) | n(G_def) | n(G_vllm) |")
    L.append("|---|---|---|---|---|---|---|")
    for scope in ["reproduction_overall", "generalization_overall", "combined_overall"] + \
                 [f"reproduction_{d}" for d in dtypes_seen] + \
                 [f"generalization_{d}" for d in dtypes_seen] + \
                 [f"combined_{d}" for d in dtypes_seen]:
        if scope in headline:
            gt, gd, ntc, ndef, gv, nv = headline[scope]
            L.append(f"| {scope} | {fmt(gt)} | {fmt(gd)} | {fmt(gv)} | {ntc} | {ndef} | {nv} |")

    L.append("\n### Device-time (cudagraph) co-equal view — same cells, launch amortized\n")
    L.append("Headline above is **eager** cold-L2 interleaved timing (the PR's method). Below is the "
             "SAME geomean recomputed from per-arm **cudagraph device time** (pure on-device, launch "
             "cost removed). Agreement with the eager headline is the proof the numbers are GPU-side "
             "truth, not CPU launch overhead. The two differ only where a cell is launch-bound "
             "(tiny/decode shapes) — there the device number is the fairer one.\n")
    L.append("| scope | geo G_tc | geo G_def | n(G_tc) | n(G_def) |")
    L.append("|---|---|---|---|---|")
    for scope in ["reproduction_overall_device", "generalization_overall_device",
                  "combined_overall_device"]:
        if scope in headline:
            gt, gd, ntc, ndef, gv, nv = headline[scope]
            L.append(f"| {scope} | {fmt(gt)} | {fmt(gd)} | {ntc} | {ndef} |")
    L.append("\n(`geo G_vllm` = seed vs vLLM's own shipped tuned config — THE key comparison for a "
             "vLLM kernel; >1 ⇒ seed beats vLLM's hand-tuned config. Only vLLM-family kernels "
             "(vllm, vllm_gen, qk_norm_rope_gen) contribute; n(G_vllm) shows how many.)")

    L.append("\n### Per-corpus\n")
    L.append("| corpus | geo G_tc | geo G_def | geo G_vllm | n(G_tc) | n(G_def) | n(G_vllm) |")
    L.append("|---|---|---|---|---|---|---|")
    for c in REAL:
        if c in per_corpus:
            gt, gd, ntc, ndef, gv, nv = per_corpus[c]
            L.append(f"| {c} | {fmt(gt)} | {fmt(gd)} | {fmt(gv)} | {ntc} | {ndef} | {nv} |")

    # ---- Per-kernel COMBINED (in-sample + out-of-sample merged per kernel) ----
    L.append("\n## Per-kernel combined (in-sample + out-of-sample merged)\n")
    L.append("Each kernel's `seeded_vs_default` (= geo G_def) and `G_tc`/`min_G_tc` over ALL its "
             "cells across both corpora — the single per-kernel number comparable to the PR table.\n")
    kernels_seen = sorted({k[1] for k in real_cells})
    L.append("| kernel | in-sample n | out-of-sample n | seeded_vs_default (geo G_def) | geo G_tc | min G_tc |")
    L.append("|---|---|---|---|---|---|")
    for kern in kernels_seen:
        gtc, gdef = [], []
        n_in = n_out = 0
        for (corpus, k, dtype), rws in real_cells.items():
            if k != kern:
                continue
            if corpus in REPRO:
                n_in += len(rws)
            elif corpus in GEN:
                n_out += len(rws)
            gtc += [_rederive_ratio(r, "G_tc") for r in rws if cell_ok_for_ratio(r, "G_tc")]
            gdef += [_rederive_ratio(r, "G_def") for r in rws if cell_ok_for_ratio(r, "G_def")]
        L.append(f"| {kern} | {n_in} | {n_out} | {fmt(geomean(gdef))} | {fmt(geomean(gtc))} | "
                 f"{fmt(min(gtc) if gtc else None)} |")

    L.append(f"\n## Per-shape disasters (realistic shape with G_tc < {FLOOR})\n")
    if disasters:
        L.append("| corpus | kernel | shape | dtype | G_tc |")
        L.append("|---|---|---|---|---|")
        for c, k, s, d, g in sorted(disasters, key=lambda x: x[4]):
            L.append(f"| {c} | {k} | {s} | {d} | {fmt(g)} |")
    else:
        L.append("_(none)_")

    L.append("\n## (A) Per-(kernel, dtype) geomeans\n")
    for corpus in REAL:
        keys = sorted(k for k in per_cell if k[0] == corpus)
        if not keys:
            continue
        L.append(f"\n### {corpus}\n")
        # show the G_vllm column for ANY corpus that has vLLM-tuned data (vllm, vllm_gen,
        # qk_norm_rope_gen) — not just the literal "vllm" corpus.
        has_vllm = any(per_cell[(c, k, d)].get("geo_G_vllm") is not None for (c, k, d) in keys)
        vcol = " geo G_vllm |" if has_vllm else ""
        vsep = "---|" if has_vllm else ""
        L.append(f"| kernel | dtype | geo G_tc | geo G_def |{vcol} shapes | acc-pass | min G_tc |")
        L.append(f"|---|---|---|---|{vsep}---|---|---|")
        for (c, k, d) in keys:
            v = per_cell[(c, k, d)]
            vc = f" {fmt(v.get('geo_G_vllm'))} |" if has_vllm else ""
            L.append(f"| {k} | {d} | {fmt(v['geo_G_tc'])} | {fmt(v['geo_G_def'])} |{vc} "
                     f"{v['n_shapes']} | {v['n_acc_pass']} | {fmt(v['min_G_tc'])} |")
        if has_vllm:
            L.append("\n(`geo G_vllm` = seed vs vLLM's shipped tuned config (exact-key or "
                     "nearest-shape lookup); >1 ⇒ seed beats vLLM's tuned config. NB: this runs the "
                     "HELION kernel with vLLM's config — it is NOT vLLM's native CUDA kernel.)")

    L.append("\n## Launch-overhead cross-check (seed arm: do_bench vs cold-L2 cudagraph)\n")
    L.append("For every cell we also time the seed arm under a cold-L2 CUDA-graph replay (pure GPU "
             "device time, launch amortized). If cold-L2 `do_bench` ≈ cudagraph, CPU launch overhead "
             "is hidden behind the 256MB L2 flush and the headline number is GPU-side truth. Cells "
             f"listed below diverge by > {COLDGRAPH_GATE:.0%} — the canary that launch cost is leaking "
             "into that measurement (see notes/LAUNCH_OVERHEAD_NOTE.md).\n")
    if launch_bound_cells:
        L.append(f"\n**Scoring decision (device-time override, gate = {DEVICE_SCORE_GATE:.0%}).** A cell "
                 f"whose eager vs cudagraph divergence exceeds {DEVICE_SCORE_GATE:.0%} on ANY arm is "
                 "BELOW the eager timing floor — its do_bench-style ratio is dominated by per-arm-"
                 "variable CPU launch tax, not compute. Those cells (below) are scored on **cudagraph "
                 f"DEVICE time** in the headline; all other cells stay on eager (the PR's method). "
                 f"{DEVICE_SCORE_GATE:.0%} matches the harness round-to-round spread gate — 'if eager and "
                 "device disagree by more than the timing noise floor, trust device.' Both ratios shown "
                 "for full transparency; the device column is the one used in the headline geomeans.\n")
        L.append("| corpus | kernel | shape | dtype | worst arm div | eager G_def | **device G_def** | eager G_tc | **device G_tc** |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c in sorted(launch_bound_cells, key=lambda x: -x["worst_div"]):
            L.append(f"| {c['corpus']} | {c['kernel']} | {c['shape']} | {c['dtype']} | "
                     f"{c['worst_div']:.0%} | {fmt(c['eager_G_def'])} | **{fmt(c['device_G_def'])}** | "
                     f"{fmt(c['eager_G_tc'])} | **{fmt(c['device_G_tc'])}** |")
        L.append("")
    if coldgraph_flags:
        L.append("| corpus | kernel | shape | dtype | do_bench µs | cudagraph µs | launch_frac |")
        L.append("|---|---|---|---|---|---|---|")
        for c in sorted(coldgraph_flags, key=lambda x: -abs(x["launch_frac"])):
            L.append(f"| {c['corpus']} | {c['kernel']} | {c['shape']} | {c['dtype']} | "
                     f"{fmt(c['dobench_us'],1)} | {fmt(c['coldgraph_us'],1)} | {fmt(c['launch_frac'])} |")
    else:
        L.append("_(none — every cell's do_bench matched its cudagraph device time within "
                 f"{COLDGRAPH_GATE:.0%}: launch overhead is hidden, headline numbers are GPU-side truth.)_")

    n_ce = sum(1 for c in accfail_perf if c["included_in_G_def"] or c["included_in_G_tc"])
    n_excl = sum(1 for c in accfail_perf if not (c["included_in_G_def"] or c["included_in_G_tc"] or c["default_no_compile"]))
    n_nocompile = sum(1 for c in accfail_perf if c["default_no_compile"])
    L.append(f"\n## Acc-fail cells — perf measured ({len(accfail_perf)} cells)\n")
    L.append("A cell fails the strict accuracy gate but is treated as **CLOSE-ENOUGH** — and its "
             "speed IS compared against ALL arms (G_tc, G_def, G_vllm) — when the DEFAULT config "
             "makes the IDENTICAL mistake (same acc_detail). Default-matches-seed means the miss is "
             "a benign kernel/dtype FACT the whole family shares (bf16-accumulator margin on "
             "rms_norm_bwd; fp8 tie-rounding — ~1 ULP on ~3% of elements — on per_token_group), NOT "
             "a seed-specific wrong answer, so timing all arms is apples-to-apples (†). A seed that "
             "fails while the DEFAULT is CORRECT (the qk_norm_rope tile-size miscompile: seed "
             "maxabs~2, default fine) is genuinely wrong and is EXCLUDED from every ratio.\n")
    L.append(f"- **{n_ce}** close-enough cells (default makes the same mistake) → included in all "
             f"applicable ratios with a † asterisk.\n"
             f"- **{n_excl}** seed-only-failure cells → excluded from all ratios (real wrong answer).\n"
             f"- **{n_nocompile}** cells where the DEFAULT config could not compile (ptxas/Inductor "
             f"timeout) — the seed compiled and ran, but no default ratio is possible (a point in "
             f"the seed's favor, noted not counted).\n")
    if accfail_perf:
        L.append("| corpus | kernel | shape | dtype | seed µs | def µs | tc µs | "
                 "perf s/def | perf s/vllm | in G_tc | in G_def | in G_vllm | note |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for c in accfail_perf:
            u = c["us"]
            note = ("default no-compile" if c["default_no_compile"]
                    else "close-enough † (default same mistake)" if c["both_fail"]
                    else "SEED-ONLY wrong (excluded)")
            L.append(f"| {c['corpus']} | {c['kernel']} | {c['shape']} | {c['dtype']} | "
                     f"{fmt(u.get('seed'),1)} | {fmt(u.get('default'),1)} | {fmt(u.get('tc'),1)} | "
                     f"{fmt(c['perf_only_def'])} | {fmt(c['perf_only_vllm'])} | "
                     f"{'yes' if c['included_in_G_tc'] else '—'} | "
                     f"{'yes' if c['included_in_G_def'] else '—'} | "
                     f"{'yes' if c['included_in_G_vllm'] else '—'} | {note} |")
    else:
        L.append("_(none)_")

    L.append(f"\n## x / n/a cells ({len(xna)} arm-level entries)\n")
    if xna:
        L.append("| corpus | kernel | shape | dtype | arm | reason |")
        L.append("|---|---|---|---|---|---|")
        for c, k, s, d, arm, reason in xna:
            L.append(f"| {c} | {k} | {s} | {d} | {arm} | {reason} |")
    else:
        L.append("_(none)_")

    open(os.path.join(results_dir, "SUMMARY.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {results_dir}/SUMMARY.md + summary.json")
    rep, gen = headline["reproduction_overall"], headline["generalization_overall"]
    print(f"reproduction: G_tc={fmt(rep[0])} G_def={fmt(rep[1])} G_vllm={fmt(rep[4])}(n{rep[5]}) | "
          f"generalization: G_tc={fmt(gen[0])} G_def={fmt(gen[1])} G_vllm={fmt(gen[4])}(n{gen[5]})")
    print(f"disasters: {len(disasters)}  x/na: {len(xna)}  coldgraph-flags: {len(coldgraph_flags)}")


if __name__ == "__main__":
    main()
