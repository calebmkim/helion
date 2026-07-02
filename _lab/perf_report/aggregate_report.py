"""Aggregate the per-(corpus,kernel) JSON cells into the manager-facing perf report.

Reads every results/*.json produced by perf_report_bench.py and emits:
  (A) Real workloads (curriculum, transfer, mreduction, vllm) — 3 arms. Per (kernel,dtype):
      per-shape rows + per-(kernel,dtype) geomean of G_tc and G_def over ACC-PASSING rows.
      Grouped by corpus. vLLM native dtype only.
  (B) Generality diagnostics (synthetic probes + adversarial) — seed-vs-default G_def only.
  Headline aggregate: overall geomean of G_tc and G_def over real cells with BOTH arms valid,
  per dtype; disasters (G_tc < 0.75 on a realistic shape); x/n/a count with reasons.

Writes: SUMMARY.md (human) + summary.json (machine) into the results dir.

Run: /home/dev/helion/.venv/bin/python aggregate_report.py <RESULTS_DIR>
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

REAL = ("curriculum", "transfer", "mreduction", "vllm")
DIAG = ("synthetic_probes", "adversarial_synth")
FLOOR = 0.75


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def _acc_ok(arm):
    """A timed arm counts iff status ok (acc True or acc absent-but-ok)."""
    return arm.get("status") == "ok"


def load_rows(results_dir):
    rows = []
    # Only the per-(corpus,kernel) bench files; skip summaries, the A/B, and the external
    # config-reproduction manifest (a different artifact, not bench rows).
    _SKIP = {"summary.json", "narrow_w1_ab.json", "config_manifest.json"}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path) in _SKIP:
            continue
        try:
            d = json.load(open(path))
        except Exception:  # noqa: BLE001
            continue
        for r in d.get("rows", []):
            rows.append(r)
    return rows


_RATIO_ARM = {"G_tc": "tc", "G_def": "default", "G_vllm": "vllm_shipped"}


def cell_ok_for_ratio(row, which):
    """G valid iff BOTH arms of the ratio timed ok. which in {'G_tc','G_def','G_vllm'}."""
    arms = row.get("arms", {})
    seed = arms.get("seed", {})
    if not _acc_ok(seed):
        return False
    other = arms.get(_RATIO_ARM[which], {})
    return _acc_ok(other) and row.get(which) is not None


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/dev/local/prompts-lab/perf-report/results"
    rows = load_rows(results_dir)

    # ---- organize ----
    # real: by (corpus, kernel, dtype) -> list of rows
    real_cells = {}
    diag_rows = []
    xna = []  # (corpus, kernel, shape, dtype, arm, reason)
    accfail_perf = []  # acc-fail cells that still have a recorded seed timing
    for r in rows:
        corpus = r.get("corpus")
        if "error" in r and "arms" not in r:
            xna.append((corpus, r.get("kernel"), r.get("shape"), r.get("dtype"),
                        "bind/build", r["error"][:80]))
            continue
        if corpus in DIAG:
            diag_rows.append(r)
            continue
        key = (corpus, r["kernel"], r.get("dtype"))
        real_cells.setdefault(key, []).append(r)
        # record x/n/a arms
        for arm in ("seed", "default", "tc"):
            a = r.get("arms", {}).get(arm, {})
            st = a.get("status", "")
            if st not in ("ok", "n/a-no-tc"):
                xna.append((corpus, r["kernel"], r.get("shape"), r.get("dtype"), arm,
                            st + (f" ({a['acc_detail']})" if a.get("acc_detail") else "")))
        # collect acc-fail cells that STILL have perf recorded (seed arm acc-fail but timed),
        # with the perf-only ratios and whether seed & default fail identically.
        seed_a = r.get("arms", {}).get("seed", {})
        if seed_a.get("status") == "acc-fail" and seed_a.get("us") is not None:
            u = r.get("us", {})
            def_a = r.get("arms", {}).get("default", {})
            accfail_perf.append({
                "corpus": corpus, "kernel": r["kernel"], "shape": r.get("shape"),
                "dtype": r.get("dtype"), "us": u,
                "perf_only_tc": r.get("perf_only_tc"), "perf_only_def": r.get("perf_only_def"),
                "default_status": def_a.get("status"),
                "both_fail": def_a.get("status") == "acc-fail",
            })

    # ---- per-(kernel,dtype) geomeans ----
    per_cell = {}
    for key, rws in real_cells.items():
        gtc = [r["G_tc"] for r in rws if cell_ok_for_ratio(r, "G_tc")]
        gdef = [r["G_def"] for r in rws if cell_ok_for_ratio(r, "G_def")]
        gvllm = [r["G_vllm"] for r in rws if cell_ok_for_ratio(r, "G_vllm")]
        per_cell[key] = {
            "n_shapes": len(rws),
            "n_acc_pass": sum(1 for r in rws if _acc_ok(r.get("arms", {}).get("seed", {}))),
            "geo_G_tc": geomean(gtc),
            "geo_G_def": geomean(gdef),
            "geo_G_vllm": geomean(gvllm) if gvllm else None,
            "n_gtc": len(gtc),
            "n_gdef": len(gdef),
            "n_gvllm": len(gvllm),
            "min_G_tc": min(gtc) if gtc else None,
        }

    # ---- headline aggregate (per dtype, over cells with both arms) ----
    def agg(dtype_filter):
        gtc, gdef = [], []
        for (corpus, kernel, dtype), rws in real_cells.items():
            if dtype_filter and dtype != dtype_filter:
                continue
            gtc += [r["G_tc"] for r in rws if cell_ok_for_ratio(r, "G_tc")]
            gdef += [r["G_def"] for r in rws if cell_ok_for_ratio(r, "G_def")]
        return geomean(gtc), geomean(gdef), len(gtc), len(gdef)

    dtypes_seen = sorted({k[2] for k in real_cells})
    headline = {"overall": agg(None)}
    for dt in dtypes_seen:
        headline[dt] = agg(dt)

    # ---- per-corpus headline (the actionable cut) ----
    def agg_corpus(corpus_filter):
        gtc, gdef = [], []
        for (corpus, kernel, dtype), rws in real_cells.items():
            if corpus != corpus_filter:
                continue
            gtc += [r["G_tc"] for r in rws if cell_ok_for_ratio(r, "G_tc")]
            gdef += [r["G_def"] for r in rws if cell_ok_for_ratio(r, "G_def")]
        return geomean(gtc), geomean(gdef), len(gtc), len(gdef)

    per_corpus = {c: agg_corpus(c) for c in REAL if any(k[0] == c for k in real_cells)}

    # ---- disasters (realistic shape below floor, per-shape) ----
    disasters = []
    for (corpus, kernel, dtype), rws in real_cells.items():
        for r in rws:
            if cell_ok_for_ratio(r, "G_tc") and r["G_tc"] < FLOOR:
                disasters.append((corpus, kernel, r["shape"], dtype, r["G_tc"]))

    # ---- diagnostics table ----
    diag = []
    for r in diag_rows:
        arms = r.get("arms", {})
        seed_ok = _acc_ok(arms.get("seed", {}))
        gdef = r.get("G_def") if seed_ok and _acc_ok(arms.get("default", {})) else None
        status = "declined" if (arms.get("seed", {}).get("status") == "no-config") else \
                 arms.get("seed", {}).get("status", "?")
        diag.append({"corpus": r.get("corpus"), "kernel": r.get("kernel"),
                     "shape": r.get("shape"), "G_def": gdef, "status": status,
                     "error": r.get("error")})

    # ================= WRITE =================
    out = {"per_cell": {f"{c}/{k}/{d}": v for (c, k, d), v in per_cell.items()},
           "headline": headline, "per_corpus": per_corpus,
           "disasters": disasters, "n_xna": len(xna),
           "xna": xna, "diagnostics": diag, "accfail_perf": accfail_perf}
    json.dump(out, open(os.path.join(results_dir, "summary.json"), "w"), indent=2, default=str)

    L = []
    L.append("# Reduction-seed perf report — SUMMARY\n")
    L.append("3 arms per real cell: **seed** (heuristic) / **default** (unseeded base) / "
             "**tc** (torch.compile default). Metric = cold-L2 median-of-9 `do_bench`, "
             "forward-only, single-process, same tensors.\n")
    L.append("- `G_tc = tc_us / seed_us` (>1 ⇒ seed beats torch.compile)\n"
             "- `G_def = default_us / seed_us` (>1 ⇒ seed beats the unseeded default — "
             "\"what the heuristic buys\")\n")

    # headline
    L.append("\n## Headline aggregate (geomean over cells with both arms valid)\n")
    L.append("| scope | geo G_tc | geo G_def | n(G_tc) | n(G_def) |")
    L.append("|---|---|---|---|---|")
    for scope in ["overall"] + dtypes_seen:
        gt, gd, ntc, ndef = headline[scope]
        L.append(f"| {scope} | {fmt(gt)} | {fmt(gd)} | {ntc} | {ndef} |")

    L.append("\n### Per-corpus\n")
    L.append("| corpus | geo G_tc | geo G_def | n(G_tc) | n(G_def) |")
    L.append("|---|---|---|---|---|")
    for c in REAL:
        if c in per_corpus:
            gt, gd, ntc, ndef = per_corpus[c]
            L.append(f"| {c} | {fmt(gt)} | {fmt(gd)} | {ntc} | {ndef} |")

    # disasters
    L.append(f"\n## Per-shape disasters (realistic shape with G_tc < {FLOOR})\n")
    if disasters:
        L.append("| corpus | kernel | shape | dtype | G_tc |")
        L.append("|---|---|---|---|---|")
        for c, k, s, d, g in sorted(disasters, key=lambda x: x[4]):
            L.append(f"| {c} | {k} | {s} | {d} | {fmt(g)} |")
    else:
        L.append("_(none)_")

    # section A: real workloads, grouped by corpus
    L.append("\n## (A) Real workloads — per-(kernel, dtype) geomeans\n")
    for corpus in REAL:
        keys = sorted(k for k in per_cell if k[0] == corpus)
        if not keys:
            continue
        L.append(f"\n### {corpus}\n")
        is_vllm = corpus == "vllm"
        vcol = " geo G_vllm |" if is_vllm else ""
        vsep = "---|" if is_vllm else ""
        L.append(f"| kernel | dtype | geo G_tc | geo G_def |{vcol} shapes | acc-pass | min G_tc |")
        L.append(f"|---|---|---|---|{vsep}---|---|---|")
        for (c, k, d) in keys:
            v = per_cell[(c, k, d)]
            vc = f" {fmt(v.get('geo_G_vllm'))} |" if is_vllm else ""
            L.append(f"| {k} | {d} | {fmt(v['geo_G_tc'])} | {fmt(v['geo_G_def'])} |{vc} "
                     f"{v['n_shapes']} | {v['n_acc_pass']} | {fmt(v['min_G_tc'])} |")
        if is_vllm:
            L.append("\n(`geo G_vllm` = seed vs vLLM's own shipped H100-tuned config, "
                     "nearest-shape lookup; >1 ⇒ seed beats vLLM's tuned config.)")

    # section B: diagnostics
    L.append("\n## (B) Generality diagnostics (seed vs default, G_def only — NOT headline perf)\n")
    L.append("| corpus | kernel | shape | G_def | status |")
    L.append("|---|---|---|---|---|")
    for r in sorted(diag, key=lambda x: (x["corpus"], x["kernel"])):
        g = fmt(r["G_def"]) if r["G_def"] is not None else "n/a"
        st = r["status"] if not r.get("error") else f"ERR:{str(r['error'])[:40]}"
        L.append(f"| {r['corpus']} | {r['kernel']} | {r['shape']} | {g} | {st} |")

    # acc-fail cells with perf still recorded (perf comparison meaningful when seed==default fail)
    L.append(f"\n## Acc-fail cells — perf still measured ({len(accfail_perf)} cells)\n")
    L.append("These cells fail the accuracy gate (excluded from the headline geomeans, footgun "
             "#6a) but timing is recorded anyway. `both_fail=yes` ⇒ seed AND default fail the gate "
             "identically, so the perf ratios below are still an apples-to-apples comparison "
             "(the failure is a kernel-source fact — bf16 accumulator / fp8 boundary — not a "
             "seed-specific wrong answer).\n")
    if accfail_perf:
        L.append("| corpus | kernel | shape | dtype | seed_us | def_us | tc_us | "
                 "perf seed/tc | perf seed/def | both_fail |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for c in accfail_perf:
            u = c["us"]
            L.append(f"| {c['corpus']} | {c['kernel']} | {c['shape']} | {c['dtype']} | "
                     f"{fmt(u.get('seed'),1)} | {fmt(u.get('default'),1)} | {fmt(u.get('tc'),1)} | "
                     f"{fmt(c['perf_only_tc'])} | {fmt(c['perf_only_def'])} | "
                     f"{'yes' if c['both_fail'] else 'NO — '+str(c['default_status'])} |")
    else:
        L.append("_(none)_")

    # x/n/a
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
    print(f"headline overall: geo G_tc={fmt(headline['overall'][0])} "
          f"geo G_def={fmt(headline['overall'][1])}")
    print(f"disasters: {len(disasters)}  x/na arm-entries: {len(xna)}")


if __name__ == "__main__":
    main()
