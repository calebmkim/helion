"""CONSOLIDATED TEST re-read (Goal 6 / Goal 1) — RUN EXACTLY ONCE on the FINAL,
FROZEN Phase-I heuristic, then re-lock TEST. Pre-authorized re-reads ONLY:
  (1) welford TEST  — v7 numbers invalid under the corrected kernel (Goal 1).
  (2) rms_norm TEST G — run-1's ~0.828 had no raw log (regenerate, backed).
NO other kernel's TEST column is re-read.

Firewall: welford TEST shapes here EXCLUDE (262144,5120) and (262144,2560) which
were promoted to in-sample-v2 (Goal 4); rms_norm TEST EXCLUDES (256,4096) (promoted).
The prime (262144,1543) is the CORRECTNESS canary (correct + fast), reported separately.
do_bench median-of-N; tc_default = torch.compile(reference) default mode; correctness
gated rtol=1e-3/atol=1e-4. Sub-25us shapes flagged NOISE-FLOOR (undefendable G).
Usage: ... python run2_TEST_reread.py
"""

from __future__ import annotations

import importlib.util
import json
import math

import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), helion.__file__
spec = importlib.util.spec_from_file_location(
    "mg", WT + "_lab/harness/run2_measure_g.py"
)
mg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mg)

# Clean TEST shapes (NOT tuned on; promoted shapes excluded per firewall).
WELFORD_TEST = [(262144, 7168), (262144, 1280), (262144, 768), (131072, 2048)]
WELFORD_CANARY = [(262144, 1543)]  # prime: correctness + must-be-fast
RMS_TEST = [
    (2048, 2560),
    (2048, 1025),
    (4096, 10240),
    (8192, 2048),
    (1, 131072),
    (65536, 512),
]


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else None


def run(kernel, shapes):
    out = []
    for m, n in shapes:
        try:
            r = mg.measure(kernel, m, n)
            r["noise_floor"] = r["tc_lat_us"] < 25.0
            out.append(r)
            print(
                f"  {kernel}({m},{n}) G={r['G_seed'] and round(r['G_seed'], 3)} "
                f"ok={r['correct']} {'NOISE-FLOOR' if r['noise_floor'] else ''} "
                f"tc={r['tc_lat_us']:.1f}us err={r['maxerr']:.1e}",
                flush=True,
            )
        except Exception as e:
            print(
                f"  {kernel}({m},{n}) ERR {type(e).__name__}: {str(e)[:80]}", flush=True
            )
            out.append({"shape": [m, n], "error": str(e)[:120]})
    return out


def main():
    print(
        f"=== CONSOLIDATED TEST RE-READ (read-once) helion={helion.__file__} ===",
        flush=True,
    )
    res = {}
    print("welford TEST (clean):", flush=True)
    res["welford_test"] = run("welford", WELFORD_TEST)
    print("welford PRIME canary:", flush=True)
    res["welford_canary"] = run("welford", WELFORD_CANARY)
    print("rms_norm TEST:", flush=True)
    res["rms_norm_test"] = run("rms_norm", RMS_TEST)
    # geomeans (all + excl noise-floor)

    def gm(rows, excl_nf=False):
        gs = [
            r["G_seed"]
            for r in rows
            if r.get("correct")
            and r.get("G_seed")
            and (not excl_nf or not r.get("noise_floor"))
        ]
        return geomean(gs)

    summary = {
        "welford_TEST_G": gm(res["welford_test"]),
        "welford_TEST_G_excl_noisefloor": gm(res["welford_test"], True),
        "welford_v7_TEST_was": 0.396,
        "rms_norm_TEST_G": gm(res["rms_norm_test"]),
        "rms_norm_TEST_G_excl_noisefloor": gm(res["rms_norm_test"], True),
        "rms_norm_run1_TEST_was": 0.828,
        "welford_prime_canary": [
            (r["shape"], r.get("G_seed"), r.get("correct"))
            for r in res["welford_canary"]
        ],
    }
    res["summary"] = summary
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    with open("/tmp/TEST_reread.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("\n[written] /tmp/TEST_reread.json — record to ledger + RE-LOCK TEST.")


if __name__ == "__main__":
    main()
