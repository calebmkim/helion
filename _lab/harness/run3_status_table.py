"""Fresh seed+tc re-bench (LIVE champion) joined against the CACHED oracle.

The oracle_cache.json `seed_us` is a palimpsest — several rows predate the edit
that fixed them (e.g. CE-wide predates EDIT-PID). The ORACLE latency is still
valid (cache key = kernel-source-hash, excludes the heuristic). So: re-measure
the live seed + tc-default for every cached shape with the established
`measure()` plumbing, and report seed/tc/oracle as a current triple.

Run from /tmp:
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_status_table.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

from run2_measure_g import measure  # noqa: E402

import helion  # noqa: E402

LOG_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "logs", "run3"))
CACHE = os.path.join(LOG_DIR, "oracle_cache.json")
OUT = os.path.join(LOG_DIR, "status_table_fresh.json")


# RUN-3 scope: another agent owns welford (the Band-C structured-combine kernel);
# PARITY this run = the OTHER 8 kernels. Skip welford so the table stays in-scope
# and we don't spend GPU on a kernel we don't report.
INSCOPE = {
    "rms_norm", "layer_norm", "sum", "long_sum", "cross_entropy",
    "softmax", "kl_div", "jsd",
}


def main():
    print(f"helion={helion.__file__}", flush=True)
    cache = json.load(open(CACHE))["entries"]

    rows = []
    for key, ent in cache.items():
        kn = ent["kernel"]
        if kn not in INSCOPE:
            print(f"  -- skip {key} (out of scope: {kn})", flush=True)
            continue
        M, N = ent["shape"]
        oracle_us = ent["oracle_us"]
        oracle_eff = ent["effort"]
        try:
            r = measure(kn, M, N)
        except Exception as e:  # OOM / hard error — record and move on
            print(f"  !! {key}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            rows.append({"key": key, "kernel": kn, "shape": [M, N],
                         "error": f"{type(e).__name__}: {e}"})
            continue
        seed_us = r["seed_lat_us"]
        tc_us = r["tc_lat_us"]
        rec = {
            "key": key, "kernel": kn, "shape": [M, N],
            "seed_us": seed_us, "tc_us": tc_us, "oracle_us": oracle_us,
            "oracle_effort": oracle_eff,
            "seed_codegen": r["seed_codegen"],
            "correct": r["correct"], "maxerr": r["maxerr"],
            "seed_oracle": (seed_us / oracle_us) if seed_us else None,
            "G_floor": (tc_us / seed_us) if seed_us else None,
            "oracle_vs_tc": oracle_us / tc_us,
        }
        rows.append(rec)
        so = rec["seed_oracle"]
        g = rec["G_floor"]
        print(f"  {key:28s} seed={seed_us:9.2f} tc={tc_us:9.2f} "
              f"orac={oracle_us:9.2f}  s/orac={so if so else 0:.3f} "
              f"G={g if g else 0:.3f}  ({r['seed_codegen']}, corr={r['correct']})",
              flush=True)

    json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
