"""Run the REAL Helion full-effort autotune on rms_norm (32768,256) fp32 and
capture (a) the final winning config, (b) the autotuner-internal perf it
recorded for the winner, then (c) FAIR re-bench the winner. This reproduces
(or refutes) the oracle's 73.6us for persistent/block=1/num_warps=32.

The autotune CSV log (HELION_AUTOTUNE_LOG) records perf_ms for every config the
search timed; we scan it for any sub-100us w32 entry.
"""

from __future__ import annotations

import csv
import os
import sys

import torch

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
sys.path.insert(0, WT)

import helion  # noqa: E402

assert helion.__file__.startswith(WT)

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402
from triton.testing import do_bench  # noqa: E402

EPS = 1e-6
M, N = 32768, 256
LOG = "/tmp/autotune_log_32768_256.csv"


def fair_med(fn, reps=5):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(reps))[reps // 2]


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32, device="cuda")
    weight = torch.ones(N, dtype=torch.float32, device="cuda")
    args = (x, weight, EPS)
    ref = rms_norm_pytorch(x, weight, EPS)

    print(f"=== FULL autotune repro ({M},{N}) fp32 GPU={gpu} ===\n")

    os.environ["HELION_AUTOTUNE_EFFORT"] = "full"
    os.environ["HELION_AUTOTUNE_LOG"] = LOG
    if os.path.exists(LOG):
        os.remove(LOG)

    k = helion.kernel(rms_norm_fwd.fn)
    bound = k.bind(args)
    bound.autotune(args)
    win = dict(bound._config)
    print(f"WINNING config: {win}\n")

    out = bound(*args)
    out0 = out[0] if isinstance(out, tuple) else out
    max_abs = float((out0.float() - ref.float()).abs().max())
    fair = fair_med(lambda: bound(*args))
    print(f"winner correctness max_abs = {max_abs:.3e}")
    print(f"winner FAIR do_bench median = {fair*1000:.2f}us\n")

    # scan the autotune log for what perf the autotuner recorded per config,
    # especially the winner's num_warps and any sub-100us w32 entries.
    if os.path.exists(LOG):
        print("--- autotune log: sorted by recorded perf_ms (top 15 fastest 'ok') ---")
        rows = []
        with open(LOG, newline="") as f:
            r = csv.DictReader(f)
            cols = r.fieldnames
            for row in r:
                rows.append(row)
        print(f"(columns: {cols})")
        # find perf + config-ish columns
        perf_key = "perf_ms" if "perf_ms" in (cols or []) else None
        ok_rows = [row for row in rows
                   if perf_key and row.get(perf_key) not in (None, "", "inf")]
        def pf(row):
            try:
                return float(row[perf_key])
            except (ValueError, TypeError):
                return float("inf")
        ok_rows.sort(key=pf)
        for row in ok_rows[:15]:
            # print the perf and the whole row compactly
            compact = {c: row[c] for c in (cols or []) if row.get(c) not in (None, "")}
            print(f"   perf={row.get(perf_key)}  {compact}")
        # any w32 sub-100us?
        print("\n--- any recorded perf < 0.100ms with num_warps=32 in config? ---")
        hits = 0
        for row in ok_rows:
            blob = " ".join(str(v) for v in row.values())
            if pf(row) < 0.100 and ("num_warps=32" in blob or "'num_warps': 32" in blob or "num_warps\": 32" in blob):
                print(f"   {row}")
                hits += 1
        if hits == 0:
            print("   NONE FOUND (no sub-100us w32 config in the autotune log).")
    else:
        print("NO autotune log written.")


if __name__ == "__main__":
    main()
