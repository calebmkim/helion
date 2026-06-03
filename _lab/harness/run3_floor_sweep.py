"""RUN-3 Phase-1 FLOOR sweep over the `train` split (machine-portable).

For every (kernel, shape) in the `train` split of shapes_v3_draft.py, measure the
Product-A FLOOR ratio:

    G = tc_default_latency / seed_latency      (G >= 1-eps  ==>  floor PASS)

using the canonical run-2 plumbing VERBATIM (the 9-kernel arg/fp32-ref builders,
the live-seed bare-config mechanism, correctness gate, codegen kind, and the
median-of-N do_bench) imported from run2_measure_g. NO autotune is run here — this
is the cheap floor pass that triages where the (expensive) oracle budget pays off.

Output: ranked table (G ascending — worst floor first) + per-kernel geomean + a
flagged list of floor losses (G < 1-eps) and correctness failures. Also writes a
machine-readable JSON checkpoint per kernel so a re-run can resume, and a merged
JSON at the end.

Each shape is measured with median-of-N (N from run2_measure_g.N_RUNS, =7). Shapes
whose 7 do_bench samples have a large spread (>SPREAD_FRAC of the median on EITHER
arm) are RE-RUN once and the re-run reported (noise discipline). The seed/oracle
victory bar is NOT computed here (Phase 2); this establishes the floor only.

fp32 everywhere (asserted by the underlying builders + a per-call dtype check on
the constructed inputs). Pin CUDA_VISIBLE_DEVICES; confirm GPU idle before trust.

Canonical invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_floor_sweep.py \
    [kernel ...]            # optional: restrict to a subset of kernels

Defaults to ALL 9 in-sample kernels if no kernel args are given.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

# Import the canonical, wiring-correct plumbing. run2_measure_g asserts helion
# resolves under this worktree (machine-portable now), so importing it is also
# our wiring check.
_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)  # local harness dir only (NOT a worktree)
_PROMPTS_DIR = os.path.join(_HARNESS_DIR, "..", "prompts")
if os.path.abspath(_PROMPTS_DIR) not in sys.path:
    sys.path.insert(0, os.path.abspath(_PROMPTS_DIR))

import helion  # noqa: E402

from run2_measure_g import (  # noqa: E402
    KERNELS,
    N_RUNS,
    geomean,
    get_seed,
    codegen_kind,
    check_correct,
)
from shapes_v3_draft import SHAPES, BANDS  # noqa: E402

from triton.testing import do_bench  # noqa: E402

EPS = 0.05  # floor tolerance: G >= 1-EPS passes the floor (== seed within 5% of tc)
SPREAD_FRAC = 0.08  # if (max-min)/median > this on either arm, re-run once
LOG_DIR = os.path.join(_HARNESS_DIR, "..", "logs", "run3")


def _band_index(kernel: str, n: int) -> int:
    edges = BANDS[kernel]
    for i, e in enumerate(edges):
        if n <= e:
            return i
    return len(edges)


def _band_label(kernel: str, n: int) -> str:
    return f"b{_band_index(kernel, n)}"


def _do_bench_samples(fn, n: int = N_RUNS):
    """n independent do_bench(median) samples (ms). Caller computes stats."""
    torch.cuda.synchronize()
    return [float(do_bench(fn, return_mode="median")) for _ in range(n)]


def _stats(samples):
    s = sorted(samples)
    med = s[len(s) // 2]
    lo, hi = s[0], s[-1]
    spread = (hi - lo) / med if med > 0 else float("inf")
    return med, lo, hi, spread


def measure_floor(kernel_name, M, N):
    """Floor measurement for one (kernel, M, N) with spread + re-run discipline.

    Mirrors run2_measure_g.measure but returns spread on both arms and re-runs a
    high-spread shape once. Returns a dict with G, both medians (us), spreads,
    codegen kind, correctness, and the resolved/normalized seed config.
    """
    fn, builder, tc_ref = KERNELS[kernel_name]
    args, ref, out_extract = builder(M, N)

    # fp32 assert on the constructed float inputs (welford/softmax defaults differ
    # upstream; these builders already force fp32 — verify it held).
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, (
                f"{kernel_name}{(M, N)} non-fp32 input {a.dtype}"
            )

    # --- live heuristic seed, bare config (no autotune) ---
    seed, _ = get_seed(fn, args)
    seeded = helion.kernel(fn.fn, configs=[helion.Config(**dict(seed))])
    bound_s = seeded.bind(args)
    bound_s.ensure_config_exists(args)
    codegen = codegen_kind(bound_s)
    norm_cfg = dict(bound_s._config)  # the NORMALIZED config actually run

    out_s = out_extract(bound_s(*args))
    correct, maxerr = check_correct(out_s, ref)

    # --- torch.compile DEFAULT of the fp32 reference ---
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    out_tc = out_extract(tc(args))
    ok_tc, _ = check_correct(out_tc, ref)
    assert ok_tc, f"tc reference correctness FAIL {kernel_name} {(M, N)}"

    def _bench_pair():
        seed_s = _do_bench_samples(lambda: bound_s(*args)) if correct else None
        tc_s = _do_bench_samples(lambda: tc(args))
        return seed_s, tc_s

    seed_samples, tc_samples = _bench_pair()
    reran = False
    if correct:
        _, _, _, sp_s = _stats(seed_samples)
        _, _, _, sp_t = _stats(tc_samples)
        if max(sp_s, sp_t) > SPREAD_FRAC:
            reran = True
            seed_samples, tc_samples = _bench_pair()  # one re-run on high spread

    if correct:
        seed_med, _, _, seed_spread = _stats(seed_samples)
    else:
        seed_med, seed_spread = None, None
    tc_med, _, _, tc_spread = _stats(tc_samples)

    g = (tc_med / seed_med) if (correct and seed_med) else None
    return {
        "kernel": kernel_name,
        "shape": [M, N],
        "band": _band_label(kernel_name, N),
        "G": g,
        "seed_us": (seed_med * 1000) if seed_med is not None else None,
        "tc_us": tc_med * 1000,
        "seed_spread": seed_spread,
        "tc_spread": tc_spread,
        "reran_highspread": reran,
        "seed_codegen": codegen,
        "correct": correct,
        "maxerr": maxerr,
        "seed_cfg_raw": dict(seed),
        "seed_cfg_norm": norm_cfg,
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    kernels = sys.argv[1:] or list(SHAPES.keys())
    os.makedirs(os.path.abspath(LOG_DIR), exist_ok=True)
    print(f"GPU={gpu}  helion={helion.__file__}", flush=True)
    print(f"kernels={kernels}  EPS={EPS}  N_RUNS={N_RUNS}\n", flush=True)

    all_results = []
    per_kernel_g = {}
    floor_losses = []  # G < 1-EPS
    fails = []
    ooms = []

    for kernel in kernels:
        shapes = SHAPES[kernel]["train"]
        gs = []
        kres = []
        for (m, n) in shapes:
            tag = f"{kernel}({m},{n})"
            try:
                r = measure_floor(kernel, m, n)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                ooms.append({"kernel": kernel, "shape": [m, n],
                             "error": f"OOM:{type(e).__name__}"})
                print(f"[OOM ] {tag}", flush=True)
                continue
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    ooms.append({"kernel": kernel, "shape": [m, n],
                                 "error": msg[:200]})
                    print(f"[OOM ] {tag}: {msg[:100]}", flush=True)
                else:
                    fails.append({"kernel": kernel, "shape": [m, n],
                                  "error": msg[:300]})
                    print(f"[ERR ] {tag}: {msg[:160]}", flush=True)
                continue

            all_results.append(r)
            kres.append(r)
            if not r["correct"]:
                fails.append({"kernel": kernel, "shape": [m, n],
                              "maxerr": r["maxerr"]})
            else:
                gs.append(r["G"])
                if r["G"] < 1 - EPS:
                    floor_losses.append(r)
            gstr = f"{r['G']:.3f}" if r["G"] is not None else "  None"
            sp = (f"sp(s/t)={r['seed_spread']:.2f}/{r['tc_spread']:.2f}"
                  if r["seed_spread"] is not None
                  else f"sp(t)={r['tc_spread']:.2f}")
            flag = "*RERUN" if r["reran_highspread"] else ""
            print(
                f"[{'OK ' if r['correct'] else 'BAD'}] {tag:>24} {r['band']} "
                f"G={gstr:>6} {r['seed_codegen']:>10} "
                f"seed={r['seed_us'] if r['seed_us'] else float('nan'):>8.1f}us "
                f"tc={r['tc_us']:>8.1f}us {sp} {flag}",
                flush=True,
            )
        per_kernel_g[kernel] = geomean(gs)
        # per-kernel checkpoint
        with open(os.path.join(os.path.abspath(LOG_DIR),
                               f"floor_{kernel}.json"), "w") as f:
            json.dump({"gpu": gpu, "kernel": kernel,
                       "geomean_G": per_kernel_g[kernel],
                       "results": kres}, f, indent=2, default=str)

    # ----- ranked table (G ascending) -----
    print("\n" + "=" * 90, flush=True)
    print("RANKED FLOOR TABLE (G ascending — worst floor first):", flush=True)
    print(f"{'G':>6}  {'kernel':>14} {'shape':>16} {'band':>4} "
          f"{'codegen':>10} {'seed_us':>9} {'tc_us':>9}  flag", flush=True)
    ranked = sorted([r for r in all_results if r["G"] is not None],
                    key=lambda r: r["G"])
    for r in ranked:
        floor = "FLOOR-LOSS" if r["G"] < 1 - EPS else ""
        print(f"{r['G']:>6.3f}  {r['kernel']:>14} "
              f"{str(tuple(r['shape'])):>16} {r['band']:>4} "
              f"{r['seed_codegen']:>10} "
              f"{r['seed_us']:>9.1f} {r['tc_us']:>9.1f}  {floor}", flush=True)

    print("\nPER-KERNEL GEOMEAN G (train):", flush=True)
    for kernel in kernels:
        gm = per_kernel_g.get(kernel)
        print(f"  {kernel:>14}: {('%.4f' % gm) if gm else 'n/a'}", flush=True)
    overall = geomean([v for v in per_kernel_g.values() if v is not None])
    print(f"  {'OVERALL':>14}: {('%.4f' % overall) if overall else 'n/a'}",
          flush=True)

    print(f"\nFLOOR LOSSES (G < {1 - EPS:.2f}):", flush=True)
    if floor_losses:
        for r in sorted(floor_losses, key=lambda r: r["G"]):
            print(f"  {r['kernel']}({r['shape'][0]},{r['shape'][1]}) "
                  f"G={r['G']:.3f} {r['seed_codegen']} band={r['band']}",
                  flush=True)
    else:
        print("  (none — all train shapes at/above floor)", flush=True)

    print("\nCORRECTNESS FAILURES:", flush=True)
    print(("  " + json.dumps(fails)) if fails else "  (none)", flush=True)
    print("\nOOM / SKIPPED:", flush=True)
    print(("  " + json.dumps(ooms)) if ooms else "  (none)", flush=True)

    merged = {
        "gpu": gpu, "eps": EPS, "n_runs": N_RUNS,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_kernel_geomean_G": per_kernel_g,
        "overall_geomean_G": overall,
        "results": all_results,
        "floor_losses": floor_losses,
        "correctness_failures": fails,
        "ooms": ooms,
    }
    with open(os.path.join(os.path.abspath(LOG_DIR), "floor_sweep_merged.json"),
              "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(os.path.abspath(LOG_DIR), 'floor_sweep_merged.json')}]",
          flush=True)


if __name__ == "__main__":
    main()
