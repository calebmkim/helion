"""RUN-3 cross_entropy persistent-vs-looped matched-lever A/B (NO autotune).

The fresh oracle (run3_oracle.py, quick) shows the CE seed's LOOPED path
(reduction_loops=[16384], forced by MULTILOAD_PERSIST_MAX_BYTES=131072) is ~1.6x
slower than a PERSISTENT oracle that ALSO beats tc-default at the persist boundary
(V<=50304). At the widest V the quick oracle stayed looped and lost to tc -- a
SUSPECT result (quick under-exploration). This script settles it WITHOUT the
autotuner: for each CE shape it benches a small matched set of explicit configs,
varying ONLY reduction_loops (+ a couple warp/stage variants), do_bench median-of-7,
correctness-gated, all in ONE process so the ratios are noise-robust.

Arms (all from the live seed config, mutating only the named field(s)):
  seed_looped   : the live seed as-is (reduction_loops=[16384], w32, ns1)
  persist       : reduction_loops=[None]   (else identical to seed)
  persist_w16   : reduction_loops=[None], num_warps=16
  persist_ns4   : reduction_loops=[None], num_warps=16, num_stages=4  (V=50304 oracle flavor)
  tc_default    : torch.compile default of F.cross_entropy

Reports per arm: median us, seed_looped/arm ratio, arm/tc ratio, correctness.
A persist arm that is FASTER than both seed_looped and tc_default confirms the cap
is wrong (the oracle's persistent win is real, not a quick-autotune artifact).

Invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_ce_persist_ab.py \
    8192x50257 4096x98304 8192x128256 2048x256000   # MxN shapes; default = a wide set
"""

from __future__ import annotations

import json
import os
import sys

import torch

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import helion  # noqa: E402

from run2_measure_g import (  # noqa: E402
    KERNELS,
    N_RUNS,
    get_seed,
    check_correct,
    codegen_kind,
)
from triton.testing import do_bench  # noqa: E402

LOG_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "logs", "run3"))


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    return s[len(s) // 2], (s[-1] - s[0]) / s[len(s) // 2] if s[len(s) // 2] else None


def _run_cfg(fn_obj, args, cfg: dict):
    """Build helion.kernel(configs=[cfg]), bind, return (bound, normalized_cfg, codegen)."""
    k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b, dict(b._config), codegen_kind(b)


def run_shape(M, N):
    fn, builder, tc_ref = KERNELS["cross_entropy"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32

    seed_raw, _ = get_seed(fn, args)
    seed = dict(seed_raw)

    arms = {
        "seed_looped": dict(seed),
        "persist": {**seed, "reduction_loops": [None]},
        "persist_w16": {**seed, "reduction_loops": [None], "num_warps": 16},
        "persist_ns4": {**seed, "reduction_loops": [None], "num_warps": 16,
                        "num_stages": 4},
    }

    # tc default
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    out_tc = out_extract(tc(args))
    ok_tc, _ = check_correct(out_tc, ref)
    assert ok_tc, f"tc FAIL ce {(M,N)}"
    tc_med, tc_sp = _bench(lambda: tc(args))

    results = {}
    for name, cfg in arms.items():
        try:
            b, norm, cg = _run_cfg(fn, args, cfg)
            out = out_extract(b(*args))
            ok, err = check_correct(out, ref)
            med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
            results[name] = {
                "cfg_rl": norm.get("reduction_loops"),
                "cfg_w": norm.get("num_warps"),
                "cfg_ns": norm.get("num_stages"),
                "codegen": cg, "correct": ok, "maxerr": err,
                "us": med * 1000 if med else None, "spread": sp,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[name] = {"oom": True}
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:200]}

    seed_us = results["seed_looped"].get("us")
    print(f"\n=== cross_entropy({M},{N}) ===  tc_default={tc_med*1000:.1f}us "
          f"(spread {tc_sp:.2f})", flush=True)
    print(f"  {'arm':>12} {'rl':>8} {'w':>3} {'ns':>3} {'codegen':>10} "
          f"{'us':>9} {'seed/arm':>9} {'arm/tc':>8} {'spread':>7} corr", flush=True)
    for name, r in results.items():
        if "us" in r and r["us"]:
            sa = seed_us / r["us"] if seed_us else float("nan")
            at = tc_med * 1000 / r["us"]
            print(f"  {name:>12} {str(r['cfg_rl']):>8} {r['cfg_w']:>3} "
                  f"{r['cfg_ns']:>3} {r['codegen']:>10} {r['us']:>9.1f} "
                  f"{sa:>9.3f} {at:>8.3f} {r['spread']:>7.2f} "
                  f"{'OK' if r['correct'] else 'BAD'}", flush=True)
        else:
            print(f"  {name:>12} -> {r}", flush=True)
    return {"shape": [M, N], "tc_us": tc_med * 1000, "arms": results}


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} helion={helion.__file__}", flush=True)
    shape_args = sys.argv[1:] or [
        "8192x49152", "4096x50304", "8192x50257", "4096x98304",
        "8192x128256", "2048x256000",
    ]
    out = []
    for s in shape_args:
        M, N = (int(x) for x in s.split("x"))
        try:
            out.append(run_shape(M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {s}: {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "ce_persist_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'ce_persist_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
