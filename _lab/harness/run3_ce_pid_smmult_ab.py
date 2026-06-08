"""RUN-3 EDIT-PID sm_mult-isolation A/B — is the derived num_sm_multiplier load-bearing?

Question (hub-driven simplification review): the EDIT-PID branch computes
num_sm_multiplier = clamp(np2(ceil(grid_rows/num_sm)), 1, 32) via a grid_rows loop +
get_num_sm import (the bulk of the branch's LOC). Could the branch drop that and
default sm_mult=1 (or use a constant 32) with no perf loss? maxnreg carries most of
the EDIT-PID win, but maxnreg is only legal on a persistent pid_type, so we KEEP
persistent_interleaved + maxnreg=64 and vary ONLY num_sm_multiplier.

Each arm starts from the EXACT LIVE SHIPPED SEED (carries eviction/warps/chunk/etc.)
and overrides only num_sm_multiplier. Arms per shape:
  flat_ref          : the pre-EDIT-PID baseline (pid_type='flat', no sm_mult/maxnreg)
  shipped           : live seed verbatim (formula sm_mult, 32/32/16 on the 3 shapes)
  smmult1           : sm_mult=1  (the proposal: drop the formula, default grid=#SMs)
  smmult32          : sm_mult=32 (constant; what the formula already emits at M>=4096)

Reported: us, shipped/arm (so >1 means arm SLOWER than shipped), arm/tc, and the
round-trip-surviving num_sm_multiplier (proves the override took, not silently popped).
do_bench median-of-7, correctness-gated (fp32 ref), ONE process, serial.

The 3 EDIT-PID firing shapes (formula sm_mult in parens):
  (4096, 98304)   sm_mult=32
  (8192, 128256)  sm_mult=32 (ceil(8192/132)=63 -> np2=64 -> clamp 32)
  (2048, 256000)  sm_mult=16 (ceil(2048/132)=16 -> np2=16)

REQ-GPU before running (hub holds the timing token; nvidia-smi idle-gated). From /tmp:
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_ce_pid_smmult_ab.py
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
    check_correct,
    codegen_kind,
    get_seed,
)
from triton.testing import do_bench  # noqa: E402

LOG_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "logs", "run3"))
CACHE_PATH = os.path.join(LOG_DIR, "oracle_cache.json")

# The 3 EDIT-PID firing CE shapes (train split): M, N.
SHAPES = [(4096, 98304), (8192, 128256), (2048, 256000)]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _run(fn_obj, args, ref, out_extract, cfg, watch_keys):
    try:
        k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        norm = dict(b._config)
        dropped = [kk for kk in watch_keys if norm.get(kk) != cfg.get(kk)]
        ok, err = check_correct(out_extract(b(*args)), ref)
        med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
        return {
            "us": med * 1000 if med else None,
            "spread": sp,
            "codegen": codegen_kind(b),
            "correct": ok,
            "maxerr": err,
            "dropped_levers": dropped,
            "pid": norm.get("pid_type"),
            "sm_mult_norm": norm.get("num_sm_multiplier"),
            "maxnreg_norm": norm.get("maxnreg"),
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def run_shape(M, N, oracle_us=None):
    fn, builder, tc_ref = KERNELS["cross_entropy"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, "fp32 required"

    # The EXACT live shipped seed (carries eviction, warps, chunk, the formula sm_mult).
    seed_live = dict(get_seed(fn, args)[0])
    shipped_smmult = seed_live.get("num_sm_multiplier")

    # flat reference: strip the 3 persistent-only keys -> pre-EDIT-PID baseline.
    flat_cfg = {
        kk: v
        for kk, v in seed_live.items()
        if kk not in ("pid_type", "num_sm_multiplier", "maxnreg")
    }
    flat_cfg["pid_type"] = "flat"

    # sm_mult variants: live seed, override ONLY num_sm_multiplier.
    def with_smmult(val):
        c = dict(seed_live)
        c["num_sm_multiplier"] = val
        return c

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    arms = {
        "flat_ref": (flat_cfg, ["pid_type"]),
        "shipped": (dict(seed_live), ["num_sm_multiplier", "maxnreg", "pid_type"]),
        "smmult1": (with_smmult(1), ["num_sm_multiplier", "maxnreg", "pid_type"]),
        "smmult32": (with_smmult(32), ["num_sm_multiplier", "maxnreg", "pid_type"]),
    }
    results = {n: _run(fn, args, ref, out_extract, c, w) for n, (c, w) in arms.items()}

    shipped_us = results["shipped"].get("us")
    print(
        f"\n=== cross_entropy({M},{N}) === tc={tc_med * 1000:.1f}us "
        f"oracle={oracle_us}us shipped_smmult={shipped_smmult} "
        f"shipped={shipped_us:.1f}us"
        if shipped_us
        else f"\n=== cross_entropy({M},{N}) === shipped FAILED: {results['shipped']}",
        flush=True,
    )
    print(
        f"  {'arm':>12} {'us':>9} {'shipped/arm':>11} {'arm/tc':>7} "
        f"{'pid':>22} {'smN':>4} {'mnr':>4} {'drop':>14} corr",
        flush=True,
    )
    for name, r in results.items():
        if r.get("us"):
            ratio = shipped_us / r["us"] if shipped_us else float("nan")
            drop = ",".join(r["dropped_levers"]) or "-"
            print(
                f"  {name:>12} {r['us']:>9.1f} {ratio:>11.3f} "
                f"{tc_med * 1000 / r['us']:>7.3f} {str(r['pid']):>22} "
                f"{str(r['sm_mult_norm']):>4} {str(r['maxnreg_norm']):>4} "
                f"{drop:>14} {'OK' if r['correct'] else 'BAD'}",
                flush=True,
            )
        else:
            print(f"  {name:>12} -> {r}", flush=True)
    return {
        "shape": [M, N],
        "tc_us": tc_med * 1000,
        "oracle_us": oracle_us,
        "shipped_smmult": shipped_smmult,
        "arms": results,
    }


def main():
    print(
        f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES', '?')} helion={helion.__file__}",
        flush=True,
    )
    cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {"entries": {}}
    out = []
    for M, N in SHAPES:
        e = cache["entries"].get(f"cross_entropy:{M}x{N}")
        oracle_us = round(e["oracle_us"], 1) if e and e.get("oracle_us") else None
        out.append(run_shape(M, N, oracle_us))
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, "ce_pid_smmult_ab.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {path}]", flush=True)


if __name__ == "__main__":
    main()
