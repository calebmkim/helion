"""RUN-3 EDIT-PID lever DECOMPOSITION — which lever(s) carry the wide-CE 1.62x?

Guardrail 5b (hub): the full-oracle CE(4096,98304) config (588us, 1.62x the looped
seed 955us, ~5% off tc) is a 7-lever bundle. Add ONE lever at a time to the CURRENT
SEED, each benched vs BOTH the verbatim oracle (588us target) AND the seed (955us
floor), to find the CARRIER lever(s). The pid CLUSTER is decomposed finely:
pid-only / +num_sm_multiplier / +maxnreg (per 5b: likely the occupancy change is the
carrier, pipelining a passenger — MEASURE). Each arm logs round-trip survival of its
levers (5a) — a lever that silently drops is flagged.

do_bench median-of-7, correctness-gated, ONE process. Eviction is handled separately
(run3_ce_evict_ab.py: the re-read eviction = 1.31x alone). This script isolates the
PID/pipelining cluster on top of (and without) eviction.

REQ-GPU before running (hub holds the timing token). Invocation (from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_ce_pid_decomp.py
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
CACHE_PATH = os.path.join(LOG_DIR, "oracle_cache.json")

# Individual oracle levers (from the full CE(4096,98304) oracle). Each is added to
# the seed independently; the pid cluster is split so we see which sub-lever carries.
PID = {"pid_type": "persistent_interleaved", "num_sm_multiplier": 32}
PID_MAXNREG = {"pid_type": "persistent_interleaved", "num_sm_multiplier": 32,
               "maxnreg": 64}
PIPELINE = {"num_stages": 4, "range_unroll_factors": [4], "range_num_stages": [2],
            "range_flattens": [False]}
CHUNK = {"reduction_loops": [4096]}
EVICT = {"load_eviction_policies": ["", "", "last", "first", "last"]}


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    return s[len(s) // 2], (s[-1] - s[0]) / s[len(s) // 2] if s[len(s) // 2] else None


def _run(fn_obj, args, ref, out_extract, cfg, watch_keys):
    try:
        k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        norm = dict(b._config)
        # round-trip survival: did the levers we set survive normalize()?
        dropped = [kk for kk in watch_keys
                   if kk in cfg and norm.get(kk) != cfg[kk]]
        ok, err = check_correct(out_extract(b(*args)), ref)
        med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
        return {"us": med * 1000 if med else None, "spread": sp,
                "codegen": codegen_kind(b), "correct": ok, "maxerr": err,
                "dropped_levers": dropped, "pid": norm.get("pid_type")}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:160]}


def run_shape(M, N, oracle_us=None):
    fn, builder, tc_ref = KERNELS["cross_entropy"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32
    seed = dict(get_seed(fn, args)[0])

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    # additive arms: seed + ONE lever (or sub-cluster)
    arms = {
        "seed": (dict(seed), []),
        "seed+pid": ({**seed, **PID}, list(PID)),
        "seed+pid+maxnreg": ({**seed, **PID_MAXNREG}, list(PID_MAXNREG)),
        "seed+pipeline": ({**seed, **PIPELINE}, list(PIPELINE)),
        "seed+chunk": ({**seed, **CHUNK}, list(CHUNK)),
        "seed+evict": ({**seed, **EVICT}, list(EVICT)),
        # carrier candidate combos
        "seed+pid+maxnreg+evict": ({**seed, **PID_MAXNREG, **EVICT},
                                   list(PID_MAXNREG) + list(EVICT)),
        "seed+pid+maxnreg+pipeline": ({**seed, **PID_MAXNREG, **PIPELINE},
                                      list(PID_MAXNREG) + list(PIPELINE)),
        "full_bundle": ({**seed, **PID_MAXNREG, **PIPELINE, **CHUNK, **EVICT},
                        list(PID_MAXNREG) + list(PIPELINE) + list(CHUNK) + list(EVICT)),
    }
    results = {n: _run(fn, args, ref, out_extract, c, w) for n, (c, w) in arms.items()}

    so = results["seed"].get("us")
    print(f"\n=== cross_entropy({M},{N}) === tc={tc_med*1000:.1f}us "
          f"oracle_target={oracle_us}us seed={so:.1f}us", flush=True)
    print(f"  {'arm':>26} {'us':>9} {'seed/arm':>9} {'arm/tc':>7} {'pid':>22} "
          f"{'dropped':>8} corr", flush=True)
    for name, r in results.items():
        if r.get("us"):
            sa = so / r["us"] if so else float("nan")
            drop = ",".join(r["dropped_levers"]) or "-"
            print(f"  {name:>26} {r['us']:>9.1f} {sa:>9.3f} {tc_med*1000/r['us']:>7.3f} "
                  f"{str(r['pid']):>22} {drop:>8} {'OK' if r['correct'] else 'BAD'}",
                  flush=True)
        else:
            print(f"  {name:>26} -> {r}", flush=True)
    return {"shape": [M, N], "tc_us": tc_med * 1000, "seed_us": so,
            "oracle_target_us": oracle_us, "arms": results}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    # oracle target for (4096,98304) from the cached full oracle
    oracle_us = None
    if os.path.exists(CACHE_PATH):
        c = json.load(open(CACHE_PATH))
        e = c["entries"].get("cross_entropy:4096x98304")
        if e:
            oracle_us = round(e.get("oracle_us"), 1)
    out = [run_shape(4096, 98304, oracle_us)]
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "ce_pid_decomp.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'ce_pid_decomp.json')}]", flush=True)


if __name__ == "__main__":
    main()
