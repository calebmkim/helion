"""RUN-3 wide-CE lever isolation: which lever(s) carry the full oracle's 1.62x?

The full-effort oracle on CE(4096,98304) = 588us (seed 955us, seed/oracle=1.624)
uses a strategy outside the seed's design space:
  reduction_loops=[4096], pid_type='persistent_interleaved', num_sm_multiplier=32,
  maxnreg=64, num_stages=4, range_unroll_factors=[4], range_num_stages=[2],
  range_flattens=[False], indexing mostly tensor_descriptor.

This A/Bs (NO autotune, do_bench median-of-7, correctness-gated):
  - SEED (baseline): the live heuristic seed.
  - ADDITIVE: seed + ONE oracle lever at a time (does that lever alone help?).
  - ABLATION: full oracle MINUS one lever at a time (does removing it hurt?).
  - FULL ORACLE (verbatim): the 588us target.
All configs are normalized (helion.Config -> bind -> ensure_config_exists) so invalid
combos are caught/repaired and the NORMALIZED config actually run is reported.

The additive arm that moves SEED toward the oracle the most, and the ablation arm
whose removal hurts the most, identify the load-bearing lever(s) -> the heuristic
branch. This is the matched-lever method (re-bench the FULL VERBATIM oracle as the
baseline; never re-pair an isolated lever).

Invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_ce_pid_ab.py \
    4096x98304 8192x128256
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

# Lever GROUPS, defined as FIELD NAMES. The baseline is the VERBATIM cached full
# oracle (reproducible to 589us); each ablation resets the group's fields to the
# seed's value (or drops them so normalize fills the default). This is the correct
# oracle-is-a-bundle method: ablate from the real verbatim oracle, never reconstruct
# a subset (a reconstruction that omits indexing/eviction measured 1005us != the
# real 589us -- those fields are part of the coupled winning bundle).
LEVER_FIELDS = {
    "chunk": ["reduction_loops"],
    "pidcluster": ["pid_type", "num_sm_multiplier", "maxnreg"],
    "pipeline": ["num_stages", "range_unroll_factors", "range_num_stages",
                 "range_flattens", "range_multi_buffers", "range_warp_specializes"],
    "indexing": ["indexing"],
    "eviction": ["load_eviction_policies"],
}


def _load_oracle_cfg(M, N):
    """The VERBATIM cached full-effort oracle config for this shape (the answer key)."""
    cache = json.load(open(CACHE_PATH))
    e = cache["entries"][f"cross_entropy:{M}x{N}"]
    return dict(e["oracle_cfg"]), e.get("effort"), e.get("oracle_us")


CACHE_PATH = os.path.join(LOG_DIR, "oracle_cache.json")


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    return s[len(s) // 2], (s[-1] - s[0]) / s[len(s) // 2] if s[len(s) // 2] else None


def _try(fn_obj, args, ref, out_extract, cfg):
    try:
        k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        ok, err = check_correct(out_extract(b(*args)), ref)
        med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
        return {"us": med * 1000 if med else None, "spread": sp,
                "codegen": codegen_kind(b), "correct": ok, "maxerr": err,
                "norm_rl": dict(b._config).get("reduction_loops"),
                "norm_pid": dict(b._config).get("pid_type"),
                "norm_ns": dict(b._config).get("num_stages")}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:160]}


def run_shape(M, N):
    fn, builder, tc_ref = KERNELS["cross_entropy"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32
    seed = dict(get_seed(fn, args)[0])
    oracle, oeffort, ocached_us = _load_oracle_cfg(M, N)  # VERBATIM cached oracle

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    arms = {"seed": dict(seed), "full_oracle": dict(oracle)}
    # ADDITIVE: seed + one lever (the lever's fields copied verbatim from the oracle).
    for name, fields in LEVER_FIELDS.items():
        add = {f: oracle[f] for f in fields if f in oracle}
        arms[f"seed+{name}"] = {**seed, **add}
    # ABLATION: VERBATIM oracle MINUS one lever (its fields reset to the seed's value
    # if the seed sets it, else dropped so normalize fills the default).
    for name, fields in LEVER_FIELDS.items():
        ablated = dict(oracle)
        for f in fields:
            if f in seed:
                ablated[f] = seed[f]
            else:
                ablated.pop(f, None)
        arms[f"oracle-{name}"] = ablated

    results = {n: _try(fn, args, ref, out_extract, c) for n, c in arms.items()}

    print(f"\n=== cross_entropy({M},{N}) ===  tc={tc_med*1000:.1f}us  "
          f"[V*4={N*4//1024}KiB]  cached_oracle({oeffort})={ocached_us:.1f}us", flush=True)
    so = results.get("seed", {}).get("us")
    oo = results.get("full_oracle", {}).get("us")
    order = (["seed", "full_oracle"]
             + [f"seed+{k}" for k in LEVER_FIELDS]
             + [f"oracle-{k}" for k in LEVER_FIELDS])
    print(f"  {'arm':>18} {'pid':>22} {'rl':>8} {'ns':>3} {'us':>9} "
          f"{'/seed':>6} {'/oracle':>7} {'/tc':>6} corr", flush=True)
    for name in order:
        r = results.get(name, {})
        if r.get("us"):
            vs_seed = so / r["us"] if so else float("nan")
            vs_or = r["us"] / oo if oo else float("nan")
            print(f"  {name:>18} {str(r['norm_pid']):>22} {str(r['norm_rl']):>8} "
                  f"{r['norm_ns']:>3} {r['us']:>9.1f} {vs_seed:>6.2f} "
                  f"{vs_or:>7.3f} {tc_med*1000/r['us']:>6.3f} "
                  f"{'OK' if r['correct'] else 'BAD'}", flush=True)
        else:
            print(f"  {name:>18} -> {r}", flush=True)
    return {"shape": [M, N], "tc_us": tc_med * 1000,
            "seed_us": so, "oracle_us": oo, "arms": results}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    shape_args = sys.argv[1:] or ["4096x98304", "8192x128256"]
    out = []
    for s in shape_args:
        M, N = (int(x) for x in s.split("x"))
        try:
            out.append(run_shape(M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {s}: {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "ce_pid_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'ce_pid_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
