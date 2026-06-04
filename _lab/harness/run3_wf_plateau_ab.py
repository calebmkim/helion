"""RUN-3 welford "is the oracle a peak or a plateau?" decomposition.

Question (the Phase-2b seedable-vs-oracle-only split): the welford oracle is an
EXOTIC config bundle (e.g. (4096,16384): block_sizes=[2,16384,16384], num_stages=7,
range_unroll=[4,4,1], range_num_stages=[0,2,0], multi_buffers=[None,True,False]).
A seed heuristic can emit block_sizes / num_warps / num_stages, but NOT the per-range
codegen knobs (range_unroll/range_num_stages/range_multi_buffers are autotuner-only).

So: how much of the oracle win is in the SEEDABLE part (block_sizes + warps + stages)
vs the UN-seedable per-range codegen bundle? Build a ladder of arms for each gap:

  seed                         -- current champion
  ob_only      = oracle block_sizes, ELSE seed/default codegen   (SEEDABLE)
  ob+warps     = + oracle num_warps                              (SEEDABLE)
  ob+warps+stg = + oracle num_stages                             (SEEDABLE)
  ob+range     = oracle block_sizes + the per-range codegen knobs (NOT seedable)
  oracle_full  = the exact verbatim oracle config                 (target)

If an early SEEDABLE rung already lands within eps of oracle_full -> the win is a
PLATEAU the heuristic can seed (skip the codegen bundle). If only oracle_full / ob+range
hits it -> it's a PEAK that needs the un-seedable knobs (honest Product-B for that part).

do_bench median-of-N, correctness-gated vs fp32 eager layer_norm ref, vs tc-default.

Invocation (from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_wf_plateau_ab.py
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
CACHE = os.path.join(LOG_DIR, "oracle_cache.json")

# the welford GAP shapes (cached FULL oracle for the wide two; quick for narrow)
SHAPES = [(4096, 16384), (32768, 8192), (16384, 768)]

RANGE_KNOBS = [
    "range_unroll_factors",
    "range_warp_specializes",
    "range_num_stages",
    "range_multi_buffers",
    "range_flattens",
]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def run_shape(M, N, oracle_cfg):
    fn, builder, tc_ref = KERNELS["welford"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"welford non-fp32 {a.dtype}"

    seed = dict(get_seed(fn, args)[0])
    ob = list(oracle_cfg["block_sizes"])
    ow = oracle_cfg.get("num_warps")
    ostg = oracle_cfg.get("num_stages")

    def base_seedish():
        # oracle block_sizes, everything else from the seed (default codegen).
        c = dict(seed)
        c["block_sizes"] = list(ob)
        return c

    arms = {}
    arms["seed"] = dict(seed)
    arms["ob_only"] = base_seedish()
    c = base_seedish(); c["num_warps"] = ow; arms["ob+warps"] = c
    c = dict(c); c["num_stages"] = ostg; arms["ob+warps+stg"] = c
    # oracle block_sizes + the per-range codegen bundle (NOT seedable), seed warps/stages
    c = base_seedish()
    for k in RANGE_KNOBS:
        if k in oracle_cfg:
            c[k] = oracle_cfg[k]
    arms["ob+rangeknobs"] = c
    # the exact verbatim oracle
    arms["oracle_full"] = dict(oracle_cfg)

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0], "tc ref incorrect"
    tc_med, _ = _bench(lambda: tc(args))

    out = {"M": M, "N": N, "tc_us": tc_med * 1000, "seed_bs": list(seed["block_sizes"]),
           "oracle_bs": ob, "arms": {}}
    for name, cfg in arms.items():
        try:
            k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
            b = k.bind(args); b.ensure_config_exists(args)
            ok, err = check_correct(out_extract(b(*args)), ref)
            if not ok:
                out["arms"][name] = {"correct": False, "maxerr": err}
                continue
            med, sp = _bench(lambda: b(*args))
            nb = dict(b._config)
            out["arms"][name] = {
                "us": med * 1000, "spread": sp, "codegen": codegen_kind(b),
                "norm_bs": nb.get("block_sizes"), "w": nb.get("num_warps"),
                "stg": nb.get("num_stages"), "correct": True, "maxerr": err,
            }
        except Exception as e:  # noqa: BLE001
            out["arms"][name] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


def main():
    print(f"helion={helion.__file__}", flush=True)
    cache = json.load(open(CACHE))["entries"]
    results = []
    for (M, N) in SHAPES:
        key = f"welford:{M}x{N}"
        oc = cache[key]["oracle_cfg"]
        eff = cache[key]["effort"]
        r = run_shape(M, N, oc)
        r["oracle_effort"] = eff
        results.append(r)
        of = r["arms"].get("oracle_full", {}).get("us")
        print(f"\n=== welford({M},{N})  tc={r['tc_us']:.1f}us  "
              f"seed_bs={r['seed_bs']} -> oracle_bs={r['oracle_bs']}  ({eff} oracle) ===",
              flush=True)
        for name, a in r["arms"].items():
            if "us" not in a:
                print(f"  {name:16s} {'CORRECT-FAIL' if a.get('correct')==False else a.get('error','?')}")
                continue
            vs_full = (a["us"] / of) if of else None
            print(f"  {name:16s} {a['us']:9.1f}us  /oracle={vs_full:.3f}  "
                  f"/tc={a['us']/r['tc_us']:.3f}  ({a['codegen']}, w{a['w']}, stg{a['stg']}, "
                  f"sp={a['spread']:.3f})", flush=True)

    json.dump({"results": results}, open(os.path.join(LOG_DIR, "wf_plateau_ab.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(LOG_DIR, 'wf_plateau_ab.json')}", flush=True)


if __name__ == "__main__":
    main()
