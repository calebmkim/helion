"""PR #2866 firing probe — the go/no-go gate BEFORE any timing.

For one representative shape per kernel, bind (compile-config only, NO timing loop) and assert:
  1. TritonPointwiseSeedHeuristic.is_eligible == True  (the fact fired: no reduction/matmul/accum)
  2. a non-empty compiler_seed_config exists
  3. seed config != base_default config (configs_differ) — the heuristic actually does something
  4. for the levers: contig_block_ids >= 2 (transposed_out_add) / sfu_ops >= 9 (heavy_transcendental_1d)

Emits a table + a PASS/FAIL verdict. Read-only-ish: it binds and reads facts; it does NOT run
the interleaved timing loop. Foreground, serial.

Usage:
  PYTHONPATH=<worktree> HELION_AUTOTUNE_EFFORT=none CUDA_VISIBLE_DEVICES=0 \
    python perf-repro/pr2866_firing_probe.py
"""

from __future__ import annotations

import os
import sys

_PERF = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(_PERF, ".."))
for _d in (_WT, os.path.join(_WT, "examples"), os.path.join(_PERF, "deps")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402

assert helion.__file__.startswith(_WT), f"helion not under worktree: {helion.__file__}"
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    TritonPointwiseSeedHeuristic,
)

import pointwise_builders as PB  # noqa: E402

# one representative shape per kernel (the PR's literal claim shape for the levers)
PROBE = {
    "swiglu": (16384, 2048),
    "geglu": (8192, 9216),
    "residual_add": (8192, 8192),
    "relu_squared": (16384, 8192),
    "bias_gelu": (16384, 4096),
    "dyt": (16384, 2048),
    "rope_fwd": (1, 32, 2048, 256),
    "transposed_out_add": (2048, 512),
    "heavy_transcendental_1d": (16777216,),
}


def probe_one(kernel, shape):
    kfn, args, ref, rt, at, tc, kind = PB.build(kernel, shape, torch.bfloat16)
    kfn.reset()
    bound = kfn.bind(args)
    env, dev_ir, spec = bound.env, bound.host_function.device_ir, bound.config_spec

    eligible = TritonPointwiseSeedHeuristic.is_eligible(env, dev_ir)
    seeds = list(spec.compiler_seed_configs)
    fired = list(spec.autotuner_heuristics)
    with bound.env:
        base = spec._base_default_config()
    seed = seeds[0] if seeds else None
    seed_bs = seed.config.get("block_sizes") if seed else None
    seed_nw = seed.config.get("num_warps") if seed else None
    base_bs = base.config.get("block_sizes") if base else None
    differ = (seed.config != base.config) if seed else False

    fact = spec.pointwise_facts[0] if getattr(spec, "pointwise_facts", None) else None
    contig = list(getattr(fact, "contig_block_ids", []) or []) if fact else None
    sfu = getattr(fact, "sfu_ops", None) if fact else None

    # per-kernel expectation
    ok = eligible and (seed is not None) and differ
    note = ""
    if kernel == "transposed_out_add":
        lever_ok = contig is not None and len(contig) >= 2
        ok = ok and lever_ok
        note = f"contig_block_ids={contig} (want >=2)"
    elif kernel == "heavy_transcendental_1d":
        lever_ok = sfu is not None and sfu >= 9
        ok = ok and lever_ok
        note = f"sfu_ops={sfu} (want >=9), num_warps={seed_nw}"
    elif kernel == "rope_fwd":
        note = f"seed_bs={seed_bs} (slab-fold -> ~[1,1])"

    del bound
    torch.cuda.empty_cache()
    return {
        "kernel": kernel, "shape": shape, "eligible": eligible,
        "fired": fired, "seed_bs": seed_bs, "seed_nw": seed_nw, "base_bs": base_bs,
        "differ": differ, "contig": contig, "sfu": sfu, "ok": ok, "note": note,
    }


def main():
    print(f"helion={helion.__file__}")
    print(f"{'kernel':24} {'elig':5} {'differ':6} {'seed_bs':18} {'base_bs':12} {'OK':3} note")
    print("-" * 110)
    results = []
    for k, shp in PROBE.items():
        try:
            r = probe_one(k, shp)
        except Exception as e:  # noqa: BLE001
            import traceback
            r = {"kernel": k, "shape": shp, "ok": False, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        results.append(r)
        if "error" in r:
            print(f"{k:24} PROBE FAILED: {r['error']}")
        else:
            print(f"{k:24} {str(r['eligible']):5} {str(r['differ']):6} "
                  f"{str(r['seed_bs']):18} {str(r['base_bs']):12} "
                  f"{'YES' if r['ok'] else 'NO':3} {r['note']}")
    n_ok = sum(1 for r in results if r.get("ok"))
    print("-" * 110)
    print(f"VERDICT: {n_ok}/{len(results)} kernels fire the pointwise seed correctly")
    if n_ok != len(results):
        print("GATE: FAIL — do NOT proceed to timing. Inspect the NO rows above.")
        sys.exit(1)
    print("GATE: PASS — pointwise seed fires on every kernel; safe to proceed to smoke.")


if __name__ == "__main__":
    main()
