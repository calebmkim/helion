"""RUN-3 — small-N occupancy->{M_BLOCK,warps} CONTOUR generality A/B (do_bench, NO autotune).

The softmax(131072,256) decomp found the carrier = COUPLED {M_BLOCK 8->16, num_warps 4->16}. Before building an
occupancy-keyed rule, confirm GENERALITY (dodge the run-2 occupancy-overfitting-trap): does coupling M_BLOCK+warps
UP help OTHER high-OCC small-N shapes, and stay NEUTRAL/worse on LOW-OCC controls (which the occ-warps A/B showed
want w4)? If high-OCC shapes improve with the coupled bump AND low-OCC don't -> a clean occupancy contour. If the
exact M_BLOCK/warps that wins varies per shape -> a fit, not a rule.

Per shape, arms = seed + coupled {M_BLOCK x2/x4, warps 8/16}. The seed's M_BLOCK is the floor (read it live).
do_bench median-of-N, correctness-gated, fp32 asserted, ONE process. Foreground (NO bg GPU).

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_smalln_occ_contour_ab.py
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
from helion.runtime import get_num_sm  # noqa: E402
from triton.testing import do_bench  # noqa: E402

LOG_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "logs", "run3"))

# (label, M, N, OCC-class). HIGH-OCC small-N should benefit from coupled bump; LOW-OCC controls should not.
CASES = [
    ("hiOcc", 131072, 256),
    ("hiOcc", 262144, 128),
    ("hiOcc", 131072, 512),
    ("loOcc-control", 16384, 512),
    ("loOcc-control", 4096, 256),
]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _run(fn_obj, args, ref, out_extract, cfg):
    try:
        k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        ok, err = check_correct(out_extract(b(*args)), ref)
        med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
        return {"us": med * 1000 if med else None, "spread": sp, "correct": ok}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:140]}


def run_case(label, M, N):
    fn, builder, tc_ref = KERNELS["softmax"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"softmax{(M, N)} non-fp32"
    seed = dict(get_seed(fn, args)[0])
    b0 = fn.bind(args)
    spec = b0.env.config_spec
    f = spec.reduction_facts[0]
    ridx = spec.block_sizes.block_id_to_index(f.block_id)
    seed_bs = list(seed["block_sizes"])
    m_idx = next(i for i in range(len(seed_bs)) if i != ridx)
    m0 = seed_bs[m_idx]
    gr = 1
    for mb in f.m_block_ids:
        gr *= b0.env.size_hint(b0.env.block_sizes[mb].size)
    occ = gr / max(1, get_num_sm(b0.env.device))

    def bs_with_m(mult):
        bs = list(seed_bs)
        bs[m_idx] = m0 * mult
        return bs

    arms = {"seed": dict(seed)}
    for mmult in (2, 4):
        for w in (8, 16):
            arms[f"M x{mmult}+w{w}"] = {**seed, "block_sizes": bs_with_m(mmult),
                                       "num_warps": w}

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    res = {nm: _run(fn, args, ref, out_extract, c) for nm, c in arms.items()}
    seed_us = res["seed"].get("us")
    best = min((r["us"] for r in res.values() if r.get("us")), default=None)
    best_nm = next((nm for nm, r in res.items() if r.get("us") == best), "?")
    print(f"\n=== softmax({M},{N}) [{label}] OCC={occ:.0f} seed_M={m0} tc={tc_med*1000:.1f}us "
          f"BEST={best_nm} ===", flush=True)
    for nm in arms:
        r = res[nm]
        if r.get("us"):
            sa = seed_us / r["us"] if seed_us else float("nan")
            star = " <-BEST" if r["us"] == best else ""
            print(f"  {nm:14s} {r['us']:8.1f}us  seed/arm={sa:6.3f}  "
                  f"{'OK' if r['correct'] else 'BAD'}{star}", flush=True)
        else:
            print(f"  {nm:14s} -> {r.get('error')}", flush=True)
    return {"label": label, "shape": [M, N], "occ": occ, "seed_M": m0,
            "tc_us": tc_med * 1000, "best": best_nm, "arms": res}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    out = []
    for label, M, N in CASES:
        try:
            out.append(run_case(label, M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {label} softmax({M},{N}): {type(e).__name__}: {e}"[:200],
                  flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "smalln_occ_contour_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n--- OCC vs BEST coupled-bump (sorted by OCC) ---", flush=True)
    for r in sorted(out, key=lambda x: x["occ"]):
        print(f"  OCC={r['occ']:>6.0f} [{r['label']:14s}] {tuple(r['shape'])} -> "
              f"BEST={r['best']}", flush=True)


if __name__ == "__main__":
    main()
