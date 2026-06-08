"""RUN-3 — softmax(131072,256) narrow-N BUNDLE lever-decomp (do_bench, NO autotune).

The FULL oracle (seed/oracle=1.170) is a BUNDLE: M_BLOCK 8->16 + num_warps 4->16 + num_stages 1->5 +
eviction ['','first'] + range pipelining. Decompose which lever(s) carry the 1.170, per the seedable-ladder
method: re-bench the oracle's FULL VERBATIM config as the baseline target, then add ONE lever group at a time
to the seed. Finds the seedable carrier + whether the occupancy (M_BLOCK) and warps are coupled.

do_bench median-of-N, correctness-gated, fp32 asserted, ONE process. Foreground (NO bg GPU).

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_softmax_smalln_decomp.py
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
M, N = 131072, 256

# The FULL VERBATIM oracle config (from oracle_cache softmax:131072x256, effort=full).
ORACLE = {
    "block_sizes": [16, 256],
    "range_unroll_factors": [0, 2],
    "range_warp_specializes": [],
    "range_num_stages": [0, 1],
    "range_multi_buffers": [None, None],
    "range_flattens": [None, False],
    "load_eviction_policies": ["", "first"],
    "num_warps": 16,
    "num_stages": 5,
    "indexing": ["pointer", "pointer", "pointer"],
    "atomic_indexing": [],
    "pid_type": "flat",
}


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _run(fn_obj, args, ref, out_extract, cfg, label):
    try:
        k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        ok, err = check_correct(out_extract(b(*args)), ref)
        med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
        return {"label": label, "us": med * 1000 if med else None, "spread": sp,
                "correct": ok, "maxerr": err, "codegen": codegen_kind(b),
                "w": dict(b._config).get("num_warps")}
    except Exception as e:  # noqa: BLE001
        return {"label": label, "error": f"{type(e).__name__}: {e}"[:160]}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    fn, builder, tc_ref = KERNELS["softmax"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"softmax{(M, N)} non-fp32"

    seed = dict(get_seed(fn, args)[0])  # the live champion seed for this shape
    print(f"SEED = {seed}", flush=True)

    # Ladder arms (each = seed + one lever group), + the full verbatim oracle.
    arms = {
        "seed": dict(seed),
        "oracle_verbatim": dict(ORACLE),
        "+Mblock16": {**seed, "block_sizes": [16, 256]},
        "+warps16": {**seed, "num_warps": 16},
        "+Mblock16+warps16": {**seed, "block_sizes": [16, 256], "num_warps": 16},
        "+Mblock16+warps16+ns5": {**seed, "block_sizes": [16, 256], "num_warps": 16,
                                  "num_stages": 5},
        "+evict": {**seed, "load_eviction_policies": ["", "first"]},
        "+Mblock16+warps16+evict": {**seed, "block_sizes": [16, 256], "num_warps": 16,
                                    "load_eviction_policies": ["", "first"]},
    }

    # tc baseline
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))
    print(f"tc={tc_med * 1000:.1f}us\n", flush=True)

    res = {nm: _run(fn, args, ref, out_extract, c, nm) for nm, c in arms.items()}
    seed_us = res["seed"].get("us")
    orac_us = res["oracle_verbatim"].get("us")
    print(f"{'arm':28s} {'us':>9} {'seed/arm':>9} {'arm/orac':>9} {'w':>3} {'codegen':>11}",
          flush=True)
    for nm in arms:
        r = res[nm]
        if r.get("us"):
            sa = seed_us / r["us"] if seed_us else float("nan")
            ao = r["us"] / orac_us if orac_us else float("nan")
            print(f"{nm:28s} {r['us']:9.1f} {sa:9.3f} {ao:9.3f} {r.get('w'):>3} "
                  f"{r.get('codegen'):>11}  {'OK' if r['correct'] else 'BAD'}",
                  flush=True)
        else:
            print(f"{nm:28s} -> {r.get('error')}", flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "softmax_smalln_decomp.json"), "w") as f:
        json.dump({"shape": [M, N], "tc_us": tc_med * 1000, "arms": res}, f, indent=2)
    print(f"\n[wrote softmax_smalln_decomp.json] tc={tc_med * 1000:.1f}us", flush=True)


if __name__ == "__main__":
    main()
