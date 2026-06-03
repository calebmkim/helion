"""RUN-3 EDIT#6 referee-reproduce — softmax T2 reread-eviction (NO autotune).

EDIT#6 routes the reread-eviction to the T2 plain path: softmax wide-looped now emits
load_eviction_policies=['last','first']. This reproduces the win for the referee:
seed_live (the SHIPPING config = WITH the EDIT#6 eviction, from get_seed) vs a
flat_noevict baseline (eviction STRIPPED) — the pure EDIT#6 delta. Plus the byte-
identity canary: kl_div/jsd (row_reread=False → no eviction) must be IDENTICAL in
both arms (EDIT#6 doesn't touch them).

Arms per shape:
  flat_noevict : the seed with load_eviction_policies removed (the pre-EDIT#6 baseline)
  seed_live    : get_seed as-is (= the EDIT#6 shipping config; softmax wide → ['last','first'])

PASS = seed_live beats flat_noevict on softmax wide (~1.31×/1.08×); kl_div/jsd identical
(seed_live ev == flat_noevict ev == None → same config → same time within noise).

Invocation (run from /tmp; AWAIT GPU-GRANTED):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_edit6_reproduce.py
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

# softmax wide-looped (EDIT#6 fires) + kl_div/jsd (T2 but row_reread=False → byte-id canary)
CASES = [
    ("softmax", 1024, 65536, True),
    ("softmax", 512, 131072, True),
    ("kl_div", 1024, 4096, False),
    ("jsd", 1024, 4096, False),
]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _run(fn_obj, args, ref, out_extract, cfg):
    k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    ok, err = check_correct(out_extract(b(*args)), ref)
    med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
    return {"us": med * 1000 if med else None, "spread": sp, "correct": ok,
            "codegen": codegen_kind(b),
            "ev": dict(b._config).get("load_eviction_policies")}


def run_case(kernel, M, N, fires):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"{kernel}{(M, N)} non-fp32"
    seed_live = dict(get_seed(fn, args)[0])  # SHIPPING config (with EDIT#6 eviction)
    flat_noevict = {k: v for k, v in seed_live.items()
                    if k != "load_eviction_policies"}

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    r_flat = _run(fn, args, ref, out_extract, flat_noevict)
    r_live = _run(fn, args, ref, out_extract, seed_live)
    base = r_flat.get("us")
    win = base / r_live["us"] if (base and r_live.get("us")) else float("nan")
    print(f"\n=== {kernel}({M},{N}) === tc={tc_med * 1000:.1f}us  fires_EDIT6={fires}",
          flush=True)
    print(f"  flat_noevict {r_flat['us']:>9.1f}us  ev={r_flat['ev']}  "
          f"{r_flat['codegen']}  {'OK' if r_flat['correct'] else 'BAD'}", flush=True)
    print(f"  seed_live    {r_live['us']:>9.1f}us  ev={r_live['ev']}  "
          f"{r_live['codegen']}  {'OK' if r_live['correct'] else 'BAD'}  "
          f"flat/live={win:.3f}", flush=True)
    verdict = ("WIN" if (fires and win >= 1.03) else
               "BYTE-ID" if (not fires and abs(win - 1.0) < 0.03) else
               "CHECK")
    print(f"  -> {verdict}", flush=True)
    return {"kernel": kernel, "shape": [M, N], "fires": fires, "tc_us": tc_med * 1000,
            "flat_noevict": r_flat, "seed_live": r_live, "win": win, "verdict": verdict}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES', '?')} helion={helion.__file__}",
          flush=True)
    out = []
    for kernel, M, N, fires in CASES:
        try:
            out.append(run_case(kernel, M, N, fires))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {kernel}({M},{N}): {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "edit6_reproduce.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'edit6_reproduce.json')}]", flush=True)


if __name__ == "__main__":
    main()
