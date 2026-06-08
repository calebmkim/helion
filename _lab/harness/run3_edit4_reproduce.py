"""RUN-3 EDIT#4 referee-reproduce — welford looped-apply cap 8192->16384 (NO autotune).

EDIT#4 raised STRUCTURED_APPLY_LOOP_CHUNK_BYTES 8192->16384 so wide-N welford emits
apply tile 4096 (was 2048). Reproduce for the referee: seed_live (= SHIPPING config,
apply 4096 from get_seed) vs apply2048 (the pre-EDIT#4 baseline, apply tile forced back
to 2048) — the pure EDIT#4 delta. Plus byte-identity canaries: narrow welford (apply
persistent, unaffected) + a non-Band-C kernel (EDIT#4 is Band-C-only).

Arms per welford shape:
  apply2048 : seed with the apply-tile block_sizes index forced to 2048 (pre-EDIT#4)
  seed_live : get_seed as-is (= EDIT#4 shipping; wide welford -> apply 4096)

PASS = seed_live beats apply2048 on wide welford (~1.05-1.07x); narrow welford + the
non-Band-C canary identical.

Invocation (run from /tmp; AWAIT GPU-GRANTED):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_edit4_reproduce.py
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

# (kernel, M, N, edit4_fires?): wide welford fires; narrow welford + rms_norm = byte-id canary
CASES = [
    ("welford", 4096, 16384, True),
    ("welford", 32768, 8192, True),
    ("welford", 16384, 768, False),   # narrow: apply persistent, EDIT#4 inert
    ("rms_norm", 8192, 4096, False),  # non-Band-C: byte-identical
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
            "codegen": codegen_kind(b), "bs": dict(b._config).get("block_sizes")}


def run_case(kernel, M, N, fires):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"{kernel}{(M, N)} non-fp32"
    seed_live = dict(get_seed(fn, args)[0])  # SHIPPING (EDIT#4: wide welford apply=4096)
    arms = {"seed_live": dict(seed_live)}
    # apply2048: force the apply-tile entry (last block_sizes index for welford) to 2048.
    if kernel == "welford":
        bs = list(seed_live["block_sizes"])
        if len(bs) >= 3 and bs[-1] > 2048:
            bs2 = list(bs)
            bs2[-1] = 2048
            arms["apply2048"] = {**seed_live, "block_sizes": bs2}

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    res = {n: _run(fn, args, ref, out_extract, c) for n, c in arms.items()}
    live = res["seed_live"]
    base = res.get("apply2048", {}).get("us")
    win = base / live["us"] if (base and live.get("us")) else None
    print(f"\n=== {kernel}({M},{N}) === tc={tc_med * 1000:.1f}us  fires_EDIT4={fires}",
          flush=True)
    for name, r in res.items():
        if r.get("us"):
            print(f"  {name:>10} {r['us']:>9.1f}us  bs={r['bs']}  {r['codegen']}  "
                  f"{'OK' if r['correct'] else 'BAD'}", flush=True)
    if win is not None:
        print(f"  apply2048/seed_live = {win:.3f}  "
              f"({'WIN' if (fires and win >= 1.03) else 'flat/n.a.'})", flush=True)
    elif not fires:
        print(f"  (single arm = seed_live; EDIT#4 inert here -> byte-identical to "
              f"pre-edit; codegen {live['codegen']}, bs={live['bs']})", flush=True)
    return {"kernel": kernel, "shape": [M, N], "fires": fires, "tc_us": tc_med * 1000,
            "win": win, "arms": res}


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
    with open(os.path.join(LOG_DIR, "edit4_reproduce.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'edit4_reproduce.json')}]", flush=True)


if __name__ == "__main__":
    main()
