"""RUN-3 EDIT-PID confirming A/B — the PHYSICS-DERIVED sm_mult vs flat (NO autotune).

Validates the proposed EDIT-PID formula BEFORE baking it into the seed (validate the
derived config, then commit):
    num_sm_multiplier = clamp( next_pow2(ceil(M_rows / num_sm)), 1, CAP )   (CAP=32)
where M_rows = the M-extent (rows the flat grid launches), num_sm = get_num_sm (H100
132). This DERIVES the multiplier from workload-M × hardware-SM — NOT the oracle's
fit-32 (the p-hacking guard). Per shape the derived value:
    CE(4096,98304):  ceil(4096/132)=32 -> np2=32
    CE(8192,128256): ceil(8192/132)=63 -> np2=64 -> cap 32
    CE(2048,256000): ceil(2048/132)=16 -> np2=16
    rms/ln(1,131072): ceil(1/132)=1 -> 1   (degenerate M=1 -> NO over-subscription,
                                            proving const-32 would be unprincipled here)

Arms per shape (on the EDIT#3-evicted seed; pid='flat' is the shipping baseline):
  flat        : the live seed (pid='flat') — BASELINE
  pid_derived : flat + pid='persistent_interleaved', num_sm_multiplier=<DERIVED>, maxnreg default
  pid_const32 : flat + persistent_interleaved + sm_mult=32 (the oracle value — to show
                derived ≈ const where M~mid, and derived is RIGHT at M=1 where const is absurd)

PASS = pid_derived net-positive (>= flat) on every shape it fires (CE-wide gains;
rms/ln tie). do_bench median-of-7, correctness-gated, fp32.

Invocation (run from /tmp; AWAIT GPU-GRANTED):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_pid_derived_ab.py
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
from helion.runtime import get_num_sm  # noqa: E402

from run2_measure_g import (  # noqa: E402
    KERNELS,
    N_RUNS,
    get_seed,
    check_correct,
    codegen_kind,
)
from triton.testing import do_bench  # noqa: E402

LOG_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "logs", "run3"))
CAP = 32

# (kernel, M, N): the T1-scoped gate fires on these (row_reread AND looped).
CASES = [
    ("cross_entropy", 4096, 98304),
    ("cross_entropy", 8192, 128256),
    ("cross_entropy", 2048, 256000),
    ("rms_norm", 1, 131072),
    ("layer_norm", 1, 131072),
]


def _np2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


def _derived_sm_mult(m_rows: int, num_sm: int) -> int:
    import math

    val = _np2(math.ceil(m_rows / max(1, num_sm)))
    return max(1, min(val, CAP))


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
        norm = dict(b._config)
        ok, err = check_correct(out_extract(b(*args)), ref)
        med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
        return {"us": med * 1000 if med else None, "spread": sp,
                "codegen": codegen_kind(b), "correct": ok,
                "pid": norm.get("pid_type"), "sm_mult": norm.get("num_sm_multiplier")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:150]}


def run_case(kernel, M, N, num_sm):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32
    base = dict(get_seed(fn, args)[0])
    derived = _derived_sm_mult(M, num_sm)
    # maxnreg=64 turned out LOAD-BEARING (the first run without it gave 1.06/1.15/0.999;
    # the 3x3 WITH it gave 1.23/1.24/1.052). So the SHIPPING config = derived sm_mult +
    # maxnreg=64. Arms isolate maxnreg's contribution.
    arms = {
        "flat": dict(base),
        "pid_derived_mnr64": {**base, "pid_type": "persistent_interleaved",
                              "num_sm_multiplier": derived, "maxnreg": 64},
        "pid_derived_nomnr": {**base, "pid_type": "persistent_interleaved",
                              "num_sm_multiplier": derived},
    }
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))
    res = {n: _run(fn, args, ref, out_extract, c) for n, c in arms.items()}
    base_us = res["flat"].get("us")
    print(f"\n=== {kernel}({M},{N}) === tc={tc_med * 1000:.1f}us  M={M} num_sm={num_sm} "
          f"-> DERIVED sm_mult={derived}", flush=True)
    for name, r in res.items():
        if r.get("us"):
            vb = base_us / r["us"] if base_us else float("nan")
            print(f"  {name:>12} {r['us']:>9.1f}us  flat/arm={vb:>6.3f}  "
                  f"arm/tc={tc_med * 1000 / r['us']:.3f}  {str(r['pid']):>22} "
                  f"sm={r['sm_mult']}  {r['codegen']:>10}  sp={r['spread']:.2f}", flush=True)
        else:
            print(f"  {name:>12} -> {r}", flush=True)
    return {"kernel": kernel, "shape": [M, N], "tc_us": tc_med * 1000,
            "derived_sm_mult": derived, "arms": res}


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    num_sm = get_num_sm(torch.device("cuda"))
    print(f"GPU={gpu} helion={helion.__file__} num_sm={num_sm} CAP={CAP}", flush=True)
    out = []
    for kernel, M, N in CASES:
        try:
            out.append(run_case(kernel, M, N, num_sm))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {kernel}({M},{N}): {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "pid_derived_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'pid_derived_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
