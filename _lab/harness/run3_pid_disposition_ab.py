"""RUN-3 EDIT-PID DISPOSITION A/B — decide (A) seed the pid-cluster vs (B) Product-B.

The FULL CE oracles (cached) showed: (4096,98304)+(8192,128256) want persistent_
interleaved@sm_mult32; (2048,256000) wants persistent_BLOCKED@sm_mult4; ALL lose to tc
(0.95/0.80/0.62 oracle/tc). The anti-giving-up answer key (task #9): does the
2-shape-consistent config (interleaved@32) BEAT flat EVERYWHERE a `row_reread AND
looped` pid gate would fire? Two outcomes:
  (A) interleaved@32 net-positive vs flat on ALL wide CE (incl 256000, the "wrong"
      variant shape) AND on the OTHER looped-reread kernels (softmax/welford/rms-ln
      wide) -> a coarse seedable pid rule keyed on the looped-reread regime. EDIT-PID.
  (B) interleaved@32 loses to flat at 256000 or on the other kernels -> the pid choice
      is per-shape fine-tuning -> Product-B, honest null for the Product-A seed.

All arms built on the EDIT#3-evicted seed (eviction is the banked carrier; pid is the
residual being tested ON TOP of it). pid_type='flat' is the current seed. The pid
variants require pid_type != flat for sm_mult/maxnreg to survive normalize (config_spec
drops them under flat) — so each pid arm sets the trio together.

do_bench median-of-7, correctness-gated, ONE process. vs the flat (EDIT#3) baseline.

CE arms (per wide CE shape):
  flat_evict   : seed + EDIT#3 evict (pid='flat') — the BASELINE (what ships today)
  interleaved32: flat_evict + pid='persistent_interleaved', num_sm_multiplier=32, maxnreg=64
  blocked4     : flat_evict + pid='persistent_blocked', num_sm_multiplier=4, maxnreg=64
Other-kernel probe (does interleaved@32 help/hurt the OTHER looped-reread kernels?):
  softmax(512,131072), welford(65536,4096), rms_norm(1,131072): flat vs interleaved32.

Invocation (run from /tmp; AWAIT GPU-GRANTED):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_pid_disposition_ab.py
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

INTERLEAVED32 = {"pid_type": "persistent_interleaved", "num_sm_multiplier": 32,
                 "maxnreg": 64}
BLOCKED4 = {"pid_type": "persistent_blocked", "num_sm_multiplier": 4, "maxnreg": 64}

CE_SHAPES = [(4096, 98304), (8192, 128256), (2048, 256000)]
# other looped-reread kernels: does interleaved@32 help or HURT them vs flat?
OTHER = [("softmax", 512, 131072), ("welford", 65536, 4096), ("rms_norm", 1, 131072)]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _run(fn_obj, args, ref, out_extract, cfg, watch):
    try:
        k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        norm = dict(b._config)
        dropped = [kk for kk in watch if kk in cfg and norm.get(kk) != cfg[kk]]
        ok, err = check_correct(out_extract(b(*args)), ref)
        med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
        return {"us": med * 1000 if med else None, "spread": sp,
                "codegen": codegen_kind(b), "correct": ok, "pid": norm.get("pid_type"),
                "dropped": dropped}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:150]}


def run_ce(M, N):
    fn, builder, tc_ref = KERNELS["cross_entropy"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32
    base = dict(get_seed(fn, args)[0])  # the live EDIT#3-evicted seed (pid='flat')
    arms = {
        "flat_evict": (dict(base), []),
        "interleaved32": ({**base, **INTERLEAVED32}, list(INTERLEAVED32)),
        "blocked4": ({**base, **BLOCKED4}, list(BLOCKED4)),
    }
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))
    res = {n: _run(fn, args, ref, out_extract, c, w) for n, (c, w) in arms.items()}
    base_us = res["flat_evict"].get("us")
    print(f"\n=== cross_entropy({M},{N}) === tc={tc_med * 1000:.1f}us  "
          f"[flat_evict = EDIT#3 baseline]", flush=True)
    for name, r in res.items():
        if r.get("us"):
            vb = base_us / r["us"] if base_us else float("nan")
            dr = ",".join(r["dropped"]) or "-"
            print(f"  {name:>14} {r['us']:>9.1f}us  flat/arm={vb:>6.3f}  "
                  f"arm/tc={tc_med * 1000 / r['us']:.3f}  {str(r['pid']):>22}  "
                  f"drop={dr}  sp={r['spread']:.2f}", flush=True)
        else:
            print(f"  {name:>14} -> {r}", flush=True)
    return {"kernel": "cross_entropy", "shape": [M, N], "tc_us": tc_med * 1000,
            "arms": res}


def run_other(kernel, M, N):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32
    base = dict(get_seed(fn, args)[0])
    arms = {
        "flat": (dict(base), []),
        "interleaved32": ({**base, **INTERLEAVED32}, list(INTERLEAVED32)),
    }
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))
    res = {n: _run(fn, args, ref, out_extract, c, w) for n, (c, w) in arms.items()}
    base_us = res["flat"].get("us")
    print(f"\n=== {kernel}({M},{N}) === tc={tc_med * 1000:.1f}us  "
          f"[does interleaved@32 help or HURT vs flat?]", flush=True)
    for name, r in res.items():
        if r.get("us"):
            vb = base_us / r["us"] if base_us else float("nan")
            dr = ",".join(r["dropped"]) or "-"
            print(f"  {name:>14} {r['us']:>9.1f}us  flat/arm={vb:>6.3f}  "
                  f"arm/tc={tc_med * 1000 / r['us']:.3f}  {str(r['pid']):>22}  "
                  f"drop={dr}  sp={r['spread']:.2f}", flush=True)
        else:
            print(f"  {name:>14} -> {r}", flush=True)
    return {"kernel": kernel, "shape": [M, N], "tc_us": tc_med * 1000, "arms": res}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES', '?')} helion={helion.__file__}",
          flush=True)
    out = []
    for M, N in CE_SHAPES:
        try:
            out.append(run_ce(M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] CE({M},{N}): {type(e).__name__}: {e}"[:200], flush=True)
    for kernel, M, N in OTHER:
        try:
            out.append(run_other(kernel, M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {kernel}({M},{N}): {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "pid_disposition_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'pid_disposition_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
