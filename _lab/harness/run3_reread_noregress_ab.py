"""RUN-3 EDIT#3 reread-eviction NO-REGRESSION A/B (NO autotune).

EDIT#3 routes the reread eviction from PROVENANCE (the re-read reduction ROW's load
slots) instead of the run-2 POSITIONAL slot rule. This is byte-identical to run-2 ONLY
where the row's first load IS slot 0 (welford). The no-regression question the hub
requires before EDIT#3 commits:

  (A) WELFORD: does the de-hacked policy ['last','first','',''] (slots 0,1 = x's two
      loads; slots 2,3 = weight/bias left default) reproduce the run-2 POSITIONAL win
      ['last','first','first','first']? Run-2's welford gain CAME from eviction
      (262144,4096) G 0.759->0.950; the de-hack must not regress it. (My provenance
      slots for welford ARE (0,1) -> the row x, so the de-hack drops the spurious
      'first' on weight/bias slots 2,3 -- expected NEUTRAL-or-better, never worse.)

  (B) rms_norm / layer_norm (1,131072) ROBUSTNESS CANARY: these are >240KiB so LOOPED
      under the row_reread cap, so EDIT#3's `row_reread and not persistent` branch NOW
      emits a reread policy for them (it did NOT before -- they had no eviction). The
      canary: the emitted policy must be CORRECT and NOT SLOWER than default (no
      regression on a robustness shape we don't perf-tune). slots -> x (the row),
      weight/bias default.

Arms (per kernel, built at the live spec length n):
  default     : None
  seed_emitted: the EXACT policy the live seed emits NOW (EDIT#3 provenance) -- ships
  pos_run2    : the run-2 POSITIONAL rule ['last']+['first']*(n-1) -- the old behavior
  all_first   : ['first']*n
  all_last    : ['last']*n

PASS = seed_emitted is correct AND (>= default) AND (>= pos_run2) on welford; correct
AND (>= default) on the rms/ln canary. do_bench median-of-7, fp32 asserted.

Invocation (run from /tmp; AWAIT GPU-GRANTED -- one GPU):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_reread_noregress_ab.py
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

# (kernel, M, N): the shapes where reread-eviction is emitted + matters.
CASES = [
    ("welford", 262144, 4096),   # run-2 eviction win shape
    ("welford", 32768, 8192),
    ("rms_norm", 1, 131072),     # robustness canary (looped, now gets reread policy)
    ("layer_norm", 1, 131072),   # robustness canary
]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _arms(n, seed_emitted):
    pos_run2 = ["last"] + ["first"] * (n - 1) if n >= 1 else []
    return {
        "default": None,
        "seed_emitted": list(seed_emitted) if seed_emitted is not None else None,
        "pos_run2": pos_run2,
        "all_first": ["first"] * n,
        "all_last": ["last"] * n,
    }


def run_case(kernel, M, N):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"{kernel}{(M, N)} non-fp32 {a.dtype}"
    seed = dict(get_seed(fn, args)[0])
    n = helion.kernel(fn.fn).bind(args).env.config_spec.load_eviction_policies.length
    seed_emitted = seed.get("load_eviction_policies")

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0], f"tc FAIL {kernel}{(M, N)}"
    tc_med, _ = _bench(lambda: tc(args))

    results = {}
    base_us = None
    for name, ev in _arms(n, seed_emitted).items():
        cfg = dict(seed)
        if ev is None:
            cfg.pop("load_eviction_policies", None)
        else:
            cfg["load_eviction_policies"] = ev
        try:
            k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
            b = k.bind(args)
            b.ensure_config_exists(args)
            ok, err = check_correct(out_extract(b(*args)), ref)
            med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
            results[name] = {
                "us": med * 1000 if med else None,
                "spread": sp,
                "codegen": codegen_kind(b),
                "correct": ok,
                "maxerr": err,
                "ev_norm": dict(b._config).get("load_eviction_policies"),
            }
            if name == "default" and med:
                base_us = med * 1000
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:160]}

    print(
        f"\n=== {kernel}({M},{N}) === tc={tc_med * 1000:.1f}us n_slots={n}  "
        f"seed_emitted={seed_emitted}",
        flush=True,
    )
    se = results.get("seed_emitted", {})
    pr = results.get("pos_run2", {})
    for name, r in results.items():
        if r.get("us"):
            vd = base_us / r["us"] if base_us else float("nan")
            print(
                f"  {name:>13} {r['us']:>9.1f}us  vs_default={vd:>5.3f}  "
                f"arm/tc={tc_med * 1000 / r['us']:.3f}  {r['codegen']:>10}  "
                f"sp={r['spread']:.2f}  {'OK' if r['correct'] else 'BAD'}  "
                f"ev={r['ev_norm']}",
                flush=True,
            )
        else:
            print(f"  {name:>13} -> {r}", flush=True)
    # verdict
    verdict = "n/a"
    if se.get("us") and base_us:
        ge_default = se["us"] <= base_us * 1.03
        if pr.get("us"):
            ge_pos = se["us"] <= pr["us"] * 1.03
            verdict = (
                "PASS" if (ge_default and ge_pos) else "REGRESS"
            )
        else:
            verdict = "PASS" if ge_default else "REGRESS"
    print(f"  -> NO-REGRESS verdict: {verdict} "
          f"(seed_emitted vs default {'>=' if verdict == 'PASS' else '<'} 1/1.03)",
          flush=True)
    return {"kernel": kernel, "shape": [M, N], "tc_us": tc_med * 1000,
            "seed_emitted": seed_emitted, "verdict": verdict, "arms": results}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES', '?')} helion={helion.__file__}",
          flush=True)
    out = []
    for kernel, M, N in CASES:
        try:
            out.append(run_case(kernel, M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {kernel}({M},{N}): {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "reread_noregress_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'reread_noregress_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
