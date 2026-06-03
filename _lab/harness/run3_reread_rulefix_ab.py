"""RUN-3 EDIT#3 reread-eviction RULE FIX A/B: Rule-A vs Rule-B (NO autotune).

The no-regress A/B found Rule-A (the current EDIT#3) REGRESSES welford 3-6% vs the
shipping positional rule, because Rule-A leaves streamed-once broadcasts (welford's
weight/bias) at DEFAULT '' instead of 'first'. Both rules keep the buffer-IDENTITY
de-hack for the 'last' slot (the fact-integrity caveat -- 'last' on the re-read
reduction ROW's first load, found via provenance, NOT a positional slot). They differ
only on the OTHER slots:

  Rule-A (current): 'last' on row-first-load; 'first' on the row's RE-READS; DEFAULT ''
                    on everything else (other buffers untouched).
  Rule-B (fix):     'last' on row-first-load; 'first' on EVERY OTHER slot (row re-reads
                    AND all streamed-once operands -> evict-first frees L2).

Rule-B reproduces the run-2 POSITIONAL win where positional was right (welford: row x is
slot 0, so Rule-B == ['last','first','first','first']) while STILL putting 'last' on the
provenance-identified row for CE (logits@slot2, not positional slot0=labels). The
question: does Rule-B tie/beat Rule-A's 1.31x on CE (i.e. does CE care whether the
labels/logits_flat gather slots are 'first' vs '')? If Rule-B >= Rule-A on CE AND
reproduces welford positional, Rule-B is the faithful rule that regresses NOTHING.

Per shape, arms built from the live seed + the fact's reread_buffer_slots:
  default        : None
  ruleA          : 'last' on slots[0]; 'first' on slots[1:]; '' elsewhere
  ruleB          : 'last' on slots[0]; 'first' on ALL other slots
  pos_run2       : ['last']+['first']*(n-1)  (the shipping positional rule)

Invocation (run from /tmp; AWAIT GPU-GRANTED):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_reread_rulefix_ab.py
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

# (kernel, M, N): the wide CE shapes (eviction win) + welford TRAIN (the regression).
CASES = [
    ("cross_entropy", 4096, 98304),
    ("cross_entropy", 8192, 128256),
    ("welford", 65536, 4096),
    ("welford", 32768, 8192),
]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _reread_slots(fn_obj, args) -> tuple[int, ...]:
    k = helion.kernel(fn_obj.fn)
    b = k.bind(args)
    return b.env.config_spec.reduction_facts[0].reread_buffer_slots


def run_case(kernel, M, N):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"{kernel}{(M, N)} non-fp32 {a.dtype}"
    seed = dict(get_seed(fn, args)[0])
    n = helion.kernel(fn.fn).bind(args).env.config_spec.load_eviction_policies.length
    slots = [s for s in _reread_slots(fn, args) if 0 <= s < n]

    ruleA = None
    ruleB = None
    if slots:
        ruleA = [""] * n
        ruleA[slots[0]] = "last"
        for s in slots[1:]:
            ruleA[s] = "first"
        ruleB = ["first"] * n
        ruleB[slots[0]] = "last"
    pos_run2 = (["last"] + ["first"] * (n - 1)) if n >= 1 else []

    arms = {
        "default": None,
        "ruleA": ruleA,
        "ruleB": ruleB,
        "pos_run2": pos_run2,
    }

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0], f"tc FAIL {kernel}{(M, N)}"
    tc_med, _ = _bench(lambda: tc(args))

    results = {}
    base_us = None
    for name, ev in arms.items():
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
                "ev_norm": dict(b._config).get("load_eviction_policies"),
            }
            if name == "default" and med:
                base_us = med * 1000
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:160]}

    print(f"\n=== {kernel}({M},{N}) === tc={tc_med * 1000:.1f}us n={n} slots={slots} "
          f"[{seed.get('reduction_loops')}]", flush=True)
    for name, r in results.items():
        if r.get("us"):
            vd = base_us / r["us"] if base_us else float("nan")
            print(f"  {name:>9} {r['us']:>9.1f}us  vs_default={vd:>5.3f}  "
                  f"arm/tc={tc_med * 1000 / r['us']:.3f}  sp={r['spread']:.2f}  "
                  f"{'OK' if r['correct'] else 'BAD'}  ev={r['ev_norm']}", flush=True)
        else:
            print(f"  {name:>9} -> {r}", flush=True)
    return {"kernel": kernel, "shape": [M, N], "tc_us": tc_med * 1000,
            "slots": slots, "arms": results}


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
    with open(os.path.join(LOG_DIR, "reread_rulefix_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'reread_rulefix_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
