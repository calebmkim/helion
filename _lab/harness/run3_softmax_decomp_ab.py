"""RUN-3 softmax-wide lever DECOMPOSITION — which lever(s) carry the full-oracle gain?

The FULL softmax oracle (run3_oracle.py, committed) beats the looped seed 1.359x
(1024,65536) / 1.097x (512,131072) AND beats tc (1.390/1.098) — a REAL tc-beating gap
the QUICK oracle (1.000/1.012) missed. The oracle's field-diff vs the seed:
  (1024,65536): block 16384->32768 ; evict ['last','first'] ; tensor_descriptor ; unroll[0,2]
  (512,131072): evict ['last',''] ; num_stages 1->4 ; num_warps 32->16 ; tensor_descriptor
Common lever = the REREAD EVICTION ('last' on slot 0 = x's first load). softmax is T2;
the seed emits NO eviction (the plain T2 path skips it). EDIT#3 already computes
softmax's reread_buffer_slots=(0,1) -> the EVICT arm here = exactly what extending the
EDIT#3 Rule B reread eviction to T2 WOULD emit (built from the live fact's slots).

Add ONE lever at a time to the seed, do_bench median-of-7, correctness-gated, ONE
process, vs BOTH the seed (floor) and tc. Finds the carrier (is the eviction enough, or
is the chunk/warps/stages doing the work?). Arms (reduction-axis block index found
generically from fact.block_id):
  seed         : the live seed as-is (no eviction)
  evict        : seed + Rule B reread eviction from fact.reread_buffer_slots
                 ('last' on slot[0], 'first' on the rest) — the EDIT#3-to-T2 candidate
  chunk32768   : seed + reduction-axis block_size -> 32768 (bigger looped chunk)
  evict+chunk  : both
  evict+w16ns4 : evict + num_warps=16 + num_stages=4 (the 512,131072 oracle flavor)
  tc_default   : torch.compile F.softmax (floor reference)

Invocation (run from /tmp; AWAIT GPU-GRANTED):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_softmax_decomp_ab.py
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

CASES = [(1024, 65536), (512, 131072)]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _reread_slots_and_redidx(fn_obj, args):
    k = helion.kernel(fn_obj.fn)
    b = k.bind(args)
    spec = b.env.config_spec
    fact = spec.reduction_facts[0]
    red_idx = spec.block_sizes.block_id_to_index(fact.block_id)
    n_ev = spec.load_eviction_policies.length
    return list(fact.reread_buffer_slots), red_idx, n_ev


def run_shape(M, N):
    fn, builder, tc_ref = KERNELS["softmax"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"softmax{(M, N)} non-fp32 {a.dtype}"
    seed = dict(get_seed(fn, args)[0])
    slots, red_idx, n_ev = _reread_slots_and_redidx(fn, args)
    seed_bs = list(seed["block_sizes"])

    # Rule B reread eviction from the fact's slots (what EDIT#3-to-T2 would emit).
    evict_policy = None
    if slots and n_ev > 0:
        evict_policy = ["first"] * n_ev
        if 0 <= slots[0] < n_ev:
            evict_policy[slots[0]] = "last"

    def _bs(val):
        bs = list(seed_bs)
        bs[red_idx] = val
        return bs

    arms = {"seed": dict(seed)}
    if evict_policy is not None:
        arms["evict"] = {**seed, "load_eviction_policies": evict_policy}
        arms["evict+chunk"] = {**seed, "load_eviction_policies": evict_policy,
                               "block_sizes": _bs(32768)}
        arms["evict+w16ns4"] = {**seed, "load_eviction_policies": evict_policy,
                                "num_warps": 16, "num_stages": 4}
    arms["chunk32768"] = {**seed, "block_sizes": _bs(32768)}

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0], f"tc FAIL softmax {(M, N)}"
    tc_med, _ = _bench(lambda: tc(args))

    results = {}
    seed_us = None
    for name, cfg in arms.items():
        try:
            k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
            b = k.bind(args)
            b.ensure_config_exists(args)
            ok, err = check_correct(out_extract(b(*args)), ref)
            med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
            results[name] = {
                "us": med * 1000 if med else None, "spread": sp,
                "codegen": codegen_kind(b), "correct": ok,
                "ev": dict(b._config).get("load_eviction_policies"),
                "bs": dict(b._config).get("block_sizes"),
            }
            if name == "seed" and med:
                seed_us = med * 1000
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:160]}

    print(f"\n=== softmax({M},{N}) === tc={tc_med * 1000:.1f}us slots={slots} "
          f"red_idx={red_idx} n_ev={n_ev}  Rule-B-evict={evict_policy}", flush=True)
    for name, r in results.items():
        if r.get("us"):
            sa = seed_us / r["us"] if seed_us else float("nan")
            print(f"  {name:>13} {r['us']:>9.1f}us  seed/arm={sa:>6.3f}  "
                  f"arm/tc={tc_med * 1000 / r['us']:.3f}  {r['codegen']:>10}  "
                  f"sp={r['spread']:.2f}  bs={r['bs']} ev={r['ev']}", flush=True)
        else:
            print(f"  {name:>13} -> {r}", flush=True)
    return {"shape": [M, N], "tc_us": tc_med * 1000, "slots": slots, "arms": results}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES', '?')} helion={helion.__file__}",
          flush=True)
    out = []
    for M, N in CASES:
        try:
            out.append(run_shape(M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] softmax({M},{N}): {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "softmax_decomp_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'softmax_decomp_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
