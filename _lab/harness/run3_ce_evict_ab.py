"""RUN-3 cross_entropy RE-READ eviction A/B (NO autotune).

CE is a 2-pass re-read kernel: loads (in the looped seed's generated Triton) are
  [0] labels_tile  [1] logits_at_target (scalar gather)  [2] logits_rows (amax pass)
  [3] logits_rows_1 (exp-sum pass = the RE-READ of the row)
(load_eviction_policies length = 5; the 5th slot is a codegen slot beyond the 4
visible tl.load — handled by building at the live spec length.)

The wide-CE full oracle picked load_eviction_policies=['','','last','first','last']
and the ablation showed eviction ALONE = 1.31x. The principled re-read policy: keep
the FIRST row-load (amax pass) L2-resident ('last') so the SECOND pass re-reads from
L2, and evict the final use ('first'). This A/Bs candidate policies (built at the
live spec length, only valid choices) vs default, do_bench median-of-7, correctness-
gated, on PERSISTENT (boundary) AND LOOPED (wide) CE shapes to see where it helps.

Arms (eviction list built at spec length n; '' = default per slot):
  default            : None (autotuner default)
  seed_emitted       : the EXACT policy the live seed emits NOW (EDIT#3 provenance —
                       the SHIPPING policy; the load-bearing arm). For wide CE this is
                       ['','','last','first','first'] ('last' on logits' first load).
  pos_slot0          : counterfactual = the run-2 POSITIONAL rule ('last' on slot 0 =
                       labels, the WRONG buffer). If seed_emitted beats pos_slot0, the
                       buffer-identity de-hack is the cause (not just "some eviction").
  oracle_exact       : the oracle's ['','','last','first','last'] (truncated/padded to n)
  all_first          : ['first']*n (stream-everything, the num_load==1 recipe)
  all_last           : ['last']*n

Invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_ce_evict_ab.py \
    4096x50304 4096x98304 8192x128256
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


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    return s[len(s) // 2], (s[-1] - s[0]) / s[len(s) // 2] if s[len(s) // 2] else None


def _evict_arms(n, seed_emitted):
    """Candidate eviction lists at spec length n. Slot 2 = the row's amax (first)
    pass = the load to keep resident ('last') for the re-read; slot 3 = the exp-sum
    re-read = final use ('first'). Slots 0/1 = scalar gathers (default).

    ``seed_emitted`` is the EXACT load_eviction_policies the live seed emits NOW (with
    EDIT#3's provenance routing) -- the SHIPPING policy. This is the load-bearing arm:
    it validates what the heuristic actually produces, not a hand-coded stand-in. The
    provenance de-hack sets slot 2 -> 'last' (logits' first load, the amax pass), slots
    3,4 -> 'first' (logits re-reads); the run-2 POSITIONAL rule would have set slot 0
    ('last' on labels), which `pos_slot0` reproduces as the wrong-buffer counterfactual."""
    oracle = ["", "", "last", "first", "last"]
    pos_slot0 = [""] * n  # run-2 positional rule: 'last' on slot 0 (= labels, WRONG)
    if n >= 1:
        pos_slot0[0] = "last"
        for i in range(1, min(n, 4)):
            pos_slot0[i] = "first"
    arms = {
        "default": None,
        "seed_emitted": list(seed_emitted) if seed_emitted is not None else None,
        "pos_slot0": pos_slot0,  # counterfactual: the run-2 positional mis-key
        "oracle_exact": (oracle + [""] * n)[:n],
        "all_first": ["first"] * n,
        "all_last": ["last"] * n,
    }
    return arms


def run_shape(M, N):
    fn, builder, tc_ref = KERNELS["cross_entropy"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32
    seed = dict(get_seed(fn, args)[0])
    n = helion.kernel(fn.fn).bind(args).env.config_spec.load_eviction_policies.length
    seed_emitted = seed.get("load_eviction_policies")  # the SHIPPING EDIT#3 policy

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    results = {}
    base_us = None
    print(f"  seed_emitted load_eviction_policies = {seed_emitted}", flush=True)
    for name, ev in _evict_arms(n, seed_emitted).items():
        cfg = dict(seed)
        if ev is None:
            cfg.pop("load_eviction_policies", None)
        else:
            cfg["load_eviction_policies"] = ev
        try:
            k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
            b = k.bind(args); b.ensure_config_exists(args)
            ok, err = check_correct(out_extract(b(*args)), ref)
            med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
            results[name] = {"us": med * 1000 if med else None, "spread": sp,
                             "codegen": codegen_kind(b), "correct": ok, "maxerr": err,
                             "ev_norm": dict(b._config).get("load_eviction_policies")}
            if name == "default" and med:
                base_us = med * 1000
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:160]}

    print(f"\n=== cross_entropy({M},{N}) === tc={tc_med*1000:.1f}us n_slots={n} "
          f"[{seed.get('reduction_loops')}]", flush=True)
    for name, r in results.items():
        if r.get("us"):
            db = base_us / r["us"] if base_us else float("nan")
            print(f"  {name:>14} {r['us']:>9.1f}us  vs_default={db:>5.3f} "
                  f"arm/tc={tc_med*1000/r['us']:.3f} {r['codegen']:>10} "
                  f"sp={r['spread']:.2f} {'OK' if r['correct'] else 'BAD'} "
                  f"ev={r['ev_norm']}", flush=True)
        else:
            print(f"  {name:>14} -> {r}", flush=True)
    return {"shape": [M, N], "tc_us": tc_med * 1000, "n_slots": n,
            "reduction_loops": seed.get("reduction_loops"), "arms": results}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    shape_args = sys.argv[1:] or ["4096x50304", "4096x98304", "8192x128256"]
    out = []
    for s in shape_args:
        M, N = (int(x) for x in s.split("x"))
        try:
            out.append(run_shape(M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {s}: {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "ce_evict_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'ce_evict_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
