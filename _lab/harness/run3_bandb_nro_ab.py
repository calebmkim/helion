"""RUN-3 EDIT#5 — Band-B num_reduction_ops accumulator-footprint cap A/B (NO autotune).

The proposed PRINCIPLED rule (hub task #14 "accumulator-row-bytes"): the Band-B R_BLOCK
cap should bound the LIVE reduction-accumulator footprint, not a single accumulator:
    bandb_cap_elems = BANDB_R_BLOCK_BYTES // (itemsize * max(1, num_reduction_ops))
  jsd  (nro=2): 16384 // (4*2) = 2048   (matches oracle: chunk 4096->2048)
  kl_div(nro=1): 16384 // (4*1) = 4096  (UNCHANGED; kl_div at parity narrow+wide)
num_reduction_ops is the FAITHFUL count of [M,R] reduction accumulators (docstring l.109),
NOT a proxy here. num_tiled_accumulators can't discriminate (both jsd/kl_div bind to 2).

This A/B does NOT edit the source yet. It constructs, from the LIVE seed, the config the
rule WOULD emit (R_BLOCK -> nro-footprint cap; and optionally num_warps/2 following the
SAME nro>=2 key), and benches it head-to-head vs the live seed, so the matched-lever gain
is proven BEFORE the heuristic change.

Arms per shape (mutating only named fields of the live seed):
  seed        : live seed (jsd/kl_div R_BLOCK=4096, w32)
  nro_chunk   : R_BLOCK -> 16384 // (itemsize * nro)   [2048 jsd / 4096 kl_div]
  nro_warps   : if nro>=2: num_warps -> num_warps//2 (32->16); else unchanged
  nro_both    : nro_chunk + nro_warps  (the full proposed EDIT#5 config)

Reports seed/arm (>1 = arm faster). The WIN test: nro_both >= 1 on jsd (narrow AND wide)
and ~1.0 (no-regression) on kl_div (narrow AND wide) + sum (a non-Band-B nro>=2 contrast:
nro_chunk must be a NO-OP there since num_tiled_accumulators==0 -> not Band-B -> the rule
only fires when num_tiled_accumulators>=1; we verify the rule's GATE here too).

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_bandb_nro_ab.py
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
BANDB_R_BLOCK_BYTES = 16384  # mirror the heuristic constant

# jsd narrow + WIDE (the World-A-vs-B decider) ; kl_div narrow + wide (no-regression) ;
# sum wide (nro>=2 but num_tiled_accumulators==0 -> rule must NOT fire = byte-id).
CASES = [
    ("jsd", 8192, 30522),
    ("jsd", 8192, 32000),
    ("jsd", 2048, 256000),
    ("kl_div", 8192, 30522),
    ("kl_div", 1024, 256000),
    ("sum", 4096, 28672),
]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _fact(fn_obj, args):
    k = helion.kernel(fn_obj.fn)
    b = k.bind(args)
    spec = b.env.config_spec
    f = spec.reduction_facts[0]
    ridx = spec.block_sizes.block_id_to_index(f.block_id)
    return f, ridx


def _np2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _run(fn_obj, args, ref, out_extract, cfg):
    try:
        k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        ok, err = check_correct(out_extract(b(*args)), ref)
        med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
        return {"us": med * 1000 if med else None, "spread": sp, "correct": ok,
                "codegen": codegen_kind(b),
                "bs": dict(b._config).get("block_sizes"),
                "w": dict(b._config).get("num_warps")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:160]}


def run_case(kernel, M, N):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"{kernel}{(M, N)} non-fp32"
    fact, ridx = _fact(fn, args)
    nro = fact.num_reduction_ops
    n_acc = fact.num_tiled_accumulators
    itemsize = fact.itemsize
    is_bandb = n_acc >= 1
    seed = dict(get_seed(fn, args)[0])
    seed_bs = list(seed["block_sizes"])
    seed_w = seed.get("num_warps")

    # the cap the PROPOSED rule would set (only fires for Band-B; for non-Band-B we still
    # SHOW what it would be, but the rule's gate is num_tiled_accumulators>=1).
    nro_cap = max(1, BANDB_R_BLOCK_BYTES // (max(1, itemsize) * max(1, nro)))
    nro_cap = _np2(nro_cap)
    cur_r = seed_bs[ridx]
    # apply the cap only when Band-B (mirror the heuristic gate); for the wide row the
    # seed R_BLOCK may already be below the cap (cap = min(extent, nro_cap)).
    new_r = min(cur_r, nro_cap) if is_bandb else cur_r

    def _bs(rblock):
        bs = list(seed_bs)
        bs[ridx] = rblock
        return bs

    fires = is_bandb and nro >= 2  # the warps half-key fires for Band-B nro>=2
    arms = {"seed": dict(seed)}
    if new_r != cur_r:
        arms["nro_chunk"] = {**seed, "block_sizes": _bs(new_r)}
    if fires and seed_w and seed_w > 1:
        arms["nro_warps"] = {**seed, "num_warps": seed_w // 2}
    if new_r != cur_r and fires and seed_w and seed_w > 1:
        arms["nro_both"] = {**seed, "block_sizes": _bs(new_r), "num_warps": seed_w // 2}

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    res = {n: _run(fn, args, ref, out_extract, c) for n, c in arms.items()}
    sus = res["seed"].get("us")
    print(f"\n=== {kernel}({M},{N}) === tc={tc_med*1000:.1f}us  nro={nro} "
          f"n_acc={n_acc} bandb={is_bandb}  R_idx={ridx} seed_R={cur_r} "
          f"-> nro_cap={nro_cap} (new_R={new_r}) seed_w={seed_w}", flush=True)
    for name, r in res.items():
        if r.get("us"):
            sa = sus / r["us"] if sus else float("nan")
            print(f"  {name:>10} {r['us']:>9.1f}us  seed/arm={sa:>6.3f}  "
                  f"arm/tc={tc_med*1000/r['us']:.3f}  bs={r['bs']} w{r['w']} "
                  f"{'OK' if r['correct'] else 'BAD'}", flush=True)
        else:
            print(f"  {name:>10} -> {r}", flush=True)
    return {"kernel": kernel, "shape": [M, N], "nro": nro, "n_acc": n_acc,
            "is_bandb": is_bandb, "nro_cap": nro_cap, "seed_R": cur_r, "new_R": new_r,
            "tc_us": tc_med * 1000, "arms": res}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    out = []
    for kernel, M, N in CASES:
        try:
            out.append(run_case(kernel, M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {kernel}({M},{N}): {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "bandb_nro_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'bandb_nro_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
