"""RUN-3 EDIT#5 lever DECOMPOSITION — jsd narrow-V Band-B (NO autotune).

Full oracle (committed) on jsd narrow-V: seed/oracle 1.214 (8192,30522) / 1.127 (8192,
32000), oracle/tc ~1.02-1.03 (seedable + beats tc). Field-diff: block_sizes [4096,1]->
[2048,1] (R_BLOCK chunk 4096->2048 fp32) + num_warps 32->16 + num_stages 1->4. Three
levers -> WHICH carries? Add ONE at a time to the seed, do_bench median-7, vs seed
(floor) + the full-oracle target. Finds the principled EDIT#5 fix (and whether it's a
clean single lever or needs the combo).

jsd is Band-B (num_tiled_accumulators>=1): block_sizes=[R_BLOCK, M_BLOCK=1]; the seed's
R_BLOCK=BANDB_R_BLOCK_BYTES/itemsize=4096. So 'chunk2048' halves the Band-B R_BLOCK cap.

Arms (from the live seed, mutating only the named field(s)):
  seed       : live seed as-is (R_BLOCK 4096, w32, ns1)
  chunk2048  : block_sizes R_BLOCK -> 2048
  warps16    : num_warps 32 -> 16
  stages4    : num_stages 1 -> 4
  chunk+warps: 2048 + w16
  all        : 2048 + w16 + ns4 (the oracle lever set)

Also probe kl_div(8192,30522) [Band-B too, but at PARITY in triage] with the SAME
levers -> does the jsd fix HURT kl_div? (the EDIT#5 gate would fire on both Band-B
narrow-V; if chunk2048 helps jsd but hurts kl_div, need a finer key than Band-B-narrow-V.)

Invocation (run from /tmp; AWAIT GPU-GRANTED):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_jsd_decomp_ab.py
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

# (kernel, M, N): jsd narrow-V (the gap) + kl_div narrow-V (Band-B peer at parity — does
# the jsd fix hurt it? = the finer-key test, like welford was for the pid).
CASES = [
    ("jsd", 8192, 30522),
    ("jsd", 8192, 32000),
    ("kl_div", 8192, 30522),
]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _r_block_index(fn_obj, args) -> int:
    """The block_sizes index that is the reduction axis (Band-B R_BLOCK), from the
    fact's block_id — so we mutate the RIGHT entry generically."""
    k = helion.kernel(fn_obj.fn)
    b = k.bind(args)
    spec = b.env.config_spec
    fact = spec.reduction_facts[0]
    return spec.block_sizes.block_id_to_index(fact.block_id)


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
                "w": dict(b._config).get("num_warps"),
                "ns": dict(b._config).get("num_stages")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:150]}


def run_case(kernel, M, N):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"{kernel}{(M, N)} non-fp32"
    seed = dict(get_seed(fn, args)[0])
    ridx = _r_block_index(fn, args)
    seed_bs = list(seed["block_sizes"])

    def _bs(rblock):
        bs = list(seed_bs)
        bs[ridx] = rblock
        return bs

    arms = {
        "seed": dict(seed),
        "chunk2048": {**seed, "block_sizes": _bs(2048)},
        "warps16": {**seed, "num_warps": 16},
        "stages4": {**seed, "num_stages": 4},
        "chunk+warps": {**seed, "block_sizes": _bs(2048), "num_warps": 16},
        "all": {**seed, "block_sizes": _bs(2048), "num_warps": 16, "num_stages": 4},
    }

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    res = {n: _run(fn, args, ref, out_extract, c) for n, c in arms.items()}
    sus = res["seed"].get("us")
    print(f"\n=== {kernel}({M},{N}) === tc={tc_med * 1000:.1f}us  R_BLOCK_idx={ridx}  "
          f"seed_bs={seed_bs}", flush=True)
    for name, r in res.items():
        if r.get("us"):
            sa = sus / r["us"] if sus else float("nan")
            print(f"  {name:>12} {r['us']:>9.1f}us  seed/arm={sa:>6.3f}  "
                  f"arm/tc={tc_med * 1000 / r['us']:.3f}  bs={r['bs']} w{r['w']} "
                  f"ns{r['ns']}  {'OK' if r['correct'] else 'BAD'}", flush=True)
        else:
            print(f"  {name:>12} -> {r}", flush=True)
    return {"kernel": kernel, "shape": [M, N], "tc_us": tc_med * 1000, "arms": res}


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
    with open(os.path.join(LOG_DIR, "jsd_decomp_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'jsd_decomp_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
