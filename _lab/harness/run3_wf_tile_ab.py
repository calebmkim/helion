"""RUN-3 welford apply/combine tile + warps matched-lever A/B (NO autotune).

The quick oracle field-diff (block_sizes=[M_block, combine_tile, apply_tile]):
  (4096,16384):  seed [1,8192,2048]  -> oracle [1,16384,4096]   (so=1.146)
  (32768,8192):  seed [2,8192,2048]  -> oracle [2,8192,4096] w32 (so=1.089)
  (16384,768):   seed [1,1024,1024]  -> oracle [1,1024,256] w1   (so=1.032)
=> wide N: bigger apply tile (seed STRUCTURED_APPLY_LOOP_CHUNK_BYTES=8192=2048fp32 is
   too small; oracle wants 4096) + maybe bigger combine; narrow N: smaller apply +
   fewer warps. This A/Bs apply-tile {seed,4096,8192} x combine {seed,np2} x warps
   {seed,oracle} vs the seed and tc, do_bench median-of-7, correctness-gated, to find
   the principled apply/combine/warps rule. Welford defaults bf16 upstream -- the
   run2 builder forces fp32 (asserted).

Each arm overrides block_sizes / num_warps on the live welford seed. Reports per arm:
us, vs_seed, arm/tc, correctness.

Invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_wf_tile_ab.py \
    4096x16384 32768x8192 16384x768
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
from helion._utils import next_power_of_2 as np2  # noqa: E402
from triton.testing import do_bench  # noqa: E402

LOG_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "logs", "run3"))


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    return s[len(s) // 2], (s[-1] - s[0]) / s[len(s) // 2] if s[len(s) // 2] else None


def run_shape(M, N):
    fn, builder, tc_ref = KERNELS["welford"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"welford non-fp32 {a.dtype}"
    seed = dict(get_seed(fn, args)[0])
    sb = list(seed["block_sizes"])  # [M_block, combine, apply] (3 entries for welford)
    sw = seed["num_warps"]
    npn = np2(N)

    # block_sizes layout for welford: index1=combine, index2=apply (index0=M_block).
    # Build arms varying apply, combine, warps around the seed.
    def mk(combine=None, apply=None, warps=None):
        bs = list(sb)
        if combine is not None:
            bs[1] = min(npn, combine)
        if apply is not None:
            bs[2] = min(npn, apply)
        c = dict(seed)
        c["block_sizes"] = bs
        if warps is not None:
            c["num_warps"] = warps
        return c

    arms = {
        "seed": dict(seed),
        "apply4096": mk(apply=4096),
        "apply8192": mk(apply=8192),
        "applyNp2": mk(apply=npn),
        "combineNp2": mk(combine=npn),
        "combineNp2_apply4096": mk(combine=npn, apply=4096),
        "warps_x2": mk(warps=min(32, sw * 2)),
        "warps_half": mk(warps=max(1, sw // 2)),
        "apply4096_warpsx2": mk(apply=4096, warps=min(32, sw * 2)),
    }

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    results = {}
    base = None
    for name, cfg in arms.items():
        try:
            k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
            b = k.bind(args); b.ensure_config_exists(args)
            ok, err = check_correct(out_extract(b(*args)), ref)
            med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
            nb = dict(b._config)["block_sizes"]
            results[name] = {"us": med * 1000 if med else None, "spread": sp,
                             "bs": nb, "w": dict(b._config)["num_warps"],
                             "codegen": codegen_kind(b), "correct": ok, "maxerr": err}
            if name == "seed" and med:
                base = med * 1000
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:160]}

    print(f"\n=== welford({M},{N}) === tc={tc_med*1000:.1f}us seed_bs={sb} seed_w={sw} "
          f"np2(N)={npn}", flush=True)
    for name, r in results.items():
        if r.get("us"):
            vs = base / r["us"] if base else float("nan")
            print(f"  {name:>22} bs={str(r['bs']):>18} w{r['w']:<2} {r['us']:>9.1f}us "
                  f"vs_seed={vs:>5.3f} arm/tc={tc_med*1000/r['us']:.3f} "
                  f"sp={r['spread']:.2f} {'OK' if r['correct'] else 'BAD'}", flush=True)
        else:
            print(f"  {name:>22} -> {r}", flush=True)
    return {"shape": [M, N], "tc_us": tc_med * 1000, "seed_bs": sb,
            "arms": results}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    shape_args = sys.argv[1:] or ["4096x16384", "32768x8192", "16384x768"]
    out = []
    for s in shape_args:
        M, N = (int(x) for x in s.split("x"))
        try:
            out.append(run_shape(M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {s}: {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "wf_tile_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'wf_tile_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
