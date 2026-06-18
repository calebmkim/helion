"""RUN-3 jsd-Band-B + softmax-small-N warps/block A/B (NO autotune).

Quick-oracle field-diffs (single-lever, simple):
  jsd(8192,30522):    seed bs=[4096,1] w32 -> oracle bs=[1024,1] w16  (so=1.196)
  softmax(131072,256):seed bs=[8,256]  w4  -> oracle bs=[8,256]  w8   (so=1.147)
=> jsd narrow-V: smaller R_BLOCK (4096->1024) + fewer warps (32->16). softmax
   small-N high-M: more warps (4->8). Both are warps/block levers (no pid). A/B each
   vs the seed + the oracle's pick + neighbors, do_bench median-of-7, correctness-
   gated, to confirm the lever is real (not quick-oracle noise) before any edit.

Generic: pass shapes as kernel:MxN; varies num_warps in {seed/2,seed,2*seed} and, for
the Band-B reduction-axis block_size, {1024,2048,4096} (the R_BLOCK).

Invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_warps_ab.py \
    jsd:8192x30522 jsd:8192x32000 softmax:131072x256 softmax:262144x128
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


def run_shape(kernel, M, N):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32
    seed = dict(get_seed(fn, args)[0])
    sw = seed["num_warps"]
    bs = list(seed["block_sizes"])

    arms = {"seed": dict(seed)}
    for w in sorted({max(1, sw // 2), sw, min(32, sw * 2), min(32, sw * 4)}):
        if w != sw:
            arms[f"w{w}"] = {**seed, "num_warps": w}
    # Band-B R_BLOCK sweep (only for jsd/kl_div: reduction axis is block_sizes[0])
    if kernel in ("jsd", "kl_div"):
        for rb in (1024, 2048, 4096):
            c = dict(seed); nbs = list(bs); nbs[0] = rb
            c["block_sizes"] = nbs
            arms[f"rb{rb}"] = c
            # combine R_BLOCK with the oracle warps
            c2 = dict(c); c2["num_warps"] = 16
            arms[f"rb{rb}_w16"] = c2

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
            results[name] = {"us": med * 1000 if med else None, "spread": sp,
                             "bs": dict(b._config)["block_sizes"],
                             "w": dict(b._config)["num_warps"],
                             "codegen": codegen_kind(b), "correct": ok}
            if name == "seed" and med:
                base = med * 1000
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:140]}

    print(f"\n=== {kernel}({M},{N}) === tc={tc_med*1000:.1f}us seed_bs={bs} seed_w={sw}",
          flush=True)
    for name, r in results.items():
        if r.get("us"):
            vs = base / r["us"] if base else float("nan")
            print(f"  {name:>12} bs={str(r['bs']):>12} w{r['w']:<2} {r['us']:>9.1f}us "
                  f"vs_seed={vs:>5.3f} arm/tc={tc_med*1000/r['us']:.3f} "
                  f"sp={r['spread']:.2f} {'OK' if r['correct'] else 'BAD'}", flush=True)
        else:
            print(f"  {name:>12} -> {r}", flush=True)
    return {"kernel": kernel, "shape": [M, N], "tc_us": tc_med * 1000, "arms": results}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    shape_args = sys.argv[1:] or ["jsd:8192x30522", "jsd:8192x32000",
                                  "softmax:131072x256", "softmax:262144x128"]
    out = []
    for s in shape_args:
        kernel, mn = s.split(":")
        M, N = (int(x) for x in mn.split("x"))
        try:
            out.append(run_shape(kernel, M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {s}: {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "warps_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'warps_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
