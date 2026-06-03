"""RUN-3 cross_entropy LOOPED-CHUNK matched-lever A/B (wide V; NO autotune).

P2: at wide vocab (V>=65536) the CE looped seed (LOOPED_CHUNK=16384, w32) is ~2x
SLOWER than tc-default, and at the 224KiB edge (V=57344) the oracle prefers a
LOOPED chunk=32768 over persistent. So the fixed looped chunk is too small for wide
rows. This script A/Bs the looped chunk x warps for the looped CE regime, do_bench
median-of-7, correctness-gated, one process -> find what looped config matches tc.

Arms (from the live seed, forcing reduction_loops=[chunk] + num_warps=w):
  loop<chunk>_w<warps> for chunk in {16384,32768,65536,131072}, w in {16,32}
  + tc_default reference.
Also tries persist (rl=[None]) as a control where feasible.

Reports per arm: median us, arm/tc ratio, correctness, codegen. Best looped arm
vs tc tells us whether ANY looped config matches tc (=> P2 is a looped-param fix) or
all lose (=> a 2-pass source ceiling: tc uses an online/fused strategy the standard
kernel cannot, separate Product-A-via-source opportunity).

Invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_ce_loopchunk_ab.py \
    8192x57344 4096x98304 8192x128256 2048x256000
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
CHUNKS = [16384, 32768, 65536, 131072]
WARPS = [16, 32]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    return s[len(s) // 2], (s[-1] - s[0]) / s[len(s) // 2] if s[len(s) // 2] else None


def run_shape(M, N):
    fn, builder, tc_ref = KERNELS["cross_entropy"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32

    seed_raw, _ = get_seed(fn, args)
    seed = dict(seed_raw)

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    out_tc = out_extract(tc(args))
    assert check_correct(out_tc, ref)[0], f"tc FAIL ce {(M,N)}"
    tc_med, tc_sp = _bench(lambda: tc(args))

    arms = {}
    for chunk in CHUNKS:
        for w in WARPS:
            arms[f"loop{chunk}_w{w}"] = {**seed, "reduction_loops": [chunk],
                                         "num_warps": w}
    arms["persist_w32"] = {**seed, "reduction_loops": [None], "num_warps": 32}

    results = {}
    for name, cfg in arms.items():
        try:
            k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
            b = k.bind(args)
            b.ensure_config_exists(args)
            out = out_extract(b(*args))
            ok, err = check_correct(out, ref)
            med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
            results[name] = {"us": med * 1000 if med else None, "spread": sp,
                             "codegen": codegen_kind(b), "correct": ok,
                             "maxerr": err,
                             "rl": dict(b._config).get("reduction_loops"),
                             "w": dict(b._config).get("num_warps")}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); results[name] = {"oom": True}
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:160]}

    print(f"\n=== cross_entropy({M},{N}) ===  tc_default={tc_med*1000:.1f}us "
          f"(spread {tc_sp:.2f})  [V*4={N*4//1024}KiB]", flush=True)
    print(f"  {'arm':>14} {'rl':>10} {'w':>3} {'codegen':>10} {'us':>9} "
          f"{'arm/tc':>7} {'spread':>7} corr", flush=True)
    ok_arms = [(n, r) for n, r in results.items() if r.get("us")]
    for name, r in sorted(results.items()):
        if r.get("us"):
            print(f"  {name:>14} {str(r['rl']):>10} {r['w']:>3} "
                  f"{r['codegen']:>10} {r['us']:>9.1f} "
                  f"{tc_med*1000/r['us']:>7.3f} {r['spread']:>7.2f} "
                  f"{'OK' if r['correct'] else 'BAD'}", flush=True)
        else:
            print(f"  {name:>14} -> {r}", flush=True)
    if ok_arms:
        best = max(ok_arms, key=lambda kv: 1.0 / kv[1]["us"])
        print(f"  -> BEST: {best[0]} {best[1]['us']:.1f}us  best/tc="
              f"{tc_med*1000/best[1]['us']:.3f}", flush=True)
    return {"shape": [M, N], "tc_us": tc_med * 1000, "arms": results}


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} helion={helion.__file__}", flush=True)
    shape_args = sys.argv[1:] or ["8192x57344", "4096x98304", "8192x128256",
                                  "2048x256000"]
    out = []
    for s in shape_args:
        M, N = (int(x) for x in s.split("x"))
        try:
            out.append(run_shape(M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {s}: {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "ce_loopchunk_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'ce_loopchunk_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
