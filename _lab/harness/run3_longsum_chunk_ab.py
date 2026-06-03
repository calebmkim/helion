"""RUN-3 long_sum looped-tail chunk A/B (NO autotune) — LOOPED_CHUNK no-regression.

LOOPED_CHUNK=16384 is SHARED by two regimes: the CE multi-load wide-V looped path
AND the long_sum >2^20 STRUCTURAL looped tail (N>2^20, persistent can't compile).
P2 wants to bump the chunk for CE (32768 is +3-8% there). Before changing the SHARED
constant, this checks whether long_sum's looped tail regresses with a bigger chunk.

long_sum(16,2097152) is the only train shape in the structural-looped regime; its
quick oracle matched the seed (seed/oracle=0.998) at chunk 16384. A/B chunk
{16384,32768,65536} x warps {16,32} vs tc, median-of-7, correctness-gated.

If long_sum's best stays at 16384, the constant must NOT be shared -> give the
multi-load looped path its OWN chunk. If long_sum is flat/better at 32768, the shared
bump is safe.

Invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_longsum_chunk_ab.py
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


def run_shape(kernel, M, N, chunks):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32
    seed = dict(get_seed(fn, args)[0])

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    arms = {}
    for c in chunks:
        for w in (16, 32):
            arms[f"loop{c}_w{w}"] = {**seed, "reduction_loops": [c], "num_warps": w}

    results = {}
    for name, cfg in arms.items():
        try:
            k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
            b = k.bind(args); b.ensure_config_exists(args)
            ok, err = check_correct(out_extract(b(*args)), ref)
            med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
            results[name] = {"us": med * 1000 if med else None, "spread": sp,
                             "codegen": codegen_kind(b), "correct": ok,
                             "rl": dict(b._config).get("reduction_loops"),
                             "w": dict(b._config).get("num_warps")}
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:160]}

    print(f"\n=== {kernel}({M},{N}) === tc={tc_med*1000:.1f}us", flush=True)
    ok_arms = [(n, r) for n, r in results.items() if r.get("us")]
    for name, r in sorted(results.items()):
        if r.get("us"):
            print(f"  {name:>14} {str(r['rl']):>10} w{r['w']:<2} {r['codegen']:>9} "
                  f"{r['us']:>9.1f}us arm/tc={tc_med*1000/r['us']:.3f} "
                  f"sp={r['spread']:.2f} {'OK' if r['correct'] else 'BAD'}", flush=True)
        else:
            print(f"  {name:>14} -> {r}", flush=True)
    if ok_arms:
        best = min(ok_arms, key=lambda kv: kv[1]["us"])
        print(f"  -> BEST {best[0]} {best[1]['us']:.1f}us", flush=True)
    return {"kernel": kernel, "shape": [M, N], "tc_us": tc_med * 1000,
            "arms": results}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    out = [run_shape("long_sum", 16, 2097152, [16384, 32768, 65536])]
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "longsum_chunk_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'longsum_chunk_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
