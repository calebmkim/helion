"""RESULTS-REFEREE independent reproduce for EDIT#4 (c3d90e8d):
STRUCTURED_APPLY_LOOP_CHUNK_BYTES 8192 -> 16384 (welford looped-apply tile 2048 -> 4096 fp32).

This is an INDEPENDENT harness (not the worker's run3_wf_tile_ab.py). It:
  (1) EMITS the live heuristic seed for each welford shape under the CURRENT helion
      (HEAD == c3d90e8d for helion/) and prints the NORMALIZED config, proving the
      wide-N apply tile == 4096 and the narrow apply tile is persistent/unchanged.
  (2) ISOLATES the apply-tile lever: benches the live seed (apply=4096) against an
      arm with apply forced to 2048 (the PARENT value 6c942ce3) holding combine /
      num_warps / M_block EQUAL, plus an applyNp2 overshoot arm (proves 16384 hurts).
  (3) My OWN bench: K independent do_bench(median) launches (default 9), reports
      median + spread (max-min)/median for BOTH arms, accuracy-gated fp32.

Accept rule (per claim 1): vs_parent = parent2048_us / seed4096_us. PASS that shape
if vs_parent >= 1.03 AND the gain exceeds the worse of the two arms' spreads.

Welford defaults bf16 upstream; the builder forces fp32 (asserted on the args).

Invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/REFEREE_edit4_apply_cap.py \
    --launches 9 --seed 1234 4096x16384 32768x8192 16384x768 8192x4096
"""

from __future__ import annotations

import argparse
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
    check_correct,
    codegen_kind,
    get_seed,
)
from helion._utils import next_power_of_2 as np2  # noqa: E402
from triton.testing import do_bench  # noqa: E402


def my_bench(fn, launches):
    """My OWN bench: `launches` independent do_bench(median) samples -> median,
    spread=(max-min)/median. Independent of the worker's _bench."""
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(launches))
    med = s[len(s) // 2]
    spread = (s[-1] - s[0]) / med if med else float("nan")
    return med, spread, s


def build_arm(seed, *, apply=None):
    """Return a Config dict identical to `seed` except the welford APPLY tile
    (block_sizes[2]) is forced to `apply` (capped at np2(N) by the caller).
    Combine (index1), M_block (index0), num_warps, num_stages all HELD EQUAL."""
    cfg = dict(seed)
    bs = list(seed["block_sizes"])
    if apply is not None:
        bs[2] = apply
    cfg["block_sizes"] = bs
    return cfg


def run_shape(M, N, launches):
    fn, builder, tc_ref = KERNELS["welford"]
    args, ref, out_extract = builder(M, N)

    # fp32 assertion on the actual args (welford defaults bf16 upstream).
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"welford arg non-fp32: {a.dtype}"

    seed, _ = get_seed(fn, args)
    sb = list(seed["block_sizes"])  # [M_block, combine, apply]
    npn = np2(N)
    seed_apply = sb[2]

    # Arms. seed == live (apply == whatever the heuristic emits). parent2048 forces
    # apply=2048 (capped at npn). applyNp2 forces apply=npn (the overshoot the worker
    # says HURTS). All hold combine/M_block/warps equal.
    arm_defs = {
        "seed_live": dict(seed),
        "parent2048": build_arm(seed, apply=min(npn, 2048)),
        "force4096": build_arm(seed, apply=min(npn, 4096)),
        "applyNp2": build_arm(seed, apply=npn),
    }

    # torch.compile reference timing (DEFAULT mode), for context only.
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0], "tc ref incorrect"
    tc_med, tc_sp, _ = my_bench(lambda: tc(args), launches)

    results = {}
    for name, cfg in arm_defs.items():
        try:
            k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
            b = k.bind(args)
            b.ensure_config_exists(args)
            ok, maxerr = check_correct(out_extract(b(*args)), ref)
            norm = dict(b._config)
            if ok:
                med, sp, samples = my_bench(lambda: b(*args), launches)
            else:
                med, sp, samples = None, None, None
            results[name] = {
                "us": med * 1000 if med is not None else None,
                "spread": sp,
                "norm_bs": list(norm["block_sizes"]),
                "norm_w": norm["num_warps"],
                "norm_stages": norm.get("num_stages"),
                "codegen": codegen_kind(b),
                "correct": ok,
                "maxerr": maxerr,
                "samples_us": [round(x * 1000, 2) for x in samples] if samples else None,
            }
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:200]}

    base = results.get("parent2048", {}).get("us")
    live = results.get("seed_live", {}).get("us")
    f4096 = results.get("force4096", {}).get("us")

    print(
        f"\n=== welford({M},{N}) === np2(N)={npn} seed_apply={seed_apply} "
        f"tc={tc_med * 1000:.1f}us(sp{tc_sp:.2f})",
        flush=True,
    )
    for name, r in results.items():
        if r.get("us"):
            vs_parent = base / r["us"] if base else float("nan")
            print(
                f"  {name:>12} bs={str(r['norm_bs']):>20} w{r['norm_w']:<2} "
                f"{r['codegen']:>10} {r['us']:>9.2f}us sp={r['spread']:.3f} "
                f"vs_parent2048={vs_parent:>6.3f} {'OK' if r['correct'] else 'BAD'} "
                f"maxerr={r['maxerr']:.1e}",
                flush=True,
            )
        else:
            print(f"  {name:>12} -> {r}", flush=True)

    # Per-shape verdict for claim 1 (wide partial win).
    verdict = None
    if base and live:
        vs_parent_live = base / live
        worst_sp = max(
            results["seed_live"].get("spread") or 0.0,
            results["parent2048"].get("spread") or 0.0,
        )
        wide = N >= 8192 and seed_apply == 4096  # wide-N looped-apply shape
        if wide:
            passes = vs_parent_live >= 1.03 and (vs_parent_live - 1.0) > worst_sp
            verdict = {
                "kind": "wide_partial_win",
                "vs_parent2048": round(vs_parent_live, 4),
                "worst_spread": round(worst_sp, 4),
                "pass": bool(passes),
            }
        else:
            # narrow/inert: expect byte-identity to parent (apply unchanged) -> vs~1.0
            inert = abs(vs_parent_live - 1.0) <= max(0.03, worst_sp)
            verdict = {
                "kind": "narrow_inert",
                "vs_parent2048": round(vs_parent_live, 4),
                "byte_identical_bs": list(results["seed_live"]["norm_bs"])
                == list(results["parent2048"]["norm_bs"]),
                "pass": bool(inert),
            }
    print(f"  VERDICT: {verdict}", flush=True)
    return {"shape": [M, N], "seed_apply": seed_apply, "np2": npn,
            "tc_us": tc_med * 1000, "arms": results, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launches", type=int, default=9)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("shapes", nargs="*",
                    default=["4096x16384", "32768x8192", "16384x768", "8192x4096"])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    print(
        f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES', '?')} helion={helion.__file__} "
        f"launches={a.launches} seed={a.seed} "
        f"APPLY_LOOP_CAP={helion._compiler.autotuner_heuristics.triton.TritonReductionHeuristic.STRUCTURED_APPLY_LOOP_CHUNK_BYTES}",
        flush=True,
    )
    out = []
    for s in a.shapes:
        M, N = (int(x) for x in s.split("x"))
        out.append(run_shape(M, N, a.launches))

    print("\n==== SUMMARY ====", flush=True)
    for r in out:
        v = r["verdict"]
        print(f"  welford{tuple(r['shape'])} seed_apply={r['seed_apply']} -> {v}",
              flush=True)
    log = os.path.join(_HARNESS_DIR, "..", "logs", "run3", "REFEREE_edit4.json")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[wrote {os.path.abspath(log)}]", flush=True)


if __name__ == "__main__":
    import helion._compiler.autotuner_heuristics.triton  # noqa: F401,E402
    main()
