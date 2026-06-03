"""RUN-3 softmax_two_pass persistent-vs-looped matched-lever A/B (NO autotune).

THE QUESTION (hub Phase-2 target): softmax_two_pass at WIDE N is currently LOOPED
by the seed (row_bytes > MULTILOAD_PERSIST_MAX_BYTES=240KiB):
  softmax(1024,65536)=256KiB -> looped (block_size_n=16384, w32), floor G~1.02 (=tc)
  softmax(512,131072) =512KiB -> looped (block_size_n=16384, w32), floor G~1.00 (=tc)
"At floor vs tc" is NOT "at oracle". The hub's physics hypothesis: softmax is a
SINGLE-OPERAND re-read (x re-read for the exp-sum pass after the max pass), so its
resident set is ONE row of x -- LIGHTER than cross_entropy's multi-pass working set
(logits + labels + target gather). softmax may therefore SPILL LESS than CE at the
same byte width and want a HIGHER persist threshold than CE's 240KiB. If a PERSISTENT
arm BEATS the looped seed here, the cap is leaving softmax-wide performance on the
table (a FURTHER gain, NOT a regression from EDIT-GATE-v2 -- which never moved these
shapes, they were already looped under num_load>=2).

softmax_two_pass is T2 (user-tiled): the reduction axis is a `block_sizes` entry
(the inner `hl.tile(n, block_size=block_size_n)`), NOT a `reduction_loops` knob.
PERSISTENT == block_size_n >= next_pow2(N) so the inner loop runs once; LOOPED ==
a capped chunk. This script finds the reduction-axis index GENERICALLY from the
ReductionFact (fact.block_id -> block_sizes index), then mutates ONLY that entry.

Arms (all from the live seed, mutating only the named field(s)):
  seed_looped   : the live seed as-is (block_size_n=16384, w32, ns1)
  persist       : block_size_n = next_pow2(N)  (whole row, one inner pass)
  persist_w16   : persistent + num_warps=16
  persist_ns2   : persistent + num_stages=2  (does pipelining help a persistent row?)
  chunk_32768   : block_size_n=32768 (a BIGGER looped chunk -- isolates chunk size
                  from full-persistence; if this alone closes the gap the cap, not
                  persistence, is the lever)
  tc_default    : torch.compile default of F.softmax(x, dim=1)

Reports per arm: median us, seed_looped/arm ratio, arm/tc ratio, codegen, correctness.
A persist arm FASTER than both seed_looped and tc_default confirms the cap is too low
for single-operand re-read softmax (the oracle then arbitrates the exact threshold).

Invocation (run from /tmp; AWAIT GPU-GRANTED first -- one GPU, hub holds the token):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_softmax_persist_ab.py \
    1024x65536 512x131072        # MxN shapes; default = the two wide looped shapes
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


def _np2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _run_cfg(fn_obj, args, cfg: dict):
    """Build helion.kernel(configs=[cfg]), bind, return (bound, normalized_cfg, codegen)."""
    k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b, dict(b._config), codegen_kind(b)


def _reduction_block_index(fn_obj, args) -> int:
    """The index into `block_sizes` that is the reduction axis (generic, from the
    fact's block_id) -- so the A/B mutates the RIGHT entry without hardcoding it."""
    k = helion.kernel(fn_obj.fn)
    b = k.bind(args)
    spec = b.env.config_spec
    fact = spec.reduction_facts[0]
    return spec.block_sizes.block_id_to_index(fact.block_id)


def run_shape(M, N):
    fn, builder, tc_ref = KERNELS["softmax"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"softmax{(M, N)} non-fp32 {a.dtype}"

    red_idx = _reduction_block_index(fn, args)
    seed_raw, _ = get_seed(fn, args)
    seed = dict(seed_raw)
    seed_bs = list(seed["block_sizes"])

    def _mut_bs(value: int) -> list[int]:
        bs = list(seed_bs)
        bs[red_idx] = value
        return bs

    np2_n = _np2(N)
    arms = {
        "seed_looped": dict(seed),
        "persist": {**seed, "block_sizes": _mut_bs(np2_n)},
        "persist_w16": {**seed, "block_sizes": _mut_bs(np2_n), "num_warps": 16},
        "persist_ns2": {**seed, "block_sizes": _mut_bs(np2_n), "num_stages": 2},
        "chunk_32768": {**seed, "block_sizes": _mut_bs(32768)},
    }

    # tc default (floor reference)
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    out_tc = out_extract(tc(args))
    ok_tc, _ = check_correct(out_tc, ref)
    assert ok_tc, f"tc FAIL softmax {(M, N)}"
    tc_med, tc_sp = _bench(lambda: tc(args))

    results = {}
    for name, cfg in arms.items():
        try:
            b, norm, cg = _run_cfg(fn, args, cfg)
            out = out_extract(b(*args))
            ok, err = check_correct(out, ref)
            med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
            results[name] = {
                "cfg_bs": norm.get("block_sizes"),
                "cfg_w": norm.get("num_warps"),
                "cfg_ns": norm.get("num_stages"),
                "codegen": cg,
                "correct": ok,
                "maxerr": err,
                "us": med * 1000 if med else None,
                "spread": sp,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[name] = {"oom": True}
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": f"{type(e).__name__}: {e}"[:200]}

    seed_us = results["seed_looped"].get("us")
    print(
        f"\n=== softmax({M},{N}) === row_bytes={N * 4 // 1024}KiB  "
        f"tc_default={tc_med * 1000:.1f}us (spread {tc_sp:.2f})  red_idx={red_idx}",
        flush=True,
    )
    print(
        f"  {'arm':>12} {'block_sizes':>16} {'w':>3} {'ns':>3} {'codegen':>10} "
        f"{'us':>9} {'seed/arm':>9} {'arm/tc':>8} {'spread':>7} corr",
        flush=True,
    )
    for name, r in results.items():
        if "us" in r and r["us"]:
            sa = seed_us / r["us"] if seed_us else float("nan")
            at = tc_med * 1000 / r["us"]
            print(
                f"  {name:>12} {str(r['cfg_bs']):>16} {r['cfg_w']:>3} "
                f"{r['cfg_ns']:>3} {r['codegen']:>10} {r['us']:>9.1f} "
                f"{sa:>9.3f} {at:>8.3f} {r['spread']:>7.2f} "
                f"{'OK' if r['correct'] else 'BAD'}",
                flush=True,
            )
        else:
            print(f"  {name:>12} -> {r}", flush=True)
    return {"shape": [M, N], "tc_us": tc_med * 1000, "red_idx": red_idx,
            "arms": results}


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} helion={helion.__file__}", flush=True)
    shape_args = sys.argv[1:] or ["1024x65536", "512x131072"]
    out = []
    for s in shape_args:
        M, N = (int(x) for x in s.split("x"))
        try:
            out.append(run_shape(M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {s}: {type(e).__name__}: {e}"[:200], flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "softmax_persist_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'softmax_persist_ab.json')}]", flush=True)


if __name__ == "__main__":
    main()
