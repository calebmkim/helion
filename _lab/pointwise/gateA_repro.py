#!/usr/bin/env python
"""Gate A axis-3 INDEPENDENT reproduction of the pointwise seed's win on swiglu.

This is a fresh, self-contained re-measurement authored by an independent engineer.
It deliberately does NOT import or copy the worker's harness (_lab/pointwise/ptw_bench.py).

Independence from the worker's method, by construction:
  * Timing mechanism is torch.cuda.Event start/stop pairs around each individual call
    -- NOT triton.testing.do_bench (the worker's mechanism).
  * Cold-L2 is enforced by EXPLICITLY zeroing a ~128MB scratch tensor before EVERY
    timed call (do_bench's internal L2-clear is not relied upon).
  * Median over >=20 independent reps (each rep = flush + one Event-timed call).
  * Reference, accuracy gate, arm construction and seed-config extraction are
    re-derived here from the public helion API, not reused from the worker's file.

Three arms on the SAME input tensors per shape:
  (a) default : helion.kernel(_swiglu_fwd.fn, config=default_cfg, static_shapes=True)
                where default_cfg = bound.config_spec.default_config()  (block_sizes=[32])
  (b) seed    : helion.kernel(_swiglu_fwd.fn, config=seed_cfg,    static_shapes=True)
                where seed_cfg = compiler_seed_configs(bound.env,
                                                       bound.host_function.device_ir)[0]
  (c) tc      : torch.compile(ref, mode='max-autotune-no-cudagraphs')

Accuracy gate (rtol=0.05, atol=0.005) runs BEFORE any timing; a failing arm is
reported as NaN time and excluded from the ratios.

Run (the driver runs this; the author does NOT):
  cwd=/tmp
  CUDA_VISIBLE_DEVICES=<n> PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-pointwise \
    /home/calebkim/.conda/envs/helion/bin/python /tmp/gate_a_axis3_swiglu_indep.py
"""

from __future__ import annotations

import json
import math
import os
import sys

# ---- environment pinning (must happen before torch picks a device) -------------------
WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Ensure the worktree helion is importable even if PYTHONPATH was not set by the caller.
if WT not in sys.path:
    sys.path.insert(0, WT)

import torch  # noqa: E402
import helion  # noqa: E402

# ---- hard independence / correctness asserts -----------------------------------------
assert helion.__file__.startswith(WT), f"WRONG helion in use: {helion.__file__}"
assert torch.cuda.is_available(), "CUDA is required for this measurement"

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from examples.swiglu import _swiglu_fwd  # noqa: E402

DEV = "cuda"
DTYPE = torch.bfloat16
RTOL, ATOL = 0.05, 0.005
TRAFFIC = 3  # swiglu reads a, reads b, writes out  -> 3 tensor-passes
REPS = 25  # >= 20 timed reps per arm
WARMUP = 5
SCRATCH_BYTES = 128 * 1024 * 1024  # ~128MB L2-flush scratch
SHAPES = [(16384, 11008), (4096, 4096), (512, 11008)]


# ---- standalone elementwise reference (fp32-internal silu, matches kernel compute) ----
def ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(a.float()).to(b.dtype) * b


# ---- cold-L2 scratch: an int8 buffer comfortably larger than any H100 L2 (50MB) -------
_SCRATCH = torch.empty(SCRATCH_BYTES, dtype=torch.int8, device=DEV)


def _flush_l2() -> None:
    """Evict the L2 by writing the whole scratch buffer (cold-cache emulation)."""
    _SCRATCH.zero_()


def time_call(call) -> float:
    """Median (over REPS) wall-time in microseconds of `call`, each rep preceded by a
    cold-L2 flush, measured with cuda Events. `call` must already be warm/compiled."""
    # warmup (also forces compilation / triton autotune if any)
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()

    samples = []
    for _ in range(REPS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        _flush_l2()  # explicit cold-L2 before the timed region
        torch.cuda.synchronize()
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)  # ms -> us
    samples.sort()
    return samples[len(samples) // 2]


def acc_ok(call, ref_out) -> bool:
    """Run the arm once and compare against the reference under the accuracy gate."""
    try:
        out = call()
        torch.testing.assert_close(out.float(), ref_out.float(), rtol=RTOL, atol=ATOL)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ACC-FAIL: {str(e)[:120]}", file=sys.stderr)
        return False


def make_helion_call(cfg, args):
    k = helion.kernel(_swiglu_fwd.fn, config=cfg, static_shapes=True)
    return lambda: k(*args)


def rd(x) -> float | None:
    return None if (x is None or x != x) else round(float(x), 4)


def main() -> None:
    print(f"helion: {helion.__file__}", file=sys.stderr)
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"device={torch.cuda.get_device_name(0)}", file=sys.stderr)
    print(f"timing: cuda.Event, {REPS} reps, explicit {SCRATCH_BYTES // (1024*1024)}MB "
          f"cold-L2 flush per rep, bf16", file=sys.stderr)

    results = []
    for (m, n) in SHAPES:
        torch._dynamo.reset()
        a = torch.randn(m, n, device=DEV, dtype=DTYPE)
        b = torch.randn(m, n, device=DEV, dtype=DTYPE)
        args = (a, b)
        ref_out = ref(a, b)

        bound = _swiglu_fwd.bind(args)
        default_cfg = bound.config_spec.default_config()
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        seed_cfg = seeds[0] if seeds else None

        print(f"\n[{m}x{n}] default block_sizes="
              f"{dict(default_cfg.config).get('block_sizes')}  "
              f"seed={dict(seed_cfg.config) if seed_cfg else None}", file=sys.stderr)

        d_call = make_helion_call(default_cfg, args)
        s_call = make_helion_call(seed_cfg, args) if seed_cfg is not None else None

        tc = torch.compile(ref, mode="max-autotune-no-cudagraphs")
        tc_call = lambda: tc(*args)

        # ---- accuracy gate BEFORE timing ----
        d_ok = acc_ok(d_call, ref_out)
        s_ok = acc_ok(s_call, ref_out) if s_call is not None else False
        t_ok = acc_ok(tc_call, ref_out)

        # ---- timing (only acc-passing arms) ----
        d_us = time_call(d_call) if d_ok else float("nan")
        s_us = time_call(s_call) if s_ok else float("nan")
        t_us = time_call(tc_call) if t_ok else float("nan")

        # ---- derived metrics ----
        bytes_moved = TRAFFIC * m * n * DTYPE.itemsize
        seed_tbps = (bytes_moved / (s_us * 1e-6)) / 1e12 if s_ok and s_us == s_us else None
        seed_vs_default = (d_us / s_us) if (d_ok and s_ok) else None  # >1 => seed faster
        g_seed_vs_tc = (t_us / s_us) if (s_ok and t_ok) else None      # >1 => seed beats tc

        row = {
            "shape": [m, n],
            "default_us": rd(d_us),
            "seed_us": rd(s_us),
            "tc_us": rd(t_us),
            "seed_vs_default": rd(seed_vs_default),
            "G_seed_vs_tc": rd(g_seed_vs_tc),
            "seed_tbps": rd(seed_tbps),
        }
        results.append(row)
        # human-readable per-shape line + cold-L2 sanity note
        cold_note = ""
        if seed_tbps is not None and seed_tbps >= 3.3:
            cold_note = "  <<< WARNING: >=3.3 TB/s implies L2 NOT cold"
        print(f"  default={rd(d_us)}us seed={rd(s_us)}us tc={rd(t_us)}us  "
              f"seed_vs_default={rd(seed_vs_default)}x  G(seed/tc)={rd(g_seed_vs_tc)}  "
              f"seed_tbps={rd(seed_tbps)}{cold_note}", file=sys.stderr)

        del a, b, args, ref_out, d_call, s_call, tc, tc_call
        torch.cuda.empty_cache()

    print("RESULT_JSON " + json.dumps(results))


if __name__ == "__main__":
    main()