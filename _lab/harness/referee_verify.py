"""INDEPENDENT results-referee verification harness (written fresh, not the worker's).

Verifies v2 TritonReductionHeuristic claims for ONE (kernel, shape) per process.
Drive it with a fresh subprocess per shape from referee_run.sh.

For the given shape it:
  1. asserts worktree helion;
  2. gets the EXACT seed the registered heuristic emits via
     compiler_seed_configs(bound.env, bound.host_function.device_ir) -- NOT by
     re-deriving the heuristic's constants;
  3. runs the seed BARE via configs=[seed] (len==1 short-circuit -> no autotune;
     a structurally-invalid seed RAISES); asserts no autotune CSV written;
  4. confirms the seed was USED: normalized bound._config + generated Triton
     (persistent vs `for roffset` loop, num_warps in launcher);
  5. correctness vs torch.sum(x,-1) (or rms_norm ref): reports max_abs AND
     max_rel, on BOTH the standard randn input AND a non-degenerate (offset)
     input where row sums are far from zero, and reports pass/fail at the
     worker's atol=1e-3 AND at a tighter atol=1e-5 (rel-only) so any hiding of
     real error behind near-zero row sums is exposed;
  6. do_bench median over N launches (default 9) for seed, un-seeded default,
     and torch.compile(reference) default -- reports median + min/max spread +
     stddev for each;
  7. prints G_seed = tc_lat/seed_lat and G_default = tc_lat/default_lat as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from statistics import pstdev
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), (
    f"helion is {helion.__file__!r}, NOT the worktree. Refusing to run."
)
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402


def reference_sum(x):
    return torch.sum(x, dim=-1)


def get_kernel_and_ref(kernel):
    if kernel == "sum":
        from examples.sum import sum_kernel
        return sum_kernel, reference_sum, "single"
    if kernel == "long_sum":
        from examples.long_sum import longsum
        return longsum, reference_sum, "single"
    if kernel == "rms_norm":
        from examples.rms_norm import rms_norm_fwd
        from examples.rms_norm import rms_norm_pytorch
        return rms_norm_fwd, rms_norm_pytorch, "rms"


def build_args(kernel, shape, offset=0.0):
    m, n = shape
    if kernel == "rms_norm":
        x = torch.randn(m, n, device="cuda", dtype=torch.float32) + offset
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        return (x, w, 1e-5)
    x = torch.randn(m, n, device="cuda", dtype=torch.float32) + offset
    return (x,)


def call_ref(kernel, ref, args):
    if kernel == "rms_norm":
        return ref(*args)
    return ref(args[0])


def looped_signature(code):
    return "for roffset" in code


def num_warps_in_launcher(code):
    import re
    m = re.search(r"num_warps=(\d+)", code)
    return int(m.group(1)) if m else None


def median_do_bench(fn, n_runs):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(n_runs)]
    samples.sort()
    return {
        "median_us": median(samples) * 1000,
        "min_us": samples[0] * 1000,
        "max_us": samples[-1] * 1000,
        "std_us": (pstdev(samples) if len(samples) > 1 else 0.0) * 1000,
        "samples_us": [round(s * 1000, 3) for s in samples],
    }


def correctness(kernel, ref, out, args):
    o = (out[0] if isinstance(out, tuple) else out).to(torch.float32)
    r = call_ref(kernel, ref, args).to(torch.float32)
    abs_err = (o - r).abs()
    max_abs = float(abs_err.max())
    max_rel = float((abs_err / (r.abs() + 1e-12)).max())
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "pass_atol1e-3_rtol1e-3": bool(torch.allclose(o, r, rtol=1e-3, atol=1e-3)),
        "pass_atol1e-5_rtol1e-4": bool(torch.allclose(o, r, rtol=1e-4, atol=1e-5)),
        "ref_abs_min": float(r.abs().min()),
        "ref_abs_max": float(r.abs().max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=["sum", "long_sum", "rms_norm"])
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--n-runs", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    os.environ.setdefault("HELION_AUTOTUNE_RANDOM_SEED", str(a.seed))
    shape = (a.m, a.n)
    fn, ref, _ = get_kernel_and_ref(a.kernel)

    # --- autotune CSV guard: must stay absent ---
    csv_prefix = f"/tmp/referee_probe_{a.kernel}_{a.m}_{a.n}"
    os.environ["HELION_AUTOTUNE_LOG"] = csv_prefix
    csv_path = Path(csv_prefix + ".csv")
    if csv_path.exists():
        csv_path.unlink()

    args = build_args(a.kernel, shape)

    # --- 2. EXACT seed from the registered heuristic path ---
    bound_probe = fn.bind(args)
    seeds = compiler_seed_configs(bound_probe.env, bound_probe.host_function.device_ir)
    assert len(seeds) == 1, f"expected exactly 1 compiler seed, got {len(seeds)}: {seeds}"
    seed = seeds[0]
    heuristics_fired = list(bound_probe.env.config_spec.autotuner_heuristics)

    # eager validation: a structurally invalid seed RAISES here
    seed_cfg = helion.Config(**dict(seed))
    bound_probe.config_spec.normalize(seed_cfg)

    # --- 3. bare run, no autotune (len(configs)==1 short-circuit) ---
    seeded = helion.kernel(fn.fn, configs=[seed_cfg])
    bound_s = seeded.bind(args)
    bound_s.ensure_config_exists(args)
    normalized = dict(bound_s._config)
    autotune_ran = csv_path.exists() and csv_path.stat().st_size > 0

    # --- 4. seed-used proof ---
    code = bound_s.to_triton_code(helion.Config(**normalized))
    want_looped = bool(normalized.get("reduction_loops", [None])[0])
    got_looped = looped_signature(code)
    want_warps = normalized.get("num_warps")
    got_warps = num_warps_in_launcher(code)
    seed_used = (want_looped == got_looped) and (want_warps == got_warps)

    # --- 5. correctness: standard input + non-degenerate (offset) input ---
    out_s = bound_s(*args)
    corr_std = correctness(a.kernel, ref, out_s, args)
    # non-degenerate: large constant offset so row sums are far from zero
    args_off = build_args(a.kernel, shape, offset=10.0)
    out_off = bound_s(*args_off)
    corr_off = correctness(a.kernel, ref, out_off, args_off)

    # --- 6. latencies (same do_bench for all three) ---
    seed_lat = median_do_bench(lambda: bound_s(*args), a.n_runs)

    default_k = helion.kernel(fn.fn)
    bound_d = default_k.bind(args)
    cfg_d = bound_d.config_spec.default_config()
    default_k2 = helion.kernel(fn.fn, configs=[cfg_d])
    bound_d2 = default_k2.bind(args)
    bound_d2.ensure_config_exists(args)
    code_d = bound_d2.to_triton_code(helion.Config(**dict(bound_d2._config)))
    default_normalized = dict(bound_d2._config)
    out_d = bound_d2(*args)
    corr_d = correctness(a.kernel, ref, out_d, args)
    default_lat = median_do_bench(lambda: bound_d2(*args), a.n_runs)

    torch._dynamo.reset()
    if a.kernel == "rms_norm":
        tc = torch.compile(ref)
        out_tc = tc(*args)
        tc_lat = median_do_bench(lambda: tc(*args), a.n_runs)
    else:
        tc = torch.compile(reference_sum)
        out_tc = tc(args[0])
        tc_lat = median_do_bench(lambda: tc(args[0]), a.n_runs)
    corr_tc = correctness(a.kernel, ref, out_tc, args)

    result = {
        "kernel": a.kernel,
        "shape": [a.m, a.n],
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
        "torch_seed": a.seed,
        "helion_file": helion.__file__,
        "heuristics_fired": heuristics_fired,
        "seed_raw": dict(seed),
        "seed_normalized": normalized,
        "default_normalized": default_normalized,
        "default_codegen": "looped" if looped_signature(code_d) else "persistent",
        "default_warps": num_warps_in_launcher(code_d),
        "autotune_ran": autotune_ran,
        "seed_used": seed_used,
        "seed_codegen": "looped" if got_looped else "persistent",
        "seed_warps_in_code": got_warps,
        "seed_warps_normalized": want_warps,
        "correctness_standard": corr_std,
        "correctness_nondegenerate_offset10": corr_off,
        "correctness_default": corr_d,
        "correctness_tc": corr_tc,
        "seed_lat": seed_lat,
        "default_lat": default_lat,
        "tc_lat": tc_lat,
        "G_seed": tc_lat["median_us"] / seed_lat["median_us"],
        "G_default": tc_lat["median_us"] / default_lat["median_us"],
    }
    print("REFEREE_JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
