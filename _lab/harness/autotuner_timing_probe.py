"""Localize the autotuner-internal vs fair do_bench discrepancy for rms_norm
(32768,256) fp32, persistent (reduction_loops=[None]) block_sizes=[1], over
num_warps in {4,8,16,32}.

For each warp count:
  (a) FAIR standalone triton.testing.do_bench (median, L2 flush, proper warmup+sync)
      of the FULL Helion wrapper (out, inv_rms = rms_norm_fwd(x, weight, eps)).
  (b) The AUTOTUNER's exact internal timing path: helion.autotuner.benchmarking.do_bench
      with warmup=1, rep=50, return_mode="median" -- the same call _benchmark_function
      makes (functools.partial(fn, *args)).
  Correctness: max_abs vs rms_norm_pytorch reference for that config's output.
  Kernel count: torch.profiler kernels/call.
Tabulates fair vs autotuner ms and G=tc_default/seed.
"""

from __future__ import annotations

import functools
import os
import sys

import torch

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
sys.path.insert(0, WT)

import helion  # noqa: E402

assert helion.__file__.startswith(WT), f"helion not worktree: {helion.__file__}"

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402

from triton.testing import do_bench as triton_do_bench  # noqa: E402
from helion.autotuner.benchmarking import do_bench as autotuner_do_bench  # noqa: E402

EPS = 1e-6
M, N = 32768, 256
WARPS = [4, 8, 16, 32]


def make_inputs():
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32, device="cuda")
    weight = torch.ones(N, dtype=torch.float32, device="cuda")
    return x, weight


def build_compiled(x, weight, num_warps):
    """Compile rms_norm_fwd at persistent/block=1/num_warps via configs=[seed]
    (len==1 short-circuit -> the seed is used verbatim, no autotune)."""
    seed = helion.Config(
        reduction_loops=[None],
        block_sizes=[1],
        num_warps=num_warps,
        num_stages=1,
    )
    kern = helion.kernel(rms_norm_fwd.fn, configs=[seed])
    bound = kern.bind((x, weight, EPS))
    # realize the normalized config + generated code
    cfg = bound._config if getattr(bound, "_config", None) is not None else seed
    compiled = bound.compile_config(cfg, allow_print=False)
    triton_code = bound.to_triton_code(cfg)
    return kern, bound, cfg, compiled, triton_code


def count_kernels(fn, n=20):
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile

    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    events = [
        e for e in prof.events()
        if e.device_type.name == "CUDA" and e.cuda_time_total > 0
    ]
    total = 0
    names = {}
    for e in events:
        cnt = e.count if e.count else 1
        names[e.key] = names.get(e.key, 0) + cnt
        total += cnt
    summary = sorted(f"{k} x{v} (~{v/n:.2f}/call)" for k, v in names.items())
    return round(total / n, 3), summary


def fair_bench(fn, reps=5):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = sorted(float(triton_do_bench(fn, return_mode="median")) for _ in range(reps))
    return samples[len(samples) // 2], samples[0], samples[-1]


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x, weight = make_inputs()
    assert x.dtype == torch.float32 and x.is_contiguous()
    ref = rms_norm_pytorch(x, weight, EPS)  # [M,N] fp32

    print(f"=== rms_norm_fwd ({M},{N}) fp32 GPU={gpu} persistent block=1 ===\n")

    # tc-default baseline for G
    module_w = torch.nn.Parameter(weight.clone(), requires_grad=False)

    def tc_ref_module(inp):
        h = inp.to(torch.float32)
        v = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(v + EPS)
        return module_w * h

    compiled_tc = torch.compile(tc_ref_module)
    tc_call = lambda: compiled_tc(x)
    tc_med, tc_min, tc_max = fair_bench(tc_call)
    print(f"tc-default fair do_bench median = {tc_med*1000:.2f}us "
          f"(min {tc_min*1000:.2f} max {tc_max*1000:.2f})\n")

    rows = []
    for w in WARPS:
        kern, bound, cfg, compiled, triton_code = build_compiled(x, weight, w)
        looped = "for roffset" in triton_code
        njit = triton_code.count("@triton.jit")
        # grid hint from code
        # the helion wrapper returns (out, inv_rms)
        full_call = lambda: kern(x, weight, EPS)
        # autotuner times functools.partial(compiled_fn, *working_args)
        # working_args = (x, weight, EPS). compiled is the CompiledConfig.
        auto_call = functools.partial(compiled, x, weight, EPS)

        # correctness of THIS config's output
        out_seed, _inv = kern(x, weight, EPS)
        max_abs = float((out_seed.float() - ref.float()).abs().max())
        # also check the compiled-config path matches
        out_c, _ = compiled(x, weight, EPS)
        max_abs_c = float((out_c.float() - ref.float()).abs().max())

        # kernel count via the full wrapper call
        kpc, knames = count_kernels(full_call)

        # FAIR end-to-end do_bench
        fair_med, fair_min, fair_max = fair_bench(full_call)

        # AUTOTUNER internal do_bench (exact path: warmup=1, rep=50, median)
        # warm up first as _benchmark_function does
        auto_call()
        torch.cuda.synchronize()
        auto_ms = float(autotuner_do_bench(
            auto_call, return_mode="median", warmup=1, rep=50,
        ))

        g_fair = tc_med / fair_med if fair_med > 0 else float("nan")
        g_auto = tc_med / auto_ms if auto_ms > 0 else float("nan")
        rows.append((w, persistent_str(looped), njit, max_abs, max_abs_c, kpc,
                     fair_med*1000, fair_min*1000, fair_max*1000,
                     auto_ms*1000, g_fair, g_auto, knames))
        print(f"--- num_warps={w}: persistent={not looped} njit={njit} "
              f"max_abs(kern)={max_abs:.3e} max_abs(compiled)={max_abs_c:.3e} "
              f"kernels/call={kpc} ---")
        for s in knames:
            print(f"     {s}")
        print(f"  FAIR    do_bench median = {fair_med*1000:8.2f}us "
              f"(min {fair_min*1000:.2f} max {fair_max*1000:.2f})  G={g_fair:.3f}")
        print(f"  AUTOTUN do_bench median = {auto_ms*1000:8.2f}us  G={g_auto:.3f}\n")

    print("\n=== SUMMARY TABLE (32768,256) fp32 persistent block=1 ===")
    print(f"{'warps':>5} {'persist':>8} {'njit':>4} {'corr_max_abs':>13} "
          f"{'k/call':>7} {'FAIR_us':>10} {'AUTOTUN_us':>11} {'G_fair':>7} {'G_auto':>7}")
    for (w, ps, njit, ma, mac, kpc, fmed, fmin, fmax, ams, gf, ga, kn) in rows:
        print(f"{w:>5} {ps:>8} {njit:>4} {ma:>13.3e} {kpc:>7.2f} "
              f"{fmed:>10.2f} {ams:>11.2f} {gf:>7.3f} {ga:>7.3f}")
    print(f"\ntc-default fair median = {tc_med*1000:.2f}us (basis for G)")


def persistent_str(looped):
    return "loop" if looped else "persist"


if __name__ == "__main__":
    main()
