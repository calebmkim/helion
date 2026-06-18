"""Re-cert bias cross-check for a T1/loss kernel: cross_entropy (8192,65536).

Mirrors crosscheck_bias.py (Step 1, Task 4) but for cross_entropy, the harder
case: the Helion CE kernel ends in a host-side `losses.mean()` -> there may be a
SECOND launched kernel (the mean reduce). We:

  1. Extract the heuristic seed config + the generated Triton it actually runs
     (persistent vs looped + num_warps), and confirm seed_used.
  2. Count the FULL set of launched CUDA kernels per call via torch.profiler on
     BOTH sides (Helion seed AND torch.compile-default of F.cross_entropy) ->
     catches the host-side mean / multi-kernel split.
  3. Hand-roll a do_bench (same primitive triton.testing.do_bench, L2 flush on,
     median) over reps timing the SAME callables.
  4. Reconcile the hand-rolled medians vs an INDEPENDENT do_bench pass (the
     measure_g_ce.py protocol: median-of-7) -> two independent do_bench
     invocations of the SAME launched kernels. Agreement <~10-15% => harness
     unbiased on this kernel too.

IDENTICAL settings both sides: fp32 logits, int64 labels; tf32 off; no
cudagraphs; do_bench L2 flush on; same process; same inputs reused.
"""

from __future__ import annotations

import os
import sys

import torch

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
sys.path.insert(0, WT)

import helion  # noqa: E402

assert helion.__file__.startswith(WT), f"helion not worktree: {helion.__file__}"

from triton.testing import do_bench  # noqa: E402

from examples.cross_entropy import cross_entropy  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

LONG = torch.int64


def make_inputs(n: int, v: int):
    logits = torch.randn(n, v, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device="cuda", dtype=LONG)
    return logits, labels


def reference(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels)


def count_kernels(fn, n: int = 20):
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
        e
        for e in prof.events()
        if e.device_type.name == "CUDA" and e.device_time_total > 0
    ]
    total = 0
    names: dict[str, int] = {}
    for e in events:
        cnt = e.count if e.count else 1
        names[e.key] = names.get(e.key, 0) + cnt
        total += cnt
    per_call = total / n
    summary = sorted(f"{k} x{v} (~{v/n:.2f}/call)" for k, v in names.items())
    return round(per_call, 3), summary


def bench(fn, reps: int = 7):
    """Median-of-reps do_bench (same primitive/policy measure_g_ce.py uses:
    L2 flush on, return_mode median)."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    samples.sort()
    return {
        "median_ms": samples[len(samples) // 2],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def run_seed(args, cfg):
    k = helion.kernel(cross_entropy.fn, configs=[helion.Config(**dict(cfg))])
    b = k.bind(args)
    b.ensure_config_exists(args)
    tcode = b.to_triton_code(helion.Config(**dict(b._config)))
    return b, tcode


def main():
    n, v = 8192, 65536
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"=== CE CROSS-CHECK shape=({n},{v}) fp32 GPU={gpu} ===\n")

    logits, labels = make_inputs(n, v)
    assert logits.dtype == torch.float32 and labels.dtype == torch.int64
    assert logits.is_contiguous() and logits.stride() == (v, 1)
    args = (logits, labels)
    ref = reference(*args)

    # ---------------- HELION SEED SIDE ----------------
    bound = cross_entropy.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    seed = seeds[0]
    bound_s, tcode = run_seed(args, seed)
    want_looped = bool(dict(bound_s._config).get("reduction_loops", [None])[0])
    got_looped = "for roffset" in tcode
    seed_used = want_looped == got_looped
    n_jit = tcode.count("@triton.jit")
    print("--- Helion seed config (heuristic) ---")
    print(dict(seed))
    print(
        f"want_looped={want_looped} got_looped={got_looped} seed_used={seed_used} "
        f"num_warps={dict(seed)['num_warps']} @triton.jit_defs={n_jit}"
    )
    out_s = bound_s(*args)
    out_s = out_s[0] if isinstance(out_s, tuple) else out_s
    err_s = float((out_s.float() - ref.float()).abs().max())
    print(f"Helion correctness vs F.cross_entropy: max_abs={err_s:.3e}")
    assert seed_used, "seed NOT used"

    helion_call = lambda: bound_s(*args)
    kpc_h, knames_h = count_kernels(helion_call)
    print(f"Helion kernels/call (profiler) = {kpc_h}")
    for s in knames_h:
        print(f"   {s}")
    bench_h = bench(helion_call)
    print(f"Helion hand-rolled do_bench: {bench_h}")
    bench_h2 = bench(helion_call)  # INDEPENDENT second do_bench pass
    print(f"Helion 2nd indep do_bench:   {bench_h2}\n")

    # ---------------- TORCH.COMPILE-DEFAULT SIDE ----------------
    torch._dynamo.reset()
    tc = torch.compile(reference)  # DEFAULT mode (matches measure_g_ce.py)
    out_tc = tc(*args)
    err_tc = float((out_tc.float() - ref.float()).abs().max())
    print("--- torch.compile-default of F.cross_entropy ---")
    print(f"tc correctness vs F.cross_entropy: max_abs={err_tc:.3e}")
    tc_call = lambda: tc(*args)
    kpc_tc, knames_tc = count_kernels(tc_call)
    print(f"tc kernels/call (profiler) = {kpc_tc}")
    for s in knames_tc:
        print(f"   {s}")
    bench_tc = bench(tc_call)
    print(f"tc hand-rolled do_bench: {bench_tc}")
    bench_tc2 = bench(tc_call)
    print(f"tc 2nd indep do_bench:   {bench_tc2}\n")

    # ---------------- RECONCILE ----------------
    def pct(a, b):
        return 100.0 * (a - b) / b

    print("=== RECONCILE (two independent do_bench passes, same kernels) ===")
    print(
        f"Helion seed: passA={bench_h['median_ms']:.5f} ms  "
        f"passB={bench_h2['median_ms']:.5f} ms  "
        f"delta={pct(bench_h['median_ms'], bench_h2['median_ms']):+.2f}%"
    )
    print(
        f"tc-default : passA={bench_tc['median_ms']:.5f} ms  "
        f"passB={bench_tc2['median_ms']:.5f} ms  "
        f"delta={pct(bench_tc['median_ms'], bench_tc2['median_ms']):+.2f}%"
    )
    g_seed = bench_tc['median_ms'] / bench_h['median_ms']
    print(
        f"\nG_seed (tc/seed) = {g_seed:.3f}   "
        f"Helion seed median = {bench_h['median_ms']*1000:.1f} us   "
        f"tc median = {bench_tc['median_ms']*1000:.1f} us"
    )
    print(
        f"kernels/call: Helion={kpc_h}  tc={kpc_tc}  "
        f"(host-side mean / multi-kernel splits all counted)"
    )


if __name__ == "__main__":
    main()
