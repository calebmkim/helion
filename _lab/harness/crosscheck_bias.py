"""Independent harness-bias cross-check (Step 1, Task 4).

For one medium compute-bound shape, for BOTH Helion-default and
torch.compile-default:

  1. Extract the generated Triton + the winning/shipped config each path runs.
  2. Count the FULL set of launched CUDA kernels per call via torch.profiler
     (catches host-side / multi-kernel splits, e.g. a final cross-block reduce).
  3. Hand-roll a do_bench with IDENTICAL overhead (same inputs/layout/dtype/
     strides, same warmup/rep, L2 flush on) timing the SAME callable TritonBench
     times (Helion: rms_norm(inp, weight, eps=1e-6) -> RMSNormFunction.apply;
     tc: compiled(input)).
  4. Reconcile hand-rolled numbers vs the TritonBench do_bench numbers.

This is the diagnostic. End-to-end TritonBench remains the optimization objective:
we time the SAME wrapper callable both here and in TritonBench, so agreement
within ~10-15% certifies the harness is not systematically biased.

IDENTICAL settings asserted: fp32 both sides; tf32 off both sides (matmul +
cudnn); no cudagraphs; do_bench default warmup=25ms/rep=100ms with L2 flush on;
same process; same input tensors reused across both sides.
"""

from __future__ import annotations

import os
import sys

import torch

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
sys.path.insert(0, WT)

import helion  # noqa: E402

assert helion.__file__.startswith(WT), f"helion not worktree: {helion.__file__}"

from examples.rms_norm import rms_norm  # noqa: E402  (RMSNormFunction.apply path)
from examples.rms_norm import rms_norm_fwd  # noqa: E402  (the @helion.kernel fn)
from examples.rms_norm import rms_norm_pytorch  # noqa: E402

from triton.testing import do_bench  # noqa: E402

EPS = 1e-6  # tritonbench rms_norm operator uses eps=1e-6


def make_inputs(m: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the tritonbench operator's input gen for rms_norm fwd-only."""
    x = torch.randn((m, h), dtype=torch.float32, device="cuda")
    weight = torch.nn.Parameter(
        torch.ones(h, dtype=torch.float32, device="cuda"), requires_grad=False
    )
    return x, weight


class LlamaRMSNorm(torch.nn.Module):
    """Identical to the tritonbench operator's reference module."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


def count_kernels(fn, n: int = 20) -> tuple[int, list[str]]:
    """Count distinct CUDA kernel launches per call via torch.profiler.

    Warms up, then profiles n calls and divides. Returns (kernels_per_call,
    sorted unique kernel names)."""
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
        if e.device_type.name == "CUDA" and e.cuda_time_total > 0
    ]
    # count actual device kernel launches: events with a CUDA self time and a
    # device association; exclude memcpy/memset by name heuristic but report all.
    total = 0
    names: dict[str, int] = {}
    for e in events:
        cnt = e.count if e.count else 1
        names[e.key] = names.get(e.key, 0) + cnt
        total += cnt
    per_call = total / n
    summary = sorted(f"{k} x{v} (~{v/n:.2f}/call)" for k, v in names.items())
    return round(per_call, 3), summary


def bench(fn, reps: int = 5) -> dict:
    """Median-of-reps do_bench (same defaults TritonBench uses: L2 flush on,
    warmup=25ms, rep=100ms, return_mode median)."""
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    samples.sort()
    return {
        "median_ms": samples[len(samples) // 2],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "samples": samples,
    }


def main() -> None:
    m, h = 4096, 8192
    # --- identical global flags both sides ---
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert not torch.backends.cuda.matmul.allow_tf32
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    x, weight = make_inputs(m, h)
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, "fp32!"
    assert x.is_contiguous() and x.stride() == (h, 1)

    print(f"=== CROSS-CHECK shape=({m},{h}) fp32 GPU={gpu} ===\n")

    # ===================== HELION-DEFAULT SIDE =====================
    # The wrapper TritonBench times: rms_norm(inp, weight, eps=1e-6).
    helion_call = lambda: rms_norm(x, weight, EPS)
    # extract the winning/shipped config + generated Triton from the fwd kernel
    # (effort=none -> default_config; same path the benchmark used since the
    # bench run set HELION_AUTOTUNE_EFFORT=none).
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")
    bound_fwd = rms_norm_fwd.bind((x, weight, EPS))
    cfg = bound_fwd.config_spec.default_config()
    triton_code = bound_fwd.to_triton_code(cfg)
    print("--- Helion-default shipped config (default_config, effort=none) ---")
    print(cfg)
    print("\n--- Helion-default generated Triton (head) ---")
    print("\n".join(triton_code.splitlines()[:8]))
    looped = "for roffset" in triton_code
    n_triton_kernels = triton_code.count("@triton.jit")
    print(
        f"\nHelion: persistent? {not looped} (for roffset present={looped}); "
        f"@triton.jit defs in module={n_triton_kernels}"
    )

    # correctness of the timed wrapper vs reference
    out_h = rms_norm(x, weight, EPS)
    ref = rms_norm_pytorch(x, weight, EPS)
    h_max_abs = float((out_h.float() - ref.float()).abs().max())
    print(f"Helion correctness vs pytorch ref: max_abs={h_max_abs:.3e}")

    kpc_h, knames_h = count_kernels(helion_call)
    print(f"Helion kernels/call (profiler) = {kpc_h}")
    for s in knames_h:
        print(f"   {s}")
    bench_h = bench(helion_call)
    print(f"Helion hand-rolled do_bench: {bench_h}\n")

    # ===================== TORCH.COMPILE-DEFAULT SIDE =====================
    # The wrapper TritonBench times: compiled(input), compiled = torch.compile(module)
    module = LlamaRMSNorm(hidden_size=h, eps=EPS).to("cuda")
    module.weight = weight
    compiled = torch.compile(module)  # DEFAULT mode (matches new variant)
    tc_call = lambda: compiled(x)

    # capture generated inductor triton via TORCH_LOGS env (set by caller) or
    # the inductor codegen; here we count kernels via profiler and report the
    # inductor output path printed by TORCH_LOGS=output_code if enabled.
    out_tc = compiled(x)
    tc_max_abs = float((out_tc.float() - ref.float()).abs().max())
    print("--- torch.compile-default ---")
    print(f"tc correctness vs pytorch ref: max_abs={tc_max_abs:.3e}")
    kpc_tc, knames_tc = count_kernels(tc_call)
    print(f"tc kernels/call (profiler) = {kpc_tc}")
    for s in knames_tc:
        print(f"   {s}")
    bench_tc = bench(tc_call)
    print(f"tc hand-rolled do_bench: {bench_tc}\n")

    # ===================== RECONCILE =====================
    print("=== HAND-ROLLED SUMMARY (reconcile vs TritonBench task-3 numbers) ===")
    print(f"Helion-default hand-rolled median = {bench_h['median_ms']:.5f} ms")
    print(f"tc-default     hand-rolled median = {bench_tc['median_ms']:.5f} ms")
    print(
        f"Helion kernels/call={kpc_h}  tc kernels/call={kpc_tc}  "
        f"(headline kernel + any host-side/cross-block splits all counted)"
    )


if __name__ == "__main__":
    main()
