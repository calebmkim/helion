"""Diagnostic: pure Triton-vs-Triton head-to-head for a single (kernel, shape, dtype) cell.

NOT part of the headline harness. Use this when a cell's G_tc looks suspicious and you want to
rule out dispatch/launch OVERHEAD as the cause — it strips overhead from BOTH sides and compares
just the generated Triton kernels:

  * Helion seed  -> the kernel the reduction heuristic emits, launched via its compiled config.
  * torch.compile -> inductor's generated Triton, launched via inductor's own `call([args])`
                     (bypasses dynamo guards + the python wrapper — the trick: run_and_get_triton_code
                     gives the source, exec it, call `call([args])` directly).

For each we report:
  full_us      : the full callable (Helion kernel(...) / torch.compile(fn)) — what the harness times
  triton_us    : the stripped Triton launch (compiled-config replay / inductor call([args]))
  overhead_us  : full - triton  (the dispatch/guard/wrapper cost cold-L2 timing already hides)
All cold-L2, CUDA-event, median. If full_us ~= triton_us, overhead is already hidden and the
harness's G_tc is a fair kernel-vs-kernel comparison. If they diverge a lot, that cell's ratio is
overhead-contaminated and should be read off the triton_us column instead.

Run (example):
  PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    python perf-repro/tools/triton_head_to_head.py --corpus curriculum --kernel rms_norm \
    --shape 8192x1280 --dtype fp32
  # vLLM:  --corpus vllm --kernel rms_norm_dynamic_per_token_quant --shape 8192x4096 --dtype native
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_PERF = os.path.dirname(_TOOLS)
sys.path.insert(0, _PERF)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

# reuse the harness's builders + config extraction + cold-L2 timer (don't reinvent).
_spec = importlib.util.spec_from_file_location("prb", os.path.join(_PERF, "perf_report_bench.py"))
prb = importlib.util.module_from_spec(_spec)
sys.modules["prb"] = prb
_spec.loader.exec_module(prb)

import torch  # noqa: E402
from triton import runtime as _tr  # noqa: E402
from torch._inductor.utils import run_and_get_triton_code  # noqa: E402

_DI = _tr.driver.active.get_device_interface()


def _cold_med_us(fn, reps=200):
    """Cold-L2 median µs — same primitive/regime as the harness (single-fn, not interleaved)."""
    clear = prb._l2_clearer()
    for _ in range(30):
        fn()
    _DI.synchronize()
    st = [_DI.Event(enable_timing=True) for _ in range(reps)]
    en = [_DI.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        clear()
        st[i].record()
        fn()
        en[i].record()
    _DI.synchronize()
    return round(statistics.median([st[i].elapsed_time(en[i]) * 1e3 for i in range(reps)]), 3)


def _inductor_call(ref_fn, tensor_inputs):
    """Return a zero-arg thunk that launches inductor's generated Triton via `call([inputs])`,
    bypassing dynamo. `ref_fn` is the cell's (possibly zero-arg closure) torch reference;
    `tensor_inputs` are the graph's tensor inputs in order (inductor's call() wants tensors only —
    python scalars are baked in as constants). Verifies call() runs before returning."""
    torch._dynamo.reset()
    compiled = torch.compile(ref_fn)
    compiled()  # trigger compile (ref_fn is a zero-arg closure)
    code = run_and_get_triton_code(compiled)
    ns: dict = {}
    exec(compile(code, "<inductor>", "exec"), ns)  # noqa: S102
    if "call" not in ns:
        raise RuntimeError("inductor code has no call() — cannot strip to Triton")
    call = ns["call"]
    inputs = list(tensor_inputs)
    call(list(inputs))  # smoke: raises here (not during timing) if the input list is wrong
    return lambda: call(list(inputs)), code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--shape", required=True, help="MxN or MxNxG")
    ap.add_argument("--dtype", default="fp32")
    a = ap.parse_args()
    shape = tuple(int(x) for x in a.shape.split("x"))
    dt = prb._DT.get(a.dtype)

    family = a.corpus[:-4] if a.corpus.endswith("_gen") else a.corpus
    print(f"helion={prb.helion.__file__}")
    print(f"cell: {a.corpus}/{a.kernel} shape={shape} dtype={a.dtype}\n")

    # --- build the cell (reuse the harness builders) ---
    if family == "curriculum":
        kfn, args, ref, acc, tc_ref = prb._cur_build(a.kernel, shape[0], shape[1], dt)
    elif family == "transfer":
        kfn, args, ref, acc, tc_ref = prb._transfer_build(a.kernel, shape, dt)
    elif family == "mreduction":
        kfn, args, ref, acc, tc_ref = prb._mred_build(a.kernel, shape, dt)
    else:
        raise SystemExit(f"corpus {a.corpus} not supported by this diagnostic "
                         "(vllm kernels are in-place; use the harness cudagraph cross-check).")

    # --- Helion seed: full callable vs compiled-config replay ---
    torch._dynamo.reset()
    seed_cfg, base_cfg, fired = prb._extract_configs(kfn, args)
    print(f"fired heuristics: {fired}")
    seed_k = prb._replay(kfn, seed_cfg)
    seed_full = _cold_med_us(lambda: seed_k(*args))
    # (a replayed helion.kernel with a fixed config already skips autotune; its per-call path IS
    #  the deployment path, so full==triton for helion — we report it for symmetry.)
    print(f"\nHelion seed   full={seed_full:8.2f} us   (fixed-config replay = deployment path)")

    # --- torch.compile: full callable vs inductor Triton-only ---
    torch._dynamo.reset()
    tc_full_fn = torch.compile(tc_ref)
    tc_full_fn()  # compile/warm
    tc_full = _cold_med_us(tc_full_fn)
    # tensor inputs inductor's call() wants = the tensors captured in the cell args.
    tensor_inputs = [t for t in args if torch.is_tensor(t)]
    triton_thunk, code = _inductor_call(tc_ref, tensor_inputs)
    tc_triton = _cold_med_us(triton_thunk)
    print(f"torch.compile full={tc_full:8.2f} us   triton_only={tc_triton:8.2f} us   "
          f"overhead={tc_full - tc_triton:+.2f} us ({(tc_full - tc_triton) / tc_full * 100:+.1f}%)")

    print(f"\nHEAD-TO-HEAD (overhead stripped): Helion seed {seed_full:.2f} us  vs  "
          f"tc-triton {tc_triton:.2f} us  ->  G_tc_triton = {tc_triton / seed_full:.3f}")
    print("(if tc full ~= tc triton, the harness G_tc is already overhead-free)")


if __name__ == "__main__":
    main()
