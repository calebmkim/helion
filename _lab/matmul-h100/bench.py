"""H100 matmul-seed replay/probe driver (cwd=/tmp, PYTHONPATH=<worktree>).

Self-contained co-bench for the three kernels (matmul / fp8_gemm /
mamba2_chunk_state) at ARBITRARY (M,K,N) / mamba shapes. Built on
atlib.cold_l2_bench (plain L2-flushed do_bench median, NO cudagraph -> the #1
footgun here) + atlib.accuracy_ok (fp32-upcast gate). One process, identical
inputs, median-of-N for stability.

It PROBES the live heuristic (bind -> config_spec) to extract the emitted seed
config(s) + the unseeded base default, then co-benches any pool of
{token|literal-config} entries accuracy-gated cold-L2 back-to-back. Tokens:
  "default"   -> spec.default_config() (== _base_default_config when promote=False)
  "seed"/"seedK" -> spec.compiler_seed_configs[K]
  a dict      -> a literal Config to replay (manual candidate)

Usage:
  python bench.py --kernel matmul --shape 2048,2048,2048 --dtype bf16 \
      --configs '["default","seed",{"block_sizes":[128,128,64],"num_warps":8}]' \
      --tc --reps 5 --out /tmp/out.json
mamba --shape is b,seq,nh,chunk,hd,ds  (dtype from --dtype).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

# Portable worktree resolution: this file lives at <worktree>/_lab/matmul-h100/bench.py,
# so the worktree is three levels up. (No hardcoded machine path -> clones/B200 just work.)
_HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.dirname(os.path.dirname(_HERE))
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

from typing import Any  # noqa: E402

import torch  # noqa: E402

import helion  # noqa: E402

# Silent-wrong-helion guard (the footgun that has burned this project): `import helion`
# MUST resolve to THIS worktree, not a system/editable install.
assert os.path.realpath(helion.__file__).startswith(
    os.path.realpath(WORKTREE) + os.sep
), f"WRONG HELION: {helion.__file__} not under {WORKTREE}; run with PYTHONPATH={WORKTREE}"


# --- cold-L2 bench + accuracy gate (inlined so this lab is self-contained/portable) ---
def _as_tensor_list(x: Any) -> list[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (tuple, list)):
        return [t for t in x if isinstance(t, torch.Tensor)]
    raise TypeError(f"cannot extract tensors from {type(x)}")


def accuracy_ok(
    out: Any, ref: Any, rtol: float = 2e-2, atol: float = 2e-2, rel_floor: float = 5e-2
) -> tuple[bool, float]:
    """Accuracy gate: upcast to fp32, fail on NaN/inf, near-zero-safe via max_abs.
    out/ref may be a tensor or tuple/list of tensors (compared pairwise). Returns
    (ok, worst_max_abs)."""
    outs = _as_tensor_list(out)
    refs = _as_tensor_list(ref)
    if len(outs) != len(refs):
        return False, float("inf")
    worst = 0.0
    ok_all = True
    for o, r in zip(outs, refs):
        o32 = o.detach().to(torch.float32)
        r32 = r.detach().to(torch.float32)
        if o32.shape != r32.shape:
            if o32.numel() == r32.numel():
                o32 = o32.reshape(r32.shape)
            else:
                return False, float("inf")
        if torch.isnan(o32).any() or torch.isinf(o32).any():
            return False, float("inf")
        if torch.isnan(r32).any() or torch.isinf(r32).any():
            return False, float("inf")
        max_abs = (o32 - r32).abs().max().item()
        worst = max(worst, max_abs)
        denom = max(r32.abs().max().item(), 1e-6)
        rel = max_abs / denom
        ok = bool(torch.allclose(o32, r32, rtol=rtol, atol=atol)) or rel < rel_floor
        ok_all = ok_all and ok
    return ok_all, worst


def cold_l2_bench(fn: Any, n_warmup: int = 25, n_rep: int = 100) -> float:
    """Cold-L2 median latency (ms) via plain triton do_bench (L2-flush between reps). NO
    cudagraph (would defeat the L2 flush -> fake 3-5 TB/s artifact). Warms first."""
    import triton.testing

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=n_warmup, rep=n_rep, return_mode="median")


DEV = torch.device("cuda")
DT = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8": torch.float8_e4m3fn,
}
WIDTH_BITS = {"bf16": 16, "fp16": 16, "fp32": 32, "fp8": 8}

# matmul/fp8 want static_shapes (faithful: examples set it). mamba: probed both
# ways in Step 0; the deployed example uses default (False) but we resolve at runtime.
STATIC_SHAPES = {
    "matmul": True, "fp8_gemm": True, "mamba2_chunk_state": True, "bmm": True,
}


def _kernel_fn(kernel: str):
    if kernel == "matmul":
        from examples.matmul import matmul

        return matmul.fn
    if kernel == "fp8_gemm":
        from examples.fp8_gemm import fp8_gemm

        return fp8_gemm.fn
    if kernel == "mamba2_chunk_state":
        from examples.mamba2_chunk_state import helion_mamba2_chunk_state_kernel

        return helion_mamba2_chunk_state_kernel.fn
    if kernel == "bmm":
        from examples.bmm import bmm

        return bmm.fn
    raise ValueError(kernel)


def make_inputs(kernel: str, shape: list[int], dtype: str):
    """Return (args, ref_output, tc_fn|None, meta). tc_fn is the roofline arm
    (torch.matmul compiled / _scaled_mm); ref_output is the accuracy reference."""
    dt = DT[dtype]
    if kernel == "matmul":
        m, k, n = shape
        x = torch.randn(m, k, device=DEV, dtype=dt)
        y = torch.randn(k, n, device=DEV, dtype=dt)
        args = (x, y)
        ref = torch.matmul(x.float(), y.float())
        # tc roofline = compile ONCE (default inductor lowering -> cuBLAS mm), warm
        # outside the timed loop; recompiling inside fn() measured dispatch, not the kernel.
        _tc = torch.compile(torch.matmul, mode="max-autotune-no-cudagraphs")
        _tc(x, y)  # trigger autotune/compile now (not while timing)

        def tc_fn():
            return _tc(x, y)

        return args, ref, tc_fn, {"mnk": (m, n, k)}

    if kernel == "fp8_gemm":
        m, k, n = shape
        from helion._testing import HALF_DTYPE

        xf = torch.randn(m, k, device=DEV, dtype=torch.float32)
        yf = torch.randn(k, n, device=DEV, dtype=torch.float32)
        x_fp8 = xf.to(torch.float8_e4m3fn)
        y_fp8 = yf.to(torch.float8_e4m3fn).T.contiguous().T  # col-major
        args = (x_fp8, y_fp8)
        # accuracy ref = the exact fp8-input matmul (what Helion computes, fp32 accum)
        ref = (x_fp8.to(torch.float32)) @ (y_fp8.to(torch.float32))
        scale_a = torch.tensor(1.0, device=DEV)
        scale_b = torch.tensor(1.0, device=DEV)

        def tc_fn():
            return torch._scaled_mm(
                x_fp8, y_fp8, scale_a, scale_b, use_fast_accum=False,
                out_dtype=HALF_DTYPE,
            )

        return args, ref, tc_fn, {"mnk": (m, n, k)}

    if kernel == "bmm":
        # shape = [B, M, K, N]; A[B,M,K] @ B[B,K,N] -> [B,M,N]
        bsz, m, k, n = shape
        a = torch.randn(bsz, m, k, device=DEV, dtype=dt)
        bmat = torch.randn(bsz, k, n, device=DEV, dtype=dt)
        args = (a, bmat)
        ref = torch.bmm(a.float(), bmat.float())
        _tc = torch.compile(torch.bmm, mode="max-autotune-no-cudagraphs")
        _tc(a, bmat)

        def tc_fn():
            return _tc(a, bmat)

        return args, ref, tc_fn, {"mnk": (m, n, k), "batch": bsz}

    if kernel == "mamba2_chunk_state":
        b, seq, nh, chunk, hd, ds = shape
        ng = 1
        nchunks = (seq + chunk - 1) // chunk
        B = torch.rand(b, seq, ng, ds, device=DEV, dtype=dt)
        x = torch.rand(b, seq, nh, hd, device=DEV, dtype=dt)
        dt_t = torch.rand(b, nh, nchunks, chunk, device=DEV, dtype=dt)
        dA = torch.rand(b, nh, nchunks, chunk, device=DEV, dtype=dt)
        args = (B, x, dt_t, dA)
        from examples.mamba2_chunk_state import ref_chunk_state

        ref = ref_chunk_state(B, x, dt_t, dA)
        # mamba's "tc" is the eager einsum reference (no cuBLAS analog); skip tc.
        return args, ref, None, {"mnk": (hd, ds, chunk), "grid": b * nchunks * nh}

    raise ValueError(kernel)


def _build(kernel: str, fn, cfg, static_shapes: bool):
    if cfg is None:
        return helion.kernel(fn, static_shapes=static_shapes)
    c = helion.Config.from_dict(cfg) if isinstance(cfg, dict) else cfg
    return helion.kernel(fn, config=c, static_shapes=static_shapes)


def _median_of(fn, reps: int) -> float:
    vals = sorted(cold_l2_bench(fn) for _ in range(reps))
    return vals[len(vals) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--shape", required=True, help="comma ints")
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--configs", default='["default","seed"]')
    ap.add_argument("--tc", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--static-shapes", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--probe-only", action="store_true",
                    help="emit facts + seed/default configs, skip all GPU timing")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    shape = [int(s) for s in a.shape.split(",")]
    fn = _kernel_fn(a.kernel)
    ss = (
        STATIC_SHAPES[a.kernel]
        if a.static_shapes == "auto"
        else (a.static_shapes == "true")
    )
    info: dict = {
        "kernel": a.kernel, "shape": shape, "dtype": a.dtype,
        "width_bits": WIDTH_BITS[a.dtype], "static_shapes": ss,
        "helion_file": helion.__file__,
    }
    try:
        args, ref, tc_fn, meta = make_inputs(a.kernel, shape, a.dtype)
        info["meta"] = meta
        # --- probe the live heuristic ---
        kp = _build(a.kernel, fn, None, ss)
        bound = kp.bind(kp.normalize_args(*args))
        spec = bound.env.config_spec
        seeds = [dict(c) for c in spec.compiler_seed_configs]
        default_cfg = dict(spec.default_config())
        facts = spec.matmul_facts
        info["matmul_facts"] = [
            {
                "lhs_ndim": f.lhs_ndim, "rhs_ndim": f.rhs_ndim,
                "static_m": f.static_m, "static_n": f.static_n, "static_k": f.static_k,
                "m_block_id": f.m_block_id, "n_block_id": f.n_block_id,
                "k_block_id": f.k_block_id,
                "lhs_dtype": str(f.lhs_dtype), "rhs_dtype": str(f.rhs_dtype),
            }
            for f in facts
        ]
        info["n_block_size_specs"] = len(spec.block_sizes)
        info["heuristics_fired"] = list(spec.autotuner_heuristics)
        info["n_seeds"] = len(seeds)
        info["seed_configs"] = seeds
        info["default_config"] = default_cfg

        if a.probe_only:
            info["ok"] = True
            info["results"] = []
            with open(a.out, "w") as fh:
                json.dump(info, fh, indent=2, default=str)
            print("PROBE " + json.dumps(
                {"ok": True, "fired": info["heuristics_fired"],
                 "n_seeds": len(seeds), "seed0": seeds[0] if seeds else None,
                 "facts": info["matmul_facts"]}, default=str))
            return

        # --- resolve the pool ---
        pool = json.loads(a.configs)
        resolved: list[tuple[str, dict | None]] = []
        for item in pool:
            if item == "default":
                resolved.append(("default", default_cfg))
            elif item == "seed":
                resolved.append(("seed0", seeds[0] if seeds else None))
            elif isinstance(item, str) and item.startswith("seed"):
                i = int(item[4:])
                resolved.append((item, seeds[i] if i < len(seeds) else None))
            elif isinstance(item, dict):
                resolved.append(("manual:" + json.dumps(item, sort_keys=True), item))
            else:
                resolved.append((str(item), None))

        results = []
        cache: dict[str, dict] = {}
        if a.tc and tc_fn is not None:
            rec = {"label": "tc", "kind": "tc"}
            try:
                out = tc_fn()
                ok, ma = accuracy_ok(out, ref)
                rec["accuracy_ok"] = ok
                rec["max_abs"] = ma
                rec["perf_ms"] = _median_of(tc_fn, a.reps) if ok else None
            except Exception as e:
                rec.update({"accuracy_ok": False, "perf_ms": None,
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-1200:]})
            results.append(rec)

        for label, cfg in resolved:
            rec = {"label": label, "config": cfg, "kind": "helion"}
            if cfg is None:
                rec.update({"accuracy_ok": False, "perf_ms": None, "error": "no config"})
                results.append(rec)
                continue
            ck = json.dumps(cfg, sort_keys=True, default=str)
            if ck in cache:
                p = cache[ck]
                rec.update({"accuracy_ok": p["accuracy_ok"], "perf_ms": p["perf_ms"],
                            "max_abs": p.get("max_abs"), "dup_of": p["label"]})
                results.append(rec)
                continue
            try:
                k = _build(a.kernel, fn, cfg, ss)
                out = k(*args)
                ok, ma = accuracy_ok(out, ref)
                rec["accuracy_ok"] = ok
                rec["max_abs"] = ma
                rec["perf_ms"] = _median_of(lambda: k(*args), a.reps) if ok else None
            except Exception as e:
                rec.update({"accuracy_ok": False, "perf_ms": None,
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-1200:]})
            cache[ck] = rec
            results.append(rec)

        info["ok"] = True
        info["results"] = results
        # ratios vs default and tc for the helion arms
        d = next((r for r in results if r["label"] == "default"), None)
        tc = next((r for r in results if r["label"] == "tc"), None)
        for r in results:
            if r.get("perf_ms"):
                if d and d.get("perf_ms"):
                    r["x_over_default"] = round(d["perf_ms"] / r["perf_ms"], 4)
                if tc and tc.get("perf_ms"):
                    r["G_vs_tc"] = round(tc["perf_ms"] / r["perf_ms"], 4)
    except Exception as e:
        info.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                     "traceback": traceback.format_exc()})

    with open(a.out, "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    # compact stdout summary
    summ = {"ok": info.get("ok"), "n_seeds": info.get("n_seeds"),
            "fired": info.get("heuristics_fired"),
            "facts": info.get("matmul_facts")}
    if info.get("ok"):
        summ["bench"] = [
            {"label": r["label"], "ms": r.get("perf_ms"), "acc": r.get("accuracy_ok"),
             "xD": r.get("x_over_default"), "G": r.get("G_vs_tc")}
            for r in info["results"]
        ]
    print("BENCH " + json.dumps(summ, default=str))


if __name__ == "__main__":
    main()
