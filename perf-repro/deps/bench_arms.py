"""Single-process A/B/C benchmark for the 5 vLLM Helion quant kernels.

Arms (all replayed via helion.kernel(fn.fn, config=cfg) for symmetry):
  A. default      -- config_spec.default_config()  (base compiler default, NO heuristic)
  B. seed         -- compiler_seed_configs(env, device_ir)[0]  (the heuristic under test)
  C. vllm_shipped -- nvidia_h100.json config via nearest-shape lookup (the authoritative target)

Method (per memory + NEW_SERVER_SETUP.md §5): single process, all arms back-to-back on
identical inputs, do_bench(return_mode='median') (cold-L2), accuracy-gate vs refs.py before
timing, fp32 accum (kernels do this), requires_grad=False. Cache CONFIGS not us.

Why explicit replay (not HELION_AUTOTUNE_EFFORT=none): the reduction/pointwise heuristics
have promote_seed_to_default=False, so effort=none returns the BASE default for BOTH A and B
(A==B, meaningless). Extracting configs explicitly makes the seed actually exercised and all
arms symmetric.

Run: /home/dev/helion/.venv/bin/python bench_arms.py [--pilot]
"""
import argparse
import importlib
import json
import os
import re
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402
import triton.testing  # noqa: E402
import helion  # noqa: E402

import refs  # noqa: E402
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max  # noqa: E402

KRT = importlib.import_module("helion.runtime.kernel")
AH = importlib.import_module("helion._compiler.autotuner_heuristics")

# vLLM's shipped per-shape tuned configs (nvidia_h100.json etc.). These belong to vLLM and are
# read LIVE (never vendored) so the "vs vLLM tuned" arm always compares against current vLLM.
# Override with VLLM_CONFIG_DIR; default points at the vLLM source checkout on this box.
VLLM_CONFIG_DIR = os.environ.get(
    "VLLM_CONFIG_DIR", "/home/dev/local/vllm-src/vllm/kernels/helion/configs"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bench_results")
os.makedirs(OUT_DIR, exist_ok=True)

FP8_MIN, FP8_MAX = get_fp8_min_max()
FP8 = torch.float8_e4m3fn

# vLLM's GPU-name -> canonical platform (mirror of vllm/kernels/helion/utils.py).
_GPU_ALIASES = {
    "nvidia_h100_pcie": "nvidia_h100", "nvidia_h100_sxm5": "nvidia_h100",
    "nvidia_h100_80gb_hbm3": "nvidia_h100", "nvidia_h100_nvl": "nvidia_h100",
}


def canonical_gpu():
    name = re.sub(r"[\s/-]+", "_", torch.cuda.get_device_name(0).lower())
    return _GPU_ALIASES.get(name, name)


PLATFORM = canonical_gpu()


# ----------------------------------------------------------------------------
# Input builders (one per kernel) — build at an ARBITRARY (tok, hidden, group),
# including shapes outside the authors' enumerated grids (e.g. hidden=7168).
# Return (args_tuple, ref_fn, out_indices, returns_output).
# ----------------------------------------------------------------------------
def build_silu(tok, inter, group=None):
    x = torch.randn(tok, 2 * inter, device="cuda", dtype=torch.bfloat16)
    scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    return (x, scale), refs.ref_silu_mul_fp8, None, True


def build_dynamic_per_token(tok, hidden, group=None):
    x = torch.randn(tok, hidden, device="cuda", dtype=torch.bfloat16)
    result = torch.empty_like(x, dtype=FP8)
    scale = torch.empty((tok, 1), device="cuda", dtype=torch.float32)
    scale_ub = torch.mean(x).to(torch.float32)
    return (result, x, scale, scale_ub), refs.ref_dynamic_per_token, [0, 2], False


def build_rms_dynamic(tok, hidden, group=None):
    x = torch.randn(tok, hidden, device="cuda", dtype=torch.bfloat16)
    result = torch.empty_like(x, dtype=FP8)
    scale = torch.empty((tok, 1), device="cuda", dtype=torch.float32)
    scale_ub = torch.mean(x).to(torch.float32)
    residual = torch.randn_like(x)
    weight = torch.normal(1.0, 1.0, (hidden,), dtype=x.dtype, device="cuda")
    eps = 1e-6
    return ((result, x, weight, scale, eps, scale_ub, residual),
            refs.ref_rms_norm_dynamic, [0, 3, 6], False)


def build_per_token_group(tok, hidden, group=128):
    x = torch.randn(tok, hidden, device="cuda", dtype=torch.bfloat16)
    output_q = torch.empty_like(x, dtype=FP8)
    output_s = torch.empty((tok, hidden // group), device="cuda", dtype=torch.float32)
    eps = 1e-10
    return ((x, output_q, output_s, group, eps, FP8_MIN, FP8_MAX, False, False, False),
            refs.ref_per_token_group, [1, 2], False)


def build_rms_per_block(tok, hidden, group=128):
    x = torch.randn(tok, hidden, device="cuda", dtype=torch.bfloat16)
    result = torch.empty_like(x, dtype=FP8)
    scale = torch.empty((tok, hidden // group), device="cuda", dtype=torch.float32)
    scale_ub = torch.mean(x).to(torch.float32)
    residual = torch.randn_like(x)
    weight = torch.normal(1.0, 1.0, (hidden,), dtype=x.dtype, device="cuda")
    eps = 1e-6
    return ((result, x, weight, scale, eps, scale_ub, residual, group, False),
            refs.ref_rms_norm_per_block, [0, 3, 6], False)


# (label, module, kernel_attr, builder, json_subdir, key_builder(tok,hidden,group))
SPECS = {
    "silu_mul_fp8": (
        "kut.silu_mul_fp8", "silu_mul_fp8", build_silu, "silu_mul_fp8",
        lambda t, h, g: {"intermediate": h, "numtokens": t}),
    "dynamic_per_token_scaled_fp8_quant": (
        "kut.dynamic_per_token_scaled_fp8_quant", "dynamic_per_token_scaled_fp8_quant",
        build_dynamic_per_token, "dynamic_per_token_scaled_fp8_quant",
        lambda t, h, g: {"hidden_size": h, "num_tokens": t}),
    "rms_norm_dynamic_per_token_quant": (
        "kut.rms_norm_dynamic_per_token_quant", "rms_norm_dynamic_per_token_quant",
        build_rms_dynamic, "rms_norm_dynamic_per_token_quant",
        lambda t, h, g: {"hidden_size": h, "num_tokens": t}),
    "per_token_group_fp8_quant": (
        "kut.per_token_group_fp8_quant", "per_token_group_fp8_quant",
        build_per_token_group, "per_token_group_fp8_quant",
        lambda t, h, g: {"hidden_size": h, "group_size": g, "num_tokens": t}),
    "rms_norm_per_block_quant": (
        "kut.rms_norm_per_block_quant", "rms_norm_per_block_quant",
        build_rms_per_block, "rms_norm_per_block_quant",
        lambda t, h, g: {"hidden_size": h, "group_size": g, "num_tokens": t}),
}


def load_json_configs(subdir):
    path = os.path.join(VLLM_CONFIG_DIR, subdir, f"{PLATFORM}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)  # list of {key, config}


def nearest_vllm_config(entries, want_key):
    """Mirror the kut pick_config nearest-shape lookup: closest hidden/intermediate,
    closest group (if present), then smallest num_tokens >= want, else largest.
    Returns (helion.Config, chosen_key_dict, is_exact_dims)."""
    if not entries:
        return None, None, None
    size_field = "intermediate" if "intermediate" in want_key else "hidden_size"
    tok_field = "numtokens" if "numtokens" in want_key else "num_tokens"
    want_size = want_key[size_field]
    want_tok = want_key[tok_field]
    want_grp = want_key.get("group_size")

    # Skip the catch-all default entry (empty key {}), matching pick_config's
    # `if key.is_default(): continue`.
    keys = [e["key"] for e in entries if e["key"] and size_field in e["key"]]
    if not keys:
        return None, None, None
    best_size = min({k[size_field] for k in keys}, key=lambda s: abs(s - want_size))
    pool = [k for k in keys if k[size_field] == best_size]
    if want_grp is not None:
        best_grp = min({k["group_size"] for k in pool}, key=lambda s: abs(s - want_grp))
        pool = [k for k in pool if k["group_size"] == best_grp]
    toks = sorted({k[tok_field] for k in pool})
    best_tok = next((n for n in toks if n >= want_tok), toks[-1])
    chosen = next(k for k in pool if k[tok_field] == best_tok)
    cfg = next(e["config"] for e in entries if e["key"] == chosen)
    is_exact = (best_size == want_size) and (want_grp is None or best_grp == want_grp)
    return helion.Config(**cfg), chosen, is_exact


def clone_args(args):
    return tuple(a.clone() if torch.is_tensor(a) else a for a in args)


def cmp_outputs(ak, ar, out_idx, returns, out_k=None, out_r=None):
    """Return (ok, detail). fp8 tolerance: exact>=0.999 OR rel<=0.131 (~1 e4m3 ULP)."""
    pairs = []
    if returns:
        pairs.append(("out", out_k, out_r))
    else:
        labels = {0: "result", 1: "out_q", 2: "scale", 3: "scale", 6: "residual"}
        for oi in out_idx:
            pairs.append((labels.get(oi, f"arg{oi}"), ak[oi], ar[oi]))
    ok_all = True
    details = []
    for lbl, a, b in pairs:
        af, bf = a.to(torch.float32), b.to(torch.float32)
        diff = (af - bf).abs()
        exact = (af == bf).float().mean().item()
        rel = (diff / bf.abs().clamp(min=1e-12)).max().item()
        ok = exact >= 0.999 or rel <= 0.131
        ok_all = ok_all and ok
        details.append(f"{lbl}:exact={exact*100:.1f}%/rel={rel:.3g}")
    return ok_all, " ".join(details)


def run_arm(fn_kernel, cfg, builder, tok, hidden, group, out_idx, returns):
    """Compile fn at cfg, accuracy-gate vs ref, then time. Returns dict."""
    if cfg is None:
        return {"status": "no_config"}
    try:
        kfn = helion.kernel(fn_kernel.fn, config=cfg)
    except Exception as e:
        return {"status": f"compile_setup_fail: {type(e).__name__}: {e}"}

    # Accuracy gate on fresh inputs.
    args = builder(tok, hidden, group)[0]
    ak, ar = clone_args(args), clone_args(args)
    try:
        if returns:
            out_k = kfn(*ak)
            out_r = refs_call(builder, tok, hidden, group, ar)
            ok, detail = cmp_outputs(ak, ar, out_idx, returns, out_k, out_r)
        else:
            kfn(*ak)
            refs_call(builder, tok, hidden, group, ar)
            ok, detail = cmp_outputs(ak, ar, out_idx, returns)
    except Exception as e:
        return {"status": f"run_fail: {type(e).__name__}: {e}",
                "trace": traceback.format_exc()}

    # Time the bare kernel on fixed inputs (cold-L2 via do_bench L2 flush).
    timing_args = clone_args(args)
    try:
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(lambda: kfn(*clone_args(timing_args)),
                                     return_mode="median")
    except Exception as e:
        return {"status": f"bench_fail: {type(e).__name__}: {e}", "accuracy_ok": ok,
                "accuracy_detail": detail}
    return {"status": "ok", "accuracy_ok": ok, "accuracy_detail": detail,
            "median_ms": ms, "median_us": ms * 1e3,
            "config": cfg.to_json() if hasattr(cfg, "to_json") else str(cfg)}


def refs_call(builder, tok, hidden, group, ar):
    """Call the ref fn for this builder with the ref-cloned args."""
    ref_fn = builder(tok, hidden, group)[1]
    return ref_fn(*ar)


def extract_default_and_seed(fn_kernel, args):
    """Bind once; read base default + heuristic seed config from the compiled env.

    NB: bind() already ran compiler_seed_configs INSIDE its `with env:` context
    (kernel.py:567) and persisted the result on config_spec. We must NOT re-call
    compiler_seed_configs here — it does CompileEnvironment.current() internally and
    raises NoCurrentEnvironment outside the context (swallowed -> looks like no-fire).
    Read the persisted values, and take the env context for default_config()."""
    bk = fn_kernel.bind(args)
    spec = bk.env.config_spec
    with bk.env:
        default_cfg = spec._base_default_config() if hasattr(spec, "_base_default_config") \
            else spec.default_config()
    seeds = list(spec.compiler_seed_configs)  # populated by bind, inside its env ctx
    fired = list(spec.autotuner_heuristics)
    seed_cfg = seeds[0] if seeds else None
    return default_cfg, seed_cfg, fired


def bench_cell(label, tok, hidden, group):
    mod_name, kern_attr, builder, json_sub, key_fn = SPECS[label]
    mod = importlib.import_module(mod_name)
    fn_kernel = getattr(mod, kern_attr)
    out_idx, returns = builder(tok, hidden, group)[2], builder(tok, hidden, group)[3]

    cell = {"kernel": label, "tok": tok, "hidden": hidden, "group": group,
            "platform": PLATFORM}

    # Arms A (default) + B (seed): extract from a bind.
    probe_args = builder(tok, hidden, group)[0]
    try:
        default_cfg, seed_cfg, fired = extract_default_and_seed(fn_kernel, probe_args)
        cell["fired_heuristics"] = fired
    except Exception as e:
        cell["extract_fail"] = f"{type(e).__name__}: {e}"
        cell["extract_trace"] = traceback.format_exc()
        return cell

    # Arm C: vLLM shipped JSON nearest-lookup.
    entries = load_json_configs(json_sub)
    want_key = key_fn(tok, hidden, group)
    vllm_cfg, vllm_chosen, vllm_exact = (None, None, None)
    if entries is not None:
        vllm_cfg, vllm_chosen, vllm_exact = nearest_vllm_config(entries, want_key)
    cell["vllm_chosen_key"] = vllm_chosen
    cell["vllm_exact_dims"] = vllm_exact

    cell["arms"] = {
        "A_default": run_arm(fn_kernel, default_cfg, builder, tok, hidden, group, out_idx, returns),
        "B_seed": run_arm(fn_kernel, seed_cfg, builder, tok, hidden, group, out_idx, returns),
        "C_vllm_shipped": run_arm(fn_kernel, vllm_cfg, builder, tok, hidden, group, out_idx, returns),
    }
    # Ratios (only when both timings present).
    def us(arm):
        a = cell["arms"].get(arm, {})
        return a.get("median_us") if a.get("status") == "ok" else None
    a, b, c = us("A_default"), us("B_seed"), us("C_vllm_shipped")
    cell["ratios"] = {
        "seed_over_default": (b / a) if (a and b) else None,
        "seed_over_vllm": (b / c) if (b and c) else None,
    }
    return cell


# Focused grid: 5 kernels x 3 token counts x 2 dims = 30 cells.
GRID = {
    "silu_mul_fp8": {"toks": [32, 256, 8192], "dims": [14336, 28672], "group": None},
    "dynamic_per_token_scaled_fp8_quant": {"toks": [64, 256, 8192], "dims": [4096, 8192], "group": None},
    "rms_norm_dynamic_per_token_quant": {"toks": [64, 256, 8192], "dims": [4096, 8192], "group": None},
    "per_token_group_fp8_quant": {"toks": [16, 256, 8192], "dims": [7168, 4096], "group": 128},
    "rms_norm_per_block_quant": {"toks": [16, 256, 8192], "dims": [7168, 4096], "group": 128},
}
PILOT = {  # 1 decode + 1 prefill per kernel = 10 cells, validates harness end-to-end.
    "silu_mul_fp8": {"toks": [32, 8192], "dims": [14336], "group": None},
    "dynamic_per_token_scaled_fp8_quant": {"toks": [64, 8192], "dims": [4096], "group": None},
    "rms_norm_dynamic_per_token_quant": {"toks": [64, 8192], "dims": [4096], "group": None},
    "per_token_group_fp8_quant": {"toks": [16, 8192], "dims": [7168], "group": 128},
    "rms_norm_per_block_quant": {"toks": [16, 8192], "dims": [7168], "group": 128},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="10-cell pilot")
    args = ap.parse_args()
    grid = PILOT if args.pilot else GRID
    tag = "pilot" if args.pilot else "focused"

    torch.manual_seed(0)
    print(f"platform={PLATFORM}  device={torch.cuda.get_device_name(0)}  grid={tag}")
    print(f"helion={helion.__file__}\n")

    results = []
    for label, g in grid.items():
        for hidden in g["dims"]:
            for tok in g["toks"]:
                t0 = time.time()
                cell = bench_cell(label, tok, hidden, g["group"])
                cell["wall_s"] = round(time.time() - t0, 1)
                results.append(cell)
                r = cell.get("ratios", {})
                arms = cell.get("arms", {})
                def fmt(k):
                    a = arms.get(k, {})
                    return f"{a['median_us']:.1f}us" if a.get("status") == "ok" else a.get("status", "?")[:18]
                sd = r.get("seed_over_default")
                sv = r.get("seed_over_vllm")
                print(f"[{label[:22]:22s}] tok={tok:5d} h={hidden:5d} | "
                      f"A={fmt('A_default'):>12s} B={fmt('B_seed'):>12s} C={fmt('C_vllm_shipped'):>12s} | "
                      f"seed/def={sd:.3f} seed/vllm={sv:.3f}" if (sd and sv) else
                      f"[{label[:22]:22s}] tok={tok:5d} h={hidden:5d} | "
                      f"A={fmt('A_default'):>12s} B={fmt('B_seed'):>12s} C={fmt('C_vllm_shipped'):>12s} | "
                      f"fired={cell.get('fired_heuristics')} exact_vllm={cell.get('vllm_exact_dims')}")

    out_path = os.path.join(OUT_DIR, f"bench_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"platform": PLATFORM, "grid": tag, "cells": results}, f, indent=2)
    print(f"\nsaved {len(results)} cells -> {out_path}")


if __name__ == "__main__":
    main()
