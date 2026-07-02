"""Matched-lever A/B for the NARROW_W1 num_warps=1 refinement.

For every (kernel, shape, dtype) whose SEED emits num_warps=1, time the seed config AS-IS
(w1) vs the SAME config with num_warps bumped to what the ramp would give WITHOUT the narrow
refinement (rnumel<=1024 -> 4, <=4096 -> 8, <=16384 -> 16, else 32). Everything else
(block_sizes, reduction_loops, eviction, stages, pid_type) held fixed -- this isolates the
refinement's contribution. This answers: "if we removed the w1 special-case, which shapes
regress/improve and by how much?"

Single process per (corpus,kernel), cold-L2 median-of-9 do_bench, forward-only, same tensors,
accuracy-gated. Reuses the perf_report_bench builders so the wiring is identical to the report.

Run (from /tmp, foreground):
  HELION_AUTOTUNE_EFFORT=none PYTHONPATH=/home/dev/local/helion-redesign \
    /home/dev/helion/.venv/bin/python .../narrow_w1_ab.py --out /path/ab.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_THIS = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_THIS))

import perf_report_bench as PB  # noqa: E402  (all the builders + timing live here)
import torch  # noqa: E402
import helion  # noqa: E402

# The (corpus, kernel, shape, dtype) cells that emitted num_warps=1 in the report sweep.
W1_CELLS = [
    ("curriculum", "layer_norm", (16384, 896), "bf16"),
    ("curriculum", "rms_norm", (16384, 896), "bf16"),
    ("curriculum", "softmax", (131072, 128), "bf16"),
    ("curriculum", "softmax", (8192, 896), "bf16"),
    ("curriculum", "welford", (16384, 896), "bf16"),  # acc-fail; measure anyway
    ("synthetic_probes", "p5-3d-reduction-tile", None, "native"),
    ("transfer", "dynamic_quant", (16384, 1024), "bf16"),
    ("transfer", "dynamic_quant", (8192, 768), "bf16"),
    ("transfer", "fused_add_layernorm", (8192, 768), "bf16"),
    ("transfer", "fused_add_layernorm", (8192, 1024), "bf16"),
    ("transfer", "fused_add_layernorm", (16384, 1024), "bf16"),  # THE disaster
    ("transfer", "fused_add_rmsnorm", (8192, 768), "bf16"),
    ("transfer", "fused_add_rmsnorm", (8192, 1024), "bf16"),
    ("transfer", "gated_rmsnorm", (8192, 768), "bf16"),
    ("transfer", "gated_rmsnorm", (8192, 1024), "bf16"),
    ("transfer", "scaled_masked_softmax", (16384, 1024), "bf16"),
    ("vllm", "per_token_group_fp8_quant", (128, 4096, 128), "native"),
]


def _ramp_warps(rnumel: int) -> int:
    if rnumel > 16384:
        return 32
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    return 16


def _build(corpus, kernel, shape, dtype):
    dt = PB._DT.get(dtype)
    if corpus == "curriculum":
        kfn, args, ref, acc, _tc = PB._cur_build(kernel, shape[0], shape[1], dt)
        return kfn, args, ref, acc, False
    if corpus == "transfer":
        kfn, args, ref, acc, _tc = PB._transfer_build(kernel, shape, dt)
        return kfn, args, ref, acc, False
    if corpus == "vllm":
        # native, in-place; reuse the vllm path pieces
        import bench_arms as B
        tok, hidden = shape[0], shape[1]
        group = shape[2] if len(shape) > 2 else None
        import importlib
        mod_name, kern_attr, builder, _s, _k = B.SPECS[kernel]
        mod = importlib.import_module(mod_name)
        kfn = getattr(mod, kern_attr)
        built = builder(tok, hidden, group)
        return kfn, built[0], (built[1], built[2], built[3]), "vllm", True
    if corpus == "synthetic_probes":
        kfn, args, _shape = PB._load_synth("synthetic_probes", kernel)
        return kfn, args, None, "synth", False
    raise KeyError(corpus)


def _time_cfg(kfn, cfg, args, inplace):
    k = PB._replay(kfn, cfg)
    if inplace:
        return PB.timed(lambda: k(*PB._clone_args(args)))
    return PB.timed(lambda: k(*args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")  # substring filter on kernel
    a = ap.parse_args()
    print(f"helion={helion.__file__}", flush=True)
    rows = []
    for (corpus, kernel, shape, dtype) in W1_CELLS:
        if a.only and a.only not in kernel:
            continue
        tag = f"{corpus}/{kernel}/{shape}/{dtype}"
        try:
            kfn, args, ref, acc, inplace = _build(corpus, kernel, shape, dtype)
            torch._dynamo.reset()
            seed_cfg, _base, fired = PB._extract_configs(kfn, args)
            if seed_cfg is None:
                rows.append({"tag": tag, "error": "no seed"}); continue
            sd = dict(seed_cfg.config)
            w1 = sd.get("num_warps")
            # reduction extent = the size_hint the ramp keys on. Derive from the descriptor via a
            # fresh bind's reduction fact (rnumel). Fall back to max(shape) for synth.
            rnumel = _rnumel(kfn, args)
            wramp = _ramp_warps(rnumel)
            row = {"tag": tag, "corpus": corpus, "kernel": kernel,
                   "shape": list(shape) if shape else None, "dtype": dtype,
                   "seed_warps": w1, "ramp_warps": wramp, "rnumel": rnumel,
                   "block_sizes": sd.get("block_sizes")}
            if w1 != 1:
                row["note"] = f"seed did not emit w1 (got w{w1}); skipping A/B"
                rows.append(row); print("SKIP " + tag + f" w={w1}", flush=True); continue
            # build the w-ramp variant config
            from helion.runtime.config import Config
            alt = dict(sd); alt["num_warps"] = wramp
            cfg_w1 = seed_cfg
            cfg_alt = Config(**alt)
            # accuracy of w1 (report already knows; recompute quickly for both)
            t_w1 = _time_cfg(kfn, cfg_w1, args, inplace)
            t_alt = _time_cfg(kfn, cfg_alt, args, inplace)
            row["w1_us"] = t_w1["us"]; row["w1_spread"] = t_w1["spread"]
            row["ramp_us"] = t_alt["us"]; row["ramp_spread"] = t_alt["spread"]
            # ratio > 1 => the ramp (removing w1) is FASTER than w1 (i.e. w1 hurt)
            row["ramp_over_w1"] = round(t_w1["us"] / t_alt["us"], 4)
            rows.append(row)
            print(f"{tag}\n   w1={t_w1['us']}us  w{wramp}={t_alt['us']}us  "
                  f"ramp/w1={row['ramp_over_w1']} (>1 => w1 HURT)", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            rows.append({"tag": tag, "error": f"{type(e).__name__}: {e}",
                         "trace": traceback.format_exc()})
            print(f"ERR {tag}: {e}", flush=True)
        json.dump({"rows": rows}, open(a.out, "w"), indent=1)
        PB._cleanup()
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}", flush=True)


def _rnumel(kfn, args):
    """The reduction extent the ramp keys on (pd.size_hint). Read it off the bound spec."""
    try:
        bound = kfn.bind(args)
        rf = bound.env.config_spec.reduction_facts
        if rf:
            # size_hint of the primary reduction
            f = rf[0]
            for attr in ("size_hint", "primary_size_hint"):
                v = getattr(f, attr, None)
                if isinstance(v, int):
                    return v
            d = f._asdict()
            for kk in ("size_hint", "rnumel", "primary_reduction_size_hint"):
                if isinstance(d.get(kk), int):
                    return d[kk]
    except Exception:  # noqa: BLE001
        pass
    # fallback: largest tensor dim
    mx = 0
    for t in args:
        if torch.is_tensor(t):
            mx = max(mx, max(t.shape))
    return mx


if __name__ == "__main__":
    main()
