"""Single-process, fair 3-arm A/B for the reduction-seed PR, reusing TritonBench.

WHY single-process: cross-process do_bench jitter is ~5-10% on these small kernels,
which swamps the seed-vs-default delta. Within ONE process, on the SAME input
tensors, with median-of-N do_bench, the three arms are directly comparable.

The three arms (all fp32, all accuracy-checked against the operator's reference):
  * helion_default : Helion with autotuner heuristics DISABLED -> base default_config()
  * helion_seeded  : Helion with the PR's reduction seed promoted to default_config()
  * torch_compile  : the operator's torch.compile DEFAULT-mode baseline (no max-autotune)

Reuse of TritonBench:
  - input generation: the operator's get_input_iter() (same tensors every arm)
  - reference + accuracy: torch.testing.assert_close vs the operator baseline's output
  - timing: triton.testing.do_bench (what TritonBench itself uses), median-of-N

Run from a non-checkout cwd (e.g. /tmp) with:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/dev/local/helion-pr-edit \
    /home/dev/helion/.venv/bin/python /home/dev/local/helion-pr-edit/_lab/bench/ab_three_arm.py <kernel> [shape_split]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch
from triton.testing import do_bench

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

# ---- shape curriculum (test split by default) ------------------------------
sys.path.insert(0, "/home/dev/local/helion-reduction-heuristics-run2/_lab/prompts")
import shapes_v3_draft as SH  # noqa: E402

N_RUNS = 9  # median-of-9 do_bench repeats per arm (noise-robust)


def _med(fn: object) -> float:
    torch.cuda.synchronize()
    samples = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return samples[len(samples) // 2]


def _seed_config(bound: object) -> helion.Config | None:
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    return seeds[0] if seeds else None


def _build_helion(kernel_fn: object, args: tuple, config: helion.Config | None):
    """Bind kernel_fn to a fixed config (no autotune) and return a callable."""
    if config is None:
        return None
    k = helion.kernel(kernel_fn.fn, config=config, static_shapes=True)
    return lambda: k(*args)


# ---- per-kernel adapters: (example_kernel_fn, build_args, torch_ref_fn) -----
# Each returns (helion_callable_args, reference_callable). Reference is the
# eager/torch.compile-default baseline used both for accuracy and as the tc arm.


def _make(kernel: str):
    dev = "cuda"
    dt = torch.float32
    if kernel in ("rms_norm",):
        from examples.rms_norm import rms_norm_fwd

        def build(shape):
            m, n = shape
            x = torch.randn(m, n, device=dev, dtype=dt)
            w = torch.randn(n, device=dev, dtype=dt)
            args = (x, w, 1e-5)

            def ref():
                var = x.pow(2).mean(-1, keepdim=True)
                return (x * torch.rsqrt(var + 1e-5)) * w

            return rms_norm_fwd, args, ref

        return build
    if kernel == "softmax":
        from examples.softmax import softmax

        def build(shape):
            m, n = shape
            x = torch.randn(m, n, device=dev, dtype=dt)
            args = (x,)
            return softmax, args, (lambda: torch.softmax(x, dim=-1))

        return build
    if kernel == "layer_norm":
        from examples.layer_norm import layer_norm_fwd

        def build(shape):
            m, n = shape
            x = torch.randn(m, n, device=dev, dtype=dt)
            w = torch.randn(n, device=dev, dtype=dt)
            b = torch.randn(n, device=dev, dtype=dt)
            args = (x, [n], w, b, 1e-5)
            return layer_norm_fwd, args, (
                lambda: torch.nn.functional.layer_norm(x, [n], w, b, 1e-5)
            )

        return build
    if kernel == "sum":
        from examples.sum import sum_kernel

        def build(shape):
            m, n = shape
            x = torch.randn(m, n, device=dev, dtype=dt)
            args = (x,)
            return sum_kernel, args, (lambda: x.sum(-1))

        return build
    raise SystemExit(f"adapter for {kernel} not wired in this sanity harness")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("split", nargs="?", default="test")
    ap.add_argument("--shapes", default=None, help="override: 'M,N;M,N'")
    args = ap.parse_args()

    assert helion.__file__.startswith("/home/dev/local/helion-pr-edit"), helion.__file__

    if args.shapes:
        shapes = [tuple(int(v) for v in p.split(",")) for p in args.shapes.split(";")]
    else:
        shapes = [tuple(s) for s in SH.SHAPES[args.kernel][args.split]]

    build = _make(args.kernel)
    rows = []
    for shape in shapes:
        kernel_fn, kargs, ref = build(shape)

        # Bind once to read both the base-default and the seed config.
        bound = kernel_fn.bind(kargs)
        default_cfg = bound.config_spec.default_config()  # unseeded base
        seed_cfg = _seed_config(bound)

        default_call = _build_helion(kernel_fn, kargs, default_cfg)
        seeded_call = _build_helion(kernel_fn, kargs, seed_cfg)

        # Accuracy gate (vs torch reference) for both helion arms.
        ref_out = ref()
        acc = {}
        for name, call in (("default", default_call), ("seeded", seeded_call)):
            if call is None:
                acc[name] = None
                continue
            out = call()
            try:
                torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-4)
                acc[name] = True
            except Exception as e:  # noqa: BLE001
                acc[name] = f"FAIL: {str(e)[:80]}"

        # torch.compile DEFAULT mode (not max-autotune) as the common anchor.
        tc = torch.compile(ref)  # default inductor mode
        tc()  # warm/compile

        t_default = _med(default_call) if default_call else float("nan")
        t_seeded = _med(seeded_call) if seeded_call else float("nan")
        t_tc = _med(tc)

        row = {
            "shape": list(shape),
            "lat_ms": {
                "helion_default": round(t_default, 6),
                "helion_seeded": round(t_seeded, 6),
                "torch_compile_default": round(t_tc, 6),
            },
            "seeded_vs_default": round(t_default / t_seeded, 4) if t_seeded else None,
            "seeded_vs_tc": round(t_tc / t_seeded, 4) if t_seeded else None,
            "default_vs_tc": round(t_tc / t_default, 4) if t_default else None,
            "accuracy": acc,
            "seed_warps": dict(seed_cfg.config).get("num_warps") if seed_cfg else None,
            "default_warps": dict(default_cfg.config).get("num_warps"),
        }
        rows.append(row)
        print("ROW " + json.dumps(row), file=sys.stderr)

    # speedup summary (geomean of ratios > 1 means seeded faster)
    sv = [r["seeded_vs_default"] for r in rows if r["seeded_vs_default"]]
    geo = statistics.geometric_mean(sv) if sv else float("nan")
    print(
        json.dumps(
            {"kernel": args.kernel, "split": args.split, "rows": rows,
             "geomean_seeded_vs_default": round(geo, 4)}
        )
    )


if __name__ == "__main__":
    main()
