"""GENERAL Product-A G-measurement harness for the 9 reduction kernels.

measure(kernel_name, M, N) -> {
    shape, G_seed, seed_lat_us, tc_lat_us, seed_codegen, seed_cfg, correct, maxerr
}

  G_seed       = tc_default_lat / seed_lat
  tc_default   = torch.compile(<fp32 reference>) DEFAULT mode, do_bench median,
                 taken as the median-of-7 (7 independent do_bench median calls).
  seed         = the LIVE heuristic seed:
                   seeds = compiler_seed_configs(bound.env,
                                                 bound.host_function.device_ir)
                   seed  = seeds[0]
                 run with NO autotune via
                   helion.kernel(fn.fn, configs=[helion.Config(**dict(seed))])
                 then ensure_config_exists + bench.
  correct      = allclose(out, ref, rtol=1e-3, atol=1e-4) (NOT loosened). If
                 incorrect, G_seed is None and correct=False (gated).
  seed_codegen = "looped" if the seed's reduction loops over the reduction axis
                 (codegen contains a `for roffset`/range step < N), else
                 "persistent".

fp32 everywhere. The fn / arg-builder / fp32 reference for every kernel are copied
verbatim (logic only, NOT the path handling) from the existing per-kernel
harnesses in this directory.

Canonical invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=1 \
    PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction-2 \
    /home/calebkim/.conda/envs/helion/bin/python \
    /home/calebkim/helion-new-heuristics/wt-reduction-2/_lab/harness/run2_measure_g.py
"""

from __future__ import annotations

import json
import math
import os

import torch

import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), helion.__file__

from triton.testing import do_bench  # noqa: E402

from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward, TorchJSDBaseline  # noqa: E402
from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.welford import welford, eager_layer_norm  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
N_RUNS = 7
LONG = torch.int64


# --------------------------------------------------------------------------- #
# Per-kernel arg builders + fp32 references (copied from per-kernel harnesses).
# Each entry: builder(M, N) -> (args_tuple, ref_tensor_or_scalar, out_extract_fn)
# out_extract_fn maps the kernel output to the comparable tensor/scalar.
# --------------------------------------------------------------------------- #


def _first(o):
    return o[0] if isinstance(o, tuple) else o


def build_rms_norm(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    args = (x, w, EPS)
    ref = rms_norm_pytorch(x, w, EPS)
    return args, ref, _first


def build_layer_norm(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    args = (x, [n], w, b, EPS)
    ref = torch.nn.functional.layer_norm(x, [n], w, b, EPS)
    return args, ref, _first


def build_welford(m, n):
    weight = torch.rand(n, device="cuda", dtype=torch.float32)
    bias = torch.rand(n, device="cuda", dtype=torch.float32)
    x = torch.rand(m, n, device="cuda", dtype=torch.float32)
    args = (weight, bias, x, EPS)
    ref = eager_layer_norm(*args)
    return args, ref, _first


def build_softmax(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    args = (x,)
    ref = torch.nn.functional.softmax(x, dim=1)
    return args, ref, _first


def build_cross_entropy(m, n):
    logits = torch.randn(m, n, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, n, (m,), device="cuda", dtype=LONG)
    args = (logits, labels)
    ref = torch.nn.functional.cross_entropy(logits, labels)
    return args, ref, _first


def build_kl_div(m, n):
    yp = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1)
    yt = torch.randn(m, n, device="cuda", dtype=torch.float32).softmax(-1)
    args = (yp, yt, False, "batchmean", 1e-10)
    ref = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to("cuda")(
        yp, yt
    )
    return args, ref, _first


_JSD_BASELINE = TorchJSDBaseline(beta=0.5, ignore_index=-100)


def build_jsd(m, n):
    lq = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1)
    lp = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1)
    args = (lq, lp, None, 0.5, -100)
    ref = _JSD_BASELINE(lq, lp)
    return args, ref, _first


def build_sum(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    args = (x,)
    ref = torch.sum(x, dim=-1)
    return args, ref, _first


def build_long_sum(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    args = (x,)
    ref = torch.sum(x, dim=-1)
    return args, ref, _first


# fn: the helion kernel object ; builder: above ; tc_ref: callable used for
# torch.compile (DEFAULT mode) returning the same fp32 reference as `ref`.
KERNELS = {
    "rms_norm": (rms_norm_fwd, build_rms_norm,
                 lambda a: rms_norm_pytorch(*a)),
    "layer_norm": (layer_norm_fwd, build_layer_norm,
                   lambda a: torch.nn.functional.layer_norm(a[0], a[1], a[2],
                                                            a[3], a[4])),
    "welford": (welford, build_welford, lambda a: eager_layer_norm(*a)),
    "softmax": (softmax_two_pass, build_softmax,
                lambda a: torch.nn.functional.softmax(a[0], dim=1)),
    "cross_entropy": (cross_entropy, build_cross_entropy,
                      lambda a: torch.nn.functional.cross_entropy(a[0], a[1])),
    "kl_div": (kl_div_forward, build_kl_div,
               lambda a: torch.nn.KLDivLoss(reduction="batchmean",
                                            log_target=False).to("cuda")(a[0],
                                                                         a[1])),
    "jsd": (jsd_forward, build_jsd, lambda a: _JSD_BASELINE(a[0], a[1])),
    "sum": (sum_kernel, build_sum, lambda a: torch.sum(a[0], dim=-1)),
    "long_sum": (longsum, build_long_sum, lambda a: torch.sum(a[0], dim=-1)),
}


# --------------------------------------------------------------------------- #
# Measurement primitives.
# --------------------------------------------------------------------------- #


def median_do_bench(fn):
    """median-of-N_RUNS independent do_bench(return_mode='median') samples."""
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


def check_correct(out, ref):
    o = torch.as_tensor(out).to(torch.float32)
    r = torch.as_tensor(ref).to(torch.float32)
    ok = bool(torch.allclose(o, r, rtol=1e-3, atol=1e-4))
    maxerr = float((o - r).abs().max())
    return ok, maxerr


def get_seed(fn, args):
    bound = fn.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    return dict(seeds[0]), bound


def codegen_kind(bound_seed):
    """persistent vs looped over the reduction axis, read off real codegen.

    - reduction_loops[0] truthy => looped (T1 looped reduction)
    - otherwise inspect the emitted Triton: a `for roffset` loop, or any
      `tl.range(0, EXTENT, STEP)` over the reduction axis where STEP < EXTENT,
      means the reduction axis is looped; else persistent.
    """
    cfg = dict(bound_seed._config)
    rl = cfg.get("reduction_loops", [None])
    if rl and rl[0]:
        return "looped"
    try:
        tcode = bound_seed.to_triton_code(helion.Config(**cfg))
    except Exception:
        return "persistent"
    if "for roffset" in tcode:
        return "looped"
    # T2-style: a tl.range step smaller than its extent => looped
    import re
    consts = {
        mm.group(1): int(mm.group(2))
        for mm in re.finditer(
            r"(_BLOCK_SIZE_\d+)\s*=\s*tl\.constexpr\((\d+)\)", tcode
        )
    }
    for mm in re.finditer(r"tl\.range\(0,\s*(\d+),\s*(_BLOCK_SIZE_\d+)\)", tcode):
        extent = int(mm.group(1))
        step = consts.get(mm.group(2))
        if step is not None and step < extent:
            return "looped"
    return "persistent"


def measure(kernel_name, M, N):
    """Measure Product-A G_seed for one (kernel, M, N).

    Returns the standard dict. On correctness failure G_seed is None and
    correct=False (gated). Raises only on hard errors (e.g. OOM) — callers
    should catch per-shape.
    """
    fn, builder, tc_ref = KERNELS[kernel_name]
    args, ref, out_extract = builder(M, N)

    # --- seed (LIVE heuristic) ---
    seed, _ = get_seed(fn, args)
    seeded = helion.kernel(fn.fn, configs=[helion.Config(**dict(seed))])
    bound_s = seeded.bind(args)
    bound_s.ensure_config_exists(args)
    codegen = codegen_kind(bound_s)

    out_s = out_extract(bound_s(*args))
    correct, maxerr = check_correct(out_s, ref)

    seed_lat = median_do_bench(lambda: bound_s(*args)) if correct else None

    # --- torch.compile DEFAULT mode of the fp32 reference ---
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    out_tc = out_extract(tc(args))
    ok_tc, _ = check_correct(out_tc, ref)
    assert ok_tc, f"tc reference correctness FAIL {kernel_name} {(M, N)}"
    tc_lat = median_do_bench(lambda: tc(args))

    g_seed = (tc_lat / seed_lat) if correct else None
    return {
        "shape": [M, N],
        "G_seed": g_seed,
        "seed_lat_us": (seed_lat * 1000) if seed_lat is not None else None,
        "tc_lat_us": tc_lat * 1000,
        "seed_codegen": codegen,
        "seed_cfg": dict(seed),
        "correct": correct,
        "maxerr": maxerr,
    }


# --------------------------------------------------------------------------- #
# Task 2 — run on the in-sample-v2 shape list.
# --------------------------------------------------------------------------- #

IN_SAMPLE_V2 = {
    "rms_norm": [(256, 4096), (512, 8192), (256, 5120), (1024, 2560)],
    "layer_norm": [(512, 4096), (512, 8192), (256, 5120), (1024, 2560)],
    "welford": [(262144, 2560), (262144, 5120), (8192, 4096), (16384, 4096)],
    "softmax": [(8192, 3072), (8192, 32768)],
    "cross_entropy": [(8192, 32000), (8192, 50257), (4096, 128256),
                      (4096, 151936)],
    "kl_div": [(8192, 32000), (4096, 50257), (4096, 128256), (2048, 151936)],
    "jsd": [(8192, 32000), (8192, 50257), (8192, 128256), (4096, 151936)],
    "sum": [(256, 4096), (512, 8192), (32, 65536), (256, 262144)],
    "long_sum": [(4, 524288), (8, 393216)],
}


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}\n", flush=True)

    results = []
    per_kernel_g = {}
    weak = []   # G_seed < 0.90
    fails = []  # correctness failures
    ooms = []   # OOM / hard errors

    for kernel, shapes in IN_SAMPLE_V2.items():
        gs = []
        for (m, n) in shapes:
            tag = f"{kernel}({m},{n})"
            try:
                r = measure(kernel, m, n)
            except torch.cuda.OutOfMemoryError as e:  # noqa: PERF203
                torch.cuda.empty_cache()
                ooms.append({"kernel": kernel, "shape": [m, n],
                             "error": f"OOM: {type(e).__name__}"})
                print(f"[OOM ] {tag}: {type(e).__name__}", flush=True)
                continue
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                if "out of memory" in str(e).lower() or "OutOfMemory" in msg:
                    torch.cuda.empty_cache()
                    ooms.append({"kernel": kernel, "shape": [m, n],
                                 "error": msg[:200]})
                    print(f"[OOM ] {tag}: {msg[:120]}", flush=True)
                else:
                    fails.append({"kernel": kernel, "shape": [m, n],
                                  "error": msg[:300]})
                    print(f"[ERR ] {tag}: {msg[:160]}", flush=True)
                continue

            r["kernel"] = kernel
            results.append(r)
            if not r["correct"]:
                fails.append({"kernel": kernel, "shape": [m, n],
                              "maxerr": r["maxerr"], "G_seed": None})
            else:
                gs.append(r["G_seed"])
                if r["G_seed"] < 0.90:
                    weak.append({"kernel": kernel, "shape": [m, n],
                                 "G_seed": round(r["G_seed"], 4),
                                 "codegen": r["seed_codegen"]})
            gflag = (f"{r['G_seed']:.3f}" if r["G_seed"] is not None
                     else "  None")
            print(
                f"[{'OK ' if r['correct'] else 'BAD'}] {tag:>26} "
                f"G_seed={gflag:>7}  codegen={r['seed_codegen']:>10}  "
                f"seed_us="
                f"{(r['seed_lat_us'] if r['seed_lat_us'] else float('nan')):>9.1f}  "
                f"tc_us={r['tc_lat_us']:>9.1f}  maxerr={r['maxerr']:.2e}",
                flush=True,
            )
        per_kernel_g[kernel] = geomean(gs)

    print("\n" + "=" * 78, flush=True)
    print("PER-KERNEL GEOMEAN G_seed (in-sample-v2):", flush=True)
    for kernel in IN_SAMPLE_V2:
        gm = per_kernel_g.get(kernel)
        print(f"  {kernel:>14}: "
              f"{('%.4f' % gm) if gm is not None else 'n/a'}", flush=True)
    overall = geomean([per_kernel_g[k] for k in per_kernel_g
                       if per_kernel_g[k] is not None])
    print(f"  {'OVERALL':>14}: "
          f"{('%.4f' % overall) if overall is not None else 'n/a'}",
          flush=True)

    print("\nWEAKNESS TARGETS (G_seed < 0.90):", flush=True)
    if weak:
        for w in weak:
            print(f"  {w['kernel']}({w['shape'][0]},{w['shape'][1]})  "
                  f"G_seed={w['G_seed']}  codegen={w['codegen']}", flush=True)
    else:
        print("  (none)", flush=True)

    print("\nCORRECTNESS FAILURES:", flush=True)
    print(("  " + json.dumps(fails)) if fails else "  (none)", flush=True)
    print("\nOOM / SKIPPED:", flush=True)
    print(("  " + json.dumps(ooms)) if ooms else "  (none)", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("FULL JSON RESULTS:", flush=True)
    print(json.dumps({
        "gpu": gpu,
        "results": results,
        "per_kernel_geomean_G": {k: per_kernel_g[k] for k in per_kernel_g},
        "overall_geomean_G": overall,
        "weakness_targets_G_lt_0.90": weak,
        "correctness_failures": fails,
        "ooms": ooms,
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
