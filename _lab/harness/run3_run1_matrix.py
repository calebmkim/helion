"""RUN-3 / RUN-1 matrix: the cheap no-autotune comparison across TEST shapes.

For each (kernel, TEST shape) measures THREE configs with the SAME do_bench
primitive tritonbench uses (triton.testing.do_bench, median-of-7), fp32, with an
allclose / rel-err accuracy gate:

  C1 = Helion DEFAULT config, no autotune  -> default_lat_us
  C4 = Helion SEED (ReductionHeuristic), no autotune (configs=[seed]) -> seed_lat_us
  C7 = torch.compile(<fp32 reference>) DEFAULT mode -> tc_lat_us

  g_seed    = tc_lat / seed_lat       g_default = tc_lat / default_lat

Reuses the VALIDATED, correctly-wired plumbing in run2_measure_g (KERNELS dict,
median_do_bench, check_correct, get_seed, codegen_kind) — that module asserts
helion is wt-reduction-2 and does NO sys.path.insert (avoids the prefix trap that
TEST_readonce.py has). TEST_SHAPES copied verbatim from TEST_readonce.py.

CHECKPOINTED: writes <out>/run1_<kernel>.json after EVERY shape; on restart it
SKIPS shapes already present (resumable / interruptible). One file per kernel ->
safe for parallel per-GPU processes (disjoint kernel subsets, no write contention).

Invocation (per GPU, disjoint kernel subset), run from a non-checkout cwd:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction-2 \
    /home/calebkim/.conda/envs/helion/bin/python \
    /home/calebkim/helion-new-heuristics/wt-reduction-2/_lab/harness/run3_run1_matrix.py \
    --kernels rms_norm,sum,long_sum --out /home/calebkim/helion-new-heuristics/wt-reduction-2/_lab/logs/run3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

import torch

import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), f"WRONG helion (prefix trap?): {helion.__file__}"

# Reuse the validated, correctly-wired plumbing.
from run2_measure_g import (  # noqa: E402
    KERNELS,
    check_correct,
    codegen_kind,
    get_seed,
    median_do_bench,
)

# Canonical TEST split (verbatim from TEST_readonce.py TEST_SHAPES).
# NOTE: welford (262144,5120) and rms_norm (256,4096) were later promoted into
# in-sample-v2 during run-2 development; kept here for a COMPLETE TEST eval table
# (flagged in the report).
TEST_SHAPES = {
    "rms_norm": [(256, 4096), (2048, 2560), (2048, 1025), (4096, 10240),
                 (8192, 2048), (1, 131072), (65536, 512)],
    "sum": [(256, 8192), (2048, 3072), (2048, 2049), (4096, 12288),
            (1, 262144), (65536, 512)],
    "long_sum": [(1, 49152), (2, 131072), (4, 196608), (8, 262144), (64, 131072)],
    "layer_norm": [(256, 4096), (2048, 2560), (2048, 1025), (4096, 10240),
                   (8192, 2048), (1, 131072), (32768, 512)],
    "cross_entropy": [(4096, 8192), (8192, 16384), (2048, 49152), (4096, 98304),
                      (8192, 49152), (16384, 16384)],
    "softmax": [(4096, 640), (4096, 3072), (4096, 1025), (8192, 8192),
                (16384, 512), (1, 131072)],
    "kl_div": [(4096, 24576), (4096, 49152), (8192, 16384), (2048, 98304),
               (4096, 262144)],
    "jsd": [(8192, 24576), (8192, 49152), (4096, 16384), (2048, 98304),
            (8192, 262144)],
    "welford": [(262144, 5120), (262144, 7168), (262144, 1280), (262144, 1543),
                (131072, 2048), (262144, 768)],
}


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else None


def _build_with_config(fn, cfg_dict, args):
    k = helion.kernel(fn.fn, configs=[helion.Config(**cfg_dict)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b


def measure_all(kernel, m, n):
    """Measure C1(default), C4(seed), C7(tc-default) on the SAME inputs."""
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(m, n)

    # --- C4: Helion SEED, no autotune (configs=[seed]) ---
    seed, _ = get_seed(fn, args)
    bound_s = _build_with_config(fn, dict(seed), args)
    codegen = codegen_kind(bound_s)
    ok_s, err_s = check_correct(out_extract(bound_s(*args)), ref)
    seed_lat = median_do_bench(lambda: bound_s(*args)) if ok_s else None

    # --- C1: Helion DEFAULT config, no autotune ---
    cfg_d = dict(fn.bind(args).config_spec.default_config())
    bound_d = _build_with_config(fn, cfg_d, args)
    ok_d, err_d = check_correct(out_extract(bound_d(*args)), ref)
    default_lat = median_do_bench(lambda: bound_d(*args)) if ok_d else None

    # --- C7: torch.compile(reference) DEFAULT mode ---
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    ok_tc, err_tc = check_correct(out_extract(tc(args)), ref)
    tc_lat = median_do_bench(lambda: tc(args)) if ok_tc else None

    return {
        "shape": [m, n],
        "default_lat_us": (default_lat * 1000) if default_lat else None,
        "seed_lat_us": (seed_lat * 1000) if seed_lat else None,
        "tc_lat_us": (tc_lat * 1000) if tc_lat else None,
        "g_seed": (tc_lat / seed_lat) if (seed_lat and tc_lat) else None,
        "g_default": (tc_lat / default_lat) if (default_lat and tc_lat) else None,
        "seed_codegen": codegen,
        "seed_cfg": dict(seed),
        "seed_correct": ok_s, "seed_err": err_s,
        "default_correct": ok_d, "default_err": err_d,
        "tc_correct": ok_tc,
    }


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save(path, blob):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(blob, f, indent=2, default=str)
    os.replace(tmp, path)


def run_kernel(kernel, out_dir, gpu):
    path = os.path.join(out_dir, f"run1_{kernel}.json")
    blob = _load(path) or {"kernel": kernel, "gpu": gpu, "rows": []}
    done = {tuple(r["shape"]) for r in blob["rows"]}
    for (m, n) in TEST_SHAPES[kernel]:
        if (m, n) in done:
            print(f"[skip] {kernel}({m},{n}) already done", flush=True)
            continue
        tag = f"{kernel}({m},{n})"
        t0 = time.time()
        try:
            r = measure_all(kernel, m, n)
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            oom = "out of memory" in str(e).lower() or "OutOfMemory" in msg
            if oom:
                torch.cuda.empty_cache()
            r = {"shape": [m, n], "error": msg[:300], "oom": oom}
            print(f"[{'OOM' if oom else 'ERR'}] {tag}: {msg[:140]}", flush=True)
            traceback.print_exc()
        else:
            gs = f"{r['g_seed']:.3f}" if r["g_seed"] is not None else "None"
            gd = f"{r['g_default']:.3f}" if r["g_default"] is not None else "None"
            print(
                f"[OK ] {tag:>22}  seed={_f(r['seed_lat_us'])}  "
                f"dflt={_f(r['default_lat_us'])}  tc={_f(r['tc_lat_us'])}  "
                f"G_seed={gs:>6} G_dflt={gd:>6}  codegen={r['seed_codegen']:>10}  "
                f"corr(s/d)={r['seed_correct']}/{r['default_correct']}  "
                f"({time.time() - t0:.0f}s)", flush=True)
        blob["rows"].append(r)
        _save(path, blob)  # checkpoint after EVERY shape
    # per-kernel geomeans (correct rows only)
    gss = [r.get("g_seed") for r in blob["rows"] if r.get("seed_correct")]
    gds = [r.get("g_default") for r in blob["rows"] if r.get("default_correct")]
    blob["geomean_g_seed"] = geomean(gss)
    blob["geomean_g_default"] = geomean(gds)
    _save(path, blob)
    print(f"=== {kernel}: geomean G_seed={blob['geomean_g_seed']} "
          f"G_default={blob['geomean_g_default']} ===", flush=True)


def _f(x):
    return f"{x:9.1f}" if x is not None else "     None"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default=",".join(TEST_SHAPES))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "logs", "run3"))
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out)
    os.makedirs(out_dir, exist_ok=True)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    kernels = [k.strip() for k in a.kernels.split(",") if k.strip()]
    bad = [k for k in kernels if k not in TEST_SHAPES]
    if bad:
        sys.exit(f"unknown kernels: {bad}")
    print(f"GPU={gpu} helion={helion.__file__}\nkernels={kernels}\nout={out_dir}\n",
          flush=True)
    for k in kernels:
        run_kernel(k, out_dir, gpu)


if __name__ == "__main__":
    main()
