"""TERMINAL TEST read-once harness (ledger-keeper). Measures bare-seed G =
tc_default_lat / seed_lat (do_bench median-of-7, fp32) on a PRISTINE TEST set,
DISJOINT from BOTH in-sample AND validation. Reuses the EXACT fair methodology of
measure_g_validation.py (and adds welford Band-C support).

Each kernel: seed fires (exactly 1 seed, welford with is_structured_combine), seed
USED (codegen persistent-vs-looped), CORRECT vs the kernel's reference (fp32 tol;
welford non-pow2/prime especially). Per-kernel TEST geomean G.

DISJOINTNESS is asserted at construction time against the recorded in-sample +
validation sets. softmax overridden to fp32 (+ assert) per SETUP.md.

Read-once: NOT for heuristic edits. Emits JSON + table.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
N_RUNS = 7
LONG = torch.int64


def median_do_bench(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def geomean(xs):
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


# ============================================================ FIREWALL SETS
# IN-SAMPLE (ledger kernels.<k>.in_sample_shapes) + VALIDATION (the shapes the
# validation sweep already read: measure_g_validation.py + welford_decline_validation.py,
# expressed at their ACTUAL measured M where shrunk). TEST must be disjoint from both.
IN_SAMPLE = {
    "rms_norm": {(2048, 1024), (2048, 2048), (2048, 4096), (2048, 8192), (2048, 16384),
                 (4096, 1536), (4096, 3584), (4096, 5120), (4096, 7168),
                 (8192, 4096), (8192, 8192), (32768, 256), (32768, 1024)},
    "sum": {(2048, 1024), (2048, 4096), (2048, 16384), (4096, 1536), (4096, 5120),
            (8192, 256), (8192, 4096), (32768, 256), (32768, 1024)},
    "long_sum": {(1, 32768), (2, 65536), (4, 130000), (8, 131072), (16, 262144)},
    "layer_norm": {(4096, 1024), (4096, 2048), (4096, 4096), (4096, 8192), (4096, 12288),
                   (4096, 15872), (2048, 3584), (2048, 8192), (8192, 4096), (8192, 5120),
                   (8192, 7168)},
    "cross_entropy": {(4096, 4096), (4096, 16384), (8192, 32768), (16384, 32768),
                      (8192, 65536), (16384, 65536), (8192, 131072)},
    "softmax": {(4096, 256), (4096, 512), (4096, 1024), (4096, 2048), (4096, 4096),
                (4096, 8192), (4096, 12288), (4096, 16384), (32768, 256), (32768, 1024)},
    "kl_div": {(4096, 4096), (4096, 8192), (4096, 16384), (4096, 32768), (4096, 65536),
               (4096, 131072)},
    "jsd": {(8192, 4096), (8192, 8192), (8192, 16384), (8192, 32768), (8192, 65536),
            (8192, 131072)},
    "welford": {(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096)},
}
VALIDATION = {
    "rms_norm": {(16, 4096), (128, 4096), (2048, 1023), (2048, 2047), (2048, 3072),
                 (2048, 6144), (4096, 12288), (1024, 32768), (8192, 256)},
    "sum": {(1, 4096), (16, 4096), (2048, 1023), (2048, 2047), (4096, 6144),
            (1024, 32768), (512, 65536), (8192, 256)},
    "long_sum": {(1, 100000), (1, 1048576), (4, 262143), (32, 65536)},
    "layer_norm": {(16, 4096), (128, 4096), (2048, 1023), (2048, 1536), (2048, 2047),
                   (4096, 6144), (1024, 32768), (1024, 36864), (8192, 36864), (8192, 256)},
    "softmax": {(16, 4096), (128, 4096), (2048, 1023), (2048, 2047), (2048, 32768),
                (512, 65536), (128, 131072), (8192, 256)},
    "cross_entropy": {(2048, 32000), (4096, 32000), (8192, 128000), (2048, 128256),
                      (4096, 129280), (2048, 151936), (1024, 256000)},
    "kl_div": {(4096, 32000), (8192, 65536), (2048, 128256), (1024, 256000)},
    "jsd": {(4096, 128256), (4096, 129280), (2048, 151936), (1024, 256000)},
    "welford": {(262144, 2560), (262144, 3072), (65536, 16384)},
}


# ============================================================ TEST SHAPES
# ~5-8 NOVEL shapes/kernel spanning regimes: small-N, medium, wide, tiny-M-huge-N,
# non-pow2 N, grid-bound large-M small-N. Prefer TritonBench-native where disjoint.
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
    # welford: native (5120,7168) + non-pow2 canaries (1280,768) + PRIME canary (1543) + (131072,2048)
    "welford": [(262144, 5120), (262144, 7168), (262144, 1280), (262144, 1543),
                (131072, 2048), (262144, 768)],
}


def assert_disjoint():
    bad = []
    for k, shapes in TEST_SHAPES.items():
        for s in shapes:
            if tuple(s) in IN_SAMPLE.get(k, set()):
                bad.append(f"{k}{s} IN in_sample")
            if tuple(s) in VALIDATION.get(k, set()):
                bad.append(f"{k}{s} IN validation")
    if bad:
        raise AssertionError("TEST not disjoint:\n  " + "\n  ".join(bad))
    print("DISJOINTNESS CHECK PASS: all TEST shapes disjoint from in-sample + "
          "validation across 9 kernels.\n", flush=True)


# ============================================================ kernel specs
def _allclose(o, r, rtol, atol):
    o = o.to(torch.float32); r = r.to(torch.float32)
    return bool(torch.allclose(o, r, rtol=rtol, atol=atol)), float((o - r).abs().max())


def _relerr_scalar(o, r):
    e = abs(float(o) - float(r)) / (abs(float(r)) + 1e-12)
    return e < 1e-3, e


def _unwrap(o):
    return o[0] if isinstance(o, tuple) else o


def spec_rms_norm():
    from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch
    def args(s):
        m, n = s
        return (torch.randn(m, n, device="cuda", dtype=torch.float32),
                torch.randn(n, device="cuda", dtype=torch.float32), EPS)
    return dict(fn=rms_norm_fwd, args=args, ref=lambda a: rms_norm_pytorch(*a),
                out=_unwrap, correct=lambda o, r: _allclose(o, r, 1e-3, 1e-4), kind="T1")


def spec_layer_norm():
    from examples.layer_norm import layer_norm_fwd
    def args(s):
        m, n = s
        return (torch.randn(m, n, device="cuda", dtype=torch.float32), [n],
                torch.randn(n, device="cuda", dtype=torch.float32),
                torch.randn(n, device="cuda", dtype=torch.float32), EPS)
    return dict(fn=layer_norm_fwd,
                args=args, ref=lambda a: torch.nn.functional.layer_norm(a[0], a[1], a[2], a[3], a[4]),
                out=_unwrap, correct=lambda o, r: _allclose(o, r, 1e-3, 1e-3), kind="T1")


def spec_softmax():
    from examples.softmax import softmax_two_pass
    def args(s):
        m, n = s
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        assert x.dtype == torch.float32  # SETUP: softmax fp32 override + assert
        return (x,)
    return dict(fn=softmax_two_pass, args=args,
                ref=lambda a: torch.nn.functional.softmax(a[0], dim=1),
                out=_unwrap, correct=lambda o, r: _allclose(o, r, 1e-3, 1e-4), kind="T2")


def spec_sum():
    from examples.sum import sum_kernel
    def args(s):
        m, n = s
        return (torch.randn(m, n, device="cuda", dtype=torch.float32),)
    return dict(fn=sum_kernel, args=args, ref=lambda a: torch.sum(a[0], dim=-1),
                out=_unwrap, correct=lambda o, r: _allclose(o, r, 1e-3, 1e-3), kind="T1")


def spec_long_sum():
    from examples.long_sum import longsum
    def args(s):
        m, n = s
        return (torch.randn(m, n, device="cuda", dtype=torch.float32),)
    return dict(fn=longsum, args=args, ref=lambda a: torch.sum(a[0], dim=-1),
                out=_unwrap, correct=lambda o, r: _allclose(o, r, 1e-3, 1e-3), kind="T1")


def spec_cross_entropy():
    from examples.cross_entropy import cross_entropy
    def args(s):
        n, v = s
        return (torch.randn(n, v, device="cuda", dtype=torch.float32),
                torch.randint(0, v, (n,), device="cuda", dtype=LONG))
    return dict(fn=cross_entropy, args=args,
                ref=lambda a: torch.nn.functional.cross_entropy(a[0], a[1]),
                out=_unwrap, correct=lambda o, r: _allclose(o, r, 1e-3, 1e-3), kind="T1")


def spec_kl_div():
    from examples.kl_div import kl_div_forward
    def args(s):
        bt, v = s
        yp = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
        yt = torch.randn(bt, v, device="cuda", dtype=torch.float32).softmax(-1)
        return (yp, yt, False, "batchmean", 1e-10)
    return dict(fn=kl_div_forward, args=args,
                ref=lambda a: torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to("cuda")(a[0], a[1]),
                out=lambda o: o[0] if isinstance(o, tuple) else o,
                correct=_relerr_scalar, kind="T2")


def spec_jsd():
    from examples.jsd import jsd_forward, TorchJSDBaseline
    baseline = TorchJSDBaseline(beta=0.5, ignore_index=-100)
    def args(s):
        bt, v = s
        return (torch.randn(bt, v, device="cuda").log_softmax(-1),
                torch.randn(bt, v, device="cuda").log_softmax(-1), None, 0.5, -100)
    return dict(fn=jsd_forward, args=args, ref=lambda a: baseline(a[0], a[1]),
                out=lambda o: o[0] if isinstance(o, tuple) else o,
                correct=_relerr_scalar, kind="T2")


def spec_welford():
    from examples.welford import welford, eager_layer_norm
    def args(s):
        m, n = s
        return (torch.rand(n, device="cuda", dtype=torch.float32),
                torch.rand(n, device="cuda", dtype=torch.float32),
                torch.rand(m, n, device="cuda", dtype=torch.float32), EPS)
    return dict(fn=welford, args=args, ref=lambda a: eager_layer_norm(*a),
                out=_unwrap, correct=lambda o, r: _allclose(o, r, 1e-3, 1e-3), kind="WELFORD")


SPECS = {
    "rms_norm": spec_rms_norm, "layer_norm": spec_layer_norm, "softmax": spec_softmax,
    "sum": spec_sum, "long_sum": spec_long_sum, "cross_entropy": spec_cross_entropy,
    "kl_div": spec_kl_div, "jsd": spec_jsd, "welford": spec_welford,
}


# ============================================================ seed-used
def t1_looped(tcode):
    return "for roffset" in tcode


def t2_persistent_used(tcode, n):
    consts = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"(_BLOCK_SIZE_\d+)\s*=\s*tl\.constexpr\((\d+)\)", tcode)}
    steps = []
    for m in re.finditer(r"tl\.range\(0,\s*(\d+),\s*(_BLOCK_SIZE_\d+)\)", tcode):
        extent = int(m.group(1)); step = consts.get(m.group(2))
        if step is not None and extent == n:
            steps.append(step)
    if not steps:
        return None
    return max(steps) >= n


def get_seed(fn, args, want_structured=False):
    bound = fn.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    if want_structured:
        sc = [f.is_structured_combine for f in bound.env.config_spec.reduction_facts]
        assert any(sc), f"welford expected is_structured_combine fact, got {sc}"
    return seeds[0]


def build(fn, cfg_dict, args):
    k = helion.kernel(fn.fn, configs=[helion.Config(**cfg_dict)])
    b = k.bind(args); b.ensure_config_exists(args)
    return b


def measure_shape(spec, shape):
    fn = spec["fn"]; args = spec["args"](shape)
    ref = spec["ref"](args)
    n = shape[1]

    seed = dict(get_seed(fn, args, want_structured=(spec["kind"] == "WELFORD")))
    bound_s = build(fn, seed, args)
    tcode = bound_s.to_triton_code(helion.Config(**dict(bound_s._config)))
    cfg = dict(bound_s._config)
    if spec["kind"] == "T1":
        want_looped = bool(cfg.get("reduction_loops", [None])[0])
        got_looped = t1_looped(tcode)
        seed_used = (want_looped == got_looped)
        codegen = "looped" if want_looped else "persistent"
    elif spec["kind"] == "T2":
        used = t2_persistent_used(tcode, n)
        seed_used = used is not None
        codegen = "persistent" if used is True else ("looped" if used is False else "?")
    else:  # WELFORD: combine tile (block 1) + apply tile (block 2). seed-used =
        # combine for-loop present with the seed's combine block size.
        nloops = tcode.count("for offset")
        bs = cfg.get("block_sizes")
        seed_used = nloops >= 1 and bs is not None
        codegen = f"bs={bs}/loops={nloops}"
    raw_s = bound_s(*args)
    out_s = spec["out"](raw_s)
    ok_s, err_s = spec["correct"](out_s, ref)
    seed_lat = median_do_bench(lambda: bound_s(*args))

    bound_d0 = fn.bind(args)
    cfg_d = dict(bound_d0.config_spec.default_config())
    bound_d = build(fn, cfg_d, args)
    raw_d = bound_d(*args)
    ok_d, _ = spec["correct"](spec["out"](raw_d), ref)
    default_lat = median_do_bench(lambda: bound_d(*args))

    torch._dynamo.reset()
    reffn = spec["ref"]
    tc = torch.compile(lambda *a: reffn(a))
    out_tc = tc(*args)
    tc_lat = median_do_bench(lambda: tc(*args))

    return {
        "shape": list(shape), "seed_warps": seed.get("num_warps"),
        "seed_block_sizes": seed.get("block_sizes"),
        "seed_reduction_loops": seed.get("reduction_loops"),
        "codegen": codegen, "seed_used": seed_used,
        "seed_correct": ok_s, "seed_err": err_s, "default_correct": ok_d,
        "seed_lat_us": seed_lat * 1000, "default_lat_us": default_lat * 1000,
        "tc_lat_us": tc_lat * 1000,
        "g_seed": tc_lat / seed_lat, "g_default": tc_lat / default_lat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=list(SPECS))
    a = ap.parse_args()
    assert_disjoint()
    spec = SPECS[a.kernel]()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    shapes = TEST_SHAPES[a.kernel]
    print(f"GPU={gpu} helion={helion.__file__} kernel={a.kernel} "
          f"kind={spec['kind']} (TEST read-once fp32)\n", flush=True)
    header = (f"{'shape':>16} {'codegen':>20} {'w':>3} {'seed_us':>10} "
              f"{'dflt_us':>10} {'tc_us':>10} {'G_seed':>7} {'G_dflt':>7} "
              f"{'used':>5} {'corr':>5} {'err':>9}")
    print(header); print("-" * len(header), flush=True)
    rows, gss, gds = [], [], []
    for shape in shapes:
        try:
            r = measure_shape(spec, tuple(shape))
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"{str(tuple(shape)):>16}  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            rows.append({"shape": list(shape), "error": f"{type(e).__name__}: {e}"})
            continue
        rows.append(r)
        gss.append(r["g_seed"]); gds.append(r["g_default"])
        print(f"{str(tuple(r['shape'])):>16} {str(r['codegen'])[:20]:>20} "
              f"{str(r['seed_warps']):>3} {r['seed_lat_us']:>10.1f} "
              f"{r['default_lat_us']:>10.1f} {r['tc_lat_us']:>10.1f} "
              f"{r['g_seed']:>7.3f} {r['g_default']:>7.3f} "
              f"{str(r['seed_used']):>5} {str(r['seed_correct']):>5} "
              f"{r['seed_err']:>9.1e}", flush=True)
    print("-" * len(header))
    summary = {}
    if gss:
        summary = {"geomean_g_seed": geomean(gss),
                   "geomean_g_default": geomean(gds), "n_shapes": len(gss)}
        all_used = all(r.get("seed_used") for r in rows if "error" not in r)
        all_corr = all(r.get("seed_correct") for r in rows if "error" not in r)
        print(f"GEOMEAN  G_seed={summary['geomean_g_seed']:.4f}  "
              f"G_default={summary['geomean_g_default']:.4f}  (n={summary['n_shapes']})  "
              f"all_used={all_used} all_correct={all_corr}", flush=True)
        summary["all_seed_used"] = all_used
        summary["all_seed_correct"] = all_corr
    blob = {"kernel": a.kernel, "kind": spec["kind"], "gpu": gpu,
            "rows": rows, "summary": summary}
    print("\n===JSON_BEGIN===")
    print(json.dumps(blob))
    print("===JSON_END===", flush=True)


if __name__ == "__main__":
    main()
