"""VALIDATION (out-of-sample) Product-A G measurement — generalization checkpoint.

READ-ONLY. Measures bare-seed G = tc_default_lat / seed_lat (do_bench median-of-7,
fp32) on the OUT-OF-SAMPLE (validation) shapes for each active kernel, reusing the
EXACT methodology of the per-kernel in-sample harnesses (measure_g_*.py):
  - bare seed via compiler_seed_configs -> configs=[seed] (no autotune)
  - default = un-seeded default_config baseline
  - tc = torch.compile default mode of the fp32 reference
  - seed-USED proof from codegen (T1: 'for roffset'; T2: tl.range step >= N)
  - correctness vs each kernel's reference (justified fp32 tol)

Emits a JSON blob to stdout (sentinel-delimited) + a human table. One kernel per
invocation (--kernel). Shapes come from list_of_kernels.md "Out-of-sample" sections
(M shrunk where noted to fit memory; N/V kept).

NOT for heuristic edits. Does NOT touch TEST shapes.
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
from helion._utils import next_power_of_2  # noqa: E402

EPS = 1e-5
N_RUNS = 7
LONG = torch.int64


def median_do_bench(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def geomean(xs):
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


# ---------------------------------------------------------------- kernel specs
# Each spec: fn, args(shape)->tuple, ref(args)->tensor/scalar, out(raw)->tensor,
# correct(out,ref)->(ok,err), kind ('T1'|'T2'), and the validation shapes.

def _allclose(o, r, rtol, atol):
    o = o.to(torch.float32); r = r.to(torch.float32)
    return bool(torch.allclose(o, r, rtol=rtol, atol=atol)), float((o - r).abs().max())


def _relerr_scalar(o, r):
    ok = abs(float(o) - float(r)) / (abs(float(r)) + 1e-12) < 1e-3
    return ok, abs(float(o) - float(r)) / (abs(float(r)) + 1e-12)


def _unwrap(o):
    """Row-output kernels (rms_norm/layer_norm) return (out, ...) tuples."""
    return o[0] if isinstance(o, tuple) else o


def spec_rms_norm():
    from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch
    def args(s):
        m, n = s
        return (torch.randn(m, n, device="cuda", dtype=torch.float32),
                torch.randn(n, device="cuda", dtype=torch.float32), EPS)
    def ref(a): return rms_norm_pytorch(*a)
    return dict(fn=rms_norm_fwd, args=args, ref=ref, out=_unwrap,
                correct=lambda o, r: _allclose(o, r, 1e-3, 1e-4), kind="T1",
                shapes=[(16, 4096), (128, 4096), (2048, 1023), (2048, 2047),
                        (2048, 3072), (2048, 6144), (4096, 12288), (1024, 32768),
                        (8192, 256), (8192, 256)],
                shape_labels=[None, None, None, None, None, None, None, None,
                              "262144,256->M=8192", "589824,256->M=8192"])


def spec_layer_norm():
    from examples.layer_norm import layer_norm_fwd
    def args(s):
        m, n = s
        return (torch.randn(m, n, device="cuda", dtype=torch.float32), [n],
                torch.randn(n, device="cuda", dtype=torch.float32),
                torch.randn(n, device="cuda", dtype=torch.float32), EPS)
    def ref(a): return torch.nn.functional.layer_norm(a[0], a[1], a[2], a[3], a[4])
    return dict(fn=layer_norm_fwd, args=args, ref=ref, out=_unwrap,
                correct=lambda o, r: _allclose(o, r, 1e-3, 1e-3), kind="T1",
                shapes=[(16, 4096), (128, 4096), (2048, 1023), (2048, 1536),
                        (2048, 2047), (4096, 6144), (1024, 32768), (1024, 36864),
                        (8192, 36864), (8192, 256)],
                shape_labels=[None, None, None, None, None, None, None, None,
                              "1152,36864->M=8192", "262144,256->M=8192"])


def spec_softmax():
    from examples.softmax import softmax_two_pass
    def args(s):
        m, n = s
        return (torch.randn(m, n, device="cuda", dtype=torch.float32),)
    def ref(a): return torch.nn.functional.softmax(a[0], dim=1)
    return dict(fn=softmax_two_pass, args=args, ref=ref, out=_unwrap,
                correct=lambda o, r: _allclose(o, r, 1e-3, 1e-4), kind="T2",
                shapes=[(16, 4096), (128, 4096), (2048, 1023), (2048, 2047),
                        (2048, 32768), (512, 65536), (128, 131072), (8192, 256)],
                shape_labels=[None, None, None, None, None, None, None,
                              "262144,256->M=8192"])


def spec_sum():
    from examples.sum import sum_kernel
    def args(s):
        m, n = s
        return (torch.randn(m, n, device="cuda", dtype=torch.float32),)
    def ref(a): return torch.sum(a[0], dim=-1)
    return dict(fn=sum_kernel, args=args, ref=ref, out=_unwrap,
                correct=lambda o, r: _allclose(o, r, 1e-3, 1e-3), kind="T1",
                shapes=[(1, 4096), (16, 4096), (2048, 1023), (2048, 2047),
                        (4096, 6144), (1024, 32768), (512, 65536), (8192, 256)],
                shape_labels=[None, None, None, None, None, None, None,
                              "262144,256->M=8192"])


def spec_long_sum():
    from examples.long_sum import longsum
    def args(s):
        m, n = s
        return (torch.randn(m, n, device="cuda", dtype=torch.float32),)
    def ref(a): return torch.sum(a[0], dim=-1)
    return dict(fn=longsum, args=args, ref=ref, out=_unwrap,
                correct=lambda o, r: _allclose(o, r, 1e-3, 1e-3), kind="T1",
                shapes=[(1, 100000), (1, 1048576), (4, 262143), (32, 65536)],
                shape_labels=[None, "4MiB-row->looped-tail", None, None])


def spec_cross_entropy():
    from examples.cross_entropy import cross_entropy
    def args(s):
        n, v = s
        return (torch.randn(n, v, device="cuda", dtype=torch.float32),
                torch.randint(0, v, (n,), device="cuda", dtype=LONG))
    def ref(a): return torch.nn.functional.cross_entropy(a[0], a[1])
    return dict(fn=cross_entropy, args=args, ref=ref, out=_unwrap,
                correct=lambda o, r: _allclose(o, r, 1e-3, 1e-3), kind="T1",
                shapes=[(2048, 32000), (4096, 32000), (8192, 128000),
                        (2048, 128256), (4096, 129280), (2048, 151936),
                        (1024, 256000)],
                shape_labels=[None] * 7)


def spec_kl_div():
    from examples.kl_div import kl_div_forward
    def args(s):
        bt, v = s
        yp = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
        yt = torch.randn(bt, v, device="cuda", dtype=torch.float32).softmax(-1)
        return (yp, yt, False, "batchmean", 1e-10)
    def ref(a):
        return torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(
            "cuda")(a[0], a[1])
    return dict(fn=kl_div_forward, args=args, ref=ref,
                out=lambda o: o[0] if isinstance(o, tuple) else o,
                correct=_relerr_scalar, kind="T2",
                shapes=[(4096, 32000), (8192, 65536), (2048, 128256),
                        (1024, 256000)],
                shape_labels=[None] * 4)


def spec_jsd():
    from examples.jsd import jsd_forward, TorchJSDBaseline
    baseline = TorchJSDBaseline(beta=0.5, ignore_index=-100)
    def args(s):
        bt, v = s
        return (torch.randn(bt, v, device="cuda").log_softmax(-1),
                torch.randn(bt, v, device="cuda").log_softmax(-1), None, 0.5, -100)
    def ref(a): return baseline(a[0], a[1])
    return dict(fn=jsd_forward, args=args, ref=ref,
                out=lambda o: o[0] if isinstance(o, tuple) else o,
                correct=_relerr_scalar, kind="T2",
                shapes=[(4096, 128256), (4096, 129280), (2048, 151936),
                        (1024, 256000)],
                shape_labels=[None] * 4)


SPECS = {
    "rms_norm": spec_rms_norm, "layer_norm": spec_layer_norm,
    "softmax": spec_softmax, "sum": spec_sum, "long_sum": spec_long_sum,
    "cross_entropy": spec_cross_entropy, "kl_div": spec_kl_div, "jsd": spec_jsd,
}


# ---------------------------------------------------------------- seed-used
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


def get_seed(fn, args):
    bound = fn.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    return seeds[0]


def build(fn, cfg_dict, args):
    k = helion.kernel(fn.fn, configs=[helion.Config(**cfg_dict)])
    b = k.bind(args); b.ensure_config_exists(args)
    return b


def measure_shape(spec, shape, label):
    fn = spec["fn"]; args = spec["args"](shape)
    ref = spec["ref"](args)
    n = shape[1]

    # --- seed ---
    seed = dict(get_seed(fn, args))
    bound_s = build(fn, seed, args)
    tcode = bound_s.to_triton_code(helion.Config(**dict(bound_s._config)))
    cfg = dict(bound_s._config)
    if spec["kind"] == "T1":
        want_looped = bool(cfg.get("reduction_loops", [None])[0])
        got_looped = t1_looped(tcode)
        seed_used = (want_looped == got_looped)
        codegen = "looped" if want_looped else "persistent"
        codegen_detail = f"want_looped={want_looped} got_looped={got_looped}"
    else:
        used = t2_persistent_used(tcode, n)
        # For T2 the seed may cap R_BLOCK (looped) for wide rows (Band-B / multi-load):
        # used True => persistent (R>=N); False => looped chunk; None => not located.
        seed_used = used is not None
        codegen = "persistent" if used is True else ("looped" if used is False else "?")
        codegen_detail = f"t2_persistent_used={used}"
    raw_s = bound_s(*args)
    out_s = spec["out"](raw_s)
    ok_s, err_s = spec["correct"](out_s, ref)
    seed_lat = median_do_bench(lambda: bound_s(*args))

    # --- default (un-seeded) ---
    bound_d0 = fn.bind(args)
    cfg_d = dict(bound_d0.config_spec.default_config())
    bound_d = build(fn, cfg_d, args)
    raw_d = bound_d(*args)
    ok_d, err_d = spec["correct"](spec["out"](raw_d), ref)
    default_lat = median_do_bench(lambda: bound_d(*args))

    # --- torch.compile default ---
    torch._dynamo.reset()
    reffn = spec["ref"]
    tc = torch.compile(lambda *a: reffn(a))
    out_tc = tc(*args)
    ok_tc, err_tc = spec["correct"](out_tc, ref)
    tc_lat = median_do_bench(lambda: tc(*args))

    return {
        "shape": list(shape), "label": label, "seed": seed,
        "warps": seed.get("num_warps"),
        "codegen": codegen, "codegen_detail": codegen_detail,
        "seed_used": seed_used, "seed_correct": ok_s, "seed_err": err_s,
        "default_correct": ok_d, "tc_correct": ok_tc,
        "seed_lat_us": seed_lat * 1000, "default_lat_us": default_lat * 1000,
        "tc_lat_us": tc_lat * 1000,
        "g_seed": tc_lat / seed_lat, "g_default": tc_lat / default_lat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=list(SPECS))
    ap.add_argument("--only", default=None, help="comma idx subset e.g. 0,1,2")
    a = ap.parse_args()
    spec = SPECS[a.kernel]()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    shapes = list(zip(spec["shapes"], spec["shape_labels"]))
    if a.only:
        idxs = [int(x) for x in a.only.split(",")]
        shapes = [shapes[i] for i in idxs]
    print(f"GPU={gpu} helion={helion.__file__} kernel={a.kernel} "
          f"kind={spec['kind']} (VALIDATION/out-of-sample fp32)\n")
    header = (f"{'shape':>16} {'label':>22} {'codegen':>10} {'w':>3} "
              f"{'seed_us':>10} {'dflt_us':>10} {'tc_us':>10} "
              f"{'G_seed':>7} {'G_dflt':>7} {'used':>5} {'corr':>5} {'err':>9}")
    print(header); print("-" * len(header))
    rows, gss, gds = [], [], []
    for shape, label in shapes:
        try:
            r = measure_shape(spec, tuple(shape), label)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"{str(tuple(shape)):>16} {(label or ''):>22}  ERROR: "
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
            rows.append({"shape": list(shape), "label": label,
                         "error": f"{type(e).__name__}: {e}"})
            continue
        rows.append(r)
        gss.append(r["g_seed"]); gds.append(r["g_default"])
        print(f"{str(tuple(r['shape'])):>16} {(r['label'] or ''):>22} "
              f"{r['codegen']:>10} {str(r['warps']):>3} "
              f"{r['seed_lat_us']:>10.1f} {r['default_lat_us']:>10.1f} "
              f"{r['tc_lat_us']:>10.1f} {r['g_seed']:>7.3f} {r['g_default']:>7.3f} "
              f"{str(r['seed_used']):>5} {str(r['seed_correct']):>5} "
              f"{r['seed_err']:>9.1e}")
    print("-" * len(header))
    summary = {}
    if gss:
        summary = {"geomean_g_seed": geomean(gss),
                   "geomean_g_default": geomean(gds), "n_shapes": len(gss)}
        print(f"GEOMEAN  G_seed={summary['geomean_g_seed']:.4f}  "
              f"G_default={summary['geomean_g_default']:.4f}  "
              f"(n={summary['n_shapes']})")
    blob = {"kernel": a.kernel, "kind": spec["kind"], "gpu": gpu,
            "rows": rows, "summary": summary}
    print("\n===JSON_BEGIN===")
    print(json.dumps(blob))
    print("===JSON_END===")


if __name__ == "__main__":
    main()
