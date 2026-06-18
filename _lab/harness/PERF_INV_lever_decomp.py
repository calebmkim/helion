"""Perf-investigator: decompose the seed->oracle gap into seed-reachable vs
codegen-only levers.

The seed vocabulary = {block_sizes (M-block, R-block), reduction_loops,
num_warps, num_stages}. The oracle ALSO sets codegen-only knobs the seed cannot:
indexing, load_eviction_policies, pid_type, num_sm_multiplier, maxnreg, range_*.

We start from the SEED and walk toward the oracle ONE seed-reachable lever at a
time, re-benching each intermediate FULL config verbatim. The residual between
"all seed-reachable levers matched to oracle" and "the full verbatim oracle" is
the codegen-only ceiling the seed structurally cannot reach.

For each tiny-rnumel shape we report:
  seed_G
  +M-block (oracle's M)            -> G
  +num_warps (oracle's warps)      -> G
  +num_stages (oracle's stages)    -> G
  +reduction_loops (oracle's)      -> G   [== all seed-reachable levers matched]
  full oracle (re-run config)      -> G   [adds codegen-only knobs]
The "seed-reachable best" vs "full oracle" delta = the codegen residual.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

LONG = torch.int64

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def build_rms(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), 1e-5)


def build_x(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


# Oracle full configs captured from the effort=full run (logs/perf_inv_oracle.out).
ORACLES = {
    ("rms_norm", (32768, 256)): {
        "fn": rms_norm_fwd, "build": build_rms,
        "ref": lambda a: rms_norm_pytorch(*a),
        "tc": lambda: torch.compile(rms_norm_pytorch),
        "oracle": {'block_sizes': [8], 'reduction_loops': [128], 'num_warps': 16,
                   'num_stages': 2,
                   'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer',
                                'pointer', 'tensor_descriptor', 'tensor_descriptor'],
                   'load_eviction_policies': ['', 'last', 'last', '', 'last'],
                   'pid_type': 'flat'},
    },
    ("sum", (8192, 256)): {
        "fn": sum_kernel, "build": build_x,
        "ref": lambda a: torch.sum(a[0], -1),
        "tc": lambda: torch.compile(lambda x: torch.sum(x, -1)),
        "oracle": {'block_sizes': [32], 'reduction_loops': [32], 'num_warps': 8,
                   'num_stages': 3,
                   'indexing': ['pointer', 'pointer', 'tensor_descriptor'],
                   'load_eviction_policies': ['last', 'first'],
                   'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 2,
                   'maxnreg': 128},
    },
    ("softmax_two_pass", (32768, 256)): {
        "fn": softmax_two_pass, "build": build_x,
        "ref": lambda a: torch.nn.functional.softmax(a[0], 1),
        "tc": lambda: torch.compile(lambda x: torch.nn.functional.softmax(x, 1)),
        "oracle": {'block_sizes': [2, 256], 'num_warps': 4, 'num_stages': 2,
                   'indexing': ['tensor_descriptor', 'tensor_descriptor',
                                'tensor_descriptor'],
                   'load_eviction_policies': ['', 'last'], 'pid_type': 'flat'},
    },
    ("cross_entropy", (4096, 4096)): {
        "fn": cross_entropy,
        "build": lambda s: (torch.randn(s[0], s[1], device="cuda", dtype=torch.float32),
                            torch.randint(0, s[1], (s[0],), device="cuda", dtype=LONG)),
        "ref": lambda a: torch.nn.functional.cross_entropy(*a),
        "tc": lambda: torch.compile(torch.nn.functional.cross_entropy),
        "oracle": {'block_sizes': [1], 'reduction_loops': [None], 'num_warps': 16,
                   'num_stages': 8,
                   'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'pointer',
                                'pointer', 'pointer'],
                   'load_eviction_policies': ['', '', 'first', 'first', 'last'],
                   'pid_type': 'flat'},
    },
}


def correct(out, ref):
    o = out[0] if isinstance(out, tuple) else out
    o = o.float(); r = ref.float()
    return bool(o.shape == r.shape and torch.allclose(o, r, rtol=1e-3, atol=1e-3))


def bench_cfg(fn, args, cfg, ref, tc_lat, label):
    try:
        k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args); b.ensure_config_exists(args)
        out = b(*args); ok = correct(out, ref)
        lat = med(lambda: b(*args))
        print(f"    {label:<40} G={tc_lat/lat:.3f}  ({lat*1000:.2f}us) ok={ok}")
        return tc_lat / lat
    except Exception as exc:
        print(f"    {label:<40} ERR {type(exc).__name__}: {str(exc)[:50]}")
        return None


def main():
    import os
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  helion={helion.__file__}\n")
    for (name, shape), spec in ORACLES.items():
        fn = spec["fn"]; args = spec["build"](shape); ref = spec["ref"](args)
        oracle = spec["oracle"]
        torch._dynamo.reset()
        tc = spec["tc"](); _ = tc(*args)
        tc_lat = med(lambda: tc(*args))
        bound0 = fn.bind(args)
        seed = dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])
        print(f"=== {name} {shape} (rnumel={shape[1]}) tc={tc_lat*1000:.2f}us ===")
        print(f"    seed={seed}")
        print(f"    oracle(seed-reachable subset)={ { k: oracle.get(k) for k in ['block_sizes','reduction_loops','num_warps','num_stages'] } }")
        # 0: seed
        bench_cfg(fn, args, seed, ref, tc_lat, "seed (baseline)")
        # progressive: match oracle's seed-reachable levers cumulatively
        cfg = dict(seed)
        cfg["block_sizes"] = list(oracle["block_sizes"])
        bench_cfg(fn, args, dict(cfg), ref, tc_lat, "+ oracle block_sizes (M-block)")
        if "reduction_loops" in oracle:
            cfg["reduction_loops"] = list(oracle["reduction_loops"])
            bench_cfg(fn, args, dict(cfg), ref, tc_lat, "+ oracle reduction_loops")
        cfg["num_warps"] = oracle["num_warps"]
        bench_cfg(fn, args, dict(cfg), ref, tc_lat, "+ oracle num_warps")
        cfg["num_stages"] = oracle["num_stages"]
        sr_g = bench_cfg(fn, args, dict(cfg), ref, tc_lat, "+ oracle num_stages [=ALL seed-reachable]")
        # full verbatim oracle (adds codegen-only knobs)
        full_g = bench_cfg(fn, args, dict(oracle), ref, tc_lat, "FULL verbatim oracle (+codegen knobs)")
        if sr_g and full_g:
            print(f"    --> codegen-only residual: {(full_g/sr_g-1)*100:+.1f}%  "
                  f"(seed-reachable best G={sr_g:.3f} vs full oracle G={full_g:.3f})")
        print()


if __name__ == "__main__":
    main()
