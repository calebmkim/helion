"""Re-cert spot-confirm of two headline G's vs the ledger, with the certified
harness (triton.testing.do_bench median-of-7, fp32):

  - G_rms_norm at (4096,8192): ledger reference G_default ~0.77, per-shape G_seed
    ~0.99 (rms_norm_fwd kernel G_seed ~0.98).
  - G_cross_entropy at (8192,65536): ledger cross_entropy_per_shape g_seed=1.0
    (looped / w32), G_default ~0.6 band.

Same seed extraction + correctness gate the canonical measure_g_*.py harnesses use.
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WT), helion.__file__
sys.path.insert(0, WT)

from triton.testing import do_bench  # noqa: E402

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402
from examples.cross_entropy import cross_entropy  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7
EPS = 1e-5


def med(fn):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2]


def seed_of(fn, args):
    b = fn.bind(args)
    seeds = compiler_seed_configs(b.env, b.host_function.device_ir)
    assert len(seeds) == 1
    return seeds[0]


def bound_cfg(fn, args, cfg):
    k = helion.kernel(fn.fn, configs=[helion.Config(**dict(cfg))])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b


def maxabs(out, ref):
    out = out[0] if isinstance(out, tuple) else out
    return float((out.float() - ref.float()).abs().max())


def measure(name, fn, args, ref):
    seed = seed_of(fn, args)
    bs = bound_cfg(fn, args, seed)
    err_s = maxabs(bs(*args), ref)
    looped = bool(dict(bs._config).get("reduction_loops", [None])[0])
    seed_lat = med(lambda: bs(*args))

    cfg_d = helion.kernel(fn.fn).bind(args).config_spec.default_config()
    bd = bound_cfg(fn, args, cfg_d)
    err_d = maxabs(bd(*args), ref)
    default_lat = med(lambda: bd(*args))

    torch._dynamo.reset()
    if name.startswith("rms"):
        tc = torch.compile(rms_norm_pytorch)
        tc_call = lambda: tc(*args)
    else:
        tc = torch.compile(lambda lg, lb: torch.nn.functional.cross_entropy(lg, lb))
        tc_call = lambda: tc(*args)
    tc_call()
    tc_lat = med(tc_call)

    print(f"\n=== {name} ===")
    print(f"  seed: codegen={'looped' if looped else 'persistent'} "
          f"warps={dict(seed)['num_warps']} err={err_s:.2e}")
    print(f"  seed_lat   = {seed_lat*1000:8.1f} us")
    print(f"  default_lat= {default_lat*1000:8.1f} us  (err {err_d:.2e})")
    print(f"  tc_lat     = {tc_lat*1000:8.1f} us")
    print(f"  G_seed    (tc/seed)    = {tc_lat/seed_lat:.3f}")
    print(f"  G_default (tc/default) = {tc_lat/default_lat:.3f}")
    return tc_lat / seed_lat, tc_lat / default_lat


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} helion={helion.__file__}")

    # rms_norm (4096,8192)
    m, n = 4096, 8192
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    ref = rms_norm_pytorch(x, w, EPS)
    measure("rms_norm (4096,8192)", rms_norm_fwd, (x, w, EPS), ref)

    # cross_entropy (8192,65536)
    N, V = 8192, 65536
    logits = torch.randn(N, V, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, V, (N,), device="cuda", dtype=torch.int64)
    ref_ce = torch.nn.functional.cross_entropy(logits, labels)
    measure("cross_entropy (8192,65536)", cross_entropy, (logits, labels), ref_ce)


if __name__ == "__main__":
    main()
