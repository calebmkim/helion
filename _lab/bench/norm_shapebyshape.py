"""Shape-by-shape rms_norm vs layer_norm A/B, single process, fp32, contention-guarded.

Benches BOTH kernels on the SAME shape set (union of their test splits) so rms vs
layer_norm are directly comparable at identical (M,N). For each (kernel, shape):
  helion_seed, helion_default, tc_default  -> latencies (us) + G = tc/helion + seed_lift.

tc references match the tritonbench operators exactly:
  rms_norm   -> LlamaRMSNorm (x.float().pow2().mean rsqrt * weight)
  layer_norm -> F.layer_norm
Also reports tc_rms vs tc_ln at the same shape (to see if the two tc baselines differ).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import torch
from triton.testing import do_bench

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

from examples.rms_norm import rms_norm_fwd
from examples.layer_norm import layer_norm_fwd

N_RUNS = 15  # more repeats: these are small/noisy kernels


def _foreign_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10).stdout
    except Exception:  # noqa: BLE001
        return 0
    me = os.getpid(); m = 0
    for line in out.splitlines():
        p = [c.strip() for c in line.split(",")]
        if len(p) == 2 and p[0].isdigit() and int(p[0]) != me:
            m = max(m, int(p[1]) if p[1].isdigit() else 0)
    return m


def _med(fn) -> float:
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2]


# union of rms_norm + layer_norm test shapes (deduped, sorted)
SHAPES = sorted({
    (16384, 896), (8192, 1280), (16384, 1536), (4096, 2560), (2048, 4096),
    (2048, 6144), (2048, 7168), (2048, 10240),                # rms test
    (4096, 2048), (4096, 3584),                               # layer_norm test extras
})


def _llama_rms_ref(x, w, eps=1e-5):
    def f():
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + eps)).to(x.dtype) * w
    return f


def bench_one(kernel: str, m: int, n: int) -> dict:
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    if kernel == "rms_norm":
        args = (x, w, 1e-5)
        kfn = rms_norm_fwd
        ref = _llama_rms_ref(x, w)
        pick = lambda o: o[0]  # (out, inv_rms)
    else:
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        args = (x, [n], w, b, 1e-5)
        kfn = layer_norm_fwd
        ref = lambda: torch.nn.functional.layer_norm(x, [n], w, b, 1e-5)
        pick = lambda o: o[0]  # (out, mean, rstd)

    bound = kfn.bind(args)
    default_cfg = bound.config_spec.default_config()
    seed_cfg = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]
    k_def = helion.kernel(kfn.fn, config=default_cfg, static_shapes=True)
    k_seed = helion.kernel(kfn.fn, config=seed_cfg, static_shapes=True)

    ref_out = ref()
    acc_s = torch.allclose(pick(k_seed(*args)), ref_out, rtol=1e-3, atol=1e-3)
    acc_d = torch.allclose(pick(k_def(*args)), ref_out, rtol=1e-3, atol=1e-3)
    tc = torch.compile(ref); tc()

    f0 = _foreign_mib()
    t_s = _med(lambda: k_seed(*args)) * 1000  # us
    t_d = _med(lambda: k_def(*args)) * 1000
    t_tc = _med(lambda: tc()) * 1000
    foreign = max(f0, _foreign_mib())

    return {
        "kernel": kernel, "shape": [m, n],
        "seed_us": round(t_s, 2), "default_us": round(t_d, 2), "tc_us": round(t_tc, 2),
        "G_seed": round(t_tc / t_s, 4), "G_default": round(t_tc / t_d, 4),
        "seed_lift": round(t_d / t_s, 4),
        "acc_seed": acc_s, "acc_default": acc_d, "foreign_mib": foreign,
        "seed_warps": dict(seed_cfg.config).get("num_warps"),
        "default_warps": dict(default_cfg.config).get("num_warps"),
        "seed_rloop": dict(seed_cfg.config).get("reduction_loops"),
    }


def main() -> None:
    assert helion.__file__.startswith("/home/dev/local/helion-pr-edit"), helion.__file__
    rows = []
    for (m, n) in SHAPES:
        for kernel in ("rms_norm", "layer_norm"):
            r = bench_one(kernel, m, n)
            rows.append(r)
            print("ROW " + json.dumps(r), file=sys.stderr)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
