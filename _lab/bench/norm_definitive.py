"""DEFINITIVE shape-by-shape rms_norm vs layer_norm, single-process, fp32, guarded.

Reconciles the two earlier harnesses. Benches the EXACT path run.py times — the
tritonbench wrapper functions `rms_norm` / `layer_norm` (autograd Functions whose
.forward calls rms_norm_fwd / layer_norm_fwd) — for three arms:

  helion_seed    : underlying fwd kernel pinned to the PR seed config
  helion_default : underlying fwd kernel pinned to base default_config()
  tc_default     : the operator's exact reference (LlamaRMSNorm / F.layer_norm),
                   torch.compile() default mode

We also time the BARE forward kernel (fwd-only) for both arms, to quantify the
autograd-wrapper overhead (the source of the run.py-vs-fwd-only discrepancy).

For each (kernel, shape): wrapper seed/default/tc us + fwd-only seed/default us,
G_seed=tc/wrapper_seed, seed_lift_wrapper, seed_lift_fwd. Contention-guarded.
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

import examples.rms_norm as RN
import examples.layer_norm as LN

N_RUNS = 15

SHAPES = sorted({
    (16384, 896), (8192, 1280), (16384, 1536), (4096, 2560), (2048, 4096),
    (2048, 6144), (2048, 7168), (2048, 10240), (4096, 2048), (4096, 3584),
})


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
    return s[len(s) // 2] * 1000.0  # us


def bench_one(kernel: str, m: int, n: int) -> dict:
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    if kernel == "rms_norm":
        fwd = RN.rms_norm_fwd            # underlying @helion.kernel
        wrapper = lambda: RN.rms_norm(x, w, eps=1e-5)   # autograd path run.py times
        fwd_args = (x, w, 1e-5)
        ref = lambda: (x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + 1e-5)).to(x.dtype) * w
        pick = lambda o: o[0]
    else:
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        fwd = LN.layer_norm_fwd
        wrapper = lambda: LN.layer_norm(x, [n], w, b, 1e-5)
        fwd_args = (x, [n], w, b, 1e-5)
        ref = lambda: torch.nn.functional.layer_norm(x, [n], w, b, 1e-5)
        pick = lambda o: o[0]

    # extract seed + base default from the fwd kernel
    bound = fwd.bind(fwd_args)
    default_cfg = bound.config_spec.default_config()
    seed_cfg = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]

    # bare fwd-only kernels (no autograd wrapper)
    k_def_fwd = helion.kernel(fwd.fn, config=default_cfg, static_shapes=True)
    k_seed_fwd = helion.kernel(fwd.fn, config=seed_cfg, static_shapes=True)
    ref_out = ref()
    acc_s = torch.allclose(pick(k_seed_fwd(*fwd_args)), ref_out, rtol=1e-3, atol=1e-3)
    acc_d = torch.allclose(pick(k_def_fwd(*fwd_args)), ref_out, rtol=1e-3, atol=1e-3)

    tc = torch.compile(ref); tc()

    # WRAPPER path: pin the module-global fwd kernel's config, call the wrapper.
    orig_configs = fwd.configs
    f0 = _foreign_mib()
    try:
        fwd.reset(); fwd.configs = [seed_cfg]
        wrapper()  # warm
        t_wrap_seed = _med(wrapper)
        fwd.reset(); fwd.configs = [default_cfg]
        wrapper()
        t_wrap_def = _med(wrapper)
    finally:
        fwd.reset(); fwd.configs = orig_configs
    t_tc = _med(tc)
    t_fwd_seed = _med(lambda: k_seed_fwd(*fwd_args))
    t_fwd_def = _med(lambda: k_def_fwd(*fwd_args))
    foreign = max(f0, _foreign_mib())

    return {
        "kernel": kernel, "shape": [m, n],
        "wrap_seed_us": round(t_wrap_seed, 2), "wrap_def_us": round(t_wrap_def, 2),
        "fwd_seed_us": round(t_fwd_seed, 2), "fwd_def_us": round(t_fwd_def, 2),
        "tc_us": round(t_tc, 2),
        "G_seed_wrap": round(t_tc / t_wrap_seed, 4),
        "G_seed_fwd": round(t_tc / t_fwd_seed, 4),
        "seed_lift_wrap": round(t_wrap_def / t_wrap_seed, 4),
        "seed_lift_fwd": round(t_fwd_def / t_fwd_seed, 4),
        "autograd_overhead_us": round(t_wrap_seed - t_fwd_seed, 2),
        "acc_seed": acc_s, "acc_default": acc_d, "foreign_mib": foreign,
        "seed_warps": dict(seed_cfg.config).get("num_warps"),
        "default_warps": dict(default_cfg.config).get("num_warps"),
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
