"""AUDITOR: held-out + no-regression + out-of-focus, with the DECISIVE 3-way:
  G_seed   = tc/v3_seed
  G_default= tc/default_config
  G_p32    = tc/simple_persistent_w32   (the auditor's simple alternative)

If v3_seed ~= simple persistent/w32 everywhere it helps, the num_load /
rnumel>16384 BRANCHES do not earn their place over "always persistent, w32 when
rnumel large". We also check rms_norm/sum no-regression and one out-of-focus
kernel (layer_norm_fwd, num_load>=2) for help/not-hurt.

One shape per (kernel) reported; correctness max_abs vs reference + tol.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch._dynamo

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.sum import sum_kernel  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7
EPS = 1e-5


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def run_cfg(kfn, args, ref, cfg):
    k = helion.kernel(kfn.fn, configs=[cfg])
    b = k.bind(args)
    b.ensure_config_exists(args)
    out = b(*args)
    out = out[0] if isinstance(out, tuple) else out
    ok = torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3)
    maxabs = float((out.float() - ref.float()).abs().max())
    t = med(lambda: b(*args)) * 1000
    return t, ok, maxabs


def measure(label, kfn, args, ref, tc_fn, seed_args):
    seed = dict(compiler_seed_configs(kfn.bind(seed_args).env,
                                      kfn.bind(seed_args).host_function.device_ir)[0])
    nl = kfn.bind(seed_args).env.config_spec.reduction_facts[0].num_load
    rn = kfn.bind(seed_args).env.config_spec.reduction_facts[0].size_hint
    torch._dynamo.reset()
    tc = torch.compile(tc_fn)
    _ = tc(*args)
    t_tc = med(lambda: tc(*args)) * 1000
    t_s, ok_s, ma_s = run_cfg(kfn, args, ref, helion.Config(**seed))
    cfg_d = kfn.bind(seed_args).config_spec.default_config()
    t_d, ok_d, _ = run_cfg(kfn, args, ref, helion.Config(**dict(cfg_d)))
    t_p, ok_p, _ = run_cfg(kfn, args, ref, helion.Config(
        block_sizes=[seed["block_sizes"][0]], reduction_loops=[None],
        num_warps=32, num_stages=1))
    print(f"{label:>26} nl={nl} rn={rn:>7} w{seed['num_warps']:<2} | "
          f"Gseed={t_tc/t_s:.3f} Gdef={t_tc/t_d:.3f} Gp32={t_tc/t_p:.3f} | "
          f"seed/p32={t_s/t_p:.3f} | ok={ok_s} maxabs={ma_s:.1e}")


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  N_RUNS={N_RUNS}\n")
    print("Gseed/Gdef/Gp32 = tc/seed, tc/default, tc/simple-persistent-w32; "
          "seed/p32~1 => seed==simple alt\n")

    # --- no-regression: rms_norm in-sample ---
    for m, n in [(2048, 16384), (8192, 8192), (32768, 256)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        args = (x, w, EPS)
        ref = rms_norm_pytorch(x, w, EPS)
        measure(f"rms_norm{(m,n)}", rms_norm_fwd, args, ref, rms_norm_pytorch, args)

    # --- no-regression: sum in-sample (incl max-rnumel row) ---
    for m, n in [(2048, 16384), (8192, 4096)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        measure(f"sum{(m,n)}", sum_kernel, (x,), x.sum(-1),
                lambda t: t.sum(-1), (x,))

    # --- held-out sum (new shapes) ---
    for m, n in [(4096, 32768), (1, 65536)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        measure(f"sum*HELD{(m,n)}", sum_kernel, (x,), x.sum(-1),
                lambda t: t.sum(-1), (x,))

    # --- held-out long_sum (new shapes) ---
    for m, n in [(32, 131072), (4, 524288)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        measure(f"long_sum*HELD{(m,n)}", longsum, (x,), x.sum(-1),
                lambda t: t.sum(-1), (x,))

    # --- out-of-focus: layer_norm_fwd (num_load>=2). help / not-hurt? ---
    def ln_ref(x, w, b, eps):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, eps)
    for m, n in [(2048, 16384), (1, 131072)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        try:
            args = (x, [n], w, b, EPS)
            ref = ln_ref(x, w, b, EPS)
            measure(f"layer_norm*OOF{(m,n)}", layer_norm_fwd, args, ref,
                    lambda xx, ns, ww, bb, ee: ln_ref(xx, ww, bb, ee), args)
        except Exception as e:  # noqa: BLE001
            print(f"{'layer_norm'+str((m,n)):>26} -> {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
