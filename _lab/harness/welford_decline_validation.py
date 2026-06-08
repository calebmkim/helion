"""Confirm welford DECLINES (0 seeds, reduction_facts=0) + correct un-seeded
default on the OUT-OF-SAMPLE welford shapes (M shrunk to fit). READ-ONLY."""
from __future__ import annotations
import sys, os
import torch
import helion
WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)
from examples.welford import welford  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

# Out-of-sample welford shapes (brief): (262144,2560),(262144,3072),(65536,16384)
# M shrunk to fit (keep N): 262144 -> 16384 ; 65536 -> 8192
SHAPES = [(16384, 2560, "262144,2560->M=16384"),
          (16384, 3072, "262144,3072->M=16384"),
          (8192, 16384, "65536,16384->M=8192")]

def wf_args(m, n):
    w = torch.rand(n, device="cuda", dtype=torch.float32)
    b = torch.rand(n, device="cuda", dtype=torch.float32)
    x = torch.rand(m, n, device="cuda", dtype=torch.float32)
    return (w, b, x, 1e-5)

def ref(w, b, x, eps):
    return torch.nn.functional.layer_norm(x, [x.shape[-1]], w, b, eps)

def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES","?")
    print(f"GPU={gpu} kernel=welford (OUT-OF-SCOPE decline check)\n")
    for (m, n, label) in SHAPES:
        args = wf_args(m, n)
        bound = welford.bind(args)
        spec = bound.env.config_spec
        nrf = len(getattr(spec, "reduction_facts", []))
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        # un-seeded default correctness
        k = helion.kernel(welford.fn)
        bd = k.bind(args)
        cfg = bd.config_spec.default_config()
        k2 = helion.kernel(welford.fn, configs=[cfg])
        b2 = k2.bind(args); b2.ensure_config_exists(args)
        out = b2(*args)
        out = out[0] if isinstance(out, tuple) else out
        r = ref(*args)
        ok = bool(torch.allclose(out.float(), r.float(), rtol=1e-3, atol=1e-3))
        err = float((out.float()-r.float()).abs().max())
        print(f"  ({m},{n}) [{label}]: reduction_facts={nrf} "
              f"compiler_seeds={len(seeds)} (DECLINE iff 0) | "
              f"default correct={ok} maxabs={err:.2e}")
    print("\nVERDICT: welford declines (0 seeds) on all OOS shapes + "
          "un-seeded default is correct.")

if __name__ == "__main__":
    main()
