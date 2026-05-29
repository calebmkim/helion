"""cross_entropy: prove the seed is USED (codegen persistent + num_warps) and
CORRECT vs torch.nn.functional.cross_entropy (fp32). Bare-seed path (configs=[seed]).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.cross_entropy import cross_entropy  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

LONG = torch.int64


def ce_args(n, v):
    logits = torch.randn(n, v, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device="cuda", dtype=LONG)
    return (logits, labels)


def run(n, v):
    print(f"=== cross_entropy ({n},{v}) ===")
    args = ce_args(n, v)
    bound = cross_entropy.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, seeds
    seed = seeds[0]
    print(f"  seed: {dict(seed)}")
    # bare-seed run (configs=[seed] -> len==1 short-circuit, no autotune)
    kern = helion.kernel(cross_entropy.fn, configs=[seed])
    out = kern(*args)
    cfg = kern.bind(args)._config if hasattr(kern, "bind") else None
    code = kern.bind(args).to_triton_code(seed)
    has_roffset = "for roffset" in code
    n_arange = code.count("tl.arange")
    # find num_warps in launcher
    import re
    warps = re.findall(r"num_warps=(\d+)", code)
    print(f"  codegen: 'for roffset' present={has_roffset} (persistent => False) "
          f"tl.arange count={n_arange} num_warps_in_code={set(warps)}")
    # correctness vs F.cross_entropy
    ref = torch.nn.functional.cross_entropy(args[0], args[1])
    maxabs = (out - ref).abs().max().item()
    relerr = (maxabs / max(ref.abs().item(), 1e-12))
    ok = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
    print(f"  out={out.item():.6f} ref={ref.item():.6f} maxabs={maxabs:.3e} "
          f"relerr={relerr:.3e} allclose(rtol1e-3,atol1e-3)={ok}")
    print()


def main():
    print(f"helion={helion.__file__}\n")
    for (n, v) in [(4096, 4096), (4096, 16384), (8192, 32768), (16384, 65536),
                   (8192, 131072)]:
        run(n, v)


if __name__ == "__main__":
    main()
