"""Decompose the long_sum win and test held-out shapes.

For each long_sum shape (in-sample AND held-out), time on the longsum kernel:
  - default_config (un-seeded baseline)
  - the heuristic SEED (what v2 ships)
  - persistent / warps32 (the alternative the fair sweep says is >= looped/32)
  - looped(16384) / warps32 (the seed recipe spelled out)
  - looped(16384) / warps4 (isolate: is it warps or loop that beats default?)
  - persistent / warps4 (isolate)
vs torch.compile default reference.

Shows whether the win is WARPS (generalizable: warps scale with rnumel) or the
LOOP FLIP (the grid-occupancy branch's stated mechanism), and whether the branch
helps/hurts on held-out (32,65536) [branch FIRES: M=32<64, 256KiB>=128KiB] and
the larger held-out rows.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch._dynamo

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.long_sum import longsum  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7

# in-sample + held-out long_sum shapes. (1,1048576) is huge; keep it.
SHAPES = [
    ("in", (1, 32768)), ("in", (2, 65536)), ("in", (4, 130000)),
    ("in", (8, 131072)), ("in", (16, 262144)),
    ("held", (1, 100000)), ("held", (4, 262143)), ("held", (32, 65536)),
    ("held", (1, 1048576)),
]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def time_cfg(x, ref, cfg):
    k = helion.kernel(longsum.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3), "correctness"
    return med(lambda: b(x)) * 1000


def get_seed(x):
    bound = longsum.bind((x,))
    return dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  helion={helion.__file__}  N={N_RUNS}\n")
    hdr = (f"{'tag':>5} {'shape':>14} {'seedcfg':>18} {'tc':>7} {'dflt':>7} "
           f"{'SEED':>7} {'P/32':>7} {'L/32':>7} {'L/4':>7} {'P/4':>7} "
           f"{'G_seed':>7} {'G_P32':>7}")
    print(hdr)
    print("-" * len(hdr))
    gseeds = []
    gp32s = []
    for tag, (m, n) in SHAPES:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        ref = x.sum(-1)
        seed = get_seed(x)
        seedcfg = f"rl={seed['reduction_loops']},w{seed['num_warps']}"
        bound_d = longsum.bind((x,))
        cfg_d = bound_d.config_spec.default_config()
        torch._dynamo.reset()

        def _sumref(t):
            return torch.sum(t, dim=-1)
        tc = torch.compile(_sumref)
        _ = tc(x)
        t_tc = med(lambda: tc(x))
        t_dflt = time_cfg(x, ref, cfg_d)
        t_seed = time_cfg(x, ref, helion.Config(**seed))
        t_p32 = time_cfg(x, ref, helion.Config(block_sizes=[1], reduction_loops=[None], num_warps=32, num_stages=1))
        t_l32 = time_cfg(x, ref, helion.Config(block_sizes=[1], reduction_loops=[16384], num_warps=32, num_stages=1))
        t_l4 = time_cfg(x, ref, helion.Config(block_sizes=[1], reduction_loops=[16384], num_warps=4, num_stages=1))
        t_p4 = time_cfg(x, ref, helion.Config(block_sizes=[1], reduction_loops=[None], num_warps=4, num_stages=1))
        g_seed = t_tc / t_seed
        g_p32 = t_tc / t_p32
        gseeds.append(g_seed)
        gp32s.append(g_p32)
        print(f"{tag:>5} {str((m,n)):>14} {seedcfg:>18} {t_tc:>7.1f} {t_dflt:>7.1f} "
              f"{t_seed:>7.1f} {t_p32:>7.1f} {t_l32:>7.1f} {t_l4:>7.1f} {t_p4:>7.1f} "
              f"{g_seed:>7.3f} {g_p32:>7.3f}")

    def geo(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs))
    print("-" * len(hdr))
    print(f"geomean(all): G_seed={geo(gseeds):.3f}  G_persistent32={geo(gp32s):.3f}")


if __name__ == "__main__":
    main()
