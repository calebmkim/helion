"""Why does layer_norm (2048,1025) want M_BLOCK=4, num_warps=4?

Sweep M_BLOCK x num_warps on three neighbors:
    (2048,1024)  exact pow2 reduction tile (rdim=1024)
    (2048,1025)  just-over-pow2 -> rdim rounds to 2048 (TEST shape, read-only)
    (2048,2048)  exact pow2 reduction tile (rdim=2048)

Hypothesis: (1025) behaves like (2048) at the hardware level because both load a
2048-wide reduction tile; (1024) loads half that. If true, (1025) and (2048)
should share an optimum and (1024) should differ.

persistent single-pass (reduction_loops=[None]), num_stages=1, pid_type=flat,
matching the seed family. fp32. do_bench median-of-7. PYTHONPATH-only wiring.
"""

from __future__ import annotations

import os

import torch

import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), helion.__file__

from triton.testing import do_bench  # noqa: E402

from examples.layer_norm import layer_norm_fwd  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7
EPS = 1e-5
SHAPES = [(2048, 1024), (2048, 1025), (2048, 2048)]
MBLOCKS = [1, 2, 4, 8]
WARPS = [2, 4, 8, 16]


def med(fn):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[N_RUNS // 2]


def build(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    args = (x, [n], w, b, EPS)
    ref = torch.nn.functional.layer_norm(x, [n], w, b, EPS)
    return args, ref


def run_cfg(args, ref, cfg):
    k = helion.kernel(layer_norm_fwd.fn, configs=[helion.Config(**cfg)])
    bound = k.bind(args)
    bound.ensure_config_exists(args)
    out = bound(*args)
    o = (out[0] if isinstance(out, tuple) else out).float()
    ok = bool(torch.allclose(o, ref, rtol=1e-3, atol=1e-4))
    if not ok:
        return None, False
    return med(lambda: bound(*args)) * 1000.0, True


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}\n", flush=True)

    for (m, n) in SHAPES:
        args, ref = build(m, n)
        rdim = next_pow2(n)
        waste = 100.0 * (1 - n / rdim)
        print(f"==== layer_norm ({m},{n})  rdim(next_pow2)={rdim}  "
              f"masked-waste={waste:.0f}%  ====", flush=True)

        # live seed for reference
        bound0 = layer_norm_fwd.bind(args)
        seed = dict(compiler_seed_configs(bound0.env,
                                          bound0.host_function.device_ir)[0])
        print(f"  seed: block_sizes={seed.get('block_sizes')} "
              f"num_warps={seed.get('num_warps')} "
              f"num_stages={seed.get('num_stages')} "
              f"reduction_loops={seed.get('reduction_loops')} "
              f"pid_type={seed.get('pid_type')}", flush=True)

        # torch.compile default reference
        torch._dynamo.reset()
        tc = torch.compile(lambda x: torch.nn.functional.layer_norm(
            x, [n], args[2], args[3], EPS))
        tc_us = med(lambda: tc(args[0])) * 1000.0
        print(f"  tc-default: {tc_us:.1f} us\n", flush=True)

        header = "  M_BLK \\ warps |" + "".join(f"{w:>9}" for w in WARPS) \
                 + "    grid(M/blk)"
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)
        best = (None, None, 1e9)
        for mb in MBLOCKS:
            cells = []
            for w in WARPS:
                cfg = dict(block_sizes=[mb], reduction_loops=[None],
                           num_warps=w, num_stages=1, pid_type="flat")
                try:
                    lat, ok = run_cfg(args, ref, cfg)
                except Exception as e:  # noqa: BLE001
                    cells.append("  ERR")
                    continue
                if not ok or lat is None:
                    cells.append("  BAD")
                    continue
                cells.append(f"{lat:>9.1f}")
                if lat < best[2]:
                    best = (mb, w, lat)
            grid = (m + mb - 1) // mb
            print(f"  M={mb:<11} |" + "".join(cells) + f"     {grid}",
                  flush=True)
        print(f"\n  >>> BEST: M_BLOCK={best[0]} num_warps={best[1]} "
              f"-> {best[2]:.1f} us   (tc-default {tc_us:.1f} us, "
              f"ratio tc/best={tc_us/best[2]:.2f})\n", flush=True)


if __name__ == "__main__":
    main()
