"""AUDITOR A6: is cross_entropy (8192,131072) G~0.54 a genuine KERNEL-SOURCE
gap (even the best CORRECT Helion config can't beat torch.compile), or config
headroom the worker skipped?

For the worst shape: (1) correctness of the v6 seed vs F.cross_entropy (honest
tol); (2) v6 seed latency; (3) a BROAD Helion-config sweep (chunks x warps x
stages, looped + persistent) -- the best CORRECT one is the Helion-max ceiling;
(4) torch.compile baseline. If best-correct-Helion >> tc -> source gap (honest).
Also a couple narrower shapes for correctness.
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

LONG = torch.int64


def med(fn, reps=7):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def ce_args(n, v, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, v, device="cuda", generator=g),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG, generator=g))


def run_cfg(args, cfg, ref):
    try:
        k = helion.kernel(cross_entropy.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        out = b(*args)
        err = (out.float() - ref.float()).abs().max().item()
        t = med(lambda: b(*args)) * 1000
        return t, err
    except Exception as e:  # noqa: BLE001
        return f"ERR:{type(e).__name__}", None


def main():
    print(f"helion={helion.__file__}\n")
    Fce = torch.nn.functional.cross_entropy

    for (n, v) in [(8192, 131072), (8192, 65536), (4096, 16384)]:
        args = ce_args(n, v, seed=0)
        logits, labels = args
        ref = Fce(logits, labels)
        # torch.compile baseline
        tc = torch.compile(Fce)
        tc(logits, labels)
        t_tc = med(lambda: tc(logits, labels)) * 1000
        t_eager = med(lambda: Fce(logits, labels)) * 1000
        print(f"### cross_entropy ({n},{v})  row={v*4//1024}KiB ###")
        print(f"  torch.compile={round(t_tc,1)}us  eager={round(t_eager,1)}us  ref={ref.item():.5f}")
        # v6 SEED (what the heuristic emits: looped 16384 w32 for >128KiB)
        v6cfg = ({"block_sizes": [1], "reduction_loops": [16384], "num_warps": 32,
                  "num_stages": 1} if v * 4 > 131072 else
                 {"block_sizes": [1], "reduction_loops": [None], "num_warps": 32,
                  "num_stages": 1})
        t6, e6 = run_cfg(args, v6cfg, ref)
        g6 = (t_tc / t6) if isinstance(t6, float) else None
        print(f"  v6_SEED {v6cfg.get('reduction_loops')}: {round(t6,1) if isinstance(t6,float) else t6}us "
              f"G_vs_tc={g6 and round(g6,3)} maxabs_err={e6 and f'{e6:.2e}'}")
        # broad Helion config sweep -> best CORRECT
        best = None; bestcfg = None
        for rl in (None, 4096, 8192, 16384, 32768, 65536):
            for w in (8, 16, 32):
                for ns in (1, 2, 3):
                    cfg = {"block_sizes": [1], "reduction_loops": [rl],
                           "num_warps": w, "num_stages": ns}
                    t, e = run_cfg(args, cfg, ref)
                    if isinstance(t, float) and e is not None and e < 1e-3:
                        if best is None or t < best:
                            best, bestcfg = t, cfg
        if best:
            print(f"  BEST-CORRECT-Helion: {round(best,1)}us  cfg={bestcfg.get('reduction_loops')}"
                  f"/w{bestcfg['num_warps']}/s{bestcfg['num_stages']}  "
                  f"G_vs_tc={round(t_tc/best,3)}  -> {'SOURCE GAP (cannot beat tc)' if best > t_tc*1.05 else 'CONFIG HEADROOM EXISTS'}")
        print()


if __name__ == "__main__":
    main()
