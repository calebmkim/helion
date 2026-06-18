"""welford ORACLE (quick autotune) at 1536 (non-pow2 canary) and 4096 (the spill
shape). Fair re-bench of the verbatim winner vs my cap-rule seed + tc. Tells us
the per-shape ceiling and whether 4096's 0.70 is config headroom or a kernel
ceiling. Also CHECKS the autotuner winner's correctness at non-pow2 (does the
oracle ever pick a masked combine -> wrong? it shouldn't, the accuracy gate
rejects wrong configs).
"""
from __future__ import annotations

import os
import sys
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402
from examples.welford import welford, eager_layer_norm  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 1e-5
SHAPE = (262144, int(os.environ.get("WF_N", "4096")))


def args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def med(fn, reps=5):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def mfloor(a):
    bs = welford.bind(a).config_spec.block_sizes[0]
    return max(1, bs.min_size, bs.autotuner_min)


def rebench(a, cfg, ref):
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
    maxabs = float((out.float() - ref.float()).abs().max())
    return med(lambda: b(*a)) * 1000, ok, maxabs


def main():
    m, n = SHAPE
    print(f"helion={helion.__file__}  shape=({m},{n})\n", flush=True)
    a = args(m, n)
    ref = eager_layer_norm(*a)
    mf = mfloor(a); P = np2(n); L = n & (-n)
    torch._dynamo.reset()
    tc = torch.compile(eager_layer_norm); tc(*a)
    tclat = med(lambda: tc(*a)) * 1000
    # my cap2048 rule seed
    seed_bs = [mf, min(L, 2048), P]
    slat, sok, smax = rebench(a, {"block_sizes": seed_bs, "num_warps": 8,
                                  "num_stages": 1}, ref)
    print(f"tc={round(tclat,1)}us   SEED(cap2048) bs={seed_bs} w8: {round(slat,1)}us "
          f"ok={sok} maxabs={smax:.1e} G={round(tclat/slat,3)}\n", flush=True)
    # oracle: quick autotune the bare welford
    os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"
    k = helion.kernel(welford.fn)
    b = k.bind(a)
    b.autotune(a)
    win = dict(b._config)
    wlat, wok, wmax = rebench(a, win, ref)
    print(f"\nORACLE winner block_sizes={win.get('block_sizes')} "
          f"num_warps={win.get('num_warps')} num_stages={win.get('num_stages')} "
          f"pid={win.get('pid_type')}", flush=True)
    print(f"ORACLE re-bench: {round(wlat,1)}us ok={wok} maxabs={wmax:.1e} "
          f"G_oracle={round(tclat/wlat,3)}  (seed/oracle={round(slat/wlat,3)})", flush=True)


if __name__ == "__main__":
    main()
