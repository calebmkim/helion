"""CLEAN welford probe: NO autotune. default baseline = config_spec.default_config()
(deterministic). Candidate seeds = helion.kernel(fn, configs=[ONE]) (short-circuit).

Per shape, A/B the CORRECT candidate configs vs the deterministic default + tc.
Correctness rule: combine tile (block 1) = pow2 DIVISOR of N (Tn==valid count, no
masked-count bug). normalize tile (block 2) = any pow2 (masked write = correct).

Candidates (all CORRECT by construction):
  combine = largest_pow2_div(N)   (= N for pow2, 512 for 1536)
  combine = min(that, CAP)        capped looped divisor chunk
  normalize = next_pow2(N) persistent, OR a capped chunk
"""
from __future__ import annotations

import math
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
_SET = {
    "small": [(262144, 1024), (262144, 1536)],
    "big": [(262144, 2048), (262144, 4096)],
    "all": [(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096)],
}
IN_SAMPLE = _SET[os.environ.get("WF_SET", "all")]


def args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def med(fn, reps=3):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def mfloor(a):
    bs = welford.bind(a).config_spec.block_sizes[0]
    return max(1, bs.min_size, bs.autotuner_min)


def lpd(n):
    return n & (-n)


def run_seed(a, bs, w, ref):
    k = helion.kernel(welford.fn, configs=[helion.Config(
        block_sizes=bs, num_warps=w, num_stages=1)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
    maxabs = float((out.float() - ref.float()).abs().max())
    lat = med(lambda: b(*a)) * 1000
    return lat, ok, maxabs


def run_default(a, ref):
    bd = welford.bind(a)
    cfg = helion.Config(**dict(bd.config_spec.default_config()))
    k = helion.kernel(welford.fn, configs=[cfg])
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
    lat = med(lambda: b(*a)) * 1000
    return lat, ok, dict(cfg)["block_sizes"]


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    gd, gs = [], []
    for (m, n) in IN_SAMPLE:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        mf = mfloor(a)
        P = np2(n)
        L = lpd(n)
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm); tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        dlat, ok_d, dbs = run_default(a, ref)
        print(f"=== ({m},{n}) np2={P} lpd={L} m_floor={mf}  tc={round(tclat,1)}us  "
              f"default(bs={dbs})={round(dlat,1)}us(ok={ok_d}) G_def={round(tclat/dlat,3)} ===",
              flush=True)
        gd.append(tclat / dlat)
        # candidate configs (all CORRECT): (combine, normalize, warps)
        cands = [
            (L, P, 8), (L, P, 16),
            (min(L, 1024), P, 8), (min(L, 1024), P, 16),
            (min(L, 1024), min(P, 1024), 16),
            (min(L, 2048), min(P, 2048), 16),
        ]
        best = None
        for (c, r2, w) in dict.fromkeys(cands):  # dedup
            lat, ok, maxabs = run_seed(a, [mf, c, r2], w, ref)
            g = tclat / lat
            flag = "" if ok else "  <-- WRONG"
            print(f"  combine={c:5d} norm={r2:5d} w{w}: {round(lat,1):8}us "
                  f"ok={ok} maxabs={maxabs:.1e} G={round(g,3)}{flag}", flush=True)
            if ok and (best is None or lat < best[0]):
                best = (lat, c, r2, w)
        if best:
            print(f"  >> BEST CORRECT: combine={best[1]} norm={best[2]} w{best[3]} "
                  f"{round(best[0],1)}us G={round(tclat/best[0],3)}", flush=True)
            gs.append(tclat / best[0])
        print(flush=True)
    def gm(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else None
    print(f"GEOMEAN  G_best_correct={gm(gs) and round(gm(gs),4)}  "
          f"G_default={gm(gd) and round(gm(gd),4)}", flush=True)


if __name__ == "__main__":
    main()
