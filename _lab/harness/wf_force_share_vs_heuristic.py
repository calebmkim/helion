"""Decision A/B: a "force apply := combine" heuristic vs the existing split heuristic.

The existing is_structured_combine seed emits [M, combine, apply] where
  combine = min(np2(N), STRUCTURED_COMBINE_CAP/4 = 8192)
  apply   = np2(N) if per-row bytes <= 12288 else min(np2(N), 2048)   # the footprint clamp
So apply diverges from combine only at wide N (>~3072), clamped to 2048.

"Force share" = drop the apply clamp, set apply := combine. This asks: does collapsing
the two tiles to one (combine's value) raise or lower OVERALL perf vs the existing split?

Tests ONLY the decision-relevant shapes — those where the existing emit has apply !=
combine (small-N shapes already share, so they're no-ops and skipped). Splits the
verdict by curriculum role (train = perf-targeted; robustness = correctness/not-
catastrophically-slow only) because the answer differs sharply between them.

Per shape: bench existing [M,combine,apply] vs forced [M,combine,combine], allclose-gate
both. Report ratio forced/existing (<1 = forced faster), geomean over train-differ
shapes, and FLAG any robustness shape where forced is catastrophically slower.
"""
from __future__ import annotations

import math
import os
import sys

import torch

import helion

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(helion.__file__)))
assert helion.__file__.startswith(WORKTREE), helion.__file__
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.welford import eager_layer_norm  # noqa: E402
from examples.welford import welford  # noqa: E402

# (shape, role) — decision-relevant welford shapes (those where apply != combine in the
# existing emit). Small-N already-shared shapes are no-ops and omitted.
SHAPES = [
    ((16384, 4096), "train"),
    ((8192, 5120), "train"),
    ((8192, 7168), "train"),
    ((8192, 8192), "train"),
    ((8192, 12288), "train"),
    ((4096, 16384), "train"),
    ((65536, 4096), "train"),
    ((32768, 8192), "train"),
    ((262144, 5120), "robust"),   # confirmed 6.6x blowup at apply=combine=8192
    ((262144, 7168), "robust"),   # UNMEASURED — combine=8192, predicted same regime
    ((65536, 16384), "robust"),   # UNMEASURED — combine=8192, lower M_BLOCK, uncertain
]


def args(m, n):
    return (
        torch.rand(n, device="cuda"),
        torch.rand(n, device="cuda"),
        torch.rand(m, n, device="cuda"),
        1e-5,
    )


def med(fn, reps=4):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def emitted_seed(a):
    k = helion.kernel(welford.fn)
    b = k.bind(a)
    seeds = b.env.config_spec.compiler_seed_configs
    assert seeds, "no compiler seed config emitted"
    cfg = seeds[0]
    return list(cfg.block_sizes), int(cfg.num_warps), int(cfg.num_stages)


def run(a, bs, w, ns, ref):
    k = helion.kernel(
        welford.fn,
        configs=[helion.Config(block_sizes=bs, num_warps=w, num_stages=ns)],
    )
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
    lat = med(lambda: b(*a)) * 1000
    return lat, ok


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    rows = []
    for (m, n), role in SHAPES:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        bs, w, ns = emitted_seed(a)
        M, combine, apply_ = bs[0], bs[1], bs[2]
        if apply_ == combine:
            print(f"  SKIP ({m},{n}) — emit already shares ({bs})", flush=True)
            continue
        ex_lat, ex_ok = run(a, [M, combine, apply_], w, ns, ref)
        fc_lat, fc_ok = run(a, [M, combine, combine], w, ns, ref)
        ratio = fc_lat / ex_lat if (ex_ok and fc_ok) else float("inf")
        verdict = ("FORCED-FASTER" if ratio < 0.97
                   else "tie" if ratio <= 1.03
                   else "FORCED-SLOWER" if ratio <= 1.5
                   else "FORCED-CATASTROPHIC")
        print(
            f"=== ({m},{n}) [{role}] existing=[{M},{combine},{apply_}] {round(ex_lat,1)}us "
            f"ok={ex_ok} | forced=[{M},{combine},{combine}] {round(fc_lat,1)}us ok={fc_ok} "
            f"| forced/existing={round(ratio,3)}  {verdict} ===",
            flush=True,
        )
        rows.append(((m, n), role, ratio, verdict, ex_ok and fc_ok))

    print("\n================ SUMMARY ================", flush=True)
    print(f"{'shape':>15} {'role':>8} {'forced/existing':>16}  verdict", flush=True)
    for sh, role, r, v, ok in rows:
        print(f"{str(sh):>15} {role:>8} {round(r,3):>16}  {v}"
              f"{'' if ok else '  (CORRECTNESS FAIL)'}", flush=True)

    train_r = [r for _, role, r, _, ok in rows if role == "train" and ok and math.isfinite(r)]
    if train_r:
        g = math.exp(sum(math.log(x) for x in train_r) / len(train_r))
        print(f"\nTRAIN geomean forced/existing = {round(g,3)} "
              f"({'forced FASTER overall on train' if g < 1 else 'forced SLOWER overall on train'})",
              flush=True)
    cats = [(sh, r) for sh, role, r, v, ok in rows
            if role == "robust" and v == "FORCED-CATASTROPHIC"]
    print(f"ROBUSTNESS catastrophic regressions (>1.5x): {len(cats)} "
          f"{[(str(s), round(r,2)) for s, r in cats]}", flush=True)
    print("\nOVERALL READ:", flush=True)
    if cats:
        print("  Forcing apply:=combine RAISES train perf but RE-OPENS the EDIT#4 footprint "
              "cliff on high-M robustness canaries -> NET a regression (fails the "
              "not-catastrophically-slow robustness bar).", flush=True)
    else:
        print("  No robustness catastrophe found -> forcing apply:=combine is a net WIN; "
              "the apply clamp is unnecessary.", flush=True)


if __name__ == "__main__":
    main()
