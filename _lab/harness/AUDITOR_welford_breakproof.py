"""AUDITOR A5: prove welford's decline is (a) for the RIGHT structural reason
(2 non-grid user tiles), and (b) HONEST -- forcing the single-axis persistent
seed onto welford is both ~10-20x SLOWER and NUMERICALLY WRONG at non-pow2 N.

Part 1: inspect block_sizes / grid_block_ids to count non-grid tiles for
        welford vs a working T2 (softmax_two_pass, which must have exactly 1).
Part 2: force the seed welford would get if the gate DID fire (floor every
        non-reduction block to 1, widen only the reduction tile) and compare
        latency + correctness vs the un-seeded default, at pow2 AND non-pow2 N.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.welford import welford, eager_layer_norm  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402

EPS = 1e-5


def med(fn, reps=5):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def inspect(name, kfn, args):
    b = kfn.bind(args)
    env = b.env
    spec = env.config_spec
    dev_ir = b.host_function.device_ir
    grid_ids = {x for bids in dev_ir.grid_block_ids for x in bids}
    all_bids = list(spec.block_sizes.valid_block_ids())
    non_grid = [x for x in all_bids if x not in grid_ids]
    print(f"{name}: block_sizes={all_bids} grid_block_ids={sorted(grid_ids)} "
          f"non_grid_tiles={non_grid} (count={len(non_grid)}) "
          f"reduction_facts={len(spec.reduction_facts)}")
    return len(non_grid)


def wf_args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def force_welford_seed(args, n):
    """Run welford with the seed it WOULD get from the single-axis heuristic:
    persistent reduction (widen reduction tile to next_pow2(N)), every other
    user block floored to 1. Returns (time_us, max_abs_err vs eager)."""
    from helion._utils import next_power_of_2 as np2
    # welford has 3 block_sizes: [tile_m, tile_n(reduce), tile_n(normalize)]
    # the single-axis seed floors the grid (tile_m) and floors whatever it does
    # NOT recognize as the reduction. The catastrophe: it widens ONE tile_n to
    # persistent and leaves the OTHER tile_n at 1.
    weight, bias, x, eps = args
    m, _ = x.shape
    # default (un-seeded)
    kd = welford
    bd = kd.bind(args)
    bd.ensure_config_exists(args)
    out_d = bd(*args)
    td = med(lambda: bd(*args)) * 1000
    ref = eager_layer_norm(weight, bias, x, eps)
    err_d = (out_d.float() - ref.float()).abs().max().item()
    # forced single-axis-persistent seed: tile_m=1, reduction tile=pow2(N),
    # normalize tile left at 1 (the multi-pass catastrophe the guard prevents)
    P = np2(n)
    results = {}
    for label, bsz in [
        ("seed[m=1,red=P,norm=1]", [1, P, 1]),
        ("seed[m=1,red=P,norm=P]", [1, P, P]),  # even if BOTH widened
    ]:
        try:
            k = helion.kernel(welford.fn, configs=[helion.Config(
                block_sizes=bsz, num_warps=32, num_stages=1)])
            bk = k.bind(args)
            bk.ensure_config_exists(args)
            out = bk(*args)
            t = med(lambda: bk(*args)) * 1000
            err = (out.float() - ref.float()).abs().max().item()
            results[label] = (t, err)
        except Exception as e:  # noqa: BLE001
            results[label] = (f"ERR:{type(e).__name__}", None)
    return td, err_d, results


def main():
    print(f"helion={helion.__file__}\n")
    print("=== Part 1: non-grid tile count (welford must be >1, softmax_two_pass ==1) ===")
    inspect("welford (4096,1024)", welford, wf_args(4096, 1024))
    inspect("softmax_two_pass (2048,4096)", softmax_two_pass,
            (torch.randn(2048, 4096, device="cuda"),))
    print()
    print("=== Part 2: forced single-axis seed vs default — speed + correctness ===")
    for (m, n, tag) in [(4096, 1024, "pow2 N"), (4096, 1000, "NON-pow2 N"),
                        (4096, 1536, "NON-pow2 N")]:
        args = wf_args(m, n)
        td, err_d, results = force_welford_seed(args, n)
        print(f"\n  welford ({m},{n}) [{tag}]")
        print(f"    default(un-seeded): {round(td,1)}us  maxabs_err={err_d:.2e}")
        for label, (t, err) in results.items():
            ts = round(t, 1) if isinstance(t, float) else t
            slow = f"{t/td:.1f}x slower" if isinstance(t, float) else ""
            es = f"{err:.2e}" if err is not None else "n/a"
            print(f"    {label}: {ts}us {slow}  maxabs_err={es}")


if __name__ == "__main__":
    main()
