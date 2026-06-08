"""Verify the FINAL structural predicate for is_structured_combine fires ONLY for
welford and NOT for softmax_two_pass / kl_div / jsd (1 non-grid tile) nor T1.

Predicate (computable inside register_user_tiled_reductions, env+device_ir only):
  - >1 non-grid user tile, AND
  - exactly one inner reduction axis (already required), AND
  - a SECOND non-grid tile with the SAME size_hint as the reduction axis that
    carries NO reduction (a reduce-then-apply two-pass structure over the same
    extent).
"""
from __future__ import annotations

import sys
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.welford import welford  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from helion._compiler.inductor_lowering import ReductionLowering  # noqa: E402


def predicate(kfn, args, label):
    b = kfn.bind(args)
    dev_ir = b.host_function.device_ir
    spec = b.env.config_spec
    env = b.env
    grid_ids = {x for bids in dev_ir.grid_block_ids for x in bids}
    all_bids = list(spec.block_sizes.valid_block_ids())
    non_grid = [x for x in all_bids if x not in grid_ids]

    red_ids = set()
    for gi in dev_ir.graphs:
        for node in gi.graph.nodes:
            low = node.meta.get("lowering")
            if isinstance(low, ReductionLowering):
                bid = getattr(low, "block_index", None)
                if bid is not None:
                    red_ids.add(bid)
    inner_red = [x for x in red_ids if x not in grid_ids]

    def sh(bid):
        try:
            return env.block_sizes[bid].size_hint()
        except Exception:  # noqa: BLE001
            return None

    sig = False
    detail = ""
    if len(inner_red) == 1 and len(non_grid) >= 2:
        red_bid = inner_red[0]
        red_sh = sh(red_bid)
        # second non-grid tile(s) that are NOT reduction axes
        apply_tiles = [x for x in non_grid if x not in red_ids]
        # at least one apply tile with the SAME extent as the reduction axis
        same_extent = [x for x in apply_tiles if sh(x) == red_sh and red_sh is not None]
        sig = len(same_extent) >= 1
        detail = (f"red_bid={red_bid}(sh={red_sh}) apply_tiles={apply_tiles} "
                  f"same_extent={same_extent}")
    print(f"{label}: non_grid={non_grid} inner_red={inner_red}  "
          f"is_structured_combine={sig}  {detail}")


def main():
    print(f"helion={helion.__file__}\n")
    for n in (1024, 1536, 2048, 4096):
        predicate(welford, (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
                            torch.rand(4096, n, device="cuda"), 1e-5),
                  f"welford N={n}")
    predicate(softmax_two_pass, (torch.randn(2048, 4096, device="cuda"),),
              "softmax_two_pass")
    try:
        from examples.kl_div import kl_div_forward as kl
        predicate(kl, (torch.randn(4096, 8192, device="cuda").log_softmax(-1),
                       torch.randn(4096, 8192, device="cuda").softmax(-1)), "kl_div")
    except Exception as e:  # noqa: BLE001
        print(f"kl_div skip: {type(e).__name__}")
    try:
        from examples.jsd import jsd_forward as jsd
        predicate(jsd, (torch.randn(8192, 4096, device="cuda").log_softmax(-1),
                        torch.randn(8192, 4096, device="cuda").log_softmax(-1)), "jsd")
    except Exception as e:  # noqa: BLE001
        print(f"jsd skip: {type(e).__name__}")


if __name__ == "__main__":
    main()
