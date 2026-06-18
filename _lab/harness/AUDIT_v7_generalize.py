"""GENERALIZABILITY probe: does is_structured_combine fire for a DIFFERENT
reduce-then-apply structured combine (NOT welford)? If the gate is structural it
must fire for any kernel with >1 non-grid tile over the same extent where one
carries the reduction and another does not. Build a synthetic 'max-normalize'
kernel (pass 1: reduce max over N; pass 2: apply subtract-max over N) -- a
totally different combine from welford -- and confirm the gate classifies it
is_structured_combine=True and emits the structured seed.

Also build a NEGATIVE control: a single-pass kernel (one tile over N, reduce
only) must classify False (like softmax_two_pass).
"""
from __future__ import annotations

import sys
import torch
import helion
import helion.language as hl

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion._compiler.inductor_lowering import ReductionLowering  # noqa: E402


# A DIFFERENT structured combine: reduce-max pass + subtract-max apply pass.
# Two hl.tile(n) loops over the SAME N: pass1 reduces (max), pass2 applies (no
# reduction). Structurally like welford but a completely different math.
@helion.kernel()
def max_normalize(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n):
            chunk = x[tile_m, tile_n]
            acc = torch.maximum(acc, torch.amax(chunk, dim=-1))
        mx = acc[:, None]
        for tile_n in hl.tile(n):
            out[tile_m, tile_n] = (x[tile_m, tile_n] - mx).to(x.dtype)
    return out


# NEGATIVE control: single-pass reduce-only (no apply pass) -> 1 non-grid tile.
@helion.kernel()
def single_pass_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m], 0.0, dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc = acc + torch.sum(x[tile_m, tile_n], dim=-1)
        out[tile_m] = acc
    return out


def classify(kfn, args, label):
    b = kfn.bind(args)
    dev_ir = b.host_function.device_ir
    spec = b.env.config_spec
    env = b.env
    grid_ids = {x for bids in dev_ir.grid_block_ids for x in bids}
    non_grid = [x for x in spec.block_sizes.valid_block_ids() if x not in grid_ids]
    red_ids = set()
    for gi in dev_ir.graphs:
        for node in gi.graph.nodes:
            low = node.meta.get("lowering")
            if isinstance(low, ReductionLowering):
                bid = getattr(low, "block_index", None)
                if bid is not None:
                    red_ids.add(bid)
    inner_red = [x for x in red_ids if x not in grid_ids]
    apply_tiles = [x for x in non_grid if x not in red_ids]
    indep = len(non_grid) > 1 and len(apply_tiles) >= 1
    seeds = compiler_seed_configs(env, dev_ir)
    facts = spec.reduction_facts
    sc = [f.is_structured_combine for f in facts]
    print(f"\n=== {label} ===", flush=True)
    print(f"  non_grid={non_grid} inner_red={inner_red} apply_tiles={apply_tiles}", flush=True)
    print(f"  INDEP is_structured_combine={indep}  FACT sc={sc}", flush=True)
    print(f"  #facts={len(facts)} seeds={[dict(s) for s in seeds]}", flush=True)
    return indep, sc


def main():
    print(f"helion={helion.__file__}", flush=True)
    x = torch.randn(8192, 1536, device="cuda")
    i1, s1 = classify(max_normalize, (x,), "max_normalize (DIFFERENT structured combine)")
    i2, s2 = classify(single_pass_sum, (x,), "single_pass_sum (negative control, 1 non-grid tile)")
    print("\n=== VERDICT ===", flush=True)
    print(f"  max_normalize fires structured-combine: indep={i1} fact={s1} "
          f"(EXPECT True -> gate generalizes beyond welford)", flush=True)
    print(f"  single_pass_sum: indep={i2} fact={s2} "
          f"(EXPECT False -> not a fence on structure)", flush=True)


if __name__ == "__main__":
    main()
