"""GEN CELL: reduce_apply_reduce

A reduce -> full-width APPLY -> SECOND reduce over the applied result. Two reductions
with a NON-REDUCTION apply pass between them, and REUSE/row_reread = TRUE (the same row
x[m, :] is re-read for both the stat reduction AND the applied second reduction).

Distinct from the existing corpus:
  - C2 reduce_then_apply_fullslice : reduce THEN apply, NO second reduce (one fact).
  - P3 materialized_two_full_slice : two full-slice reductions, but NO apply pass between
                                     and the 2nd does not consume the 1st's stat.
  - P5 twopass_usertiled           : two user-tiled reductions in SEPARATE loops, user-tiled,
                                     no full-width apply pass, 2nd independent of 1st.

Here the SECOND reduction is data-dependent on the FIRST (numerically-stable softmax-style):
    m  = max_n x[row, :]                       # reduce #1 (full-slice over N)
    p  = exp(x[row, :] - m)                    # full-width APPLY (non-reduction, row re-read)
    s  = sum_n p                               # reduce #2 (full-slice over N, over applied row)
This stresses the non_reduction_loop + row_reread interplay: both reductions live in the
SAME grid-pinned loop, the row is materialized/re-read, and the apply pass is a full-width
elementwise op that is neither a reduction nor a tiled loop.
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROBES = os.path.abspath(os.path.join(_THIS, ".."))
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import check_kernel  # noqa: E402

torch.manual_seed(0)
DEV = "cuda"
BF16 = torch.bfloat16


# reduce -> full-width apply -> second reduce (REUSE/row_reread, non-reduction apply pass)
@helion.kernel(static_shapes=False)
def reduce_apply_reduce(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        row = x[tile_m, :].to(torch.float32)          # row materialized
        m = torch.amax(row, dim=-1)                   # reduce #1 over N (the stat)
        applied = torch.exp(row - m[:, None])         # full-width APPLY (non-reduction pass)
        s = applied.sum(-1)                           # reduce #2 over N (over applied result)
        out[tile_m, 0] = s
    return out


def main():
    print(f"helion={helion.__file__}\n")
    assert os.path.abspath(helion.__file__).startswith(
        "/home/dev/local/helion-unify" + os.sep
    ), helion.__file__
    x = torch.randn(8192, 4096, device=DEV, dtype=BF16)
    intended = {
        "cell": "reduce_apply_reduce",
        "access": "full-slice",
        "origin": "inner/grid-pinned",
        "extent": "static",
        "n_reduce": 2,
        "non_red_loop": True,
        "row_reread": True,
        "reuse": "2nd-reduce-consumes-1st-stat",
    }
    v = check_kernel("reduce_apply_reduce", reduce_apply_reduce, (x.clone(),), intended)
    obs = v["observed"]
    ns = obs.get("normalized_cfg", {})
    red = v["red"] or "green"
    print(f"[{red:13s}] reduce_apply_reduce")
    print(f"  fired={obs.get('fired')}")
    print(f"  n_reduction_facts={obs.get('n_reduction_facts')}")
    print(f"  n_matmul_facts={obs.get('n_matmul_facts')}")
    print(f"  lowering_reduction_axes={obs.get('lowering_reduction_axes')}")
    print(f"  grid_block_ids={obs.get('grid_block_ids')}")
    print(f"  block_sizes_valid_ids={obs.get('block_sizes_valid_ids')}")
    print(f"  reduction_loops_valid_ids={obs.get('reduction_loops_valid_ids')}")
    print(f"  fact={obs.get('fact')}")
    print(f"  normalized_cfg={ns}")
    for r in v["reasons"]:
        print(f"  reason: {r}")
    return v


if __name__ == "__main__":
    main()
