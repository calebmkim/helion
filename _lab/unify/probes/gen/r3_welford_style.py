"""TARGET CELL: r3_welford_style -- OP-VARIETY probe of the welford/var_mean reduction.

OP-VARIETY: ``torch.var_mean`` is the ONE reduction op that "promotes differently": it is
lowered (``inductor_lowering_extra.var_mean_helper_``) by casting the input to the COMPUTE
dtype (fp32 for a half-precision input via ``get_computation_dtype``) and decomposing into a
SUM-based mean/variance (sum_x, sum_x2). It is the reduce-then-apply norm idiom (layernorm):
reduce to mean+var, then a full-width apply pass re-reads the row to normalize. This cell asks
the two task questions for that op-variety:

  (1) Does var_mean FIRE and size SANELY (a justified standard-reduction seed, no NO_FIRE, no
      FLOOR1_TILED, no grid collapse)?
  (2) Is the DTYPE handling correct for var_mean -- specifically, does the persist-vs-loop
      decision key on the genuine fp32-accumulator footprint, NOT on the bf16 input width?

WHY var_mean is the sharp dtype test. A naked Helion reduction tree accumulates in fp32
regardless of input dtype, so the residency byte caps were recently floored at
``max(fact.itemsize, 4)`` (``_resident_itemsize``) to bound the genuine fp32 resident tile.
var_mean is the op whose internal promotion that fix is explicitly described against ("only
var_mean differs"). REPO-VERIFIED finding of this probe: even though var_mean promotes
internally, the recorded ``fact.itemsize`` is still 2 for a bf16 input (the multi-node
sum-decomposition's reduction INPUT is read as the bf16 tensor at ``_reduction_input_itemsize``,
last-wins=2 -- the docstring claim that this is "4 at both bf16 and fp32" for var_mean does NOT
hold at the fact level). So var_mean is saved EXACTLY like a plain bf16 reduction: only by the
``max(itemsize,4)`` floor. This cell pins the persist/loop boundary to prove that floor fires.

PROPERTY POINT (all §1 axes at the common corpus point; only OP-VARIETY + DTYPE varied):
  ACCESS         = standard (Helion-rolled var_mean over the inner axis, rdim rides reduction_loops)
  ORIGIN         = inner (N is the inner reduced dim; grid_reduction_origin=False)
  EXTENT         = static, at/above the persist-vs-loop boundary (see N below)
  CARRIED-RESIDENT = none (scalar-per-row mean/var -> num_carried_2d_tiles=0)
  CO-RESIDENCY   = single reduction axis
  REUSE          = row_reread=True (the normalize apply re-reads the row -> persist-eligible)
  NON-REDUCTION-LOOP = none (single combined apply via the broadcast write-back)
  DIMS           = 2
  PINNED-GRID    = none (M is a plain tunable hl.tile(M) grid axis)
  OP-VARIETY     = var_mean (mean+variance, the promote-differently op)
  DTYPE          = bfloat16 (itemsize 2; the fp32-accumulator footprint is 2x the input width)

DTYPE-FAITHFUL CHECK (run a fresh bind per N -- never re-bind one kernel object across shapes
in one process; that freezes the first config and reads back a stale warps=1 persist).
ROW_PERSIST_MAX_BYTES = 245760. The fp32-accumulator row = N*4:
  - N=61440: fp32 row = 245760 = cap  -> PERSIST (reduction_loops=[None])  [<= cap is allowed]
  - N=65536: fp32 row = 262144 > cap  -> LOOP    (reduction_loops=[16384]) even though the bf16
             INPUT row (131072 B) is well under the cap. If the cap keyed on the raw bf16 width
             this would WRONGLY persist a spilling fp32 tile -- the dtype hole the floor closes.
We size N=65536 so the seed MUST loop on the fp32 footprint, and assert it does (the headline
dtype-correctness claim), plus dump the boundary sibling N=61440 for the minter.
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
ROW_PERSIST_MAX_BYTES = 245760


@helion.kernel(static_shapes=False)
def welford_var_mean_normalize(x: torch.Tensor) -> torch.Tensor:
    """x: [M, N] bfloat16. Welford/layernorm-style reduce-then-apply: ``torch.var_mean`` over
    the inner axis N (the promote-differently op -- lowered as a fp32-cast sum-decomposition),
    then a full-width normalize apply that re-reads the row (``row_reread=True``). M is a plain
    tunable ``hl.tile(M)`` grid axis; N is the standard rolled inner reduction. The reduction
    consumes bf16 DIRECTLY (var_mean does the fp32 cast internally), so ``fact.itemsize`` records
    2 while the live accumulator tile is fp32 -- the exact case the ``max(itemsize,4)`` residency
    floor must catch so the persist/loop decision keys on the fp32 footprint, not the bf16 width."""
    M, N = x.shape
    out = torch.empty([M, N], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M):
        row = x[tile_m, :]                          # [M_BLOCK, N] bf16
        var, mean = torch.var_mean(row, dim=-1)     # var_mean: promotes to fp32 internally
        out[tile_m, :] = (row.to(torch.float32) - mean[:, None]) * torch.rsqrt(
            var[:, None] + 1e-5
        )                                           # full-width re-read apply (row_reread)
    return out


def main() -> None:
    print(f"helion={helion.__file__}\n")
    _wt_root = os.path.abspath(os.path.join(_THIS, "..", "..", "..", ".."))
    assert os.path.abspath(helion.__file__).startswith(_wt_root + os.sep), (
        f"{helion.__file__} not under {_wt_root}"
    )

    M = 512
    # HEADLINE N: fp32-accumulator row (N*4 = 262144) EXCEEDS the cap (245760) while the bf16
    # input row (131072) does not -> the seed MUST loop on the fp32 footprint.
    N = 65536
    x = torch.randn(M, N, device=DEV, dtype=BF16)
    intended = {
        "cell": "r3_welford_style",
        "op_variety": "var_mean (mean+variance, promote-differently)",
        "access": "standard(rolled var_mean)",
        "origin": "inner",
        "extent": f"static N={N} (fp32 row {N * 4}B > cap {ROW_PERSIST_MAX_BYTES})",
        "carried_resident": False,
        "co_residency": "single",
        "reuse": "row_reread(full-width normalize apply)",
        "non_reduction_loop": None,
        "dims": 2,
        "pinned_grid": None,
        "dtype": "bfloat16(itemsize=2; fp32-accumulator footprint=2x)",
    }
    v = check_kernel("welford_var_mean_normalize", welford_var_mean_normalize, (x,), intended)

    import json

    print(json.dumps(v, indent=2, default=repr))

    obs = v["observed"]
    cfg = obs.get("normalized_cfg") or {}
    fact = obs.get("fact") or {}
    rl = cfg.get("reduction_loops")
    fp32_row = N * 4
    looped = rl != [None]
    print("\n=== r3_welford_style DTYPE-FAITHFUL CHECK ===")
    print(
        f"fired={obs.get('fired')} n_reduction_facts={obs.get('n_reduction_facts')} "
        f"n_matmul_facts={obs.get('n_matmul_facts')}"
    )
    print(
        f"itemsize(recorded)={fact.get('itemsize')} input_load_itemsize="
        f"{fact.get('input_load_itemsize')} row_reread={fact.get('row_reread')} "
        f"num_carried_2d_tiles={fact.get('num_carried_2d_tiles')} "
        f"grid_reduction_origin={fact.get('grid_reduction_origin')}"
    )
    print(
        f"block_sizes={cfg.get('block_sizes')} reduction_loops={rl} "
        f"num_warps={cfg.get('num_warps')}"
    )
    print(
        f"fp32_accumulator_row={fp32_row}B > cap({ROW_PERSIST_MAX_BYTES})="
        f"{fp32_row > ROW_PERSIST_MAX_BYTES}; seed={'LOOP' if looped else 'PERSIST'} "
        f"-> dtype handling {'CORRECT (loops on fp32 footprint)' if looped else 'WRONG (persists spilling fp32 tile)'}"
    )
    assert v["red"] is None, f"unexpected RED: {v['red']} {v['reasons']}"
    assert looped, "DTYPE HOLE: fp32 footprint exceeds cap but seed persists (bf16-width cap)"
    print("\n=== VERDICT: GREEN (var_mean fires + sizes sanely; dtype-faithful) ===")


if __name__ == "__main__":
    main()
