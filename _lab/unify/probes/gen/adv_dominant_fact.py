"""CELL: adv_dominant_fact (minted probe).

ADVERSARIAL-against-the-relaxed-gate + the ``_reduction_primary_fact`` max-extent tie-break.

A genuinely MULTI-FACT kernel (two independent reductions in two separate top-level grid loops,
over two DIFFERENT tensors) where the DOMINANT (max-``size_hint``) fact is the WRONG one to seed:

  - LOOP 1 (the WIDE one): a ROLLED full-slice reduction ``a[m, :].sum(-1)`` over a HUGE axis
    Na -> a STANDARD-track fact (the roller rolls the rdim into ``reduction_loops``; its axis is
    NOT a ``block_sizes`` tile). This is the DOMINANT fact (largest size_hint). It is CHEAP per
    element (a single streamed sum) -- not the perf-critical loop.
  - LOOP 2 (the NARROW, perf-critical one): a USER-TILED reduction
    ``for tk in hl.tile(Nb): acc += b[m, tk].sum(-1)`` over a much SMALLER axis Nb -> a
    USER-TILED-track fact (the rdim IS a tunable ``block_sizes`` tile, a REAL TILED reduction in
    the LOWERING role).

The seed routes to the DOMINANT (wide, standard, cheap) fact:
  - ``_is_standard_reduction(primary)`` is True -> the STANDARD heuristic owns the seed, so
    ``red_values`` carries only the primary's (empty) secondaries -- the NARROW user-tiled axis
    is NOT in ``red_values`` (it belongs to a SEPARATE fact, not this primary's
    ``secondary_reduction_block_ids``).
  - The narrow axis therefore falls to ``_build_block_sizes``'s catch-all branch, whose
    residency cap (``_resident_tile_cap``) divides the per-program byte budget by
    ``r_block_resident = next_pow2(primary.size_hint)`` -- i.e. by the WIDE dominant extent.
  - ADVERSARIAL HYPOTHESIS: a huge dominant Na shrinks that residency cap so far that the
    NARROW critical loop's TILED reduction axis (extent Nb > 1) is FLOORED to block_size=1
    -> FLOOR1_TILED (a real tiled reduction silently serialized by the wrong-fact seed).

QUESTION (compile-time): does the dominant-fact seed FIRE (relaxed >=1 gate) AND avoid flooring
the NARROW user-tiled reduction's real tiled axis? If FLOOR1_TILED fires, the dominant pick
mis-sizes a real reduction = a RED. If GREEN, confirm the narrow axis got a sane (>1) tile.

Run:
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-unify /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-unify/_lab/unify/probes/gen/adv_dominant_fact.py
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

_WT = "/home/dev/local/helion-unify"
assert os.path.abspath(helion.__file__).startswith(_WT + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT})"
)

torch.manual_seed(0)
DEV = "cuda"
BF16 = torch.bfloat16
F32 = torch.float32


# WIDE rolled full-slice reduction (LOOP 1, dominant/standard) + NARROW user-tiled reduction
# (LOOP 2, perf-critical). Two SEPARATE top-level grid loops over two DIFFERENT tensors =>
# two genuinely-independent reductions. The dominant (max size_hint = Na) is the WRONG one to
# seed: it is the cheap wide sum, while the narrow user-tiled loop over Nb is the critical one.
@helion.kernel(static_shapes=False)
def adv_dominant_fact(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, Na = a.shape
    _, Nb = b.shape
    out_a = torch.empty([M], dtype=torch.float32, device=a.device)
    out_b = torch.empty([M], dtype=torch.float32, device=a.device)
    # LOOP 1: WIDE rolled full-slice sum over Na (DOMINANT, standard track, cheap).
    for tile_m in hl.tile(M, block_size=1):
        out_a[tile_m] = a[tile_m, :].to(torch.float32).sum(-1)
    # LOOP 2 (separate grid loop): NARROW user-tiled reduction over Nb (critical, user-tiled
    # track -- the rdim IS a tunable block_sizes tile, a REAL TILED reduction).
    for tile_m2 in hl.tile(M, block_size=1):
        acc = hl.zeros([tile_m2], dtype=torch.float32)
        for tile_nb in hl.tile(Nb):
            acc = acc + b[tile_m2, tile_nb].to(torch.float32).sum(-1)
        out_b[tile_m2] = acc
    return out_a + out_b


def _dump(tag: str, v: dict) -> None:
    obs = v["observed"]
    ns = obs.get("normalized_cfg") or {}
    print(f"\n--- {tag}: {v['name']} ---")
    print(f"  red                       = {v['red']}")
    print(f"  reasons                   = {v['reasons']}")
    print(f"  fired                     = {obs.get('fired')}")
    print(f"  n_reduction_facts         = {obs.get('n_reduction_facts')}")
    print(f"  n_matmul_facts            = {obs.get('n_matmul_facts')}")
    print(f"  lowering_reduction_axes   = {obs.get('lowering_reduction_axes')}")
    print(f"  grid_block_ids            = {obs.get('grid_block_ids')}")
    print(f"  block_sizes_valid_ids     = {obs.get('block_sizes_valid_ids')}")
    print(f"  reduction_loops_valid_ids = {obs.get('reduction_loops_valid_ids')}")
    print(f"  block_sizes (norm)        = {ns.get('block_sizes')}")
    print(f"  reduction_loops (norm)    = {ns.get('reduction_loops')}")
    print(f"  num_warps (norm)          = {ns.get('num_warps')}")
    print(f"  fact[0]                   = {obs.get('fact')}")


def main():
    print(f"helion={helion.__file__}\n")
    M = 8192
    # ADVERSARIAL: Na (the DOMINANT rolled-standard extent) is HUGE; Nb (the narrow user-tiled
    # critical extent) is much smaller. The wide dominant's residency width (next_pow2(Na))
    # is what _resident_tile_cap divides the budget by when sizing the narrow tile.
    a = torch.randn(M, 65536, device=DEV, dtype=BF16)   # Na=65536 (DOMINANT, cheap wide sum)
    b = torch.randn(M, 512, device=DEV, dtype=BF16)     # Nb=512   (NARROW user-tiled, critical)
    intended = {
        "cell": "adv_dominant_fact",
        "access": "rolled-fullslice (wide,dominant) + user-tiled (narrow,critical)",
        "origin": "grid",
        "co_residency": "different-loop",
        "n_facts": ">=2 intended",
        "dominant_extent": 65536,
        "narrow_critical_extent": 512,
        "saturation_axis": "reduction-OP identity / dominant-pick correctness (not in §1 axes)",
        "hypothesis": "huge dominant residency floors the narrow tiled reduction to block_size=1",
    }
    v = check_kernel("adv_dominant_fact", adv_dominant_fact, (a, b), intended)
    _dump("adv_dominant_fact", v)
    return v


if __name__ == "__main__":
    main()
