"""CELL multifact_3way — THREE independent reductions in THREE separate loops over THREE
different tensors, of three DIFFERENT extents.

Stresses the relaxed ``is_eligible`` gate (``len(reduction_facts) >= 1`` -- the old ``== 1``
fence would have DECLINED this 3-fact kernel into the upstream default) and the
``_reduction_primary_fact`` dominant pick (``max(facts, key=size_hint)``). Three separate
``hl.tile(M, block_size=1)`` grid-pinned loops, each doing a full-slice reduction over a
DIFFERENT tensor with a distinct reduction extent (Na > Nb > Nc), so the dominant fact is
unambiguous (the widest, Na). The intended property-point: ORIGIN=grid (each row pinned),
ACCESS=full-slice (materialized rdim, standard track), EXTENT=static, CO-RESIDENCY=different-loop,
N_FACTS=3.

QUESTION: does the dominant-fact seed FIRE (relaxed >=1 gate) and size the primary axis (Na)
sanely (no NO_FIRE, no FLOOR1 on a tiled axis)?

Run:
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-unify /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-unify/_lab/unify/probes/gen/multifact_3way.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/dev/local/helion-unify/_lab/unify/probes")

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


# Three independent full-slice reductions, each in its OWN grid-pinned loop, over three
# DIFFERENT tensors with three DIFFERENT reduction extents (Na > Nb > Nc).
@helion.kernel(static_shapes=False)
def multifact_3way(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    M, Na = a.shape
    _, Nb = b.shape
    _, Nc = c.shape
    # Three SEPARATE outputs so the loops are genuinely independent (no loop-carried
    # dependency on a shared buffer): each loop reduces a DIFFERENT tensor.
    oa = torch.empty([M], dtype=torch.float32, device=a.device)
    ob = torch.empty([M], dtype=torch.float32, device=a.device)
    oc = torch.empty([M], dtype=torch.float32, device=a.device)
    # Loop 1: reduce over Na (the DOMINANT / widest extent) -- sum of squares
    for tile_m in hl.tile(M, block_size=1):
        oa[tile_m] = (a[tile_m, :].to(torch.float32) ** 2).sum(-1)
    # Loop 2: reduce over Nb (mid extent) -- amax of abs
    for tile_m in hl.tile(M, block_size=1):
        ob[tile_m] = b[tile_m, :].to(torch.float32).abs().amax(-1)
    # Loop 3: reduce over Nc (smallest extent) -- mean (sum then scale)
    for tile_m in hl.tile(M, block_size=1):
        oc[tile_m] = c[tile_m, :].to(torch.float32).sum(-1) / Nc
    return oa + ob + oc


def main():
    print(f"helion={helion.__file__}\n")
    M = 8192
    a = torch.randn(M, 4096, device=DEV, dtype=BF16)   # Na=4096 (dominant)
    b = torch.randn(M, 2048, device=DEV, dtype=BF16)   # Nb=2048
    c = torch.randn(M, 1024, device=DEV, dtype=BF16)   # Nc=1024
    intended = {
        "cell": "multifact_3way",
        "access": "full-slice",
        "origin": "grid",
        "extent": "static",
        "co_residency": "different-loop",
        "n_facts": 3,
        "dominant_extent": 4096,
    }
    v = check_kernel("multifact_3way", multifact_3way, (a, b, c), intended)
    obs = v["observed"]
    ns = obs.get("normalized_cfg", {})
    red = v["red"] or "green"
    print(f"[{red}] multifact_3way")
    print(f"  fired                  = {obs.get('fired')}")
    print(f"  n_reduction_facts      = {obs.get('n_reduction_facts')}")
    print(f"  n_matmul_facts         = {obs.get('n_matmul_facts')}")
    print(f"  lowering_reduction_axes= {obs.get('lowering_reduction_axes')}")
    print(f"  grid_block_ids         = {obs.get('grid_block_ids')}")
    print(f"  block_sizes_valid_ids  = {obs.get('block_sizes_valid_ids')}")
    print(f"  reduction_loops_valids = {obs.get('reduction_loops_valid_ids')}")
    print(f"  fact[0]                = {obs.get('fact')}")
    print(f"  raw_seed               = {obs.get('raw_seed')}")
    print(f"  normalized block_sizes = {ns.get('block_sizes') if ns else None}")
    print(f"  normalized reduction_loops = {ns.get('reduction_loops') if ns else None}")
    print(f"  reasons                = {v['reasons']}")
    return v


if __name__ == "__main__":
    main()
