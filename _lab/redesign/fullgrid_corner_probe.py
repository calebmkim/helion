"""Stress-test the FULL_GRID-corners family. Binds candidate kernels, prints Stage-1
categorization + co-residency groups + the emitted seed config so we can validate the
predicted taxonomy + spot allocator mis-sizing before committing to the candidates."""

from __future__ import annotations

import os
import sys

_HARNESS = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS, "..", ".."))
for _d in (_HARNESS, _WT_ROOT):
    if os.path.abspath(_d) not in sys.path:
        sys.path.insert(0, os.path.abspath(_d))

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402
from helion._compiler.compile_environment import FixedBlockSizeSource  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), helion.__file__


# ---------------- Candidate C1: two FULL_GRID reductions, same body, two specialized axes
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def c1_two_fullgrid(x: torch.Tensor, y: torch.Tensor):
    """x:[T,G,GS] per-(token,group) max over GS; y:[T,H,HS] per-(token,head) sum over HS.
    BOTH reductions are over specialized full-extent grid axes (FULL_GRID), in ONE outer body."""
    T, G, GS = x.shape
    _, H, HS = y.shape
    hl.specialize(GS); hl.specialize(G)
    hl.specialize(HS); hl.specialize(H)
    o1 = torch.empty([T, G], dtype=torch.float32, device=x.device)
    o2 = torch.empty([T, H], dtype=torch.float32, device=x.device)
    for tile_t, tile_g, tile_c, tile_h, tile_d in hl.tile(
        [T, G, GS, H, HS], block_size=[1, None, GS, None, HS]
    ):
        o1[tile_t, tile_g] = torch.amax(x[tile_t, tile_g, tile_c].to(torch.float32), -1)
        o2[tile_t, tile_h] = y[tile_t, tile_h, tile_d].to(torch.float32).sum(-1)
    return o1, o2


def c1_args(T=8192, G=32, GS=128, H=16, HS=256, dtype=torch.bfloat16, device="cuda"):
    return (torch.randn(T, G, GS, device=device, dtype=dtype),
            torch.randn(T, H, HS, device=device, dtype=dtype))


# ---------------- Candidate C2: FULL_GRID co-resident with a LARGE rolled FULL_SLICE
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def c2_fullgrid_plus_bigslice(x: torch.Tensor, w: torch.Tensor):
    """x:[T,G,GS] per-(token,group) max over GS (FULL_GRID, tiny extent 128);
    w:[T,N] per-token sum over a LARGE N (full slice, N=16384) in the SAME outer body,
    re-read for a normalize so it wants to persist. Does the FULL_GRID extent factor the
    budget (pinned_inner=GS) without crushing the big slice's persistence?"""
    T, G, GS = x.shape
    _, N = w.shape
    hl.specialize(GS); hl.specialize(G)
    o1 = torch.empty([T, G], dtype=torch.float32, device=x.device)
    o2 = torch.empty([T, N], dtype=torch.float32, device=x.device)
    for tile_t, tile_g, tile_c in hl.tile([T, G, GS], block_size=[1, None, GS]):
        o1[tile_t, tile_g] = torch.amax(x[tile_t, tile_g, tile_c].to(torch.float32), -1)
        row = w[tile_t, :].to(torch.float32)
        s = row.sum(-1)                       # FULL_SLICE over N (re-read below -> persist)
        o2[tile_t, :] = (row / (s[:, None] + 1.0)).to(torch.float32)
    return o1, o2


def c2_args(T=8192, G=32, GS=128, N=16384, dtype=torch.bfloat16, device="cuda"):
    return (torch.randn(T, G, GS, device=device, dtype=dtype),
            torch.randn(T, N, device=device, dtype=dtype))


# ---------------- Candidate C3: nested specialized grid axes, ONE 2-axis FULL_GRID reduction
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def c3_nested_fullgrid_2axis(x: torch.Tensor):
    """x:[T,G,A,B] per-(token,group) sum over BOTH inner specialized axes A and B (a 2-axis
    FULL_GRID reduction: both A and B are specialized full-extent grid axes). Grid = [T,G],
    reduce the resident [A,B] block. Tests the FULL_GRID exclusion + pinned-extent footprint
    when TWO grid axes are full-extent reduced together."""
    T, G, A, B = x.shape
    hl.specialize(A); hl.specialize(B); hl.specialize(G)
    out = torch.empty([T, G], dtype=torch.float32, device=x.device)
    for tile_t, tile_g, tile_a, tile_b in hl.tile(
        [T, G, A, B], block_size=[1, None, A, B]
    ):
        blk = x[tile_t, tile_g, tile_a, tile_b].to(torch.float32)
        out[tile_t, tile_g] = blk.sum(-1).sum(-1)
    return out


def c3_args(T=4096, G=16, A=32, B=32, dtype=torch.bfloat16, device="cuda"):
    return (torch.randn(T, G, A, B, device=device, dtype=dtype),)


def dump(name, fn, args):
    print(f"\n===== {name} =====")
    bound = fn.bind(args)
    env = bound.env
    spec = env.config_spec
    with env:
        grid_ids = {b for bids in bound.host_function.device_ir.grid_block_ids for b in bids}
        print(f"grid_block_ids={sorted(grid_ids)}")
        kf = spec.reduction_kernel_fact
        if kf is None:
            print("kernel_fact=None")
        else:
            for d in kf.reductions:
                ext = env.block_sizes[d.block_id].size_hint()
                print(f"  desc block_id={d.block_id} cat={d.category.value} "
                      f"graph_id={d.graph_id} extent={ext} pinned={d.pinned} "
                      f"rollable={d.rollable} carried_2d_count={d.carried_2d_count} "
                      f"row_reread={d.row_reread}")
            print(f"  coresidency_groups={[sorted(g.descriptor_indices) for g in kf.coresidency_groups]}")
        for f in spec.reduction_facts:
            print(f"  LEGACY fact: primary={f.primary_reduction_block_id} "
                  f"m_block_ids={f.m_block_ids} size_hint={f.size_hint} "
                  f"full_width_output={f.full_width_output} "
                  f"num_carried_2d={f.num_carried_2d_tiles} "
                  f"secondary={f.secondary_reduction_block_ids}")
        # seed config from whichever heuristic is eligible
        try:
            from helion._compiler.autotuner_heuristics import triton as T
            for cls in (T.TritonStandardReductionHeuristic, T.TritonUserTiledReductionHeuristic):
                if cls.is_eligible(env, bound.host_function.device_ir):
                    cfg = cls.get_seed_config(env, bound.host_function.device_ir)
                    print(f"  SEED[{cls.name}] block_sizes={cfg.block_sizes} "
                          f"reduction_loops={getattr(cfg,'reduction_loops',None)} "
                          f"num_warps={cfg.num_warps}")
        except Exception as e:
            print(f"  seed error: {type(e).__name__}: {e}")
    # also run it for correctness
    out = fn(*args)
    torch.cuda.synchronize()
    print("  compiled+ran OK")
    return out


def main():
    dump("C1 two_fullgrid", c1_two_fullgrid, c1_args())
    dump("C2 fullgrid_plus_bigslice", c2_fullgrid_plus_bigslice, c2_args())
    dump("C3 nested_fullgrid_2axis", c3_nested_fullgrid_2axis, c3_args())


if __name__ == "__main__":
    main()
