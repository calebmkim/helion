"""Targeted: does the M-widen cap on the tunable group-row (G/H) of a multi-axis / multi-FULL_GRID
kernel actually SEE the pinned inner extents (GS/HS/A/B)? Push the pinned extents up and watch the
widened block_sizes + the binding cap. If the widen is unbounded by the pinned footprint, the
group row widens regardless of pinned size -> the claimed under-count is real."""

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


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def c3_nested_fullgrid_2axis(x: torch.Tensor):
    T, G, A, B = x.shape
    hl.specialize(A); hl.specialize(B); hl.specialize(G)
    out = torch.empty([T, G], dtype=torch.float32, device=x.device)
    for tile_t, tile_g, tile_a, tile_b in hl.tile([T, G, A, B], block_size=[1, None, A, B]):
        blk = x[tile_t, tile_g, tile_a, tile_b].to(torch.float32)
        out[tile_t, tile_g] = blk.sum(-1).sum(-1)
    return out


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def c1_two_fullgrid(x: torch.Tensor, y: torch.Tensor):
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


def probe(name, fn, args):
    from helion._compiler.autotuner_heuristics import triton as TT
    print(f"\n===== {name} =====")
    bound = fn.bind(args)
    env = bound.env
    spec = env.config_spec
    with env:
        kf = spec.reduction_kernel_fact
        for d in kf.reductions:
            print(f"  desc bid={d.block_id} cat={d.category.value} extent={env.block_sizes[d.block_id].size_hint()} pinned={d.pinned}")
        fact = spec.reduction_facts[0]
        print(f"  primary={fact.primary_reduction_block_id} m_block_ids={fact.m_block_ids} secondary={fact.secondary_reduction_block_ids}")
        cls = TT.TritonStandardReductionHeuristic
        # inner pinned count for each tunable m_block axis
        for mbid in fact.m_block_ids:
            if mbid in spec.block_sizes.valid_block_ids():
                inner = cls._pinned_inner_resident_elems(spec, fact, mbid)
                rtc = cls._resident_tile_cap(spec, fact, inner, r_block_resident=1)
                print(f"  m_block axis bid={mbid} extent={env.block_sizes[mbid].size_hint()} "
                      f"pinned_inner={inner} resident_tile_cap(r_res=1)={rtc}")
        cfg = cls.get_seed_config(env, bound.host_function.device_ir)
        print(f"  SEED block_sizes={cfg.block_sizes} reduction_loops={getattr(cfg,'reduction_loops',None)} num_warps={cfg.num_warps}")


def main():
    # C3 baseline 32x32, then BIG pinned 128x128 (claim: widen unchecked by A*B)
    probe("C3 A=32 B=32 G=16", c3_nested_fullgrid_2axis,
          (torch.randn(4096, 16, 32, 32, device="cuda", dtype=torch.bfloat16),))
    probe("C3 A=128 B=128 G=64", c3_nested_fullgrid_2axis,
          (torch.randn(4096, 64, 128, 128, device="cuda", dtype=torch.bfloat16),))
    # C1 baseline, then BIG GS/HS
    probe("C1 GS=128 HS=256 G=32 H=16", c1_two_fullgrid,
          (torch.randn(8192, 32, 128, device="cuda", dtype=torch.bfloat16),
           torch.randn(8192, 16, 256, device="cuda", dtype=torch.bfloat16)))
    probe("C1 GS=512 HS=1024 G=64 H=64", c1_two_fullgrid,
          (torch.randn(8192, 64, 512, device="cuda", dtype=torch.bfloat16),
           torch.randn(8192, 64, 1024, device="cuda", dtype=torch.bfloat16)))


if __name__ == "__main__":
    main()
