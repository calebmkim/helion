"""Targeted: C2's persistence-undercount claim. The persistence gate in _reduction_rblock checks
ONLY the slice's own resident row (m*extent*itemsize), never the co-resident pinned FULL_GRID tile.
Push N up (and G/GS up) and check: does persistence stay granted while the TRUE resident set
(slice row + [1,G,GS] pinned tile) blows past the byte budget? Also confirm itemsize used."""

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
def c2(x: torch.Tensor, w: torch.Tensor):
    T, G, GS = x.shape
    _, N = w.shape
    hl.specialize(GS); hl.specialize(G)
    o1 = torch.empty([T, G], dtype=torch.float32, device=x.device)
    o2 = torch.empty([T, N], dtype=torch.float32, device=x.device)
    for tile_t, tile_g, tile_c in hl.tile([T, G, GS], block_size=[1, None, GS]):
        o1[tile_t, tile_g] = torch.amax(x[tile_t, tile_g, tile_c].to(torch.float32), -1)
        row = w[tile_t, :].to(torch.float32)
        s = row.sum(-1)
        o2[tile_t, :] = (row / (s[:, None] + 1.0)).to(torch.float32)
    return o1, o2


def probe(name, T, G, GS, N, dtype=torch.bfloat16):
    from helion._compiler.autotuner_heuristics import triton as TT
    print(f"\n===== {name}  T={T} G={G} GS={GS} N={N} dtype={dtype} =====")
    args = (torch.randn(T, G, GS, device="cuda", dtype=dtype),
            torch.randn(T, N, device="cuda", dtype=dtype))
    bound = c2.bind(args)
    env = bound.env
    spec = env.config_spec
    with env:
        fact = spec.reduction_facts[0]
        cls = TT.TritonStandardReductionHeuristic
        m_block = cls._m_block_product(spec, fact)
        rb, pers = cls._reduction_rblock(env, fact, m_block, footprint_factor=fact.body_live_tiles)
        # true resident bytes: persistent slice row [m_block, N] fp32 + pinned [1,G,GS] fp32 amax tile
        slice_bytes = m_block * fact.size_hint * fact.itemsize
        pinned_bytes = G * GS * 4   # fp32 amax tile resident
        print(f"  primary={fact.primary_reduction_block_id} extent={fact.size_hint} itemsize={fact.itemsize} "
              f"m_block={m_block} body_live_tiles={fact.body_live_tiles} row_reread={fact.row_reread}")
        print(f"  -> r_block={rb} PERSISTENT={pers}  ROW_PERSIST_MAX_BYTES={cls.ROW_PERSIST_MAX_BYTES} "
              f"LIVE_PERSIST_BUDGET={cls.LIVE_PERSIST_BUDGET}")
        print(f"  gate slice-only bytes = m*ext*isz = {slice_bytes}  (<= {cls.ROW_PERSIST_MAX_BYTES}? {slice_bytes <= cls.ROW_PERSIST_MAX_BYTES})")
        print(f"  IGNORED co-resident pinned FULL_GRID tile bytes = G*GS*4 = {pinned_bytes}")
        print(f"  TRUE resident (slice + pinned) = {slice_bytes + pinned_bytes}  ratio over LIVE_BUDGET = {(slice_bytes+pinned_bytes)/cls.LIVE_PERSIST_BUDGET:.2f}")
    return pers


def main():
    # shipped extents (claim: does not bite)
    probe("shipped", 8192, 32, 128, 16384)
    # push N high: fp32 promoted persistent row; itemsize is the INPUT (bf16=2). 245760/2 = 122880 elems max.
    probe("N=49152 (claim N where it bites)", 8192, 32, 128, 49152)
    probe("N=122880 just under cap (bf16 isz=2)", 8192, 32, 128, 122880)
    # large pinned FULL_GRID tile alongside a near-cap slice
    probe("N=49152 + BIG pinned G=128 GS=512", 8192, 128, 512, 49152)
    probe("N=98304 + BIG pinned G=128 GS=1024", 4096, 128, 1024, 98304)
    # ADVERSARIAL: slice at ROW cap (N=61440 fp32 -> 245760 B exactly) + HUGE pinned tile
    probe("N=61440(cap) + HUGE pinned G=256 GS=2048", 4096, 256, 2048, 61440)
    probe("N=61440(cap) + HUGE pinned G=512 GS=4096", 4096, 512, 4096, 61440)


if __name__ == "__main__":
    main()
