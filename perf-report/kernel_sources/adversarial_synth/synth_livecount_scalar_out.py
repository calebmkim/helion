import torch
import helion
import helion.language as hl


# LENS: register-spill-working-set (body_live_tiles).
#
# CLAIM: the persist-vs-chunk cutoff is a single working-set ceiling ~= the 256KB register
# file, applied to  itemsize * body_live_tiles * raw_ext  -- NOT itemsize * scale * raw_ext.
# body_live_tiles = peak count of simultaneously-live rdim-shaped [m,N] tiles in the reduction
# body (already computed in device_ir._graph_peak_live_by_axis, stored on the descriptor, but
# NEVER read by the ext_held gate). softmax has body_live_tiles>=2 (probs/exp numerator live
# WITH the row + a full-width output), cross_entropy has body_live_tiles==1 (reduces to a scalar
# as it goes). That is why softmax flips at ~2x-SMALLER N despite an identical scale=2 footprint.
#
# THIS KERNEL ISOLATES body_live_tiles. It is scalar-output (NOT full_width_output -- kills the
# "store-traffic / full-width output" lens), single logical reduction target, and re-reads the
# row (row_reread True, matching cross_entropy's structure -- kills the "#passes" and
# "_apply_reread L2" lenses as the discriminator). The ONLY thing that differs from a
# cross_entropy-shaped baseline is that the body holds THREE simultaneously-live full-width
# [m,N] intermediates (t0,t1,t2 all live at the reduce step) instead of one. The co-residency
# footprint scale is BLIND to this (it sums per-shape tiles at the roller peak and calls it
# scale~2 for everything), so the current gate treats this exactly like cross_entropy and will
# PERSIST. My lens predicts it must CHUNK: WS = 4 * 3 * 32768 = 384KB > ~288KB ceiling.
@helion.kernel(static_shapes=False)
def kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=bm):
        # PASS 1: a plain amax over the row (one fused reduction, scalar carry [m]).
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            mi = torch.maximum(mi, torch.amax(v, dim=1))
        # PASS 2: RE-READ the row (row_reread=True) and reduce to a SCALAR [m]. The body holds
        # THREE full-width [m,N] intermediates SIMULTANEOUSLY LIVE at the reduction step: each of
        # t0,t1,t2 is used AFTER all three are defined, so a liveness sweep counts 3 co-live
        # rdim-shaped tiles -> body_live_tiles == 3. None is a full-width STORE (out is [m]).
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=bn):
            v2 = x[tile_m, tile_n]
            z = v2 - mi[:, None]
            t0 = torch.exp(z)
            t1 = z * z
            t2 = torch.sin(z)
            # all three co-live here, then collapse to a scalar (no full-width output tile):
            acc += (t0 + t1 - t2).sum(dim=1)
        out[tile_m] = acc
    return out


def make_args():
    # DISCRIMINATING SHAPE N=32768: cross_entropy (body_live_tiles=1) PERSIST-wins here and
    # softmax (blt=2) also still PERSIST-wins here (+30%). NONE of the competing lenses predict a
    # flip at this N for a scalar-output single-target kernel. My lens does: blt=3 -> WS=384KB >
    # ceiling -> CHUNK must WIN. fp32.
    return (torch.randn(2048, 32768, device="cuda", dtype=torch.float32),)
