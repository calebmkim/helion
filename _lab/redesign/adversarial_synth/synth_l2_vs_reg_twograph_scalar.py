import torch

import helion
import helion.language as hl


# LENS: l2-vs-register-residency.
#
# The persist budget is a DIFFERENT physical resource depending on whether the reduction row is
# reused WITHIN ONE graph (registers/SRAM, ~256 KB CTA-local reg file) or RE-READ ACROSS TWO graphs
# (an L2 round-trip: budget = L2_capacity / concurrent_CTAs, ~160 KB per CTA on H100).
#
# This kernel ISOLATES the graph-count variable while nailing everything else that the competing
# lenses key on to cross_entropy's values:
#   - OUTPUT IS SCALAR [m]  (NOT full-width [m,N]) -> kills the "full-width-output store" lens: no
#     [m,N] store tile resident, no store-bound occupancy. Same output shape as cross_entropy.
#   - SAME #reductions as cross_entropy: exactly 2 (amax over the row, then sum-exp over the row).
#     Kills the "#passes / maxfork" lens (ce also has 2).
#   - SAME dtype (fp32), SAME footprint form (scale=2: the [m,R] read tile + rank-1 carries).
#
# The ONE thing that differs from cross_entropy: the max and the sum-exp are written as TWO SEPARATE
# `hl.tile(n)` loops => TWO DISTINCT graph_ids, so x is LOADED TWICE (an L2 round-trip). cross_entropy
# does both reductions on ONE `logits[tile_n, :]` full-slice load = ONE graph_id (register reuse).
#
# SEED-TIME DISCRIMINATOR: n_graphs = |{f.graph_id : f.kind=='load', f.tensor_name==row_tensor}|.
#   this kernel -> 2 (L2 budget) ;  cross_entropy -> 1 (register budget).
# NOTE: because BOTH loops reduce (neither is store-only), today's `_apply_reread` returns FALSE here
# -> it hands this kernel the BIG (737280) ceiling and PERSISTS it. My lens predicts that is WRONG at
# the shape below: the L2 round-trip makes persist LOSE well before the register-file limit.
@helion.kernel(static_shapes=False)
def kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=bm):
        # PASS 1 (graph G1): re-read-able x load feeding ONLY amax -> the running row max.
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            mi = torch.maximum(mi, torch.amax(v, dim=1))
        # PASS 2 (graph G2, SEPARATE loop -> DISTINCT graph_id): re-read x, feed ONLY sum-exp.
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=bn):
            v2 = x[tile_m, tile_n]
            di += torch.exp(v2 - mi[:, None]).sum(dim=1)
        # SCALAR output [m] (log-sum-exp), exactly cross_entropy's output rank — NOT a full-width store.
        out[tile_m] = mi + torch.log(di)
    return out


def make_args():
    # N=49152 -> row = 49152*4 = 196608 B = 192 KB.
    #   ABOVE softmax's measured L2 cutoff (~160 KB, chunk wins at 49152) and
    #   BELOW cross_entropy's register-file spill cutoff (256 KB, persist still wins at 65536).
    # MY LENS predicts CHUNK wins here (two-graph L2 round-trip -> L2/concurrency budget ~160 KB).
    # The full-width-output lens predicts PERSIST wins (scalar out, register budget, 192 KB < 256 KB).
    return (torch.randn(2048, 49152, device="cuda", dtype=torch.float32),)
