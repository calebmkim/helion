import torch

import helion
import helion.language as hl


# ============================================================================
# LENS: working-set-undercount.
#
# WHY THE FOOTPRINT IS BLIND (the exact code path):
# _group_live_tiles (device_ir.py) combines a group's owned graphs by
# MAX-BY-PROFILE, NEVER sum (rule 4), over the peak-live step WITHIN each graph
# (_graph_peak_live_tiles). softmax_two_pass has its reduction (amax+sumexp) in
# for-loop body #1 and its output BUILD+STORE in for-loop body #2 -- two distinct
# ForLoopGraphInfo graphs driven by the same co-residency group. The footprint is
# therefore MAX(peak(pass1), peak(pass2)), NOT their sum, and neither pass's peak
# includes the OTHER pass's full-N tiles. But PERSIST's whole reason to exist is
# to hold the row x[m,N] resident ACROSS both passes so it is read from HBM once
# (VERIFIED: persistent softmax emits ONE tl.load, chunk emits TWO). So at the
# moment persist is deciding fit, the SIMULTANEOUSLY-RESIDENT set is:
#     pass-1's persisted row x[m,N]  +  pass-2's built output out[m,N]  +  probs.
# The MAX-combine sees each pass at scale 2 and reports scale 2, missing the +1
# full-N output tile that only exists because full_width_output=True. Result:
# softmax's TRUE resident set is ~3 full-N tiles while the footprint says 2.
#
# cross_entropy is ONE fused loop reducing to a SCALAR [m]: no second pass, no
# full-N output tile. Its scale=2 (row + register accumulator) is FAITHFUL. Same
# footprint FORM (both scale 2), different TRUTH -> softmax's row-bytes budget
# should be ~1.5x tighter, which is exactly the measured paradox (softmax flips
# at row bytes ~128-192KB; cross_entropy persists to ~256-384KB).
#
# ISOLATION. The ONE variable is the presence of the uncounted cross-pass full-N
# OUTPUT tile (full_width_output). Everything else is pinned to a softmax/ce
# midpoint:
#   * raw row bytes (N * 4)               CONSTANT across the two arms
#   * reduction extent + #reductions (2)  CONSTANT (amax + sumexp, pass 1)
#   * row_reread = True                   CONSTANT (pass 2 reloads x)
#   * _apply_reread -> SMALL hold_ceiling  CONSTANT (pass-2 load feeds a store,
#                                                     no reduction)
#   * two physical passes / two graphs    CONSTANT (both arms two-pass)
#   * carried_2d_count = 0                CONSTANT (only scalar mi,di carried)
#   * footprint scale AS THE ALLOCATOR COMPUTES IT = 2 in BOTH arms, because
#     pass 2 holds only { v2[m,r], probs[m,r] } at its peak (scale 2) exactly
#     like pass 1, and MAX(2,2)=2 -- the extra resident tile is the PERSISTED
#     ROW from the OTHER graph, which MAX-combine never adds.
#
# The two arms differ ONLY in the sink of pass 2:
#   ARM S (full_width_output=True, softmax-like): stores probs[m,N] -> the built
#     output tile is co-resident with the persisted row -> TRUE scale 3.
#   ARM C (scalar out, cross_entropy-like):        reduces probs to a scalar and
#     stores out[m] -> no full-N output tile -> TRUE scale 2.
# This file is ARM S; flip the two marked lines for ARM C. (The orchestrator runs
# one kernel/process; the A/B is S-persist-vs-chunk THEN C-persist-vs-chunk.)
#
# PREDICTION IF TRUE: ARM S flips to CHUNK at a smaller N than ARM C, and ARM S's
# flip N matches softmax's (~49152 / row bytes ~192KB) while ARM C's matches
# cross_entropy's (~higher N). Because the allocator footprint is scale=2 for
# BOTH, a single byte budget cannot reproduce this -- only a signal that counts
# the cross-pass full-N output tile can. At the shipped shape (N=40960, row bytes
# 160KB) ARM S should already prefer CHUNK (its true 3x160=480KB row-triple busts
# the 256KB register file -> spill, VERIFIED as the softmax loss mechanism at
# N>=49152) while ARM C should still PERSIST (its true 2x160=320KB... note ce
# persists to 384KB row bytes i.e. 768KB triple? no: ce true scale 2 -> 320KB,
# fits the read-once win). Rough: ARM S persist LOSES ~15-34%; ARM C persist WINS
# ~+7 to +30%.
#
# PREDICTION IF FALSE: if the cutoff is set by refetch bytes / #passes / raw row
# bytes / occupancy (competing lenses), ARM S and ARM C flip at the SAME N,
# because full_width_output changes none of those -- both re-read x, both are
# two-pass, both hold the same row bytes. Same flip N across arms falsifies this
# lens.
# ============================================================================


@helion.kernel(static_shapes=False)
def kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    # ARM S (this file): full-width output, softmax-like. full_width_output=True.
    out = torch.empty_like(x)  # ARM C: torch.empty([m], dtype=..., device=...)
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=bm):
        # PASS 1: online softmax stats -> two SCALAR carries mi,di [m]. Peak-live
        # step is { v[m,r], exp(v-mn)[m,r] } -> scale 2. Identical in both arms.
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            a = torch.amax(v, dim=1)
            mn = torch.maximum(mi, a)
            di = di * torch.exp(mi - mn) + torch.exp(v - mn[:, None]).sum(dim=1)
            mi = mn
        # PASS 2: SEPARATE loop, re-reads x (row_reread=True). Peak-live step is
        # { v2[m,r], probs[m,r] } -> scale 2, SAME as pass 1, SAME as ARM C's
        # pass 2. The ONLY difference is the sink below.
        # ARM C body (flip): acc = hl.zeros([tile_m]); ... acc += probs.sum(1);
        #                    out[tile_m] = acc   (scalar store, no full-N output).
        for tile_n in hl.tile(n, block_size=bn):
            v2 = x[tile_m, tile_n]
            p = torch.exp(v2 - mi[:, None]) / di[:, None]
            out[tile_m, tile_n] = p  # ARM S: full-width store (uncounted full-N tile)
    return out


def make_args():
    # N=40960 -> row bytes 40960*4 = 160KB (between softmax's 128KB-persist-wins
    # and 192KB-chunk-wins boundary). ARM S's TRUE resident triple ~480KB busts
    # the 256KB register file at this N (persist should already LOSE); ARM C's
    # true double ~320KB is the read-once win (persist should WIN). m=2048 gives a
    # wide grid so program-count occupancy is NOT the lever under test. fp32.
    return (torch.randn(2048, 40960, device="cuda", dtype=torch.float32),)
