import torch
import helion
import helion.language as hl


# ARITHMETIC-INTENSITY DISCRIMINATOR.
#
# Structure is FROZEN to a two-pass softmax_two_pass isomorph (row_reread=True,
# carried_2d_count=0, full_width_output=True, TWO physical x loads, scale=2 footprint)
# so it is byte-for-byte and pass-for-pass identical to the measured softmax baseline.
# The ONLY knob is COMPUTE PER ELEMENT over the row: a compile-time ``HEAVY`` flag adds
# extra transcendental (exp/log) ops per element in BOTH passes WITHOUT changing:
#   - resident bytes  (same tiles live, same scale=2/flat footprint),
#   - #physical loads  (still two rolled loops over x),
#   - the output shape (still a full-width [m,n] store),
#   - the reduction structure (still amax+online-denom in pass1, normalize+store in pass2).
#
# LENS PREDICTION: persist's ONLY benefit is saving one HBM read of the row. When
# per-element ALU/SFU work is HIGH, the saved read is hidden behind compute (compute-bound),
# so persist's benefit erodes -> the persist->chunk cutoff moves to a SMALLER N. When
# per-element work is LOW (LEAN), the kernel is memory-bound, the saved read is the whole
# story, and persist stays winning to a LARGER N. => at a fixed intermediate N chosen at the
# LEAN cutoff, HEAVY should already prefer CHUNK while LEAN still prefers PERSIST.
#
# This isolates FLOPs/byte from the #passes/reuse-distance confound (both HEAVY and LEAN
# re-read twice) and from the byte-footprint (identical). If the A/B verdict is INSENSITIVE
# to HEAVY at fixed N, arithmetic intensity is NOT the governing quantity (some structural /
# occupancy lens is) and this hypothesis is falsified.

HEAVY = True  # flip to False for the LEAN arm of the A/B


@helion.kernel(static_shapes=False)
def kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=bm):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        # PASS 1: online-softmax stats (amax + running denom). One x load, one exp over row.
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            if HEAVY:
                # Extra per-element SFU work that FEEDS the same reduction (does not change
                # which reduction the load feeds, only the FLOPs/byte). ~3 extra exp/element.
                v = torch.exp(torch.exp(torch.exp(v * 0.1) * 0.1) * 0.1) * 10.0
            a = torch.amax(v, dim=1)
            mn = torch.maximum(mi, a)
            di = di * torch.exp(mi - mn) + torch.exp(v - mn[:, None]).sum(dim=1)
            mi = mn
        # PASS 2: SEPARATE loop RE-READS x, normalizes, stores the full-width row.
        for tile_n in hl.tile(n, block_size=bn):
            v2 = x[tile_m, tile_n]
            if HEAVY:
                v2 = torch.exp(torch.exp(torch.exp(v2 * 0.1) * 0.1) * 0.1) * 10.0
            p = torch.exp(v2 - mi[:, None]) / di[:, None]
            out[tile_m, tile_n] = p
    return out


def make_args():
    # N=40960: chosen BETWEEN the two measured softmax cutoff endpoints (32768 persist +30%,
    # 49152 chunk +34%) — i.e. straddling the LEAN cutoff. Row bytes = 40960*4 = 160 KiB,
    # footprint scale=2 => ~320 KiB, below the 256KB reg-file spill wall (so any HEAVY-vs-LEAN
    # flip here is COMPUTE-driven, not the ce spill cliff). Prediction: LEAN persists (near the
    # memory-bound softmax N<=32768 regime), HEAVY chunks (compute pushes the cutoff below N).
    return (torch.randn(2048, 40960, device="cuda", dtype=torch.float32),)
