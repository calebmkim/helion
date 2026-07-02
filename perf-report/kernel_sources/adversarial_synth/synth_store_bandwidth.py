import torch
import helion
import helion.language as hl

# LENS: output-store bandwidth. Two identical 2-pass reductions over x (same loads, same
# row-reread, same PASS-1 reductions). The ONLY difference is the PASS-2 OUTPUT WIDTH:
#   FULL_OUTPUT=True  -> store full [m,N] normalized probs (a full-width write EVERY pass; this is
#                        the softmax structure). The write bandwidth EQUALS the read the persist is
#                        trying to save -> persist's NET benefit ~= 0 -> should CHUNK.
#   FULL_OUTPUT=False -> reduce the SAME probs to a scalar and store [m] (the cross_entropy
#                        structure). The write is negligible -> persist's read-saving is pure
#                        profit -> should PERSIST (row still fits the register file at this N).
# Everything else (row extent, reread, footprint scale=2, #passes, arithmetic) is held constant, so
# the persist-vs-chunk A/B FLIP between the two variants isolates OUTPUT-STORE WIDTH as the governing
# quantity. Flip the flag to run each side.

FULL_OUTPUT = True


@helion.kernel(static_shapes=False)
def kernel(x: torch.Tensor):
    m, n = x.size()
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    if FULL_OUTPUT:
        out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    else:
        out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m, block_size=bm):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        # PASS 1: online-softmax stats (amax + running denom) — one x load, two reductions.
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            a = torch.amax(v, dim=1)
            mn = torch.maximum(mi, a)
            di = di * torch.exp(mi - mn) + torch.exp(v - mn[:, None]).sum(dim=1)
            mi = mn
        # PASS 2: SEPARATE loop re-reads x (the refetch persist avoids) and builds probs p.
        if FULL_OUTPUT:
            for tile_n in hl.tile(n, block_size=bn):
                v2 = x[tile_m, tile_n]
                p = torch.exp(v2 - mi[:, None]) / di[:, None]
                out[tile_m, tile_n] = p  # FULL-WIDTH store: write == saved read
        else:
            ent = hl.zeros([tile_m], dtype=torch.float32)
            for tile_n in hl.tile(n, block_size=bn):
                v2 = x[tile_m, tile_n]
                p = torch.exp(v2 - mi[:, None]) / di[:, None]
                ent += -(p * torch.log(p + 1e-9)).sum(dim=1)
            out[tile_m] = ent  # SCALAR store: write negligible
    return out


def make_args():
    # N=49152: full-width row = 192KB (past softmax's persist->chunk flip); scalar row = 192KB but
    # holds only the read row, still under cross_entropy's ~256KB persist ceiling. This is where the
    # two variants MAXIMALLY diverge: FULL_OUTPUT should CHUNK-win, scalar should PERSIST-win.
    return (torch.randn(2048, 49152, device="cuda", dtype=torch.float32),)
