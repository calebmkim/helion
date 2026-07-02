import torch
import helion
import helion.language as hl


# Candidate #2: two physical passes over x (softmax_two_pass isomorph), but PASS 2 REDUCES the
# normalized probs (entropy) instead of storing them -> pass-2 load is (R,-), evades _apply_reread
# -> BIG ceiling. But it genuinely re-reads the row (refetch cliff) -> should want SMALL.
@helion.kernel(static_shapes=False)
def kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=bm):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        # PASS 1: online softmax stats (amax + running denom) — one x load forks to two reductions.
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            a = torch.amax(v, dim=1)
            mn = torch.maximum(mi, a)
            di = di * torch.exp(mi - mn) + torch.exp(v - mn[:, None]).sum(dim=1)
            mi = mn
        # PASS 2: SEPARATE loop re-reads x, reduces the normalized probs into an entropy scalar.
        ent = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=bn):
            v2 = x[tile_m, tile_n]
            p = torch.exp(v2 - mi[:, None]) / di[:, None]
            ent += -(p * torch.log(p + 1e-9)).sum(dim=1)
        out[tile_m] = ent
    return out


def make_args():
    # N=49152: the shape where softmax measurably prefers CHUNK (persist -34%). fp32.
    return (torch.randn(2048, 49152, device="cuda", dtype=torch.float32),)
