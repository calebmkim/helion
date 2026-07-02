import torch
import helion
import helion.language as hl


# Candidate #1: two physical passes over x. Pass 1 = mean (fork sum + count on axis N).
# Pass 2 = re-read x, reduce sum of squared deviations. Both loads (R,-) -> evades _apply_reread.
@helion.kernel(static_shapes=False)
def kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=bm):
        s = hl.zeros([tile_m], dtype=torch.float32)
        cnt = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            s += v.sum(dim=1)
            cnt += torch.ones_like(v).sum(dim=1)
        mean = s / cnt
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=bn):
            v2 = x[tile_m, tile_n]
            d = v2 - mean[:, None]
            acc += (d * d).sum(dim=1)
        out[tile_m] = acc / cnt
    return out


def make_args():
    return (torch.randn(2048, 49152, device="cuda", dtype=torch.float32),)
