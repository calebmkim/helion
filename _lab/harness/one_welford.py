from __future__ import annotations
import sys
import torch
import helion
WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), helion.__file__
from examples.welford import welford, eager_layer_norm

bm, bn, w = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
M, N = 16384, 5120
EPS = 1e-5
weight = torch.rand(N, device="cuda", dtype=torch.float32)
bias = torch.rand(N, device="cuda", dtype=torch.float32)
x = torch.rand(M, N, device="cuda", dtype=torch.float32)
args = (weight, bias, x, EPS)
cfg = dict(block_sizes=[bm, bn, bn],
           load_eviction_policies=['last', 'first', 'first', 'first'],
           num_warps=w, num_stages=1, pid_type='flat')
k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
b = k.bind(args)
b.ensure_config_exists(args)
for _ in range(5):           # warmup; ncu will skip these
    b(*args)
torch.cuda.synchronize()
b(*args)                     # the one ncu captures
torch.cuda.synchronize()
