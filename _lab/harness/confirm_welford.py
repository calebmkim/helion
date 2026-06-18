from __future__ import annotations
import sys
import torch
import helion
WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), helion.__file__
from examples.welford import welford, eager_layer_norm
from triton.testing import do_bench

M, N = int(sys.argv[1]), int(sys.argv[2])
EPS = 1e-5
weight = torch.rand(N, device="cuda", dtype=torch.float32)
bias = torch.rand(N, device="cuda", dtype=torch.float32)
x = torch.rand(M, N, device="cuda", dtype=torch.float32)
args = (weight, bias, x, EPS)
ref = eager_layer_norm(*args)

def med(fn, reps=7):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(reps))[reps // 2]

def run(label, bm, cbn, abn, w):  # combine-block, apply-block separate
    cfg = dict(block_sizes=[bm, cbn, abn],
               load_eviction_policies=['last', 'first', 'first', 'first'],
               num_warps=w, num_stages=1, pid_type='flat')
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args); b.ensure_config_exists(args)
    out = b(*args)
    ok = bool(torch.allclose(out, ref, rtol=1e-3, atol=1e-4))
    print(f"{label:44s} {med(lambda: b(*args))*1000:8.1f} us  ok={ok}  cfg=[{bm},{cbn},{abn}] w{w}")

print(f"\n==== welford ({M},{N}) ====")
run("A REAL SEED [16,8192,2048] w16", 16, 8192, 2048, 16)
run("B spill-removed (combine 8192->2048)", 16, 2048, 2048, 16)
run("C block_m only (16->1, seed combine/apply)", 1, 8192, 2048, 16)
run("D bm=1 + combine 2048", 1, 2048, 2048, 16)
run("E bm=1 + w16->w4 (BEST)", 1, 2048, 2048, 4)
torch._dynamo.reset()
tc = torch.compile(lambda a: eager_layer_norm(*a)); tc(args); torch.cuda.synchronize()
print(f"{'F torch.compile default':44s} {med(lambda: tc(args))*1000:8.1f} us")
