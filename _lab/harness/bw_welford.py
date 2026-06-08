from __future__ import annotations
import sys
import torch
import helion
WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), helion.__file__
from examples.welford import welford, eager_layer_norm
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from triton.testing import do_bench

M, N = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (262144, 5120)
EPS = 1e-5
weight = torch.rand(N, device="cuda", dtype=torch.float32)
bias = torch.rand(N, device="cuda", dtype=torch.float32)
x = torch.rand(M, N, device="cuda", dtype=torch.float32)
args = (weight, bias, x, EPS)
ref = eager_layer_norm(*args)

bytes_x = M * N * 4

def med(fn, reps=7):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(reps))[reps // 2]

def bw(ms, passes):
    # GB/s assuming `passes` * (M*N*4) bytes moved
    return passes * bytes_x / (ms * 1e-3) / 1e9

def run_cfg(label, cfg, x_passes):
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    out = b(*args)
    ok = bool(torch.allclose(out, ref, rtol=1e-3, atol=1e-4))
    ms = med(lambda: b(*args))
    # 2 reads of x (reduce+apply) + 1 write
    print(f"{label:30s} {ms*1000:8.1f} us  ok={ok}  ~{bw(ms,3):6.0f} GB/s (3-pass model)  cfg_bn={cfg['block_sizes']}")
    return ms

bound = welford.bind(args)
seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
print(f"shape ({M},{N})  x={bytes_x/1e9:.2f} GB per pass\n")

run_cfg("SEED (bn1=8192 persistent)", seed, 3)
# streaming variants
for bm in (1, 4, 16, 32):
    for bn in (256, 512, 1024):
        cfg = {'block_sizes': [bm, bn, bn], 'load_eviction_policies': ['last','first','first','first'], 'num_warps': 8, 'num_stages': 1, 'pid_type': 'flat'}
        try:
            run_cfg(f"stream bm={bm} bn={bn} w8", cfg, 3)
        except Exception as e:
            print(f"stream bm={bm} bn={bn} w8  ERR {type(e).__name__}")

# torch.compile
torch._dynamo.reset()
tc = torch.compile(lambda a: eager_layer_norm(*a))
out = tc(args); torch.cuda.synchronize()
ms = med(lambda: tc(args))
print(f"\n{'torch.compile default':30s} {ms*1000:8.1f} us  ~{bw(ms,3):6.0f} GB/s (3-pass model) / ~{bw(ms,4):6.0f} (4-pass)")
