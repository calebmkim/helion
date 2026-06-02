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
def bw(ms, p): return p * bytes_x / (ms*1e-3) / 1e9

def run_cfg(label, cfg):
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args); b.ensure_config_exists(args)
    out = b(*args)
    ok = bool(torch.allclose(out, ref, rtol=1e-3, atol=1e-4))
    ms = med(lambda: b(*args))
    print(f"{label:34s} {ms*1000:8.1f} us  ok={ok}  ~{bw(ms,3):5.0f} GB/s")

print(f"shape ({M},{N})\n")
for bn in (1024, 2048, 4096):
    for w in (4, 8, 16):
        cfg = {'block_sizes':[1,bn,bn],'load_eviction_policies':['last','first','first','first'],'num_warps':w,'num_stages':1,'pid_type':'flat'}
        try: run_cfg(f"bm=1 bn={bn} w={w}", cfg)
        except Exception as e: print(f"bm=1 bn={bn} w={w}  ERR {type(e).__name__}")
# asymmetric: small reduce block, larger apply block
for rbn,abn in ((1024,2048),(2048,1024),(1024,4096)):
    cfg = {'block_sizes':[1,rbn,abn],'load_eviction_policies':['last','first','first','first'],'num_warps':8,'num_stages':1,'pid_type':'flat'}
    try: run_cfg(f"bm=1 rbn={rbn} abn={abn} w8", cfg)
    except Exception as e: print(f"asym ERR {type(e).__name__}")

torch._dynamo.reset()
tc = torch.compile(lambda a: eager_layer_norm(*a))
tc(args); torch.cuda.synchronize()
ms = med(lambda: tc(args))
print(f"\n{'torch.compile default':34s} {ms*1000:8.1f} us  ~{bw(ms,3):5.0f} GB/s (3-pass)")
