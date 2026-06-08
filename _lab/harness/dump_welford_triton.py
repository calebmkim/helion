from __future__ import annotations
import os, sys, io, contextlib
import torch
import helion
WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), helion.__file__
from examples.welford import welford, eager_layer_norm
from helion._compiler.autotuner_heuristics import compiler_seed_configs

M, N = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (262144, 5120)
EPS = 1e-5

weight = torch.rand(N, device="cuda", dtype=torch.float32)
bias = torch.rand(N, device="cuda", dtype=torch.float32)
x = torch.rand(M, N, device="cuda", dtype=torch.float32)
args = (weight, bias, x, EPS)

# --- Helion seeded ---
bound = welford.bind(args)
seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
seed = dict(seeds[0])
print("=" * 80)
print(f"SEED CONFIG for welford ({M},{N}):")
print(seed)
print("=" * 80)
seeded = helion.kernel(welford.fn, configs=[helion.Config(**seed)])
bs = seeded.bind(args)
code = bs.to_triton_code(helion.Config(**seed))
with open(f"/tmp/welford_seed_{M}x{N}.triton.py", "w") as f:
    f.write(code)
print(f"Helion seeded triton -> /tmp/welford_seed_{M}x{N}.triton.py  ({len(code)} bytes)")

# --- torch.compile default ---
os.environ["TORCH_LOGS"] = ""
torch._dynamo.reset()
from torch._inductor import config as ind_cfg
tc = torch.compile(lambda a: eager_layer_norm(*a))
# Capture inductor output code
import logging
out = tc(args)
torch.cuda.synchronize()
print("torch.compile ran. To get its triton, use TORCH_LOGS=output_code env.")
