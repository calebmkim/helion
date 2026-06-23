"""Gate D 4th-pass: confirm the ACTUAL lab corpus kernels (relu_squared/bias_gelu/dyt, all
bind-once) give the CORRECT traffic byte count despite using the bound value many times
(_gelu_tanh does v*v*v). Then contrast with the inline double-count. Compile-only."""
from __future__ import annotations
import torch
import helion
import helion.language as hl
import sys
sys.path.insert(0, "/home/calebkim/helion-new-heuristics/helion-pointwise/_lab/pointwise")
import ptw_kernels as K
from helion._compiler.autotuner_heuristics import compiler_seed_configs

assert helion.__file__.startswith(
    "/home/calebkim/helion-new-heuristics/helion-pointwise/"
), helion.__file__


def dump(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    f = spec.pointwise_facts[0]
    s = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    print("%-20s bytes/elem=%-2d n_load=%d n_store=%d seed=%s" % (
        name, f.bytes_per_elem, f.n_load, f.n_store,
        dict(s[0].config).get("block_sizes") if s else None))


def main():
    M, N = 4096, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    print("--- ACTUAL lab corpus (bind-once idiom) ---")
    dump("relu_squared", K.relu_squared, (x,))
    dump("bias_gelu", K.bias_gelu, (x, bias))
    dump("dyt", K.dyt, (x, gamma, beta, 1.0))
    print("  (all traffic-2 = bytes/elem 4; gelu_tanh's v*v*v does NOT add loads)")


if __name__ == "__main__":
    main()
