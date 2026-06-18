"""AUDITOR: confirm the heuristic emits a Band-B capped seed (R_BLOCK<=4096) for
held-out kl_div/jsd shapes, and that the cap fires due to num_tiled_accumulators.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402


def emit_kl(BT, V):
    yp = torch.randn(BT, V, device="cuda", dtype=torch.float32).log_softmax(-1)
    yt = torch.randn(BT, V, device="cuda", dtype=torch.float32).softmax(-1)
    b = kl_div_forward.bind((yp, yt, False, "batchmean", 1e-10))
    f = b.env.config_spec.reduction_facts[0]
    s = dict(compiler_seed_configs(b.env, b.host_function.device_ir)[0])
    print(f"  kl_div ({BT},{V}): ntiled={f.num_tiled_accumulators} "
          f"size_hint={f.size_hint} -> seed block_sizes={s['block_sizes']} "
          f"w={s['num_warps']}  (R_BLOCK<=4096? {min(s['block_sizes'])==1 and max(s['block_sizes'])<=4096})")


def emit_jsd(BT, V):
    lq = torch.randn(BT, V, device="cuda", dtype=torch.float32).log_softmax(-1)
    lp = torch.randn(BT, V, device="cuda", dtype=torch.float32).log_softmax(-1)
    b = jsd_forward.bind((lq, lp, None, 0.5, -100))
    f = b.env.config_spec.reduction_facts[0]
    s = dict(compiler_seed_configs(b.env, b.host_function.device_ir)[0])
    print(f"  jsd    ({BT},{V}): ntiled={f.num_tiled_accumulators} "
          f"size_hint={f.size_hint} -> seed block_sizes={s['block_sizes']} "
          f"w={s['num_warps']}  (R_BLOCK<=4096? {min(s['block_sizes'])==1 and max(s['block_sizes'])<=4096})")


def main():
    print("HELD-OUT seed emission (Band-B cap should fire):")
    for (m, n) in [(4096, 32000), (8192, 65536), (2048, 128256), (1024, 256000)]:
        emit_kl(m, n)
    for (m, n) in [(4096, 128256), (4096, 129280), (2048, 151936), (1024, 256000)]:
        emit_jsd(m, n)


if __name__ == "__main__":
    main()
