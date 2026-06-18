"""TASK 4: prove welford small-N (apply persistent) is BYTE-IDENTICAL to v7, and ONLY
the wide N=4096 shape changes. Compares the v8 heuristic's emitted seed vs the recorded
v7 ledger seed per shape.

v7 ledger seeds (ledger.json welford.seed):
  (262144,1024) -> [16,1024,1024] w4
  (262144,1536) -> [16,512,2048]  w8
  (262144,2048) -> [16,2048,2048] w8
  (262144,4096) -> [16,2048,4096] w8   (v8 CHANGES this to [16,4096,2048])
"""
from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.welford import welford  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
V7 = {
    (262144, 1024): {"block_sizes": [16, 1024, 1024], "num_warps": 4, "num_stages": 1},
    (262144, 1536): {"block_sizes": [16, 512, 2048], "num_warps": 8, "num_stages": 1},
    (262144, 2048): {"block_sizes": [16, 2048, 2048], "num_warps": 8, "num_stages": 1},
    (262144, 4096): {"block_sizes": [16, 2048, 4096], "num_warps": 8, "num_stages": 1},
}


def args(m, n):
    return (
        torch.rand(n, device="cuda"),
        torch.rand(n, device="cuda"),
        torch.rand(m, n, device="cuda"),
        EPS,
    )


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    for (m, n), v7 in V7.items():
        bound = welford.bind(args(m, n))
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        v8 = dict(seeds[0])
        v8c = {k: v8.get(k) for k in ("block_sizes", "num_warps", "num_stages")}
        identical = v8c == v7
        verdict = "BYTE-IDENTICAL" if identical else "CHANGED (expected only at N=4096)"
        print(f"({m},{n}): v7={v7['block_sizes']} v8={v8c['block_sizes']} "
              f"w{v8c['num_warps']} -> {verdict}", flush=True)
    print("\nExpected: 1024/1536/2048 BYTE-IDENTICAL (apply persistent, N<=2048<=cap);"
          " ONLY 4096 changes (apply looped + combine uncapped).", flush=True)


if __name__ == "__main__":
    main()
