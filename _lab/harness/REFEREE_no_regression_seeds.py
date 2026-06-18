"""Referee no-regression spot-check: emit the heuristic seed for rms_norm/sum/
long_sum on representative in-sample shapes and assert the heuristic still fires
(exactly 1 seed, expected persistent/looped + num_warps per the documented ramp).

This is belt-and-suspenders on top of the git-level byte-identical proof
(`git diff 37b27f67 HEAD -- helion/` is empty).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.rms_norm import rms_norm_fwd  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
import helion.language as hl  # noqa: E402


@helion.kernel
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].to(torch.float32).sum(dim=-1)
    return out


# long_sum: same op, but the "long" curriculum just uses very wide rows.
long_sum = sum_kernel


def emit(fn, args, label):
    bound = fn.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    names = bound.env.config_spec.autotuner_heuristics
    fact = bound.env.config_spec.reduction_facts[0]
    seed = dict(seeds[0]) if seeds else None
    print(f"{label:>28}  n_seeds={len(seeds)}  heuristics={names}")
    print(f"{'':>28}  rnumel={fact.size_hint}  seed={seed}")
    return len(seeds), seed, fact.size_hint


def expected_warps(rnumel):
    if rnumel > 16384:
        return 32
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    return 16


def main():
    print("=== Referee no-regression: other-3-kernel seed emission ===\n")
    all_ok = True

    # rms_norm representative shapes (persistent low-N, persistent w32 high-N)
    for (m, n) in [(2048, 1024), (2048, 4096), (2048, 16384), (8192, 8192)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        nseed, seed, rnumel = emit(rms_norm_fwd, (x, w, 1e-5), f"rms_norm ({m},{n})")
        ok = nseed == 1 and seed["reduction_loops"] == [None] and seed["num_warps"] == expected_warps(rnumel) and seed["num_stages"] == 1
        all_ok &= ok
        print(f"{'':>28}  -> persistent+w{expected_warps(rnumel)} expected: {'OK' if ok else 'MISMATCH'}\n")

    # sum representative shapes
    for (m, n) in [(2048, 1024), (2048, 16384), (8192, 256)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        nseed, seed, rnumel = emit(sum_kernel, (x,), f"sum ({m},{n})")
        ok = nseed == 1 and seed["reduction_loops"] == [None] and seed["num_warps"] == expected_warps(rnumel)
        all_ok &= ok
        print(f"{'':>28}  -> persistent+w{expected_warps(rnumel)} expected: {'OK' if ok else 'MISMATCH'}\n")

    # long_sum representative shapes (huge rows; still <= 2**20 cap => persistent w32)
    for (m, n) in [(1, 32768), (8, 131072), (16, 262144)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        nseed, seed, rnumel = emit(long_sum, (x,), f"long_sum ({m},{n})")
        # all <= 2**20 (=1048576) so persistent; rnumel>16384 => w32
        below_cap = rnumel <= 2 ** 20
        exp_rl = [None] if below_cap else [16384]
        exp_w = 32  # all these rnumel > 16384
        ok = nseed == 1 and seed["reduction_loops"] == exp_rl and seed["num_warps"] == exp_w
        all_ok &= ok
        print(f"{'':>28}  -> {'persistent' if below_cap else 'looped'}+w{exp_w} expected: {'OK' if ok else 'MISMATCH'}\n")

    print(f"\nALL_OTHER_3_SEEDS_AS_EXPECTED = {all_ok}")


if __name__ == "__main__":
    main()
