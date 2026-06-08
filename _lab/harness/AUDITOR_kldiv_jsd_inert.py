"""AUDITOR A7: kl_div/jsd are nl>=2 AND have in-sample rows > 128KiB (524288B),
so the v6 multi-load cap COULD fire for them. But they are T2 Band-B (num_tiled
_accumulators>=1) whose R_BLOCK is already capped at 16KiB. Verify the v6 seed
is BYTE-IDENTICAL to what v5 would emit (the cap must be inert for them too).

Strategy: classify -> print num_load + num_tiled_accumulators, then compute BOTH
the v6 live seed AND a manual v5-equivalent seed (cap removed: can_persist by
structural rule only) and assert equality, across all in-sample shapes.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    TritonReductionHeuristic as H,
)
from helion._utils import next_power_of_2 as np2  # noqa: E402


def v5_equiv_seed(env, dev_ir, spec):
    """Replicate get_seed_config WITHOUT the multi-load cap (= v5 behavior)."""
    fact = spec.reduction_facts[0]
    persist_cap = env.backend.max_tensor_numel
    can_persist = persist_cap is None or fact.size_hint <= persist_cap
    # NOTE: v5 had NO `if fact.num_load>=2: ...` block here.
    if can_persist:
        extent = np2(fact.size_hint)
        num_warps = H._num_warps(fact)
        persistent = True
    else:
        extent = H.LOOPED_CHUNK
        num_warps = H.LOOPED_NUM_WARPS
        persistent = False
    is_t1 = fact.block_id in spec.reduction_loops.valid_block_ids()
    if is_t1:
        rl = [None] if persistent else [H.LOOPED_CHUNK]
        return {"block_sizes": [H._m_block_size(env)], "reduction_loops": rl,
                "num_warps": num_warps, "num_stages": 1}
    r_block = extent
    if fact.num_tiled_accumulators >= 1:
        bandb_cap = max(1, H.BANDB_R_BLOCK_BYTES // max(1, fact.itemsize))
        r_block = min(r_block, np2(bandb_cap))
    red_idx = spec.block_sizes.block_id_to_index(fact.block_id)
    from typing import cast
    bsl = []
    for i in range(len(spec.block_sizes)):
        bs_spec = spec.block_sizes[i]
        bsl.append(r_block if i == red_idx else H._block_floor(bs_spec))
    return {"block_sizes": bsl, "num_warps": num_warps, "num_stages": 1}


def check(name, kfn, args):
    b = kfn.bind(args)
    env = b.env
    dev_ir = b.host_function.device_ir
    spec = env.config_spec
    if len(spec.reduction_facts) != 1:
        print(f"{name}: reduction_facts={len(spec.reduction_facts)} (no fact)")
        return
    f = spec.reduction_facts[0]
    with env:
        v6 = H.get_seed_config(env, dev_ir)
    v6d = dict(v6)
    v5d = v5_equiv_seed(env, dev_ir, spec)
    ident = (v6d == v5d)
    fires_cap = (f.num_load >= 2 and f.size_hint * max(1, f.itemsize) > H.MULTILOAD_PERSIST_MAX_BYTES)
    print(f"{name}: nl={f.num_load} accum={f.num_tiled_accumulators} "
          f"row_bytes={f.size_hint*max(1,f.itemsize)} cap_would_fire={fires_cap}")
    print(f"    v6={v6d}")
    print(f"    v5={v5d}")
    print(f"    BYTE-IDENTICAL={ident}{'  <-- MISMATCH!' if not ident else ''}")


def main():
    print(f"helion={helion.__file__}\n")
    from examples.kl_div import kl_div_forward
    from examples.jsd import jsd_forward

    def kl_args(bt, v):
        return (torch.randn(bt, v, device="cuda").log_softmax(-1),
                torch.randn(bt, v, device="cuda").softmax(-1))

    def jsd_args(bt, v):
        return (torch.randn(bt, v, device="cuda").log_softmax(-1),
                torch.randn(bt, v, device="cuda").log_softmax(-1))

    print("=== kl_div (nl>=2, in-sample up to 131072=512KiB > cap) ===")
    for (bt, v) in [(4096, 4096), (4096, 65536), (4096, 131072)]:
        try:
            check(f"kl_div ({bt},{v})", kl_div_forward, kl_args(bt, v))
        except Exception as e:  # noqa: BLE001
            print(f"kl_div ({bt},{v}): ERR {type(e).__name__}: {e}")
    print("\n=== jsd (nl>=2, in-sample up to 131072=512KiB > cap) ===")
    for (bt, v) in [(8192, 4096), (8192, 65536), (8192, 131072)]:
        try:
            check(f"jsd ({bt},{v})", jsd_forward, jsd_args(bt, v))
        except Exception as e:  # noqa: BLE001
            print(f"jsd ({bt},{v}): ERR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
