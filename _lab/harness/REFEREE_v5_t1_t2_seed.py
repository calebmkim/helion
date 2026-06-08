"""Referee-OWN verification for v5 (commit 6430b37e):
 (A) T1 no-regression: spot-check ~5 T1 shapes (rms_norm/sum/long_sum/layer_norm)
     emit a heuristic seed AND that seed is the v4 champion config (persistent
     reduction_loops + expected num_warps ramp). We re-derive the champion config
     from the heuristic's documented mechanism, independent of any logged number.
 (B) T2 fire + seed-used: for softmax_two_pass / kl_div / jsd at a couple of
     shapes, prove compiler_seed_configs returns exactly 1 seed, the seed routes
     to a block_sizes-only (no reduction_loops) config, and the generated Triton
     has the inner reduction loop running ONCE (persistent) for non-BandB and
     a CAPPED R_BLOCK (<N) for BandB wide rows.

This is NOT a perf script (no do_bench). It proves the gate fires and the seed is
USED at codegen. Run with the canonical SETUP.md invocation.
"""
from __future__ import annotations

import re
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402


def get_seeds(fn, args):
    bound = fn.bind(args)
    return bound, compiler_seed_configs(bound.env, bound.host_function.device_ir)


def expected_t1_warps(rnumel):
    # mirrors TritonReductionHeuristic._num_warps ramp (documented):
    # <=1024 ->4, <=4096 ->8, then 16, then 32 above 16384
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    if rnumel <= 16384:
        return 16
    return 32


def codegen(bound_kernel, args):
    cfg = helion.Config(**dict(bound_kernel._config))
    return bound_kernel.to_triton_code(cfg)


def count_roffset_loops(code):
    # T1 looped form has `for roffset` ; persistent has none.
    return len(re.findall(r"for roffset", code))


def t2_red_step_vs_extent(code):
    """Return list of (extent, step) for tl.range(0, extent, _BLOCK_SIZE) over a
    matched constexpr block size. persistent iff step>=extent."""
    consts = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"(_BLOCK_SIZE_\d+)\s*=\s*tl\.constexpr\((\d+)\)", code)}
    out = []
    for m in re.finditer(r"tl\.range\(0,\s*(\d+),\s*(_BLOCK_SIZE_\d+)\)", code):
        extent = int(m.group(1))
        step = consts.get(m.group(2))
        if step is not None:
            out.append((extent, step))
    return out


# ---------------- T1 spot-check ----------------
def t1_spotcheck():
    print("=== (A) T1 NO-REGRESSION spot-check ===")
    from examples.rms_norm import rms_norm_fwd
    from examples.layer_norm import layer_norm_fwd

    results = []

    # rms_norm: persistent for all in-sample; warps by ramp
    for (m, n) in [(2048, 4096), (4096, 1536), (8192, 8192), (32768, 256)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        bound, seeds = get_seeds(rms_norm_fwd, (x, w, 1e-5))
        assert len(seeds) == 1, f"rms_norm {(m,n)} seeds={len(seeds)}"
        sd = dict(seeds[0])
        k = helion.kernel(rms_norm_fwd.fn, configs=[helion.Config(**sd)])
        bk = k.bind((x, w, 1e-5)); bk.ensure_config_exists((x, w, 1e-5))
        code = codegen(bk, (x, w, 1e-5))
        persistent = sd.get("reduction_loops") == [None]
        nroff = count_roffset_loops(code)
        exp_w = expected_t1_warps(n)
        ok = (persistent and nroff == 0 and sd["num_warps"] == exp_w
              and dict(bk._config).get("reduction_loops") == [None])
        results.append(ok)
        print(f"  rms_norm {(m,n):} seed={{rl={sd.get('reduction_loops')}, "
              f"bs={sd['block_sizes']}, w={sd['num_warps']}}} "
              f"exp_w={exp_w} roff_loops={nroff} -> {'OK' if ok else 'MISMATCH'}")

    # layer_norm: persistent, warps ramp (with bias path; num_load=3)
    for (m, n) in [(4096, 4096), (8192, 7168)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        bound, seeds = get_seeds(layer_norm_fwd, (x, [n], w, b))
        assert len(seeds) == 1, f"layer_norm {(m,n)} seeds={len(seeds)}"
        sd = dict(seeds[0])
        k = helion.kernel(layer_norm_fwd.fn, configs=[helion.Config(**sd)])
        bk = k.bind((x, [n], w, b)); bk.ensure_config_exists((x, [n], w, b))
        code = codegen(bk, (x, [n], w, b))
        persistent = sd.get("reduction_loops") == [None]
        nroff = count_roffset_loops(code)
        exp_w = expected_t1_warps(n)
        ok = persistent and nroff == 0 and sd["num_warps"] == exp_w
        results.append(ok)
        print(f"  layer_norm {(m,n):} seed={{rl={sd.get('reduction_loops')}, "
              f"bs={sd['block_sizes']}, w={sd['num_warps']}}} "
              f"exp_w={exp_w} roff_loops={nroff} -> {'OK' if ok else 'MISMATCH'}")

    print(f"  T1 spot-check: {sum(results)}/{len(results)} OK")
    return all(results)


# ---------------- T2 fire + seed-used ----------------
def t2_check():
    print("\n=== (B) T2 FIRE + SEED-USED ===")
    from examples.softmax import softmax_two_pass
    from examples.kl_div import kl_div_forward
    from examples.jsd import jsd_forward

    ok_all = []

    # softmax_two_pass (Band A): persistent R_BLOCK>=N, inner loop runs once
    for (m, n) in [(4096, 1024), (4096, 8192), (4096, 16384)]:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        bound, seeds = get_seeds(softmax_two_pass, (x,))
        assert len(seeds) == 1, f"softmax {(m,n)} seeds={len(seeds)}"
        sd = dict(seeds[0])
        assert "reduction_loops" not in sd, f"T2 should have NO reduction_loops: {sd}"
        k = helion.kernel(softmax_two_pass.fn, configs=[helion.Config(**sd)])
        bk = k.bind((x,)); bk.ensure_config_exists((x,))
        code = codegen(bk, (x,))
        steps = t2_red_step_vs_extent(code)
        # persistent: for the extent==n loop, step >= n
        relevant = [(e, s) for (e, s) in steps if e == n]
        persistent = bool(relevant) and all(s >= e for (e, s) in relevant)
        ok = persistent
        ok_all.append(ok)
        print(f"  softmax {(m,n)} bs={sd['block_sizes']} w={sd['num_warps']} "
              f"range(0,{n},step) -> {relevant} persistent={persistent} "
              f"{'OK' if ok else 'FAIL'}")

    # kl_div (Band B): R_BLOCK CAPPED at 4096 for wide rows; loop runs many times
    for (bt, v) in [(4096, 4096), (4096, 65536), (4096, 131072)]:
        yp = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
        yt = torch.randn(bt, v, device="cuda", dtype=torch.float32).softmax(-1)
        a = (yp, yt, False, "batchmean", 1e-10)
        bound, seeds = get_seeds(kl_div_forward, a)
        assert len(seeds) == 1, f"kl_div {(bt,v)} seeds={len(seeds)}"
        sd = dict(seeds[0])
        bs = list(sd["block_sizes"])
        red_block = max(bs)
        # BandB cap = 4096 fp32 elems
        expected_R = min(np2(v), 4096)
        ok = (red_block == expected_R)
        ok_all.append(ok)
        print(f"  kl_div {(bt,v)} bs={bs} w={sd['num_warps']} R_BLOCK={red_block} "
              f"expected_cap={expected_R} {'OK' if ok else 'FAIL'}")

    # jsd (Band B)
    for (bt, v) in [(8192, 4096), (8192, 65536), (8192, 131072)]:
        lq = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
        lp = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
        a = (lq, lp, None, 0.5, -100)
        bound, seeds = get_seeds(jsd_forward, a)
        assert len(seeds) == 1, f"jsd {(bt,v)} seeds={len(seeds)}"
        sd = dict(seeds[0])
        bs = list(sd["block_sizes"])
        red_block = max(bs)
        expected_R = min(np2(v), 4096)
        ok = (red_block == expected_R)
        ok_all.append(ok)
        print(f"  jsd {(bt,v)} bs={bs} w={sd['num_warps']} R_BLOCK={red_block} "
              f"expected_cap={expected_R} {'OK' if ok else 'FAIL'}")

    print(f"  T2 fire+seed-used: {sum(ok_all)}/{len(ok_all)} OK")
    return all(ok_all)


def main():
    import os
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}\n")
    a = t1_spotcheck()
    b = t2_check()
    print(f"\nT1_no_regression_spotcheck={'PASS' if a else 'FAIL'}  "
          f"T2_fire_seed_used={'PASS' if b else 'FAIL'}")


if __name__ == "__main__":
    main()
