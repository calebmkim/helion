"""STEP A (v3): re-derive the persistent-vs-looped crossover with num_warps HELD EQUAL.

The v2 auditor showed the long_sum win was entirely num_warps=32, NOT the looped
or grid-occupancy branches: persistent/w32 beat the shipped looped/w32 on 8/9
shapes. So here we A/B persistent vs looped with warps HELD EQUAL (sweep {16,32}
for BOTH), over the huge-rnumel regime, at several M. We answer:

  (a) the rnumel where LOOPED genuinely starts beating PERSISTENT at equal warps
      (auditor evidence: only the 4MiB row wanted looped) -- this sets the looped
      byte ceiling.
  (b) the best num_warps for the PERSISTENT path per rnumel (ramp breakpoints).

Workload: sum_kernel (num_load=1) -- same memory-op class as long_sum. This is
in-sample-legitimate workload characterization (synthetic sweep, in-sample firewall).

One process; fresh tensors per (M,rnumel). Tiny-M huge-row latencies are small so
we use a robust median-of-N do_bench and also report per-warp so noise is visible.
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.sum import sum_kernel  # noqa: E402

N_RUNS = 9
# fp32 elems. 32768=128KiB ... 4194304=16MiB
# NOTE: PERSISTENT is STRUCTURALLY IMPOSSIBLE above 1048576 elems (Triton
# TRITON_MAX_TENSOR_NUMEL=1048576 -> tl.arange over the whole row fails). So for
# rnumel > 1048576 only the looped path exists -- a hard physics bound on the
# looped threshold. We catch the persistent compile failure and report it.
RNUMELS = [131072, 262144, 393216, 524288, 786432, 1048576, 2097152, 4194304]
MS = [1, 4, 16, 64, 256]
WARPS = [16, 32]
LOOPED_CHUNK = 16384
PERSIST_NUMEL_MAX = 1048576  # TRITON_MAX_TENSOR_NUMEL


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def time_cfg(x, ref, reduction_loops, warps):
    cfg = helion.Config(block_sizes=[1], reduction_loops=reduction_loops,
                        num_warps=warps, num_stages=1)
    k = helion.kernel(sum_kernel.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3), "correctness"
    return med(lambda: b(x)) * 1000


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  N_RUNS={N_RUNS}  chunk={LOOPED_CHUNK}")
    print("metric: best_P/best_L over warps{16,32}; >1 => LOOPED wins. Also pure-lever P/L at each warp.\n")
    for n in RNUMELS:
        kib = n * 4 // 1024
        print(f"========== rnumel={n} ({kib}KiB) ==========")
        print(f"{'M':>5} | {'P/16':>6} {'P/32':>6} | {'L/16':>6} {'L/32':>6} | "
              f"{'P32/L32':>8} {'P16/L16':>8} | {'bestP':>6} {'bestL':>6} {'bestP/bestL':>11} {'win':>5}")
        persistent_ok = n <= PERSIST_NUMEL_MAX
        for m in MS:
            x = torch.randn(m, n, device="cuda", dtype=torch.float32)
            ref = x.sum(-1)
            if persistent_ok:
                p = {w: time_cfg(x, ref, [None], w) for w in WARPS}
            else:
                p = {w: float("nan") for w in WARPS}
            # only loop if chunk < rnumel (else looped==persistent)
            if LOOPED_CHUNK < n:
                lp = {w: time_cfg(x, ref, [LOOPED_CHUNK], w) for w in WARPS}
            else:
                lp = {w: float("nan") for w in WARPS}
            valid_p = [v for v in p.values() if v == v]
            best_p = min(valid_p) if valid_p else float("nan")
            valid_l = [v for v in lp.values() if v == v]
            best_l = min(valid_l) if valid_l else float("nan")
            p32l32 = p[32] / lp[32] if lp[32] == lp[32] else float("nan")
            p16l16 = p[16] / lp[16] if lp[16] == lp[16] else float("nan")
            ratio = best_p / best_l if best_l == best_l else float("nan")
            win = "LOOP" if (ratio == ratio and ratio > 1.0) else "PERS"
            print(f"{m:>5} | {p[16]:>6.1f} {p[32]:>6.1f} | "
                  f"{lp[16]:>6.1f} {lp[32]:>6.1f} | "
                  f"{p32l32:>8.3f} {p16l16:>8.3f} | "
                  f"{best_p:>6.1f} {best_l:>6.1f} {ratio:>11.3f} {win:>5}")
        print()


if __name__ == "__main__":
    main()
