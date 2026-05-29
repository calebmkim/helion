"""Test the autotuner's INTERLEAVED bench path (population rebenchmark) on the
4 warp configs for rms_norm (32768,256) fp32 persistent block=1, to see if the
interleaved path is where w32 gets mis-ranked as ~73us.

interleaved_bench(fns, repeat=R) is the code path PopulationBasedSearch.rebenchmark
uses. It warms each fn ONCE, then for repeat iters records back-to-back CUDA events
per fn with an L2 flush. We feed it exactly the 4 compiled configs and compare to
the autotuner do_bench and a fair do_bench.
"""

from __future__ import annotations

import functools
import os
import sys

import torch

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
sys.path.insert(0, WT)

import helion  # noqa: E402

assert helion.__file__.startswith(WT)

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402

from triton.testing import do_bench as triton_do_bench  # noqa: E402
from helion.autotuner.benchmarking import do_bench as autotuner_do_bench  # noqa: E402
from helion.autotuner.benchmarking import interleaved_bench  # noqa: E402

EPS = 1e-6
M, N = 32768, 256
WARPS = [4, 8, 16, 32]


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32, device="cuda")
    weight = torch.ones(N, dtype=torch.float32, device="cuda")

    print(f"=== interleaved_bench probe ({M},{N}) fp32 GPU={gpu} ===\n")

    compiled_fns = []
    for w in WARPS:
        seed = helion.Config(
            reduction_loops=[None], block_sizes=[1], num_warps=w, num_stages=1
        )
        kern = helion.kernel(rms_norm_fwd.fn, configs=[seed])
        bound = kern.bind((x, weight, EPS))
        cfg = bound._config if getattr(bound, "_config", None) is not None else seed
        compiled = bound.compile_config(cfg, allow_print=False)
        compiled_fns.append((w, compiled))

    # callables exactly as rebenchmark builds them: functools.partial(m.fn, *args)
    iterator = [functools.partial(c, x, weight, EPS) for (_w, c) in compiled_fns]

    # what repeat would rebenchmark pick? base_repeat=int(200/best_perf). best_perf
    # here is the fastest = w4 ~0.036ms -> int(200/0.036)=5555 -> capped 1000.
    for repeat in [50, 200, 1000]:
        timings = interleaved_bench(iterator, repeat=repeat)
        print(f"interleaved_bench repeat={repeat}:")
        for (w, _c), t in zip(compiled_fns, timings):
            print(f"   w={w:>2}: {t*1000:8.2f}us")
        print()

    # contrast: autotuner do_bench (warmup=1 rep=50) one at a time, and fair
    print("per-config autotuner do_bench(warmup=1,rep=50) vs fair triton do_bench:")
    for (w, c) in compiled_fns:
        call = functools.partial(c, x, weight, EPS)
        call(); torch.cuda.synchronize()
        a = float(autotuner_do_bench(call, return_mode="median", warmup=1, rep=50))
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        f = float(triton_do_bench(call, return_mode="median"))
        print(f"   w={w:>2}: autotuner={a*1000:8.2f}us   fair={f*1000:8.2f}us")


if __name__ == "__main__":
    main()
