from __future__ import annotations

import sys
import statistics
import json

import torch
import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
import helion.runtime

sys.path.insert(0, "/home/dev/local/helion-pr-with-lab/examples")
from softmax import softmax_two_pass
from rms_norm import rms_norm_fwd
from layer_norm import layer_norm_fwd
from sum import sum_kernel
from welford import welford

DEV = torch.device("cuda")
NUM_SM = helion.runtime.get_num_sm(DEV)

# CUDA-graph device time: capture CALLS_PER_GRAPH per graph, median over REPS reps.
CALLS_PER_GRAPH = 50
REPS = 21          # >= 15
WARMUP_GRAPH_LAUNCHES = 5


def make_kernel(fn, args, W):
    b = fn.bind(args)
    seed = compiler_seed_configs(b.env, b.host_function.device_ir)[0]
    cfg = dict(seed)
    cfg["num_warps"] = W
    return helion.kernel(fn.fn, config=helion.Config(**cfg), static_shapes=True)


def cudagraph_device_time_ms(callable_fn):
    """Return median per-call device time in ms via CUDA graph capture."""
    # Warmup (also forces compile / allocations) on a side stream.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            callable_fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS_PER_GRAPH):
            callable_fn()

    # warmup graph launches
    for _ in range(WARMUP_GRAPH_LAUNCHES):
        g.replay()
    torch.cuda.synchronize()

    times = []
    for _ in range(REPS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / CALLS_PER_GRAPH)
    del g
    return statistics.median(times)


def build_inputs(kernel_name, M, N, dt):
    x = torch.randn([M, N], device=DEV, dtype=dt)
    w = torch.randn([N], device=DEV, dtype=dt)
    bvec = torch.randn([N], device=DEV, dtype=dt)
    if kernel_name == "softmax_two_pass":
        return softmax_two_pass, (x,)
    if kernel_name == "rms_norm_fwd":
        return rms_norm_fwd, (x, w, 1e-5)
    if kernel_name == "layer_norm_fwd":
        return layer_norm_fwd, (x, [N], w, bvec, 1e-5)
    if kernel_name == "sum_kernel":
        return sum_kernel, (x,)
    if kernel_name == "welford":
        # welford(weight, bias, x) — x is fp32 in example but we honor dt
        return welford, (w, bvec, x, 1e-5)
    raise ValueError(kernel_name)


KERNELS = ["softmax_two_pass", "rms_norm_fwd", "layer_norm_fwd", "sum_kernel", "welford"]
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
MS = [2048, 8192, 16384, 32768, 65536, 131072, 262144]
WARPS = [1, 2, 4, 8, 16, 32]
N = 512
N_SUM = 1024  # sum uses N=1024 per task


def main():
    results = {}
    # allow restricting via argv: kernel dtype
    only_kernel = sys.argv[1] if len(sys.argv) > 1 else None
    only_dtype = sys.argv[2] if len(sys.argv) > 2 else None

    for kname in KERNELS:
        if only_kernel and kname != only_kernel:
            continue
        for dname, dt in DTYPES.items():
            if only_dtype and dname != only_dtype:
                continue
            n = N_SUM if kname == "sum_kernel" else N
            for M in MS:
                fn, args = build_inputs(kname, M, n, dt)
                row = {}
                for W in WARPS:
                    try:
                        k = make_kernel(fn, args, W)
                        t = cudagraph_device_time_ms(lambda: k(*args))
                        row[W] = t
                    except Exception as e:
                        row[W] = None
                        sys.stderr.write(f"FAIL {kname} {dname} M={M} W={W}: {type(e).__name__}: {e}\n")
                key = f"{kname}|{dname}|{M}"
                results[key] = row
                occ = M / NUM_SM
                # report
                valid = {w: v for w, v in row.items() if v is not None}
                best_w = min(valid, key=valid.get) if valid else None
                w1 = row.get(1); w4 = row.get(4); w8 = row.get(8)
                r14 = (w1 / w4) if (w1 and w4) else None
                r18 = (w1 / w8) if (w1 and w8) else None
                print(f"{kname:16s} {dname} M={M:7d} occ={occ:7.1f} "
                      f"best=w{best_w} "
                      f"w1/w4={r14:.3f} " if r14 else f"{kname:16s} {dname} M={M:7d} occ={occ:7.1f} best=w{best_w} w1/w4=NA ", end="")
                print(f"w1/w8={r18:.3f} | " if r18 else "w1/w8=NA | ", end="")
                print("times(us): " + " ".join(f"w{w}={row[w]*1000:.2f}" if row[w] else f"w{w}=NA" for w in WARPS))
                sys.stdout.flush()
    with open("/tmp/occ_results.json", "w") as f:
        json.dump({"num_sm": NUM_SM, "results": results}, f, indent=2)
    print("\nWROTE /tmp/occ_results.json")


if __name__ == "__main__":
    main()
