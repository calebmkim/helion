"""Validate the NEW welford seeds: per shape, extract the live seed, run it
(configs=[seed], no autotune), check correctness vs eager layer_norm, measure
G = tc_default/seed_lat. Prime-N canary included (must be correct AND fast).
Usage: ... python run2_wf_validate.py "262144,1024" "262144,1536" ...
"""
from __future__ import annotations
import sys, os, json
from statistics import median
import torch
import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2"
assert helion.__file__.startswith(WT + "/"), helion.__file__
from examples.welford import welford, eager_layer_norm
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from triton.testing import do_bench

EPS = 1e-5
N_RUNS = 7


def args_for(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def med(fn):
    torch.cuda.synchronize()
    return median([float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)])


def one(m, n):
    a = args_for(m, n)
    ref = eager_layer_norm(*a)
    bound = welford.bind(a)
    seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
    k = helion.kernel(welford.fn, configs=[helion.Config(**seed)])
    bk = k.bind(a); bk.ensure_config_exists(a)
    out = bk(*a); out = out[0] if isinstance(out, tuple) else out
    err = float((out.float() - ref.float()).abs().max())
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-4))
    seed_lat = med(lambda: bk(*a)) if ok else None
    torch._dynamo.reset()
    tc = torch.compile(eager_layer_norm); tc(*a)
    tc_lat = med(lambda: tc(*a))
    return {"shape": [m, n], "seed_blocks": dict(bk._config)["block_sizes"],
            "warps": dict(bk._config)["num_warps"], "ok": ok, "maxerr": err,
            "seed_us": (seed_lat * 1e3) if seed_lat else None,
            "tc_us": tc_lat * 1e3, "G": (tc_lat / seed_lat) if seed_lat else None}


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    for arg in sys.argv[1:]:
        m, n = (int(v) for v in arg.split(","))
        r = one(m, n)
        print(json.dumps({**r, "gpu": gpu}), flush=True)


if __name__ == "__main__":
    main()
