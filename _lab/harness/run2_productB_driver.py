"""Run-2 Product B driver: ONE cold-cache autotune (seeded|unseeded, quick|full),
writes the per-generation convergence CSV via HELION_AUTOTUNE_LOG. Fresh subprocess
per cell. Adapted from run-1 productB_driver: wt-reduction-2 path (NO sys.path.insert
footgun), + welford + cross_entropy_online, eviction-aware injection check.

SEEDED   = registered TritonReductionHeuristic active (default) -> seed in gen0.
UNSEEDED = HELION_DISABLE_AUTOTUNER_HEURISTICS=1 -> no compiler seed in gen0.
Budget via HELION_AUTOTUNE_EFFORT + HELION_AUTOTUNE_MAX_GENERATIONS; force=True.
Usage: python run2_productB_driver.py --kernel K --M M --N N --mode seeded|unseeded --rand-seed R --log /path
"""
from __future__ import annotations
import argparse, os
import torch
import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2"
assert helion.__file__.startswith(WT + "/"), helion.__file__

from examples.cross_entropy import cross_entropy, cross_entropy_online
from examples.jsd import jsd_forward
from examples.kl_div import kl_div_forward
from examples.layer_norm import layer_norm_fwd
from examples.long_sum import longsum
from examples.rms_norm import rms_norm_fwd
from examples.softmax import softmax_two_pass
from examples.sum import sum_kernel
from examples.welford import welford
from helion._compiler.autotuner_heuristics import compiler_seed_configs


def b_rms(s):
    m, n = s
    return (torch.randn(m, n, device="cuda"), torch.randn(n, device="cuda"), 1e-5)


def b_reduce(s):
    m, n = s
    return (torch.randn(m, n, device="cuda"),)


def b_ln(s):
    m, n = s
    return (torch.randn(m, n, device="cuda"), [n], torch.randn(n, device="cuda"),
            torch.randn(n, device="cuda"), 1e-5)


def b_ce(s):
    n, v = s
    return (torch.randn(n, v, device="cuda"),
            torch.randint(0, v, (n,), device="cuda", dtype=torch.int64))


def b_kl(s):
    bt, v = s
    return (torch.randn(bt, v, device="cuda").log_softmax(-1),
            torch.randn(bt, v, device="cuda").softmax(-1), False, "batchmean", 1e-10)


def b_jsd(s):
    bt, v = s
    return (torch.randn(bt, v, device="cuda").log_softmax(-1),
            torch.randn(bt, v, device="cuda").log_softmax(-1), None, 0.5, -100)


def b_wf(s):
    m, n = s
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), 1e-5)


KERNELS = {
    "rms_norm": (rms_norm_fwd, b_rms), "sum": (sum_kernel, b_reduce),
    "long_sum": (longsum, b_reduce), "layer_norm": (layer_norm_fwd, b_ln),
    "cross_entropy": (cross_entropy, b_ce),
    "cross_entropy_online": (cross_entropy_online, b_ce),
    "softmax_two_pass": (softmax_two_pass, b_reduce),
    "kl_div": (kl_div_forward, b_kl), "jsd": (jsd_forward, b_jsd),
    "welford": (welford, b_wf),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=list(KERNELS))
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=["seeded", "unseeded"])
    ap.add_argument("--rand-seed", type=int, required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    fn, build = KERNELS[a.kernel]
    args = build((a.M, a.N))
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[driver] GPU={gpu} helion={helion.__file__} kernel={a.kernel} "
          f"shape={(a.M, a.N)} mode={a.mode} rand_seed={a.rand_seed} "
          f"disable_heur={os.environ.get('HELION_DISABLE_AUTOTUNER_HEURISTICS')!r} "
          f"effort={os.environ.get('HELION_AUTOTUNE_EFFORT')!r} "
          f"max_gen={os.environ.get('HELION_AUTOTUNE_MAX_GENERATIONS')!r} "
          f"log={os.environ.get('HELION_AUTOTUNE_LOG')!r}", flush=True)

    # seed-injection verification (eviction-aware)
    probe = helion.kernel(fn.fn)
    bound = probe.bind(args)
    recomputed = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    heur = list(bound.env.config_spec.autotuner_heuristics)
    if a.mode == "seeded":
        assert len(recomputed) >= 1, f"SEEDED expected >=1 seed, got {len(recomputed)}"
        s0 = dict(recomputed[0])
        print(f"[verify] SEEDED n_seeds={len(recomputed)} heur={heur} "
              f"seed0={s0} evict={s0.get('load_eviction_policies')} "
              f"pid={s0.get('pid_type')}", flush=True)
    else:
        assert len(recomputed) == 0, f"UNSEEDED expected 0, got {len(recomputed)}"
        assert heur == [], heur
        print("[verify] UNSEEDED OK: no compiler seed in gen0", flush=True)

    k = helion.kernel(fn.fn)
    bk = k.bind(args)
    best = bk.autotune(args, force=True)
    print(f"[driver] DONE best_config={dict(best)}", flush=True)


if __name__ == "__main__":
    main()
