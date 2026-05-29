"""Product B driver: seeded vs unseeded quick-autotune convergence trace.

For ONE (kernel, shape, mode, random_seed) it runs a single cold-cache quick
autotune, writing the per-generation convergence CSV via HELION_AUTOTUNE_LOG.
Designed to be invoked as a FRESH SUBPROCESS per cell (cold Triton/inductor
cache, one GPU, clean timing). The matrix runner (productB_run.sh) launches it.

SEEDED   = our registered TritonReductionHeuristic active (default). The seed
           lands in generation 0 of the initial population (compiler_seed_configs
           -> seed_flat_config_pairs -> initial population, alongside `default`).
UNSEEDED = HELION_DISABLE_AUTOTUNER_HEURISTICS=1 -> compiler_seed_configs returns
           []  -> no compiler seed in gen0 (verified: env.config_spec
           .compiler_seed_configs == [] and autotuner_heuristics == []).

The ONLY difference between the two modes is the presence of the compiler seed
config in the initial population. Same budget knobs, same random seed, same
shape, cold cache, different autotune_log paths.

This script ALSO prints a SEED-INJECTION VERIFICATION block: it computes the
heuristic's seed config for the shape, and reports whether
env.config_spec.compiler_seed_configs contains it (seeded) or is empty
(unseeded). The matrix runner cross-checks this against gen0 of the CSV.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.jsd import jsd_forward  # noqa: E402
from examples.kl_div import kl_div_forward  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402


def build_rms_norm(shape):
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, 1e-5)


def build_reduce(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def build_layer_norm(shape):
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, [n], w, b, 1e-5)


def build_cross_entropy(shape):
    # shape = (N rows, V vocab); int64 labels (matches measure_g_ce.py)
    n, v = shape
    logits = torch.randn(n, v, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device="cuda", dtype=torch.int64)
    return (logits, labels)


def build_softmax(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def build_kl_div(shape):
    # (BT, V); log-prob input, prob target, batchmean, eps (matches measure_g_lossk.py)
    bt, v = shape
    yp = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
    yt = torch.randn(bt, v, device="cuda", dtype=torch.float32).softmax(-1)
    return (yp, yt, False, "batchmean", 1e-10)


def build_jsd(shape):
    # (BT, V); two log-prob inputs, beta=0.5, ignore_index=-100 (matches measure_g_jsd.py)
    bt, v = shape
    lq = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
    lp = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
    return (lq, lp, None, 0.5, -100)


KERNELS = {
    "rms_norm": {"fn": rms_norm_fwd, "build": build_rms_norm},
    "sum": {"fn": sum_kernel, "build": build_reduce},
    "long_sum": {"fn": longsum, "build": build_reduce},
    "layer_norm": {"fn": layer_norm_fwd, "build": build_layer_norm},
    "cross_entropy": {"fn": cross_entropy, "build": build_cross_entropy},
    "softmax_two_pass": {"fn": softmax_two_pass, "build": build_softmax},
    "kl_div": {"fn": kl_div_forward, "build": build_kl_div},
    "jsd": {"fn": jsd_forward, "build": build_jsd},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=list(KERNELS))
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=["seeded", "unseeded"])
    ap.add_argument("--rand-seed", type=int, required=True)
    ap.add_argument("--log", required=True, help="HELION_AUTOTUNE_LOG base path")
    a = ap.parse_args()

    spec = KERNELS[a.kernel]
    fn = spec["fn"]
    shape = (a.M, a.N)
    args = spec["build"](shape)

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[driver] GPU={gpu} helion={helion.__file__}")
    print(f"[driver] kernel={a.kernel} shape={shape} mode={a.mode} "
          f"rand_seed={a.rand_seed}")
    print(f"[driver] disable_heuristics_env="
          f"{os.environ.get('HELION_DISABLE_AUTOTUNER_HEURISTICS')!r} "
          f"force={os.environ.get('HELION_FORCE_AUTOTUNE')!r} "
          f"effort={os.environ.get('HELION_AUTOTUNE_EFFORT')!r} "
          f"max_gen={os.environ.get('HELION_AUTOTUNE_MAX_GENERATIONS')!r} "
          f"rand_seed_env={os.environ.get('HELION_AUTOTUNE_RANDOM_SEED')!r} "
          f"log={os.environ.get('HELION_AUTOTUNE_LOG')!r}")

    # --- SEED-INJECTION VERIFICATION (anti-silent-drop) ---------------------
    # Bind a throwaway kernel to read what compiler_seed_configs returns under
    # the CURRENT settings (seeded vs unseeded). This is what gets injected into
    # generation 0 of the real search below.
    probe = helion.kernel(fn.fn)
    bound = probe.bind(args)
    # compiler_seed_configs is invoked during bind() and stored on the spec.
    seeds_on_spec = list(bound.env.config_spec.compiler_seed_configs)
    heur_names = list(bound.env.config_spec.autotuner_heuristics)
    # Also recompute directly (independent of bind ordering) for cross-check.
    recomputed = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    print(f"[verify] config_spec.compiler_seed_configs (n={len(seeds_on_spec)}): "
          f"{[dict(c) for c in seeds_on_spec]}")
    print(f"[verify] config_spec.autotuner_heuristics: {heur_names}")
    print(f"[verify] recomputed compiler_seed_configs (n={len(recomputed)}): "
          f"{[dict(c) for c in recomputed]}")
    if a.mode == "seeded":
        assert len(recomputed) == 1, (
            f"SEEDED expected exactly 1 compiler seed, got {len(recomputed)}")
        assert heur_names == ["triton_reduction_tile"], heur_names
        print(f"[verify] SEEDED OK: seed config = {dict(recomputed[0])}")
        expected_seed_str = str(recomputed[0])
        # Report what the seed becomes after the autotuner's flat encode/decode
        # (this is what actually lands in gen0). reduction_loops=[None]
        # (persistent) is NOT representable in the flat space for rnumel>4096 and
        # is degraded to a looped chunk (default 4096). num_warps survives.
        from helion.autotuner.config_generation import ConfigGeneration
        cg = ConfigGeneration(bound.env.config_spec)
        flat_norm = cg.unflatten(cg.flatten(recomputed[0]))
        raw = dict(recomputed[0])
        # T1 kernels carry the persistent lever in reduction_loops=[None];
        # T2 (user-tiled) kernels carry it in block_sizes (R_BLOCK>=rnumel).
        # Round-trip BOTH levers through flatten/unflatten so the verification is
        # meaningful for either track.
        raw_rl = raw.get("reduction_loops")
        norm_rl = flat_norm.get("reduction_loops")
        raw_bs = raw.get("block_sizes")
        norm_bs = flat_norm.get("block_sizes")
        is_t1 = raw_rl is not None
        if is_t1:
            lever_raw, lever_norm, lever_name = raw_rl, norm_rl, "reduction_loops"
        else:
            lever_raw, lever_norm, lever_name = raw_bs, norm_bs, "block_sizes"
        degraded = lever_raw != lever_norm
        print(f"[verify] SEED-FLAT-ENCODE: track={'T1' if is_t1 else 'T2'} "
              f"lever={lever_name} raw={lever_raw} "
              f"num_warps={raw.get('num_warps')} -> "
              f"flat-normalized {lever_name}={lever_norm} "
              f"num_warps={flat_norm.get('num_warps')}  "
              f"{'DEGRADED(persistent->looped)' if degraded else 'PRESERVED'}")
    else:
        assert len(recomputed) == 0, (
            f"UNSEEDED expected 0 compiler seeds, got {len(recomputed)}")
        assert heur_names == [], heur_names
        print("[verify] UNSEEDED OK: no compiler seed in initial population")
        expected_seed_str = None
    print(f"[verify] EXPECTED_SEED_STR={expected_seed_str}")

    # --- run the real quick-autotune search --------------------------------
    k = helion.kernel(fn.fn)
    bk = k.bind(args)
    best = bk.autotune(args, force=True)
    print(f"[driver] DONE best_config={dict(best)}")


if __name__ == "__main__":
    main()
