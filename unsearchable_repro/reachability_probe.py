"""Reference reachability probe: is a given config reachable by the CuTe autotuner?

This is the ONE example script in this directory, provided as a reference for the method.
The other classes in UNREACHABLE_CONFIGS.md are left for you to probe yourself — writing
your own is a genuine check on this one.

It answers a pure CONFIG-SPACE question and never times anything, so there is no
benchmarking bias to worry about: either the search can produce the config or it cannot.

Four independent surfaces are checked, because "unreachable" can mean different things:

  1. SEEDS      -- is the config among ``spec.compiler_seed_configs``? A seeded config is
                   benchmarked even if the search could never draw it.
  2. DRAWS      -- does ``random_population(N)`` ever produce it? This is the search proper.
  3. STRICT     -- does ``normalize(_fix_invalid=False)`` accept it? If this FAILS, the config
                   is not even expressible via set_config; some gate rejects it outright.
  4. SEARCH-PATH-- does ``normalize(_fix_invalid=True)`` preserve it? A config can pass (3)
                   and still be silently REWRITTEN here, which is the subtlest failure mode
                   and the one worth looking for.

Usage:
    CUDA_VISIBLE_DEVICES=1 python reachability_probe.py                  # the C5 fp8 case
    CUDA_VISIBLE_DEVICES=1 python reachability_probe.py --case c5-bf16   # the C5 bf16 case

Add your own case to CASES below. Set TREE=<path> to probe a different checkout.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys

ROOT = os.environ.get(
    "TREE", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/benchmarks/cute")
os.environ.setdefault("HELION_BACKEND", "cute")
os.environ.setdefault("HELION_AUTOTUNE_BENCHMARK_SUBPROCESS", "0")

import helion  # noqa: E402

# Guard against silently importing an installed copy instead of this checkout --
# a wrong tree here produces a confidently wrong answer.
assert helion.__file__.startswith(ROOT), (
    f"imported {helion.__file__}, expected a file under {ROOT}"
)

import compare_matmul_backends as bench  # noqa: E402

from helion.autotuner.config_generation import ConfigGeneration  # noqa: E402

# Each case: the problem, plus the config UNREACHABLE_CONFIGS.md claims is unreachable.
# Configs are abbreviated to the keys that matter for the claim; the full versions are in
# ../unreachable_winners.json.
CASES = {
    "c5-fp8": {
        "problem": {
            "m": 512,
            "k": 2048,
            "n": 4096,
            "dtype": "float8_e4m3fn",
            "epilogue": "scaled_mm",
        },
        "kernel": ("examples.fp8_matmul", "fp8_matmul"),
        "claim": "block_n=128 with cluster_m=2 is rewritten to block_n=256",
        "config": {
            "block_sizes": [256, 128, 128],
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": 8,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
            "l2_groupings": [1],
            "pid_type": "persistent_interleaved",
        },
    },
    "c5-bf16": {
        "problem": {
            "m": 512,
            "k": 4096,
            "n": 4096,
            "dtype": "bfloat16",
            "epilogue": "none",
        },
        "kernel": ("examples.matmul", "matmul"),
        "claim": "block_n=128 with cluster_m=2 is rewritten to block_n=256",
        "config": {
            "block_sizes": [256, 128, 128],
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": 4,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
            "l2_groupings": [1],
            "pid_type": "persistent_interleaved",
        },
    },
}


def canonical_tile(config: dict[str, object]) -> tuple[object, ...]:
    block_sizes = config.get("block_sizes") or []
    return tuple(list(block_sizes)[:3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="c5-fp8", choices=sorted(CASES))
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    case = CASES[args.case]

    print(f"tree   : {ROOT}")
    print(f"helion : {helion.__file__}")
    print(f"case   : {args.case} -- {case['claim']}")
    print(f"problem: {case['problem']}")

    problem = argparse.Namespace(seed=0, **case["problem"])
    _, lhs, rhs, bias, residual = bench._make_matmul_problem(problem)
    kernel_args = bench._helion_matmul_args(problem, lhs, rhs, bias, residual)

    module_name, attr = case["kernel"]
    __import__(module_name)
    kernel = getattr(sys.modules[module_name], attr)
    bound = kernel.bind(kernel_args)
    _ = bound.host_function.device_ir
    spec = bound.config_spec

    target_tile = canonical_tile(case["config"])
    target_cluster_m = case["config"]["tcgen05_cluster_m"]

    # 1. SEEDS
    seeds = [c.config for c in spec.compiler_seed_configs]
    seeded = any(
        canonical_tile(s) == target_tile
        and s.get("tcgen05_cluster_m") == target_cluster_m
        for s in seeds
    )
    print(f"\n[1] compiler seeds: {len(seeds)}")
    for s in seeds:
        print(
            f"      {canonical_tile(s)} cluster_m={s.get('tcgen05_cluster_m')} "
            f"ab={s.get('tcgen05_ab_stages')}"
        )
    print(f"    target seeded: {seeded}")

    # 2. DRAWS
    random.seed(args.seed)
    population = ConfigGeneration(spec).random_population(args.draws)
    hits = sum(
        1
        for c in population
        if canonical_tile(c.config) == target_tile
        and c.config.get("tcgen05_cluster_m") == target_cluster_m
    )
    tiles: dict[tuple[object, ...], int] = {}
    for c in population:
        if c.config.get("tcgen05_cluster_m") == target_cluster_m:
            tile = canonical_tile(c.config)
            tiles[tile] = tiles.get(tile, 0) + 1
    print(f"\n[2] target in {args.draws} draws: {hits}")
    print(
        f"    cluster_m={target_cluster_m} tiles drawn: {dict(sorted(tiles.items()))}"
    )

    # 3 + 4. STRICT vs SEARCH PATH
    full = dict(case["config"])
    full.setdefault("indexing", ["tensor_descriptor"] * spec.indexing.length)
    base = helion.Config(**full)

    strict = copy.deepcopy(base)
    try:
        spec.normalize(strict, _fix_invalid=False)
        print(f"\n[3] strict set_config : ACCEPTED -> {canonical_tile(strict.config)}")
    except Exception as exc:
        print(f"\n[3] strict set_config : REJECTED {type(exc).__name__}: {exc}")

    projected = copy.deepcopy(base)
    try:
        spec.normalize(projected, _fix_invalid=True)
        after = canonical_tile(projected.config)
        verdict = "PRESERVED" if after == target_tile else f"REWRITTEN -> {after}"
        print(
            f"[4] search path       : {verdict} "
            f"(cluster_m={projected.config.get('tcgen05_cluster_m')}, "
            f"ab={projected.config.get('tcgen05_ab_stages')})"
        )
    except Exception as exc:
        print(f"[4] search path       : RAISED {type(exc).__name__}: {exc}")

    print(
        "\nA config that is not seeded, never drawn, and rewritten by the search path is "
        "unreachable\nby the autotuner even though set_config can run it. To test the "
        "PERFORMANCE claim, write\nyour own harness -- see UNREACHABLE_CONFIGS.md."
    )


if __name__ == "__main__":
    main()
