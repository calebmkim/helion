"""Launcher around benchmarks/run.py that optionally promotes the reduction-seed
heuristic to the kernel's *default* config, so the seed is exercised directly at
HELION_AUTOTUNE_EFFORT=none through the normal TritonBench harness (do_bench +
accuracy gate + torch.compile baselines) — no autotuning search.

Three arms (all measured by TritonBench's do_bench, fp32, accuracy-gated):
  * Helion SEEDED   : HELION_PROMOTE_REDUCTION_SEED=1  + HELION_AUTOTUNE_EFFORT=none
                      -> default_config() returns the PR's seed config.
  * Helion DEFAULT  : HELION_DISABLE_AUTOTUNER_HEURISTICS=1 + HELION_AUTOTUNE_EFFORT=none
                      -> default_config() returns the upstream base config (unseeded).
  * torch.compile   : the operator's torch_compile_* baseline (same invocation).

Usage: identical to benchmarks/run.py; pass through all args.
"""

from __future__ import annotations

import os
import sys


def _maybe_promote_reduction_seed() -> None:
    if os.environ.get("HELION_PROMOTE_REDUCTION_SEED", "0") != "1":
        return
    import helion._compiler.autotuner_heuristics.triton as triton_heuristics

    # Promote both reduction seed heuristics (T1 tile + T2 user-tile) so that
    # compiler_seed_configs() also sets compiler_default_config, which
    # ConfigSpec.default_config() returns at effort=none.
    triton_heuristics.TritonReductionTileHeuristic.promote_seed_to_default = True
    triton_heuristics.TritonReductionUserTileHeuristic.promote_seed_to_default = True
    print(
        "[run_seeded] reduction seed promoted to default config (seeded arm)",
        file=sys.stderr,
    )


def main() -> None:
    _maybe_promote_reduction_seed()
    # Import after the patch so the heuristic classes carry the promoted flag.
    from benchmarks.run import main as run_main

    run_main()


if __name__ == "__main__":
    main()
