"""VERBATIM port of main's ``TritonReductionTileHeuristic`` (the NARROW seed).

Source: origin/main @ SHA 8d5cc261 (landed via upstream pytorch/helion PR #2648,
commit ea35dfdd). The two source files copied here line-for-line:

  * helion/_compiler/autotuner_heuristics/triton.py
      - class ``TritonReductionTileHeuristic`` @ lines 299-336
          name = "triton_reduction_tile", backend = "triton"
          is_eligible:    return is_canonical_row_reduction(env)          (line 315)
          get_seed_config (lines 318-336):
              seed = {"block_sizes": [1], "reduction_loops": [None]}
              eviction = spec.load_eviction_policies
              if (eviction.length
                  and isinstance(eviction.inner, EnumFragment)
                  and "last" in eviction.inner.choices):
                  seed["load_eviction_policies"] = ["last"] * eviction.length
              return Config(**seed)
      - import (line 10):  from ...autotuner.config_fragment import EnumFragment
      - import (line 11):  from ...runtime.config import Config

  * helion/_compiler/autotuner_heuristics/common.py
      - def ``is_canonical_row_reduction(env)`` @ lines 38-55:
          spec = env.config_spec
          if len(spec.block_sizes) != 1 or len(spec.reduction_loops) != 1:
              return False
          if spec.matmul_facts:
              return False
          bs_spec = spec.block_sizes[0]
          return max(bs_spec.min_size, bs_spec.autotuner_min) <= 1

This module reimplements ONLY what main's heuristic reads:
  spec.block_sizes (len + [0].min_size/.autotuner_min),
  spec.reduction_loops (len), spec.matmul_facts (truthiness),
  spec.load_eviction_policies (.length, .inner is EnumFragment, 'last' in .choices).
It does NOT touch ``reduction_facts`` (which exists only on MINE's tree).

``Config`` and ``EnumFragment`` are imported from the LIVE helion package so the
emitted config object is identical in type to what the real heuristic returns.

This is a behavior-only port for compile-time config comparison. No autotune,
no do_bench.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

# Imported from the live helion package (same modules main imports from).
from helion.autotuner.config_fragment import EnumFragment
from helion.runtime.config import Config

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR


def main_is_canonical_row_reduction(env: CompileEnvironment) -> bool:
    """VERBATIM port of common.is_canonical_row_reduction (main @ 8d5cc261).

    Whether the kernel is a canonical single-tile row reduction: a single
    non-reduction tile + a single reduction loop, with no matmul facts, and an
    M-axis that admits one row per program (block_size == 1).
    """
    spec = env.config_spec
    # Single non-reduction tile + single reduction dim.
    if len(spec.block_sizes) != 1 or len(spec.reduction_loops) != 1:
        return False
    # No matmul facts (this seeds reduction kernels, not GEMMs, and not a fused
    # matmul+reduction, which keeps its own tiling).
    if spec.matmul_facts:
        return False
    bs_spec = spec.block_sizes[0]
    # M-axis must accept block_size=1 (one row per program).
    return max(bs_spec.min_size, bs_spec.autotuner_min) <= 1


def main_is_eligible(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    """VERBATIM port of TritonReductionTileHeuristic.is_eligible (main @ 8d5cc261).

    main line 315:  return is_canonical_row_reduction(env)
    (``device_ir`` is part of the heuristic signature but unread by main here.)
    """
    return main_is_canonical_row_reduction(env)


def main_get_seed_config(
    env: CompileEnvironment, device_ir: DeviceIR
) -> Config | None:
    """VERBATIM port of TritonReductionTileHeuristic.get_seed_config (main @ 8d5cc261).

    The real heuristic is only invoked when ``is_eligible`` returned True, so we
    mirror that: return ``None`` when not eligible (matching the
    ``compiler_seed_configs`` flow where a non-eligible heuristic emits nothing).
    """
    if not main_is_eligible(env, device_ir):
        return None

    spec = env.config_spec
    seed: dict[str, Any] = {
        "block_sizes": [1],
        "reduction_loops": [None],
    }
    # Emit 'last' only where the backend supports it; backends that restrict
    # eviction to ("",) keep the spec default so the seed stays valid.
    eviction = spec.load_eviction_policies
    if (
        eviction.length
        and isinstance(eviction.inner, EnumFragment)
        and "last" in eviction.inner.choices
    ):
        seed["load_eviction_policies"] = ["last"] * eviction.length
    return Config(**seed)
