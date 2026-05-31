from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class AutotunerHeuristic:
    """Base class for compiler-owned autotuner heuristics."""

    name: ClassVar[str]
    backend: ClassVar[str]

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        raise NotImplementedError

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        return None

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        """Optional MULTI-seed hook: return a list of structurally-distinct seed
        Configs to inject into the autotuner's gen-0 population, or None to use the
        single ``get_seed_config`` (the default — backward compatible).

        ``compiler_seed_configs`` calls this first and falls back to
        ``[get_seed_config()]`` when it returns None, so Product-A (the bare
        single seed) and existing single-seed heuristics are unaffected. A
        heuristic overrides this only to contribute an aggressive PORTFOLIO of
        seeds (run-2 Goal 3b — beat max-effort autotune): several seeds, each
        embodying one falsifiable structural hypothesis, so the search explores
        around all of them and the ceiling rises (not just convergence speed).
        ``dedupe_configs`` removes exact duplicates while preserving distinct seeds.
        """
        return None


AutotunerHeuristicType = type[AutotunerHeuristic]
