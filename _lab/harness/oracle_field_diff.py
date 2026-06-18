"""Oracle field-diff: Helion full-autotune winning config vs the heuristic seed.

For representative shapes, run Helion effort=full to get the MAX winning config
(``bound._config`` after autotune), then diff against the heuristic's seed on the
levers: reduction_loops (persistent vs looped), block_sizes (R/M), num_warps,
num_stages. Reports the oracle latency and ``G_oracle = tc_default_lat / oracle_lat``.

LEVER-ISOLATION GUARD (harness-integrity, 2026-05-28)
-----------------------------------------------------
The autotuner winner is a SINGLE coupled point in config space: ``num_warps`` is
coupled to ``block_sizes`` (a tiny M-block on a huge-M shape forces a huge grid;
``raise_grid_block_minimums`` floors the M-block at 2+ for large M, so configs
like ``block_sizes=[1]`` may NEVER be tested by the search). A field-diff that
re-benches a FABRICATED config — e.g. the oracle's ``num_warps`` paired with
``block_sizes=[1]`` that the oracle never paired it with — measures a config that
does not exist on the search frontier and yields a nonsense latency (this is the
exact bug that produced the bogus "oracle num_warps=32 is a measurement artifact"
claim: block=1/w32 = 1174us was a config the autotuner never tested).

RULE, enforced by ``_assert_full_verbatim_config`` below: ALWAYS re-bench the
FULL verbatim oracle config (every lever, exactly as the autotuner emitted it).
NEVER re-bench a config built by taking ONE oracle lever and pinning the rest
(especially never ``block_sizes=[1]``). To A/B a single lever, vary it AROUND a
full verbatim config and keep every other lever at the verbatim value.

Run with the canonical invocation (set CUDA_VISIBLE_DEVICES to a free GPU).
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

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
N_RUNS = 7
SHAPES = [(32768, 256), (4096, 5120), (2048, 16384)]
LEVERS = ["block_sizes", "reduction_loops", "num_warps", "num_stages"]


def build_args(shape):
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, EPS)


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def _assert_full_verbatim_config(bound_kernel, oracle_cfg: dict) -> None:
    """Guard: the config we are about to bench MUST be the FULL verbatim oracle.

    Refuses to proceed if the resolved/normalized config that will actually run
    differs from the autotuner's emitted winner on ANY of the coupled levers.
    This makes it structurally impossible to re-bench a fabricated single-lever
    config (e.g. the oracle's num_warps with block pinned to 1) — the bug that
    produced the bogus w32 'artifact'. warps x block are coupled: always bench
    the whole point, never an isolated lever.
    """
    resolved = dict(bound_kernel._config)
    coupled = ["block_sizes", "reduction_loops", "num_warps", "num_stages"]
    mismatches = {
        lever: (oracle_cfg.get(lever), resolved.get(lever))
        for lever in coupled
        if oracle_cfg.get(lever) != resolved.get(lever)
    }
    assert not mismatches, (
        "LEVER-ISOLATION GUARD TRIPPED: about to bench a config that differs "
        f"from the verbatim oracle winner on {list(mismatches)}: {mismatches}. "
        "Re-bench the FULL verbatim oracle config (all levers together), never a "
        "single isolated lever (esp. never block_sizes=[1])."
    )
    # Hard backstop: never bench block_sizes=[1] unless the oracle itself emitted
    # it (raise_grid_block_minimums floors M-block at 2 for large M, so [1] is
    # usually a config the search never tested).
    if resolved.get("block_sizes") == [1]:
        assert oracle_cfg.get("block_sizes") == [1], (
            "Refusing to bench block_sizes=[1]: the oracle did not emit it. "
            "This is the fabricated-config footgun."
        )


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}\n")
    for shape in SHAPES:
        args = build_args(shape)
        x, w, e = args
        ref = rms_norm_pytorch(x, w, e)

        # seed
        bound = rms_norm_fwd.bind(args)
        seed = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]

        # tc default
        torch._dynamo.reset()
        tc = torch.compile(rms_norm_pytorch)
        tc(x, w, e)
        tc_lat = med(lambda: tc(x, w, e))

        # Helion full autotune (the oracle)
        os.environ["HELION_AUTOTUNE_EFFORT"] = "full"
        k = helion.kernel(rms_norm_fwd.fn)
        bound_o = k.bind(args)
        bound_o.autotune(args)  # runs the full search
        oracle_cfg = dict(bound_o._config)

        # GUARD: re-bench ONLY the full verbatim oracle config (all levers). The
        # config that will actually run is bound_o._config — assert it equals the
        # emitted winner so no isolated/fabricated lever can sneak in.
        _assert_full_verbatim_config(bound_o, oracle_cfg)

        out_o = bound_o(*args)
        out_o = out_o[0] if isinstance(out_o, tuple) else out_o
        ok = torch.allclose(out_o.float(), ref.float(), rtol=1e-3, atol=1e-4)
        oracle_lat = med(lambda: bound_o(*args))

        print(f"=== shape {shape}  (correct={ok}) ===")
        print(f"  tc_default_lat = {tc_lat*1000:.1f}us")
        print(f"  oracle_lat     = {oracle_lat*1000:.1f}us  G_oracle={tc_lat/oracle_lat:.3f}")
        print(f"  {'lever':>16} {'seed':>16} {'oracle':>16}")
        sd = dict(seed)
        for lever in LEVERS:
            print(f"  {lever:>16} {str(sd.get(lever)):>16} {str(oracle_cfg.get(lever)):>16}")
        print(f"  full oracle config: {oracle_cfg}\n")


if __name__ == "__main__":
    main()
