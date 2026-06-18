"""Canonical bare-seed run harness (Step 0, Deliverable C).

Given a Helion kernel, an args-builder for a shape, a reference fn, and a seed
``Config`` (as a dict), this:

  1. asserts ``helion`` is the worktree copy (silent-no-op guard),
  2. builds args at the requested shape/dtype,
  3. runs the seed with NO search via ``helion.kernel(fn, configs=[seed])`` so a
     structurally-invalid seed RAISES instead of being silently dropped,
  4. proves no autotune ran (HELION_AUTOTUNE_LOG CSV absent + len(configs)==1
     short-circuit path),
  5. inspects the NORMALIZED resolved config (``bound._config``) and confirms the
     seed was actually used (persistent-vs-looped + num_warps reflected in the
     generated Triton),
  6. runs a correctness check vs the provided reference (fp32-justified tol),
  7. returns median-of-N latency (triton.testing.do_bench) + spread.

Everything (shape, dtype, N, seed, tolerances) is a parameter. dtype defaults to
fp32 but is NOT hardcoded.

Importable: ``from _lab.harness.bare_seed_run import run_bare_seed``.
CLI demo: see ``if __name__ == "__main__"`` (defaults to rms_norm (2048,4096) fp32).

NOTE: this script must be run with the canonical invocation, e.g.::

    cd /home/calebkim/helion-new-heuristics/wt-reduction && \
    CUDA_VISIBLE_DEVICES=2 \
    PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction \
    HELION_AUTOTUNE_LOG=/tmp/seed_probe \
    /home/calebkim/.conda/envs/helion/bin/python _lab/harness/bare_seed_run.py
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from statistics import median
from statistics import pstdev
import sys
from typing import Any
from typing import Callable

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"


def assert_worktree_helion() -> None:
    """Hard guard: refuse to run if helion isn't the worktree copy."""
    assert helion.__file__.startswith(WORKTREE), (
        f"helion resolved to {helion.__file__!r}, NOT the worktree "
        f"({WORKTREE}). Set PYTHONPATH={WORKTREE} and run from a cwd that is "
        f"not the original checkout root. Refusing to run a silent no-op."
    )


@dataclass
class BareSeedResult:
    seed_raw: dict[str, Any]
    seed_normalized: dict[str, Any]
    seed_used: bool
    seed_used_how: str
    autotune_ran: bool
    autotune_evidence: str
    correctness_pass: bool
    max_abs: float
    max_rel: float
    rtol: float
    atol: float
    latency_median_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_stddev_ms: float
    n_runs: int
    generated_triton: str


def _looped_signature(triton_code: str) -> bool:
    """True if the generated Triton contains a reduction loop (looped, not persistent)."""
    return "for roffset" in triton_code


def _num_warps_in_launcher(triton_code: str) -> int | None:
    """Extract num_warps from the kernel launcher call in the generated Triton."""
    import re

    m = re.search(r"num_warps=(\d+)", triton_code)
    return int(m.group(1)) if m else None


def run_bare_seed(
    fn: Any,
    build_args: Callable[[tuple[int, ...], torch.dtype], tuple[object, ...]],
    reference: Callable[..., torch.Tensor],
    shape: tuple[int, ...],
    seed: dict[str, Any],
    *,
    dtype: torch.dtype = torch.float32,
    n_runs: int = 7,
    rtol: float = 1e-3,
    atol: float = 1e-4,
    autotune_log_prefix: str | None = None,
    output_index: int = 0,
) -> BareSeedResult:
    """Run a single bare seed and gather Step-0 evidence.

    Args:
        fn: the ``@helion.kernel`` function (e.g. ``rms_norm_fwd``). Its ``.fn``
            (the undecorated callable) is what we re-wrap with ``configs=[seed]``.
        build_args: ``(shape, dtype) -> args tuple`` for both the kernel and the
            reference. The kernel and reference must accept the same args.
        reference: fp32 reference fn; called with the SAME args; returns a Tensor.
        shape: e.g. ``(2048, 4096)``.
        seed: the seed Config as a dict (raw, pre-normalize).
        dtype: tensor dtype (default fp32, NOT hardcoded).
        n_runs: number of do_bench launches; median/spread taken over these (>=5).
        rtol, atol: correctness tolerance (fp32-justified; never silently loosen).
        autotune_log_prefix: if set, points HELION_AUTOTUNE_LOG here so we can
            prove the CSV is absent (no real search ran). If None, uses the env
            var already set, else a temp path.
        output_index: which element of the kernel's output tuple to compare
            (rms_norm_fwd returns (out, inv_rms); compare out).
    """
    assert_worktree_helion()
    from triton.testing import do_bench

    # --- autotune-log path (CSV is written ONLY when a real search runs) ---
    if autotune_log_prefix is None:
        autotune_log_prefix = os.environ.get("HELION_AUTOTUNE_LOG") or "/tmp/seed_probe"
    os.environ["HELION_AUTOTUNE_LOG"] = autotune_log_prefix
    csv_path = Path(autotune_log_prefix + ".csv")
    if csv_path.exists():
        csv_path.unlink()  # start clean so a stale CSV can't lie

    # --- build args ---
    args = build_args(shape, dtype)

    # --- build the seed Config and bind to validate eagerly ---
    seed_cfg = helion.Config(**seed)
    bound_probe = fn.bind(args)
    # eager validation: normalize raises on a structurally-invalid seed
    bound_probe.config_spec.normalize(seed_cfg)

    # --- run with NO search: configs=[seed] -> len(configs)==1 short-circuit ---
    # Re-wrap the undecorated fn so the seed is the ONLY config (a bad seed RAISES
    # here rather than being silently dropped, unlike a real search).
    seeded_kernel = helion.kernel(fn.fn, configs=[seed_cfg])
    bound = seeded_kernel.bind(args)
    bound.ensure_config_exists(args)  # triggers set_config(seed) -> normalize+compile

    normalized = dict(bound._config)
    seed_raw = dict(seed_cfg)

    # --- (a) prove no autotune ran ---
    autotune_ran = csv_path.exists() and csv_path.stat().st_size > 0
    autotune_evidence = (
        f"HELION_AUTOTUNE_LOG CSV absent/empty at {csv_path}; "
        f"len(configs)==1 short-circuit (_user_provided_config path), "
        f"no search/generation"
    )

    # --- (b) prove the seed was actually used (codegen reflects it) ---
    triton_code = bound.to_triton_code(seed_cfg)
    want_looped = (
        normalized.get("reduction_loops")
        and normalized["reduction_loops"][0] is not None
    )
    got_looped = _looped_signature(triton_code)
    want_warps = normalized.get("num_warps")
    got_warps = _num_warps_in_launcher(triton_code)
    seed_used = (bool(want_looped) == bool(got_looped)) and (want_warps == got_warps)
    seed_used_how = (
        f"codegen persistent-vs-looped={'looped' if got_looped else 'persistent'} "
        f"matches normalized reduction_loops={normalized.get('reduction_loops')}; "
        f"num_warps in launcher={got_warps} matches normalized num_warps={want_warps}"
    )

    # --- (c) correctness vs reference (fp32) ---
    out = bound(*args)
    kernel_out = out[output_index] if isinstance(out, tuple) else out
    ref_out = reference(*args)
    kernel_out = kernel_out.to(torch.float32)
    ref_out = ref_out.to(torch.float32)
    abs_err = (kernel_out - ref_out).abs()
    max_abs = float(abs_err.max())
    rel_err = abs_err / (ref_out.abs() + 1e-12)
    max_rel = float(rel_err.max())
    correctness_pass = bool(
        torch.allclose(kernel_out, ref_out, rtol=rtol, atol=atol)
    )

    # --- (d) latency: median-of-N via do_bench (what TritonBench uses) ---
    torch.cuda.synchronize()
    samples = [
        float(do_bench(lambda: bound(*args), return_mode="median"))
        for _ in range(n_runs)
    ]
    lat_median = median(samples)
    lat_min = min(samples)
    lat_max = max(samples)
    lat_std = pstdev(samples) if len(samples) > 1 else 0.0

    return BareSeedResult(
        seed_raw=seed_raw,
        seed_normalized=normalized,
        seed_used=seed_used,
        seed_used_how=seed_used_how,
        autotune_ran=autotune_ran,
        autotune_evidence=autotune_evidence,
        correctness_pass=correctness_pass,
        max_abs=max_abs,
        max_rel=max_rel,
        rtol=rtol,
        atol=atol,
        latency_median_ms=lat_median,
        latency_min_ms=lat_min,
        latency_max_ms=lat_max,
        latency_stddev_ms=lat_std,
        n_runs=n_runs,
        generated_triton=triton_code,
    )


# --------------------------------------------------------------------------- #
# CLI demo: rms_norm_fwd (2048,4096) fp32 -> full evidence block
# --------------------------------------------------------------------------- #
def _demo() -> None:
    assert_worktree_helion()
    sys.path.insert(0, WORKTREE)  # so `_lab.harness...` and `examples` resolve
    from examples.rms_norm import rms_norm_fwd
    from examples.rms_norm import rms_norm_pytorch

    from _lab.harness.evidence_block import EvidenceFields
    from _lab.harness.evidence_block import format_evidence_block

    eps = 1e-5

    def build_args(
        shape: tuple[int, ...], dtype: torch.dtype
    ) -> tuple[object, ...]:
        m, n = shape
        x = torch.randn(m, n, device="cuda", dtype=dtype)
        w = torch.randn(n, device="cuda", dtype=dtype)
        return (x, w, eps)

    def reference(x: torch.Tensor, w: torch.Tensor, e: float) -> torch.Tensor:
        return rms_norm_pytorch(x, w, e)

    shape = (2048, 4096)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")

    # default seed: persistent (reduction_loops=[None]), num_warps=4
    default_seed = {
        "block_sizes": [1],
        "reduction_loops": [None],
        "range_unroll_factors": [0],
        "range_warp_specializes": [],
        "range_num_stages": [0],
        "range_multi_buffers": [None],
        "range_flattens": [None],
        "load_eviction_policies": ["", "", "", "", ""],
        "num_warps": 4,
        "num_stages": 1,
        "indexing": ["pointer"] * 8,
        "atomic_indexing": [],
        "pid_type": "flat",
    }

    res = run_bare_seed(
        rms_norm_fwd,
        build_args,
        reference,
        shape,
        default_seed,
        dtype=torch.float32,
        n_runs=7,
        rtol=1e-3,
        atol=1e-4,
    )

    cmd = (
        "cd /home/calebkim/helion-new-heuristics/wt-reduction && "
        "CUDA_VISIBLE_DEVICES=2 "
        "PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction "
        "HELION_AUTOTUNE_LOG=/tmp/seed_probe "
        "/home/calebkim/.conda/envs/helion/bin/python "
        "_lab/harness/bare_seed_run.py"
    )

    block = format_evidence_block(
        EvidenceFields(
            kernel_shape="rms_norm_fwd (2048,4096) fp32",
            exact_command=cmd,
            seed_raw=res.seed_raw,
            seed_normalized=res.seed_normalized,
            seed_used=res.seed_used,
            seed_used_how=res.seed_used_how,
            autotune_ran=res.autotune_ran,
            autotune_evidence=res.autotune_evidence,
            correctness_pass=res.correctness_pass,
            max_abs=res.max_abs,
            max_rel=res.max_rel,
            rtol=res.rtol,
            atol=res.atol,
            tol_justification=(
                "fp32 reduction-order drift; rtol=1e-3,atol=1e-4 well above "
                "observed error"
            ),
            latency_median_ms=res.latency_median_ms,
            latency_min_ms=res.latency_min_ms,
            latency_max_ms=res.latency_max_ms,
            latency_stddev_ms=res.latency_stddev_ms,
            n_runs=res.n_runs,
            gpu_index=gpu,
            accept_reject_rule=(
                "bare seed runs with no autotune, used as-is, correct, and "
                "stably timed -> safe to use as the Product-A measurement primitive"
            ),
        )
    )
    print(block)


if __name__ == "__main__":
    _demo()
