"""Canonical EVIDENCE BLOCK formatter (Step 0 house style).

Every agent that reports a bare-seed measurement emits an identical receipt via
``format_evidence_block`` so the hub / referee can audit at a glance. This is the
fixed format defined by the measurement-harness-verifier (Step 0, Deliverable D).

The block is intentionally plain text (not JSON) so it reads cleanly inline in an
agent message; the structured numbers behind it live in the ledger.

Usage::

    from _lab.harness.evidence_block import format_evidence_block, EvidenceFields
    print(format_evidence_block(EvidenceFields(...)))
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class EvidenceFields:
    """All fields of one EVIDENCE BLOCK. Keep names stable: agents parse these."""

    kernel_shape: str  # e.g. "rms_norm_fwd (2048,4096) fp32"
    exact_command: str  # copy-pasteable command that reproduces this
    seed_raw: dict[str, Any]  # the seed Config as a dict, before normalize()
    seed_normalized: dict[str, Any]  # bound_kernel._config dict, after normalize()
    seed_used: bool  # was the seed actually the config that ran?
    seed_used_how: str  # how that was verified (codegen match / _config inspect)
    autotune_ran: bool  # should always be False for a bare-seed run
    autotune_evidence: str  # e.g. "CSV absent at <path>; short-circuit len(configs)==1"
    correctness_pass: bool
    max_abs: float
    max_rel: float
    rtol: float
    atol: float
    tol_justification: str  # justify any loosening; "" if defaults used
    latency_median_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_stddev_ms: float
    n_runs: int
    gpu_index: str  # the CUDA_VISIBLE_DEVICES index used
    measured_by: str = "do_bench"
    accept_reject_rule: str = ""  # the rule this evidence supports
    spread_style: str = "minmax"  # "minmax" or "stddev" — which to show inline


def _yn(b: bool) -> str:
    return "YES" if b else "NO"


def format_evidence_block(f: EvidenceFields) -> str:
    """Render the fixed EVIDENCE BLOCK. Stable layout — do not reorder lines."""
    if f.spread_style == "stddev":
        spread = f"stddev={f.latency_stddev_ms:.5f}ms"
    else:
        spread = f"[{f.latency_min_ms:.5f}..{f.latency_max_ms:.5f}]"

    seed_used_line = (
        f"{f.seed_normalized}   seed_used: {_yn(f.seed_used)}  ({f.seed_used_how})"
    )

    corr_pass = "PASS" if f.correctness_pass else "FAIL"
    tol = f"tol(rtol={f.rtol:g},atol={f.atol:g})"
    corr = (
        f"{corr_pass}  max_abs={f.max_abs:.3e} max_rel={f.max_rel:.3e}  {tol}"
    )
    if f.tol_justification:
        corr += f"  [{f.tol_justification}]"

    autotune = f"{_yn(f.autotune_ran)}  (evidence: {f.autotune_evidence})"

    latency = (
        f"median={f.latency_median_ms:.5f}ms  spread={spread}  "
        f"N={f.n_runs}  GPU={f.gpu_index}  measured_by={f.measured_by}"
    )

    lines = [
        "EVIDENCE BLOCK",
        f" kernel/shape:        {f.kernel_shape}",
        f" exact command:       {f.exact_command}",
        f" seed (raw):          {f.seed_raw}",
        f" seed (normalized):   {seed_used_line}",
        f" autotune ran:        {autotune}",
        f" correctness:         {corr}",
        f" latency:             {latency}",
        f" accept/reject rule:  {f.accept_reject_rule}",
    ]
    return "\n".join(lines)
