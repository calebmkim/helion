"""Round-trip proof for the persistent-seed (reduction_loops=[None]) autotuner
flat-encode fix.

Binds a real reduction kernel (rms_norm / long_sum) at several rnumel and asserts
that the persistent choice survives the autotuner's flatten/unflatten round-trip:

    unflatten(flatten(Config(..., reduction_loops=[None]))) == [None]

PRE-FIX this fails for rnumel>4096 (None -> looped chunk 4096) because
ReductionLoopSpec._encode_flat_value(None) returned the fragment default (capped
at 4096), and unflatten only restores None when flat_int >= size_hint.

POST-FIX it returns the fragment .high (= next_power_of_2(size_hint) >= size_hint),
the one flat value that decodes back to None for ALL size_hints.

We also keep a GUARD that size_hint <= 4096 still round-trips None (must not break).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.long_sum import longsum  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402

from helion import Config  # noqa: E402
from helion.autotuner.config_generation import ConfigGeneration  # noqa: E402


def build_rms_norm(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, 1e-5)


def build_reduce(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def roundtrip_reduction_loops(bound, rl_value):
    """Return reduction_loops after one flatten/unflatten round-trip."""
    cg = ConfigGeneration(bound.env.config_spec)
    # Build a Config that carries reduction_loops=rl_value. We start from the
    # default config (so all OTHER fields are valid for this kernel) and only
    # override reduction_loops.
    default_cfg = bound.env.config_spec.default_config()
    base = dict(default_cfg)
    n_loops = len(base.get("reduction_loops", [None]))
    base["reduction_loops"] = [rl_value] * max(1, n_loops)
    cfg = Config(**base)
    norm = cg.unflatten(cg.flatten(cfg))
    return norm.get("reduction_loops")


def size_hint_of(bound):
    """The single reduction loop's size_hint for these 1-reduction-loop kernels."""
    rl = bound.env.config_spec.reduction_loops
    return [spec.size_hint for spec in rl]


def main():
    cases = [
        # (label, build, kernel_fn, m, n)  -- n == rnumel
        ("rms_norm", build_rms_norm, rms_norm_fwd, 2048, 8192),    # >4096
        ("rms_norm", build_rms_norm, rms_norm_fwd, 2048, 65536),   # >4096
        ("rms_norm", build_rms_norm, rms_norm_fwd, 2048, 262144),  # >4096
        ("long_sum", build_reduce, longsum, 8, 8192),              # >4096
        ("rms_norm", build_rms_norm, rms_norm_fwd, 2048, 256),     # <=4096 GUARD
        ("rms_norm", build_rms_norm, rms_norm_fwd, 2048, 4096),    # ==4096 GUARD
    ]

    print(f"helion={helion.__file__}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print("=" * 92)
    print(f"{'kernel':10} {'rnumel':>8} {'size_hint':>10} "
          f"{'[None]->':>10} {'verdict':>14}")
    print("-" * 92)

    all_ok = True
    for label, build, kfn, m, n in cases:
        args = build(m, n)
        k = helion.kernel(kfn.fn)
        bound = k.bind(args)
        hints = size_hint_of(bound)
        rt = roundtrip_reduction_loops(bound, None)
        # round-trips to None for every reduction loop entry?
        ok = rt == [None] * len(rt)
        verdict = "ROUND-TRIPS" if ok else "DEGRADED"
        all_ok = all_ok and ok
        print(f"{label:10} {n:>8} {str(hints):>10} {str(rt):>10} {verdict:>14}")

    print("=" * 92)
    print("ALL_ROUND_TRIP_NONE:", all_ok)
    # Exit non-zero if any case failed (so callers can detect before/after).
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
