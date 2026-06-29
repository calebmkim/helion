"""P4 Tier-1 assertion harness — the durable fired-right-path check for the 13 probes.

Encodes each probe's EXPECTED Stage-1 classification + which heuristic should fire (the
docstring's "IMPLEMENTER'S ASSERTIONS / Tier 1"). Tier-2 (perf >= default) is in probe_perf.py
(GPU). This is the structural check: it binds each probe and asserts the categorization landed
on the intended taxonomy point and the right path fired (NOT silently the default).

Run from /tmp:
  HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=<worktree> \
    python probe_assertions.py
Exit 0 = all GREEN; exit 1 = a probe regressed its fired-right-path.
"""

from __future__ import annotations

import os
import sys

_HARNESS = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS, "..", ".."))
if _HARNESS not in sys.path:
    sys.path.insert(0, _HARNESS)

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
from helion.autotuner.config_spec import ReductionCategory as RC  # noqa: E402

import ir_introspect as II  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep)


def _cats(spec):
    """Map block_id -> ReductionCategory from the kernel fact."""
    kf = spec.reduction_kernel_fact
    if kf is None:
        return {}, []
    cats = {}
    for d in kf.reductions:
        cats.setdefault(d.block_id, d.category)
    return cats, kf.coresidency_groups


def _check(slug, expect):
    """expect: dict with keys fired (heuristic name or None), categories (set of category
    names that must ALL be present among the reductions), and optional same_group / n_groups."""
    fn, args = II._probe_fn_args(slug)
    bound = fn.bind(args)
    spec = bound.env.config_spec
    fails = []
    with bound.env:
        fired = list(spec.autotuner_heuristics)
        cats, groups = _cats(spec)
        present = {c.value for c in cats.values()}
        # fired-right-path
        want_fired = expect.get("fired")
        if want_fired is None:
            if fired:
                fails.append(f"expected DECLINE, got fired={fired}")
        else:
            if want_fired not in fired:
                fails.append(f"expected fired {want_fired!r}, got {fired}")
        # categories present
        for c in expect.get("categories", set()):
            if c not in present:
                fails.append(f"category {c} absent (present={sorted(present)})")
        # co-residency
        if "n_groups" in expect and len(groups) != expect["n_groups"]:
            fails.append(f"n_groups={len(groups)} expected {expect['n_groups']}")
    del bound
    torch.cuda.empty_cache()
    return fails


# Tier-1 expectations per probe (from the kernel docstrings + the verified IR).
_EXPECT = {
    "p1-outer-product-coresident": {
        "fired": "triton_reduction_tile",
        "categories": {"full_slice"},
        "n_groups": 1,  # two FULL_SLICE co-resident in ONE group
    },
    "p2-feature-plus-rowaccum-offcorpus": {
        "fired": "triton_reduction_tile",
        "categories": {"full_slice", "grid_tile"},  # feature + cross-row accum, co-resident
    },
    "p3-full-grid-nonquant": {
        "fired": "triton_reduction_tile",
        "categories": {"full_grid"},
    },
    "p4-two-rollable-sequential": {
        "fired": "triton_reduction_tile",
        "categories": {"full_slice"},
        "n_groups": 2,  # two sequential rollable reductions
    },
    "p5-3d-reduction-tile": {
        "fired": "triton_reduction_tile",
        "categories": {"full_slice"},
    },
    "p6-mixed-coresident-plus-sequential": {
        "fired": "triton_reduction_tile",
        "categories": {"full_slice", "grid_tile"},
        "n_groups": 2,  # co-resident pair + a sequential pass
    },
    "p7-gridtile-then-usertile": {
        "fired": "triton_reduction_tile",
        "categories": {"full_slice"},
        "n_groups": 2,  # two sequential rolled reductions (the relaxed gate)
    },
    "p8-fullgrid-plus-usertile": {
        "fired": "triton_reduction_tile",
        "categories": {"full_grid", "full_slice"},
    },
    "p9-nonred-loop-then-fullextent": {
        "fired": "triton_reduction_tile",
        "categories": {"full_slice"},
    },
    "p10-usertile-and-gridtile": {
        "fired": "triton_reduction_tile",
        "categories": {"full_slice"},
    },
    "p11-fullextent-then-nonred-loop": {
        "fired": "triton_reduction_tile",
        "categories": {"full_slice"},
    },
    "oos1-jagged-declined": {
        "fired": None,  # correctly DECLINED
    },
    "oos2-strided-dim0": {
        "fired": "triton_reduction_tile",  # the known cliff, still fires (left as-is)
        "categories": {"full_slice"},
    },
}


def main() -> None:
    print(f"helion={helion.__file__}\n", flush=True)
    n_pass = n_fail = 0
    for slug, expect in _EXPECT.items():
        try:
            fails = _check(slug, expect)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            fails = [f"EXC {type(e).__name__}: {e}"]
        if fails:
            n_fail += 1
            print(f"[FAIL] {slug}\n       " + "\n       ".join(fails), flush=True)
        else:
            n_pass += 1
            print(f"[ ok ] {slug}", flush=True)
    print(f"\n=== Tier-1: {n_pass} pass, {n_fail} fail ===", flush=True)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
