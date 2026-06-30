"""#2/#3 zero-diff guard: for every corpus kernel, compare the CURRENT position-based
_carried_leading_dims / _carried_m_block_cap against the proposed MEMBERSHIP-based versions,
per grid axis, AND confirm the final size_axis(grid M) result is identical.

Membership rules (CARRIED_AND_GREEDY_FINDINGS #2/#3):
 - carried grid dims = any grid axis appearing ANYWHERE in a >=2D carried accumulator's dims
   (not just dim_block_ids[0]).
 - carried_m_block_cap: a buffer carries M iff m_axis in dim_block_ids (not == [0]); footprint
   contribution per buffer = PRODUCT of its OTHER tiled dims classified by MEMBERSHIP (rdim ->
   sized r_block else padded extent; other block_id -> padded extent; None static dim -> 1, as
   today it is unrecoverable from dim_block_ids); SUM across buffers; M_BLOCK <= budget /
   (Σ_buffers(∏ other) * itemsize).

Usage (from /tmp): PYTHONPATH=/home/dev/local/helion-redesign python carried_membership_trace.py
"""

from __future__ import annotations

import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(_HARNESS_DIR, "..", "harness")))
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep)

import unified_config_recorder as REC  # noqa: E402
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    _TritonReductionSeedBase as B,
)
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    _primary_descriptor_selected,
)
from helion._utils import next_power_of_2 as np2  # noqa: E402
from helion._utils import prev_power_of_2 as pp2  # noqa: E402


def _is_carried_reduction_acc(a, rdims) -> bool:
    """A buffer is a CARRIED reduction tile iff it is >=2D and contains >=1 reduction dim
    (an rdim). A per-row scalar accum ([M_BLOCK, None]) or a pure grid-product accum (grpo's
    [0,1], no rdim) is NOT a carried reduction footprint."""
    return len(a.dim_block_ids) >= 2 and any(d in rdims for d in a.dim_block_ids)


def membership_leading_dims(spec) -> set[int]:
    """Any grid axis appearing ANYWHERE in a >=2D CARRIED REDUCTION accumulator's dims."""
    kf = spec.reduction_kernel_fact
    grid = set(kf.grid_axis_block_ids) if kf else set()
    rdims = {d.block_id for d in kf.reductions} if kf else set()
    out: set[int] = set()
    for a in spec.accumulator_facts:
        if _is_carried_reduction_acc(a, rdims):
            for d in a.dim_block_ids:
                if d is not None and d in grid:
                    out.add(d)
    return out


def membership_m_cap(env, spec, m_axis, red_values) -> int:
    kf = spec.reduction_kernel_fact
    rdims = {d.block_id for d in kf.reductions} if kf else set()
    total = 0
    itemsize = 1
    for a in spec.accumulator_facts:
        if not _is_carried_reduction_acc(a, rdims) or m_axis not in a.dim_block_ids:
            continue
        prod = 1
        for d in a.dim_block_ids:
            if d == m_axis or d is None:
                # the axis we solve for; a None static dim is unrecoverable from dim_block_ids
                # (== today's behavior: only block-id'd dims contribute), so skip both.
                continue
            if d in rdims:
                w = red_values.get(d)
                if w is None:
                    w = np2(env.block_sizes[d].size_hint())
            else:
                w = np2(env.block_sizes[d].size_hint())
            prod *= max(1, w)
        total += prod
        itemsize = max(itemsize, a.itemsize)
    if total <= 0:
        return 1 << 30
    return max(1, pp2(max(1, B.CARRIED_TILE_MAX_BYTES // max(1, total * max(1, itemsize)))))


def main() -> None:
    print(f"helion={helion.__file__}\n")
    mismatches = 0
    for corpus in REC._CORPORA:
        for (cps, kn, shape, dtype, fn, kargs, _split) in REC._CORPORA[corpus](None):
            bound = fn.bind(kargs)
            env = bound.env
            spec = env.config_spec
            with env:
                # any >=2D accumulator at all?
                has2d = any(len(a.dim_block_ids) >= 2 for a in spec.accumulator_facts)
                if not has2d:
                    del bound
                    torch.cuda.empty_cache()
                    continue
                pd = _primary_descriptor_selected(env)
                if pd is None:
                    del bound
                    torch.cuda.empty_cache()
                    continue
                # reconstruct red_values exactly as the allocator does
                rv = {} if pd.category.value in ("full_slice", "full_grid") else {
                    pd.block_id: B._reduction_rblock(env, pd, B._m_block_product(spec, pd))[0]
                }
                rv.update(
                    B._secondary_red_values(
                        env, spec, pd, B._m_block_product(spec, pd), exclude_block_id=pd.block_id
                    )
                )
                from helion._compiler.autotuner_heuristics.triton import Cap, size_axis

                cur_ld = B._carried_leading_dims(spec)
                new_ld = membership_leading_dims(spec)
                # r_block_resident for the primary (mirror _build_block_sizes)
                if pd.category.value == "full_slice":
                    rbr = np2(pd.size_hint)
                elif pd.category.value == "user_tile":
                    rbr = rv.get(pd.block_id, 1) or 1
                else:
                    rbr = 1
                n_carried = pd.carried_2d_count
                is_carried = n_carried >= 1
                carried_cap = max(
                    1, B.CARRIED_TILE_MAX_BYTES // (max(1, rbr) * max(1, pd.itemsize) * max(1, n_carried))
                )
                for mbid in B._grid_axis_block_ids(spec):
                    if mbid not in spec.block_sizes.valid_block_ids():
                        continue
                    idx = spec.block_sizes.block_id_to_index(mbid)
                    bs_spec = spec.block_sizes[idx]
                    inner = B._pinned_inner_resident_elems(spec, pd, mbid)

                    def final(co_set, m_cap_fn):
                        co = (not is_carried) or (mbid in co_set)
                        arbr = rbr if co else 1
                        ca = is_carried and co
                        caps = [
                            Cap("resident_tile", True, B._resident_tile_cap(spec, pd, inner, r_block_resident=arbr)),
                            Cap("carried_2d", ca, pp2(carried_cap)),
                            Cap("carried_m", True, m_cap_fn),
                            Cap("occupancy", True, B._m_axis_occupancy_cap(env, pd, mbid)),
                            Cap("extent", True, np2(bs_spec.size_hint)),
                            Cap("m_block_register", True, B._m_block_cap(spec, pd)),
                        ]
                        return size_axis(B._block_floor(bs_spec), caps)

                    cur_v, cur_b = final(cur_ld, B._carried_m_block_cap(spec, pd, mbid, rv))
                    new_v, new_b = final(new_ld, membership_m_cap(env, spec, mbid, rv))
                    tag = "OK" if cur_v == new_v else "DIFF"
                    if cur_v != new_v:
                        mismatches += 1
                    print(
                        f"  [{tag}] {cps}/{kn}/{shape} m_axis={mbid} "
                        f"FINAL cur={cur_v}({cur_b}) new={new_v}({new_b}) | "
                        f"carried_m cur={B._carried_m_block_cap(spec, pd, mbid, rv)} "
                        f"new={membership_m_cap(env, spec, mbid, rv)} | "
                        f"ld cur={sorted(cur_ld)} new={sorted(new_ld)}"
                    )
            del bound
            torch.cuda.empty_cache()
    print(f"\n=== {mismatches} axes where cur != new (value or co_holds) ===")


if __name__ == "__main__":
    main()
