"""OFFLINE numerical model of size_reduction_tiles, driven by /tmp/corpus_facts.json.

Two jobs:
  1. `current_block_sizes(rec)` — replicate the CURRENT allocator's block_sizes vector exactly,
     validated against the recorded seed (/tmp/before_rewrite.json) so the model is trusted.
  2. `candidate_block_sizes(rec)` — the TWO-REGIME additive-Σ footprint (the #1 rewrite), so its
     corpus-wide config diff vs current can be inspected BEFORE editing helion.

No GPU, no bind — pure arithmetic over the dumped facts. Run:
  python model_alloc.py --facts /tmp/corpus_facts.json [--mode current|candidate|diff]
"""
from __future__ import annotations

import argparse
import json


def np2(n: int) -> int:
    n = max(1, int(n))
    p = 1
    while p < n:
        p <<= 1
    return p


def pp2(n: int) -> int:
    n = max(1, int(n))
    p = 1
    while p * 2 <= n:
        p <<= 1
    return p


# Budget constants (mirror _TritonReductionSeedBase).
ROW = 245760
LIVE = 3 * 245760
LOOPED = 16384
MIN_WAVES = 8
WIDEN_MAX_ROWS = 8

FULL_EXTENT = {"full_slice", "full_grid"}
SIZED = {"full_slice", "full_grid", "user_tile"}


def _ext_of(rec, bid):
    """np2-padded extent for a block_id (matches extent_of)."""
    e = rec["ext"].get(str(bid))
    if e is None:
        return 1
    return np2(e)


def _is_carried_reduction_tile(acc, reduction_ids):
    dims = acc["dims"]
    return len(dims) >= 2 and any(d in reduction_ids for d in dims)


def _alloc(rec, *, candidate: bool):
    """Compute the full block_sizes vector. `candidate` selects the additive-Σ two-regime
    footprint; otherwise the current num_live×carried_mult×prod footprint."""
    if "kf" in rec or rec.get("error") or rec.get("pd_block_id") is None:
        return None
    reds = rec["reductions"]
    by_bid = {d["block_id"]: d for d in reds}
    reduction_ids = set(rec["reduction_ids"])
    grid_ids = set(rec["grid_ids"])
    nrl_ids = set(rec["nrl_ids"])
    accs = rec["accumulators"]
    itemsize = max(1, rec["pd_itemsize"])
    pd_bid = rec["pd_block_id"]
    pd_cat = rec["pd_category"]
    bs_valid = set(rec["bs_valid"])
    grid_rows = rec["grid_rows"]
    num_sm = rec["num_sm"]
    occ_floor = num_sm * MIN_WAVES
    element_cap = rec["element_cap"]
    feature_axes = rec["feature_axes"]
    feature_footprint = 1
    for v in feature_axes.values():
        feature_footprint *= max(1, v or 1)

    # provisional grid seats
    seated = {int(k): v for k, v in rec["grid_seats"].items()}
    sizes = {}
    red_values = {}
    primary_r_block = 1
    persistent = False

    def in_acc(bid):
        return any(bid in a["dims"] for a in accs)

    # ---- the carried-regime detector (regime selection for the candidate) ----
    has_carried_red_tile = any(_is_carried_reduction_tile(a, reduction_ids) for a in accs)

    # group iteration order
    groups = rec["groups"]

    def group_key(idxs):
        descs = [reds[i] for i in idxs]
        sized = [d for d in descs if d["category"] in SIZED]
        n_full = sum(1 for d in sized if d["category"] in FULL_EXTENT)
        max_ext = max([d["size_hint"] for d in sized], default=0)
        return (-n_full, -max_ext)

    for idxs in sorted(groups, key=group_key):
        descs = [reds[i] for i in idxs]
        sized = [d for d in descs if d["category"] in SIZED]
        if not sized:
            continue
        num_live = max([d["body_live_tiles"] for d in sized] + [1])

        def group_footprint_excluding(axis, _num_live, *, _candidate=candidate):
            """Returns the budget denominator for sizing `axis`."""
            if not _candidate:
                # ----- CURRENT formula -----
                prod = feature_footprint
                for d2 in sized:
                    if d2["block_id"] == axis or d2["category"] == "grid_tile":
                        continue
                    prod *= max(1, seated.get(d2["block_id"], _ext_of(rec, d2["block_id"])))
                for gbid in grid_ids:
                    if gbid == axis:
                        continue
                    if in_acc(gbid):
                        prod *= max(1, seated.get(gbid, 1))
                carried_mult = 1
                if feature_footprint == 1:
                    carried_mult = max(
                        1,
                        sum(
                            1
                            for a in accs
                            if _is_carried_reduction_tile(a, reduction_ids)
                            and axis in a["dims"]
                        ),
                    )
                return max(1, itemsize * _num_live * carried_mult * prod)
            else:
                # ----- CANDIDATE two-regime additive-Σ footprint -----
                return _candidate_footprint(axis, _num_live)

        def _candidate_footprint(axis, _num_live):
            # The resident working set per element of `axis`, times num_live (the body's live-tile
            # peak). Common multiplicative base = the materialized feature tensors + the OTHER
            # seated reductions in this group (resident in every working tile).
            prod = feature_footprint
            for d2 in sized:
                if d2["block_id"] == axis or d2["category"] == "grid_tile":
                    continue
                prod *= max(1, seated.get(d2["block_id"], _ext_of(rec, d2["block_id"])))
            if has_carried_red_tile:
                # CARRIED regime: the loop-carried reduction tiles are SEPARATE resident buffers
                # that ADD (jsd carries 2 -> tighter; grpo's working tile vs accumulator add).
                # Σ over carried buffers (∏ buffer dims EXCEPT axis, classified by membership).
                # This single Σ subsumes the old `carried_mult` (= buffer count for [grid,R]
                # buffers) AND the lossy `in_accumulator`-multiplied grid term (the grid dim is now
                # a buffer dim counted inside the ∏).
                carried_resident = 0
                for a in accs:
                    if not _is_carried_reduction_tile(a, reduction_ids):
                        continue
                    buf = 1
                    for d in a["dims"]:
                        if d is None or d == axis:
                            continue
                        if d in reduction_ids:
                            buf *= max(1, seated.get(d, _ext_of(rec, d)))
                        elif d in grid_ids:
                            buf *= max(1, seated.get(d, 1))
                        else:
                            buf *= max(1, _ext_of(rec, d))
                    carried_resident += buf
                prod *= max(1, carried_resident)
            else:
                # STREAMED regime: the resident grid rows enter MULTIPLICATIVELY (one working
                # tile). Same in-accumulator membership test as before.
                for gbid in grid_ids:
                    if gbid == axis:
                        continue
                    if in_acc(gbid):
                        prod *= max(1, seated.get(gbid, 1))
            return max(1, itemsize * _num_live * prod)

        order = sorted(
            sized,
            key=lambda d: (
                0 if d["category"] in FULL_EXTENT else (1 if d["category"] == "user_tile" else 2),
                -d["size_hint"],
            ),
        )
        for d in order:
            raw_ext = d["size_hint"]
            ext = _ext_of(rec, d["block_id"])
            if d["category"] == "grid_tile":
                seated[d["block_id"]] = 1
                continue
            coeff_single = group_footprint_excluding(d["block_id"], 1)
            coeff_live = group_footprint_excluding(d["block_id"], num_live)
            ext_held = (
                d["row_reread"]
                and d["carried_2d_count"] == 0
                and (element_cap is None or raw_ext <= element_cap)
                and coeff_single * raw_ext <= ROW
                and coeff_live * raw_ext <= LIVE
            )
            if ext_held:
                r = ext
            else:
                byte_budget = pp2(max(1, ROW // coeff_live))
                r = max(1, min(LOOPED, byte_budget, ext))
            seated[d["block_id"]] = r
            if d["block_id"] == pd_bid:
                primary_r_block = r
                persistent = r >= ext and d["category"] in FULL_EXTENT
            if d["block_id"] in bs_valid:
                red_values[d["block_id"]] = r

        # grid-M widen
        for mbid in sorted(grid_ids):
            if mbid not in bs_valid:
                continue
            ext = _ext_of(rec, mbid)
            floor = rec["floor"].get(str(mbid), 1)
            reduced_away = len(accs) > 0 and not in_acc(mbid)
            if reduced_away:
                collapse = np2(max(1, grid_rows // num_sm)) if grid_rows > 0 else 1
                blk = max(floor, min(collapse, ext))
            else:
                is_reduce_then_apply = bool(nrl_ids)
                widen_live = 1 if is_reduce_then_apply else num_live
                byte_widen = pp2(max(1, ROW // group_footprint_excluding(mbid, widen_live)))
                if grid_rows > 0:
                    occ_widen = pp2(max(1, grid_rows // occ_floor))
                else:
                    occ_widen = 1
                rows_ceiling = ext if pd_cat == "full_grid" else WIDEN_MAX_ROWS
                blk = max(floor, min(byte_widen, occ_widen, rows_ceiling, ext))
            seated[mbid] = blk
            sizes[mbid] = blk

    # non-reduction / independent loops
    loop_budget = pp2(max(1, ROW // max(1, itemsize * feature_footprint)))
    # need spec block order; we only have ext/floor per bid. Reconstruct vector from bs order:
    bs_order = rec["bs_valid"]  # NOTE: bs_valid is sorted block_ids, not spec index order!
    # We can't perfectly reconstruct spec index order from facts, but the recorded seed gives it.
    # For independent loops we mirror the logic per-bid:
    for bid in bs_order:
        if bid in red_values or bid in grid_ids or bid in reduction_ids:
            continue
        if bid in nrl_ids or bid not in seated:
            sizes[bid] = max(1, min(_ext_of(rec, bid), loop_budget))

    return {
        "block_sizes_by_bid": {**{b: red_values[b] for b in red_values}, **sizes},
        "primary_r_block": primary_r_block,
        "persistent": persistent,
        "red_values": red_values,
        "sizes": sizes,
    }


def _ext_of_raw(rec, bid):
    e = rec["ext"].get(str(bid))
    return e if e is not None else 1


def reconstruct_vector(rec, alloc):
    """Build the block_sizes list in spec INDEX order (matches the recorded seed)."""
    order = rec.get("bs_index_to_bid")
    if order is None:
        return None
    sizes = alloc["sizes"]
    red_values = alloc["red_values"]
    out = []
    for bid in order:
        if bid in sizes:
            out.append(sizes[bid])
        elif bid in red_values:
            out.append(red_values[bid])
        else:
            out.append(rec["floor"].get(str(bid), 1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--facts", default="/tmp/corpus_facts.json")
    p.add_argument("--mode", default="diff", choices=["current", "candidate", "diff", "validate"])
    p.add_argument("--kernels", default="")
    a = p.parse_args()
    facts = json.load(open(a.facts))
    kf = set(a.kernels.split(",")) if a.kernels else None

    def key(r):
        return f"{r['corpus']}/{r['kernel']}/{tuple(r['shape'])}/{r['dtype']}"

    if a.mode == "validate":
        # Reconstruct the current-model block_sizes vector in spec index order and compare to the
        # recorded seed's block_sizes. Mismatches mean the OFFLINE model diverges from the live
        # allocator (a model bug to fix before trusting the candidate diff).
        n_ok = n_bad = n_skip = 0
        for r in facts:
            if kf and r.get("kernel") not in kf:
                continue
            m = _alloc(r, candidate=False)
            if m is None:
                n_skip += 1
                continue
            seed = r.get("seed") or {}
            vec = reconstruct_vector(r, m)
            seed_bs = seed.get("block_sizes")
            if seed_bs is None:
                n_skip += 1
                continue
            if vec == seed_bs:
                n_ok += 1
            else:
                n_bad += 1
                print(f"MISMATCH {key(r)}\n   model={vec}\n   seed ={seed_bs}")
        print(f"\n=== validate: {n_ok} match, {n_bad} mismatch, {n_skip} skipped ===")
        return

    n_diff = 0
    for r in facts:
        if kf and r.get("kernel") not in kf:
            continue
        cur = _alloc(r, candidate=False)
        if cur is None:
            continue
        if a.mode == "current":
            print(f"{key(r):55s} {cur['block_sizes_by_bid']} r={cur['primary_r_block']}")
            continue
        cand = _alloc(r, candidate=True)
        if a.mode == "candidate":
            print(f"{key(r):55s} {cand['block_sizes_by_bid']} r={cand['primary_r_block']}")
            continue
        # diff
        if cur["block_sizes_by_bid"] != cand["block_sizes_by_bid"]:
            n_diff += 1
            print(f"DIFF {key(r)}")
            print(f"   cur ={cur['block_sizes_by_bid']} r={cur['primary_r_block']}")
            print(f"   cand={cand['block_sizes_by_bid']} r={cand['primary_r_block']}")
    if a.mode == "diff":
        print(f"\n=== {n_diff} cells differ (candidate vs current) ===")


if __name__ == "__main__":
    main()
