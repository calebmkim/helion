"""Sweep candidate two-regime footprint formulas against the validated oracle (current model),
to find the EXACT additive-Σ formulation that is config-neutral on all 443 cells.

Each variant is a footprint(axis, num_live) callable; we run the full allocator with it and diff
the reconstructed block_sizes vector vs the current model. A variant with 0 diffs is the #1 spec.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/home/dev/local/helion-redesign/_lab/redesign")
import model_alloc as M  # noqa: E402

np2 = M.np2
pp2 = M.pp2
ROW, LIVE, LOOPED, MIN_WAVES, WIDEN_MAX_ROWS = (
    M.ROW,
    M.LIVE,
    M.LOOPED,
    M.MIN_WAVES,
    M.WIDEN_MAX_ROWS,
)
FULL_EXTENT, SIZED = M.FULL_EXTENT, M.SIZED


def run_variant(rec, footprint_fn):
    """Run the allocator using footprint_fn(ctx, axis, num_live) for the byte budget."""
    if "kf" in rec or rec.get("error") or rec.get("pd_block_id") is None:
        return None
    reds = rec["reductions"]
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
    feature_footprint = 1
    for v in rec["feature_axes"].values():
        feature_footprint *= max(1, v or 1)
    feature_axes = {int(k) for k in rec["feature_axes"]}
    feature_extent = {int(k): max(1, v or 1) for k, v in rec["feature_axes"].items()}
    seated = {int(k): v for k, v in rec["grid_seats"].items()}
    sizes = {}
    red_values = {}

    def ext_of(b):
        e = rec["ext"].get(str(b))
        return np2(e) if e else 1

    def in_acc(b):
        return any(b in a["dims"] for a in accs)

    def is_carried(a):
        dims = a["dims"]
        return len(dims) >= 2 and any(d in reduction_ids for d in dims)

    has_carried = any(is_carried(a) for a in accs)
    ctx = dict(
        reds=reds, reduction_ids=reduction_ids, grid_ids=grid_ids, accs=accs,
        itemsize=itemsize, feature_footprint=feature_footprint, seated=seated,
        ext_of=ext_of, in_acc=in_acc, is_carried=is_carried, has_carried=has_carried,
        feature_axes=feature_axes, feature_extent=feature_extent,
    )

    def group_key(idxs):
        descs = [reds[i] for i in idxs]
        s = [d for d in descs if d["category"] in SIZED]
        return (
            -sum(1 for d in s if d["category"] in FULL_EXTENT),
            -max([d["size_hint"] for d in s], default=0),
        )

    for idxs in sorted(rec["groups"], key=group_key):
        descs = [reds[i] for i in idxs]
        sized = [d for d in descs if d["category"] in SIZED]
        if not sized:
            continue
        num_live = max([d["body_live_tiles"] for d in sized] + [1])
        ctx["sized"] = sized

        def gfe(axis, nl):
            return footprint_fn(ctx, axis, nl)

        order = sorted(
            sized,
            key=lambda d: (
                0 if d["category"] in FULL_EXTENT else (1 if d["category"] == "user_tile" else 2),
                -d["size_hint"],
            ),
        )
        for d in order:
            raw = d["size_hint"]
            e = ext_of(d["block_id"])
            if d["category"] == "grid_tile":
                seated[d["block_id"]] = 1
                continue
            cs = gfe(d["block_id"], 1)
            cl = gfe(d["block_id"], num_live)
            held = (
                d["row_reread"]
                and d["carried_2d_count"] == 0
                and (element_cap is None or raw <= element_cap)
                and cs * raw <= ROW
                and cl * raw <= LIVE
            )
            r = e if held else max(1, min(LOOPED, pp2(max(1, ROW // cl)), e))
            seated[d["block_id"]] = r
            if d["block_id"] in bs_valid:
                red_values[d["block_id"]] = r
        for mbid in sorted(grid_ids):
            if mbid not in bs_valid:
                continue
            e = ext_of(mbid)
            floor = rec["floor"].get(str(mbid), 1)
            reduced = len(accs) > 0 and not in_acc(mbid)
            if reduced:
                coll = np2(max(1, grid_rows // num_sm)) if grid_rows > 0 else 1
                blk = max(floor, min(coll, e))
            else:
                wl = 1 if nrl_ids else num_live
                bw = pp2(max(1, ROW // gfe(mbid, wl)))
                ow = pp2(max(1, grid_rows // occ_floor)) if grid_rows > 0 else 1
                rc = e if pd_cat == "full_grid" else WIDEN_MAX_ROWS
                blk = max(floor, min(bw, ow, rc, e))
            seated[mbid] = blk
            sizes[mbid] = blk

    loop_budget = pp2(max(1, ROW // max(1, itemsize * feature_footprint)))
    for bid in rec["bs_valid"]:
        if bid in red_values or bid in grid_ids or bid in reduction_ids:
            continue
        if bid in nrl_ids or bid not in seated:
            sizes[bid] = max(1, min(ext_of(bid), loop_budget))
    return {"sizes": sizes, "red_values": red_values}


# ---------------- candidate footprint variants ----------------
def fp_current(ctx, axis, nl):
    """Replicate current exactly (sanity: must be 0-diff)."""
    seated, ext_of, in_acc = ctx["seated"], ctx["ext_of"], ctx["in_acc"]
    prod = ctx["feature_footprint"]
    for d2 in ctx["sized"]:
        if d2["block_id"] == axis or d2["category"] == "grid_tile":
            continue
        prod *= max(1, seated.get(d2["block_id"], ext_of(d2["block_id"])))
    for g in ctx["grid_ids"]:
        if g == axis:
            continue
        if in_acc(g):
            prod *= max(1, seated.get(g, 1))
    cm = 1
    if ctx["feature_footprint"] == 1:
        cm = max(
            1,
            sum(
                1
                for a in ctx["accs"]
                if ctx["is_carried"](a) and axis in a["dims"]
            ),
        )
    return max(1, ctx["itemsize"] * nl * cm * prod)


def fp_additive(ctx, axis, nl):
    """Two-regime additive-Σ. STREAMED: num_live × ∏(reductions) × feature × ∏(in-acc grid).
    CARRIED (feature==1 pure carried): num_live × Σ_buffers(∏ non-axis dims by membership).
    GRAD-COLLAPSE (feature!=1, but has carried tiles): falls back to the multiplicative working
    tile (the carried multiplicity is NOT counted — matches current cm=1)."""
    seated, ext_of, in_acc = ctx["seated"], ctx["ext_of"], ctx["in_acc"]
    feature = ctx["feature_footprint"]
    # the multiplicative working tile (streamed / grad-collapse)
    prod = feature
    for d2 in ctx["sized"]:
        if d2["block_id"] == axis or d2["category"] == "grid_tile":
            continue
        prod *= max(1, seated.get(d2["block_id"], ext_of(d2["block_id"])))
    for g in ctx["grid_ids"]:
        if g == axis:
            continue
        if in_acc(g):
            prod *= max(1, seated.get(g, 1))
    # pure carried regime: feature==1 AND a carried reduction tile holds axis
    if feature == 1 and ctx["has_carried"]:
        carried = [
            a for a in ctx["accs"] if ctx["is_carried"](a) and axis in a["dims"]
        ]
        if carried:
            total = 0
            for a in carried:
                buf = 1
                for d in a["dims"]:
                    if d is None or d == axis:
                        continue
                    if d in ctx["reduction_ids"]:
                        buf *= max(1, seated.get(d, ext_of(d)))
                    elif d in ctx["grid_ids"]:
                        buf *= max(1, seated.get(d, 1))
                    else:
                        buf *= max(1, ext_of(d))
                total += buf
            return max(1, ctx["itemsize"] * nl * total)
    return max(1, ctx["itemsize"] * nl * prod)


def fp_inline_feature(ctx, axis, nl):
    """The IMPLEMENTED form: features iterated inline (skipping fbid==axis), carried gate keys on
    `mat_feature_axes` non-empty. Must be 0-diff vs current to ship #1 config-neutral."""
    seated, ext_of, in_acc = ctx["seated"], ctx["ext_of"], ctx["in_acc"]
    feat_axes = ctx["feature_axes"]
    feat_ext = ctx["feature_extent"]
    base = 1
    for fbid in feat_axes:
        if fbid == axis:
            continue
        base *= feat_ext[fbid]
    for d2 in ctx["sized"]:
        if d2["block_id"] == axis or d2["category"] == "grid_tile":
            continue
        base *= max(1, seated.get(d2["block_id"], ext_of(d2["block_id"])))
    carried = [a for a in ctx["accs"] if ctx["is_carried"](a) and axis in a["dims"]]
    if not feat_axes and carried:
        total = 0
        for a in carried:
            buf = 1
            for d in a["dims"]:
                if d is None or d == axis:
                    continue
                if d in ctx["reduction_ids"]:
                    buf *= max(1, seated.get(d, ext_of(d)))
                elif d in ctx["grid_ids"]:
                    buf *= max(1, seated.get(d, 1))
                else:
                    buf *= max(1, ext_of(d))
            total += buf
        return max(1, ctx["itemsize"] * nl * total)
    for g in ctx["grid_ids"]:
        if g == axis:
            continue
        if in_acc(g):
            base *= max(1, seated.get(g, 1))
    return max(1, ctx["itemsize"] * nl * base)


def fp_resident_set(ctx, axis, nl, *, dedup):
    """The CLEAN unified form: footprint = itemsize × nl × Σ_resident_tensors ∏_(dim≠axis) width(dim).
    The two regimes ARE the resident-tensor set; ONE membership width(), ONE `dim==axis` skip,
    no per-kind loops, no GRID_TILE skip (sized never contains grid_tile)."""
    seated, ext_of, in_acc = ctx["seated"], ctx["ext_of"], ctx["in_acc"]
    reduction_ids, grid_ids = ctx["reduction_ids"], ctx["grid_ids"]
    feat_ext = ctx["feature_extent"]

    def width(d):
        if d in reduction_ids:
            return max(1, seated.get(d, ext_of(d)))
        if d in grid_ids:
            return max(1, seated.get(d, 1))
        return max(1, feat_ext.get(d, 1))

    carried = [a for a in ctx["accs"] if ctx["is_carried"](a) and axis in a["dims"]]
    if not feat_ext and carried:
        resident_tensors = [a["dims"] for a in carried]
    else:
        working = (
            [d["block_id"] for d in ctx["sized"]]
            + [g for g in grid_ids if in_acc(g)]
            + list(feat_ext)
        )
        resident_tensors = [set(working) if dedup else working]
    footprint = 0
    for tensor in resident_tensors:
        prod = 1
        for d in tensor:
            if d is None or d == axis:
                continue
            prod *= width(d)
        footprint += prod
    return max(1, ctx["itemsize"] * nl * footprint)


def fp_streamed_only(ctx, axis, nl):
    """NO carried regime, NO feature gate: ONE faithful formula everywhere —
        itemsize × num_live × ∏(one working tile's dims except axis).
    working tile = feature tiles (extent) + every seated reduction + in-acc grid rows.
    num_live (body_live_tiles) ALREADY counts the carried buffers among the live tiles, so there
    is no separate additive-Σ-over-buffers term (that double-counts them). The carried/streamed
    distinction then survives ONLY where it is physical: the persistence gate (carried_2d_count==0),
    which this footprint does not touch."""
    seated, ext_of, in_acc = ctx["seated"], ctx["ext_of"], ctx["in_acc"]
    grid_ids = ctx["grid_ids"]
    feat_ext = ctx["feature_extent"]

    def red_width(bid):
        if bid in grid_ids:
            return max(1, seated.get(bid, 1))
        return max(1, seated.get(bid, ext_of(bid)))

    prod = 1
    for f, fext in feat_ext.items():
        if f != axis:
            prod *= fext
    for d in ctx["sized"]:
        if d["block_id"] != axis:
            prod *= red_width(d["block_id"])
    for g in grid_ids:
        if g != axis and in_acc(g):
            prod *= red_width(g)
    return max(1, ctx["itemsize"] * nl * prod)


def fp_unified_tagged(ctx, axis, nl):
    """ONE footprint formula: itemsize × nl × Σ_resident_tensors ∏_(dim≠axis) width(dim). The two
    regimes ARE the resident-tensor set (CARRIED: each loop-carried buffer is its own tensor, they
    ADD; STREAMED: ONE combined working tile). Dims are tagged with their size-source so the SAME
    block_id can be both a feature (extent) and a reduction (seated) — no dedup, no per-kind loops,
    no GRID_TILE skip. The only skip is `dim != axis` (the footprint EXCLUDES axis), once."""
    seated, ext_of, in_acc = ctx["seated"], ctx["ext_of"], ctx["in_acc"]
    reduction_ids, grid_ids = ctx["reduction_ids"], ctx["grid_ids"]
    feat_ext = ctx["feature_extent"]
    RED, GRID, FEAT = "red", "grid", "feat"

    def width(bid, src):
        if src is FEAT:
            return feat_ext[bid]
        if src is GRID:
            return max(1, seated.get(bid, 1))
        return max(1, seated.get(bid, ext_of(bid)))  # RED (seated rdim, else its extent)

    carried = [a for a in ctx["accs"] if ctx["is_carried"](a) and axis in a["dims"]]
    if not feat_ext and carried:
        # CARRIED: each loop-carried >=2-D reduction buffer is a SEPARATE resident tensor (ADD).
        resident_tensors = [
            [(d, GRID if d in grid_ids else RED) for d in a["dims"] if d is not None]
            for a in carried
        ]
    else:
        # STREAMED: ONE combined working tile — features + seated reductions + resident grid rows.
        resident_tensors = [
            [(f, FEAT) for f in feat_ext]
            + [(d["block_id"], RED) for d in ctx["sized"]]
            + [(g, GRID) for g in grid_ids if in_acc(g)]
        ]
    footprint = 0
    for tensor in resident_tensors:
        prod = 1
        for bid, src in tensor:
            if bid == axis:
                continue
            prod *= width(bid, src)
        footprint += prod
    return max(1, ctx["itemsize"] * nl * footprint)


def fp_additive_v2(ctx, axis, nl):
    """The CLEAN unified form to implement: base = feature × ∏(other seated reductions); then
    × Σ_carried_buffers (pure-carried regime) OR × ∏(in-acc grid rows) (streamed/grad-collapse)."""
    seated, ext_of, in_acc = ctx["seated"], ctx["ext_of"], ctx["in_acc"]
    feature = ctx["feature_footprint"]
    base = feature
    for d2 in ctx["sized"]:
        if d2["block_id"] == axis or d2["category"] == "grid_tile":
            continue
        base *= max(1, seated.get(d2["block_id"], ext_of(d2["block_id"])))
    if feature == 1 and ctx["has_carried"]:
        carried = [a for a in ctx["accs"] if ctx["is_carried"](a) and axis in a["dims"]]
        if carried:
            total = 0
            for a in carried:
                buf = 1
                for d in a["dims"]:
                    if d is None or d == axis:
                        continue
                    if d in ctx["reduction_ids"]:
                        buf *= max(1, seated.get(d, ext_of(d)))
                    elif d in ctx["grid_ids"]:
                        buf *= max(1, seated.get(d, 1))
                    else:
                        buf *= max(1, ext_of(d))
                total += buf
            return max(1, ctx["itemsize"] * nl * base * total)
    for g in ctx["grid_ids"]:
        if g == axis:
            continue
        if in_acc(g):
            base *= max(1, seated.get(g, 1))
    return max(1, ctx["itemsize"] * nl * base)


def diff_variant(facts, fn, label):
    nd = 0
    examples = []
    for r in facts:
        cur = M._alloc(r, candidate=False)
        if cur is None:
            continue
        v = run_variant(r, fn)
        vc = M.reconstruct_vector(r, cur)
        vv = M.reconstruct_vector(r, v)
        if vc != vv:
            nd += 1
            if len(examples) < 12:
                examples.append(
                    f"  {r['kernel']}/{tuple(r['shape'])}/{r['dtype']}: {vc} -> {vv}"
                )
    print(f"[{label}] {nd} diffs vs current")
    for e in examples:
        print(e)
    return nd


if __name__ == "__main__":
    facts = json.load(open("/tmp/corpus_facts.json"))
    diff_variant(facts, fp_current, "fp_current (sanity)")
    diff_variant(facts, fp_additive, "fp_additive (two-regime)")
    diff_variant(facts, fp_additive_v2, "fp_additive_v2 (clean unified)")
    diff_variant(facts, fp_inline_feature, "fp_inline_feature (IMPLEMENTED)")
    diff_variant(
        facts, lambda c, a, n: fp_resident_set(c, a, n, dedup=False),
        "fp_resident_set (unified, working=list)",
    )
    diff_variant(
        facts, lambda c, a, n: fp_resident_set(c, a, n, dedup=True),
        "fp_resident_set (unified, working=set-dedup)",
    )
    diff_variant(facts, fp_unified_tagged, "fp_unified_tagged (ONE formula)")
