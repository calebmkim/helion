"""SPIKE the FAITHFUL Σ-over-resident-tiles footprint offline.

footprint(axis) = itemsize × [ Σ over resident TILES of ∏(that tile's dim widths, EXCLUDING axis) ]

Resident tiles = two kinds, uniformly summed (∏ within a tile, Σ across tiles):
  1. READ/COMPUTE tiles: the reduction body holds ``num_live`` simultaneously-live copies of the
     reduction tile, shape [<all sized reduction dims in the group>]. So ``num_live`` copies, each
     ∏(seated width of every sized reduction dim except axis).
  2. ACCUMULATOR tiles: each ``accumulator_fact`` is a resident buffer; contributes ∏(width of its
     dims except axis). Dims classified by membership (reduction->seated, grid->seated, else extent).

Grid rows: a resident grid axis multiplies the read tile (a program handling k rows holds k× the
tile). Modeled by including resident (in-accumulator) grid dims in the read-tile ∏. A reduced-away
grid axis is handled in the grid-M loop (collapse), not here.

This treats accumulators as TILES THAT ADD (not multiplied into the read tile) — faithful to bytes.
Reports the full corpus config delta vs current.
"""
from __future__ import annotations

import json
import math
import sys

sys.path.insert(0, "/home/dev/local/helion-redesign/_lab/redesign")
import model_alloc as M  # noqa: E402

np2, pp2 = M.np2, M.pp2
ROW = 245760
LIVE = 3 * 245760
LOOPED = 16384
MIN_WAVES = 8
WMR = 8
CARR = 122880
FULL = {"full_slice", "full_grid"}
SIZED = FULL | {"user_tile"}


def run(rec, *, carried_budget=CARR, grad_param_budget=393216):
    if "kf" in rec or rec.get("error") or rec.get("pd_block_id") is None:
        return None
    reds = rec["reductions"]
    rids = set(rec["reduction_ids"])
    grid = set(rec["grid_ids"])
    nrl = set(rec["nrl_ids"])
    accs = rec["accumulators"]
    itemsize = max(1, rec["pd_itemsize"])
    pd = rec["pd_block_id"]
    pdcat = rec["pd_category"]
    bsv = set(rec["bs_valid"])
    grid_rows = rec["grid_rows"]
    num_sm = rec["num_sm"]
    occ_floor = num_sm * MIN_WAVES
    ec = rec["element_cap"]
    feat = {int(k): max(1, v or 1) for k, v in rec["feature_axes"].items()}
    seated = {int(k): v for k, v in rec["grid_seats"].items()}
    sizes, redv = {}, {}

    def ext(b):
        e = rec["ext"].get(str(b))
        return np2(e) if e else 1

    def in_acc(b):
        return any(b in a["dims"] for a in accs)

    def is_c(a):
        return len(a["dims"]) >= 2 and any(d in rids for d in a["dims"])

    has_carried = any(is_c(a) for a in accs)
    # THREE budgets keyed on two faithful structural flags (no recognizer):
    #  - carried-2D (a >=2-D loop-carried reduction tile, NO materialized feature): TIGHT — kl_div/
    #    jsd hold [grid,R] resident the whole loop (transient, want a small chunk).
    #  - grad-param (a materialized feature accumulator present): LOOSE — a big [N] accumulator is
    #    amortized across many rows, so the program genuinely runs at a higher byte ceiling; the
    #    read tile shares SRAM with it and wants inner>=2.
    #  - streamed (neither): ROW.
    if feat:
        budget = grad_param_budget
    elif has_carried:
        budget = carried_budget
    else:
        budget = ROW

    def width(b):
        # a MATERIALIZED axis is resident at its FULL extent (hl.specialize'd — never chunked),
        # even when it is ALSO flagged a reduction. So feature width wins over any chunk seating.
        if b in feat:
            return feat[b]
        if b in grid:
            return max(1, seated.get(b, 1))
        return max(1, seated.get(b, ext(b)))

    cs_ = []

    def read_tile_prod(axis):
        # The read/compute tile spans AXIS and the OTHER resident dims: the UNION (deduped) of the
        # group's sized reductions, the resident (in-accumulator) grid rows, and the materialized
        # feature axes. ∏ of their widths EXCEPT axis (axis is the R factor). A materialized axis
        # contributes its FULL extent (width() handles that), even if it is also a sized reduction —
        # counted ONCE (set union).
        dims = {d["block_id"] for d in cs_}
        dims |= {g for g in grid if in_acc(g)}
        dims |= set(feat)
        p = 1
        for d in dims:
            if d != axis:
                p *= width(d)
        return p

    def footprint_terms(axis, nl):
        """Return (SCALE, FLAT): total resident ELEMENTS = SCALE × R_axis + FLAT.
        A tile that CONTAINS the axis scales with R (contributes ∏(its other dims) to SCALE); a
        tile that does NOT contain the axis is FLAT (contributes ∏(all its dims)). The read/compute
        tile always contains the axis (both reductions and resident grid rows are its dims), in
        num_live copies. Each accumulator_fact is its own tile; a feature not in any accumulator is
        its own [feat] tile. This is the faithful byte model: R-scaling terms and constant terms are
        SEPARATE (add), not multiplied together."""
        # read/compute tile: num_live copies, ∏ of its OTHER dims (the axis contributes the R factor)
        scale = nl * read_tile_prod(axis)
        flat = 0
        acc_dims_seen = set()
        for a in accs:
            dims = [d for d in a["dims"] if d is not None]
            acc_dims_seen.update(dims)
            if axis in dims:
                p = 1
                for d in dims:
                    if d != axis:
                        p *= width(d)
                scale += p  # this accumulator scales with R
            else:
                p = 1
                for d in dims:
                    p *= width(d)
                flat += p  # constant in R
        for f, fe in feat.items():
            if f in acc_dims_seen:
                continue  # already counted via an accumulator tile
            if f == axis:
                continue  # the axis itself is the R factor, not a flat term
            flat += fe
        return scale, flat

    def gk(idxs):
        s = [reds[i] for i in idxs if reds[i]["category"] in SIZED]
        return (
            -sum(1 for d in s if d["category"] in FULL),
            -max([d["size_hint"] for d in s], default=0),
        )

    for idxs in sorted(rec["groups"], key=gk):
        sized = [reds[i] for i in idxs if reds[i]["category"] in SIZED]
        if not sized:
            continue
        cs_ = sized
        nl = max([d["body_live_tiles"] for d in sized] + [1])
        order = sorted(
            sized,
            key=lambda d: (
                0 if d["category"] in FULL else (1 if d["category"] == "user_tile" else 2),
                -d["size_hint"],
            ),
        )
        for d in order:
            raw = d["size_hint"]
            e = ext(d["block_id"])
            if d["category"] == "grid_tile":
                seated[d["block_id"]] = 1
                continue
            if d["category"] == "full_grid":
                seated[d["block_id"]] = e
                if d["block_id"] in bsv:
                    redv[d["block_id"]] = e
                continue
            # total resident BYTES(R) = itemsize × (scale × R + flat). Persistence holds R=extent
            # iff that fits; else chunk to the largest pow2 R that fits: R ≤ (budget/isz − flat)/scale.
            s1, f1 = footprint_terms(d["block_id"], 1)
            sl, fl_ = footprint_terms(d["block_id"], nl)
            held = (
                d["row_reread"]
                and d["carried_2d_count"] == 0
                and (ec is None or raw <= ec)
                and itemsize * (s1 * raw + f1) <= budget
                and itemsize * (sl * raw + fl_) <= LIVE
            )
            if held:
                r = e
            else:
                avail = budget // itemsize - fl_
                r = max(1, min(LOOPED, pp2(max(1, avail // max(1, sl))), e))
            seated[d["block_id"]] = r
            if d["block_id"] in bsv:
                redv[d["block_id"]] = r
        for mbid in sorted(grid):
            if mbid not in bsv:
                continue
            e = ext(mbid)
            fl = rec["floor"].get(str(mbid), 1)
            if len(accs) > 0 and not in_acc(mbid):
                coll = np2(max(1, grid_rows // num_sm)) if grid_rows > 0 else 1
                blk = max(fl, min(coll, e))
            else:
                wl = 1 if nrl else nl
                s_w, f_w = footprint_terms(mbid, wl)
                avail = budget // itemsize - f_w
                bw = pp2(max(1, avail // max(1, s_w)))
                ow = pp2(max(1, grid_rows // occ_floor)) if grid_rows > 0 else 1
                rc = e if pdcat == "full_grid" else WMR
                blk = max(fl, min(bw, ow, rc, e))
            seated[mbid] = blk
            sizes[mbid] = blk
    lb = pp2(max(1, ROW // max(1, itemsize * (math.prod(feat.values()) if feat else 1))))
    for bid in rec["bs_valid"]:
        if bid in redv or bid in grid or bid in rids:
            continue
        if bid in nrl or bid not in seated:
            sizes[bid] = max(1, min(ext(bid), lb))
    return {"sizes": sizes, "red_values": redv}


def main():
    facts = json.load(open("/tmp/corpus_facts.json"))
    nd = 0
    movers = []
    for r in facts:
        cur = M._alloc(r, candidate=False)
        if cur is None:
            continue
        v = run(r)
        vc = M.reconstruct_vector(r, cur)
        vv = M.reconstruct_vector(r, v)
        if vc != vv:
            nd += 1
            movers.append((r["kernel"], tuple(r["shape"]), vc, vv))
    print(f"FAITHFUL Σ-over-tiles: {nd} cells differ from current committed")
    byk = {}
    for k, sh, vc, vv in movers:
        byk.setdefault(k, []).append((sh, vc, vv))
    for k in sorted(byk):
        print(f"\n  {k}: {len(byk[k])} cells")
        for sh, vc, vv in byk[k][:6]:
            print(f"     {sh} {vc} -> {vv}")
        if len(byk[k]) > 6:
            print(f"     ... (+{len(byk[k]) - 6})")


if __name__ == "__main__":
    main()
