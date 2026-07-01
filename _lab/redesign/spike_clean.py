"""SPIKE the clean redesign OFFLINE before touching helion.

ONE uniform footprint everywhere (no Σ_buffers, no carried *formula*):
    footprint(axis) = itemsize × num_live × ∏(merged working-tile dims, except axis)
where the working tile's dims are ONE merged list: materialized features (extent) +
every seated reduction in the group + the in-accumulator grid rows. `num_live`
(`body_live_tiles`) is ALREADY axis-resolved (peak tiles whose shape spans the rdim —
verified in device_ir `_graph_peak_live_by_axis`), so scalar carries are not counted and
there is NO separate buffer-multiplicity term.

The carried/streamed distinction becomes a BUDGET CONSTANT, not a formula: a kernel with a
>=2-D loop-carried reduction tile AND no materialized feature tile (pure kl_div/jsd) uses a
tighter CARRIED budget; everything else uses ROW_PERSIST. Tuned so kl_div->4096, jsd->2048.

Reports the full corpus config delta vs current (validated model_alloc) for a swept budget.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/home/dev/local/helion-redesign/_lab/redesign")
import model_alloc as M  # noqa: E402

np2, pp2 = M.np2, M.pp2
ROW = 245760
LIVE = 3 * 245760
LOOPED = 16384
MIN_WAVES = 8
WIDEN_MAX_ROWS = 8
FULL = {"full_slice", "full_grid"}
SIZED = {"full_slice", "full_grid", "user_tile"}


def alloc(rec, *, carried_budget, drop_widen_live_for_rta):
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

    def is_carried(a):
        return len(a["dims"]) >= 2 and any(d in rids for d in a["dims"])

    has_carried = any(is_carried(a) for a in accs)
    # BUDGET REGIME (not a formula): pure-carried (>=2-D carried reduction tile, no feature) ->
    # tighter carried budget; everything else (streamed + grad-collapse) -> ROW.
    budget = carried_budget if (has_carried and not feat) else ROW

    def wdt(b):  # working-tile dim width by membership (grid row block / seated rdim / extent)
        if b in grid:
            return max(1, seated.get(b, 1))
        return max(1, seated.get(b, ext(b)))

    def working_tile_dims():
        # ONE merged list of (block_id, width): features + seated reductions + in-acc grid rows.
        return (
            [(f, fe) for f, fe in feat.items()]
            + [(d["block_id"], wdt(d["block_id"])) for d in cur_sized]
            + [(g, wdt(g)) for g in grid if in_acc(g)]
        )

    def footprint(axis, nl):
        prod = 1
        for b, w in working_tile_dims():
            if b != axis:
                prod *= w
        return max(1, itemsize * nl * prod)

    def gk(idxs):
        s = [reds[i] for i in idxs if reds[i]["category"] in SIZED]
        return (
            -sum(1 for d in s if d["category"] in FULL),
            -max([d["size_hint"] for d in s], default=0),
        )

    cur_sized = []
    for idxs in sorted(rec["groups"], key=gk):
        sized = [reds[i] for i in idxs if reds[i]["category"] in SIZED]
        if not sized:
            continue
        cur_sized = sized
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
            cs = footprint(d["block_id"], 1)
            cl = footprint(d["block_id"], nl)
            held = (
                d["row_reread"]
                and d["carried_2d_count"] == 0
                and (ec is None or raw <= ec)
                and cs * raw <= budget
                and cl * raw <= LIVE
            )
            r = e if held else max(1, min(LOOPED, pp2(max(1, budget // cl)), e))
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
                wl = 1 if (drop_widen_live_for_rta and nrl) else nl
                bw = pp2(max(1, budget // footprint(mbid, wl)))
                ow = pp2(max(1, grid_rows // occ_floor)) if grid_rows > 0 else 1
                rc = e if pdcat == "full_grid" else WIDEN_MAX_ROWS
                blk = max(fl, min(bw, ow, rc, e))
            seated[mbid] = blk
            sizes[mbid] = blk
    loop_budget = pp2(max(1, ROW // max(1, itemsize * _prod(feat.values()))))
    for bid in rec["bs_valid"]:
        if bid in redv or bid in grid or bid in rids:
            continue
        if bid in nrl or bid not in seated:
            sizes[bid] = max(1, min(ext(bid), loop_budget))
    return {"sizes": sizes, "red_values": redv}


def _prod(vals):
    p = 1
    for v in vals:
        p *= v
    return p


def main():
    facts = json.load(open("/tmp/corpus_facts.json"))
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--carried", type=int, default=122880)
    ap.add_argument("--keep-widen-live", action="store_true",
                    help="do NOT drop widen_live for reduce-then-apply (test if budget subsumes F)")
    a = ap.parse_args()
    nd = 0
    by_kernel = {}
    for r in facts:
        cur = M._alloc(r, candidate=False)
        if cur is None:
            continue
        v = alloc(r, carried_budget=a.carried,
                  drop_widen_live_for_rta=not a.keep_widen_live)
        vc = M.reconstruct_vector(r, cur)
        vv = M.reconstruct_vector(r, v)
        if vc != vv:
            nd += 1
            by_kernel.setdefault(r["kernel"], []).append((tuple(r["shape"]), vc, vv))
    print(f"carried_budget={a.carried} keep_widen_live={a.keep_widen_live}: "
          f"{nd} cells differ from current\n")
    for k in sorted(by_kernel):
        lst = by_kernel[k]
        print(f"  {k}: {len(lst)} cells")
        for sh, vc, vv in lst[:4]:
            print(f"     {sh} cur={vc} -> new={vv}")
        if len(lst) > 4:
            print(f"     ... (+{len(lst) - 4} more)")


if __name__ == "__main__":
    main()
