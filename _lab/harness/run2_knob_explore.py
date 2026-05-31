"""Goal-2 codegen-knob exploration (matched-lever A/B).

For each (kernel,M,N): hold block_sizes/num_warps/reduction_loops = the LIVE
heuristic seed and vary ONE knob at a time:
  - num_stages in {1(default),2,3,4}
  - indexing=tensor_descriptor on the WIDE x load slot(s) (+ stores left pointer)
  - indexing=tensor_descriptor on ALL slots
  - range_num_stages on every range loop in {1,2,3,4}
  - TD(best-eligible) + best-num_stages combo
G = tc_default_lat / variant_lat, median-of-7. Correctness gated allclose
rtol=1e-3/atol=1e-4 (NOT loosened). Inert variants (codegen identical to seed)
are flagged. Raw numbers persisted to /tmp/knob_explore.json.

Usage: ... python run2_knob_explore.py KERNEL M N [KERNEL M N ...]
"""
from __future__ import annotations
import sys, os, json, re
import torch
import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), helion.__file__
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run2_measure_g", WT + "_lab/harness/run2_measure_g.py")
mg = importlib.util.module_from_spec(spec); spec.loader.exec_module(mg)

KERNELS = mg.KERNELS
get_seed = mg.get_seed
check_correct = mg.check_correct

OUT_PATH = "/tmp/knob_explore.json"

# Robust median-of-N timer. Small reduction kernels (~6us) are launch/measure
# noise dominated, so we use more do_bench samples than the canonical harness.
from triton.testing import do_bench  # noqa: E402
N_RUNS = 15


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median", rep=200))
               for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


def x_load_slots(bound, seed, length):
    """Find indexing slots that map to the reduction-input x load.

    Strategy: toggle each indexing slot to tensor_descriptor in isolation and
    see which one (a) changes codegen and (b) attaches a tensor descriptor to a
    `tl.load(x ...)` line. Returns (x_slots, td_engages_for_slot dict, store_slots).
    """
    store_slots = list(getattr(bound.env.config_spec, "store_indices", []))
    base = bound.to_triton_code(helion.Config(**seed))
    x_slots = []
    engages = {}
    for i in range(length):
        idx = ["pointer"] * length
        idx[i] = "tensor_descriptor"
        try:
            code = bound.to_triton_code(helion.Config(**{**seed, "indexing": idx}))
        except Exception:
            engages[i] = False
            continue
        changed = code != base
        ntd = code.count("make_tensor_descriptor")
        engages[i] = changed and ntd > 0
        if not engages[i]:
            continue
        # does a tensor-descriptor-backed load read x (the wide reduction input)?
        # heuristic: the descriptor block built from x base appears and a desc.load
        # exists; mark as x slot if this slot is NOT a store slot.
        if i not in store_slots:
            x_slots.append(i)
    return x_slots, engages, store_slots


def measure_variant(fn, args, ref, cfg_overrides, seed, extract):
    cfg = {**seed, **cfg_overrides}
    try:
        k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        out = extract(b(*args))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, "OOM"
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if "out of memory" in msg.lower() or "OutOfMemory" in msg:
            torch.cuda.empty_cache()
            return None, None, "OOM"
        return None, None, msg[:120]
    ok, err = check_correct(out, ref)
    if not ok:
        return None, err, "INCORRECT"
    lat = median_do_bench(lambda: b(*args))
    return lat, err, "ok"


def codegen_inert(bound, seed, cfg_overrides):
    """True if the override produces codegen identical to the seed (knob inert)."""
    try:
        base = bound.to_triton_code(helion.Config(**seed))
        var = bound.to_triton_code(helion.Config(**{**seed, **cfg_overrides}))
    except Exception:
        return None
    return base == var


def probe(kernel, m, n):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, extract = builder(m, n)
    seed, bound = get_seed(fn, args)
    cs = bound.env.config_spec
    L = cs.indexing.length
    n_range = len(cs.range_num_stages)

    x_slots, td_engages, store_slots = x_load_slots(bound, seed, L)

    # tc baseline
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    _ = tc(args)
    tc_lat = median_do_bench(lambda: tc(args))

    # build variant configs
    variants = {}
    variants["default"] = {}
    for ns in (2, 3, 4):
        variants[f"num_stages={ns}"] = {"num_stages": ns}
    # TD on x load slots only (stores + small operands left pointer)
    if x_slots:
        idx = ["pointer"] * L
        for s in x_slots:
            idx[s] = "tensor_descriptor"
        variants["TD_x_load"] = {"indexing": idx}
    # TD on all slots
    if L > 0:
        variants["TD_all"] = {"indexing": ["tensor_descriptor"] * L}
    # range_num_stages on all range loops
    if n_range > 0:
        for rns in (1, 2, 3):
            variants[f"range_num_stages={rns}"] = {
                "range_num_stages": [rns] * n_range}

    # add a 2nd default measurement as a noise control
    variants["default_recheck"] = {}

    res = {}
    best_ns = None
    best_ns_g = None
    seed_lat = None
    for name, ov in variants.items():
        inert = codegen_inert(bound, seed, ov) if ov else False
        lat, err, status = measure_variant(fn, args, ref, ov, seed, extract)
        g = (tc_lat / lat) if lat else None
        if name == "default":
            seed_lat = lat
        rel = (seed_lat / lat) if (lat and seed_lat) else None
        res[name] = {
            "G": round(g, 4) if g is not None else None,
            "rel_to_seed": round(rel, 4) if rel is not None else None,
            "status": status,
            "inert": inert,
            "lat_us": round(lat * 1e3, 2) if lat else None,
            "maxerr": round(err, 6) if err is not None else None,
        }
        if name.startswith("num_stages=") and g is not None and not inert:
            if best_ns_g is None or g > best_ns_g:
                best_ns_g = g
                best_ns = int(name.split("=")[1])

    # combo: TD_x_load + best num_stages
    if x_slots and best_ns is not None:
        idx = ["pointer"] * L
        for s in x_slots:
            idx[s] = "tensor_descriptor"
        ov = {"indexing": idx, "num_stages": best_ns}
        lat, err, status = measure_variant(fn, args, ref, ov, seed, extract)
        g = (tc_lat / lat) if lat else None
        rel = (seed_lat / lat) if (lat and seed_lat) else None
        res[f"TD_x+num_stages={best_ns}"] = {
            "G": round(g, 4) if g is not None else None,
            "rel_to_seed": round(rel, 4) if rel is not None else None,
            "status": status, "inert": False,
            "lat_us": round(lat * 1e3, 2) if lat else None,
            "maxerr": round(err, 6) if err is not None else None,
        }

    return {
        "kernel": kernel, "shape": [m, n],
        "seed_block_sizes": seed.get("block_sizes"),
        "seed_num_warps": seed.get("num_warps"),
        "seed_reduction_loops": seed.get("reduction_loops"),
        "indexing_len": L, "store_slots": store_slots,
        "x_load_slots": x_slots, "td_engages_per_slot": td_engages,
        "n_range_loops": n_range,
        "tc_us": round(tc_lat * 1e3, 2),
        "results": res,
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    a = sys.argv[1:]
    triples = [(a[i], int(a[i + 1]), int(a[i + 2])) for i in range(0, len(a), 3)]

    # load existing json (append/update)
    try:
        with open(OUT_PATH) as f:
            store = json.load(f)
    except Exception:
        store = {"gpu": gpu, "probes": []}
    store.setdefault("probes", [])

    for kernel, m, n in triples:
        try:
            r = probe(kernel, m, n)
        except Exception as e:  # noqa: BLE001
            r = {"kernel": kernel, "shape": [m, n],
                 "err": f"{type(e).__name__}: {e}"[:200]}
        # replace existing entry for same kernel+shape
        store["probes"] = [p for p in store["probes"]
                           if not (p.get("kernel") == kernel
                                   and p.get("shape") == [m, n])]
        store["probes"].append(r)
        with open(OUT_PATH, "w") as f:
            json.dump(store, f, indent=2, default=str)
        print(json.dumps(r, default=str), flush=True)


if __name__ == "__main__":
    main()
