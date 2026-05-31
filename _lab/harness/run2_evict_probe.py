"""Goal-2 GENERAL eviction probe. Tests whether the welford eviction win
generalizes via a principled RULE: per-load eviction = 'last' if that tensor is
RE-READ later in the kernel (keep L2-resident), else 'first' (last use -> stream).

For (kernel,M,N): map each load_eviction slot -> base tensor (from generated
Triton), compute the rule, and A/B {default, all_last, all_first, rule} at MATCHED
block_sizes/warps (live seed), G = tc_default/seed_lat (median-of-7).

Usage: ... python run2_evict_probe.py KERNEL M N [KERNEL M N ...]
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
median_do_bench = mg.median_do_bench
check_correct = mg.check_correct


def slot_tensors(bound, seed, length):
    """Map each eviction slot (in order) -> base tensor name, by generating Triton
    with a unique eviction marker and parsing tl.load lines that carry it."""
    cfg = helion.Config(**{**seed, "load_eviction_policies": ["last"] * length})
    code = bound.to_triton_code(cfg)
    bases = []
    for ln in code.splitlines():
        if "tl.load(" in ln and "eviction_policy='evict_last'" in ln:
            m = re.search(r"tl\.load\(([A-Za-z_][A-Za-z0-9_]*)\s*\+", ln)
            bases.append(m.group(1) if m else "?")
    return bases  # in textual (== slot) order; len should == length


def rule_policies(bases):
    """re-read rule: 'last' if base appears again at a LATER slot, else 'first'."""
    out = []
    for i, b in enumerate(bases):
        reread = b in bases[i + 1:]
        out.append("last" if reread else "first")
    return out


def measure_variant(fn, args, ref, seed, evict, extract):
    cfg = {**seed} if evict is None else {**seed, "load_eviction_policies": evict}
    k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args); b.ensure_config_exists(args)
    out = extract(b(*args))
    ok, err = check_correct(out, ref)
    if not ok:
        return None, err
    return median_do_bench(lambda: b(*args)), err


def probe(kernel, m, n):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, extract = builder(m, n)
    seed, bound = get_seed(fn, args)
    length = bound.env.config_spec.load_eviction_policies.length
    if length == 0:
        return {"kernel": kernel, "shape": [m, n], "note": "no eviction slots"}
    bases = slot_tensors(bound, seed, length)
    rule = rule_policies(bases)
    # tc baseline
    torch._dynamo.reset(); tc = torch.compile(tc_ref); _ = tc(args)
    tc_lat = median_do_bench(lambda: tc(args))
    variants = {
        "default": None,
        "all_last": ["last"] * length,
        "all_first": ["first"] * length,
        "rule": rule,
    }
    res = {}
    for name, ev in variants.items():
        lat, err = measure_variant(fn, args, ref, seed, ev, extract)
        res[name] = {"G": (tc_lat / lat) if lat else None, "evict": ev}
    return {"kernel": kernel, "shape": [m, n], "evict_len": length,
            "slot_tensors": bases, "rule": rule, "tc_us": tc_lat * 1e3,
            "seed_blocks": seed.get("block_sizes"), "results": res}


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    a = sys.argv[1:]
    triples = [(a[i], int(a[i + 1]), int(a[i + 2])) for i in range(0, len(a), 3)]
    for kernel, m, n in triples:
        try:
            r = probe(kernel, m, n)
        except Exception as e:  # noqa: BLE001
            r = {"kernel": kernel, "shape": [m, n], "err": f"{type(e).__name__}: {e}"[:160]}
        print(json.dumps({**r, "gpu": gpu}), flush=True)


if __name__ == "__main__":
    main()
