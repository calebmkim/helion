"""Goal-2 welford codegen-knob probe: from the G1 block_sizes seed, sweep
indexing (tensor_descriptor) + load_eviction_policies variants at MATCHED
block_sizes/warps, measure G vs tc_default. Find what reaches the oracle ~0.96
and WHICH slots/policies drive it (workload property for the seed rule).
Usage: ... python run2_wf_knobs.py M N
"""
from __future__ import annotations
import sys, os, json
from statistics import median
import torch
import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2"
assert helion.__file__.startswith(WT + "/"), helion.__file__
from examples.welford import welford, eager_layer_norm
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from triton.testing import do_bench

EPS = 1e-5
N_RUNS = 7


def args_for(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def med(fn):
    torch.cuda.synchronize()
    return median([float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)])


def run(a, cfg, ref):
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    bk = k.bind(a); bk.ensure_config_exists(a)
    out = bk(*a); out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-4))
    if not ok:
        return None, dict(bk._config)
    return med(lambda: bk(*a)), dict(bk._config)


def main():
    m, n = int(sys.argv[1]), int(sys.argv[2])
    a = args_for(m, n); ref = eager_layer_norm(*a)
    bound = welford.bind(a)
    spec = bound.env.config_spec
    nidx = spec.indexing.length
    nevict = spec.load_eviction_policies.length
    store_idx = list(spec.store_indices)
    seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
    print(f"# welford ({m},{n}) indexing.length={nidx} evict.length={nevict} "
          f"store_indices={store_idx} valid_idx={spec.valid_indexing_types()}", flush=True)
    print(f"# G1 seed = {seed}", flush=True)
    torch._dynamo.reset(); tc = torch.compile(eager_layer_norm); tc(*a)
    tc_lat = med(lambda: tc(*a))

    POINTER = "pointer"; TD = "tensor_descriptor"
    load_slots = [i for i in range(nidx) if i not in store_idx]
    variants = {}
    variants["default"] = {}
    variants["idx_all_TD"] = {"indexing": [TD] * nidx}
    variants["idx_loads_TD"] = {"indexing": [TD if i in load_slots else POINTER for i in range(nidx)]}
    variants["evict_all_last"] = {"load_eviction_policies": ["last"] * nevict}
    variants["evict_all_first"] = {"load_eviction_policies": ["first"] * nevict}
    variants["evict_last_first"] = {"load_eviction_policies": (["last", "first"] * nevict)[:nevict]}
    variants["loadsTD+evict_last"] = {"indexing": [TD if i in load_slots else POINTER for i in range(nidx)],
                                       "load_eviction_policies": ["last"] * nevict}
    variants["loadsTD+evict_lastfirst"] = {"indexing": [TD if i in load_slots else POINTER for i in range(nidx)],
                                            "load_eviction_policies": (["last", "first"] * nevict)[:nevict]}
    variants["allTD+evict_lastfirst"] = {"indexing": [TD] * nidx,
                                          "load_eviction_policies": (["last", "first"] * nevict)[:nevict]}
    # per-slot TD probes (which load slot matters)
    for s in load_slots:
        variants[f"idx_TD_slot{s}"] = {"indexing": [TD if i == s else POINTER for i in range(nidx)]}

    results = []
    for name, extra in variants.items():
        cfg = {**seed, **extra}
        try:
            lat, norm = run(a, cfg, ref)
        except Exception as e:  # noqa: BLE001
            results.append({"name": name, "err": f"{type(e).__name__}: {e}"[:80]}); continue
        if lat is None:
            results.append({"name": name, "incorrect": True}); continue
        results.append({"name": name, "G": tc_lat / lat, "lat_us": lat * 1e3,
                        "idx": norm.get("indexing"), "evict": norm.get("load_eviction_policies")})
    results.sort(key=lambda r: -(r.get("G") or 0))
    print(json.dumps({"shape": [m, n], "tc_us": tc_lat * 1e3, "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
