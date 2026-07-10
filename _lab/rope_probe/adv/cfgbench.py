"""Bench explicit Helion configs for one adversarial kernel+shape, cold-L2 med-of-9, in ONE process.
Reuses oracle_vs_tc.tc_ref for tc-parity where available (else ratios are vs default only).
Accuracy-gated vs the helion DEFAULT output.

argv: kernel_name shard_json M [N ...]  --cfgs "BM,BN[,W]" "BM,BN[,W]" ...
   (a cfg with no W defaults to num_warps=4; block dims are given outer->inner, matching hl.tile order)
Emits RESULT json: per-cfg {block_sizes,num_warps,ms,ok,G_tc,vs_default} + seed + default + tc.
"""
from __future__ import annotations
import sys, json, importlib.util, os
_WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"; sys.path.insert(0, _WT)
import torch, helion
import helion.language as hl
assert helion.__file__.startswith(_WT), helion.__file__
from helion._compiler.autotuner_heuristics.triton import TritonPointwiseSeedHeuristic as H
from helion.autotuner.benchmarking import do_bench
sys.path.insert(0, os.path.dirname(__file__))
import oracle_vs_tc as ovt  # for tc_ref

def load(name, shard):
    e = [x for x in json.load(open(shard)) if x["name"] == name][0]
    os.makedirs("/tmp/cfgb", exist_ok=True); p = f"/tmp/cfgb/{name}.py"
    open(p, "w").write("from __future__ import annotations\nimport torch,helion\nimport helion.language as hl\n" + e["code"])
    sp = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return getattr(m, name), m.make_inputs

def bench(fn, args, n=9):
    return sorted(do_bench(lambda: fn(*args), return_mode="median") for _ in range(n))[n // 2]

def main():
    name, shard = sys.argv[1], sys.argv[2]
    i = 3; dims = []
    while i < len(sys.argv) and sys.argv[i] != "--cfgs":
        dims.append(int(sys.argv[i])); i += 1
    cfgs = []
    if i < len(sys.argv) and sys.argv[i] == "--cfgs":
        for tok in sys.argv[i + 1:]:
            parts = [int(x) for x in tok.split(",")]
            if len(parts) == len(dims):  # no warp given
                cfgs.append((parts, 4))
            else:
                cfgs.append((parts[:-1], parts[-1]))
    shape = tuple(dims)
    fn, mk = load(name, shard); args = tuple(mk(shape))
    bk = fn.bind(args)
    seed = H.get_seed_config(bk.env, bk.host_function.device_ir)
    dflt = bk.config_spec.default_config()
    fact = bk.config_spec.pointwise_facts[0] if bk.config_spec.pointwise_facts else None
    dref = bk.compile_config(dflt)(*args)
    def agree(o):
        a = o if isinstance(o, torch.Tensor) else o[0]
        b = dref if isinstance(dref, torch.Tensor) else dref[0]
        return bool(torch.isfinite(a.float()).all() and torch.allclose(a.float(), b.float(), atol=2e-2, rtol=2e-2))
    d_ms = bench(bk.compile_config(dflt), args)
    s_run = bk.compile_config(seed); s_ms = bench(s_run, args); s_ok = agree(s_run(*args))
    tc_ms = None
    try:
        torch._dynamo.reset()
        cf = torch.compile(ovt.tc_ref(name), mode="max-autotune-no-cudagraphs")
        cf(*args); torch.cuda.synchronize(); tc_ms = bench(cf, args)
    except Exception as e:
        tc_ms = None
    res = []
    for b, w in cfgs:
        try:
            r = bk.compile_config(helion.Config(block_sizes=b, num_warps=w))
            o = r(*args); ok = agree(o); ms = bench(r, args)
            res.append({"cfg": b, "w": w, "ms": ms, "ok": ok,
                        "G_tc": (tc_ms / ms) if tc_ms else None, "vs_default": d_ms / ms})
        except Exception as e:
            res.append({"cfg": b, "w": w, "status": f"{type(e).__name__}: {str(e)[:100]}"})
    print("RESULT " + json.dumps({
        "kernel": name, "shape": list(shape),
        "seed_cfg": seed.config["block_sizes"], "seed_ms": s_ms, "seed_ok": s_ok,
        "default_ms": d_ms, "tc_ms": tc_ms,
        "G_seed": (tc_ms / s_ms) if tc_ms else None, "seed_vs_default": d_ms / s_ms,
        "fact_bpe": getattr(fact, "bandwidth_bytes_per_elem", None), "fact_reg": getattr(fact, "register_bytes_per_elem", None),
        "cfgs": res}))

if __name__ == "__main__":
    main()
