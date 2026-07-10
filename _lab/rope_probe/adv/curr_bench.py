"""Full-curriculum DoD bench: iterate shapes_adversarial.CURRICULUM, bench seed vs tc vs default
(cold-L2 med-of-9, accuracy-gated vs default), report G_seed per (kernel, shape). Reuses cfgbench/ovt.

argv: [split1,split2,...]  (default: train,val ; NEVER pass 'test' unless the Gate-E freeze read)
Emits per-row JSON to stdout (prefix ROW) + a RESULT summary; also writes /tmp/curr_bench_<splits>.json.
"""
from __future__ import annotations
import sys, json, os, importlib.util
_WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"; sys.path.insert(0, _WT)
sys.path.insert(0, os.path.join(_WT, "_lab", "adversarial"))
import torch, helion
import helion.language as hl
assert helion.__file__.startswith(_WT), helion.__file__
from helion._compiler.autotuner_heuristics.triton import TritonPointwiseSeedHeuristic as H
from helion.autotuner.benchmarking import do_bench
sys.path.insert(0, os.path.dirname(__file__))
import oracle_vs_tc as ovt
import shapes_adversarial as SA
SH = "/home/calebkim/helion-new-heuristics/local/rope_probe/adv/shards/"

def load(shard, name):
    e = [x for x in json.load(open(SH + shard)) if x["name"] == name][0]
    os.makedirs("/tmp/cb", exist_ok=True); p = f"/tmp/cb/{name}.py"
    open(p, "w").write("from __future__ import annotations\nimport torch,helion\nimport helion.language as hl\n" + e["code"])
    sp = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return getattr(m, name), m.make_inputs

def bench(fn, args, n=9):
    return sorted(do_bench(lambda: fn(*args), return_mode="median") for _ in range(n))[n // 2]

def main():
    splits = sys.argv[1].split(",") if len(sys.argv) > 1 else ["train", "val"]
    rows = []
    for name, spec in SA.CURRICULUM.items():
        shard = spec["shard"]
        try:
            fn, mk = load(shard, name)
        except Exception as e:
            print("ROW " + json.dumps({"kernel": name, "load_error": str(e)[:120]})); continue
        try:
            tcref = ovt.tc_ref(name)
        except Exception:
            tcref = None
        for sp in splits:
            for shape in spec.get(sp, []):
                shape = tuple(shape)
                row = {"kernel": name, "cls": spec["cls"], "split": sp, "shape": list(shape)}
                try:
                    args = tuple(mk(shape))
                    bk = fn.bind(args)
                    if not bk.config_spec.pointwise_facts:
                        row["status"] = "not_pointwise"; rows.append(row); print("ROW " + json.dumps(row)); continue
                    seed = H.get_seed_config(bk.env, bk.host_function.device_ir)
                    dflt = bk.config_spec.default_config()
                    dref = bk.compile_config(dflt)(*args)
                    def agree(o):
                        a = o if isinstance(o, torch.Tensor) else o[0]
                        b = dref if isinstance(dref, torch.Tensor) else dref[0]
                        return bool(torch.isfinite(a.float()).all() and torch.allclose(a.float(), b.float(), atol=2e-2, rtol=2e-2))
                    srun = bk.compile_config(seed); s_ok = agree(srun(*args))
                    s_ms = bench(srun, args); d_ms = bench(bk.compile_config(dflt), args)
                    tc_ms = None
                    if tcref is not None:
                        torch._dynamo.reset()
                        cf = torch.compile(tcref, mode="max-autotune-no-cudagraphs")
                        cf(*args); torch.cuda.synchronize(); tc_ms = bench(cf, args)
                    row.update({"seed_cfg": seed.config.get("block_sizes"), "num_warps": seed.config.get("num_warps"),
                                "seed_ok": s_ok, "seed_ms": round(s_ms, 5), "default_ms": round(d_ms, 5),
                                "tc_ms": round(tc_ms, 5) if tc_ms else None,
                                "G_seed": round(tc_ms / s_ms, 3) if tc_ms else None,
                                "seed_vs_default": round(d_ms / s_ms, 3)})
                    del args, dref, srun; torch.cuda.empty_cache()
                except Exception as e:
                    row["status"] = f"{type(e).__name__}: {str(e)[:100]}"
                rows.append(row); print("ROW " + json.dumps(row))
    outp = f"/tmp/curr_bench_{'_'.join(splits)}.json"
    json.dump(rows, open(outp, "w"), indent=1)
    # summary: min G per class, below-floor count
    from collections import defaultdict
    byc = defaultdict(list)
    for r in rows:
        if r.get("G_seed") is not None and r.get("seed_ok"):
            byc[r["cls"]].append((r["G_seed"], r["kernel"], r["shape"]))
    print("RESULT " + json.dumps({"out": outp, "classes": {c: {"min_G": min(v)[0], "min_at": min(v)[1:], "n": len(v)} for c, v in byc.items()},
                                  "below_floor": [(r["kernel"], r["shape"], r["G_seed"]) for r in rows if r.get("G_seed") is not None and r["G_seed"] < 0.75 and r.get("seed_ok")]}))

if __name__ == "__main__":
    main()
