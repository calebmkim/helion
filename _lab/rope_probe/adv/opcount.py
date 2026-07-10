"""Bind-only (NO GPU launch) profiler: count device-graph compute ops per kernel, categorized, to
design the num_warps arithmetic-intensity ramp. Prints per-kernel: total call_function nodes, load/store
count, SFU/transcendental count, other-math count, + the pointwise fact fields.

argv: none (runs a fixed set). Reads shard kernels + Tier-A kernels.
"""
from __future__ import annotations
import sys, json, os, importlib.util
_WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"; sys.path.insert(0, _WT)
sys.path.insert(0, os.path.join(_WT, "_lab", "pointwise"))
import torch, helion
import helion.language as hl
assert helion.__file__.startswith(_WT), helion.__file__
SH = "/home/calebkim/helion-new-heuristics/local/rope_probe/adv/shards/"
DEV = "cuda"

# op-name buckets (by fx node target __name__ / str)
SFU = {"sin", "cos", "tanh", "exp", "log", "sqrt", "rsqrt", "sigmoid", "erf", "expm1", "log1p", "reciprocal", "pow"}


def load_shard(shard, name):
    e = [x for x in json.load(open(SH + shard)) if x["name"] == name][0]
    os.makedirs("/tmp/oc", exist_ok=True); p = f"/tmp/oc/{name}.py"
    open(p, "w").write("from __future__ import annotations\nimport torch,helion\nimport helion.language as hl\n" + e["code"])
    sp = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return getattr(m, name), m.make_inputs


def profile(fn, args, label):
    from helion._compiler.autotuner_heuristics.triton import TritonPointwiseSeedHeuristic as H
    bk = fn.bind(args)
    dir_ = bk.host_function.device_ir
    total = load = store = sfu = other = 0
    from helion.language import memory_ops
    for gi in dir_.graphs:
        for node in gi.graph.nodes:
            if node.op != "call_function":
                continue
            tgt = node.target
            nm = getattr(tgt, "__name__", str(tgt)).lower()
            if tgt is memory_ops.load:
                load += 1
            elif tgt is memory_ops.store:
                store += 1
            else:
                total += 1
                base = nm.split(".")[0]
                if any(s in nm for s in SFU):
                    sfu += 1
                else:
                    other += 1
    fires = bool(bk.config_spec.pointwise_facts)
    f = bk.config_spec.pointwise_facts[0] if fires else None
    seed = H.get_seed_config(bk.env, dir_) if fires else None
    print(f"{label:34s} fires={fires} seed={seed.config.get('block_sizes') if seed else None} "
          f"compute_ops={total} sfu={sfu} other={other} loads={load} stores={store} "
          f"bpe={getattr(f,'bandwidth_bytes_per_elem',None)} reg={getattr(f,'register_bytes_per_elem',None)} total_numel={getattr(f,'total_numel',None)}")


def main():
    def rr(*shape, dt=torch.bfloat16):
        return torch.randn(*shape, device=DEV, dtype=dt)
    # compute-heavy probes
    for shard, name, shp in [
        ("shard_warps.json", "wide_row_trig_chain", (2048, 2048)),
        ("shard_warps.json", "wide_row_moderate_chain", (2048, 2048)),
        ("shard_warps.json", "skinny_m_wide_n_chain", (64, 65536)),
        ("shard_temporaries.json", "horner_poly24", (4096, 4096)),
        ("shard_temporaries.json", "manydiff_chains", (4096, 4096)),
        ("shard_temporaries.json", "wide_fanin_combine", (4096, 2048)),
        ("shard_compute.json", "chained_activation_2d", (4096, 4096)),
        ("shard_compute.json", "iterated_map_fp32", (4096, 4096)),
        ("shard_broadcast.json", "bcast_rowvec_transcend", (8192, 8192)),
    ]:
        try:
            fn, mk = load_shard(shard, name); profile(fn, tuple(mk(shp)), f"{name}{list(shp)}")
        except Exception as e:
            print(f"{name}: ERR {type(e).__name__}: {str(e)[:100]}")
    # Tier-A flat (light intensity)
    try:
        from examples.swiglu import _swiglu_fwd
        from examples.geglu import _geglu
        import ptw_kernels as PK
        profile(_swiglu_fwd, (rr(8192, 4096), rr(8192, 4096)), "swiglu[8192,4096]")
        profile(_geglu, (rr(8192, 4096), rr(8192, 4096)), "geglu[8192,4096]")
        profile(PK.relu_squared, (rr(8192, 8192),), "relu_squared[8192,8192]")
        profile(PK.bias_gelu, (rr(8192, 8192), rr(8192)), "bias_gelu[8192,8192]")
    except Exception as e:
        print(f"tierA ERR {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
