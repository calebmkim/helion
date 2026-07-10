"""Config-recorder over the FULL active matrix (Gate R tool). Records the SEED config (no bench) for
every curriculum cell so a BEFORE/AFTER diff scopes the re-bench: byte-identical cells are
perf-invariant (skip), changed cells get re-benched. Also records whether the pointwise fact fires
(negatives must stay 0 / not fire).

Usage:
  python cfg_recorder.py record  <out.json>
  python cfg_recorder.py diff    <before.json> <after.json>
"""
from __future__ import annotations
import sys, json, os, importlib.util
# HELION source root (override with HELION_SRC to record a BEFORE at a baseline worktree). The
# curriculum (_lab, shards) is gitignored so it ALWAYS comes from the main worktree — only the
# helion package + examples resolve from _WT, so a BEFORE/AFTER diff isolates the heuristic change.
_MAIN = "/home/calebkim/helion-new-heuristics/helion-pointwise"
_WT = os.environ.get("HELION_SRC", _MAIN)
sys.path.insert(0, os.path.join(_MAIN, "_lab", "pointwise"))
sys.path.insert(0, os.path.join(_MAIN, "_lab", "prompts"))
sys.path.insert(0, _WT)  # helion + examples resolve from the (possibly baseline) source root
import torch, helion
import helion.language as hl
assert helion.__file__.startswith(_WT), helion.__file__
from helion._compiler.autotuner_heuristics import compiler_seed_configs
DEV = "cuda"
SHARDS = "/home/calebkim/helion-new-heuristics/local/rope_probe/adv/shards/"


def _ab(m, n, dt=torch.bfloat16):
    return (torch.randn(m, n, device=DEV, dtype=dt), torch.randn(m, n, device=DEV, dtype=dt))


def _load_shard_fn(shard, name):
    e = [x for x in json.load(open(SHARDS + shard)) if x["name"] == name][0]
    os.makedirs("/tmp/cfgrec", exist_ok=True); p = f"/tmp/cfgrec/{name}.py"
    open(p, "w").write("from __future__ import annotations\nimport torch,helion\nimport helion.language as hl\n" + e["code"])
    sp = importlib.util.spec_from_file_location("rec_" + name, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return getattr(m, name), m.make_inputs, e["shapes"]


def tierA_registry():
    from examples.add import add as _add
    from examples.geglu import _geglu
    from examples.swiglu import _swiglu_fwd
    import ptw_kernels as PK
    import shapes_pointwise_draft as SH
    reg = {
        "swiglu": (_swiglu_fwd, _ab),
        "geglu": (_geglu, _ab),
        "residual_add": (_add, _ab),
        "relu_squared": (PK.relu_squared, lambda m, n: (torch.randn(m, n, device=DEV, dtype=torch.bfloat16),)),
        "bias_gelu": (PK.bias_gelu, lambda m, n: (torch.randn(m, n, device=DEV, dtype=torch.bfloat16), torch.randn(n, device=DEV, dtype=torch.bfloat16))),
        "dyt": (PK.dyt, lambda m, n: (torch.randn(m, n, device=DEV, dtype=torch.bfloat16), torch.randn(n, device=DEV, dtype=torch.bfloat16), torch.randn(n, device=DEV, dtype=torch.bfloat16), 0.5)),
    }
    out = []
    for k, (fn, mk) in reg.items():
        shapes = []
        for sp in ("train", "val", "test", "robustness"):
            shapes += [tuple(s) for s in SH.SHAPES[k].get(sp, [])]
        out.append(("A:" + k, fn, mk, shapes))
    return out


def rope_registry():
    sp = importlib.util.spec_from_file_location("rope_ex", os.path.join(_WT, "examples", "rope.py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    def mk(seq, hd):
        return (torch.randn(1, 32, seq, hd, device=DEV, dtype=torch.bfloat16),
                torch.randn(1, 8, seq, hd, device=DEV, dtype=torch.bfloat16),
                torch.randn(1, seq, hd, device=DEV, dtype=torch.bfloat16),
                torch.randn(1, seq, hd, device=DEV, dtype=torch.bfloat16))
    shapes = [(2048, 256), (4096, 256), (8192, 256), (16384, 256), (1024, 256), (2048, 64), (2048, 512), (2048, 2048)]
    return [("B:rope_fwd", m.rope_fwd, mk, shapes)]


def negatives_registry():
    out = []
    try:
        from examples.softmax import softmax
        out.append(("N:softmax", softmax, lambda m, n: (torch.randn(m, n, device=DEV, dtype=torch.bfloat16),), [(2048, 4096), (4096, 8192)]))
    except Exception as e:
        out.append(("N:softmax", None, None, str(e)))
    try:
        from examples.rms_norm import rms_norm_fwd
        out.append(("N:rms_norm", rms_norm_fwd, lambda m, n: (torch.randn(m, n, device=DEV, dtype=torch.bfloat16), torch.randn(n, device=DEV, dtype=torch.bfloat16)), [(2048, 4096), (4096, 8192)]))
    except Exception as e:
        out.append(("N:rms_norm", None, None, str(e)))
    try:
        from examples.matmul import matmul
        out.append(("N:matmul", matmul, lambda m, n: (torch.randn(m, n, device=DEV, dtype=torch.bfloat16), torch.randn(n, m, device=DEV, dtype=torch.bfloat16)), [(1024, 1024), (2048, 2048)]))
    except Exception as e:
        out.append(("N:matmul", None, None, str(e)))
    return out


def adversarial_registry():
    out = []
    for shard in sorted(os.listdir(SHARDS)):
        if not shard.endswith(".json"):
            continue
        for e in json.load(open(SHARDS + shard)):
            name = e["name"]
            try:
                fn, mk, shapes = _load_shard_fn(shard, name)
                out.append(("C:" + name, fn, lambda *s, _mk=mk: tuple(_mk(tuple(s[0]))), [tuple(x) for x in shapes]))
            except Exception as ex:
                out.append(("C:" + name, None, None, str(ex)))
    return out


def record(out_path):
    reg = tierA_registry() + rope_registry() + adversarial_registry() + negatives_registry()
    data = {}
    for entry in reg:
        label, fn = entry[0], entry[1]
        if fn is None:
            data[label] = {"load_error": entry[3]}; continue
        mk, shapes = entry[2], entry[3]
        cells = {}
        for shape in shapes:
            key = "x".join(str(s) for s in shape)
            try:
                args = tuple(mk(*shape)) if not label.startswith("C:") else tuple(mk(shape))
                bk = fn.bind(args)
                spec = bk.config_spec
                fires = bool(spec.pointwise_facts)
                seeds = compiler_seed_configs(bk.env, bk.host_function.device_ir)
                pw_seed = None
                # find the pointwise seed among the collected seeds (block_sizes-only, no reduction knobs)
                bs = None; nw = None
                if fires and seeds:
                    for s in seeds:
                        if "block_sizes" in s.config:
                            bs = s.config.get("block_sizes"); nw = s.config.get("num_warps"); break
                f = spec.pointwise_facts[0] if fires else None
                cells[key] = {
                    "fires": fires, "block_sizes": bs, "num_warps": nw,
                    "contig": list(getattr(f, "contig_block_ids", ()) or ()) if f else None,
                    "bpe": (getattr(f, "bandwidth_bytes_per_elem", None) if getattr(f, "bandwidth_bytes_per_elem", None) is not None else getattr(f, "bytes_per_elem", None)) if f else None,
                    "reg": (getattr(f, "register_bytes_per_elem", None) if getattr(f, "register_bytes_per_elem", None) is not None else getattr(f, "reg_bytes_per_elem", None)) if f else None,
                }
            except Exception as ex:
                cells[key] = {"error": f"{type(ex).__name__}: {str(ex)[:120]}"}
            torch.cuda.empty_cache()
        data[label] = cells
    json.dump(data, open(out_path, "w"), indent=1)
    n_fire = sum(1 for lab, cs in data.items() if isinstance(cs, dict) and any(c.get("fires") for c in cs.values() if isinstance(c, dict)))
    print(f"RECORDED {out_path}: {len(data)} kernels, {sum(len(cs) for cs in data.values() if isinstance(cs,dict))} cells, {n_fire} kernels fire pointwise")


def _cfgkey(c):
    if not isinstance(c, dict) or "error" in c or "load_error" in c:
        return ("ERR",)
    return (c.get("fires"), tuple(c.get("block_sizes") or []), c.get("num_warps"))


def diff(before_path, after_path):
    b = json.load(open(before_path)); a = json.load(open(after_path))
    changed = 0; errs = 0
    for label in sorted(set(b) | set(a)):
        bc, ac = b.get(label, {}), a.get(label, {})
        for key in sorted(set(bc) | set(ac)):
            cb, ca = bc.get(key, {}), ac.get(key, {})
            if isinstance(cb, dict) and ("error" in cb or "error" in ca):
                errs += 1
                print(f"  ERR   {label} [{key}] before={cb.get('error')} after={ca.get('error')}")
                continue
            if _cfgkey(cb) != _cfgkey(ca):
                changed += 1
                print(f"  CHANGE {label} [{key}]: {cb.get('block_sizes')} w{cb.get('num_warps')} fires={cb.get('fires')}  ->  {ca.get('block_sizes')} w{ca.get('num_warps')} fires={ca.get('fires')}")
    print(f"DIFF: {changed} changed cells, {errs} errors")


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(sys.argv[2])
    elif sys.argv[1] == "diff":
        diff(sys.argv[2], sys.argv[3])
