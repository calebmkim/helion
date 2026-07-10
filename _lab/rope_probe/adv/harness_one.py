"""Trusted, deterministic measurement of ONE adversarial kernel candidate, isolated per process.

Compares the pointwise SEED config vs the compiler DEFAULT config (block_size=32 = "the bar,
what happens without the heuristic"), accuracy-gated (seed output must match default output),
cold-L2 do_bench. A FINDING = seed >1.10x slower than default, OR seed compile/run fails while
default works. This is exactly the failure class RoPE was (seed [1,256] worse than default [1,32]).

argv: shard_json entry_index shape_index
Emits one JSON line.
"""
from __future__ import annotations

import json
import sys

_WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
sys.path.insert(0, _WT)

import torch
import helion
assert helion.__file__.startswith(_WT), helion.__file__
import helion.language as hl  # noqa: F401  (exposed to exec'd kernel code)
from helion._compiler.autotuner_heuristics.triton import TritonPointwiseSeedHeuristic
from helion.autotuner.benchmarking import do_bench


def _as_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x,)
    if isinstance(x, (list, tuple)):
        return tuple(t for t in x if isinstance(t, torch.Tensor))
    return ()


def _outputs_agree(a, b):
    ta, tb = _as_tuple(a), _as_tuple(b)
    if len(ta) != len(tb) or not ta:
        return False
    for x, y in zip(ta, tb):
        if x.shape != y.shape:
            return False
        if not torch.isfinite(x.float()).all():
            return False
        if not torch.allclose(x.float(), y.float(), atol=2e-2, rtol=2e-2):
            return False
    return True


def _bench(run, args):
    return do_bench(lambda: run(*args), return_mode="median")


def main():
    shard, ei, si = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    entry = json.load(open(shard))[ei]
    name = entry["name"]
    shape = entry["shapes"][si]
    out = {"lens": entry.get("targeted_gap"), "name": name, "shape": shape,
           "hypothesis": entry.get("hypothesis", "")[:160]}
    try:
        # helion.kernel needs inspect.getsource → write to a REAL module file and import it.
        import importlib.util
        import os
        gen_dir = os.path.join(os.path.dirname(shard), "_gen")
        os.makedirs(gen_dir, exist_ok=True)
        mod_path = os.path.join(gen_dir, f"advk_{ei}_{name}.py")
        header = ("from __future__ import annotations\n"
                  "import torch\nimport helion\nimport helion.language as hl\n\n")
        with open(mod_path, "w") as f:
            f.write(header + entry["code"])
        spec_ = importlib.util.spec_from_file_location(f"advk_{ei}_{name}", mod_path)
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        fn = getattr(mod, name)
        args = tuple(mod.make_inputs(tuple(shape) if isinstance(shape, list) else shape))
    except Exception as e:
        out["status"] = f"bad_code: {type(e).__name__}: {str(e)[:180]}"
        print(json.dumps(out)); return

    try:
        bk = fn.bind(args)
        env, dev_ir, spec = bk.env, bk.host_function.device_ir, bk.config_spec
        if not TritonPointwiseSeedHeuristic.is_eligible(env, dev_ir):
            out["status"] = "not_pointwise (fact did not fire — reduction/matmul/accum present)"
            print(json.dumps(out)); return
        seed_cfg = TritonPointwiseSeedHeuristic.get_seed_config(env, dev_ir)
        default_cfg = spec.default_config()
        out["seed"] = seed_cfg.config.get("block_sizes")
        out["default"] = default_cfg.config.get("block_sizes")
        fact = spec.pointwise_facts[0]
        out["fact_bpe"] = fact.bandwidth_bytes_per_elem
        out["fact_reg_bpe"] = getattr(fact, "register_bytes_per_elem", None)
        out["fact_total_numel"] = fact.total_numel
    except Exception as e:
        out["status"] = f"bind_fail: {type(e).__name__}: {str(e)[:180]}"
        print(json.dumps(out)); return

    # default arm (the baseline / bar)
    try:
        drun = bk.compile_config(default_cfg)
        dref = drun(*args)
        default_ms = _bench(drun, args)
        out["default_ms"] = default_ms
    except Exception as e:
        out["status"] = f"default_failed: {type(e).__name__}: {str(e)[:140]}"
        print(json.dumps(out)); return

    # seed arm
    try:
        srun = bk.compile_config(seed_cfg)
        sref = srun(*args)
    except Exception as e:
        out["status"] = "FINDING: seed_compile/run_FAIL_default_OK"
        out["detail"] = f"{type(e).__name__}: {str(e)[:160]}"
        print(json.dumps(out)); return

    out["outputs_agree"] = _outputs_agree(sref, dref)
    try:
        seed_ms = _bench(srun, args)
        out["seed_ms"] = seed_ms
        out["ratio_seed_over_default"] = seed_ms / default_ms
    except Exception as e:
        out["status"] = f"seed_bench_fail: {type(e).__name__}: {str(e)[:140]}"
        print(json.dumps(out)); return

    if not out["outputs_agree"]:
        out["status"] = "output_mismatch (seed vs default differ — config-dependent correctness)"
    elif out["ratio_seed_over_default"] > 1.10:
        out["status"] = f"FINDING: seed {out['ratio_seed_over_default']:.2f}x SLOWER than default"
    else:
        out["status"] = f"ok (seed {1.0/out['ratio_seed_over_default']:.2f}x vs default)"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
