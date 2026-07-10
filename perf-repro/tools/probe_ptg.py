"""Standalone probe: reproduce the per_token_group seed-config flip.

Mimics the harness exactly: one module-level kfn reused across shapes,
torch._dynamo.reset() per cell, _extract_configs per cell, NO kfn.reset().

Usage:
  probe_ptg.py "1x2048x128,4096x4096x128"          # in-process sequence
  probe_ptg.py "4096x4096x128" --reset             # kfn.reset() per cell
  probe_ptg.py "4096x4096x128"                      # isolated single
"""
import os, sys, argparse, importlib

_PERF = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WT = os.path.abspath(os.path.join(_PERF, ".."))
_DEPS = os.path.join(_PERF, "deps")
for _d in (_WT, os.path.join(_WT, "examples"), _DEPS, os.path.join(_DEPS, "kut")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402
import helion  # noqa: E402
import bench_arms as B  # noqa: E402


def extract(kfn, args):
    bound = kfn.bind(args)
    spec = bound.env.config_spec
    seeds = list(spec.compiler_seed_configs)
    fired = list(spec.autotuner_heuristics)
    seed = seeds[0] if seeds else None
    del bound
    return (dict(seed.config) if seed else None), fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes")
    ap.add_argument("--reset", action="store_true", help="kfn.reset() per cell")
    args = ap.parse_args()

    shapes = []
    for tok in args.shapes.split(","):
        p = [int(x) for x in tok.split("x")]
        shapes.append(p)

    mod_name, kern_attr, builder, sub, key_fn = B.SPECS["per_token_group_fp8_quant"]
    mod = importlib.import_module(mod_name)
    kfn = getattr(mod, kern_attr)
    print(f"kfn id={id(kfn)} reset_per_cell={args.reset}")

    for shape in shapes:
        tok, hidden = shape[0], shape[1]
        group = shape[2] if len(shape) > 2 else 128
        built = builder(tok, hidden, group)
        a = built[0]
        torch._dynamo.reset()
        if args.reset:
            kfn.reset()
        seed_cfg, fired = extract(kfn, a)
        bs = seed_cfg.get("block_sizes") if seed_cfg else None
        gpr = hidden // group
        print(f"  shape={shape} groups_per_row={gpr} -> block_sizes={bs} fired={fired}")


if __name__ == "__main__":
    main()
