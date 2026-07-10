"""Blast-radius probe: for a vLLM kernel, compare seed config obtained in the
FULL in-process harness sequence vs ISOLATED re-extraction (one bind on a fresh
kfn per shape). Reports per-shape SAME/DIFF.

Two modes controlled by --mode:
  full     : reuse ONE module-level kfn across all shapes (NO reset) -> harness behavior
  isolated : kfn.reset() before each shape -> ground truth (== fresh process)

Usage: probe_blast.py <kernel> <shapesCSV like 1x2048x128,...> --mode full|isolated
"""
import os, sys, argparse, importlib, json
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
    seeds = list(bound.env.config_spec.compiler_seed_configs)
    seed = dict(seeds[0].config) if seeds else None
    del bound
    return seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("shapes")
    ap.add_argument("--mode", choices=["full", "isolated"], default="full")
    args = ap.parse_args()

    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]
    mod_name, kern_attr, builder, sub, key_fn = B.SPECS[args.kernel]
    mod = importlib.import_module(mod_name)
    kfn = getattr(mod, kern_attr)

    out = []
    for shape in shapes:
        tok, hidden = shape[0], shape[1]
        group = shape[2] if len(shape) > 2 else 128
        built = builder(tok, hidden, group)
        a = built[0]
        torch._dynamo.reset()
        if args.mode == "isolated":
            kfn.reset()
        seed = extract(kfn, a)
        out.append({"shape": list(shape), "seed": seed})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
