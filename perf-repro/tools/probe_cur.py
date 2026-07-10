"""Curriculum blast probe: full-in-process-sequence vs isolated seed configs,
using the harness's own _cur_build (so kfn objects match the harness exactly).

Usage: probe_cur.py <kernel> <shapesCSV mxn> --mode full|isolated --dtype fp32
"""
import os, sys, argparse, json
_PERF = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WT = os.path.abspath(os.path.join(_PERF, ".."))
_DEPS = os.path.join(_PERF, "deps")
for _d in (_PERF, _WT, os.path.join(_WT, "examples"), _DEPS, os.path.join(_DEPS, "kut")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")
import torch  # noqa: E402
import helion  # noqa: E402
import perf_report_bench as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("shapes")
    ap.add_argument("--mode", choices=["full", "isolated"], default="full")
    ap.add_argument("--dtype", default="fp32")
    args = ap.parse_args()
    dt = P._DT[args.dtype]
    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]

    seen_static = None
    out = []
    for (m, n) in shapes:
        kfn, kargs, ref, acc, tc = P._cur_build(args.kernel, m, n, dt)
        seen_static = kfn.settings.static_shapes
        torch._dynamo.reset()
        if args.mode == "isolated":
            kfn.reset()
        bound = kfn.bind(kargs)
        seeds = list(bound.env.config_spec.compiler_seed_configs)
        seed = dict(seeds[0].config) if seeds else None
        del bound
        out.append({"shape": [m, n], "seed": seed})
    print(json.dumps({"static_shapes": seen_static, "rows": out}))


if __name__ == "__main__":
    main()
