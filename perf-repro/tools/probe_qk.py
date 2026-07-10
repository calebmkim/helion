"""qk_norm_rope blast probe using the harness _qk_build."""
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

ap = argparse.ArgumentParser()
ap.add_argument("shapes")
ap.add_argument("--mode", choices=["full", "isolated"], default="full")
a = ap.parse_args()
shapes = [tuple(int(x) for x in s.split("x")) for s in a.shapes.split(",")]
out = []
for (qh, kvh, nt) in shapes:
    kfn, args, ref = P._qk_build(qh, kvh, nt)
    torch._dynamo.reset()
    if a.mode == "isolated":
        kfn.reset()
    bound = kfn.bind(args)
    seeds = list(bound.env.config_spec.compiler_seed_configs)
    seed = dict(seeds[0].config) if seeds else None
    del bound
    out.append({"shape": [qh, kvh, nt], "seed": seed, "static_shapes": kfn.settings.static_shapes})
print(json.dumps(out))
