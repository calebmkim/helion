"""Inspect the bind cache key + seed config for two shapes."""
import os, sys, importlib
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

mod_name, kern_attr, builder, sub, key_fn = B.SPECS["per_token_group_fp8_quant"]
mod = importlib.import_module(mod_name)
kfn = getattr(mod, kern_attr)

SHAPES = [tuple(int(x) for x in s.split("x")) for s in sys.argv[1].split(",")]

for shape in SHAPES:
    tok, hidden, group = shape
    args = builder(tok, hidden, group)[0]
    torch._dynamo.reset()
    sig = kfn._base_specialization_key(args)
    ck = kfn._get_bound_kernel_cache_key(args, sig)
    print(f"shape={shape} gpr={hidden//group}")
    print(f"  static_shapes={kfn.settings.static_shapes} index_dtype={kfn.settings.index_dtype}")
    print(f"  base_sig={sig}")
    print(f"  cache_key(before bind)={ck}")
    bound = kfn.bind(args)
    sig2 = kfn._base_specialization_key(args)
    ck2 = kfn._get_bound_kernel_cache_key(args, sig2)
    seeds = list(bound.env.config_spec.compiler_seed_configs)
    print(f"  cache_key(after bind) ={ck2}")
    print(f"  seed block_sizes={dict(seeds[0].config)['block_sizes'] if seeds else None}")
    print(f"  n_bound_kernels={len(kfn._bound_kernels)} keys={list(kfn._bound_kernels.keys())}")
    print()
