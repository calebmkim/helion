from __future__ import annotations
import sys
import torch
import helion
WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), helion.__file__
from examples.welford import welford, eager_layer_norm
from triton.testing import do_bench

M, N = 16384, 5120   # per-program physics is M-independent; small M for speed
EPS = 1e-5
weight = torch.rand(N, device="cuda", dtype=torch.float32)
bias = torch.rand(N, device="cuda", dtype=torch.float32)
x = torch.rand(M, N, device="cuda", dtype=torch.float32)
args = (weight, bias, x, EPS)
ref = eager_layer_norm(*args)

def med(fn, reps=7):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(reps))[reps // 2]

def introspect(label, cfg):
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args); b.ensure_config_exists(args)
    out = b(*args)
    ok = bool(torch.allclose(out, ref, rtol=1e-3, atol=1e-4))
    ms = med(lambda: b(*args))
    # find the compiled triton kernel(s) in the JIT cache
    import triton
    found = []
    for obj in list(globals().values()):
        pass
    # walk triton's JITFunction registry via the module's cache
    import gc
    for o in gc.get_objects():
        if isinstance(o, triton.runtime.jit.JITFunction) and o.__name__ == "_helion_welford":
            for ck in o.cache.values() if hasattr(o, "cache") else []:
                for v in (ck.values() if isinstance(ck, dict) else [ck]):
                    md = getattr(v, "metadata", None)
                    nr = getattr(v, "n_regs", None)
                    ns = getattr(v, "n_spills", None)
                    sh = getattr(md, "shared", None) if md else None
                    nw = getattr(md, "num_warps", None) if md else None
                    if nr is not None:
                        found.append((nr, ns, sh, nw))
    found = sorted(set(found))
    print(f"\n{label}: {ms*1000:8.1f} us  ok={ok}  cfg={cfg['block_sizes']} w={cfg['num_warps']}")
    for nr, ns, sh, nw in found:
        threads = (nw or 0) * 32
        print(f"    regs/thread={nr:4d}  spills={ns:4d}  shared={sh}B  num_warps={nw}  -> regs/SM_used≈{nr*threads}")

base = dict(load_eviction_policies=['last','first','first','first'], num_stages=1, pid_type='flat')
introspect("SEED bm=16 bn=8192 w16", {**base, 'block_sizes':[16,8192,2048], 'num_warps':16})
introspect("bm=1 bn=2048 w16", {**base, 'block_sizes':[1,2048,2048], 'num_warps':16})
introspect("bm=1 bn=2048 w4 (BEST)", {**base, 'block_sizes':[1,2048,2048], 'num_warps':4})
introspect("bm=16 bn=2048 w16", {**base, 'block_sizes':[16,2048,2048], 'num_warps':16})
introspect("bm=16 bn=2048 w4", {**base, 'block_sizes':[16,2048,2048], 'num_warps':4})
