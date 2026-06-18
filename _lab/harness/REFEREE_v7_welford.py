"""INDEPENDENT results-referee for v7 welford Band-C structured-combine.

For each shape:
  - pull the REAL emitted seed via compiler_seed_configs (not a hardcoded approx),
  - confirm is_structured_combine=True + apply_block_ids,
  - run bare configs=[seed] (no autotune), inspect codegen (for-loops, persistent
    apply, combine tile constexpr) to confirm the seed is USED,
  - correctness vs torch.nn.functional.layer_norm fp32: max_abs + max_rel,
  - G = tc_default_lat / seed_lat (median-of-medians do_bench).
Tests in-sample (incl non-pow2 1536) PLUS extra non-pow2 canaries.
"""
from __future__ import annotations

import math
import re
import sys
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402
from examples.welford import welford  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
IN_SAMPLE = [(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096)]
# Extra non-pow2 canaries (validation N + a smaller-M non-pow2)
CANARY = [(65536, 1536), (262144, 2560), (262144, 3072)]


def ref_layer_norm(weight, bias, x):
    return torch.nn.functional.layer_norm(
        x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=EPS
    )


def args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def med(fn, reps=5):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def measure(m, n, tag):
    a = args(m, n)
    weight, bias, x, _ = a
    ref = ref_layer_norm(weight, bias, x).float()

    bound = welford.bind(a)
    spec = bound.env.config_spec
    facts = spec.reduction_facts
    nfacts = len(facts)
    sc = scinfo = None
    if nfacts == 1:
        f = facts[0]
        sc = f.is_structured_combine
        scinfo = (f.block_id, f.size_hint, f.apply_block_ids)

    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    if not seeds:
        print(f"[{tag}] ({m},{n}) facts={nfacts} sc={sc} -> NO SEED (declined)")
        return None
    seed = dict(seeds[0])

    # Run bare seed (configs=[seed]) -> NO autotune
    k = helion.kernel(welford.fn, configs=[helion.Config(**seed)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    code = b.to_triton_code(helion.Config(**dict(b._config)))
    nloops = len(re.findall(r"for \w+ in tl\.range|for offset", code))
    out = b(*a)
    out = (out[0] if isinstance(out, tuple) else out).float()

    max_abs = float((out - ref).abs().max())
    denom = ref.abs().clamp_min(1e-4)
    max_rel = float(((out - ref).abs() / denom).max())
    # tol: fp32 layer_norm welford vs eager; rtol=1e-3 atol=1e-3
    ok = bool(torch.allclose(out, ref, rtol=1e-3, atol=1e-3))

    torch._dynamo.reset()
    tc = torch.compile(ref_layer_norm)
    tc(weight, bias, x)
    tclat = med(lambda: tc(weight, bias, x)) * 1000
    slat = med(lambda: b(*a)) * 1000
    g = tclat / slat

    flag = "" if ok else "   <<< WRONG"
    print(f"[{tag}] ({m},{n}) facts={nfacts} sc={sc} apply={scinfo[2] if scinfo else None}")
    print(f"        SEED bs={seed.get('block_sizes')} w={seed.get('num_warps')} "
          f"st={seed.get('num_stages')}  for-loops={nloops}")
    print(f"        CORRECT ok={ok} max_abs={max_abs:.2e} max_rel={max_rel:.2e} (tol r1e-3 a1e-3)")
    print(f"        G={g:.3f}  seed={slat:.1f}us  tc_default={tclat:.1f}us{flag}\n")
    return g, ok, seed.get("block_sizes")


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    print("=== IN-SAMPLE ===")
    gs = []
    all_ok = True
    for (m, n) in IN_SAMPLE:
        r = measure(m, n, "IN")
        if r:
            g, ok, _ = r
            all_ok = all_ok and ok
            if ok:
                gs.append(g)
    gm = math.exp(sum(math.log(v) for v in gs) / len(gs)) if gs else None
    print(f"G_welford GEOMEAN (in-sample) = {gm:.4f} over {len(gs)} CORRECT shapes\n")

    print("=== EXTRA NON-POW2 CANARIES ===")
    for (m, n) in CANARY:
        r = measure(m, n, "CANARY")
        if r:
            _, ok, _ = r
            all_ok = all_ok and ok

    print(f"\nALL CORRECT (in-sample + canary) = {all_ok}")


if __name__ == "__main__":
    main()
