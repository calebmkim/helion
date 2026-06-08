"""GENERALITY (NEW REDUCTION OP): static classification + v8-fires + seed-USED +
correctness probe for the max/min fixtures (a genuinely different accumulator than
the sum/mean curriculum). Reuses the classify_ce_welford / gen_classify_new probe.

For each (max,min) shape:
  - dump ReductionFact (num_load / itemsize / num_reduction_ops / is_structured_combine)
    to PROVE the workload class (expect num_load=1, is_structured_combine=False -> same
    class as sum_kernel; the OP is not a heuristic input).
  - compiler_seed_configs -> assert exactly 1 seed (v8 FIRES).
  - build a config=[seed] kernel, normalize, inspect codegen -> assert the seed's
    persistent-vs-looped choice is what codegen emits (seed USED).
  - run + check vs torch.amax/amin (fp32 EXACT: max/min select an existing element,
    so the result is bitwise-identical; tol justified == 0).
  - classify T1 (block_id in reduction_loops.valid_block_ids()).

Run with the canonical invocation (SETUP.md).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from _lab.harness.fixture_maxmin import max_kernel  # noqa: E402
from _lab.harness.fixture_maxmin import min_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402


def args(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    return (x,)


def reference(which, x):
    return torch.amax(x, dim=-1) if which == "max" else torch.amin(x, dim=-1)


def codegen_persistent(triton_code):
    """T1 persistent iff there is NO inner reduction `for roffset` loop."""
    return "for roffset" not in triton_code


def probe(which, kern, m, n):
    x, = args(m, n)
    bound = kern.bind((x,))
    spec = bound.env.config_spec
    dev = bound.host_function.device_ir

    rf = getattr(spec, "reduction_facts", [])
    print(f"--- {which} ({m},{n}) ---")
    print(f"  block_sizes={len(spec.block_sizes)} reduction_loops={len(spec.reduction_loops)} "
          f"reduction_facts={len(rf)}")
    for f in rf:
        print(f"  FACT block_id={f.block_id} size_hint={f.size_hint} itemsize={f.itemsize} "
              f"num_load={f.num_load} num_store={f.num_store} "
              f"num_reduction_ops={f.num_reduction_ops} "
              f"num_tiled_accumulators={f.num_tiled_accumulators} "
              f"static_rnumel={f.static_rnumel} "
              f"is_structured_combine={getattr(f, 'is_structured_combine', '?')}")

    # FIRES: exactly 1 seed
    seeds = compiler_seed_configs(bound.env, dev)
    assert len(seeds) == 1, f"v8 did NOT fire 1 seed: got {len(seeds)}"
    sd = dict(seeds[0])
    names = list(spec.autotuner_heuristics)
    print(f"  v8 FIRES: 1 seed; heuristics_used={names}")
    print(f"    seed: block_sizes={sd.get('block_sizes')} "
          f"reduction_loops={sd.get('reduction_loops')} "
          f"num_warps={sd.get('num_warps')} num_stages={sd.get('num_stages')}")

    # SEED USED: build config=[seed], normalize, compare codegen vs seed intent
    k = helion.kernel(kern.fn, configs=[helion.Config(**sd)])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    cfg = dict(b._config)
    tcode = b.to_triton_code(helion.Config(**cfg))
    cg_persist = codegen_persistent(tcode)
    seed_persist = cfg.get("reduction_loops", [None])[0] is None
    assert cg_persist == seed_persist, (
        f"SEED-USED MISMATCH: codegen_persist={cg_persist} seed_persist={seed_persist} "
        f"reduction_loops={cfg.get('reduction_loops')}")
    # T1 classification
    rl_valid = spec.reduction_loops.valid_block_ids()
    is_t1 = bool(rf) and rf[0].block_id in rl_valid
    print(f"  SEED USED: codegen {'persist' if cg_persist else 'looped'} "
          f"== seed ({'persist' if seed_persist else 'looped'}); "
          f"T1={is_t1} (rl_valid={rl_valid}) warps={cfg.get('num_warps')}")

    # CORRECTNESS: max/min select an existing element -> exact equality
    out = b((x,)[0] if False else x)
    ref = reference(which, x)
    maxabs = float((out.to(torch.float32) - ref).abs().max())
    exact = bool(torch.equal(out, ref))
    print(f"  CORRECT: exact={exact} maxabs={maxabs:.3e} "
          f"(allclose@0={'OK' if maxabs == 0.0 else 'NONZERO'})")
    assert maxabs == 0.0, f"max/min must be EXACT vs torch (got {maxabs})"
    print()


def main():
    print(f"helion={helion.__file__}\n")
    shapes = [(2048, 1024), (2048, 16384), (8192, 256), (8192, 8192),
              (32768, 256), (256, 131072)]
    for which, kern in (("max", max_kernel), ("min", min_kernel)):
        print(f"##### {which.upper()} #####")
        for (m, n) in shapes:
            probe(which, kern, m, n)


if __name__ == "__main__":
    main()
