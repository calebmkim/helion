"""AUDITOR independent probe (v7 / 53ed8762).

For each of the 8 active kernels + welford, compile and print:
  - non_grid_tiles, inner reduction axis, apply_tiles (non-grid non-reduction),
    each tile's size_hint
  - is_structured_combine, apply_block_ids on each ReductionFact
  - the seed config(s) emitted
This is an INDEPENDENT recomputation of the gate predicate from env+device_ir,
NOT trusting the worker's harness.
"""
from __future__ import annotations

import sys
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward  # noqa: E402
from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.welford import welford  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion._compiler.inductor_lowering import ReductionLowering  # noqa: E402

EPS = 1e-5


def rms_args(m, n):
    return (torch.randn(m, n, device="cuda"), torch.randn(n, device="cuda"), EPS)


def sum_args(m, n):
    return (torch.randn(m, n, device="cuda"),)


def ln_args(m, n):
    return (torch.randn(m, n, device="cuda"), [n], torch.randn(n, device="cuda"),
            torch.randn(n, device="cuda"), EPS)


def kl_args(m, v):
    return (torch.log_softmax(torch.randn(m, v, device="cuda"), -1),
            torch.softmax(torch.randn(m, v, device="cuda"), -1))


def jsd_args(m, v):
    return (torch.log_softmax(torch.randn(m, v, device="cuda"), -1),
            torch.log_softmax(torch.randn(m, v, device="cuda"), -1))


def ce_args(m, v):
    return (torch.randn(m, v, device="cuda"),
            torch.randint(0, v, (m,), device="cuda", dtype=torch.int64))


def wf_args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


CASES = [
    ("rms_norm", rms_norm_fwd, rms_args, (2048, 1024)),
    ("sum", sum_kernel, sum_args, (2048, 16384)),
    ("long_sum", longsum, sum_args, (16, 262144)),
    ("layer_norm", layer_norm_fwd, ln_args, (4096, 15872)),
    ("softmax", softmax_two_pass, sum_args, (4096, 16384)),
    ("kl_div", kl_div_forward, kl_args, (4096, 131072)),
    ("jsd", jsd_forward, jsd_args, (8192, 131072)),
    ("cross_entropy", cross_entropy, ce_args, (8192, 131072)),
    ("welford", welford, wf_args, (262144, 1536)),
]


def probe(name, kern, args):
    b = kern.bind(args)
    dev_ir = b.host_function.device_ir
    spec = b.env.config_spec
    env = b.env
    grid_ids = {x for bids in dev_ir.grid_block_ids for x in bids}
    all_bids = list(spec.block_sizes.valid_block_ids())
    non_grid = [x for x in all_bids if x not in grid_ids]

    red_ids = set()
    for gi in dev_ir.graphs:
        for node in gi.graph.nodes:
            low = node.meta.get("lowering")
            if isinstance(low, ReductionLowering):
                bid = getattr(low, "block_index", None)
                if bid is not None:
                    red_ids.add(bid)
    inner_red = [x for x in red_ids if x not in grid_ids]
    apply_tiles = [x for x in non_grid if x not in red_ids]

    def sh(bid):
        try:
            return env.block_sizes[bid].size_hint()
        except Exception:  # noqa: BLE001
            return None

    # Independent recomputation of the predicate
    indep_isc = len(non_grid) > 1 and len(apply_tiles) >= 1

    seeds = compiler_seed_configs(env, dev_ir)
    facts = spec.reduction_facts

    print(f"\n=== {name} {args[0].shape if hasattr(args[0],'shape') else ''} ===", flush=True)
    print(f"  grid_ids={sorted(grid_ids)} all_block_ids={sorted(all_bids)}", flush=True)
    print(f"  non_grid_tiles={non_grid} sizes={[sh(x) for x in non_grid]}", flush=True)
    print(f"  red_block_ids(raw)={sorted(red_ids)} inner_red={inner_red} "
          f"sizes={[sh(x) for x in inner_red]}", flush=True)
    print(f"  apply_tiles(non-grid non-red)={apply_tiles} sizes={[sh(x) for x in apply_tiles]}", flush=True)
    print(f"  INDEP is_structured_combine={indep_isc}", flush=True)
    print(f"  #reduction_facts={len(facts)}", flush=True)
    for f in facts:
        print(f"    FACT block_id={f.block_id} size_hint={f.size_hint} "
              f"static_rnumel={f.static_rnumel} num_tiled_acc={f.num_tiled_accumulators} "
              f"is_structured_combine={f.is_structured_combine} apply_block_ids={f.apply_block_ids}", flush=True)
    for s in seeds:
        print(f"    SEED {dict(s)}", flush=True)
    return name, [dict(s) for s in seeds], [f.is_structured_combine for f in facts]


def main():
    print(f"helion={helion.__file__}", flush=True)
    print(f"torch={torch.__version__} dev={torch.cuda.get_device_name(0)}", flush=True)
    results = {}
    for name, kern, argfn, shape in CASES:
        try:
            nm, seeds, sc = probe(name, kern, argfn(*shape))
            results[nm] = (seeds, sc)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  ERROR {name}: {e}", flush=True)
            traceback.print_exc()
    print("\n=== SUMMARY is_structured_combine per kernel ===", flush=True)
    for name in results:
        print(f"  {name}: sc={results[name][1]}", flush=True)


if __name__ == "__main__":
    main()
