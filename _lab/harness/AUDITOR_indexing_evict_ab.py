"""AUDITOR Task 3: MATCHED-LEVER A/B of indexing + load_eviction_policies.

The v3/pid lesson: hold ALL OTHER levers EQUAL to the v6 seed; flip ONLY the
codegen knob under test. Baseline = the exact v6 seed (block_sizes,
reduction_loops, num_warps, num_stages from compiler_seed_configs). Then, on top
of that identical bundle:

  indexing variants (build at spec.indexing.length; store slots held 'pointer'
  unless forced; load slots = complement of spec.store_indices):
    - idx_all_td     : every LOAD slot -> tensor_descriptor, stores pointer
    - idx_lastload_td: only the LAST load slot (the widest/most-contiguous
                       reduction-axis load) -> tensor_descriptor
    - idx_oracle     : the verbatim oracle indexing list (lever-isolated: only
                       indexing differs from v6; all else = v6 seed)

  eviction variants (build at spec.load_eviction_policies.length):
    - ev_all_first   : every load-eviction slot -> 'first' (single-pass stream)
    - ev_all_last    : every load-eviction slot -> 'last'  (reused operands)
    - ev_oracle      : the verbatim oracle eviction list

Metric: ratio v6/variant (>1 => variant FASTER => real win). Correctness checked
vs fp32 ref for every config; INCORRECT/ERR variants reported (not silently
dropped). do_bench median-of-N, fresh process, idle GPU.
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
import helion.runtime as rt  # noqa: E402

N_RUNS = 9
LONG = torch.int64
NUM_SM = rt.get_num_sm(torch.device("cuda"))


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def build_rms(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), 1e-5)


def build_x(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def build_ce(shape):
    n, v = shape
    return (torch.randn(n, v, device="cuda", dtype=torch.float32),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG))


# Verbatim oracle indexing/eviction lists (from logs/perf_inv_oracle.out).
ORACLE_IDX = {
    ("sum", (8192, 256)): ['pointer', 'pointer', 'tensor_descriptor'],
    ("rms_norm", (32768, 256)): ['pointer', 'pointer', 'pointer', 'pointer', 'pointer',
                                 'pointer', 'tensor_descriptor', 'tensor_descriptor'],
    ("softmax_two_pass", (32768, 256)): ['tensor_descriptor', 'tensor_descriptor',
                                         'tensor_descriptor'],
    ("cross_entropy", (4096, 4096)): ['tensor_descriptor', 'pointer', 'pointer',
                                      'pointer', 'pointer', 'pointer'],
    ("rms_norm", (2048, 16384)): ['pointer', 'tensor_descriptor', 'pointer',
                                  'tensor_descriptor', 'pointer', 'pointer', 'pointer',
                                  'tensor_descriptor'],
    ("long_sum", (256, 131072)): ['tensor_descriptor', 'tensor_descriptor', 'pointer'],
}
ORACLE_EV = {
    ("sum", (8192, 256)): ['last', 'first'],
    ("rms_norm", (32768, 256)): ['', 'last', 'last', '', 'last'],
    ("softmax_two_pass", (32768, 256)): ['', 'last'],
    ("cross_entropy", (4096, 4096)): ['', '', 'first', 'first', 'last'],
    ("rms_norm", (2048, 16384)): ['', '', 'first', '', 'first'],
    ("long_sum", (256, 131072)): ['first', 'last'],
}

CASES = [
    ("sum", (8192, 256), sum_kernel, build_x, lambda a: torch.sum(a[0], -1)),
    ("rms_norm", (32768, 256), rms_norm_fwd, build_rms, lambda a: rms_norm_pytorch(*a)),
    ("softmax_two_pass", (32768, 256), softmax_two_pass, build_x,
     lambda a: torch.nn.functional.softmax(a[0], 1)),
    ("cross_entropy", (4096, 4096), cross_entropy, build_ce,
     lambda a: torch.nn.functional.cross_entropy(*a)),
    ("rms_norm", (2048, 16384), rms_norm_fwd, build_rms, lambda a: rms_norm_pytorch(*a)),
    ("long_sum", (256, 131072), longsum, build_x, lambda a: torch.sum(a[0], -1)),
]


def correct(out, ref):
    o = out[0] if isinstance(out, tuple) else out
    o = o.float(); r = ref.float()
    return bool(o.shape == r.shape and torch.allclose(o, r, rtol=1e-3, atol=1e-3))


def bench_cfg(fn, args, cfg, ref):
    try:
        k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args); b.ensure_config_exists(args)
        out = b(*args)
        if not correct(out, ref):
            return None, "INCORRECT"
        return med(lambda: b(*args)) * 1000, "ok"
    except Exception as exc:
        return None, f"ERR:{type(exc).__name__}"


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} NUM_SM={NUM_SM} N_RUNS={N_RUNS}\n")
    for name, shape, fn, build, reffn in CASES:
        args = build(shape); ref = reffn(args)
        bound = fn.bind(args)
        spec = bound.env.config_spec
        v6 = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
        idx_len = spec.indexing.length
        ev_len = spec.load_eviction_policies.length
        store_idx = set(spec.store_indices)
        load_slots = [i for i in range(idx_len) if i not in store_idx]

        # build indexing variants (all at idx_len; stores stay 'pointer')
        idx_all_td = ['tensor_descriptor' if i in load_slots else 'pointer'
                      for i in range(idx_len)]
        idx_last_td = ['pointer'] * idx_len
        if load_slots:
            idx_last_td[load_slots[-1]] = 'tensor_descriptor'
        idx_oracle = ORACLE_IDX[(name, shape)]

        ev_all_first = ['first'] * ev_len
        ev_all_last = ['last'] * ev_len
        ev_oracle = ORACLE_EV[(name, shape)]

        variants = [
            ("v6 (baseline)", v6),
            ("idx_all_td", {**v6, "indexing": idx_all_td}),
            ("idx_lastload_td", {**v6, "indexing": idx_last_td}),
            ("idx_oracle", {**v6, "indexing": idx_oracle}),
            ("ev_all_first", {**v6, "load_eviction_policies": ev_all_first}),
            ("ev_all_last", {**v6, "load_eviction_policies": ev_all_last}),
            ("ev_oracle", {**v6, "load_eviction_policies": ev_oracle}),
        ]
        print(f"=== {name} {shape} (rnumel={shape[1]}) v6={v6}")
        print(f"    idx_len={idx_len} ev_len={ev_len} store_idx={sorted(store_idx)} "
              f"load_slots={load_slots}")
        base = None
        for label, cfg in variants:
            lat, status = bench_cfg(fn, args, cfg, ref)
            if label.startswith("v6"):
                base = lat
            ratio = (base / lat) if (base and lat) else float("nan")
            latstr = f"{lat:8.2f}us" if lat else f"{status:>10}"
            print(f"    {label:>16}: {latstr}  v6/var={ratio:5.3f}"
                  + ("  <-- WIN" if (lat and base and base/lat > 1.02) else ""))
        print()


if __name__ == "__main__":
    main()
