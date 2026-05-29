"""AUDITOR Task 3b: can a GENERALIZABLE per-slot eviction rule capture the
ev_oracle win, or is the win only reachable by the per-shape autotuned pattern?

ev_oracle wins big (+7-23%) on 5/6 shapes at matched-v6 levers, but ev_oracle is
a DIFFERENT mixed list per shape -> not directly seedable. A heuristic must emit a
deterministic list. We test candidate PRINCIPLED rules (built at
spec.load_eviction_policies.length) on top of the exact v6 seed, all else equal:

  - last_load_first : the LAST eviction slot (widest contiguous reduction load)
                      -> 'first', rest ''        (single-pass-stream-the-row idea)
  - first_load_first: the FIRST eviction slot -> 'first', rest ''
  - last_load_last  : the LAST eviction slot -> 'last', rest ''
  - all_first / all_last : uniform (regression-prone, for reference)
  - ev_oracle        : the per-shape autotuned pattern (upper bound)

Goal: find ONE rule that (a) wins broadly like ev_oracle and (b) does NOT regress
the wide rms_norm (2048,16384) / long_sum. If none -> the win is autotuner-only.
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
        L = spec.load_eviction_policies.length

        def mk(setval, pos):
            x = [''] * L
            if 0 <= pos < L:
                x[pos] = setval
            return x

        variants = [
            ("v6 (baseline)", v6),
            ("last_load_first", {**v6, "load_eviction_policies": mk('first', L - 1)}),
            ("first_load_first", {**v6, "load_eviction_policies": mk('first', 0)}),
            ("last_load_last", {**v6, "load_eviction_policies": mk('last', L - 1)}),
            ("all_first", {**v6, "load_eviction_policies": ['first'] * L}),
            ("all_last", {**v6, "load_eviction_policies": ['last'] * L}),
            ("ev_oracle", {**v6, "load_eviction_policies": ORACLE_EV[(name, shape)]}),
        ]
        print(f"=== {name} {shape} (rnumel={shape[1]}, ev_len={L}, "
              f"oracle_ev={ORACLE_EV[(name, shape)]}) ===")
        base = None
        for label, cfg in variants:
            lat, status = bench_cfg(fn, args, cfg, ref)
            if label.startswith("v6"):
                base = lat
            ratio = (base / lat) if (base and lat) else float("nan")
            latstr = f"{lat:8.2f}us" if lat else f"{status:>10}"
            tag = ""
            if lat and base:
                if base / lat > 1.02:
                    tag = "  WIN"
                elif base / lat < 0.98:
                    tag = "  REGRESS"
            print(f"    {label:>17}: {latstr}  v6/var={ratio:5.3f}{tag}")
        print()


if __name__ == "__main__":
    main()
