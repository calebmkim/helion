"""Rigorous N-arm cold-L2 comparison for ONE cell: reuses mmperf's M1 (cudagraph
graph-diff) + M2 (do_bench) machinery to time an arbitrary set of {label: config|token}
arms interleaved, R rounds, raw arrays retained, accuracy-gated. Tokens: 'seed',
'helion_default', 'tc' (the library arm); a dict is a literal Config (e.g. the autotuned one).

Usage:
  CUDA_VISIBLE_DEVICES=0 MMPERF_WORKTREE=<wt> python compare_configs.py \
      --kernel bmm --shape 16,4096,128,4096 --dtype bf16 \
      --arms '{"seed":"seed","autotuned":{...cfg...},"helion_default":"helion_default","cuBLAS":"tc"}' \
      --rounds 9 --out /tmp/cmp.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WT = os.path.dirname(os.path.dirname(_HERE))
WORKTREE = os.environ.get("MMPERF_WORKTREE", _DEFAULT_WT)
for p in (WORKTREE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import helion  # noqa: E402
import mmperf  # noqa: E402

assert os.path.realpath(helion.__file__).startswith(os.path.realpath(WORKTREE) + os.sep)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shape = [int(s) for s in a.shape.split(",")]
    ss = mmperf.STATIC_SHAPES[a.kernel]
    fn = mmperf._kernel_fn(a.kernel)
    arms_spec = json.loads(a.arms)  # ordered dict {label: token|dict}
    args, ref, tc_fn, meta = mmperf.make_inputs(a.kernel, shape, a.dtype)

    # resolve seed/default configs from a live probe
    seed_cfg, default_cfg, fired, facts, n_seeds = mmperf.probe(a.kernel, fn, args, ss)

    # build thunks (accuracy-gate each helion arm first)
    thunks, acc, cfgs = {}, {}, {}
    for label, tok in arms_spec.items():
        try:
            if tok == "tc":
                if tc_fn is None:
                    acc[label] = ("skip", None)
                    continue
                out = tc_fn()
                ok, ma = mmperf.accuracy_ok(out, ref)
                acc[label] = ("ok" if ok else "acc_fail", ma)
                thunks[label] = tc_fn
                cfgs[label] = "tc_max_autotune"
            else:
                cfg = seed_cfg if tok == "seed" else default_cfg if tok == "helion_default" else tok
                k = mmperf._build(a.kernel, fn, cfg, ss)
                out = k(*args)
                ok, ma = mmperf.accuracy_ok(out, ref)
                acc[label] = ("ok" if ok else "acc_fail", ma)
                thunks[label] = (lambda kk: (lambda: kk(*args)))(k)
                cfgs[label] = cfg
        except Exception as e:
            acc[label] = (f"fail:{type(e).__name__}", None)

    order = [l for l in arms_spec if l in thunks and acc[l][0] == "ok"]
    flush_buf = mmperf.make_flush_buf()
    R = a.rounds

    # ---- M1 ----
    g_flush = mmperf.capture_flush(flush_buf)
    graphs, inner = {}, {}
    for l in order:
        g, _ = mmperf.capture_graph(thunks[l], flush_buf)
        graphs[l] = g
        inner[l] = mmperf.pick_inner(g, g_flush)
    m1 = {l: [] for l in order}
    for _r in range(R):
        for l in order:
            ii = inner[l]
            tf = mmperf.time_graph_ms(graphs[l], ii)
            tfl = mmperf.time_graph_ms(g_flush, ii)
            m1[l].append(round(max(0.0, (tf - tfl) / ii * 1000.0), 4))
    del graphs, g_flush
    torch.cuda.synchronize()

    # ---- M2 ----
    m2 = {l: [] for l in order}
    for _r in range(R):
        for l in order:
            m2[l].append(round(mmperf.do_bench_ms(thunks[l]) * 1000.0, 4))

    info = {"kernel": a.kernel, "shape": shape, "dtype": a.dtype, "R": R,
            "seed_cfg": seed_cfg, "arms": {}}
    for l in arms_spec:
        info["arms"][l] = {
            "status": acc.get(l, ("missing", None))[0],
            "max_abs": acc.get(l, (None, None))[1],
            "config": cfgs.get(l),
            "M1_t_us": m1.get(l, []), "M1_median": med(m1.get(l, [])),
            "M2_t_us": m2.get(l, []), "M2_median": med(m2.get(l, [])),
        }
    with open(a.out, "w") as fh:
        json.dump(info, fh, default=str)

    # print a compact table
    print(f"\n=== {a.kernel} {shape} {a.dtype}  (R={R}, M1=cudagraph coldL2 canonical) ===")
    print(f"{'arm':<16}{'M1 us':>10}{'M2 us':>10}{'acc_maxabs':>12}")
    for l in arms_spec:
        d = info["arms"][l]
        m1s = f"{d['M1_median']:.2f}" if d["M1_median"] else "n/a"
        m2s = f"{d['M2_median']:.2f}" if d["M2_median"] else "n/a"
        mas = f"{d['max_abs']:.4f}" if d["max_abs"] is not None else "-"
        print(f"{l:<16}{m1s:>10}{m2s:>10}{mas:>12}  status={d['status']}")
    # ratios vs cuBLAS + vs seed (M1)
    def m1med(l):
        return info["arms"].get(l, {}).get("M1_median")
    tc_l = next((l for l, t in arms_spec.items() if t == "tc"), None)
    seed_l = next((l for l, t in arms_spec.items() if t == "seed"), None)
    print()
    for l in order:
        s = m1med(l)
        line = f"  {l}: "
        if tc_l and m1med(tc_l) and s:
            line += f"G_vs_cuBLAS={m1med(tc_l)/s:.3f}  "
        if seed_l and m1med(seed_l) and s:
            line += f"speedup_vs_seed={m1med(seed_l)/s:.3f}x"
        print(line)


if __name__ == "__main__":
    main()
