"""Bare-forward seeded-Helion vs torch.compile-default — matches the PRIOR run's method.

The PR's TritonBench numbers route rms_norm/layer_norm through an autograd Function
wrapper (RMSNormFunction.apply), adding host overhead the forward-only torch.compile
reference doesn't pay. The PRIOR run (run3_task2_replay_bench.py) instead timed the
BARE forward kernel. This script reproduces that method but with the PR's LIVE seed
(compiler_seed_configs), for an apples-to-apples comparison to the prior tables.

Per (kernel, test-shape), single process, fresh dynamo per shape, contention-guarded:
  seed  = helion.kernel(fwd.fn, config=compiler_seed_configs(...)[0]); do_bench(bare call)
  tc    = torch.compile(fp32_reference)  (default mode)
G_seed = tc_lat / seed_lat. Build fns + tc refs copied verbatim from the prior harness.
"""

from __future__ import annotations

import json
import os
import statistics as st
import subprocess
import sys

import torch
from triton.testing import do_bench

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

sys.path.insert(0, "/home/dev/local/helion-reduction-heuristics-run2/_lab/prompts")
import shapes_v3_draft as SH  # noqa: E402

from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.welford import welford, eager_layer_norm  # noqa: E402
from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402
from examples.long_sum import longsum  # noqa: E402

EPS = 1e-5
LONG = torch.int64
N_RUNS = 15


def _first(o):
    return o[0] if isinstance(o, tuple) else o


def _foreign_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10).stdout
    except Exception:  # noqa: BLE001
        return 0
    me = os.getpid(); m = 0
    for line in out.splitlines():
        p = [c.strip() for c in line.split(",")]
        if len(p) == 2 and p[0].isdigit() and int(p[0]) != me:
            m = max(m, int(p[1]) if p[1].isdigit() else 0)
    return m


def _med(fn) -> float:
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2] * 1000.0


# build fns: (args, ref_tensor, out_extract) — verbatim from run3_task2_replay_bench.py
def b_rms(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32); w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, EPS), rms_norm_pytorch(x, w, EPS), _first


def b_ln(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32); w = torch.randn(n, device="cuda", dtype=torch.float32); b = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, [n], w, b, EPS), torch.nn.functional.layer_norm(x, [n], w, b, EPS), _first


def b_welford(m, n):
    w = torch.rand(n, device="cuda", dtype=torch.float32); b = torch.rand(n, device="cuda", dtype=torch.float32); x = torch.rand(m, n, device="cuda", dtype=torch.float32)
    a = (w, b, x, EPS); return a, eager_layer_norm(*a), _first


def b_softmax(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32); return (x,), torch.nn.functional.softmax(x, dim=1), _first


def b_ce(m, n):
    lg = torch.randn(m, n, device="cuda", dtype=torch.float32); lb = torch.randint(0, n, (m,), device="cuda", dtype=LONG)
    return (lg, lb), torch.nn.functional.cross_entropy(lg, lb), _first


def b_kl(m, n):
    yp = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1); yt = torch.randn(m, n, device="cuda", dtype=torch.float32).softmax(-1)
    return (yp, yt, False, "batchmean", 1e-10), torch.nn.KLDivLoss(reduction="batchmean").to("cuda")(yp, yt), _first


def _jsd_ref(lq, lp):
    # JSD(beta=0.5): 0.5*KL(p||m)+0.5*KL(q||m), m=0.5(p+q); p=exp(lq), q=exp(lp). Mean over rows.
    p, q = lq.exp(), lp.exp()
    mm = 0.5 * (p + q)
    return (0.5 * (p * (lq - mm.log())).sum(-1) + 0.5 * (q * (lp - mm.log())).sum(-1)).mean()


def b_jsd(m, n):
    lq = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1); lp = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1)
    return (lq, lp, None, 0.5, -100), _jsd_ref(lq, lp), _first


def b_sum(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32); return (x,), torch.sum(x, dim=-1), _first


def b_longsum(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32); return (x,), torch.sum(x, dim=-1), _first


KERNELS = {
    "rms_norm": (rms_norm_fwd, b_rms, lambda a: rms_norm_pytorch(*a)),
    "layer_norm": (layer_norm_fwd, b_ln, lambda a: torch.nn.functional.layer_norm(a[0], a[1], a[2], a[3], a[4])),
    "softmax": (softmax_two_pass, b_softmax, lambda a: torch.nn.functional.softmax(a[0], dim=1)),
    "sum": (sum_kernel, b_sum, lambda a: torch.sum(a[0], dim=-1)),
    "cross_entropy": (cross_entropy, b_ce, lambda a: torch.nn.functional.cross_entropy(a[0], a[1])),
    "long_sum": (longsum, b_longsum, lambda a: torch.sum(a[0], dim=-1)),
    "welford": (welford, b_welford, lambda a: eager_layer_norm(*a)),
    "kl_div": (kl_div_forward, b_kl, lambda a: torch.nn.KLDivLoss(reduction="batchmean").to("cuda")(a[0], a[1])),
    "jsd": (jsd_forward, b_jsd, lambda a: _jsd_ref(a[0], a[1])),
}


def bench(kn: str) -> dict:
    fn, build, tc_ref = KERNELS[kn]
    rows = []
    for (m, n) in SH.SHAPES[kn]["test"]:
        torch._dynamo.reset()
        args, ref, extract = build(m, n)
        bound0 = fn.bind(args)
        seeds = compiler_seed_configs(bound0.env, bound0.host_function.device_ir)
        seed = seeds[0] if seeds else bound0.config_spec.default_config()
        k_seed = helion.kernel(fn.fn, config=seed, static_shapes=True)
        out = extract(k_seed(*args))
        tol = 2e-2 if kn in ("kl_div", "jsd", "welford") else 1e-3
        acc = bool(torch.allclose(out.float(), ref.float(), rtol=tol, atol=tol))
        tcfn = torch.compile((lambda: tc_ref(args)) if tc_ref else (lambda: ref))
        if tc_ref:
            tcfn()
        f0 = _foreign_mib()
        t_seed = _med(lambda: k_seed(*args))
        t_tc = _med(tcfn) if tc_ref else float("nan")
        foreign = max(f0, _foreign_mib())
        rows.append({"shape": [m, n], "seed_us": round(t_seed, 2),
                     "tc_us": round(t_tc, 2),
                     "G_seed": round(t_tc / t_seed, 4) if tc_ref else None,
                     "acc": acc, "foreign_mib": foreign})
        print("ROW " + json.dumps(rows[-1]), file=sys.stderr)
    gs = [r["G_seed"] for r in rows if r["G_seed"]]
    return {"kernel": kn, "rows": rows,
            "median_G_seed": round(st.median(gs), 4) if gs else None,
            "geo_G_seed": round(st.geometric_mean(gs), 4) if gs else None,
            "min_G_seed": round(min(gs), 4) if gs else None,
            "max_G_seed": round(max(gs), 4) if gs else None,
            "any_acc_fail": any(not r["acc"] for r in rows),
            "max_foreign_mib": max(r["foreign_mib"] for r in rows)}


def main() -> None:
    assert helion.__file__.startswith("/home/dev/local/helion-pr-edit"), helion.__file__
    out = []
    for kn in (sys.argv[1:] or list(KERNELS)):
        sys.stderr.write(f"\n===== {kn} =====\n")
        try:
            r = bench(kn)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"  {kn} FAILED: {e}\n"); out.append({"kernel": kn, "error": str(e)}); continue
        sys.stderr.write(f"  median_G_seed={r['median_G_seed']} geo={r['geo_G_seed']} "
                         f"({r['min_G_seed']}-{r['max_G_seed']}) acc_fail={r['any_acc_fail']} foreign={r['max_foreign_mib']}\n")
        out.append(r)
        json.dump(out, open("/tmp/barefwd_out.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
