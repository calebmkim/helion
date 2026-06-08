"""Single-process 3-arm A/B for the loss kernels kl_div and jsd (test split).

The tritonbench kl_div/jsd operators fix B=8,T=512 and only sweep V (no --shapes),
so they can't drive the test-split (BT,V) shapes through run.py. We bench directly,
in ONE process, on the SAME (BT,V) tensors, with median-of-N do_bench:

  * helion_default : config-free clone -> base default_config() (unseeded)
  * helion_seeded  : the PR's reduction seed (compiler_seed_configs)
  * torch_compile  : torch.compile(default mode) of the reference loss

Accuracy: each helion arm is checked against the torch reference.
G = tc_default_latency / helion_latency ; seed_lift = G_seed / G_default.
A foreign-GPU-process guard flags contamination (other agents sharing the GPU).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys

import torch
from triton.testing import do_bench

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

sys.path.insert(0, "/home/dev/local/helion-reduction-heuristics-run2/_lab/prompts")
import shapes_v3_draft as SH  # noqa: E402

from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward  # noqa: E402

N_RUNS = 9


def _foreign_gpu_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return 0
    me = os.getpid()
    m = 0
    for line in out.splitlines():
        p = [c.strip() for c in line.split(",")]
        if len(p) == 2 and p[0].isdigit() and int(p[0]) != me:
            m = max(m, int(p[1]) if p[1].isdigit() else 0)
    return m


def _med(fn) -> float:
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2]


def _kl_inputs(bt: int, v: int):
    yp = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
    yt = torch.randn(bt, v, device="cuda", dtype=torch.float32).softmax(-1)
    args = (yp, yt)

    def ref():
        # torch.nn.KLDivLoss(reduction="batchmean")(yp, yt)
        return torch.nn.functional.kl_div(yp, yt, reduction="batchmean")

    return args, ref, lambda out: out


def _jsd_inputs(bt: int, v: int):
    xi = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
    tg = torch.randn(bt, v, device="cuda", dtype=torch.float32).log_softmax(-1)
    args = (xi, tg)

    def ref():
        # JSD(beta=0.5): 0.5*KL(m||p)+0.5*KL(m||q), m=0.5(p+q); p=exp(xi),q=exp(tg)
        p, q = xi.exp(), tg.exp()
        m = 0.5 * (p + q)
        kl_pm = (p * (xi - m.log())).sum(-1)
        kl_qm = (q * (tg - m.log())).sum(-1)
        return (0.5 * kl_pm + 0.5 * kl_qm).mean()

    # jsd_forward returns (loss, dX); compare the loss only
    return args, ref, lambda out: out[0]


def bench(kernel: str):
    kfn = {"kl_div": kl_div_forward, "jsd": jsd_forward}[kernel]
    make = {"kl_div": _kl_inputs, "jsd": _jsd_inputs}[kernel]
    shapes = [tuple(s) for s in SH.SHAPES[kernel]["test"]]
    rows = []
    for (bt, v) in shapes:
        args, ref, pick = make(bt, v)
        bound = kfn.bind(args)
        default_cfg = bound.config_spec.default_config()
        seed_cfg = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]
        k_def = helion.kernel(kfn.fn, config=default_cfg, static_shapes=True)
        k_seed = helion.kernel(kfn.fn, config=seed_cfg, static_shapes=True)

        ref_out = ref()
        out_d = pick(k_def(*args))
        out_s = pick(k_seed(*args))
        acc_d = torch.allclose(out_d, ref_out, rtol=2e-2, atol=2e-2)
        acc_s = torch.allclose(out_s, ref_out, rtol=2e-2, atol=2e-2)

        tc = torch.compile(ref)
        tc()

        f0 = _foreign_gpu_mib()
        t_d = _med(lambda: k_def(*args))
        t_s = _med(lambda: k_seed(*args))
        t_tc = _med(lambda: tc())
        foreign = max(f0, _foreign_gpu_mib())

        g_seed, g_default = t_tc / t_s, t_tc / t_d
        rows.append({
            "shape": [bt, v], "lat_seed": round(t_s, 6),
            "lat_default": round(t_d, 6), "lat_tc": round(t_tc, 6),
            "G_seed": round(g_seed, 4), "G_default": round(g_default, 4),
            "seed_lift": round(g_seed / g_default, 4),
            "seed_vs_default_raw": round(t_d / t_s, 4),
            "acc_seed": acc_s, "acc_default": acc_d, "foreign_mib": foreign,
            "seed_rloop": dict(seed_cfg.config).get("reduction_loops"),
        })
        print("ROW " + json.dumps(rows[-1]), file=sys.stderr)

    g = statistics.geometric_mean
    return {
        "kernel": kernel, "split": "test", "rows": rows,
        "geo_G_seed": round(g([r["G_seed"] for r in rows]), 4),
        "geo_G_default": round(g([r["G_default"] for r in rows]), 4),
        "geo_seed_lift": round(g([r["seed_lift"] for r in rows]), 4),
        "geo_seed_vs_default_raw": round(g([r["seed_vs_default_raw"] for r in rows]), 4),
        "any_acc_fail": any(not r["acc_seed"] or not r["acc_default"] for r in rows),
        "max_foreign_mib": max(r["foreign_mib"] for r in rows),
        "contaminated": max(r["foreign_mib"] for r in rows) > 300,
    }


def main() -> None:
    assert helion.__file__.startswith("/home/dev/local/helion-pr-edit"), helion.__file__
    out = [bench(k) for k in (sys.argv[1:] or ["kl_div", "jsd"])]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
