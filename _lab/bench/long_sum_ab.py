"""Single-process 3-arm A/B for long_sum (test split).

long_sum's example kernel ships a BAKED @helion.kernel(config=...), so the run.py
seed/default toggle can't take effect (a user-provided config short-circuits
default_config()). We therefore build the three arms directly, in ONE process, on
the SAME input tensors, with median-of-N do_bench (TritonBench's timer):

  * helion_default : config-free clone -> base default_config() (unseeded)
  * helion_seeded  : the PR's reduction seed (compiler_seed_configs)
  * torch_compile  : torch.compile(default mode) of x.sum(-1)

Reduction is the row-sum (dim=-1), matching the `sum` operator's --reduce-dim 1.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys

import torch
from triton.testing import do_bench


def _foreign_gpu_mib() -> int:
    """Max MiB used by any GPU process that is NOT this process."""
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
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != me:
            m = max(m, int(parts[1]) if parts[1].isdigit() else 0)
    return m

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

sys.path.insert(0, "/home/dev/local/helion-reduction-heuristics-run2/_lab/prompts")
import shapes_v3_draft as SH  # noqa: E402

from examples.long_sum import longsum  # noqa: E402

N_RUNS = 9


def _med(fn) -> float:
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2]


def main() -> None:
    assert helion.__file__.startswith("/home/dev/local/helion-pr-edit"), helion.__file__
    shapes = [tuple(s) for s in SH.SHAPES["long_sum"]["test"]]
    rows = []
    for (m, n) in shapes:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        ref = x.sum(-1)

        # base default (config-free clone) + seed (from the same bind)
        kfree = helion.kernel(longsum.fn, static_shapes=True)
        bound = kfree.bind((x,))
        default_cfg = bound.config_spec.default_config()
        seed_cfg = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]

        k_default = helion.kernel(longsum.fn, config=default_cfg, static_shapes=True)
        k_seeded = helion.kernel(longsum.fn, config=seed_cfg, static_shapes=True)

        out_d, out_s = k_default(x), k_seeded(x)
        acc_d = torch.allclose(out_d, ref, rtol=1e-3, atol=1e-3)
        acc_s = torch.allclose(out_s, ref, rtol=1e-3, atol=1e-3)

        tc = torch.compile(lambda t: t.sum(-1))
        tc(x)  # warm

        f0 = _foreign_gpu_mib()
        t_d = _med(lambda: k_default(x))
        t_s = _med(lambda: k_seeded(x))
        t_tc = _med(lambda: tc(x))
        foreign = max(f0, _foreign_gpu_mib())

        g_seed, g_default = t_tc / t_s, t_tc / t_d
        rows.append({
            "shape": [m, n],
            "lat_seed": round(t_s, 6), "lat_default": round(t_d, 6),
            "lat_tc": round(t_tc, 6),
            "G_seed": round(g_seed, 4), "G_default": round(g_default, 4),
            "seed_lift": round(g_seed / g_default, 4),
            "seed_vs_default_raw": round(t_d / t_s, 4),
            "acc_seed": acc_s, "acc_default": acc_d,
            "foreign_mib": foreign,
            "seed_rloop": dict(seed_cfg.config).get("reduction_loops"),
            "default_rloop": dict(default_cfg.config).get("reduction_loops"),
        })
        print("ROW " + json.dumps(rows[-1]), file=sys.stderr)

    g = statistics.geometric_mean
    summary = {
        "kernel": "long_sum", "split": "test", "rows": rows,
        "geo_G_seed": round(g([r["G_seed"] for r in rows]), 4),
        "geo_G_default": round(g([r["G_default"] for r in rows]), 4),
        "geo_seed_lift": round(g([r["seed_lift"] for r in rows]), 4),
        "geo_seed_vs_default_raw": round(g([r["seed_vs_default_raw"] for r in rows]), 4),
        "any_acc_fail": any(not r["acc_seed"] or not r["acc_default"] for r in rows),
        "max_foreign_mib": max(r["foreign_mib"] for r in rows),
        "contaminated": max(r["foreign_mib"] for r in rows) > 300,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
