"""Single-process 3-arm A/B for welford (LayerNorm via Welford), test split, fp32.

The tritonbench welford operator hardcodes bf16 inputs and its torch.compile
baseline compile-storms at the wide test shapes, so we bench directly in ONE
process at fp32 (the regime the seed's residency caps were tuned for):

  * helion_default : config-free clone -> base default_config() (unseeded)
  * helion_seeded  : the PR's reduction seed (compiler_seed_configs)
  * torch_compile  : torch.compile(default) of F.layer_norm

G = tc_default_latency / helion_latency ; seed_lift = G_seed / G_default.
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

from examples.welford import welford  # noqa: E402

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


def main() -> None:
    assert helion.__file__.startswith("/home/dev/local/helion-pr-edit"), helion.__file__
    shapes = [tuple(s) for s in SH.SHAPES["welford"]["test"]]
    rows = []
    for (m, n) in shapes:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        args = (w, b, x, 1e-5)
        ref = lambda: torch.nn.functional.layer_norm(x, [n], w, b, 1e-5)  # noqa: E731

        bound = welford.bind(args)
        default_cfg = bound.config_spec.default_config()
        seed_cfg = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]
        k_def = helion.kernel(welford.fn, config=default_cfg, static_shapes=True)
        k_seed = helion.kernel(welford.fn, config=seed_cfg, static_shapes=True)

        ref_out = ref()
        acc_d = torch.allclose(k_def(*args), ref_out, rtol=1e-3, atol=1e-3)
        acc_s = torch.allclose(k_seed(*args), ref_out, rtol=1e-3, atol=1e-3)

        tc = torch.compile(ref)
        tc()

        f0 = _foreign_gpu_mib()
        t_d = _med(lambda: k_def(*args))
        t_s = _med(lambda: k_seed(*args))
        t_tc = _med(lambda: tc())
        foreign = max(f0, _foreign_gpu_mib())

        g_seed, g_default = t_tc / t_s, t_tc / t_d
        rows.append({
            "shape": [m, n], "lat_seed": round(t_s, 6),
            "lat_default": round(t_d, 6), "lat_tc": round(t_tc, 6),
            "G_seed": round(g_seed, 4), "G_default": round(g_default, 4),
            "seed_lift": round(g_seed / g_default, 4),
            "seed_vs_default_raw": round(t_d / t_s, 4),
            "acc_seed": acc_s, "acc_default": acc_d, "foreign_mib": foreign,
        })
        print("ROW " + json.dumps(rows[-1]), file=sys.stderr)

    g = statistics.geometric_mean
    out = {
        "kernel": "welford", "split": "test", "rows": rows,
        "geo_G_seed": round(g([r["G_seed"] for r in rows]), 4),
        "geo_G_default": round(g([r["G_default"] for r in rows]), 4),
        "geo_seed_lift": round(g([r["seed_lift"] for r in rows]), 4),
        "geo_seed_vs_default_raw": round(g([r["seed_vs_default_raw"] for r in rows]), 4),
        "any_acc_fail": any(not r["acc_seed"] or not r["acc_default"] for r in rows),
        "max_foreign_mib": max(r["foreign_mib"] for r in rows),
        "contaminated": max(r["foreign_mib"] for r in rows) > 300,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
