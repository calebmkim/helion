"""§7 THREE-SHAPE HEADROOM TEST — standalone elementwise SwiGLU, bf16, cold-L2.

Confirms the pointwise niche is real BEFORE building the heuristic:
  default(bs=32)  vs  mini block_size/warps/stages grid (oracle proxy)  vs  tc(max-autotune).
Expect ~5-7x default->oracle and ~1.0x oracle-vs-tc (task §2/§7).

Run (cwd=/tmp):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-pointwise \
    /home/calebkim/.conda/envs/helion/bin/python \
    /home/calebkim/helion-new-heuristics/helion-pointwise/_lab/pointwise/headroom3.py
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys

import torch

import helion

WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
assert helion.__file__.startswith(WT), f"WRONG helion: {helion.__file__}"

from examples.swiglu import _swiglu_fwd  # noqa: E402

DEV = "cuda"
N_RUNS = 3  # calibration only (Step-1 re-measures fresh with med-of-9)
DT = torch.bfloat16
SHAPES = [(16384, 11008), (512, 11008), (4096, 4096)]
# bytes moved per shape: read a + read b + write out = 3 * M*N*itemsize
TRAFFIC = 3


def ref_swiglu(a, b):
    # faithful to the kernel: silu computed in fp32, cast to b.dtype, * b
    return (torch.nn.functional.silu(a.float())).to(b.dtype) * b


def med(fn):
    from triton.testing import do_bench

    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2]


def force(cfg, a, b):
    k = helion.kernel(_swiglu_fwd.fn, config=cfg, static_shapes=True)
    return lambda: k(a, b)


def acc_ok(call, ref_out):
    try:
        torch.testing.assert_close(call().float(), ref_out.float(), rtol=0.05, atol=0.005)
        return True
    except Exception as e:  # noqa: BLE001
        return f"FAIL:{str(e)[:80]}"


def gbps(lat_ms, m, n):
    return TRAFFIC * m * n * DT.itemsize / (lat_ms * 1e-3) / 1e12  # TB/s


def run():
    print(f"helion: {helion.__file__}")
    BS = [256, 1024, 2048, 4096, 8192, 16384]
    WARPS = [4, 8, 16]
    STAGES = [1, 2]
    results = []
    for (m, n) in SHAPES:
        torch._dynamo.reset()
        a = torch.randn(m, n, device=DEV, dtype=DT)
        b = torch.randn(m, n, device=DEV, dtype=DT)
        ref_out = ref_swiglu(a, b)

        # --- default arm (bs=32) ---
        bound = _swiglu_fwd.bind((a, b))
        default_cfg = bound.config_spec.default_config()
        d_call = force(default_cfg, a, b)
        d_acc = acc_ok(d_call, ref_out)
        td = med(d_call)

        # --- oracle proxy: mini grid ---
        best = None
        grid_rows = []
        for bs, w, s in itertools.product(BS, WARPS, STAGES):
            cfg = helion.Config(block_sizes=[bs], num_warps=w, num_stages=s)
            try:
                call = force(cfg, a, b)
                if acc_ok(call, ref_out) is not True:
                    continue
                t = med(call)
            except Exception as e:  # noqa: BLE001
                continue
            grid_rows.append({"bs": bs, "w": w, "s": s, "us": round(t * 1e3, 2)})
            if best is None or t < best[0]:
                best = (t, cfg, bs, w, s)
        to, ocfg, obs, ow, os_ = best

        # --- tc max-autotune-no-cudagraphs ---
        torch._dynamo.reset()
        tc = torch.compile(ref_swiglu, mode="max-autotune-no-cudagraphs")
        tc(a, b)  # warmup/autotune
        tt = med(lambda: tc(a, b))

        row = {
            "shape": [m, n],
            "default_us": round(td * 1e3, 2),
            "default_acc": d_acc,
            "default_gbps": round(gbps(td, m, n), 3),
            "oracle_us": round(to * 1e3, 2),
            "oracle_cfg": {"bs": obs, "w": ow, "s": os_},
            "oracle_gbps": round(gbps(to, m, n), 3),
            "tc_us": round(tt * 1e3, 2),
            "tc_gbps": round(gbps(tt, m, n), 3),
            "default_to_oracle": round(td / to, 3),
            "oracle_vs_tc": round(tt / to, 3),  # >1 = oracle beats tc
            "best5_grid": sorted(grid_rows, key=lambda r: r["us"])[:5],
        }
        results.append(row)
        print("RESULT_ROW " + json.dumps(row), file=sys.stderr)
        del a, b, ref_out
        torch.cuda.empty_cache()

    print("\n===== HEADROOM SUMMARY =====")
    for r in results:
        print(f"  shape={r['shape']}  default={r['default_us']}us(bs=32,{r['default_gbps']}TB/s)  "
              f"oracle={r['oracle_us']}us({r['oracle_cfg']},{r['oracle_gbps']}TB/s)  "
              f"tc={r['tc_us']}us({r['tc_gbps']}TB/s)  "
              f"default->oracle={r['default_to_oracle']}x  oracle_vs_tc={r['oracle_vs_tc']}x")
    print("RESULT_JSON " + json.dumps(results))
    return results


if __name__ == "__main__":
    run()
