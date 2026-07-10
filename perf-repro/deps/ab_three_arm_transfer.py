"""Transfer-test harness for the merged reduction seed heuristic (PR #2762).

Two modes:
  verify  : correctness gate only (no timing). For each (kernel, shape, dtype) it checks the
            seed config compiles and matches a pure-torch reference, and reports whether the
            seed FIRED (compiler_seed_configs non-empty) and what config it emitted.
  bench   : single-process 3-arm timing on the real-world shape curriculum:
              helion_default (heuristics off -> base default_config())
              helion_seeded  (the PR reduction seed)
              torch_compile  (torch.compile DEFAULT mode, not max-autotune)
            reports seeded_vs_default (the floor that matters: defaults are bad) and
            seeded_vs_tc (the goal), median-of-9 do_bench, same inputs across arms.

Run (from /tmp, against the MERGED heuristic worktree):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-pr-merge \
    /home/calebkim/.conda/envs/helion/bin/python \
    /home/calebkim/helion-new-heuristics/helion-pr-merge/_lab/transfer/ab_three_arm_transfer.py verify
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

# B200/sm100 ONLY: the merged reduction seed gates on sm90, so on a B200 it falls back
# (standard track -> conservative _narrow_seed; user-tiled track -> declines entirely).
# Set HELION_TRANSFER_FORCE_SEED=1 to force the sm90 seed to fire on non-sm90 hardware, to
# MEASURE whether the H100-tuned constants (240KiB SMEM cap, 132-SM occupancy limits, the
# warp ramp) transfer to B200. EXPERIMENT-ONLY: those constants are H100-specific, so a
# suboptimal B200 config is itself the finding (it motivates a B200 reduction heuristic).
if os.environ.get("HELION_TRANSFER_FORCE_SEED") == "1":
    import helion._compiler.autotuner_heuristics.triton as _tri

    _tri.matches_hardware = lambda *a, **k: True  # noqa: ARG005

# import the sibling kernel + shape modules regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shapes_transfer as SH  # noqa: E402
import transfer_kernels as TK  # noqa: E402

DEV = "cuda"
N_RUNS = 9
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _med(fn) -> float:
    from triton.testing import do_bench

    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2]


def _seed_cfg(bound):
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    return seeds[0] if seeds else None


def _bind(kernel_fn, args, cfg):
    if cfg is None:
        return None
    k = helion.kernel(kernel_fn.fn, config=cfg, static_shapes=True)
    return lambda: k(*args)


# ---- default accuracy checkers ------------------------------------------------
def _close(rtol, atol):
    def chk(out, ref):
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    return chk


def _chk_quant(out, ref):
    (q, s), (qr, sr) = out, ref
    torch.testing.assert_close(s, sr, rtol=1e-4, atol=1e-6)
    diff = (q.float() - qr.float()).abs().max().item()
    assert diff <= 1.0, f"int8 quant differs by {diff} (> 1 ULP)"


def _chk_first(rtol, atol):
    # compare only output[0] (the loss) for multi-output loss kernels
    def chk(out, ref):
        o = out[0] if isinstance(out, (tuple, list)) else out
        r = ref[0] if isinstance(ref, (tuple, list)) else ref
        torch.testing.assert_close(o.float(), r.float(), rtol=rtol, atol=atol)
    return chk


# ---- per-kernel adapters: build(shape, dtype) -> (kernel_fn, args, ref_callable, check)
def _make(kernel):
    def randn(*shp, dtype):
        return torch.randn(*shp, device=DEV, dtype=dtype)

    if kernel == "fused_add_rmsnorm":
        def build(shape, dt):
            m, n = shape
            x, res, w = randn(m, n, dtype=dt), randn(m, n, dtype=dt), randn(n, dtype=dt)
            args = (x, res, w, 1e-6)
            return (TK.fused_add_rmsnorm_fwd, args,
                    lambda: TK.ref_fused_add_rmsnorm(*args), _close(2e-2, 2e-2))
        return build
    if kernel == "fused_add_layernorm":
        def build(shape, dt):
            m, n = shape
            x, res = randn(m, n, dtype=dt), randn(m, n, dtype=dt)
            w, b = randn(n, dtype=dt), randn(n, dtype=dt)
            args = (x, res, w, b, 1e-5)
            return (TK.fused_add_layernorm_fwd, args,
                    lambda: TK.ref_fused_add_layernorm(*args), _close(2e-2, 2e-2))
        return build
    if kernel == "gated_rmsnorm":
        def build(shape, dt):
            m, n = shape
            x, g, w = randn(m, n, dtype=dt), randn(m, n, dtype=dt), randn(n, dtype=dt)
            args = (x, g, w, 1e-6)
            return (TK.gated_rmsnorm_fwd, args,
                    lambda: TK.ref_gated_rmsnorm(*args), _close(2e-2, 2e-2))
        return build
    if kernel == "scaled_masked_softmax":
        def build(shape, dt):
            m, n = shape
            x = randn(m, n, dtype=dt)
            # realistic additive mask: ~30% positions masked to -inf-ish
            mask = torch.where(
                torch.rand(m, n, device=DEV) < 0.3,
                torch.full((m, n), -1e4, device=DEV, dtype=dt),
                torch.zeros(m, n, device=DEV, dtype=dt),
            )
            args = (x, mask, 0.125)
            return (TK.scaled_masked_softmax_fwd, args,
                    lambda: TK.ref_scaled_masked_softmax(*args), _close(2e-2, 2e-3))
        return build
    if kernel == "cross_entropy_ls_zloss":
        def build(shape, dt):
            m, v = shape
            logits = randn(m, v, dtype=dt)
            labels = torch.randint(0, v, (m,), device=DEV, dtype=torch.int64)
            args = (logits, labels, 0.1, 1e-4)
            return (TK.cross_entropy_ls_zloss_fwd, args,
                    lambda: TK.ref_cross_entropy_ls_zloss(*args), _close(2e-3, 2e-3))
        return build
    if kernel == "dynamic_quant":
        def build(shape, dt):
            m, n = shape
            x = randn(m, n, dtype=dt)
            args = (x, 127.0)
            return (TK.dynamic_quant_fwd, args,
                    lambda: TK.ref_dynamic_quant(*args), _chk_quant)
        return build
    if kernel == "fused_linear_jsd":
        from examples.fused_linear_jsd import jsd_kernel

        def build(shape, dt):
            m, v = shape
            sl, tl = randn(m, v, dtype=dt), randn(m, v, dtype=dt)
            beta, ignore_index, temp = 0.5, -100, 1.0
            args = (beta, ignore_index, temp, sl, tl)

            def ref():
                ss, ts = sl.float() / temp, tl.float() / temp
                sp, tp = torch.softmax(ss, -1), torch.softmax(ts, -1)
                slp, tlp = torch.log_softmax(ss, -1), torch.log_softmax(ts, -1)
                mm = (1 - beta) * sp + beta * tp
                logm = torch.log(mm)
                skl = (sp * (slp - logm)).sum(-1)
                tkl = (tp * (tlp - logm)).sum(-1)
                return (1 - beta) * skl + beta * tkl  # [M] loss

            return (jsd_kernel, args, ref, _chk_first(3e-2, 3e-2))
        return build
    if kernel == "grpo":
        from examples.grpo_loss import (
            extract_selected_logits_pytorch,
            grpo_loss_forward,
            torch_grpo_loss,
        )

        def build(shape, dt):
            b, ll, v = shape
            temp, beta, eps_lo, eps_hi = 0.9, 0.2, 0.2, 0.4
            logits = randn(b, ll + 1, v, dtype=dt)
            cids = torch.randint(0, v - 1, (b, ll), device=DEV, dtype=torch.int64)
            cmask = torch.ones(b, ll, device=DEV, dtype=torch.float32)
            ref_logp = randn(b, ll, dtype=torch.float32)
            old_logp = randn(b, ll, dtype=torch.float32)
            adv = randn(b, dtype=torch.float32)
            sel = extract_selected_logits_pytorch(logits[:, :-1, :], cids, temp)
            args = (logits, sel, old_logp, ref_logp, adv, cmask,
                    temp, beta, eps_lo, eps_hi)

            def ref():
                lr = logits.float()
                loss, kl, _ = torch_grpo_loss(
                    lr, old_logp, ref_logp, cids, adv, cmask,
                    temp, beta, eps_lo, eps_hi)
                return loss  # [B, L]

            return (grpo_loss_forward, args, ref, _chk_first(3e-2, 3e-2))
        return build
    raise SystemExit(f"no adapter for {kernel}")


ALL = list(SH.SHAPES.keys())


def verify(kernels, dtypes):
    print(f"helion: {helion.__file__}")
    fails = 0
    for kernel in kernels:
        build = _make(kernel)
        for dt_name in dtypes:
            dt = _DTYPES[dt_name]
            for shape in SH.SMOKE[kernel]:
                kfn, args, ref, chk = build(shape, dt)
                try:
                    bound = kfn.bind(args)
                    seed = _seed_cfg(bound)
                    fired = seed is not None
                    call = _bind(kfn, args, seed) if fired else None
                    ref_out = ref()
                    status = "PASS"
                    if not fired:
                        status = "SEED-DID-NOT-FIRE(under-fire!)"
                        fails += 1
                    else:
                        out = call()
                        chk(out, ref_out)
                    warps = dict(seed.config).get("num_warps") if seed else None
                    rl = dict(seed.config).get("reduction_loops") if seed else None
                    print(f"  {status:14} {kernel:24} {str(shape):18} {dt_name:5} "
                          f"warps={warps} reduction_loops={rl}")
                except Exception as e:  # noqa: BLE001
                    fails += 1
                    print(f"  FAIL           {kernel:24} {str(shape):18} {dt_name:5} "
                          f":: {type(e).__name__}: {str(e)[:120]}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    return fails


def bench(kernel, dtypes):
    build = _make(kernel)
    out_rows = []
    for dt_name in dtypes:
        dt = _DTYPES[dt_name]
        for shape in SH.SHAPES[kernel]:
            # Footgun #2/#11: reset dynamo per shape so the tc arm recompiles for THIS
            # shape (else it caches multiple shapes -> slower dynamic-shapes mode = unfair).
            torch._dynamo.reset()
            kfn, args, ref, chk = build(shape, dt)
            bound = kfn.bind(args)
            default_cfg = bound.config_spec.default_config()
            seed_cfg = _seed_cfg(bound)
            d_call = _bind(kfn, args, default_cfg)
            s_call = _bind(kfn, args, seed_cfg)
            ref_out = ref()
            # Footgun #6: accuracy gate BEFORE timing; an acc-FAIL row is excluded from the
            # geomean (never let a wrong-output latency into the headline).
            acc = {}
            for nm, c in (("default", d_call), ("seeded", s_call)):
                if c is None:
                    acc[nm] = "no-config"
                    continue
                try:
                    chk(c(), ref_out)
                    acc[nm] = True
                except Exception as e:  # noqa: BLE001
                    acc[nm] = f"FAIL:{str(e)[:60]}"
            tc = torch.compile(ref)  # default inductor mode (NOT max-autotune); footgun #3/#10
            tc()
            td = _med(d_call) if d_call else float("nan")
            ts = _med(s_call) if s_call else float("nan")
            tt = _med(tc)
            row = {
                "kernel": kernel, "shape": list(shape), "dtype": dt_name,
                "lat_us": {"default": round(td * 1e3, 2), "seeded": round(ts * 1e3, 2),
                           "tc": round(tt * 1e3, 2)},
                "seeded_vs_default": round(td / ts, 4) if ts else None,
                "seeded_vs_tc": round(tt / ts, 4) if ts else None,
                "acc": acc,
                "seed_fired": seed_cfg is not None,
                "seed_config": dict(seed_cfg.config) if seed_cfg else None,
                # sub-25us rows are noise-floor unreliable (median-of-9 helps; re-run if needed)
                "noisy_sub25us": min(td, ts, tt) * 1e3 < 25.0,
            }
            out_rows.append(row)
            print("ROW " + json.dumps(row), file=sys.stderr)
            # Footgun #8: free big tensors between (multi-GB) shapes.
            del kfn, args, ref, ref_out, d_call, s_call, tc
            torch.cuda.empty_cache()
    # geomean over accuracy-passing seeded rows only (footgun #6a).
    ok = [r for r in out_rows if r["acc"].get("seeded") is True]
    sv = [r["seeded_vs_default"] for r in ok if r["seeded_vs_default"]]
    st = [r["seeded_vs_tc"] for r in ok if r["seeded_vs_tc"]]
    print(json.dumps({
        "kernel": kernel, "rows": out_rows,
        "n_acc_pass": len(ok), "n_total": len(out_rows),
        "geomean_seeded_vs_default": round(statistics.geometric_mean(sv), 4) if sv else None,
        "geomean_seeded_vs_tc": round(statistics.geometric_mean(st), 4) if st else None,
    }))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "bench"])
    ap.add_argument("kernel", nargs="?", default="all")
    ap.add_argument("--dtypes", default="bf16,fp32")
    a = ap.parse_args()
    dtypes = a.dtypes.split(",")
    kernels = ALL if a.kernel == "all" else [a.kernel]
    if a.mode == "verify":
        raise SystemExit(1 if verify(kernels, dtypes) else 0)
    else:
        for k in kernels:
            bench(k, dtypes)


if __name__ == "__main__":
    main()
