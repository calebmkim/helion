"""GATE A — INDEPENDENT reproduction (authored fresh by the adversarial reviewer; NOT a
clone of mm_epilogue_bench.py). The DRIVER runs this foreground-serial on one idle GPU;
the analytical skeptics never touch the GPU.

Purpose: positively confirm, from primitives, that the Stage-3
``TritonMatmulReductionEpilogueHeuristic`` SEED (a) is the config the heuristic actually
emits (read via ``compiler_seed_configs`` — not hand-typed), (b) beats the N-blind Helion
default ~1.4-1.9x, and (c) beats a torch.compile(reference) baseline, on
matmul_rms_norm bf16 at (131072,256,256) and (131072,256,512).

Independence from the worker harness:
  - re-implements the matmul_rms_norm kernel + its fp32 reference INLINE here (does not
    import matmul_epilogue_kernels), so a bug in the corpus cannot launder a number;
  - times with a fresh cold-L2 median-of-9 do_bench written here;
  - re-derives the accuracy gate (rel-to-output-RMS, bf16 floor 0.07) inline and gates
    BEFORE timing; an acc-fail arm reports NaN, never a fabricated speedup;
  - same input tensors across all arms; forward-only, no autograd wrapper; dynamo reset
    per shape; empty_cache between shapes.

RUN (driver, from /tmp, one idle GPU):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-3stage \
    /home/calebkim/.conda/envs/helion/bin/python \
    /home/calebkim/helion-new-heuristics/helion-3stage/_lab/stage3_epilogue/gate_A_repro.py

PASS criteria printed at the end:
  seed config == the heuristic's emitted seed (must be True),
  seed_vs_default in ~[1.3, 2.1], seed_vs_tc_default > 1.0  (per shape, acc-gated).
"""

from __future__ import annotations

import statistics

import torch

import helion
import helion.language as hl
from helion._compiler.autotuner_heuristics import compiler_seed_configs

DEV = "cuda"
N_RUNS = 9
BF16_REL_RMS_TOL = 0.07
SHAPES = [(131072, 256, 256), (131072, 256, 512)]


@helion.kernel(static_shapes=True)
def matmul_rms_norm(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))
    assert k == k2
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        eps = 1e-6
        ms = (acc * acc).sum(dim=-1, keepdim=True) / n
        normed = acc * torch.rsqrt(ms + eps)
        out[tile_m, :] = normed * weight[:].to(torch.float32)
    return out


def ref_rms_norm(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mm = torch.matmul(x, y).to(torch.float32)
    ms = mm.pow(2).mean(dim=-1, keepdim=True)
    normed = mm * torch.rsqrt(ms + 1e-6)
    return (normed * weight.to(torch.float32)).to(
        torch.promote_types(x.dtype, y.dtype)
    )


def cold_l2_median_us(fn) -> float:
    """median-of-9 of do_bench's median (this Triton build flushes L2 between reps)."""
    from triton.testing import do_bench

    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2] * 1e3  # ms -> us


def acc_ok(out: torch.Tensor, ref_out: torch.Tensor) -> tuple[bool, float]:
    o, r = out.float(), ref_out.float()
    if not torch.isfinite(o).all():
        return False, float("inf")
    d = (o - r).abs().max().item()
    rms = r.pow(2).mean().sqrt().item()
    rel = d / rms if rms > 1e-9 else d
    return rel <= BF16_REL_RMS_TOL, rel


def main() -> None:
    print(f"helion: {helion.__file__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    for (m, k, n) in SHAPES:
        torch._dynamo.reset()
        x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
        y = torch.randn(k, n, device=DEV, dtype=torch.bfloat16)
        w = torch.randn(n, device=DEV, dtype=torch.bfloat16)
        args = (x, y, w)
        ref_out = ref_rms_norm(*args)

        bound = matmul_rms_norm.bind(args)
        default_cfg = bound.config_spec.default_config()
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        assert seeds, "heuristic emitted no seed (eligibility regressed)"
        seed_cfg = seeds[0]

        def mk(cfg):
            return helion.kernel(matmul_rms_norm.fn, config=cfg, static_shapes=True)

        d_kernel = mk(default_cfg)
        s_kernel = mk(seed_cfg)
        d_out = d_kernel(*args)
        s_out = s_kernel(*args)
        torch.cuda.synchronize()
        d_ok, d_rel = acc_ok(d_out, ref_out)
        s_ok, s_rel = acc_ok(s_out, ref_out)

        torch._dynamo.reset()
        tc = torch.compile(lambda: ref_rms_norm(*args))
        tc_out = tc()
        torch.cuda.synchronize()
        tc_ok, tc_rel = acc_ok(tc_out, ref_out)

        td = cold_l2_median_us(lambda: d_kernel(*args)) if d_ok else float("nan")
        ts = cold_l2_median_us(lambda: s_kernel(*args)) if s_ok else float("nan")
        ttc = cold_l2_median_us(tc) if tc_ok else float("nan")

        seed_dict = dict(seed_cfg)
        print(f"=== matmul_rms_norm ({m},{k},{n}) bf16 ===")
        print(f"  emitted SEED config  : {seed_dict}")
        print(f"  default config       : {dict(default_cfg)}")
        print(f"  acc rel-to-rms  default={d_rel:.4f}({d_ok}) "
              f"seed={s_rel:.4f}({s_ok}) tc={tc_rel:.4f}({tc_ok})")
        print(f"  lat us  default={td:.2f}  seed={ts:.2f}  tc_default={ttc:.2f}")
        if ts == ts and td == td:
            print(f"  seed_vs_default = {td / ts:.3f}")
        if ts == ts and ttc == ttc:
            print(f"  seed_vs_tc_default = {ttc / ts:.3f}")
        print()

        del x, y, w, args, ref_out, d_kernel, s_kernel
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
