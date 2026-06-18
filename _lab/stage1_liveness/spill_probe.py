"""Spill probe: compile a standard-track reduction with persistent vs looped reduction_loops
and read n_regs/n_spills off the compiled Triton kernel(s). Tests:
  (1) flj persistent spills hard, looped doesn't (the Part-B mechanism / Gate-F evidence).
  (2) does cross_entropy fp32 wide-V ALSO spill when persistent? -> if yes, ff=peak_live
      flipping it to looped is a win/neutral (the secondary gap), not a regression.
  (3) rms_norm stays cheap either way.
Compile-only metadata read (no timing). NO autotune.
"""

from __future__ import annotations

import gc
import os
import sys

import torch
import triton

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

_WT = "/home/calebkim/helion-new-heuristics/helion-3stage"
assert os.path.realpath(helion.__file__).startswith(_WT), helion.__file__
sys.path.insert(0, os.path.join(_WT, "_lab", "bench"))
sys.path.insert(0, os.path.join(_WT, "_lab", "transfer"))
import bare_fwd_dtype as BF  # noqa: E402


def _spills_for(kname_prefix):
    found = []
    for o in gc.get_objects():
        if isinstance(o, triton.runtime.jit.JITFunction) and o.__name__.startswith(
            kname_prefix
        ):
            cache = getattr(o, "cache", {})
            for ck in cache.values() if isinstance(cache, dict) else []:
                for v in ck.values() if isinstance(ck, dict) else [ck]:
                    nr = getattr(v, "n_regs", None)
                    ns = getattr(v, "n_spills", None)
                    if nr is not None:
                        found.append((nr, ns))
    return sorted(set(found))


def probe(label, fn, args, base_cfg, reduction_loops, ref, extract):
    cfg = dict(base_cfg)
    cfg["reduction_loops"] = reduction_loops
    gc.collect()
    before = set(_spills_for("_helion_"))
    k = helion.kernel(fn.fn, config=helion.Config(**cfg), static_shapes=True)
    out = extract(k(*args))
    acc = bool(torch.allclose(out.float(), ref.float(), rtol=3e-2, atol=3e-2))
    after = _spills_for("_helion_")
    new = [x for x in after if x not in before]
    worst = max((ns or 0 for _, ns in new), default=-1)
    maxreg = max((nr for nr, _ in new), default=-1)
    print(f"  {label:36} rl={str(reduction_loops):9} acc={acc} "
          f"max_regs={maxreg} max_spills={worst}  variants={new}")


def build_flj(m, v, dt):
    from examples.fused_linear_jsd import jsd_kernel

    sl = torch.randn(m, v, device="cuda", dtype=dt)
    tl = torch.randn(m, v, device="cuda", dtype=dt)

    def ref():
        ss, ts = sl.float(), tl.float()
        sp, tp = torch.softmax(ss, -1), torch.softmax(ts, -1)
        slp, tlp = torch.log_softmax(ss, -1), torch.log_softmax(ts, -1)
        mm = 0.5 * sp + 0.5 * tp
        logm = torch.log(mm)
        return (0.5 * (sp * (slp - logm)).sum(-1) + 0.5 * (tp * (tlp - logm)).sum(-1))

    return jsd_kernel, (0.5, -100, 1.0, sl, tl), ref(), (lambda o: o[0])


def main():
    dt = torch.float32
    # flj (standard track) narrow-V: persistent vs looped
    print("=== fused_linear_jsd fp32 (4096,50257) ===")
    fn, args, ref, ex = build_flj(4096, 50257, dt)
    seed = compiler_seed_configs(fn.bind(args).env, fn.bind(args).host_function.device_ir)[0]
    base = dict(seed.config)
    for rl in ([None], [16384], [2048]):
        probe("flj", fn, args, base, rl, ref, ex)

    # cross_entropy fp32 wide-V: does persistent spill?
    print("=== cross_entropy fp32 (8192,50257) ===")
    fn, build, _ = BF.KERNELS["cross_entropy"]
    args, ref, ex = build(8192, 50257, dt)
    seed = compiler_seed_configs(fn.bind(args).env, fn.bind(args).host_function.device_ir)[0]
    base = dict(seed.config)
    print(f"  (seed reduction_loops={base.get('reduction_loops')})")
    for rl in ([None], [16384]):
        probe("cross_entropy", fn, args, base, rl, ref, ex)

    # rms_norm fp32 mid-N: cheap either way
    print("=== rms_norm fp32 (8192,8192) ===")
    fn, build, _ = BF.KERNELS["rms_norm"]
    args, ref, ex = build(8192, 8192, dt)
    seed = compiler_seed_configs(fn.bind(args).env, fn.bind(args).host_function.device_ir)[0]
    base = dict(seed.config)
    for rl in ([None], [16384]):
        probe("rms_norm", fn, args, base, rl, ref, ex)


if __name__ == "__main__":
    main()
