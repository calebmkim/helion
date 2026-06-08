"""Correctness + seed-used probe for kl_div and jsd (Band-B T2 loss kernels).

Loss kernels return a SCALAR; justify tol by comparing the SEED output to the
liger-style torch baseline (kl_div: torch.nn.KLDivLoss batchmean; jsd:
TorchJSDBaseline) AND to the bare default config (so we separate seed error from
algorithmic fp32 drift). Confirm the seed routes T2 persistent (R_BLOCK>=V, inner
loop runs once) and M_BLOCK at floor (Band-B numel constraint).
"""

from __future__ import annotations

import re
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.kl_div import kl_div_forward, HelionKLDivLoss  # noqa: E402
from examples.jsd import jsd_forward, HelionJSD, TorchJSDBaseline  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

DEV = "cuda"


def t2_persistent_used(code, v):
    consts = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"(_BLOCK_SIZE_\d+)\s*=\s*tl\.constexpr\((\d+)\)", code)}
    steps = []
    for m in re.finditer(r"tl\.range\(0,\s*(\d+),\s*(_BLOCK_SIZE_\d+)\)", code):
        if int(m.group(1)) == v:
            s = consts.get(m.group(2))
            if s is not None:
                steps.append(s)
    return (max(steps) >= v) if steps else None


def kl_inputs(BT, V):
    yp = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(dim=-1)
    yt = torch.randn(BT, V, device=DEV, dtype=torch.float32).softmax(dim=-1)
    return yp, yt


def jsd_inputs(BT, V):
    lq = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(dim=-1)
    lp = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(dim=-1)
    return lq, lp


def probe_kl(BT, V):
    print(f"\n--- kl_div ({BT},{V}) ---")
    yp, yt = kl_inputs(BT, V)
    args = (yp, yt, False, "batchmean", 1e-10)
    bound0 = kl_div_forward.bind(args)
    seed = dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])
    print(f"  seed={seed}")
    k = helion.kernel(kl_div_forward.fn, configs=[helion.Config(**seed)])
    b = k.bind(args); b.ensure_config_exists(args)
    code = b.to_triton_code(helion.Config(**dict(b._config)))
    used = t2_persistent_used(code, V)
    out_seed = b(*args)
    # default
    cfgd = dict(bound0.config_spec.default_config())
    kd = helion.kernel(kl_div_forward.fn, configs=[helion.Config(**cfgd)])
    bd = kd.bind(args); bd.ensure_config_exists(args)
    out_def = bd(*args)
    # torch baseline
    ref = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(DEV)(yp, yt)
    print(f"  persistent_used={used}  seed_out={float(out_seed):.6f} "
          f"def_out={float(out_def):.6f} ref={float(ref):.6f}")
    print(f"  |seed-ref|={abs(float(out_seed)-float(ref)):.3e}  "
          f"|def-ref|={abs(float(out_def)-float(ref)):.3e}  "
          f"|seed-def|={abs(float(out_seed)-float(out_def)):.3e}")
    rel = abs(float(out_seed)-float(ref))/(abs(float(ref))+1e-12)
    print(f"  seed rel-err vs torch={rel:.3e}")


def probe_jsd(BT, V):
    print(f"\n--- jsd ({BT},{V}) ---")
    lq, lp = jsd_inputs(BT, V)
    args = (lq, lp, None, 0.5, -100)
    bound0 = jsd_forward.bind(args)
    seed = dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])
    print(f"  seed={seed}")
    k = helion.kernel(jsd_forward.fn, configs=[helion.Config(**seed)])
    b = k.bind(args); b.ensure_config_exists(args)
    code = b.to_triton_code(helion.Config(**dict(b._config)))
    used = t2_persistent_used(code, V)
    loss_seed, dX_seed = b(*args)
    cfgd = dict(bound0.config_spec.default_config())
    kd = helion.kernel(jsd_forward.fn, configs=[helion.Config(**cfgd)])
    bd = kd.bind(args); bd.ensure_config_exists(args)
    loss_def, dX_def = bd(*args)
    ref = TorchJSDBaseline(beta=0.5, ignore_index=-100)(lq, lp)
    print(f"  persistent_used={used}  loss_seed={float(loss_seed):.6f} "
          f"loss_def={float(loss_def):.6f} ref={float(ref):.6f}")
    print(f"  |loss_seed-ref|={abs(float(loss_seed)-float(ref)):.3e}  "
          f"|loss_def-ref|={abs(float(loss_def)-float(ref)):.3e}")
    rel = abs(float(loss_seed)-float(ref))/(abs(float(ref))+1e-12)
    print(f"  seed loss rel-err vs torch={rel:.3e}  "
          f"dX max|seed-def|={float((dX_seed-dX_def).abs().max()):.3e}")


def main():
    print(f"helion={helion.__file__}")
    torch.manual_seed(0)
    for V in [4096, 65536, 131072]:
        probe_kl(4096, V)
    for V in [4096, 65536, 131072]:
        probe_jsd(8192, V)


if __name__ == "__main__":
    main()
