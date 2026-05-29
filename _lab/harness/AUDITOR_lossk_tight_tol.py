"""AUDITOR: independent tight-tolerance correctness for kl_div/jsd SEED output.
Asserts the seed output passes a TIGHT rtol on well-conditioned inputs (vs the
torch baseline), proving the worker's tol is not hiding real error.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward, TorchJSDBaseline  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

DEV = "cuda"
TIGHT_RTOL = 1e-4  # far tighter than run_example's 1e-2


def seed_kernel(fn, args):
    bound = fn.bind(args)
    seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
    k = helion.kernel(fn.fn, configs=[helion.Config(**seed)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b, seed


def main():
    torch.manual_seed(0)
    allok = True
    print(f"TIGHT_RTOL={TIGHT_RTOL}")
    for (BT, V) in [(4096, 32000), (4096, 65536), (4096, 128256)]:
        yp = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(-1)
        yt = torch.randn(BT, V, device=DEV, dtype=torch.float32).softmax(-1)
        args = (yp, yt, False, "batchmean", 1e-10)
        b, seed = seed_kernel(kl_div_forward, args)
        out = float(b(*args))
        ref = float(torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(DEV)(yp, yt))
        rel = abs(out - ref) / (abs(ref) + 1e-12)
        ok = rel <= TIGHT_RTOL
        allok &= ok
        print(f"  kl_div ({BT},{V}) R={seed['block_sizes']}: out={out:.6f} ref={ref:.6f} "
              f"rel={rel:.2e} {'OK' if ok else 'FAIL'}")
    for (BT, V) in [(4096, 32000), (8192, 65536), (4096, 128256)]:
        lq = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(-1)
        lp = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(-1)
        args = (lq, lp, None, 0.5, -100)
        b, seed = seed_kernel(jsd_forward, args)
        loss, dX = b(*args)
        out = float(loss)
        ref = float(TorchJSDBaseline(beta=0.5, ignore_index=-100)(lq, lp))
        rel = abs(out - ref) / (abs(ref) + 1e-12)
        ok = rel <= TIGHT_RTOL
        allok &= ok
        print(f"  jsd    ({BT},{V}) R={seed['block_sizes']}: out={out:.6f} ref={ref:.6f} "
              f"rel={rel:.2e} {'OK' if ok else 'FAIL'}")
    print(f"\nALL_TIGHT_TOL_PASS = {allok}")


if __name__ == "__main__":
    main()
