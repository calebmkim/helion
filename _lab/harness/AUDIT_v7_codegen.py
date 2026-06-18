"""Inspect generated Triton code to confirm the divisor-chunk mechanism:
  - SEED (combine = pow2 divisor): the combine loop has NO mask on the count, and
    Tn lowers to the tile constexpr == divisor (a full chunk), so the per-chunk
    count is the TRUE count.
  - prime N=1543 (combine=1): seed is USED -- a real combine for-loop appears.
"""
from __future__ import annotations

import sys
import re
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.welford import welford  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5


def args(m, n):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (torch.rand(n, device="cuda", generator=g),
            torch.rand(n, device="cuda", generator=g),
            torch.rand(m, n, device="cuda", generator=g), EPS)


def show(m, n):
    a = args(m, n)
    bound = welford.bind(a)
    seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
    k = helion.kernel(welford.fn, configs=[helion.Config(**seed)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    code = b.to_triton_code(helion.Config(**dict(b._config)))
    print(f"\n===== N={n} seed block_sizes={seed['block_sizes']} =====", flush=True)
    forlines = [ln.strip() for ln in code.splitlines() if "for " in ln and "range" in ln]
    print(f"  for-loops: {forlines}", flush=True)
    # find lines mentioning the count / Tn / size
    for ln in code.splitlines():
        s = ln.strip()
        if any(t in s for t in ["chunk.size", "Tn", "_RDIM", "tl.sum", "tl.where", "to_copy", "/ "]) and "def " not in s:
            if re.search(r"(sum|where|num|cnt|count|/ |size)", s):
                pass
    # just print lines that contain a division or sum, the welford count math
    snippets = [ln.rstrip() for ln in code.splitlines()
                if ("sum_x" in ln or "new_cnt" in ln or "acc_cnt" in ln or "= 4" in ln
                    or "Tn" in ln) ]
    for s in snippets[:25]:
        print("    | " + s.strip(), flush=True)
    del a, b, k
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print(f"helion={helion.__file__}", flush=True)
    for (m, n) in [(8192, 1536), (8192, 1543)]:
        show(m, n)
