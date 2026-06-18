"""Inspect T2 codegen for softmax_two_pass: compare the SEED (R_BLOCK=next_pow2(N),
persistent) vs an explicit small R_BLOCK (looped) so we can see what 'inner loop
runs once' looks like in the generated Triton. Also correctness vs fp32 softmax.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.softmax import softmax_two_pass  # noqa: E402

DEV = "cuda"


def show(name, seed_dict, x):
    print(f"\n========== {name}: {seed_dict} ==========")
    seed = helion.Config(**seed_dict)
    k = helion.kernel(softmax_two_pass.fn, configs=[seed])
    bound = k.bind((x,))
    code = bound.to_triton_code(seed)
    # Count for-loops in the device kernel body (proxy for inner-loop iteration).
    nloops = code.count("for ")
    has_roffset = "roffset" in code
    print(f"  total 'for ' in code: {nloops}; 'roffset' present: {has_roffset}")
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("for ") or "tl.range" in s or "range(" in s and "for" in s:
            print(f"    LOOP: {s}")
    out = bound((x,)) if False else bound(x)
    ref = torch.nn.functional.softmax(x, dim=1)
    maxabs = float((out.to(torch.float32) - ref).abs().max())
    ok = torch.allclose(out.to(torch.float32), ref, rtol=1e-3, atol=1e-4)
    print(f"  correctness vs F.softmax fp32: maxabs={maxabs:.2e} allclose={ok}")
    return code


def main():
    print(f"helion={helion.__file__}")
    torch.manual_seed(0)
    M, N = 4096, 2560  # next_pow2(N)=4096
    x = torch.randn(M, N, device=DEV, dtype=torch.float32)
    # Seed: persistent R_BLOCK = next_pow2(N) = 4096 -> inner loop runs once
    code_p = show("SEED persistent (R_BLOCK=4096)",
                  {"block_sizes": [1, 4096], "num_warps": 8, "num_stages": 1}, x)
    # Explicit small R_BLOCK = 512 -> looped (ceil(2560/512)=5 iters)
    code_l = show("LOOPED (R_BLOCK=512)",
                  {"block_sizes": [1, 512], "num_warps": 8, "num_stages": 1}, x)
    print("\n--- DIFF MARKER ---")
    print(f"persistent code len={len(code_p)} looped code len={len(code_l)}")
    # dump a snippet of each device-loop region
    for tag, code in [("PERSISTENT", code_p), ("LOOPED", code_l)]:
        print(f"\n=== {tag} device-kernel for-loop lines ===")
        for i, ln in enumerate(code.splitlines()):
            if "for " in ln and "tl_offset" not in ln:
                print(f"  {ln.rstrip()}")


if __name__ == "__main__":
    main()
