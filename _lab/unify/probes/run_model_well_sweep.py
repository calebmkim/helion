"""Run the MODEL-WELL tripwire (perf_tripwire.tripwire) over the durable newly-fired-region probe
kernels — the cells Gate T certified compile-only (Blocker B closure). One kernel per call ideally,
but this runner does them sequentially in fresh process if invoked per-kernel; here we accept a small
batch since these are distinct kernel objects (not the same-object-rebind footgun).

Reports RED if the compiler DEFAULT beats the SEED past the ~5% noise band on any newly-fired kernel.
foreground-serial GPU (the only GPU step in the totality machinery).
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from perf_tripwire import tripwire  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep)
DEV = "cuda"


# The newly-fired-region kernels (multi-fact, grid-collapse, grid-tile, dtype) re-authored inline so
# this runner is self-contained + durable (the /tmp Gate-T mints are ephemeral).
@helion.kernel(static_shapes=False)
def two_tensor_two_loop(x, y):  # len==1 BROADEN: 2 independent reductions
    m1, _ = x.shape
    m2, _ = y.shape
    o1 = torch.empty([m1], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m2], dtype=torch.float32, device=y.device)
    for tm in hl.tile(m1):
        o1[tm] = x[tm, :].to(torch.float32).sum(-1)
    for tn in hl.tile(m2):
        o2[tn] = y[tn, :].to(torch.float32).sum(-1)
    return o1, o2


@helion.kernel(static_shapes=False)
def backed_col_sum_collapse(x):  # grid-tile-reduction (PARTIAL_GRID floor)
    m, n = x.shape
    mb = hl.register_block_size(m)
    nb = (m + mb - 1) // mb
    blocks = x.new_zeros([nb, n], dtype=torch.float32)
    for cta in hl.tile(m, block_size=mb):
        blocks[cta.id, :] = torch.sum(x[cta, :].to(torch.float32), dim=0)
    return torch.sum(blocks, dim=0)


@helion.kernel(static_shapes=False)
def dual_grid(x):  # joint multi-tunable-grid-axis occupancy
    m, p, _r = x.shape
    out = torch.empty([m, p], dtype=torch.float32, device=x.device)
    for tm, tp in hl.tile([m, p]):
        out[tm, tp] = x[tm, tp, :].to(torch.float32).sum(-1)
    return out


@helion.kernel(static_shapes=False)
def two_rolled_dominant_second(x, y):  # multi-rolled reduction_loops by-slot
    m1, _ = x.shape
    m2, _ = y.shape
    o1 = torch.empty([m1], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m2], dtype=torch.float32, device=y.device)
    for tm in hl.tile(m1):
        o1[tm] = x[tm, :].to(torch.float32).sum(-1)
    for tn in hl.tile(m2):
        o2[tn] = y[tn, :].to(torch.float32).sum(-1)
    return o1, o2


def main():
    print(f"helion={helion.__file__}")
    nvidia = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader").read().strip()
    print(f"GPU mem.used={nvidia}\n")
    cases = [
        ("two_tensor_two_loop@4096,2048+2048,4096", two_tensor_two_loop,
         (torch.randn(4096, 2048, device=DEV), torch.randn(2048, 4096, device=DEV))),
        ("two_tensor_two_loop@8192,4096+4096,8192", two_tensor_two_loop,
         (torch.randn(8192, 4096, device=DEV), torch.randn(4096, 8192, device=DEV))),
        ("backed_col_sum_collapse@8192,2048", backed_col_sum_collapse,
         (torch.randn(8192, 2048, device=DEV),)),
        ("backed_col_sum_collapse@16384,4096", backed_col_sum_collapse,
         (torch.randn(16384, 4096, device=DEV),)),
        ("dual_grid@512,512,16", dual_grid, (torch.randn(512, 512, 16, device=DEV),)),
        ("dual_grid@1024,1024,32", dual_grid, (torch.randn(1024, 1024, 32, device=DEV),)),
        ("two_rolled_dominant_second@256,512+256,262144", two_rolled_dominant_second,
         (torch.randn(256, 512, device=DEV), torch.randn(256, 262144, device=DEV))),
    ]
    n_red = 0
    for label, fn, args in cases:
        try:
            r = tripwire(fn, args, label)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERR ] {label}: {type(e).__name__}: {e}")
            continue
        if not r.get("fired"):
            print(f"  [decl] {label}: {r.get('note')}")
            continue
        red = r["RED_default_beats_seed"]
        if red:
            n_red += 1
        print(f"  [{'RED ' if red else 'ok  '}] {label}: seed={r['seed_us']}us "
              f"default={r['default_us']}us seed/default={r['seed_over_default']} "
              f"match={r['outputs_match']} bs={r['seed_block_sizes']}")
    print(f"\n=== model-well sweep: {n_red} RED (default beats seed >5%) of {len(cases)} cells ===")


if __name__ == "__main__":
    main()
