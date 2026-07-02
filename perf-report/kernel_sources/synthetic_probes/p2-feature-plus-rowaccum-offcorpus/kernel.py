"""STRESS-TEST KERNEL — P2: the per_feature_accumulator WITNESS (off-corpus).

TAXONOMY POINT: a FULL_SLICE feature reduction co-resident with a cross-row (grad-accum)
reduction — the SAME structure as rms/layer/group/instance-norm-backward, but on a kernel
that is NOT one of those 4. (Here: a fused "scale-and-grad" backward-style op — per-row
feature mean over N, AND a per-feature accumulation summed across the M rows.)

WHY IT MATTERS: this is the DEFECT-2 witness (PROMPT.md §6 Q4). The human's claim is that
`per_feature_accumulator` is NOT a real property — it is just "a FULL_SLICE reduction
co-resident with a partial-GRID_TILE reduction." The old `per_feature_accumulator` recognizer
is keyed on the 4 norm-bwd kernels' shape, so it would NOT fire here — but the general
allocator (full-extent bids first; grid-tile claims ~1; m_block occupancy-sized) SHOULD size
this correctly with no override. If the general rule recreates a sane config here, the
recognizer is proven redundant.

IMPLEMENTER'S ASSERTIONS (add when Stage 1/2 exist):
  Tier 1: feature reduction -> FULL_SLICE; cross-row accum -> partial GRID_TILE; the two are
          co-resident (same graph_id). NO `per_feature_accumulator`-style recognizer fires.
  Tier 2: config matches the norm-bwd FAMILY shape (occupancy-sized grid-M, byte-capped inner)
          purely from the allocator. This is the GREEN that proves Defect 2 is absorbed.

NOTE: off-corpus by construction (a generic scale-grad, not a named norm). Keep it a PLAUSIBLE
ML shape (a backward over a scaled activation) so it is a convincing generality witness.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def scale_grad_bwd(
    grad_out: torch.Tensor, x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of  y = x * scale[None, :] * mean(x, -1, keepdim=True).
    grad_x  : needs a per-row FULL_SLICE feature reduction (mean over N).
    grad_scale[N] : needs a cross-row accumulation (sum over the M grid axis).
    Same co-residency shape as the norm-bwds, but NOT a named norm kernel."""
    M, N = x.shape
    grad_x = torch.empty_like(x)
    grad_scale = torch.empty([N], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M):
        xt = x[tile_m, :].to(torch.float32)
        go = grad_out[tile_m, :].to(torch.float32)
        row_mean = (xt).mean(-1)                       # FULL_SLICE feature reduction over N
        sc = scale[None, :].to(torch.float32)
        gx = go * sc * row_mean[:, None]
        grad_x[tile_m, :] = gx.to(x.dtype)
        # cross-row accumulation -> per-feature grad over the M grid axis
        grad_scale[:] += (go * xt * row_mean[:, None]).sum(0)
    return grad_x, grad_scale


def make_args(M: int = 2048, N: int = 4096, dtype=torch.float16, device="cuda"):
    return (
        torch.randn(M, N, device=device, dtype=dtype),
        torch.randn(M, N, device=device, dtype=dtype),
        torch.randn(N, device=device, dtype=dtype),
    )


def main() -> None:
    scale_grad_bwd(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
