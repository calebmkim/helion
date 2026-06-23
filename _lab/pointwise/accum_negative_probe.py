"""Negative-recognizer #3: a CARRIED-STATE kernel (loop-carried tensor accumulator, NO reduction,
NO matmul) must NOT get a PointwiseElementwiseFact (the disjointness accumulator clause, tested in
isolation — rms_norm only tests it transitively alongside a reduction)."""
from __future__ import annotations
import torch
import helion
import helion.language as hl
from helion._compiler.autotuner_heuristics import compiler_seed_configs


@helion.kernel()
def running_carry(x: torch.Tensor) -> torch.Tensor:
    """out[m, n-chunk] = x + carry; carry = last column of the chunk (a loop-carried [tile_m,1]
    tensor across the inner N loop). No sum/max/var -> no reduction; no dot -> no matmul; but the
    inner loop carries a tensor -> AccumulatorFact fires."""
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        carry = hl.zeros([tile_m, 1], dtype=torch.float32)  # loop-carried [tile_m,1] tensor
        for tile_n in hl.tile(n):
            v = x[tile_m, tile_n].to(torch.float32) + carry  # carry broadcasts over the chunk
            out[tile_m, tile_n] = v.to(x.dtype)
            carry = carry + 1.0  # carried update (same shape) — no reduction, no matmul
    return out


def main():
    x = torch.randn(2048, 4096, device="cuda", dtype=torch.bfloat16)
    bound = running_carry.bind((x,))
    spec = bound.config_spec
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    print("running_carry: reduction=%d matmul=%d accum=%d pointwise=%d heuristics=%s" % (
        len(spec.reduction_facts), len(spec.matmul_facts), len(spec.accumulator_facts),
        len(spec.pointwise_facts), spec.autotuner_heuristics))
    assert len(spec.accumulator_facts) >= 1, "expected an AccumulatorFact (carried state)"
    assert len(spec.pointwise_facts) == 0, "DISJOINTNESS VIOLATION: pointwise fired on carried-state!"
    print("PASS: carried-state kernel correctly EXCLUDED from the pointwise track")


if __name__ == "__main__":
    main()
