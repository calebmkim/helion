"""Gate evidence (compile-only, NO GPU timing) for the pointwise seed heuristic.

Run in BOTH worktrees to diff (Gate R no-regression on negative recognizers):
  AFTER  = helion-pointwise (this change)
  BEFORE = helion-3stage    (base 9bf7b1f9, no pointwise heuristic)

Emits machine-readable EVID lines:
  - facts + seed config for negative recognizers (rms_norm/matmul/softmax/layer_norm)
  - Gate D divergence: per-token fp8 quant (amax over N -> REDUCTION -> pointwise ABSENT)
  - Gate D broadcast : bias_gelu N-D (bias[N] broadcast -> pointwise PRESENT, traffic-2)
"""

from __future__ import annotations

import json
import sys

import torch

import helion
import helion.language as hl

print(f"helion: {helion.__file__}", file=sys.stderr)
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

DEV = "cuda"


@helion.kernel()
def per_token_fp8_quant(x: torch.Tensor):
    """Per-token fp8 activation quant: scale = amax(row)/448 (a REDUCTION over N).
    Looks like a pointwise quant but the amax makes a ReductionFact fire — the Gate D
    divergence kernel: PointwiseElementwiseFact MUST be absent (disjointness rule)."""
    m, n = x.shape
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty([m], device=x.device, dtype=torch.float32)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        amax = torch.amax(torch.abs(row), dim=-1)
        s = amax / 448.0
        scale[tile_m] = s
        out[tile_m, :] = (row / s[:, None]).to(torch.float8_e4m3fn)
    return out, scale


@helion.kernel()
def bias_gelu_nd(x: torch.Tensor, bias: torch.Tensor):
    """GELU(x + bias[N]) on a PRE-PROJECTED x[M,N]; bias[N] BROADCASTS over rows.
    Gate D positive: PointwiseElementwiseFact PRESENT, bytes_per_elem == traffic-2 (the
    broadcast bias is amortized -> excluded; only x-read + out-write count)."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = (x[tm, tn].to(torch.float32) + bias[tn].to(torch.float32))
        g = 0.5 * v * (1.0 + torch.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)))
        out[tm, tn] = g.to(x.dtype)
    return out


def dump(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    pf = getattr(spec, "pointwise_facts", [])  # _lab probe: BEFORE worktree lacks the attr
    rec = {
        "name": name,
        "reduction": len(spec.reduction_facts),
        "matmul": len(spec.matmul_facts),
        "accum": len(spec.accumulator_facts),
        "pointwise": len(pf),
        "heuristics": list(spec.autotuner_heuristics),
        "seed": [dict(s.config) for s in seeds],
    }
    if pf:
        f = pf[0]
        rec["pw_fact"] = {"total_numel": f.total_numel, "bytes_per_elem": f.bytes_per_elem}
    print("EVID " + json.dumps(rec))


def main():
    from examples.matmul import matmul
    from examples.rms_norm import rms_norm_fwd

    x = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(4096, device=DEV, dtype=torch.bfloat16)
    # negative recognizers (no-regression: must be byte-identical BEFORE vs AFTER)
    dump("rms_norm(4096,4096)", rms_norm_fwd, (x, w, 1e-5))
    m1 = torch.randn(1024, 1024, device=DEV, dtype=torch.bfloat16)
    m2 = torch.randn(1024, 1024, device=DEV, dtype=torch.bfloat16)
    dump("matmul(1024,1024)", matmul, (m1, m2))
    try:
        from examples.softmax import softmax
        dump("softmax(4096,4096)", softmax, (x,))
    except Exception as e:  # noqa: BLE001
        print("EVID " + json.dumps({"name": "softmax", "skip": str(e)[:80]}))
    try:
        from examples.layer_norm import layer_norm_fwd
        b = torch.randn(4096, device=DEV, dtype=torch.bfloat16)
        dump("layer_norm(4096,4096)", layer_norm_fwd, (x, [4096], w, b))
    except Exception as e:  # noqa: BLE001
        print("EVID " + json.dumps({"name": "layer_norm", "skip": str(e)[:80]}))
    # Gate D divergence (only meaningful in AFTER worktree; harmless in BEFORE)
    try:
        dump("DIVERGENCE_per_token_fp8_quant(4096,4096)", per_token_fp8_quant, (x,))
    except Exception as e:  # noqa: BLE001
        print("EVID " + json.dumps({"name": "per_token_fp8_quant", "compile_err": str(e)[:120]}))
    try:
        bias = torch.randn(4096, device=DEV, dtype=torch.bfloat16)
        dump("BROADCAST_bias_gelu_nd(4096,4096)", bias_gelu_nd, (x, bias))
    except Exception as e:  # noqa: BLE001
        print("EVID " + json.dumps({"name": "bias_gelu_nd", "compile_err": str(e)[:120]}))


if __name__ == "__main__":
    main()
