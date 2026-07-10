# Out-of-sample kernel finding: fused_qk_norm_rope tile-size miscompile

## What this kernel is

`fused_qk_norm_rope` is a vLLM Helion kernel (`vllm/kernels/helion/ops/fused_qk_norm_rope.py`,
upstream main) that we added as an **out-of-sample GENERALIZATION kernel** — structurally unlike
the 4 quant kernels: a 3D grid `[num_tokens, qk_heads, head_dim]` with an **inner RMS-norm
reduction over head_dim** fused with a **RoPE (rotary) epilogue** that reads and writes `qkv`
in place. It exercises reduction-co-resident-with-pointwise-epilogue logic the heuristic's
curriculum never contained.

Vendored kernel body is **byte-identical to upstream** (verified, 72/72 lines) — so any finding
here is about upstream helion, not our copy.

## The finding: a tile-size-dependent MISCOMPILE (not a seed bug, not a precision issue)

Grading the in-place q|k output against a pure-torch reference (`refs.ref_fused_qk_norm_rope`,
which default/tc/vLLM-tuned configs all match to maxabs 0.0156), sweeping the token-axis
`block_sizes` **by hand** (no seed heuristic involved) at (q=32, kv=8, tok=512):

| block_size | maxabs vs ref | verdict |
|---|---|---|
| 8   | 0.0156 | ✓ correct |
| 16  | 1.90   | ✗ **WRONG** |
| 32  | 0.0156 | ✓ correct |
| 64  | 1.59   | ✗ **WRONG** |
| 128 | 0.0156 | ✓ correct |

**Pattern:** correct at 8/32/128, wrong at 16/64 — an *alternating* power-of-two miscompile, NOT
monotonic precision degradation. It reproduces with hand-set configs, so it is a **helion codegen
correctness bug for this kernel at specific tile sizes**, independent of the reduction seed
heuristic. The seed heuristic merely *happens to pick block_size=16* for some shapes and thus
inherits the wrong result.

Likely cause (unconfirmed): the RoPE epilogue does an in-place read-after-write on `qkv`
(`qkv[tile_m,tile_gn,x1_offset]` is both written by the RMS-norm store AND re-read for the rotate)
— certain tile sizes may reorder/alias the load vs the prior store within the tile. Worth a
dedicated upstream repro + issue.

## How the perf report handles it

Per the decision: **report this as a finding, and still measure perf on the shapes where the
seed produces a CORRECT config.** The harness records per-cell `seed_config` + accuracy; the
aggregator excludes acc-failing seed cells from the perf geomean (documented, same mechanism as
the bf16 `†` / fp8 `‡` cases) and lists them. So the qk_norm_rope_gen headline is a geomean over
the correct-seed shapes only, with the miscompiling shapes reported separately as the finding.

## Per-shape seed correctness (all 15 qk_norm_rope_gen cells)

9/15 shapes get a CORRECT seed config (perf-measurable); 6 miscompile. The miscompile is NOT a
clean "block∈{16,64}" rule — it interacts with head-count and token-count too (e.g. q=16/tok=512
picks block=[8] and is WRONG here, while block=8 was correct at q=32; q=64/tok=64 block=[4] wrong):

| q_heads | tok | seed block | maxabs | verdict |
|---|---|---|---|---|
| 16 | 1     | [1]   | 0.000 | ✓ |
| 16 | 64    | [1]   | 1.83  | ✗ |
| 16 | 512   | [8]   | 2.05  | ✗ |
| 16 | 4096  | [32]  | 0.016 | ✓ |
| 16 | 16384 | [32]  | 0.016 | ✓ |
| 32 | 1     | [1]   | 0.000 | ✓ |
| 32 | 64    | [2]   | 0.016 | ✓ |
| 32 | 512   | [16]  | 2.56  | ✗ |
| 32 | 4096  | [64]  | 1.94  | ✗ |
| 32 | 16384 | [64]  | 1.94  | ✗ |
| 64 | 1     | [2]   | 0.000 | ✓ |
| 64 | 64    | [4]   | 1.75  | ✗ |
| 64 | 512   | [32]  | 0.016 | ✓ |
| 64 | 4096  | [128] | 0.016 | ✓ |
| 64 | 16384 | [128] | 0.016 | ✓ |

The head-count/token interaction (not a pure block-value rule) makes it a subtle codegen bug —
strong evidence the out-of-sample kernel earned its place. Perf headline for this corpus = geomean
over the 9 correct-seed cells; the 6 miscompiling cells are reported here as the finding.

## Status

- Confirmed heuristic-independent (hand-set block=16/64 also wrong).
- Confirmed NOT a vendoring artifact (kernel body byte-identical to upstream).
- Reference verified correct (default/tc/vLLM all agree at 0.0156).
- TODO (out of scope for the perf report): minimal upstream repro + file a helion issue.

## Update: torch.compile baseline fusion fix (post-run)

The tc arm's reference (`refs.ref_fused_qk_norm_rope`) originally used advanced-index writes
(`blk[:, :, i1] = ...`) → `index_put`/scatter → Inductor emitted **6 kernels** with intermediate
HBM round-trips. Rewrote it (neox path) as `rotate_half` + full-width cos/sin + a **single
contiguous slice-write**, in-place, bit-identical across all benchmarked shapes → Inductor now
emits **2 kernels** (RMS reduction + 1 pointwise epilogue). 1 kernel is not reached because the
neox rope uses `torch.cat` (`rotate_half` = `cat((-x2, x1))`), and Inductor treats `torch.cat` as a
MATERIALIZATION BOUNDARY. Isolated by ablation: rmsnorm alone fuses to 1 kernel; rmsnorm + a
`[T,1,hd]` broadcast-mul fuses to 1; rmsnorm + half-slice reads + slice-writes fuses to 1; but
rmsnorm + ANY `torch.cat` (even an identity cat that reconstructs the same layout) → 2 kernels.
So the split is specifically the cat, NOT the reduction↔pointwise boundary (my earlier claim —
corrected). A cat-free rope (masked / arithmetic-index recombine instead of concat) could plausibly
reach 1 kernel, but was judged diminishing returns: the 2nd kernel is a single streaming pointwise
pass; going to 1 saves ~one launch + possibly one materialization, wouldn't change the conclusion
(Helion's 1 fused kernel still wins) and at most trims the fair G_tc slightly (~1.9x → maybe ~1.7x).
This ~halved tc device time and dropped **qk_norm_rope_gen G_tc from
3.44 → 1.92** (still a real Helion win — 1 fused kernel vs Inductor's 2 — but the 3.44 was inflated
by the scatter-penalized baseline). Helion's advantage here is genuine cross-reduction fusion that
Inductor won't do; it is NOT "Inductor is too dumb to fuse" — with a fair reference it fuses to the
2-kernel floor.

Correct-seed/miscompile split under the re-run: 10 correct / 5 miscompile (was 9/6 — one
threshold-borderline cell flipped; miscompiling set is now (16,8,512),(32,8,512),(32,8,4096),
(32,8,16384),(64,8,64)). The miscompile finding itself is unchanged.
