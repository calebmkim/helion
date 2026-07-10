export const meta = {
  name: 'adversarial-pointwise-hunt',
  description: 'Parallel designer agents write reduction-free pointwise kernels that each break a distinct heuristic assumption, to expose configs where the pointwise seed is >10% worse than the compiler default',
  phases: [
    { title: 'Ideate', detail: 'one designer per assumption-lens writes an adversarial kernel shard' },
  ],
}

const SHARD_DIR = '/home/calebkim/helion-new-heuristics/local/rope_probe/adv/shards'

const PREAMBLE = `You are an ADVERSARIAL kernel designer hunting for coverage gaps in Helion's pointwise
seed heuristic (PR #2866, plus a just-added partial-tiling + register-cap fix). Your job: author
REDUCTION-FREE elementwise Helion kernels + input shapes that make the heuristic emit a config
that is DRASTICALLY WORSE (>10% slower, or fails to compile) than the compiler DEFAULT
(block_size=32). This is the failure class RoPE was: its seed [1,256] was far worse than the
default [1,32] because the byte model was blind to the untiled heads*head_dim slab.

FIRST, read these files to understand the current (patched) heuristic and what fires it:
- /home/calebkim/helion-new-heuristics/helion-pointwise/helion/_compiler/autotuner_heuristics/triton.py
  (class TritonPointwiseSeedHeuristic: get_seed_config, _seed_block_sizes, _clamp_dim, constants
   TILE_BYTES=8192, MIN_WAVES=8, BLOCK_FLOOR=256, REGISTER_BYTES=65536)
- /home/calebkim/helion-new-heuristics/helion-pointwise/helion/_compiler/device_ir.py
  (method build_pointwise_facts — how bytes_per_elem / reg_bytes_per_elem / total_numel are derived,
   the accessed_numel//total_numel slab folding, and the disjointness rule)
- /home/calebkim/helion-new-heuristics/helion-pointwise/helion/autotuner/config_spec.py
  (class PointwiseElementwiseFact docstring)
- /home/calebkim/helion-new-heuristics/helion-pointwise/examples/rope.py (the kernel that fires
  pointwise despite full-extent inner dims — it flows them through reshape/hl.split as elementwise)
- /home/calebkim/helion-new-heuristics/local/rope_probe/adv/harness_one.py (the EXACT harness that
  will score your kernels — read it so your JSON matches its contract)

CRITICAL — your kernel MUST fire the pointwise fact, or it is worthless for this hunt:
- It must be REDUCTION-FREE and matmul-free: NO sum/mean/max/amax/min/argmax/softmax/logsumexp/
  var/norm/matmul/dot/hl.dot, and NO loop-carried accumulator (no hl.zeros + += across an inner
  hl.tile loop). Any of those routes the kernel to another family and NO pointwise fact is built.
- WARNING (verified): a naive full-slice inner dim like \`out[tb,tt,:] = x[tb,tt,:] * s[tb,tt,None]\`
  where only [tb,tt] are tiled actually triggers a REDUCTION+ACCUMULATOR fact (the untiled \`:\` dim
  becomes an rdim) and does NOT fire pointwise. So to fire pointwise, EITHER:
    (a) tile ALL dims of the problem: \`for tm,tn in hl.tile(x.size()): out[tm,tn]=f(x[tm,tn],...)\`
        (this reliably fires pointwise — use it for lenses about warps/dtype/fan-in/transpose/compute), OR
    (b) mimic RoPE EXACTLY: tile the OUTER dims, load the inner full-extent slab and flow it through
        .reshape(...)/.permute(...)/hl.split(...)/hl.join(...) as pure elementwise (use this for the
        partial-tiling lens). Copy rope.py's structure closely.

OUTPUT CONTRACT — write a JSON file (via the Write tool) to:
  {SHARD_PATH}
containing a JSON array of 2-3 kernel entries. Each entry is an object:
  {
    "name": "<the @helion.kernel function name, unique>",
    "targeted_gap": "<the assumption you are breaking, one line>",
    "hypothesis": "<why you expect seed >10% worse than default: what tile the seed picks and why it hurts>",
    "code": "<a python snippet defining @helion.kernel def NAME(...) AND def make_inputs(shape)->tuple>",
    "shapes": [[...], [...]]   // 2-3 shapes; each is passed to make_inputs(shape)
  }
Rules for "code":
- Do NOT include imports (the harness prepends: torch, helion, import helion.language as hl).
- Define exactly: the decorated kernel \`@helion.kernel()\ndef NAME(...):\` and \`def make_inputs(shape):\`
  returning a tuple of CUDA tensors matching the kernel's args. Kernels OUT-OF-PLACE (return new tensors).
- Vary shapes to include at least one LARGE (total_numel > 4M, the heuristic's target-wins regime)
  and where relevant a small/decode one. Use realistic-ish sizes.
- Keep it VALID Helion: tile-index preserves rank; use \`x[tm,tn]\` style; fp32-internal compute via .to(torch.float32) then .to(out.dtype) is fine.
- Prefer 2-3 DISTINCT kernels that each stress the lens a different way.

After writing the file, return the structured summary (schema). Be concrete and diverse — the more
genuinely different structural stressors, the better. Do NOT benchmark anything (no GPU); the harness does that.`

const LENSES = [
  { key: 'warps', brief: `LENS = num_warps starvation. The seed emits ONLY block_sizes and leaves num_warps=4. Design flat all-dims-tiled kernels with WIDE contiguous rows so the seed picks a big inner tile (e.g. [1,1024]/[1,2048]); with only 4 warps a big tile may have low occupancy / poor ILP, so the default's small [32,32] tile (also 4 warps but more programs) could be FASTER. Favor moderate compute so the tile-vs-occupancy tradeoff bites. Try wide-N 2D kernels and 1D-flatten kernels.` },
  { key: 'transpose', brief: `LENS = broken coalescing / strided access. Flat all-dims-tiled, but arrange so the seed's big INNER tile is NOT contiguous in memory: e.g. write to a transposed output view, or read a transposed/strided input, so the innermost tiled dim has a large stride. The seed assumes innermost-block == contiguous; if it isn't, a big inner tile coalesces terribly and may lose to the default's small tile.` },
  { key: 'fanin', brief: `LENS = high fan-in. Many input tensors summed/combined elementwise (e.g. 8-16 inputs), all full-extent → bytes_per_elem becomes large → seed target = TILE_BYTES//bytes_per_elem becomes small (maybe near/below the default 32 after the floor). Probe whether a huge-traffic kernel gets a starved tile that loses to default, or whether the floor mis-sizes it.` },
  { key: 'dtype_small', brief: `LENS = tiny dtypes. Use int8/uint8/bool (itemsize 1) elementwise ops (e.g. bitwise/int add, clamp) so bytes_per_elem is tiny → seed target = TILE_BYTES//1 huge → a very large tile. Probe register/occupancy blowups or block-cap issues from an oversized tile the byte model green-lit.` },
  { key: 'dtype_big', brief: `LENS = wide dtypes. Use float64 or complex64 (itemsize 8) elementwise ops so bytes_per_elem is large and the register width is wide. Probe whether the seed tile is mis-sized (too big for registers, or too small vs default) for wide-element kernels.` },
  { key: 'compute', brief: `LENS = compute-bound (not bandwidth-bound). The heuristic assumes pointwise = bandwidth-bound. Design flat kernels with VERY high arithmetic intensity per byte: many transcendentals (exp, log, tanh, erf, sin, cos, pow, division) chained. For these the bandwidth-saturating big tile may be the wrong objective and (with fixed 4 warps) lose to the default small tile.` },
  { key: 'temporaries', brief: `LENS = register spill from many live temporaries. Flat all-dims-tiled kernel whose body keeps MANY intermediates live simultaneously (a high-degree Horner polynomial with many coefficients, or many independent sub-expressions combined at the end). Even at a "flat" 1024-2048 tile the byte/reg model (which counts per-op slab, not the count of live temporaries) may under-predict register pressure → spill → worse than default.` },
  { key: 'multi_output', brief: `LENS = many outputs / fan-out. A flat kernel that writes MANY output tensors from the same inputs (like RoPE writes q_out AND k_out) — 4-8 outputs. Lots of simultaneously-live results; probe whether the seed tile is too big once you account for all the live output tiles.` },
  { key: 'broadcast', brief: `LENS = broadcast-dominated. Kernels where MOST reads are broadcast operands (accessed_numel < total_numel → EXCLUDED from bytes_per_elem), so the seed thinks traffic is tiny and picks a huge tile, but the broadcast values are re-materialized per element and/or dominate register/compute. e.g. out[m,n] = f(x[m,n], a[n], b[n], c[n], d[n], ...) with many per-column vectors. Also try broadcast along the tiled dim.` },
  { key: 'partial_tile', brief: `LENS = partial-tiling variants (stress the just-added fix + the 64KB REGISTER_BYTES calibration). Mimic rope.py structure (tile outer dims, flow the full-extent inner slab through reshape/hl.split/hl.join as elementwise) but vary: (1) TWO untiled inner dims of different sizes; (2) a MODERATE untiled slab where the fp32 register width vs fp16 storage width matters (the register cap should bite but the HBM budget alone might not); (3) an untiled slab whose size sits right where REGISTER_BYTES=65536 flips the tile. Goal: find a slab size where the seed still picks a tile that spills / loses >10% to default.` },
]

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    shard_path: { type: 'string' },
    lens: { type: 'string' },
    n_kernels: { type: 'integer' },
    wrote_file: { type: 'boolean' },
    kernels: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          name: { type: 'string' },
          targeted_gap: { type: 'string' },
          hypothesis: { type: 'string' },
          n_shapes: { type: 'integer' },
        },
        required: ['name', 'targeted_gap', 'hypothesis', 'n_shapes'],
      },
    },
  },
  required: ['shard_path', 'lens', 'n_kernels', 'wrote_file', 'kernels'],
}

phase('Ideate')
log(`Spawning ${LENSES.length} adversarial designers, one per assumption-lens.`)

const results = await parallel(
  LENSES.map((lens) => () => {
    const shardPath = `${SHARD_DIR}/shard_${lens.key}.json`
    const prompt =
      PREAMBLE.replace(/\{SHARD_PATH\}/g, shardPath).replace('{SHARD_PATH}', shardPath) +
      `\n\n=== YOUR LENS ===\n${lens.brief}\n\nWrite your shard to: ${shardPath}`
    return agent(prompt, {
      label: `design:${lens.key}`,
      phase: 'Ideate',
      agentType: 'general-purpose',
      schema: SCHEMA,
    })
  })
)

const ok = results.filter(Boolean)
const total = ok.reduce((s, r) => s + (r.n_kernels || 0), 0)
log(`Designers done: ${ok.length}/${LENSES.length} shards, ${total} candidate kernels.`)
return { shards: ok, total_kernels: total }
