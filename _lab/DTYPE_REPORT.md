# bf16/fp16 reduction-seed dtype climb — REPORT (living; updated as the climb proceeds)

## EXECUTIVE SUMMARY (milestone banked @ f1869309 — D4 narrow-N w1 SHIPPED)
The bf16/fp16 frontier is **transfer + correctness + THREE gate-confirmed perf wins** (jsd correctness,
CE wide-V w8, and — newest — D4 narrow-N occupancy-gated w1, +20-60% on a broad narrow-row class). The
inherited fp32 champion transfers free; on top of that the warps lever yielded two faithful wins at
opposite extents (wide-V w8 via `full_width_output`; narrow-N w1 via `grid_rows` + `input_load_itemsize`):
- **The inherited fp32 champion's configs transfer to bf16/fp16 for FREE.** The seed already beats
  torch.compile-default on the large majority of curriculum shapes at bf16 (geo G_cg: softmax 1.31,
  CE 1.19, layer_norm 1.09, kl_div 1.08, sum 1.07, rms_norm 1.06) and fp16 tracks bf16 (transfer
  hypothesis CONFIRMED). The byte-caps that key on `itemsize` self-adjust correctly because the
  resident reduction tile is at input width.
- **Shipped win #1 (correctness):** jsd bf16/fp16 fix (was ControlFlowTensorMismatch; true fp32 no-op).
- **Shipped win #2 (perf): CE bf16 wide-V w8 (+49%)** via a new FAITHFUL fact `full_width_output`
  (does a store write the result back over the reduction extent [M,N], or collapse to a scalar [M]).
  A re-read + scalar-output + wide + half-precision reduction (cross_entropy at V=32k/50k — the GPT2/
  Llama2 vocabs) is reduction-tree-bound, so w8 beats w32 by ~35-49% (ncu: w32's cross-warp shared-mem
  tree throttles reads). geo CE bf16 1.19→1.36. Config-invariant on the CURRICULUM's fp32 shapes (all
  CE vocabs ≥30522 → bytes >102400 cap → stay w32); faithfully excludes layer_norm/softmax/welford/rms
  (full_width) + sum/kl/jsd (not re-read). NOTE: the mechanism is dtype-AGNOSTIC — the gate also fires
  on fp32 CE at V∈[16384,25600] (bytes ≤cap), where w8 also wins (V=18000 fp32 −12%); the curriculum
  just has no realistic fp32 vocab there, so bf16 didn't *create* the penalty, it moved the common
  vocabs into the window where it bites (V=32k fp32=128KB >cap, but V=32k bf16=64KB <cap). [Gate A+D pending.]
  - This is the win the warps lever resisted for 3 attempts (bytes-ramp = occupancy-flip; num_load≥3 =
    false proxy that regressed layer_norm +32%). It landed once the FAITHFUL property (output-width ×
    re-read) was found — exactly the method's "find the workload property, never fence on identity."
  - GATE-CONFIRMED 3/3 (b11e57bd): Gate A no-regression (0% worst, zero non-CE flips), Gate A win
    (+30-38% latency, M-stable), Gate D fact-integrity (full_width_output faithful on synthetic + all
    9 kernels). The `NOT full_width_output` scope is also CONFIRMED-necessary: dropping it to "broaden"
    to softmax/ln is a TRAP — those full_width kernels' w8-preference is a narrow non-monotonic per-N
    resonance (softmax 24576→w8 −23% but 32768→w8 **+33% regress**; ln 24576/32768→w8 +30% regress).
    The scalar-output (CE) win is clean+monotonic; full_width is chaotic. The fact correctly separates them.
- **Every genuine bf16 loser is structurally not a seed win** (per-shape, not geomean) — verified
  taxonomy in "Why each remaining bf16 loser loses" below: long_sum few-row = codegen-structural
  (no split-reduction primitive); softmax_two_pass/welford narrow = kernel-source (2× HBM re-read,
  one-pass siblings match tc); rms/ln (16384,896) = ~3% near-roofline tie. jsd all-V = quick-oracle
  ≤tc (Band-B, not full-confirmed). NOTE: rms_norm (2048,4096) / layer_norm (2048,6144/10240) are NOT
  losers under CUDA-graph (their do_bench "losses" were host-overhead artifacts).
- **Shipped win #3 (perf): D4 narrow-N occupancy-gated num_warps=1 (+20-60%)** — the formerly-DEFERRED
  occupancy win, BUILT once the human asked for it (@ `3408c4f5`, gate-confirmed). Two new FAITHFUL
  `ReductionFact` fields close the gap the prior feasibility assessment said was *required* but unbuilt:
  `grid_rows` (product of static M-axis extents → occupancy `grid_rows//num_sm`) and
  `input_load_itemsize` (the **HBM input-row-load** element width = 2 bf16/fp16, 4 fp32 — read via
  `_accessed_tensor_fake` on the reduction-fed load; DISTINCT from `fact.itemsize`, the fp32-promoted
  accumulator width = 4 at BOTH dtypes for softmax/rms/ln). Rule: **w1 IF `input_load_itemsize>0` AND
  `rnumel*input_load_itemsize<=2048` AND `grid_rows//num_sm <= 512//input_load_itemsize`** (bf16:
  rnumel≤1024 & occ≤256; fp32: rnumel≤512 & occ≤128). Both caps key on the input-load byte width, so they
  scale ~2× with dtype FAITHFULLY (no dtype-kind branch). This STRUCTURALLY fixes the two refuted paths:
  the dtype-faithful occ cap excludes the Path-A poison cell softmax fp32 (32768,512) occ248 (fp32 cap
  128 < 248 → stays w4, no +19.6% regression), and the `input_load_itemsize==0` guard excludes kl_div/jsd
  (where forcing w1 regresses up to +46%). The earlier "4× crossover" worry was an artifact of the wrong
  signal (`fact.itemsize`): with the true input-load byte width the crossover is a clean ~2× shift, fit by
  a single 2048-byte row cap + a 512-occ-byte cap (BYTE_CAP=2048 had ZERO bad cells; 3072/4096 introduced
  regressions). GATE-CONFIRMED: Gate A 3/3 NOT-refuted (softmax bf16 +20-26%, welford +33%, held-out
  (8192,768) +61%, danger cells excluded, noise≪signal), Gate D faithful (both facts; divergence from
  `fact.itemsize` is exactly the discriminator), Gate F mechanism (cross-warp reduction tree overhead;
  boundary verified occ124 w1+52% → occ497 w8+2% → occ993 w8+29%), Gate E no overfit (round HW thresholds,
  no fences). NO-REGRESSION config-proof (777-cell BEFORE/AFTER snapshot): 39 diffs, ALL num_warps→1 in
  the predicted narrow zone, ZERO non-warp/anomaly diffs, fp32 invariant except softmax (16384,512) (a
  genuine fp32 win). It fires on layer_norm/rms_norm/softmax/sum/welford × bf16/fp16/fp32 at real model
  dims (768-1024) — BROADER than the dtype curriculum's 6 shapes — and RESCUES 3 shapes the report had
  filed "exempt" (welford (16384,896); rms/ln narrow). Reframe-relevant: these were seed↔oracle gaps the
  old "oracle≤tc exempt" labeling wrongly shelved.
- **Method finding:** CUDA-graph device time is the correct seed-vs-tc metric; plain do_bench
  mis-attributes Python host-enqueue cost to the kernel at low-M, inventing phantom losses.

---

Baseline: inherited run3 fp32 champion @ `e9251140`. This climb adds bf16/fp16 support.
Authoritative method/task: `_lab/prompts/{hillclimb-method,local-setup,gate-prompts,dtype-task}.md`.
Source of truth: `_lab/dtype_notebook.md` + ledger key `dtype`. All perf = **CUDA-graph device time**
(do_bench mis-attributes host overhead at low-M — see "Harness" below).

## Headline — see EXECUTIVE SUMMARY at top. Two gate-confirmed shipped wins:
1. jsd bf16/fp16 correctness fix (true fp32 no-op). All 9 kernels compile+correct at half precision.
2. CE bf16 wide-V num_warps=8 (+49% on V=32k/50k) via the faithful `full_width_output` fact —
   SURVIVED Gate A+D 3/3. The warps lever took 4 attempts (3 gate-killed proxies) before the faithful
   property (output-width × re-read) landed it. Deferred: narrow-N occupancy-gated w1 (D4, needs
   grid_rows//num_sm). fp32 config-invariant on the curriculum (all CE vocabs >cap; the gate's
   mechanism is dtype-agnostic but no realistic fp32 vocab lands in its byte window); every genuine
   bf16 loser is structurally not a seed win (codegen-structural / kernel-source / near-roofline tie —
   verified taxonomy below; not all are strict oracle-confirmed ceilings).

## Shipped win #1 — the jsd correctness fix, explained

**The one-line change** (`examples/jsd.py`, the per-row accumulator declaration inside the JSD kernel):

```python
# before:
intermediate_dX = hl.zeros([tile_bt, block_size_n], dtype=_input.dtype)
# after:
intermediate_dX = hl.zeros([tile_bt, block_size_n], dtype=torch.float32)
```

**What broke (and why only at bf16/fp16).** `jsd_forward` carries two accumulators across its inner
`for tile_v in hl.tile(V)` loop: `intermediate_loss` (already declared `torch.float32`) and
`intermediate_dX` (declared `dtype=_input.dtype` — i.e. the *input's* dtype). Inside the loop the
gradient term is computed from transcendental math — `Q = exp(X)`, `M = beta*P + (1-beta)*Q`,
`log_M = log(M)`, `kl_q_m = (1-beta)*Q * (X - log_M)` — and Helion (following PyTorch/inductor) does
those `exp`/`log` ops in **fp32** regardless of input dtype, so `kl_q_m` is an **fp32** value. The loop
then does `intermediate_dX += kl_q_m`.

A Helion loop carries its accumulators across iterations, and the compiler requires each carried
variable to keep the **same dtype** on every iteration (it merges the value flowing out of the loop
body with the value declared before it). At **fp32 input** the declared accumulator is already fp32, so
`fp32 += fp32` matches and everything compiles. At **bf16/fp16 input** the accumulator is declared
**bf16/fp16** but the loop body produces **fp32** — the two don't match, and Helion's type-propagation
raises:

```
helion.exc.ControlFlowTensorMismatch: Tensor mismatch in control flow for variable
'intermediate_dX': dtype torch.float32 != torch.bfloat16
```

So at half precision the kernel **failed to compile at all** — it never even ran. That's why the bug
was invisible at fp32 (the dtype this heuristic was originally tuned on) and only surfaced when this
dtype climb first tried to bind `jsd` at bf16.

**Why declaring it fp32 is the right fix (not a workaround).**
- It matches the sibling accumulator `intermediate_loss`, which is *already* `torch.float32` for exactly
  the same reason (it accumulates the same kind of fp32 transcendental terms).
- It matches the **output** tensor: `dX` is allocated `torch.empty_like(loss)` where `loss` is
  `torch.float32`, so the final `dX[tile_bt] = sum(intermediate_dX ...)` is an fp32→fp32 store — no
  down-cast needed, no precision lost.
- It is **numerically correct**: accumulating the gradient in fp32 is strictly at-least-as-accurate as
  accumulating in bf16/fp16 (the half-precision accumulator would have *added* rounding error).
- This is a **kernel-authoring** fix, not a heuristic change — the seed heuristic never sees it. The task
  brief explicitly anticipates this class: "an input-dtype (non-promoted) accumulator… is a
  kernel-authoring question, not a heuristic one."

**Proven a true fp32 no-op.** At fp32 input, `_input.dtype` *is* `torch.float32`, so the new line is
the same value as the old — the change cannot alter fp32 behavior. Verified empirically by dumping the
generated Triton for the fp32 seed config before vs after the edit and diffing: the emitted code is
**byte-identical** apart from the source-line-number comments that shift because the patch added 4
comment lines (a `# src[jsd.py:NNN]` renumbering, not a code change). So fp32 users are completely
unaffected; only the previously-broken bf16/fp16 paths change (from "fails to compile" to "correct").

**Result.** jsd now compiles and is accurate at all three dtypes (β=0.5, the value the curriculum and
tritonbench use): fp32 `max_abs=0.0` (exact no-op), bf16 `max_abs≈1e-4`, fp16 `max_abs≈2e-5`, all well
inside the half-precision tolerance. With this fix, **all 9 reduction kernels compile and are correct at
bf16 and fp16**. (Two residual half-precision caveats, both out of heuristic scope and documented below:
jsd's β=0 / β=1 branches and kl_div/jsd at very wide vocab produce fp16 `NaN` from an un-shifted
`exp`/`log` — fp16's 5-bit exponent overflows; bf16 is fine. The curriculum uses β=0.5, where jsd is
correct at all dtypes.)

**Why the reference (liger_kernel) doesn't hit this — and what's structurally different.** The
upstream Triton JSD (liger_kernel `ops/jsd.py`) allocates its gradient buffer `dX = torch.empty_like(
_input)` — i.e. at the *input* (bf16) dtype — and is fine, which looks like it contradicts the fix
above. It doesn't: liger's bf16 `dX` is an **output buffer written by cast-on-store**, not a carried
accumulator. Liger is a *map-then-reduce-in-PyTorch* kernel:

```
# liger: per vocab tile, no cross-tile accumulator
X = tl.load(...).to(tl.float32); Y = tl.load(...).to(tl.float32)   # cast UP on load
... all math in fp32 ...
tl.store(dX_ptr + offsets, dX, mask=mask)     # fp32 value -> bf16 buffer (implicit one-shot cast)
loss = torch.sum(loss)                         # reduction done in PyTorch, AFTER the kernel
```

The JSD gradient is **per-element** (`dX[j]` depends only on element `j`), so liger writes each tile's
slice independently — there is *no* variable carried across the vocab loop. The fp32 value meets the
bf16 buffer exactly once, at `tl.store`, which is a harmless implicit down-cast; and the loss
reduction is deferred to `torch.sum` outside the kernel, so the kernel carries no loss accumulator
either (it materializes the full `[BT, V]` loss + dX tensors to HBM).

Helion's example instead does *reduce-in-kernel*: it carries `intermediate_loss` and `intermediate_dX`
across `for tile_v in hl.tile(V)` with `+=`, then `torch.sum(..., dim=1)` to a `[BT]` output. A Helion
loop-carried variable must hold one invariant dtype across iterations — so a bf16-declared accumulator
fed an fp32 loop body is illegal, where liger's cast-on-store is not. **It is the accumulation pattern,
not the buffer dtype, that differs**: Helion *can* write a bf16 output via cast-on-store (welford and
layer_norm in the same examples dir do exactly `out[...] = y.to(x.dtype)`); the bf16 buffer was never
the problem. The trade-off is the usual one: liger spends HBM bandwidth materializing full `[BT, V]`
loss+dX with no register-carried state; Helion keeps the accumulator in registers (less HBM, more
register pressure) and reduces in-kernel — the same "reduce-in-kernel, scalar `[BT]` output" structure
that makes `cross_entropy` the kernel the `full_width_output` fact gates for w8 (shipped win #2).
(Minor: liger's `dX` is the true `[BT, V]` per-element gradient; Helion's `dX=empty_like(loss)` is
`[BT]` — the gradient *summed over vocab* — an untested second return the harness ignores; cosmetic.)

## Per-kernel bf16 status — SHIPPED state (baseline warps + jsd fix + CE-wide-V w8)
CUDA-graph G_cg = tc/seed (≥1.0 = seed ≥ tc), test split, geomean. This is the INHERITED fp32
champion's config evaluated at bf16 — it transfers well (the seed already beats tc on most shapes).
| kernel | geo G_cg | notes |
|---|---|---|
| softmax | 1.31 | crushes tc at wide-N (up to 2.1×); losers (8192,896)=0.65 & (131072,128)=0.95 are two-pass-algorithm / narrow-N — the deferred occupancy-warps win would lift (8192,896) ~23% but still algorithm-bound vs tc |
| layer_norm | 1.09 | beats tc broadly; losers (16384,896),(2048,6144-10240) pending oracle |
| cross_entropy | **1.36** (was 1.19) | EDIT#2-v2 w8 lifts wide-V +49% (V=32k/50k); widest-V (4096,151936)=0.96 codegen-bound exempt |
| kl_div | 1.08 | beats tc; widest-V (4096,151936)=0.96 pending oracle |
| sum | 1.07 | beats/ties tc; bf16 accuracy is a kernel floor (input-width accumulator, char.) |
| rms_norm | 1.06 | near-roofline; (16384,896)=0.95 pending oracle |
| welford | 0.99 | losers (16384,896/5120/7168, 8192,14336) ORACLE-CONFIRMED codegen-bound (oracle/tc 0.40-0.66); accuracy = chunk*chunk input-width (char.) |
| long_sum | 0.80 | few-row huge-N ORACLE-CONFIRMED codegen-bound (oracle==seed, oracle/tc=0.53; no split-K in Helion) |
| jsd | 0.82 | loses tc on ALL shapes (correct) but ORACLE-CONFIRMED codegen-bound (oracle/tc 0.92-0.94 — Band-B heavy epilogue, oracle also loses tc). EXEMPT, consistent with run3 fp32 jsd=tie. |

fp16 mirrors bf16 for ALL 9 kernels (transfer confirmed incl. welford+long_sum, the dtype-sensitive
ones). fp16 EXCEPTION: kl_div/jsd wide-V (V≥50257) produce kernel-authoring NaN (5-bit exponent) — those
rows are NOT valid perf comparisons and are excluded from fp16 geomeans; fp16 kl_div/jsd validated only
at narrow-V (V≤49152).

ROUND-ONE per-shape honesty: most shapes beat tc. Remaining genuine losers split into 3 verified
classes (see "Why each remaining bf16 loser loses"): codegen-structural (long_sum few-row — no
split-reduction primitive), kernel-source (softmax_two_pass/welford narrow — 2× HBM re-read in the
example), near-roofline tie (rms/ln (16384,896) ~3%). jsd all-V is quick-oracle ≤tc (Band-B,
not full-confirmed). rms_norm (2048,4096) & layer_norm (2048,6144/10240) were de-listed — CUDA-graph
shows they beat tc (do_bench host-overhead artifacts). None is an un-pursued clean SEED win. The
reachable-but-deferred wins (need new facts): occupancy-gated narrow-N warps (D4); CE bf16 wide-V
w8 was the D5 candidate and is now SHIPPED as EDIT#2-v2.

## What transferred free vs needed re-tuning
- **bf16 ≈ fp16**: confirmed — itemsize=2 for both, near-identical optimal configs everywhere measured.
  The task's transfer hypothesis holds; fp16 is mostly a correctness question (rms/ln/softmax/CE
  fp16-safe; kl_div/jsd wide-V have kernel-authoring fp16 NaN — narrow 5-bit exponent).
- **Most fp32-tuned levers transferred for FREE**: persistent/looped, eviction, Band-B/C caps all
  faithful at bf16 — the byte-caps that key on itemsize self-adjust correctly because the resident
  reduction tile IS at input width, so cap//itemsize is right; the "welford 2× cap" thesis was
  FALSIFIED (matched-lever A/B). The inherited fp32 champion's configs make the seed beat tc on most
  bf16/fp16 shapes with NO heuristic change. This run's main perf finding is "it transfers."
- **num_warps — the one lever where a dtype-aware rule could help, but it's OCCUPANCY-gated not
  bytes-gated, so no static rnumel/itemsize rule ships cleanly.** Two byte-ramp attempts were built and
  BOTH reverted (Gate A caught real >10% regressions across the M axis). The win (narrow-N fewer warps,
  10-20%) is real + ncu-explained but needs a grid_rows//num_sm fact — deferred (D4). NET WARPS CHANGE
  SHIPPED: none (logic AST-identical to baseline).

## Open questions resolved
- "Are bf16 byte-caps mis-sized 2×?" → NO for resident-tile caps (reduction tile is input-width; welford
  cap thesis falsified by matched-lever A/B). The warps ramp IS dtype-suboptimal but the fix is
  occupancy-gated, not a clean byte re-key (both byte-ramps reverted by Gate A).
- "Do low-M bf16 norms really lose to tc?" → NO — do_bench host-overhead artifact; CUDA-graph shows
  Helion ≥ tc on-device (rms_norm (2048,4096) bf16: do_bench 0.76 → CUDA-graph 1.06).
- "Is fewer-warps-at-narrow-N real?" → YES, ncu-confirmed (cross-warp reduction shared-mem overhead) —
  but occupancy(M)-gated: inverts to >10% regression at low AND very-high M (Gate A caught this).

## Why each remaining bf16 loser loses to torch.compile — verified per-case taxonomy
CORRECTION (the earlier blanket "every loser is oracle-confirmed CODEGEN-bound exempt" was too strong;
verified generated-Triton + CUDA-graph re-measurement gives **three distinct classes**, only one truly
codegen-bound). Loser set is also smaller than first claimed — see the de-listed shapes below.

**Class A — genuinely codegen-structural (Helion's DSL cannot express it via any config):**
- **long_sum few-row huge-N** (e.g. (8,2097152) G_cg≈0.36): Helion maps one row → one program, so
  grid = M = 8 programs on 132 SMs (~6% occupancy), looping the 2M reduction *inside* each program.
  torch.compile emits a **two-stage split reduction** — kernel 1 = 512 programs (8 rows × 64 splits)
  writing (8,64) partials, kernel 2 reduces (8,64)→(8) — splitting ONE row's reduction across SMs.
  Helion has **no split-reduction / cooperative-grid / cross-program-atomic construct**: `reduction_loops`
  only chunks *within* a program, the grid comes solely from `hl.tile(m)`. All 3 long_sum example
  variants launch grid (8,). Unreachable by seed OR autotuner; needs a different hand-written kernel.
  **Legitimately exempt.** (Full oracle == seed, oracle/tc=0.53.)

**Class B — kernel-AUTHORING / source-bound (NOT codegen-bound — the seed is optimal for the kernel as
written, but the EXAMPLE re-reads HBM; a one-pass sibling kernel matches tc):**
- **softmax_two_pass narrow-N** (e.g. (8192,896) G_cg≈0.65): the example has TWO `for tile_n` loops →
  loads the row from HBM **twice** (max+sum pass, then normalize pass); tc is one-pass (1 load). Verified:
  the one-pass `softmax` example compiles to a single load and matches tc. The 2× traffic is in the
  source, not a seed knob → the right label is source-bound, not codegen-bound.
- **welford narrow/mid-N** (e.g. (16384,896) G_cg≈0.75): the example re-reads the row (combine + normalize
  passes, 2 loads) AND runs the serial Welford count/mean/M2 recurrence; tc uses a cheap one-load
  sum+sum-of-squares (1 x-load). The `layer_norm_fwd` example (1 load) matches tc. Source choice, not knob.
- **sum / welford bf16 ACCURACY** (separate from perf): input-width accumulator (sum reduces bf16; welford
  `chunk*chunk` at input width) — kernel-authoring (fixable by upcast-before-reduce like rms_norm).

**Class C — near-roofline tie (not a limit at all, ~3%):**
- **rms_norm / layer_norm (16384,896)** (G_cg≈0.969): structurally IDENTICAL to tc (one-pass, single
  load, and tc *also* picks XBLOCK=1/one-row-per-program). Both run ~2.0–2.2 TB/s (60–66% of HBM3 peak).
  The 3% is micro-scheduling/launch-wrapper noise, not a missing construct, not a flippable knob.

**De-listed (NOT actually losers — earlier "losses" were do_bench host-overhead artifacts; CUDA-graph
shows they BEAT tc):** rms_norm (2048,4096) G_cg=1.077, layer_norm (2048,6144) 1.067, (2048,10240) 1.014.

**Method caveats on the exemption verdicts:** the oracle harness times with do_bench (unreliable at
low-M) and several entries used `quick` (not `full`) effort — insufficient for a strict ceiling claim.
The Class-A long_sum verdict is full-oracle + structural (solid); Class-B is structural-from-source
(solid, doesn't depend on the oracle); jsd's "oracle/tc 0.92-0.94" is quick-oracle and would want a
full CUDA-graph oracle to call a strict ceiling (it's Band-B, plausibly source/codegen-bound but
not rigorously confirmed). Honest status: not every loser is a *proven* ceiling, but every loser is
either structurally unreachable by the seed (A/B) or a near-tie (C) — none is an un-pursued seed win.

## Harness note (method §4 specialization)
Plain `triton.testing.do_bench` mis-attributes Python host-enqueue cost (~70µs/call for Helion vs ~47
for tc) to the KERNEL at low-M bandwidth-bound shapes, inventing "losses" that vanish under device-time
timing. The headline metric for this run is **CUDA-graph per-call device time** (captures both arms
identically, fair), do_bench kept as recorded secondary.

TRIPLE-CONFIRMED + tritonbench correction (2026-06-08): on the canonical case rms_norm bf16 (2048,4096),
one process, 3 methods — do_bench G=0.681 (seed "loses"), **CUDA-graph G=1.068**, **torch-profiler
device-time G=1.121** (kernel is ~8µs on-device both arms; do_bench's ~21µs is host dispatch). The
profiler reads CUPTI kernel durations — independent of BOTH do_bench and graph capture — and agrees with
CUDA-graph, so the host-overhead artifact is real, not a graph-capture quirk. CORRECTION to an earlier
overstatement: tritonbench does NOT "lack" CUDA-graph timing — it has `--cudagraph` and
`--latency-measure-mode {triton_do_bench, inductor_benchmarker, profiler, gpu_events}`; it merely
DEFAULTS to `triton_do_bench`. The lab's `seed_vs_tc.py` cross-check ran the default mode, so its low-M
numbers carry the same host-overhead caveat; passing `--cudagraph` (or `--latency-measure-mode
gpu_events`) would give tritonbench-native device time. So the right statement is "do_bench DEFAULT
mis-times low-M; use a device-time mode," not "tritonbench can't."

## #1 DEFERRED OPPORTUNITY — CE bf16 wide-V wants w8 (+35-49%, fully ncu-characterized)
The single biggest perf win found. cross_entropy bf16 at V=32000/50257 (GPT2/Llama2 vocabs) runs
~35-49% faster at num_warps=8 than the seed's w32, M-stable (occ 15-124). NOT shipped: every static
gate keyed on an EXISTING fact was Gate-killed (bytes-ramp = occupancy-flip + fp32 under-warp;
num_load≥3 = false proxy, regressed layer_norm +32%). ncu mechanism: CE is a PERSISTENT full-row
register reduction with a DOUBLE reduction (amax+sum); at w32 the cross-warp shared-mem reduction tree
is 4× costlier (23× bank conflicts) and throttles the DRAM read pipeline. The faithful gate is a 5-way
conjunction — persistent (reduction_loops=[None]) × multiple reduction trees × wide × half-precision ×
scalar output — that needs NEW facts (reduction-tree count + scalar-vs-full-width output) which don't
exist (`num_reduction_ops` was falsified by run3's fact-integrity gate). Capturing it = a dedicated
fact-building effort, deferred (D5) with the full mechanism documented. Controls that prove the
conjunction is faithful, not a fence: sum (1 reduction → w32), layer_norm (full-width output → w32),
kl_div/jsd (streaming not persistent → warp-neutral) all correctly excluded.

## The warps lever — overall finding (3 attempts, all gate-killed or deferred)
The reduction `num_warps` ramp is genuinely dtype-suboptimal in several regimes, but the optimum is a
multi-dimensional function (extent × itemsize × occupancy/M × reduction-tree-count × output-width) that
no simple seed-ramp captures cleanly. Three faithful-looking ramps were built and rejected by the
gates/measurement (narrow-N occupancy-flip; wide byte-ramp fp32 under-warp; num_load false-proxy). The
honest conclusion: improving it needs either richer facts (occupancy grid_rows//num_sm, reduction-tree
count, output-width) or per-shape autotune — a research effort, not a ramp tweak. Net warps change
shipped: NONE (logic AST-identical to baseline; fp32 provably non-regressed). The gates worked.

## Mechanistic discoveries (hard-won, reusable — validated by measurement/ncu this run + the fork)
- **Cross-warp reduction overhead governs narrow-row warps.** At a narrow reduction extent the
  cross-warp shared-memory reduction tree (+`__syncthreads`) is pure overhead — w1 reduces in-register
  via a shuffle (0 shared traffic, 0 barriers); ncu shows w8 there issues ~262k shared ld/st + barriers
  and is ~1.5× slower at *higher* occupancy. This is WHY narrow-N wants fewer warps (the D4/CE wins).
- **The narrow-N w1 win is OCCUPANCY-gated with a BIMODAL CLIFF, not a gentle slope.** w1 wins at
  low/moderate `occ=grid_rows//num_sm` then *inverts hard* past a per-class ceiling — and the inversion
  is a CLIFF: softmax bf16 N=512 is w1 −24% at occ≤248 but **w32 +6× at occ≈1985** (a catastrophe, not
  a wiggle). The ceiling is BIMODAL by kernel structure: re-read-from-HBM-each-pass kernels {softmax
  ~occ496, welford ~992} flip ~8× EARLIER than register-resident-reuse kernels {rms_norm, layer_norm
  >1985}. `occ×num_load` does NOT normalize it (12× spread). This bimodality + the cliff are why a
  single occ threshold can't ship and why a wrong threshold is dangerous (high-occ shapes fall off it).
- **The crossover is also DTYPE-shifted ~4× for re-read kernels.** softmax N=512 flips at fp32 occ~124
  vs bf16 occ~496 — a ~4× shift while input bytes differ only 2× (the two-pass HBM re-read amplifies the
  input-bytes effect). So the faithful occupancy threshold keys on INPUT-LOAD itemsize (dtype-agnostic),
  but with an empirically-fit (~4×), not literal-bytes, scaling. See D4 in the notebook.
- **`fact.itemsize` is the REDUCTION/accumulator width, not the input-load width** — =4 for
  softmax/rms/ln at BOTH fp32 and bf16 (they reduce fp32-promoted values), =2 only for input-width
  reducers (sum/CE/welford bf16). This is why caps keyed on `fact.itemsize` (Path A) can't discriminate
  fp32 from bf16; the dtype-faithful per-byte signal is `MemoryOpFact.dtype` (the input load).
- **CUDA-graph timing is fair to torch.compile** (fork audit): a torch-profiler device-time read agrees
  with CUDA-graph for non-self-graphing arms (rms_norm bf16 (2048,4096): graph 1.068 ≈ profiler 1.121,
  do_bench 0.681). The do_bench host-overhead artifact is real; the only timing TRAP is double-wrapping
  a reduce-overhead/already-graphed tc baseline. tritonbench HAS `--cudagraph`/`--latency-measure-mode`
  but DEFAULTS to do_bench — so "use a device-time mode", not "tritonbench can't".

## Remaining (open, lower-priority)
- welford narrow/mid-N tc-losses: ORACLE-CONFIRMED codegen-bound (oracle/tc 0.40-0.66) — exempt.
- The occupancy-gated narrow-N w1 win (D4) and the CE-wide-V w8 win (D5): both real, both need new
  facts, both deferred with full characterization.

## Fairness-to-torch.compile audit (CUDA-graph timing) — 2026-06-08
Worry (correct instinct): does CUDA-graph wrapping penalize torch.compile, or does it miss tc's
graph-using fast path? Resolved on rms_norm bf16 (2048,4096), all arms × {do_bench, my CUDA-graph,
torch-profiler device-time (CUPTI, imposes NO graph — neutral judge)}:
  seed              : do_bench 22.2  cudagraph 8.4  profiler_dev 7.82us
  tc_default        : do_bench 23.0  cudagraph 8.8  profiler_dev 8.66us   -> G_profiler = 1.107 (seed faster)
  tc_reduce_overhead: do_bench 72.1  cudagraph  n/a  profiler_dev 30.3us
FINDINGS: (1) For seed and tc_default (both plain eager-launch callables), the CUDA-graph number == the
profiler device-time number — so graph-wrapping did NOT distort either arm; the comparison is fair, and
the seed's ~11% device win is confirmed by a graph-FREE metric. (2) The real trap your instinct points at
is DOUBLE-WRAPPING: torch.compile(mode='reduce-overhead') lays down its OWN cudagraph; wrapping that in an
outer cudagraph fails/garbles (n/a) and its device-time balloons to 30us (graph machinery is pure overhead
for a single 8us bandwidth-bound op — reduce-overhead pays off for many-small-kernel models, not one tiny
kernel). So: benchmark via DEVICE TIME (profiler/gpu_events), OR cudagraph-wrap only non-self-graphing arms
(seed + tc_default), NEVER cudagraph + a reduce-overhead baseline. (3) tritonbench --cudagraph /
--latency-measure-mode gpu_events is a valid device-time path and would reproduce seed>=tc here; just don't
combine it with a reduce-overhead tc baseline. CONCLUSION: the run's seed-vs-tc-DEFAULT comparison is fair;
tc's own fast path is actually slower on-device for these tiny ops, so comparing to tc_default favors tc.
