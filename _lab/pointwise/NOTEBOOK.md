# POINTWISE SEED HEURISTIC — worker notebook (source of truth; trust over context, method §6.1)

Run: unattended hill-climb for a NEW pointwise/elementwise autotuner seed heuristic.
Base SHA: **9bf7b1f9ec742aa9e776ce59d4c7df2d20d79e7f** (branch `pointwise-seed-heuristic`,
worktree `/home/calebkim/helion-new-heuristics/helion-pointwise`, off `reduction-4pr-stack` tip).
Interpreter: `/home/calebkim/.conda/envs/helion/bin/python`. GPU: 1×H100, pin CUDA_VISIBLE_DEVICES.
PYTHONPATH = the worktree; run scripts cwd=/tmp; assert helion.__file__ under worktree.

Goal hierarchy (lexical): FAITHFULNESS > GENERALITY > PERFORMANCE.
Bar: per-shape floor G≥0.75 vs tc (here oracle≈tc, so target ≈ seeded≈oracle); kernel geomean is a
diagnostic (~0.85 healthy). DoD = Phase1 (swiglu+geglu) banked, then keep climbing through P2 + overtime.

═══════════════════════════════════════════════════════════════════════
## STEP-0 CALIBRATION (3-shape headroom test, swiglu bf16 cold-L2) — DONE, niche CONFIRMED
═══════════════════════════════════════════════════════════════════════
Script: `_lab/pointwise/headroom3.py` (med-of-3, do_bench cold-L2).
| shape | default bs=32 | oracle (cfg) | tc | def→oracle | oracle_vs_tc |
|---|---|---|---|---|---|
| (16384,11008) | 3393µs 0.32TB/s | 483µs (bs2048,w16,s2) 2.24TB/s | 484.6µs | **7.02×** | 1.003× |
| (512,11008)   | 110µs 0.31TB/s  | 20.7µs (bs1024,w4,s1) 1.63TB/s  | 21.15µs | **5.32×** | 1.022× |
| (4096,4096)   | 319.6µs 0.32TB/s| 50.24µs (bs8192,w4,s1) 2.00TB/s | 51.04µs | **6.36×** | 1.016× |

Findings:
- default→oracle 5.3–7.0× (matches task §2). Niche REAL. Root cause = bs=32 (config_spec.py
  `BlockSizeSpec._fragment` 1842-1868: total_ndim<=2 & reduction_numel<=128 → default=32).
- oracle ≈ tc parity (retarget seed→oracle/tc parity; do NOT chase >1× vs tc — HBM ceiling).
- COLD-L2 verified: (512,11008)=33.8MB < 50MB L2 reads 1.63 TB/s (not a 3-5TB/s hot artifact) →
  triton do_bench flushes L2 this build. ✅
- Oracle clusters across bs∈[1024,8192] @ w4-w16 (best5 within noise). Lever = land bs in KB range
  + sane warps; warps/stages secondary. swiglu facts: 0 reduction/matmul/accumulator, 3 memory_op
  (2 load + 1 store, bf16, ndim=1, inner_extent=None for 1-D flatten). DISJOINTNESS GATE CLEAN.

═══════════════════════════════════════════════════════════════════════
## COMMIT TRAIL (pointwise-seed-heuristic branch)
═══════════════════════════════════════════════════════════════════════
- dda82b3  Phase 1: PointwiseElementwiseFact + populator + TritonPointwiseSeedHeuristic
- f49543d  Gate D fix #1 (accessed_numel == total_numel, exclude full-rank broadcast) + Gate F (emit block_sizes only)
- 8dc7889  Gate D fix #2 (stride-aware accessed_numel — exclude .expand()/broadcast_tensors stride-0)
- abba851  N-D perf: outer dims -> 1, inner covers contiguous row (residual_add 0.906->0.984)
- a1b5402  TILE_BYTES 16384 -> 8192 (robust byte budget; lifts bias_gelu/relu_squared, 1-D neutral)
- b4a4a681  Gate D fix #3 (accessed_numel >= total_numel — count oversized operands) (DoD banked, all gates PASS)
- 1ef6c5c0  refactor-critic: trim fact 6→2 fields (behavior-preserving)
- 268edf3b  spill-to-outer distribute: fix tall-skinny (BROADEN, corpus byte-identical)  ← CHAMPION

TALL-SKINNY BROADEN (268edf3b, measured breadth — completeness-critic 2nd-pass find): the outer→1
distribute starved short-inner tensors (image RGBA N=4, per-head N=64, head-dim slices: [M,8]→[1,8],
32× below the [32,8] default → below floor). Fixed by spilling leftover budget outward ([M,8]→[128,8],
a coalesced 1024-tile). MEASURED on cool GPU: residual_add (1048576,8) seed [128,8] beats default 1.07×
G_vs_tc=1.013; (262144,64) [16,64] 1.14× G=0.973; (131072,256) [4,256] 1.15× G=1.005. Corpus byte-
identical (no N<budget). The distribute now handles ALL aspect ratios. (tallskinny_probe.py)

## CURRENT HEURISTIC STATE  (committed @ a1b5402 on pointwise-seed-heuristic)
═══════════════════════════════════════════════════════════════════════
- **Fact**: `PointwiseElementwiseFact` (config_spec.py) — DERIVED, reads `memory_op_facts` slices +
  block_sizes (no graph walk). Fields: total_numel, n_block_dims, block_size_hints, bytes_per_elem,
  n_load, n_store. Populator `build_pointwise_facts` (device_ir.py Phase 5) — built ONLY on absence
  of reduction/matmul/accumulator facts (disjointness). bytes_per_elem = Σ itemsize over FULL-EXTENT
  ops (ndim==n_block_dims; broadcast inputs excluded).
- **Heuristic**: `TritonPointwiseSeedHeuristic` (triton.py), registered in __init__.py.
  is_eligible=bool(pointwise_facts). Lever: target=max(BLOCK_FLOOR=256, TILE_BYTES=16384//bytes/elem),
  capped by grid (total_numel//(num_sm*MIN_WAVES=8)), distributed across block dims (outer→floor,
  innermost→remainder). Config(block_sizes, num_warps=4, num_stages=1, pid='flat'). ONLY non-default
  field = block_sizes.
- levers_since_refactor = 1 (the seed lever); fire refactor-critic every K≈4-5 levers / at freeze.
- Decisions logged: (D1) fire BROADLY (no HARDWARE_TARGETS gate) — lever keys on bytes/occupancy/
  num_sm which is hardware-general; prefer-to-fire (method §2). Off-H100 = best-guess (accepted).
  (D2) promote_seed_to_default=False (match reduction heuristics; force configs=[seed] for bench).
  (D3) authored corpus kernels (relu_squared/bias_gelu/dyt) live in _lab/pointwise/ptw_kernels.py,
  NOT committed examples/ — the heuristic doesn't depend on them; keeps PR to the 4 helion files.

═══════════════════════════════════════════════════════════════════════
## PER-SHAPE STATUS TABLE  (G = tc/seed; floor 0.75; anchor ref)
═══════════════════════════════════════════════════════════════════════
(empty — populated once the seed is built + benched on the curriculum)

═══════════════════════════════════════════════════════════════════════
## PER-(KERNEL,DTYPE) GEOMEAN TABLE  (worst-kernel-first triage signal)
═══════════════════════════════════════════════════════════════════════
CLEAN champion @ a1b5402 (TILE_BYTES=8192), train+val, bf16, do_bench med-of-9 cold-L2, COOL GPUs
(<40°C — see thermal note below; earlier hot-GPU-1 numbers were CORRUPTED up to 50%).
| kernel,dtype | seeded_vs_default | G_vs_tc (geo) | min_G | acc | below_floor | seed @8k |
|---|---|---|---|---|---|---|
| swiglu,bf16 | 6.83× | 0.998 | 0.995 | 23/23 | none | [1024] |
| geglu,bf16  | 6.84× | 0.996 | 0.993 | 17/17 | none | [1024] |
| residual_add,bf16 | 1.35× | 0.986 | 0.964 | 16/16 | none | [1,1024] |
| relu_squared,bf16 | 9.99× | 0.994 | 0.988 | 12/12 | none | [2048] |
| bias_gelu,bf16 | 1.21× | 0.968 | 0.903 | 15/15 | none | [1,2048] |
| dyt,bf16 | HELD-OUT (freeze read only) | — | — | — | — | [1,2048] |
ALL 5 fit kernels: near tc PARITY (G 0.968-0.998), beat default (1.21-9.99×), clear the 0.75 floor
with huge margin (min_G 0.90-0.995), 0 acc fails, 0 disasters. The oracle-retarget target (seed≈tc
parity) is MET on all. (residual_add/bias_gelu lower svd because their N-D default [32,32]=1024 is
already a decent tile, vs the 1-D default [32]=32.)

⚠️ THERMAL: GPU-1 ran hot (65°C) from sustained back-to-back benching → corrupted timings up to 50%,
ASYMMETRICALLY across configs, FLIPPING A/B signs. It briefly made bias_gelu look like seed<default
(svd 0.37-0.96) + spawned a bogus "broadcast needs multi-row tile" theory. On COOL GPUs all of that
vanished: bias_gelu seed BEATS default 1.21×, [1,inner] is correct for broadcast kernels too (bias[N]
is L2-resident, NOT extra HBM traffic). See [[gpu-thermal-measurement-corruption]]. RULE: cool GPU
(<~35°C) + reproduce-twice for any trusted timing.

═══════════════════════════════════════════════════════════════════════
## FROZEN-CHAMPION ANCHOR  (per (kernel,dtype) cell; ratchets up only)
═══════════════════════════════════════════════════════════════════════
(to be SET at the Phase-1 freeze, pending Gate R accept @ dda82b3:
  swiglu,bf16 G_geo=0.9948 ; geglu,bf16 G_geo=0.9946)

═══════════════════════════════════════════════════════════════════════
## GATE EVIDENCE (Phase 1, @ dda82b3) — gathered, verdicts pending workflow wf_6cff2546
═══════════════════════════════════════════════════════════════════════
- Gate D divergence (gate_evidence.py, compile-only): per_token_fp8_quant (amax over N) → pointwise=0
  (routes to triton_reduction_tile, disjointness holds); bias_gelu_nd broadcast → pointwise=1
  bytes/elem=4 (traffic-2, broadcast bias excluded); swiglu/geglu bytes/elem=6 (traffic-3). Population
  reads MemoryOpFact itemsizes, not op name/dtype literal.
- Gate R no-regression: rms_norm/matmul/softmax/layer_norm facts+seed BYTE-IDENTICAL in
  helion-3stage(BEFORE) vs helion-pointwise(AFTER). Purely additive.
- Gate F codegen (to_triton_code): bs32→524288 programs×64B/tensor (0.32TB/s) vs bs2048→8192
  programs×4KB/tensor coalesced (2.24TB/s). block_sizes is the ONLY non-default field; carries the
  whole win (revert→32 = default arm = 6.8× slower). No dead knobs.

═══════════════════════════════════════════════════════════════════════
## BANKED WINS
═══════════════════════════════════════════════════════════════════════
(none yet — Step-0 calibration banked as scaffolding commit)

═══════════════════════════════════════════════════════════════════════
## TRIED-AND-REJECTED  (compact: what / why / evidence pointer)
═══════════════════════════════════════════════════════════════════════
- num_warps ramp (BROADEN #1): DEFER/REJECT — at the seed's tiles (1024-2048 elem) w4 is optimal;
  w8 within-noise (x1.000-1.005), w16 REGRESSES ~10% (x0.90). No >5% gain → adding the knob would be
  a dead knob (Gate H rule 2). Keep w4 default. (warp_ramp_probe.py, cool GPU med-of-9.)
- "broadcast needs a multi-row tile" (bias_gelu): REJECTED — was a GPU-thermal artifact; on cool GPUs
  [1,inner] beats the multi-row default (bias[N] is L2-resident). [[gpu-thermal-measurement-corruption]]
- raising the byte budget so N-D covers the full row ([1,8192]): mixed — [1,8192] tanks on (4096,8192)
  G=0.865 (cliff) while helping others; TILE_BYTES=8192 ([1,2048]/[1,1024]) more robust. (nd_tile_probe.py)
- flatten_loops=[True] for N-D (treat N-D pointwise as 1-D): DEFER — only ~1.7% on residual_add
  (190->187us), near-noise, and it flips a structural default (Gate F) for <2% on an already-0.986 kernel.
  Not worth the complexity. N-D is at the codegen ceiling. (flatten_loops probe inline.)

CEILING NOTE: every faithful perf lever probed in overtime (warp ramp, flatten_loops, N-D budget) is at
the codegen ceiling or within-noise. seed = oracle = tc ~= parity ~= HBM peak (~2.2 TB/s) on ALL kernels.
Per Step-3 invariant (seed ~= oracle => nothing more reachable), the pointwise family is at its ceiling.
The oracle-retarget target (seed ~= tc parity) is MET; chasing >1x vs tc is chasing noise (fixed HBM BW).

═══════════════════════════════════════════════════════════════════════
## GATE VERDICTS — ALL PASS @ champion b4a4a681 (DoD milestone BANKED)
═══════════════════════════════════════════════════════════════════════
- Gate A (adversarial verify, wf_1b971801 @ f49543d): PASS — 3/3 skeptics refuted=false; independent
  CUDA-Event repro (different method) confirms tc-parity 0.96-0.99. Win real, reproduced, general.
- Gate F (mechanism, wf_6cff2546 @ dda82b3): PASS — bs32→bs2048 in lowered code (524288 tiny programs
  → 8192 coalesced), block_sizes the ONLY non-default field; flagged num_warps/stages/pid inert → dropped.
- Gate H (generality, wf_6cff2546 + wf_beb416c3): KEEP — faithful keys (bytes/numel/occupancy/num_sm),
  broad firing, byte-budget+occupancy form; 6 BROADEN items queued.
- Gate R (regression-referee, wf_beb416c3 @ a1b5402): accept — negatives byte-identical, no realistic
  shape below floor, 5/5 cells up vs default; anchor SET.
- Gate D (fact faithfulness): PASS @ b4a4a681 after THREE adversarial fixes (each a real divergence):
  (1) f49543d full-rank [M,1]/[1,N] broadcast (was counted by a bare-ndim proxy) → accessed_numel==total;
  (2) 8dc7889 stride-0 .expand()/broadcast_tensors → stride-aware accessed_numel;
  (3) b4a4a681 oversized operand (padded/sliced buffer wider than tile) → accessed_numel >= total.
  Final Gate-D (4th pass, agent a3f0104d): refuted=FALSE. population faithful (dtype.itemsize, mixed→8),
  threshold faithful (divisor in byte budget). Only residual corner = re-subscripting the SAME tile from
  HBM (non-idiomatic; corpus binds-once → correct bytes; lands in robust zone) → scoped_deferral, logged.
- Gate E (overfit firewall, FREEZE read @ b4a4a681, ONCE): PASS — see FREEZE below.

═══════════════════════════════════════════════════════════════════════
## FREEZE / DoD (Gate E sole TEST+held-out read, ONCE @ b4a4a681, cool GPUs)
═══════════════════════════════════════════════════════════════════════
TEST split (held-out shapes, first read) — 5 fit kernels:
| kernel | G_vs_tc | min_G | svd | acc |
|---|---|---|---|---|
| swiglu | 0.997 | 0.994 | 6.82× | 7/7 |
| geglu | 0.996 | 0.995 | 6.78× | 5/5 |
| residual_add | 0.977 | 0.972 | 1.36× | 5/5 |
| relu_squared | 0.998 | 0.991 | 9.93× | 4/4 |
| bias_gelu | 0.977 | 0.952 | 1.20× | 5/5 |
HELD-OUT KERNEL dyt (never fitted, first read, 18 shapes): G=0.967, min_G=0.876, svd 1.18× (1.05-1.32),
  18/18 acc, cfg [1,N-covered]. CLEARS the floor everywhere; the activation-blind bytes/numel seed
  GENERALIZES to an unseen kernel = interpolation, not memorization. NO overfit.
FREEZE VERDICT: PASS. Test tracks train; no held-out shape below floor where train clears it.
DoD MET + BANKED (champion b4a4a681). Per method §6.0, this is a milestone — KEEP CLIMBING (overtime).

═══════════════════════════════════════════════════════════════════════
## OVERTIME LOG (post-DoD, champion 1ef6c5c0; never-stop §6.0)
═══════════════════════════════════════════════════════════════════════
- refactor-critic (wf_9fede741): heuristic near-minimal; ONE win = trim fact 6→2 fields (drop
  n_block_dims/block_size_hints/n_load/n_store, read by no branch). DONE @ 1ef6c5c0 (configs byte-identical).
- completeness-critic (wf_9fede741, 11 gaps) — closures:
  ✅ robustness/decode CORRECTNESS canaries: all 6 kernels pass acc on decode M=1..256 + odd/prime M
     + non-pow2 N (swiglu/geglu/residual_add 8/8, relu² 5/5, bias_gelu 6/6, dyt 6/6). res_robustness.json.
  ✅ accumulator negative recognizer (the task's 3rd, tested in ISOLATION): running_carry (carried
     [tile_m,1] tensor, no reduction/matmul) → accum=1, pointwise=0, excluded. accum_negative_probe.py.
  ✅ dtype fp16/fp32: seed FIRES + correct + dtype-FAITHFUL tile (fp32 traffic-3 bytes=12 → [512],
     smaller than bf16 [1024]; fp16 bytes=6 → [1024]). The bytes-aware budget reads dtype.itemsize.
  Remaining (logged, low/med): MIN_WAVES=8 unaudited (small-problem zone, BROADEN); oversized/concat
  real curriculum shape (Gate D fix #3 covers it but task pins the corpus → human-review queue);
  sub-parity push residual_add/bias_gelu ~0.95-0.97 (N-D codegen ceiling, no clean lever).

## BROADEN-AND-REFACTOR QUEUE  (Priority-2 standing work)
═══════════════════════════════════════════════════════════════════════
From Gate H @ dda82b3 (the 6 self-flagged items) + measurements:
1. num_warps ramp for large tiles (currently flat w4). Headroom: w4 best/tied on 2/3 calib shapes;
   w16 marginal on largest (within noise). LIKELY within-noise → DEFER unless A/B shows >5%.
2. N-D inner-tile tuning — DONE (outer→1 + cover-row @ abba851; residual_add 0.906→0.984).
3. TILE_BYTES scaling with total_numel — partially addressed (8192 robust zone @ a1b5402).
4. BLOCK_FLOOR/MIN_WAVES small-problem + non-H100 SM-count sweep (decode shapes correctness-only).
5. wire n_load/n_store into the budget for many-input fused elementwise (currently diagnostic).
6. per-dtype TILE_BYTES validation (8192 tuned on bf16; fp16/fp32/fp8 unmeasured) — dtype climb is
   a separate axis; pointwise corpus is bf16 (real serving). Defer.

═══════════════════════════════════════════════════════════════════════
## DEFERRED HARD-PILE + BORDERLINE
═══════════════════════════════════════════════════════════════════════
- (RESOLVED — was a THERMAL ARTIFACT, not a real issue) bias_gelu "seed<default / broadcast needs
  multi-row tile": the svd 0.37-0.96 + min_G 0.79 were GPU-1 thermal corruption. On a cool GPU
  bias_gelu seed [1,2048] BEATS default 1.21×, G=0.968, min 0.903. [1,inner] is correct for broadcast
  kernels (bias[N] L2-resident). NO broadcast lever needed. See [[gpu-thermal-measurement-corruption]].
- (none active)

═══════════════════════════════════════════════════════════════════════
## HUMAN-REVIEW QUEUE  (append-only; one line each)
═══════════════════════════════════════════════════════════════════════
- {tall-skinny curriculum gap} The 168-shape corpus has min inner N=768 (all LLM FFN/proj widths), so
  it never sampled short-inner pointwise (image RGBA N=4, per-head N=64, head-dim slices). The seed now
  handles them (spill-to-outer @ 268edf3b, measured beats-default+parity), but the task PINS the
  curriculum so I did NOT add them to shapes_pointwise_draft.py (val/test envelope check forbids
  N<768). RECOMMEND: add a few real tall-skinny shapes (e.g. residual_add/SiLU on [tokens,head_dim=64/128]
  per-head, or vision elementwise N=3/4) to a 'narrow' robustness set. Where to reverse: _seed_block_sizes
  in triton.py + tallskinny_probe.py.
- {oversized/concat curriculum} Gate D fix #3 (accessed_numel>=total) handles oversized operands but no
  real oversized/concat shape is in the corpus (task pins it). RECOMMEND a padded/concat elementwise shape.
- {stride-blindness, low-value} The seed (like the compiler DEFAULT + Helion's identity loop_orders) is
  stride-blind: a TRANSPOSED/col-major short-inner input gets a tile optimized for the wrong physical dim.
  NOT seed-introduced (pipeline-wide), correctness-neutral, uncommon (pointwise inputs are usually
  contiguous activations; outputs always contiguous). A stride-aware reorder / seeded loop_orders is a
  FUTURE BROADEN (new lever, measured justification), not a current gap. (completeness 3rd pass.)

═══════════════════════════════════════════════════════════════════════
## NEXT ACTION  (champion 268edf3b; family DRY at a justified steady state; never-stop §6.0)
═══════════════════════════════════════════════════════════════════════
The pointwise family is COMPLETE: built, gated (A/D/F/H/R/E all PASS; Gate D after 3 faithfulness
fixes), reported, and overtime-validated (refactor-critic simplify; robustness correctness; accumulator-
negative; dtype-faithful; warp-ramp + flatten_loops measured-DEFER; tall-skinny BROADEN). Completeness
loop-until-dry: pass2 found+fixed tall-skinny, pass3 DRY. seed = oracle = tc parity ≈ HBM ceiling on the
corpus + held-out dyt; all aspect ratios + dtypes handled; robust on edge shapes.

What remains (all LOW-VALUE / human-review, logged above): MIN_WAVES audit; per-dtype TILE_BYTES;
stride-aware loop_orders BROADEN (transposed inputs, pipeline-inherited); tall-skinny + oversized
curriculum shapes (task pins corpus → human-review). None high-value; all need measured justification
the pinned curriculum can't supply.

This is a clean banked checkpoint (§6.1 proactive-recycle point): the durable log (NOTEBOOK/ledger/
REPORT) IS the state; a fresh context resumes from here. A FRESH-CONTEXT continuation could: (a) pursue
the stride-aware loop_orders BROADEN with its own micro-curriculum; (b) scout a NEW heuristic family
(chunked-scan/SSM per [[ws4-family-ranking]]) if the human widens scope. Do NOT re-read TEST/dyt (Gate E
firewall spent). NEVER push (human assembles the PR from the 4 helion files @ 268edf3b).
