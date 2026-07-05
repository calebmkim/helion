# H100 matmul-seed hill-climb — NOTEBOOK (source of truth on resume)

Branch: `matmul-h100-seed` (off `reduction-4pr-stack` @ abc10a00). GPU: pin ONE H100
(`CUDA_VISIBLE_DEVICES=0`), all GPU work serial. Interpreter:
`/home/calebkim/.conda/envs/helion/bin/python`. Scripts cwd=/tmp, PYTHONPATH=worktree.

Task: `prompts-lab/tasks/matmul-h100-task.md`. Key = literal shape (M,N,K,width_bits∈{8,16,32}).
FIREWALL: main agent climbs TRAIN only; VAL/TEST via referee subagent ONLY.

Harness: `_lab/matmul-h100/` — bench.py (probe+cold-L2 co-bench), sweep.py (multi-shape jobs),
autotune_one.py (oracle). Floor = beat default (structural, catch-all). Target = within ~10% of
pooled ceiling. tc = roofline column (fp8 tc=_scaled_mm; fp32 tc=fp32-no-TF32).
**FOOTGUN (cost me a sweep): ALWAYS set num_warps+num_stages on manual candidates — the s1
default leaves the MMA un-pipelined (3-7x slower); a bare block_sizes candidate is meaningless.**

---

## DELIVERABLE COMPLETE — HEAD 0162291a (heuristic frozen @945ed729; tests+golden on top)
Status: DoD EXCEEDED + overtime + fully validated + tested. Branch matmul-h100-seed, 10 commits
off abc10a00. tree clean (only unrelated examples/cross_entropy.py uncommitted, per instructions).
- Product A (no-autotune seed): TRAIN all-above-floor (geomean G bf16 0.959/fp16 0.956/fp32 1.465/
  fp8 0.995; mamba 4.5-7.7x default); VAL 13/13 within 10% ceiling; TEST 25/25 beat default, 23/25
  within 10% (both stress-corner gaps CLOSED in overtime).
- Product B (seeded search): real quick-autotune of 4096^3 converged to the primary seed in 19s.
- Tests: 155 pass + new TestTritonH100MatmulHeuristic + 4 examples pass; ruff clean.
- REMAINING (LOW-VALUE, for a resuming context): (1) full Product-B convergence comparison
  seeded-vs-unseeded (secondary per task, expensive); (2) 3072^3 non-pow2 at-floor 0.754 (oracle
  0.86 = [128,128,32]w4, shape-specific, can't generalize from 1 pt — hard-pile); (3) tiny-M (M<=4)
  decode ~2-3% within noise. No high-value per-shape climb remains (all realistic shapes at/near the
  Helion ceiling; Helion structurally loses bare-GEMM to cuBLAS). Maintain via BROADEN/refactor queue.

## POST-DELIVERABLE EXTENSIONS (HEAD 761dcbfb)  — lineage 0162291a → 091bb56a → 761dcbfb
1. **num_stages refactor (091bb56a)** — folded the dangling "step 7" saturated num_stages cap into
   step 3'. It was an artifact of incremental dev (the cap was added @9cc6cfa8 before step 3' was
   rewritten to SMEM-maximize @19b46d84, so num_stages got assigned twice). Now decided ONCE in 3':
   num_stages = deepest pipeline fitting SMEM, ceiling = `2 if saturated_batched else MAX_STAGES(6)`.
   BYTE-IDENTICAL across all 45 TRAIN seed[0] (before/after config-recorder diff); pure readability.
2. **Batched-matmul expansion + pin-batch-to-1 (761dcbfb)** — new precondition
   `_batched_static_matmul_fact`: fire on ANY static matmul whose extra tunable axes are all
   batch/outer GRID axes (pinnable to 1), not just the 2-D case. Size the dot's M/N/K by the budget,
   **pin every batch axis to 1** (build_block_sizes already floors non-M/N/K). The pinned grid then
   drives the saturation levers → a batched dot and mamba are the SAME case. Rationale: batch is a
   no-reuse parallel axis (one CTA/batch maxes the grid) AND the fp32 acc is [batch,bm,bn] so the
   bm*bn budget REQUIRES batch=1 (block_b>1 measured = 50x cliff / OOM). Now fires on bmm (baddbmm
   3-D dot) + matmul/fp8/broadcast/mamba; declines dynamic/jagged (grouped_gemm: static_shapes=False).
   VALIDATED: matmul/mamba/fp8 TRAIN BYTE-IDENTICAL (45, purely additive); bmm 8-30x over default,
   G vs torch.bmm 0.84-0.95 on 5/6 (4x2048^3 seed within ~7% of Helion ceiling; cuBLAS-batched
   structurally unreachable like bare GEMM). B200/skinny untouched (_single_2d_static_matmul_fact
   unchanged). New test test_fires_on_batched_dot_and_pins_batch_to_one; 33 heuristic tests + golden
   + bmm/broadcast examples pass; ruff clean. NOTE: pointwise epilogues (cast/relu/bias/gelu/scale)
   keep a plain MatmulFact -> the heuristic fires (verified); reduction epilogues (rms/layernorm/
   softmax over output) route to triton_matmul_reduction_epilogue instead.

## CURRENT HEURISTIC STATE  (commit 3071909b) — DoD EXCEEDED + overtime
Champion 3071909b. State: TRAIN all-above-floor (geomean G bf16 0.959 / fp16 0.956 / fp32 1.465 /
fp8 0.995; mamba 4.5-7.7x default); VAL 13/13 within 10% ceiling (3 referee passes); TEST 25/25
beat default, 23/25 within 10% (Gate E @1ebf4bbc) with both stress-corner gaps CLOSED in overtime.
Overtime since DoD: sat batched-dot tile cap (59cc706d), SMEM-max num_stages<=6 (19b46d84), l2 gate
BROADEN 8x->3x (3071909b, +3-5% on TRAIN tall shapes, VAL-verified). levers_since_refactor=3 (refactor
due ~4-5). REMAINING (low-value): 3072^3 0.754 at-floor (oracle 0.86 shape-specific, hard-pile);
tiny-M decode ~2-3% within noise. NEXT: refactor-critic (cadence) -> then climb is at diminishing
returns (everything realistic at/near ceiling); maintain via BROADEN/refactor queue + proactive recycle.

## SUPERSEDED STATE  (commit 19b46d84)
Champion 19b46d84. Levers (all faithful, gate-checked): register-budget wide-N tile + spill-outward;
saturated batched-dot occupancy tile cap (bm<=64,bn<=128 when pinned_grid>=4*num_sm); wave-quant
occupancy fill; SMEM+pipeline-capped block_k; num_warps ramp; SMEM-maximized num_stages (<=6, capped
to 2 for saturated batched dots); tall-grid l2_groupings; ranked multi-seed alternates.
TEST (Gate E @1ebf4bbc, single read): 25/25 beat default, 23/25 within 10% ceiling. 2 stress-corner
gaps since CLOSED in overtime (general fixes, validated on TRAIN+constructed shapes, NOT TEST-tuned):
- gap#2 large fused mamba dot -> saturated tile cap (59cc706d): [128,256]->[64,128] +12-13%.
- gap#1 deep-K K>>M.N -> SMEM-maximized num_stages (19b46d84): G 0.49->0.62-0.77 (one above floor).
Refactor-critic ran @c82d2ab4 (dead code/annotation/simplify). levers_since_refactor=2.
NEXT: full TRAIN re-bench @champion (running bcnhhcwoy) -> VAL referee re-check (regression on
held-out from the 2 overtime edits) -> update REPORT -> keep climbing (3072^3, simplify).

## PRIOR HEURISTIC STATE  (commit 9cc6cfa8)
Levers added since 521a6261:
- tall-grid l2_groupings=[2] when grid_m>=8*grid_n (7a0bc7f5): rescues tall-skinny 0.69->0.97;
  reversal boundary measured (wide/square regress). tensor_descriptor from oracle was INERT (dropped).
- grid-saturation num_stages cap: <=2 when pinned_grid>=4*num_sm AND tile<32768 (9cc6cfa8):
  mamba fused dot 24%-off -> ceiling (VAL-referee-diagnosed). Keyed on PINNED grid not total tiles
  (3072^3 bare GEMM pinned=1 keeps s4; s2 there = G0.58 disaster). xD mamba 4.1-5.0 -> 4.2-7.4.
VAL referee #1 (@7a0bc7f5): 13/13 beat default; 10/13 within 10% ceiling; sole gap = mamba s4
(now fixed @9cc6cfa8). fp8 cross-transfer perfect. Re-check pending.

## PRIOR HEURISTIC STATE  (commit 521a6261)
TritonH100MatmulHeuristic, fires on every clean 2-D static MatmulFact (sm90), promote=False.
Budget formula `_h100_matmul_tile(M,N,K,itemsize,num_sm,pinned_grid)`:
- [bm,bn]: register-budget ACC_BUDGET=32768 fp32-acc elems, base wide-N [128,256], clamp to
  shape (pow2), spill-outward, wave-quantization occupancy fill (counts pinned grid axes;
  shrink larger dim while wave_eff<0.8).
- bk: largest pow2 <= min(BK_CAP=256, K/PIPE=K/4) fitting [bm,bk]+[bk,bn] in SMEM(228K) at
  num_stages=4 (width via itemsize). Deep bk for small-M large-K; K/4 cap keeps small-K (mamba) shallow.
- num_warps = 8 if bm*bn>=16384 else 4 (LOAD-BEARING: w4 on big tile = 7x slower).
- num_stages = 4, dropped only if min_bk overflows SMEM (LOAD-BEARING: s1 = 3x slower).
matmul_h100.json overrides = EMPTY (formula is sole catch-all so far). levers_since_refactor = 0.

## PER-(KERNEL,DTYPE) GEOMEAN (G=tc/seed; worst-kernel-first triage)  [TRAIN, seed @521a6261]
- matmul bf16 : ~0.93 geomean (range 0.69 tall-skinny .. 1.055 decode-M32); 2 below-floor shapes
- matmul fp16 : ~0.935 (tracks bf16 — 16-bit merge confirmed)
- matmul fp32 : ~1.46 (Helion TF32 beats tc fp32; 1.03-1.75)
- fp8_gemm    : ~0.99 (0.877 .. 1.095; beats/matches _scaled_mm) — BEST band
- mamba fp16  : ~4.6x over default (no cuBLAS analog; cross-kernel transfer CONFIRMED)

## PER-SHAPE STATUS (matmul bf16 unless noted; seed=formula @521a6261; G vs tc)
TRAIN — at/above ceiling (G>=0.88), DONE:
  4096^3 .937 | 8192^3 .935 | 8192,4096,4096 .908 | 4096,4096,11008 .991 | 4096,11008,4096 .959
  4096,4096,14336 .933 | 4096,14336,4096 .958 | 8192,8192,28672 .954 | 4096,4096,12288 .952
  8192,5120,5120 .927 | 4096,4096,128256(vocab) .945 | 2048,4096,32000(vocab) .992
  512,4096,16384(wide) .988 | M=1 1.035 | M=8 .986 | M=32 1.055 | M=128 .882 | M=256 .97
  fp16 4096^3 ~.937 | fp16 4096,4096,14336 ~.933
  fp32 2048^3 1.734 | fp32 4096^3 1.747 | fp32 M=8 1.028
  fp8 2048^3 1.095 | fp8 4096^3 1.045 | fp8 8192^3 .963 | fp8 8192,4096,4096 .94
  fp8 4096,4096,11008 .949 | fp8 4096,14336,4096 1.058 | fp8 4096,4096,14336 1.018
  fp8 4096,4096,28672 1.009 | fp8 M256 .979 | fp8 M8 .976 | fp8 M32 .877
  mamba(all 8): xD 4.1-5.0 over default
BELOW-FLOOR (G<0.75), oracle-in-progress:
  16384,8192,512 (tall-skinny) G=0.69  -> autotune running (bx6o5skdc)
  3072^3 (non-pow2) G=0.747            -> autotune next
  2048^3 bf16: seed=[128,256,64] not yet seed-vs-tc benched (assume ~.93) -> bench

## FROZEN-CHAMPION ANCHOR  (last banked freeze @521a6261)
- Set at first VAL-referee milestone. Current per-cell geomeans above are the provisional anchor.

## BANKED WINS (full lineage off abc10a00)
- 317f69c9: scaffold (formula+family+multi-seed loader+json), fires all 45 TRAIN.
- 521a6261: SMEM-budgeted pipeline-capped bk -> small-M G 0.46-0.84 -> 0.88-1.06; mamba chunk128
  [64,128,32] 13% faster; big GEMM unchanged. num_warps=8/num_stages=4 attributed load-bearing.
- 7a0bc7f5: tall-grid l2_groupings (rescues tall-skinny 0.69->0.97).
- 9cc6cfa8: grid-saturation num_stages cap (mamba fused dot 24% -> ceiling).
- c82d2ab4: refactor-critic pass #1 (dead code, return annotation, simplify num_stages gate).
- 1ebf4bbc: ranked multi-seed alternates (Product-B search diversity).
- 59cc706d: saturated batched-dot occupancy tile cap (closes TEST large-fused-dot gap).
- 19b46d84: SMEM-maximized num_stages (deepen to <=6 when SMEM allows; deep-K/small-M wins).
- 3071909b: BROADEN l2_grouping gate 8x->3x (refactor-critic; +3-5% on TRAIN tall shapes).
- 945ed729: multi-seed alt num_stages perturbs DOWN only (refactor-critic #1 fix).
- 484657b8 / 0162291a: TestTritonH100MatmulHeuristic + adapt skinny/golden tests (155 pass).
- 091bb56a: **num_stages refactor — fold the dangling step-7 saturated cap into step 3'**
  (assigned-once; byte-identical across all 45 TRAIN; pure readability).
- 761dcbfb: **batched-matmul expansion — _batched_static_matmul_fact fires on any static matmul,
  pinning batch/outer axes to 1** (bmm + matmul/fp8/broadcast/mamba; declines dynamic/jagged).
  matmul/mamba/fp8 byte-identical; bmm 8-30x over default, G 0.84-0.95 on 5/6 vs torch.bmm.

## TRIED-AND-REJECTED
- width-constant bk {1:128,2:64,4:32}: superseded by SMEM-budgeted bk (gave small-M only G~0.46-0.84).
- deep bk on mamba ([64,128,128]) : REGRESSED 356->413us; mamba wants bk=64 (small K). K/PIPE cap fixes.
- deep bk on tall-skinny: REGRESSED ([128,256,128]s2=472 vs seed 296); not the fix.

## BROADEN-AND-REFACTOR QUEUE
- (none yet — formula is already broad/catch-all; revisit after VAL.)

## DEFERRED-HARD-PILE-AND-BORDERLINE
- tall-skinny 16384,8192,512 G=0.69: oracle running to find ceiling / confirm weird.
- 3072^3 G=0.747: oracle next.

## HUMAN-REVIEW QUEUE
- mamba example uses @helion.kernel() (static_shapes=False) -> seed won't fire as-written
  (fact non-static). Needs static_shapes=True (matmul/fp8 examples already set it). 1-line, out of scope.
- fp32 win (1.7x) is Helion-TF32 vs tc-true-fp32: a precision difference (both pass fp32-exact
  gate within 5e-2 tol). Legitimate but note the TF32 caveat in the report.

## NEXT ACTION
1. Read autotune oracle for tall-skinny (bx6o5skdc) -> if a config beats seed 294us, fold into
   formula/override; else confirm weird (retarget floor 0.75x ceiling). 2. Autotune 3072^3.
3. Spawn VAL referee (milestone) for mechanistic diagnosis. 4. Continue to DoD + overtime.
