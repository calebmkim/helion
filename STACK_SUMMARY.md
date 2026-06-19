# 3-STAGE REDUCTION-HEURISTIC STACK — summary

**Branch:** `reduction-3stage-stack` (worktree `helion-3stage`). **NOT pushed** (remotes inverted:
origin=pytorch/helion, fork=calebmkim/helion; the human assembles/pushes the PR).

**Base SHA: `0676dd32`** — `reduction-seed-heuristic` = the merged PR #2762 contents (local `main`
`82bee2fa` + `a96a53f4` reduction-fact layer + `0676dd32` reduction seed heuristic). `origin/main`
(`db78b17a`) did **not** yet contain the merge, so per the driver's explicit fallback this local
branch IS the "current main + merged reduction seed" base. Box: 4×H100 (sm90), L2 50 MB, conda env
`helion`.

## Commit ranges → stages (contiguous, non-straddling; `[stageN-lab]` = lab/log/report only, excluded
from the eventual PR which is the `helion/` + `examples/` changes)

| stage | PR code commits (helion/ + examples/) | lab commits | what landed |
|---|---|---|---|
| **Stage 1 — liveness chunk** | `48ac703e` (Part A), `e99ade11` (Part B), `26427709` (doc/format), `3c239777` (welford m_block cap + fp32 stats), `440e8717` (fold carried-cap into `_reduction_rblock`) | `010ebf05`, `b003d7ee`, trailing `e0d98abb` + `28672138` (post-hoc records) | liveness-aware persistent decision; welford huge-M m_block persist-cap (+13% fp32) + fp32 stat accumulation (bf16/fp16 accuracy fix) |
| **Stage 2 — unify M-reduction** | `afac0b81` (A: materialized recognizer), `9bc8757c` (B: bias_grad M-collapse), `c8429c85` (C: multi-materialized), `5f34e9d1` (faithful `per_feature_accumulator` + dyt fix), **`883aa50f` (D: dual-axis M-collapse perf recovery — 2026-06-19)** | `223bc3cd`, **`e0c293d6` (D)** | materialized + M-collapse recognizers (m_reduction unnecessary) **+ rms/ln/instance/group backward recover old m_reduction perf** |
| **Stage 3 — matmul+epilogue** | `ea5cf150` (composed fact + heuristic), `000b02db` (dtype-aware M_BLOCK overtime) | `6749108d`, `bee15408`, `8d738eae`, `51045a1b`, `64222faa` | the first composed fact + heuristic |

(`git log --oneline 0676dd32..HEAD` is the full stack. To cut the 3-stage PR, split on the
`[stage1-*]` / `[stage2-*]` / `[stage3-*]` prefixes; drop the `_lab/` paths. NOTE: the two trailing
`[stage1-lab]` post-hoc commits (`e0d98abb`, `28672138`) sit physically after Stage 3 — they are
**lab-only** (no `helion/` changes) so they don't affect the code split. SHAs `ea5cf150`→`28672138`
were regenerated when sub-problem D was inserted at the end of Stage 2; backup of the prior tip:
branch `backup-pre-mcta-insert`.)

## HARD INVARIANT — held end-to-end (with ONE intentional, benchmarked exception: welford)
**8 of the 9** standard reduction kernels (rms_norm, layer_norm, softmax, sum, long_sum,
cross_entropy, kl_div, jsd) are **byte-identical from BASE → Stage 3**. **welford is intentionally
changed in Stage 1** by the welford commit `3c239777` (huge-M `m_block` persist-cap + fp32 stat
accumulation): config_recorder over the full active matrix (× fp32/bf16/fp16 × train/val/robustness =
739 cells) shows **exactly 15 changed cells, all welford**; the other 8 standard kernels and the 8
Task-1 transfer kernels are **0 changed**. The welford change is benchmarked, not a regression — fp32
**+13% geomean (up to +37.5%)** at huge-M, and it **fixes** the pre-existing bf16 (0.082→0.0156) and
fp16 (NaN→0.0020) accuracy failures at wide reduction chunks. The `m_block` cap is full-width-only and
is a **provable no-op** for the Stage-2 backward kernels (bias_grad/dyt are scalar-output; group_norm/
instance_norm are small-M so `m_floor=1`, `fires=False`) and Stage 3. Every other stage-to-stage diff
is 0. Backup of the pre-welford stack: branch `backup-pre-welford-rebase`.

## Per-stage status

**Stage 1 — liveness-aware chunk decision (DONE, gated D/R/H/A/F).** Part A refactors `_reduction_rblock`
into one budgeted `(r_block, persistent)` decision (zero config diff). Part B adds a walker-fact
liveness signal `ReductionFact.body_live_tiles` (peak simultaneously-live rdim-shaped tiles, computed
once in the collect pass) — the standard track routes a heavy reduction body (fused_linear_jsd's ~7
live full-width fp32 tiles) from persistent `[None]` to looped `[8192]` via a `LIVE_PERSIST_BUDGET`
spill ceiling. **Headline:** `fused_linear_jsd` narrow-V flips persistent→looped, **2.3–3.4× over the
old persistent seed**, beats grad-fair torch.compile on 4/5 shapes (G 1.17–1.76); n_spills 480→0
(byte-identical result tile); 9 curriculum + 8 transfer untouched. Gate F found the flip couples
`load_eviction_policies` (recorded). All gates pass.

**Stage 2 — unify the backward M-reduction kernels into the standard/user-tiled track (DONE, gated
D/H/F/R/E).** A PORT, not a deletion — `m_reduction` never exists on main (grep-clean). (A) the
materialized-feature recognizer routes rms/ln/instance backward to standard T1 `[1,1]` (0.14→0.90,
0.15→0.67, 0.03→0.48 — blows away the catastrophic default); (B) a hill-climbed M-collapse occupancy
seed makes bias_grad good on the user-tiled track (geo 0.35→0.94, beats default; dyt excluded by
num_load=1, already beats its default); (C) the multi-materialized recognizer fires group_norm
(0.04→1.16–1.89 large-N). **Every former-m_reduction kernel now beats its default via the standard or
user-tiled track**, so a bespoke M-reduction heuristic is unnecessary. Wide-S group/instance shapes
characterized as kernel-authoring-bound (the `[32,32]` default fails to compile there; the seed still
beats it). All gates pass. **(D — 2026-06-19, commit `883aa50f`) The vanilla-T1 perf gap to old
m_reduction is now CLOSED for the 4 dual-axis backward norms.** An M-collapse occupancy lever
(grid M block → `next_pow2(grid_rows/num_sm)`) + inner byte-cap + narrow-w1 bypass in the standard
MATERIALIZED branch, gated on `per_feature_accumulator` (+ `is_materialized` + `full_width_output` +
a non-grid inner tile), lifts fp32 G=tc/seed on N≤4096 to **rms 0.90→1.49 / 0.72→1.81, ln 0.68→1.49 /
0.49→1.13, instance 0.67→1.65 / 0.43→1.13, group 2.45→2.82 / 1.42→1.76** — matching/exceeding old
m_reduction. The kernels already had the two-level tiling (`register_block_size` + inner `hl.tile`), so
it is heuristic-only. Wide-N (N=8192) neutral. **Invariant held: 739/739 byte-identical** (gate
disjoint — `per_feature_accumulator=False` for all 9 standard + 8 transfer); bias_grad/dyt unchanged.

**Stage 3 — composed-fact seed for matmul + reduction-epilogue (DONE, gated D/F/H/A/R/E — the genuine
hill-climb).** The first COMPOSED fact: `MatmulWithReductionEpilogueFact` holds a MatmulFact + the
over-output epilogue ReductionFact (registered by relaxing the matmul guard so the Stage-2 materialized
branch picks up the `.sum(-1)` on the register-resident accumulator). `TritonMatmulReductionEpilogue-
Heuristic` emits a footprint-aware tile (M_BLOCK sized to the `[M_BLOCK,N]` fp32 accumulator budget,
num_stages=3). **Headline:** seeded_vs_default **1.3–2.4×**, seed ≈ grid optimum (0.96–1.00), **beats
torch.compile 1.5–1.73×** (driver-run repro), across 7 epilogues (rms_norm/layernorm/softmax/
l2_normalize/sum FIT + logsumexp/max HELD-OUT) and the held-out test shape — the FACT is blind to which
reduction it is. Disjoint: composed fact fires only on the fused family (pure matmul/reduction
unchanged). Gate D proposed a §2 doctrine clarification for the "composed fact" category. All gates
pass.

## Overtime (post-DoD, completeness-critic-driven — method §6.0 keep-climbing)
- **Stage 3 fp32 dtype fix (commit `c265d421`)** — the dtype gap the task flagged. The M_BLOCK
  ceiling was dtype-blind; at fp32 the seed emitted [64,32] and the 2×-bigger fp32 operand spilled
  (svd 0.53, seed/best 0.34 = 3× slower than grid best). Fixed by scaling the row ceiling by the
  input itemsize (a register-budget FACTOR, not a dtype literal): fp32 svd 0.53→1.51, seed/best
  0.34→0.997; generalizes (matmul_sum fp32 svd 1.44); bf16/fp16 unchanged; 9 curriculum byte-identical.
- **Loss corner (131072,512,1024)** — seed svd=2.87 over default, seed/best=0.997; does NOT regress
  (it matches Helion's best, which loses to tc only because the small-N fusion win has narrowed — the
  task's predicted boundary, a codegen ceiling not a seed failure).
- **Correctness backstop** (the suite no stage had run): `test_autotuner_heuristics` (24 passed),
  `test_matmul_heuristics` (passed), `test_reductions` (28 passed) — the new liveness sweep,
  recognizers, and composed fact don't break existing tests.
- **Completeness critic** re-verified the hard invariant 0/739 across every hop, and the group_norm
  TEST split (recognizer generalizes: large-N 1.21 beats default; wide-S kernel-authoring-bound).
- **Stage 2 m_reduction perf recovery — now DONE** (sub-problem D, commit `883aa50f`, 2026-06-19):
  the M_CTA-occupancy + inner-byte-cap recovery, re-opened by the human. Deliberately uses the
  `AccumulatorFact`-derived `per_feature_accumulator` signal the original run was told to defer. fp32
  G=tc/seed 0.49–0.90 → 1.07–2.82 on N≤4096; 739/739 byte-identical. Residual: wide-N (N=8192) neutral
  (m_reduction's HBM-bytes warp ramp would close it); full curriculum sweep before landing.
- **Logged, not chased (task-deprioritized):** Stage 1 flj bf16 V=50257 wide-V tail (chunk [2048] →
  ~1.15 vs current ~0.8; a secondary bonus the task says is not a loop target) — real headroom left
  as future overtime with its re-add recipe in the per-stage notebook.

## Reports & logs
Per-stage report + notebook + ledger (gate verdicts AS-RETURNED): `_lab/stage1_liveness/`,
`_lab/stage2_unify/`, `_lab/stage3_epilogue/` (`REPORT.md` / `NOTEBOOK.md` / `ledger.json` +
`gate_*_verdict.json`). Lab infra ported onto the merged base in `010ebf05`.
