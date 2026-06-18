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

| stage | PR code commits (helion/) | lab commits | what landed |
|---|---|---|---|
| **Stage 1 — liveness chunk** | `48ac703e` (Part A), `e99ade11` (Part B), `26427709` (doc/format) | `010ebf05`, `b003d7ee`, `26427709` | a liveness-aware persistent decision |
| **Stage 2 — unify M-reduction** | `a90aa7b4` (A), `ef8b390f` (B), `faecdbfc` (C) | `6a4e091a` | materialized + M-collapse recognizers; m_reduction unnecessary |
| **Stage 3 — matmul+epilogue** | `d204b1a5` | `e6cf4670` | the first composed fact + heuristic |

(`git log --oneline 0676dd32..HEAD` is the full stack. To cut the 3-stage PR, split on the
`[stage1-*]` / `[stage2-*]` / `[stage3-*]` prefixes; drop the `_lab/` paths.)

## HARD INVARIANT — held end-to-end
The 9 standard reduction kernels (rms_norm, layer_norm, softmax, sum, long_sum, cross_entropy,
welford, kl_div, jsd) are **byte-identical from BASE → Stage 3**: config_recorder over the full active
matrix (× fp32/bf16/fp16 × train/val/robustness = 739 cells) shows **0 changed**, and every stage-to-
stage diff is also 0. The 8 Task-1 transfer kernels are non-regressed throughout. Every stage proved
this after each edit (config + `--triton` hash for fact/source edits = selection-only).

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
beats it). Vanilla-T1 perf short of old m_reduction is accepted (overtime). All gates pass.

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

## Reports & logs
Per-stage report + notebook + ledger (gate verdicts AS-RETURNED): `_lab/stage1_liveness/`,
`_lab/stage2_unify/`, `_lab/stage3_epilogue/` (`REPORT.md` / `NOTEBOOK.md` / `ledger.json` +
`gate_*_verdict.json`). Lab infra ported onto the merged base in `010ebf05`.
