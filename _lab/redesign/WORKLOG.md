# Reduction heuristic redesign — WORKLOG

> Gated log of work on the two-stage reduction-seed redesign. Every entry is a VERIFIED fact
> (measured / read from code / config-diffed), anchored on an immutable SHA where possible.
> Design lives in `/home/dev/local/prompts-lab/reduction-generality/PROMPT.md` (read §0 first).
> This file is the respawn-reconstruct artifact: baton + ledger + notebook in one.

## Fixed coordinates (verified)
- **Worktree:** `/home/dev/local/helion-redesign`, branch `reduction-redesign`.
- **Base SHA:** `de2c545f` = pristine `fc1dbaa0` `helion/` (`git diff fc1dbaa0 -- helion/` EMPTY, verified) + `_lab/` infra.
- **Interpreter:** `/home/dev/helion/.venv/bin/python`. Run scripts from `cwd=/tmp` with
  `PYTHONPATH=/home/dev/local/helion-redesign`; assert `helion.__file__` is under the worktree at top of every script.
- **Probe kernels:** `/home/dev/local/prompts-lab/reduction-generality/kernels/<slug>/kernel.py` (16 dirs: p1-p11, oos1, oos2, defect1, defect2, defect3). 13 stress = p1-p11 + oos1 + oos2.
- **Baseline configs:** `_lab/unify/baseline_fc1dbaa0_configs.json` (447 cells, frozen).
- **Config recorder:** `_lab/harness/unified_config_recorder.py` (the byte-identical / changed-cell oracle).
- **Heuristic code:** `helion/_compiler/autotuner_heuristics/triton.py` (reduction classes @ 1107–~1418).
  Stage-1 fact-building: `helion/_compiler/device_ir.py`. Fact structs: `helion/autotuner/config_spec.py:88+`.

## Hard rules (from PROMPT §0 + CLAUDE.md + memory)
- Commit after each green step (immutable SHA). config-diff EVERY edit. Never `pip install` / `git push` / `git commit` unless asked... NOTE: PROMPT §0 says "commit after each green step" — that IS the standing instruction for this task, so commits to `reduction-redesign` are authorized.
- GPU foreground-serial, NEVER detached/backgrounded (memory: detached GPU job → silent 13h stall + no notify).
- After each PHASE: run `test_reductions.py` + `test_autotuner_heuristics.py` + matmul-epilogue tests in `test_examples.py` green.
- Per probe kernel acceptance = BOTH (1) fired-right-path AND (2) perf ≥ default.
- Don't break `TritonMatmulReductionEpilogueHeuristic` (§2.9): keep its single-full-extent-reduction guard; don't leak the `>=1` relaxation into it.

## Phase status
- [x] P0 — runway verify + RED baseline of 13 probes (DONE @ commit P0; see log)
- [ ] P1 — Stage-1 categorizing fact-builder (GATE: zero config diffs)
- [ ] P2 — Stage-2 cap-set + greedy allocator (GATE: within 10% of champion)
- [ ] P3 — delete special cases, ordered (Defect-1 re-key BEFORE deleting per_feature_accumulator)
- [ ] P4 — probes GREEN + two-check verify

---

## Log

### 2026-06-29 — P0 DONE
- Verified HEAD `de2c545f`; `git diff fc1dbaa0 -- helion/` empty (pristine base confirmed).
- Read PROMPT.md fully (§0–§7). Read all heuristic code: `triton.py` reduction classes (Pointwise@304,
  reduction base@454, Standard@1107, UserTiled@1284, MatmulEpilogue@1421), `config_spec.py` ReductionFact@88,
  `device_ir.py` fact-builders (ReductionRole@99, classify@1098, register_unrolled@1132, build_reduction_facts@1313,
  assemble@1542, phase orchestration@3459-3515). Registry: `autotuner_heuristics/__init__.py` (order: matmul-epi,
  splitjoin, standard, usertiled, pointwise).
- **GATE INSTRUMENT VALIDATED:** config recorder reproduces frozen baseline **ZERO-DIFF, 447/447 byte-identical**
  on pristine base. (`_lab/redesign/p0_repro.json` vs `baseline_fc1dbaa0_configs.json`.)
- **Probe corpus = 13 stress kernels** = p1-p11 + oos1 + oos2. (`defect1/2/3` dirs are TODO.md placeholders, NOT
  kernels — manifest §2.8 confirms 13 = p1-p11+oos1+oos2.)
- FIXED portability: oos1 hardcoded `/home/dev/local/helion-unify/examples` → now derives examples/ from active
  `helion.__file__`. (per portable-lab-state rule.)
- RED baseline recorded: `_lab/redesign/probe_red_baseline.json` (recorder `_lab/redesign/probe_recorder.py`,
  reusable as GREEN recorder later). All 13 bind without crashing. Diagnoses:

| probe | fired today | n_red | RED symptom (what the redesign must fix) |
|---|---|---|---|
| p1 outer-product-coresident | **triton_pointwise** | 0 | co-resident 2 FULL_SLICE over grid-tile axes → both FLOORED_ROW → no red fact → **mis-seen as pointwise** |
| p2 feature+rowaccum (Defect-2 witness) | triton_reduction_tile | 1 | fires standard via *secondary* path, `pfa=False`. graph_ids `{1:[0,1]}` (co-resident). bs=[2048] |
| p3 full-grid-nonquant | triton_reduction_tile | 1 | prim=2 (group axis) sh=128 FULL_GRID, m_block=[0,1]. bs=[32] — grid sibling widened (looks OK-ish) |
| p4 two-rollable-sequential | triton_reduction_tile | 1 | TWO rollable reductions but **only 1 fact** (graph_ids `{1:[0,1,2,3]}` — both rdims rolled to 1 fact?); bs=[8,8] |
| p5 3d-reduction-tile | triton_pointwise | 0 | reduced inner axes not seen as reduction → **pointwise** bs=[1,32,64] |
| p6 mixed-coresident+seq | triton_reduction_tile | 1 | only 1 fact for the mixed shape; bs=[2048,8] |
| **p7 gridtile-then-usertile** | **[] NONE** | **2** | **`==1` gate DECLINES a real 2-reduction kernel → falls to default.** graph_ids `{1:[0,2],3:[1,3]}` (2 seq groups). Headline witness for `>=1` relaxation. |
| p8 fullgrid+usertile | triton_reduction_tile | 1 | prim=3 (K) FULL_SLICE, group FULL_GRID dropped. bs=[1] rl=[128] |
| p9 nonred-loop-then-fullextent | triton_reduction_tile | 1 | bs=[1,4096] rl=[None] (nrl handling) |
| p10 usertile+gridtile | triton_reduction_tile | 1 | only 1 fact (one declined); bs=[4096] |
| p11 fullextent-then-nonred-loop | triton_reduction_tile | 1 | bs=[1,4096] rl=[None] |
| oos1 jagged | [] NONE | 0 | correctly DECLINED (✓ expected) |
| oos2 strided-dim0 | triton_reduction_tile | 1 | fires (the known cliff, left as-is); bs=[4] w=16 |

- **KEY RED themes:** (a) the `==1` gate (p7 declined, p4/p6/p10 collapse multi→1 fact); (b) co-resident reductions
  over grid-tile axes mis-classified FLOORED_ROW→dropped (p1) or kernel seen pointwise (p1/p5); (c) `pfa` recognizer
  absent off the 4 norm-bwds (p2). The redesign's first-class N-reduction fact + graph_id co-residency + positive
  taxonomy must turn these GREEN.
- **NOTE on p4:** two rollable reductions over different axes landed in ONE fact (graph_ids all 4 graphs). Per §2.7
  invariant "two rollable reductions are NEVER co-resident" → they should be 2 sequential facts. Investigate in P1
  (may be the `_rollable_reduction_records` stashing one rdim, or both rolled into one). Flag, don't block P0.
