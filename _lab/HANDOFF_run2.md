# HANDOFF — RUN 2 (read this FIRST). Triton reduction seed-heuristic, H100/fp32.

> Single entry point for the next agent. Run 2 is COMPLETE + gated + PR-ready (never pushed). This compresses
> everything; deep detail is in the docs listed in §1. ALL file:line refs are approximate — grep the symbol.

## 0. TL;DR
- **Worktree** `/home/calebkim/helion-new-heuristics/wt-reduction-2`, **branch** `reduction-heuristics-run2`,
  forked off the run-1 v8 tip `25561778` (branch `reduction-heuristics-autotuner`, worktree `wt-reduction`).
  **28 run-2 commits, NEVER pushed.** Working tree clean.
- **Heuristic** = `TritonReductionHeuristic` v8 + run-2 changes (welford Band-C re-derive, load-eviction
  seeding, pid='flat', multi-seed plumbing). 9 forward inner-reduction kernels.
- **Results vs v8:** in-sample **O 0.9786→0.998**; TEST **O 0.8628→0.946** (0.964 excl noise-floor);
  welford TEST 0.396→0.892; **welford prime-N (262144,1543) 0.082 (numerically WRONG) → 0.905 (correct+fast)**;
  in-sample↔TEST gap −0.115→−0.05.
- Every accepted change was independently **results-referee + adversarial-auditor** gated; a **capstone**
  auditor (pre-Phase-II) and a **Product-B** auditor both PASSed. 187 tests pass; ruff/pyrefly clean; diff vs
  v8 = 8 files, +527/−123 (shippable). 6 of the 9 kernels' seeds are byte-identical to v8 (only welford,
  sum, long_sum changed — welford via Band-C, sum/long_sum via eviction).

## 1. Where the knowledge lives (read in this order)
1. **`_lab/HANDOFF_run2.md`** (this) — orientation + traps.
2. **`_lab/FINAL_REPORT_run2.md`** — the deliverable report (headline, what changed, Product B, residuals).
3. **`_lab/run2_notebook.md`** — the live reasoning trace (decisions + empirical why + tried/rejected,
   incl. the eviction property derivation + 3b pre-registered hypotheses + the FIREWALL MAP).
4. **`_lab/ledger.json` key `"run2"`** — durable structured state: `run2_iteration_history`,
   `run2_gate_verdicts`, `eviction`, `codegen_knobs_other`, `in_sample_v2`, `product_A_O`, `product_B_3a`,
   `product_B_3b`, `TEST_reread`, `pr_readiness`, `goal5_generality`, `STATUS`.
5. **`_lab/HUB_LOG.md`** — dated run arc (run-1 then the `## 2026-05-30/31 (RUN 2 …)` sections at the bottom).
6. **RUN-1 reference (still in force):** `_lab/HANDOFF.md` (§4 TRAPS — matched-lever A/B, oracle-is-a-bundle,
   do_bench noise floor), `_lab/FINAL_REPORT.md`, `_lab/SETUP.md` (env), `_lab/step2_code_map.md`,
   `_lab/t2_code_map.md`, `_lab/codegen_knob_map.md`. **`wt-reduction` (run-1) is READ-ONLY reference; never edit.**
7. Raw Product-B autotune CSVs + matrices: `_lab/logs/run2/pb/` (63 CSVs); knob/evict/TEST raw JSON:
   `_lab/logs/run2/{evict_verify,knob_explore,TEST_reread,standardize_probe,ce_online_probe}.json`.
8. Curriculum: `/home/calebkim/helion-new-heuristics/local/list_of_kernels.md` (the in-sample-v2 split is
   appended there; a committed snapshot is `_lab/list_of_kernels_run2.md`).

## 2. ENVIRONMENT / WIRING — the #1 trap (re-prove before trusting any edit)
- **Interpreter:** `/home/calebkim/.conda/envs/helion/bin/python` (conda env `helion`; torch 2.12 dev cu128,
  triton 3.7). System python lacks deps.
- **4× H100.** GPU 0 has a ~1 GB co-tenant (idle util). GPUs 1/2/3 idle. **Pin `CUDA_VISIBLE_DEVICES` per
  job; NEVER run two TIMING runs on one GPU (corrupts do_bench).** Re-check `nvidia-smi` before trusting a delta.
- **CANONICAL INVOCATION** (run from `/tmp`, NOT a checkout root):
  ```
  cd /tmp && CUDA_VISIBLE_DEVICES=<1|2|3> \
    PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction-2 \
    /home/calebkim/.conda/envs/helion/bin/python <script>
  ```
- **THE PREFIX TRAP (silent no-op #2):** `helion` is an editable PLAIN-PATH install pointing at the ORIGINAL
  checkout `/home/calebkim/helion-new-heuristics/local/helion`. `PYTHONPATH=wt-reduction-2` makes `helion`
  AND `examples.*` resolve to the worktree. BUT `"wt-reduction-2".startswith("wt-reduction")` is **True**, and
  run-1 harness scripts hardcode `WORKTREE="…/wt-reduction"` AND do `sys.path.insert(0, WORKTREE)` — so
  reusing them verbatim **silently runs the OLD worktree's code**. EVERY run-2 script asserts
  `helion.__file__.startswith("/home/.../wt-reduction-2/")` and relies on PYTHONPATH only (NO `sys.path.insert`).
- **tritonbench** resolves to the ORIGINAL checkout (hardcoded meta-path finder; PYTHONPATH can't shadow).
  Operator-level edits (e.g. `torch_compile_<op>_default` baselines, already present from run-1) go in the
  ORIGINAL checkout's `benchmarks/tritonbench/.../operators/<op>/operator.py`. Pure-Python; no pip install.
- Proven at run-2 setup: helion+examples resolve to wt-reduction-2; a sentinel edit to `_num_warps` flowed
  into the live seed; tritonbench rms_norm operator resolves to original w/ `torch_compile_rms_norm_default`;
  bare-seed mechanism sound. Step-1 5-way sanity reproduced run-1 exactly (rms_norm 8192,8192: default 336 /
  seed 252.6 / tc-default 250.3 / tc-max 248.4 us). See ledger `run2.wiring_proof`/`step1_sanity`.

## 3. THE HEURISTIC (helion/_compiler/autotuner_heuristics/triton.py :: TritonReductionHeuristic)
`is_eligible` = `_triton_reduction_eligible` (gate: `len(reduction_facts)==1 and not matmul_facts`). Branches
in `get_seed_config` (ALL keyed on `ReductionFact` workload props, NEVER kernel identity — capstone verified
zero kernel-name literals in executable code):
1. **persistent-vs-looped**: persistent (`reduction_loops=[None]` T1 / `block_sizes[red]=np2(N)` T2) up to the
   Triton `max_tensor_numel` (2²⁰) structural cap. Multi-load extra perf cap: `num_load>=2 AND row_bytes >
   MULTILOAD_PERSIST_MAX_BYTES (131072)` → looped (fires for wide cross_entropy).
2. **num_warps ramp** `_num_warps(fact)` by rnumel: ≤1024→4, ≤4096→8, ≤16384→16, else 32
   (`STREAM_WARPS32_MIN_ELEMS=16384`).
3. **Band-B R_BLOCK cap** (T2, `num_tiled_accumulators>=1`: kl_div/jsd) → R_BLOCK ≤ `BANDB_R_BLOCK_BYTES
   (16384)/itemsize`.
4. **Band-C structured combine** (`fact.is_structured_combine`: welford, standardize) — RUN-2 RE-DERIVED to
   two INDEPENDENT byte caps (divisor + apply↔combine coupling DELETED): `combine = min(np2(N),
   STRUCTURED_COMBINE_CAP_BYTES(32768)/itemsize)` [persistent — looping it regresses via the serial
   recurrence]; `apply = np2(N) if n_valid*itemsize <= STRUCTURED_APPLY_PERSIST_MAX_BYTES(12288) else
   min(np2(N), STRUCTURED_APPLY_LOOP_CHUNK_BYTES(8192)/itemsize)`.
5. **load_eviction_policies** (RUN-2, `_eviction_policies(env, kind)`): kind `"stream"` (T1 `num_load==1`:
   sum/long_sum) → `['first']*len`; kind `"reread"` (is_structured_combine: welford) → `['last']+['first']*
   (len-1)`. Others (rms/ln/softmax/kl/jsd/CE) → DEFAULT (None). Built at EXACT `spec.load_eviction_policies.
   length` (NOT length-validated by normalize).
6. **pid_type='flat'** emitted explicitly in every branch (principled constant; behavior-neutral — verified
   generated Triton byte-identical with/without; run-1 matched-lever A/B rejected persistent pid).
7. **M-block** at the autotuner floor (`_m_block_size` / `_block_floor`).
8. **`get_seed_configs()`** (RUN-2 Goal 3b, opt-in via env `HELION_REDUCTION_SEED_PORTFOLIO`): base + warp
   {4,8,16,32} + eviction {none, all-last} + num_stages=2 variants (7 distinct for welford). OFF by default →
   Product-A/3a use the single seed unchanged.

Other heuristic-adjacent code:
- `helion/_compiler/autotuner_heuristics/registry.py`: base `get_seed_configs()` → None (default).
- `helion/_compiler/autotuner_heuristics/__init__.py`: `compiler_seed_configs` uses `get_seed_configs` when
  non-None else falls back to `[get_seed_config()]`; `dedupe_configs` at the end.
- `helion/autotuner/config_spec.py`: `ReductionFact` NamedTuple (docstring updated — divisor language removed).
- `helion/_compiler/device_ir.py`: `register_rollable_reductions` (T1), `register_user_tiled_reductions` (T2 +
  is_structured_combine gate ~L970: `len(non_grid_tiles)>1 and len(apply_tiles)>=1`), `_count_reduction_
  workload` (RUN-2: `_is_reduction_extent` helper with the int-equality `last==size_hint` caveat comment +
  symbolic-fallback symmetry at the dtype site).

## 4. SOURCE-KERNEL changes (examples/)
- **`examples/welford.py`** (Goal 1, ships): `Tn = chunk.size(-1)` → `Tn = (tile_n.index < n).sum()` (masked
  valid count; Helion masks OOB loads with other=0 so sum_x/sum_x2 were already correct — only the
  divisor/count was wrong). Fixes mean/var/count at non-divisor N. Validated correct at well-factored/odd/PRIME
  (1543) for BOTH the seed and the un-seeded default. THE pow2-divisor combine constraint is no longer needed.
- **`examples/standardize.py`** (Goal 5, NEW — generality probe): plain two-moment LayerNorm (sum_x + sum_xsq,
  then normalize) — a genuinely DIFFERENT combine PATH from welford's online recurrence. Fires
  is_structured_combine; the Band-C recipe transfers within 1.2% on every shape → resolves the auditor's
  "is_structured_combine fit to one kernel" flag. (Kept as a documented probe; layer_norm-equivalent workload,
  not wired into the shipped curriculum.)
- **`examples/cross_entropy.py`** (Goal 5, ADDED `cross_entropy_online` alongside the unchanged existing):
  single-pass online/flash logsumexp (running max + rescaled sum; reads the row ONCE — verified ONE logits
  load in the generated Triton). Closes the wide-vocab SOURCE ceiling: (8192,131072) **0.539→0.956**,
  (4096,128256) 0.580→1.042; CE in-sample geomean 0.917→0.975 (regime-best 0.995). KNOWN: −7% at V=65536
  (register-heavy recurrence; no clean fact to key a lower warp count — disclosed). Both variants ship.

## 5. PER-GOAL STATUS (all DONE + gated)
- **G1 welford** — DONE, both gates PASS. Source fix + Band-C re-derive. welford in-sample G(orig4)
  0.911→0.926; per-shape wins 1536 +6.7%, 2560 +12.2%; prime 0.082(wrong)→0.905. Commits ddf8fc34, 43492809,
  18bdab57. Validated: correctness (referee re-ran prime + non-pow2; auditor proved the SAME config is err≈0.67
  under OLD source vs 1.4e-6 under NEW). Confirmed the v8 welford ORACLE was divisor-confounded (corrected
  oracle picks non-divisor combine at non-pow2 N). Harness: `_lab/harness/run2_wf_{sweep,validate,knobs}.py`.
- **G2 codegen knobs** — DONE, both gates PASS. **WIN: load_eviction_policies** (separating property = per-load
  cache RESIDENCY, overturns run-1's "autotuner-only"). sum +29% / long_sum +16% geomean; welford 4096
  0.76→0.95. pid='flat' owned. **NULL (honest, recorded with raw A/B): num_stages, tensor_descriptor, range_***
  — no clean workload-keyed win; rms/ln "weak" in-sample-v2 shapes are sub-25µs NOISE-FLOOR (±25% same-config
  swing); TD is inert unless ≥2 rows in the leading tile (seed tiles rows at 1) and OOMs on welford. Commits
  136d187a, b9548c29, 45db9997. In-sample O 0.9786→0.998. Validated: matched-lever in-process A/B
  (`run2_evict_probe.py`/`run2_wf_knobs.py`); a full-curriculum verification agent (RULE A/B/C with noise
  rechecks); auditor confirmed eviction in generated Triton + 8 kernels byte-identical. Raw:
  `_lab/logs/run2/{evict_verify,knob_explore}.json`, `ledger.run2.eviction`/`codegen_knobs_other`.
- **G4 in-sample-v2** — DONE. Real-AI-workload shapes (Llama/GPT vocabs, hidden dims, small-M, non-pow2) in
  `list_of_kernels.md` + `_lab/list_of_kernels_run2.md` + `ledger.run2.in_sample_v2`. Firewall-validated
  disjoint from sealed TEST for the 7 non-re-read kernels (see the FIREWALL MAP in run2_notebook).
- **G5 generality** — DONE. structured-combine validated via `standardize` (the key flag); multi-load
  (CE+rms+ln) and Band-B (kl_div+jsd) shown already-multi-kernel; `cross_entropy_online` closes the source
  ceiling. Commits f13168e4, 5cb57cba, + the band-generality summary. Raw: `standardize_probe.json`,
  `ce_online_probe.json`, `ledger.run2.goal5_generality`.
- **G6 device_ir + TEST** — DONE. device_ir robustness (behavior-neutral). **Consolidated TEST RE-READ done +
  RE-LOCKED** (the two pre-authorized re-reads): welford TEST 0.396→0.892, prime 0.905; rms_norm TEST
  0.828→0.841 (0.992 excl noise-floor; now backed by a raw log). TEST O 0.863→0.946. Harness:
  `_lab/harness/run2_TEST_reread.py`; raw `_lab/logs/run2/TEST_reread.json`; `ledger.run2.TEST_reread`. Commits
  c07d9751, 58724e98.
- **G3a Product B budget reduction** — DONE, auditor PASS. seeded-QUICK matches unseeded-FULL optimum within
  0.1–0.9% on **5 kernels across all bands** (welford/kl_div/rms_norm/cross_entropy/softmax) → budget full→quick.
  Convergence DRAMATIC where the optimum is hard (welford/kl_div/rms: seed at-optimum gen0 vs unseeded
  115–380s), modest where easy (softmax 1.35×, CE 1.2–1.36×). Corroborates+exceeds run-1's 1.94×.
- **G3b beat-max-effort** — DONE = **HONEST NULL**, auditor PASS. Pre-registered pilot (welford 262144,4096
  hard eviction coupling + sum 2048,16384 Band-A control), N=5/arm, full budget: BOTH seeded-portfolio AND
  unseeded reach the optimum reliably (5/5); seed within 1.3% = at ceiling → no beat. A preliminary "beat" (an
  unseeded 6.117ms "miss") was an INCOMPLETE-DATA artifact (a mid-run autotune CSV's cumulative-min before
  convergence — the anti-lucky-run protocol of complete-N runs + fair re-bench CAUGHT it). Multi-seed plumbing
  ships (opt-in). `ledger.run2.product_B_3b`.

## 6. HARNESS SCRIPTS (`_lab/harness/run2_*` — the canonical run-2 tools)
- `run2_measure_g.py` — `measure(kernel,M,N)` → G_seed=tc_default/seed_lat (live seed, configs=[seed], no
  autotune), correctness-gated, codegen kind. The 9-kernel fn/arg/fp32-ref plumbing all other run-2 scripts import.
- `run2_seed_dump.py` — path-AGNOSTIC seed dump for all 9 kernels (run under both worktrees + diff to prove
  byte-identical).
- `run2_wf_sweep.py` (combine×apply×warps grid + oracle), `run2_wf_validate.py` (live welford seeds vs tc),
  `run2_wf_knobs.py` (welford eviction/indexing A/B).
- `run2_evict_probe.py`(+`2`) — general per-load eviction A/B (default/all_last/all_first/rule[/xonly_first]),
  maps slot→tensor from generated Triton.
- `run2_TEST_reread.py` — the read-once TEST re-read (welford+rms_norm; firewall-aware; noise-floor flagged).
- `run2_productB_driver.py` — ONE cold-cache autotune (seeded|unseeded, quick|full) → per-gen CSV;
  `run2_productB_matrix.sh KERNEL M N GPU OUTDIR NREPS` — the matrix runner; `run2_productB_analyze.py` (3a:
  budget gap + convergence to 95/99% on gen + wall-clock axes); `run2_productB_3b_analyze.py` (3b: best-of-N /
  median / spread / P(reach optimum) per arm).
  UNSEEDED = `HELION_DISABLE_AUTOTUNER_HEURISTICS=1`; cold cache + `HELION_FORCE_AUTOTUNE=1`; budget via
  `HELION_AUTOTUNE_EFFORT={quick,full}`; per-gen CSV via `HELION_AUTOTUNE_LOG`.

## 7. TRAPS / GOTCHAS (run-2 specific; run-1 §4 also in force)
1. **The prefix trap** (§2) — pin `wt-reduction-2` exactly; no `sys.path.insert`.
2. **Incomplete-CSV trap** (NEW, caused a 3b false-positive): a Product-B autotune CSV read MID-RUN has a
   cumulative-min that is NOT the final best (it converges later). ONLY analyze COMPLETE runs; use complete-N
   + fair-process median-of-7 re-bench before any beat claim. `run2_productB_*analyze` parse only `status==ok`.
3. **Noise floor** — sub-25µs / tiny-M shapes swing ±25% on the SAME config. The rms/ln in-sample-v2
   "weakness" (0.71–0.88) was this; fresh median-of-7 shows them at parity. Lift M or flag/exclude.
4. **Eviction/list knobs** are NOT length-validated by normalize → build at EXACTLY
   `spec.load_eviction_policies.length` / `spec.indexing.length`. Verified the eviction seed survives the
   autotuner flatten/unflatten round-trip PRESERVED (so Product-B's seeded arm carries it).
5. **welford warps residual** — the rnumel ramp gives w16 at N=5120 but w8 is ~3.5% better (register-heavy
   combine); left simple (no regression). A possible future refinement, but no clean fact distinguishes it.
6. **Background-bash GPU access** — verified subagents DO have GPU access; but for trusted timing the hub used
   background bash on pinned GPUs (concurrency ≤ #idle GPUs). Parallel matrices ran one kernel per GPU.

## 8. HONEST RESIDUALS / NULLS (not bugs — attributed)
- cross_entropy wide-vocab source ceiling → CLOSED by `cross_entropy_online` (ships); existing CE untouched.
- long_sum few-row (4,524288)=0.66 — grid-starved (4–8 programs); needs split-K (cross-CTA) — wip-DEFERRED.
- welford wide-N (5120/8192)=0.69–0.81 + small-N — codegen-OOM (TD) / measurement-noise ceilings.
- num_stages / tensor_descriptor / range_* — no seedable win (recorded null with raw A/B + mechanism).
- 3b beat-max-effort — honest null on the tested couplings (autotuner thorough at full budget).

## 9. ORCHESTRATION (how the run was driven — no SendMessage in this harness)
Hub (main session) drove directly + maintained the durable `_lab/` state; spawned one-shot `Agent` calls as
results-referee / adversarial-auditor / perf-investigators / measurement workers (they HAVE GPU access);
heavy GPU sweeps ran as **background bash** pinned to distinct GPUs; analysis/decisions/commits by the hub.
**Acceptance independently gated:** nothing entered the champion without a referee re-run AND an auditor pass;
gates caught/corrected the welford-oracle confound, the eviction-vs-noise distinction, and the 3b
incomplete-data false-positive. Commit early/often; NEVER push.

## 10. WHAT'S NEXT (deferred = human scope decisions, not omissions)
- **Push** `reduction-heuristics-run2` (the human handles git push). Shippable diff: `git diff 25561778 --
  helion/ examples/`. `_lab/` is lab scratch (don't ship). The run-1 tc-default baseline patch is in the
  ORIGINAL checkout (see run-1 `_lab/harness/patches/`).
- **bf16/fp16 expansion** — the heuristic is dtype-general (all caps via `itemsize`; reads `dtype`); expected
  straightforward. Assert precision per call; softmax defaults to fp16 (override to test fp32).
- **Backward / outer reductions (Band D)** — DEFERRED; rms_norm_bwd/layer_norm_bwd weight/bias grads (sum over
  M) need their own fact/branch.
- **long_sum split-K / >2²⁰ looped tail** — the grid-starved few-row + the >2²⁰ structural tail; needs a
  split-K (cross-CTA + atomic/2nd-kernel combine) recipe — a separate workstream (the brief said: live with it
  but provably attribute it, which is done).
- Lower-priority/diminishing: more 3a kernels (pattern is robust across 5/all-bands), more 3b shapes (the
  hardest coupling already showed no beat at full — don't rabbit-hole), wire standardize into the benchmark
  path, a 3rd Band-B kernel.

## 11. REPRODUCE / VERIFY (canonical)
- In-sample O: `cd /tmp && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=…/wt-reduction-2 <py> /tmp/run2_O.py rms_norm sum
  long_sum layer_norm welford` (+ a 2nd GPU for softmax kl_div jsd cross_entropy) → per-kernel geomean; O=0.998.
  (run2_O.py is in /tmp; the IN_SAMPLE dict is inline — re-create from ledger.run2 if gone.)
- Byte-identical check: run `_lab/harness/run2_seed_dump.py` under PYTHONPATH=wt-reduction-2 vs PYTHONPATH=
  wt-reduction and diff (8 non-welford identical).
- Tests: `cd wt-reduction-2 && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD <py> -m pytest -q test/test_reductions.py
  test/test_autotuner.py test/test_best_available.py test/test_autotuner_heuristics.py` → 187 passed.
- Lint/type: `ruff check helion/ examples/`; `ruff format --check …`; `pyrefly check helion/_compiler/
  autotuner_heuristics/triton.py` (0 errors; device_ir shows pre-existing missing-import noise only).
