# HANDOFF — Triton Reduction Seed-Heuristic (read this FIRST)

> Single entry point for the next agent. Compresses the hard-won cross-run knowledge; points to the
> deep docs for detail. The forward reduction-seed task is COMPLETE & at its deterministic-seed ceiling.
> Worktree `/home/calebkim/helion-new-heuristics/wt-reduction`, branch `reduction-heuristics-autotuner`,
> HEAD `7f08d70f` (44 commits, **never pushed**). Heuristic = **v8 `TritonReductionHeuristic`**.

## 0. Where the knowledge lives (read in this order)
1. **`_lab/HANDOFF.md`** (this) — orientation + the traps/lessons that aren't obvious from code.
2. **`_lab/FINAL_REPORT.md`** — the deliverable report: heuristic structure + per-branch why, per-kernel
   G (in-sample/validation/TEST), Products A+B, generalization, ceilings, rejected ideas, out-of-scope.
3. **`_lab/SETUP.md`** — the VERIFIED working env (interpreter, PYTHONPATH wiring, GPU pin, bare-seed
   mechanism, the two import gotchas). Use it EXACTLY.
4. **`_lab/ledger.json`** — durable state: `champion`, per-kernel `kernels`, `oracle_cache`,
   `product_A_measurements` (v1..v8), `product_B_measurements`, `validation_measurements`,
   `generality_stress_test`, `*_gate_verdicts`, `iteration_history`.
5. **`_lab/notebook.md`** — the worker's reasoning trace (decisions + empirical why, tried-and-rejected).
6. **`_lab/HUB_LOG.md`** — the whole run's arc (newest at bottom of the dated sections), incl. every gate verdict.
7. Code maps: `_lab/step2_code_map.md` (T1/ReductionFact), `_lab/t2_code_map.md` (T2 user-tiled),
   `_lab/codegen_knob_map.md` (the codegen-knob surface + why it's autotuner-only).

## 1. What the deliverable IS (high-level idea)
A **general compile-time autotuner seed-heuristic** for **forward inner reductions** (H100/sm_90, Triton,
tuned at fp32 but dtype-general via `itemsize`). It reads pre-digested **workload facts** about a kernel
(`ReductionFact`) and emits **one seed `Config`**. That seed serves both products:
- **Product A (skip autotune):** the seed IS the config (`configs=[seed]`, no search). Goal: beat
  torch.compile-default; approach max-autotune. Achieved **in-sample O = geomean Gₖ = 0.9786** over 9
  forward kernels (Gₖ = tc_default_lat/seed_lat); **fresh-oracle seed/oracle = 1.007** (at ceiling).
- **Product B (seed the autotuner):** the seed is injected into the autotuner's gen-0 population →
  reaches a good config in **~1.94× less wall-clock** (median time-to-95%); ~halves the budget.

**Core idea of the heuristic:** persistent reduction is the workhorse (the un-seeded Helion
`default_config` needlessly LOOPS at chunk 4096 and loses 20-41% on wide rows); seed persistent up to the
backend's structural compile cap (2²⁰ elems), ramp `num_warps` with rnumel, and force LOOPED only where
the **working set would spill** (three caps: multi-load re-read, multi-accumulator, structured-combine).

## 2. Code entrypoints (the actual changes — 892 lines across 4 helion/ files + 2 tests)
- **`helion/_compiler/autotuner_heuristics/triton.py`** — `TritonReductionHeuristic` (the heuristic).
  `is_eligible` = `_triton_reduction_eligible` (gate: `len(reduction_facts)==1 and not matmul_facts`).
  `get_seed_config` routes T1 vs T2 and picks the config. `_num_warps(fact)` = the rnumel ramp. Constants:
  `STREAM_WARPS32_MIN_ELEMS=16384`, `MULTILOAD_PERSIST_MAX_BYTES=131072`, `BANDB_R_BLOCK_BYTES=16384`,
  `STRUCTURED_APPLY_CAP_BYTES=8192`, `STRUCTURED_COMBINE_CAP_BYTES_LOOPED_APPLY=16384`. Every branch is
  commented with its empirical why + the lab script that proved it.
- **`helion/autotuner/config_spec.py`** — (a) `ReductionFact` NamedTuple + `self.reduction_facts` list
  (fields: block_id, size_hint, m_block_ids, static_rnumel, dtype, itemsize, num_load, num_store,
  num_reduction_ops, num_tiled_accumulators, is_structured_combine, apply_block_ids). (b) **CORE FIX:**
  `ReductionLoopSpec._encode_flat_value(None)` returns `.high` not `.default()` (see §4 trap #6).
- **`helion/_compiler/device_ir.py`** — fact population: `register_rollable_reductions` (T1) +
  `register_user_tiled_reductions` (T2, called right after, GUARDED `if not config_spec.reduction_facts`
  so T1/T2 are mutually exclusive and exactly 1 fact exists) + shared `_count_reduction_workload` +
  `_build_reduction_fact`. T2 reduction axis is found via `ReductionLowering.block_index` filtered against
  `grid_block_ids` (the filter is LOAD-BEARING for jsd — see `_lab/t2_code_map.md`). `is_structured_combine`
  gate: `>1 non-grid tile AND ≥1 apply tile over the same static extent`.
- **`helion/_compiler/autotuner_heuristics/__init__.py`** — registration: `HEURISTICS_BY_BACKEND['triton']
  = (TritonSkinnyGemmHeuristic, TritonReductionHeuristic)`. `compiler_seed_configs` dispatches by backend.
- **Tests:** `test/test_autotuner_heuristics.py::TestTritonReductionHeuristic` (fires + branch selection);
  `test/test_best_available.py` (the persistent round-trip test, fails pre-fix).
- **OUTSIDE worktree git:** the `torch_compile_rms_norm_default` baseline variant is in the ORIGINAL
  checkout's tritonbench rms_norm operator (`/home/calebkim/helion-new-heuristics/local/helion/benchmarks/
  tritonbench/.../operators/rms_norm/operator.py`). Reproducible patch + why: `_lab/harness/patches/`.

The branches, in order (all keyed on workload facts, NEVER kernel identity):
1. **Persistent vs structural-looped:** persistent (`reduction_loops=[None]` T1 / `block_sizes[red]=
   next_pow2(N)` T2) iff `size_hint <= max_tensor_numel (2²⁰)`; else looped (only huge rows can't compile persistent).
2. **num_warps ramp** by rnumel: ≤1024→4, ≤4096→8, ≤16384→16, else 32. (Keyed on rnumel ALONE — see trap #2.)
3. **Multi-load persist byte-cap:** `num_load≥2 AND rnumel*itemsize>128KiB` → looped (re-read working-set spills).
4. **Band-B R_BLOCK cap:** `num_tiled_accumulators≥1` (kl_div/jsd's [M,R] accumulators) → R_BLOCK ≤ 16KiB/itemsize.
5. **is_structured_combine (welford):** combine tile = `min(largest_pow2_div(N), cap)` (MUST divide N — trap #7);
   apply tile = `min(next_pow2(N), 8KiB)` looped, with the combine cap raised to 16KiB when apply is looped.
6. **M-block** at the autotuner floor (`max(1,min_size,autotuner_min)`); never raised (raising it is a
   regime-conflict fence — see rejected ideas).

## 3. Design decisions & tradeoffs (the WHY — most of this is also in FINAL_REPORT §3-§7)
- **Persistent-first.** The dominant win. Default loops at 4096 and re-streams; persistent keeps the whole
  contiguous row resident. Recovered e.g. rms_norm (2048,16384) 0.70→0.99.
- **Caps express working-set pressure, generically.** Three independent spill sources, three byte-caps keyed
  on the *direct* fact: re-read loads (num_load), live [M,R] accumulators (num_tiled_accumulators), the
  structured-combine apply pass. All in BYTES via `itemsize` so they generalize to bf16/fp16.
- **num_warps keys on rnumel, not num_load.** (v3 tried gating w32 on num_load and it was a fence — trap #2.)
- **Correctness is gated before perf, every time.** welford's combine-divides-N is a CORRECTNESS constraint
  (trap #7), not a perf choice.
- **What we deliberately do NOT seed (the residual is real but not deterministic-seedable):** the codegen
  knobs `indexing`/`load_eviction_policies`/`pid_type`/`num_sm_multiplier`/`maxnreg`. The autotuner's
  oracle wins the last 3-30% on small-N/welford via these, but they are **autotuner-only**: e.g. optimal
  eviction patterns are mutually contradictory across kernels (sum wants all_last, cross_entropy all_first)
  → no single deterministic rule. This is Product-B territory. Verified exhaustively (`_lab/codegen_knob_map.md`,
  the pid/indexing/eviction matched-lever rejections in `ledger.gate_verdicts`).

## 4. Known hacks / fragile assumptions / TRAPS (read every one — these are how you avoid re-making mistakes)
1. **Worktree wiring:** `helion` + `examples.*` resolve to the WORKTREE only via
   `PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction` from a non-original cwd. **tritonbench
   operators resolve to the ORIGINAL checkout** (hardcoded meta-path finder; PYTHONPATH can't shadow). So
   operator-level edits (e.g. tc-default baselines) go in the original checkout. ALWAYS
   `assert helion.__file__.startswith(".../wt-reduction")`. (SETUP.md "import/tritonbench edit wiring".)
2. **MATCHED-LEVER A/B is mandatory** (the #1 lesson — it caught v2/v3/pid/M-block confounds). When testing
   a lever, hold ALL OTHERS EQUAL and A/B against the best SIMPLE alternative (e.g. persistent/w32), NEVER
   the catastrophic un-seeded default. A bundled comparison (persistent/w16 vs looped/w32) wrongly credits
   the loop flip for a warps win.
3. **The oracle is a BUNDLE.** max-autotune winners couple block_sizes + codegen knobs. Field-diffing MUST
   re-bench the FULL VERBATIM config; isolating one lever and RE-PAIRING it fabricates an unmeasured config
   (this produced the bogus "w32=1174us artifact" and the "welford 0.968 seedable" overstatement). To split
   seedable vs autotuner-only: measure the oracle's block_sizes at DEFAULT codegen knobs, then +knobs.
4. **Autotuner-internal do_bench is NOT biased** (proven 3 ways). Earlier "anomalies" were bugs in our
   field-diff scripts, not the timer.
5. **Shared machine / co-tenants.** Pin `CUDA_VISIBLE_DEVICES`, re-check `nvidia-smi` before+after, re-run
   any shape with >5% do_bench spread. GPU 0 usually busy; GPU 2 picked up a co-tenant mid-run once. Tiny-M
   / sub-25µs shapes are noise-floor — lift M to time them; treat their wins/losses skeptically.
6. **Persistent-seed round-trip (CORE FIX, keep it):** before the fix, a `reduction_loops=[None]` seed was
   silently flattened to looped-4096 on autotuner injection for rnumel>4096 (Product B was crippled). Fix =
   `_encode_flat_value(None) -> .high`. Product A (configs=[seed]) never flattens, so it was unaffected.
   Guarded by `test/test_best_available.py`.
7. **welford correctness:** the kernel divides by the tile constexpr `chunk.size(-1)`, NOT the masked count,
   so a masked (next_pow2) combine tile is NUMERICALLY WRONG at non-pow2 N. The seed MUST pick a combine
   tile that DIVIDES N (largest pow2 divisor; =1 at prime N → correct but slow). This is the welford
   prime-N residual (a kernel-structure limit, not seedable-fixable without a kernel rewrite).
8. **fp32 + softmax:** softmax defaults to fp16 — override + assert fp32 on every softmax call.
9. **`_lab/harness/` has 143 scripts, many scratch.** Trust the LIVE `compiler_seed_configs` output, not any
   hardcoded-constant approximations in older scripts (e.g. a "2048" literal that happens to match for fp32).
   Canonical: `measure_g_*.py`, `bare_seed_run.py`, `evidence_block.py`, `productB_driver.py`.
10. **num_store / num_reduction_ops** are collected in ReductionFact but NOT gated on (diagnostic). `_m_extent`
    was removed (dead). Don't reintroduce M-block-raising (regime-conflict fence).

## 5. How to run tests / reproduce
Canonical invocation (from SETUP.md):
```
cd /home/calebkim/helion-new-heuristics/wt-reduction && CUDA_VISIBLE_DEVICES=<idle> \
  PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction \
  /home/calebkim/.conda/envs/helion/bin/python <script>
```
- **Unit/integration tests:** `python -m pytest test/test_reductions.py test/test_autotuner.py
  test/test_best_available.py test/test_autotuner_heuristics.py` (all green: 24/107/46/10 at HEAD).
- **Lint/type (pre-commit):** `ruff check helion/...`, `ruff format --check helion/...`, `pyrefly check` — clean.
- **Product-A G per kernel:** `_lab/harness/measure_g_rms_norm.py`, `measure_g_reduction.py --kernel {sum,long_sum}`,
  `measure_g_softmax.py`, `measure_g_lossk.py`, `measure_g_jsd.py`, `measure_g_ce.py`, `wf_final_seed.py`.
- **Bare-seed primitive:** `_lab/harness/bare_seed_run.py` (configs=[seed], no autotune, correctness, do_bench).
- **Product B:** `_lab/harness/productB_driver.py` + `productB_full_run.sh` (seeded vs unseeded;
  UNSEEDED = `HELION_DISABLE_AUTOTUNER_HEURISTICS=1`; cold cache + `HELION_FORCE_AUTOTUNE=1`, NOT SKIP_CACHE).
- **TEST read (already done once):** `_lab/harness/TEST_readonce.py` (don't re-read TEST without reason — firewall).
- **End-to-end via TritonBench:** `benchmarks/run.py --kernel <k> --metrics latency,accuracy,speedup
  --precision fp32 --M <M> --H <N>` with `HELION_AUTOTUNE_EFFORT={none|quick|full}`.

## 6. Orchestration reality (if you are the next HUB/manager)
- **`SendMessage` is NOT available** in this harness → there is no persistent agent. The "persistent worker"
  = fresh `Agent` invocations each briefed off `_lab/notebook.md` + `_lab/ledger.json` (the wip blesses this
  as lossless — the notebook IS the state). Helpers (referee/auditor/investigators) are one-shot Agent calls.
- **Acceptance is independently gated** and that is what kept the run honest: nothing entered the champion
  without the **results-referee** reproducing it AND the **adversarial-auditor** passing it. The auditor
  caught real issues SEVEN times (v2 looped/grid-occ misattribution, v3 num_load fence, pid confound,
  indexing no-win, M-block regime-conflict, two welford overstatements). **Do not skip the auditor.**
- **GPU partitioning:** run parallel agents on distinct idle GPUs (1/2/3); NEVER two timing runs on one GPU
  (it corrupts do_bench). Read-only/code agents can share.
- Commit early/often; **never `git push`** (human handles it).

## 7. Status & what's next
- **DONE & at ceiling:** all 9 forward kernels seeded (rms_norm, sum, long_sum, layer_norm, softmax_two_pass,
  kl_div, jsd, cross_entropy, welford). O=0.9786. Generalizes across **shapes** (TEST gap −0.036 ex-welford),
  **structures** (new softmax T1 +34%), **reduction ops** (max/min, op-agnostic), and correctly **declines**
  on unsupported (jagged dynamic rdim). Harness re-certified unbiased; code PR-ready.
- **Irreducible residuals (NOT bugs):** cross_entropy (8192,131072) 0.54 = kernel-source (Helion re-reads
  the row vs tc online-softmax; needs a kernel rewrite). welford prime-N + small-N = autotuner-only codegen
  knobs / kernel-structure. Don't chase these with a deterministic seed.
- **NEXT WORKSTREAM — Band D (backward), wip-DEFERRED, scope decision for the human:** rms_norm_bwd /
  layer_norm_bwd weight/bias gradients are the only OUTER-shaped reductions (sum over M, not the contiguous
  last dim). The forward heuristic does NOT handle outer reductions — Band D needs its own fact/branch
  (a separate `MatmulFact`-style analysis for the M-reduction). In-sample bwd shapes are in `list_of_kernels.md`.
- **To turn this into a PR:** `git diff 1ec3193d -- helion/ test/` is the shippable change (892+158 lines,
  4 helion files + 2 tests). The tc-default baseline patch (original checkout) is in `_lab/harness/patches/`.
  `_lab/` is lab scratch (don't ship). Rebase onto the latest helion main before pushing.

## 8. One-paragraph TL;DR for a fresh agent
A general Triton reduction seed-heuristic (v8, branch `reduction-heuristics-autotuner`, HEAD 7f08d70f) is
COMPLETE for all 9 forward inner-reduction kernels: persistent workhorse + rnumel warps ramp + three
working-set byte-caps (multi-load / Band-B accumulators / structured-combine) + a welford structured-combine
treatment, all keyed on `ReductionFact` workload properties (never kernel identity), plus a core autotuner
round-trip fix so persistent seeds survive injection. Product A O=0.9786 (at the fresh-oracle ceiling),
Product B 1.94× faster time-to-target, generalizes across shapes/structures/ops, harness re-certified, code
PR-ready. The remaining headroom is autotuner-only codegen knobs (not deterministic-seedable) or kernel-source
limits. The next workstream is backward/outer reductions (Band D), explicitly deferred. READ §4 (traps) before
touching anything — especially matched-lever A/B, the oracle-is-a-bundle trap, and welford-divides-N.
