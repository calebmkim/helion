# RUN 2 — Live Notebook (reasoning trace, decisions + empirical why, tried/rejected)

> The DURABLE source of truth for run 2. A fresh worker/agent reads THIS + the run-1 reference
> (`HANDOFF.md` §4 traps, `FINAL_REPORT.md`, `ledger.json`) to continue losslessly. Newest at bottom.
> All file:line refs are approximate — grep the symbol.

## Orchestration (this harness)
- Hub = main session (only spawner). No `SendMessage` → no persistent agents; "persistent worker" =
  fresh one-shot `Agent` calls briefed off this notebook + ledger, OR the hub drives directly.
- **Workflow** for deterministic parallel fan-outs + gates (G-sweeps, A/B batteries, oracle runs,
  Product-B, multi-skeptic adversarial verification). GPU-partitioned: timing concurrency ≤ #idle GPUs
  (1/2/3; GPU0 has a co-tenant). NEVER 2 timing runs on one GPU (corrupts do_bench). Read-only agents share.
- Acceptance independently gated: nothing enters champion unless results-referee reproduces it AND
  adversarial-auditor passes. Reject cheats/unreproducible/correctness-fails; never loosen; keep going.
- Never stop. "stuck/converged/broken infra" = a prompt for the next move, not an exit.

## Canonical invocation (run 2 — note the -2 and NO sys.path.insert footgun)
```
cd /tmp && CUDA_VISIBLE_DEVICES=<idle 1|2|3> \
  PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction-2 \
  /home/calebkim/.conda/envs/helion/bin/python <script>
```
Every run-2 script: `assert helion.__file__.startswith("/home/.../wt-reduction-2/")`. Do NOT reuse
run-1 harness scripts verbatim — they hardcode the OLD path and `sys.path.insert(0, OLD)`, and
`"wt-reduction-2".startswith("wt-reduction")==True`, so they'd SILENTLY run old code.

## v8 baseline (the floor; O_in_sample = 0.9785 over 9 forward kernels)
Heuristic `TritonReductionHeuristic` (helion/_compiler/autotuner_heuristics/triton.py). Branches (all on
ReductionFact, never kernel identity): persistent-vs-structural-looped (cap 2^20) → num_warps rnumel ramp
4/8/16/32 → multi-load persist byte-cap (num_load≥2 & >128KiB → looped) → Band-B R_BLOCK cap
(num_tiled_accumulators≥1 → 16KiB) → Band-C welford (is_structured_combine; combine=largest_pow2_div(N)
capped, apply=min(np2(N),8KiB) looped-at-wide, combine cap 8→16KiB when apply looped) → M-block at floor.
Seed emits only block_sizes/reduction_loops/num_warps/num_stages (codegen knobs NOT seeded = Goal 2 target).

## TRAPS carried from run 1 (HANDOFF §4 — apply to EVERY new A/B)
1. Matched-lever A/B: vary ONE lever, hold others equal, vs the best SIMPLE alt (persistent/w32), NOT the
   catastrophic default strawman.
2. Oracle is a BUNDLE: re-bench the FULL VERBATIM oracle config; never isolate-a-lever-and-re-pair-it.
   To split seedable-vs-oracle-only: oracle block_sizes at DEFAULT codegen knobs, then +candidate knob.
3. do_bench noise floor: sub-25µs / tiny-M shapes — lift M or exclude; any "win" there is undefendable.
4. Shared machine: re-check nvidia-smi before/after; re-run >5% spread; GPU0 co-tenant.
5. Persistent-seed round-trip fix in place (_encode_flat_value(None)→.high) — keep it.
6. fp32 fixed; assert dtype every call (softmax defaults fp16). Heuristic reads dtype/itemsize.

## Goal 1 — welford source fix + Band-C re-derivation  [STARTED 2026-05-30]
THE BUG: examples/welford.py L52 `Tn = chunk.size(-1)` = constexpr tile width, NOT masked valid count →
wrong mean/count/M2 on the last tile when block size doesn't divide N. Helion masks OOB loads with other=0
so sum_x/sum_x2 are already correct over valid cols; ONLY Tn is wrong. Fix: `Tn = (tile_n.index < n).sum()`
(grep `tile.index < bound` in helion/language/loops.py + examples/segment_reduction.py for the idiom).
CASCADE to undo: the buggy Tn forced combine tile to be a pow2 DIVISOR of N (→1 at prime N → cliff), which
spawned `is_structured_combine` + `apply_block_ids` + `largest_pow2_div` machinery. With the source fixed
the divisor constraint is GONE — combine tile can be a normal byte-capped min(N,cap). KEEP the legit
two-pass need: seed the APPLY pass wide (general T2 floors non-reduction blocks to 1 → catastrophic for
apply). Simplify/generalize Band-C; if is_structured_combine fits exactly one kernel that's a Goal-5 flag.
Plan: (1) fix+verify OOB zero-fill + correctness on well-factored/odd/PRIME(1543); (2) re-validate welford
oracle under corrected kernel (re-run fresh quick-autotune oracle on (262144,2048), compare to v8 cache; if
>3% beyond noise → oracle was confounded, re-run all welford oracles); (3) re-derive Band-C; (4) re-measure
welford in-sample G + pre-authorized welford TEST re-read; (5) commit (source fix ships w/ deliverable).

## FIREWALL MAP (from TEST_readonce.py — guard Goal 4 + TEST re-reads)
- Authorized TEST re-reads (ONLY these two): **welford TEST** (invalid under corrected kernel) +
  **rms_norm TEST G ~0.828** (no raw log). No other kernel's TEST column is re-read.
- welford: in-sample {1024,1536,2048,4096}@262144; VALIDATION {2560,3072}@262144,(65536,16384);
  TEST {(262144,5120),(262144,7168),(262144,1280),(262144,1543),(131072,2048),(262144,768)}.
  → Goal-4 promotes 2560(val)+5120(TEST) to **in-sample-v2** (+M-var (512,4096),(8192,4096)). I have swept
  5120,2560,1024,2048,4096,(8192,4096) — OK as in-sample/in-sample-v2. KEEP CLEAN for welford TEST re-read
  (do NOT tune on): **7168, 1280, 1543(canary), 768, (131072,2048)** (+ can add 3072).
- rms_norm TEST: {(256,4096),(2048,2560),(2048,1025),(4096,10240),(8192,2048),(1,131072),(65536,512)} —
  re-read authorized (regen the ~0.828). Goal-4 tiny-M (256,4096) may go in-sample-v2 (rms TEST regenerated).
- DISCIPLINE for the 7 NON-re-read kernels (layer_norm/sum/softmax/cross_entropy/kl_div/jsd): in-sample-v2
  shapes MUST be disjoint from their sealed TEST. Brief collisions to AVOID/substitute: softmax (4096,3072)
  IS softmax TEST → use a different non-pow2 (e.g. (8192,3072)); layer_norm (256,4096) IS layer_norm TEST →
  use rms_norm for the tiny-M (256,4096) only, pick a distinct tiny-M for layer_norm. Loss-kernel real-vocab
  in-sample-v2: pick BT not already in VALIDATION/TEST (validation already has 2048/4096 × {32000,128256,...}).

## Goal 1 — DONE + GATED (2026-05-31). Commits ddf8fc34 (source), 43492809 (band-C).
RESULT: welford source bug fixed (`Tn=(tile_n.index<n).sum()`), Band-C re-derived to two INDEPENDENT
byte caps (combine=min(np2(N),32KiB/it) [persistent — looping it regresses via serial recurrence];
apply=persistent np2(N) if per-row-valid-bytes<=12KiB else looped 8KiB chunk). Deleted largest_pow2_div +
apply<->combine coupling (bug-artifacts). welford in-sample G(orig 1024/1536/2048/4096) 0.911->0.926;
1536 +6.7% (combine 512->2048), 2560(v2) +12.2% (->0.973), 5120(v2) ~tie 0.696; prime-1543 0.082(WRONG)
->0.958(CORRECT+FAST). 8 non-welford kernels BYTE-IDENTICAL (20/20). Both gates PASS (referee ACCEPT,
auditor PASS). Auditor proved the is_structured_combine GATE generalizes (2 synthetic structured-combine
kernels fire); FLAG: byte-cap VALUES are welford-curriculum-fit -> validate w/ real 2nd kernel in Goal 5.
ORACLE CONFOUND CONFIRMED: corrected welford oracle picks non-divisor combine=4096 at N=2560 (buggy
kernel's accuracy gate would reject) -> v8 welford oracle was artificially divisor-constrained.
GOAL-2 SEED FOR WELFORD: big codegen-knob residual — N=4096 seedable 0.76 vs corrected-oracle 0.961
(tensor_descriptor indexing + load_eviction last/first); N=2560 oracle 0.987 (eviction); N=8192 0.737.
DEFERRED: welford TEST re-read (clean shapes 7168,1280,768,(131072,2048); 1543 canary already 0.958) ->
consolidated TEST pass w/ rms_norm TEST G (Goal 6). warps residual: N=5120 ramp gives w16 (0.696) vs w8
optimum (0.721) — left simple (no regression); possible later refinement.

## Goal 2 — codegen knobs [STARTED 2026-05-31]. BIG FINDING: load_eviction_policies is seedable.
OVERTURNS run-1's "eviction = autotuner-only, no seedable rule". welford (262144,4096), MATCHED block_sizes
(G1 seed [16,4096,2048]w8), eviction-only (no TD): default G=0.760; **evict=[last,first,last,first] G=0.947
(+18.7%)**; all_last=0.842; all_first=0.669. So per-slot alternating last/first is the win (not a global rule).
tensor_descriptor OOMs on welford's wide combine load (needs 262KB shared > limit) → TD not usable at
combine=4096 (oracle used combine=1024 to fit TD, reaching 0.961; eviction-only at our block_sizes already
gets 0.947). pid_type stays flat. num_sm_multiplier/maxnreg N/A (flat).
SLOT MAP (welford, indexing.length=5 [4 loads + store@idx4], evict.length=4): loads in graph order. HYPOTHESIS
(cache residency, brief's hint): welford RE-READS x in the apply pass, so x-in-combine wants 'last' (keep in
L2 for the apply re-read) and x-in-apply wants 'first' (last use → stream/evict). weight/bias small. MUST map
slots→tensors empirically (generated Triton) to find the GENERAL workload property (NOT copy [last,first,...]).
PLAN: (1) map slots; (2) test eviction generality across welford shapes (2560/5120/8192/1024) + on
rms_norm/sum/cross_entropy small-N & wide (run-1 noted +11-23% there); (3) co-design a ReductionFact eviction
rule keyed on the residency property; (4) matched-lever A/B + persist raw numbers + DISJOINT-shape validation
+ auditor (anti-fit-to-oracle). Make pid_type='flat' explicit in the seed (principled constant, run-1 lock).

## Goal 4 — in-sample-v2 added + baselined (2026-05-31). list_of_kernels.md updated (snapshot _lab/list_of_kernels_run2.md).
Baseline G_seed (current G1 heuristic) geomean per kernel: rms 0.776, ln 0.867, wf 0.876, softmax 1.005,
CE 0.715, kl 1.054, jsd 1.030, sum 1.050, long_sum 0.726; OVERALL 0.890. All correct, no OOM.
Goal-2 targets (G<0.90): rms/ln small-M+medium (persistent, eviction re-read target), welford 5120 (eviction),
CE wide-vocab 0.54-0.80 (SOURCE ceiling -> Goal5 online-logsumexp), long_sum few-row 0.67 (grid-starved ->
split-K DEFERRED, attribute not chase). softmax/kl/jsd/sum all >=1.0 (no codegen headroom needed).

## Goal 2 — EVICTION the separating property = PER-TENSOR REUSE (2026-05-31). Slot map via generated Triton.
Confirmed clean, principled wins (matched block_sizes, eviction-only):
- num_load==1 PURE STREAM (sum/long_sum, only x): 'first' (evict_first) -> sum (512,8192) 0.925->1.451,
  (2048,16384) 0.931->1.087. Stream once, no reuse -> evict_first frees L2.
- RE-READ reduction tensor (welford x across 2 tile-loops; softmax_two_pass x across 2 passes): 'last' on
  non-final read(s), 'first' on final read -> welford(262144,4096) 0.759->0.950, (5120) 0.696->0.807;
  softmax(8192,32768) 0.983->1.010. (Per-load reuse rule, computed by run2_evict_probe.)
NOISY / leave DEFAULT: rms_norm/layer_norm FUSE the two x-uses into ONE load (slots=[x,weight(,bias)]); x
wants 'first' but reused weight/bias want 'last'/default -> blanket policy REGRESSES (rms 256,5120 first=-8%;
512,8192 last=-25%). all_last/all_first/default each win at different shapes = run-1's "contradictory" (REAL
for these). cross_entropy eviction-neutral (~1.05; its gap is SOURCE ceiling not eviction).
RULE DESIGN (co-design device_ir per-load fact -> heuristic emit load_eviction_policies):
  per load slot (graph order): if tensor re-read LATER in kernel -> 'last'; elif it is the (final read of a)
  re-read tensor -> 'first'; elif num_load==1 single streamed reduction input -> 'first'; else '' (default).
  i.e. ONLY set policies that are CLEAN wins (stream + re-read); leave multi-load-with-operands at default.
pid_type -> emit 'flat' explicitly (run-1 lock, principled constant). tensor_descriptor: OOMs on welford wide
combine; defer/limited. POSSIBLE refinement: rms_norm reduction-input(x)-only 'first' (leave weight default) —
needs per-load classification (reduction-input vs broadcast operand); verify separately.

## Goal 2 — eviction IMPLEMENTED (uncommitted, pending gate). triton.py _eviction_policies helper.
RULE (matched-lever, in-process A/B): num_load==1 -> ['first']*len (sum/long_sum); is_structured_combine ->
['last']+['first']*(len-1) (welford, slot0=combine-x re-read). Others left default.
LIVE-seed G uplift (run2_measure_g): welford 4096 0.760->0.951(+19%), 5120 0.694->0.806(+11%), 2560 ->0.980,
1536 in-process 0.963->0.975, 1024/2048 ~tie (NO regression — earlier cross-process 1536 dip was noise);
sum 2048,16384 0.931->1.011, 512,8192 ->0.970, 32768,256 ->1.132. 8 non-evict kernels byte-identical (evict=None).
ruff/pyrefly clean; 56 unit tests pass. PENDING: verification agent RULE A/B/C (full curriculum no-regression +
rms_norm x-only-first test) -> then add rms/ln eviction if clean, then referee+auditor gate, commit.

## Product-A O milestone (2026-05-31): in-sample O 0.9786 -> 0.9980 (+2.0%). Champion = v8+G1+evict.
Per-kernel G: rms 0.979, sum 1.019, long_sum 1.138, ln 0.985, welford 0.975, softmax 0.960, kl 1.028,
jsd 0.997, CE 0.916. Residuals: CE (8192,131072) 0.539 SOURCE ceiling (Goal5); rms (2048,2048) 0.871 +
small-N (codegen-knob explorer). welford TEST re-read pending (consolidated pass).

## Goal 5 — new-kernel generality probes (CANDIDATES, design — propose profiles to auditor before impl).
Each must be STRUCTURALLY DISTINCT (different (num_load,num_tiled_accumulators,num_reduction_ops,
is_structured_combine) profile or path), NOT a cosmetic wrapper. Validate each band generalizes; if a band's
seed is >10% off on the new kernel's in-sample shapes -> REWORK heuristic (find the distinguishing workload
property), NO kernel-identity fence.
- STRUCTURED-COMBINE band (welford's is_structured_combine — auditor flag: byte-cap VALUES welford-fit):
  candidate **two-pass standardize / plain mean-var layernorm** (combine = 2 plain reductions sum_x, sum_x2;
  apply = normalize) -> is_structured_combine=True but num_reduction_ops=2 (vs welford 3-stat recurrence) +
  num_load differs -> tests whether the combine/apply byte caps + re-read eviction generalize beyond welford.
- MULTI-LOAD band (cross_entropy's MULTILOAD_PERSIST_MAX_BYTES + re-read): candidate **standalone logsumexp**
  (row max-pass + exp-sum-pass over x = 2 reads, re-read; scalar output, no apply) -> num_load=2, re-read,
  not structured-combine -> tests multi-load persist cap on a different identity; also tests whether re-read
  eviction should extend to multi-pass multi-load reductions (it regressed softmax mid-N -> watch).
- BAND-B accumulator (kl_div/jsd's num_tiled_accumulators): candidate **2nd heavy-epilogue loss** carrying
  [M,R] accumulator(s) with a different num_store/num_reduction_ops (e.g. a generalized-KL or a cosine/dot
  loss). -> tests BANDB_R_BLOCK_BYTES generalizes.
Impl site: examples/ + benchmarks/run.py KERNEL_MAPPINGS + tritonbench operator w/ torch_compile default
baseline (operator edits in ORIGINAL checkout) + _lab/harness measure + in-sample-v2 shapes. Correctness-gate
vs eager FIRST. Propose each ReductionFact profile + "why it tests the band differently" to auditor pre-impl.

## PHASE I COMPLETE (2026-05-31). in-sample O 0.998, TEST O 0.946, prime-N 0.905. Heuristic FROZEN.
Phase-II prereq: seed (incl eviction+pid) round-trips through autotuner flatten/unflatten PRESERVED (welford/
sum/long_sum verified) -> seeded Product-B arm carries eviction. Capstone auditor running (Phase-II gate).
PHASE II PLAN (Goal 3, on the FROZEN seed): 3a budget-reduction — (a) seeded-QUICK vs unseeded-FULL (does
quick-seeded match full-unseeded optimum? budget gap=savings); (b) convergence curves seeded vs unseeded at
full effort (best-perf vs generation AND vs wall-clock SEPARATELY; gens/time to 95-99%). Pilot seeded-FULL vs
unseeded-FULL on 1-2 shapes. 3b beat-max-effort (depends on Goal-2 eviction bundles): add get_seed_configs()
multi-seed portfolio (base class), pre-register N + shapes, probabilistic beat def, fresh-process median-of-7
re-bench, exclude noise-floor, expanded anti-lucky-run auditor. Driver: adapt run-1 productB_driver (wt-2 path,
NO sys.path.insert, +welford +cross_entropy_online, eviction-aware). UNSEEDED=HELION_DISABLE_AUTOTUNER_HEURISTICS=1.

## Goal 3b — beat-max-effort. PRE-REGISTERED portfolio + hypotheses (2026-05-31, BEFORE the seeded 3b runs).
Plumbing: get_seed_configs() (opt-in HELION_REDUCTION_SEED_PORTFOLIO) returns base + structural variants.
3a showed unseeded-FULL usually REACHES the optimum (welford/softmax tie at full) -> 3b beats are only where
the blind search UNDER-SAMPLES a hard coupling run-to-run. Characterizing unseeded-full variance (welford
262144,4096 N=5) to find such shapes BEFORE finalizing the comparison (this measures unseeded variance, NOT
which config wins -> not p-hacking; portfolio derived from principle below).
PORTFOLIO (each = one falsifiable hypothesis; seed = base perturbed on ONE lever):
- H0 base = best deterministic seed. Prove/disprove: seeded best-of-N >= unseeded best-of-N at stated conf.
- H1/H2 warps {4,8,16,32}: HYP the rnumel ramp's warp is 1 step off the optimum on register-heavy combines/
  streamed wide rows and the bounded search doesn't reliably land the other step. Disproof: unseeded best-of-N
  matches seeded (search finds the warp anyway).
- H3 eviction {rule, none, all-last}: HYP the per-load eviction coupling is a large space the bounded search
  under-samples; seeding the coupling -> seeded reliably hits it. Disproof: unseeded best-of-N finds eviction.
- H4 num_stages=2: HYP marginal (memory-bound). Likely disproved (inert).
PROTOCOL (brief): pilot N in {3,5,10,20} on 1 at-ceiling Band-A + 1 harder (welford/Band-B); COMMIT one N for
all; pre-register shapes; beat = P(seeded best-of-N >= X) > P(unseeded best-of-N >= X) at stated conf, X=99% of
unseeded-full oracle; report best/median/spread both arms; fresh-process median-of-7 re-bench winners; exclude
noise-floor; matched knob sets. Banned: re-tuning the portfolio to the observed unseeded winners.
