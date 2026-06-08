# Hub Log — non-blocking breadcrumbs for the human

> Passive status notes the human sees on check-in. Convergence flags. Never a stop, never a request
> for input. Newest at top.
>
> **>>> NEXT HUB: read `_lab/HANDOFF_run3.md` FIRST. <<<** Live state: Phase 2, 5 advances banked, EDIT#5
> (jsd) ACK'd-to-build (immediate next: gate it with the full-jsd-flip-set referee condition). EDIT#4 was
> REJECTED+reverted (apply-cap=8192). Worklist to PARITY: EDIT#5 → EDIT#4b welford (M-keyed) → narrow-N #15
> (hard/parked) → long_sum-2M (anti-giving-up) → quick-VICTORY confirms. NEVER `git push`.

## 2026-06-03 (RUN 3 begins — STRICT per-shape seed≈oracle bar; new orchestration)
> Run 3 inherits the run-2 champion (in TritonReductionHeuristic) and RE-OPENS it against a stricter
> bar. The originating work-order is `_lab/prompts/reduction-heuristics-standalone.md` (+ server-specific
> -setup.md + shapes_v3_draft.py). The hub is THIS Claude Code session (Opus 4.8 1M); orchestration is a
> real Agent/Team harness (SendMessage continue-in-place worker, peer investigators, fresh-per-claim gates)
> — unlike run-1/2 which had no SendMessage and ran fresh-worker-per-iteration.
- **The reframe (why run 3 exists):** run 2 declared COMPLETE gated on the AGGREGATE `O` vs tc-default
  (in-sample 0.998, TEST 0.946) and a GEOMEAN "fresh-oracle seed/oracle=1.007". The new work-order resets
  the bar to **per-shape `seed ≈ oracle` within ε≈3-5% on EVERY measurable shape** and names "done because
  the aggregate looks good" as failure mode #9 (premature convergence). So run 3 is NOT a restart — it is
  re-opening the inherited champion against the per-shape bar to surface the gaps the geomean buried, climb
  them to parity, freeze+bank (Phase 3), then beat-oracle overtime (Phase 4).
- **Inherited state read through the new bar (the worklist seed):**
  - Per-shape `seed/oracle` is essentially UNMEASURED. The `oracle_cache` ledger key is sparse (a few
    shapes, 3 kernels) and PRE-DATES the run-2 heuristic edits → STALE/untrustworthy (staleness rule). The
    run-2 "victory" rests on a geomean, not a per-shape table. Phase 2 must rebuild the oracle cache fresh,
    keyed by source-hash, and produce a per-shape table.
  - Floor itself has real sub-1.0 per-kernel G (seed/tc_default): softmax 0.960, cross_entropy 0.916,
    rms_norm 0.979, welford 0.975; realistic in-sample-v2 baseline 0.89 (rms_norm 0.776, CE 0.715, long_sum
    0.726 pre-lift).
  - Give-up claims to RE-LITIGATE vs a FRESH oracle (anti-giving-up): rms_norm (2048,2048)=0.871 /
    (1,131072)=0.512 "noise floor"; CE (8192,131072)=0.539 "source ceiling"; welford (262144,7168)=0.69
    TEST; long_sum (4,524288)=0.66 "split-K deferred". Each is failure mode #6/#7/#8 until disproven.
- **Setup:** Step 0+1 (wiring + measurement mechanism + harness sanity) DELEGATED to a one-time background
  agent (keeps hub context clean per the work-order NOTE). Worker (timing-bound) NOT spawned until setup's
  GPU work finishes — one GPU, serial timing. Team `reduction-heuristics` created; macro task list 1→{2,3}
  →4→5→6→7 wired. All spawns pass `model:"opus"`.
- **Step 0+1 VERDICT: GREEN (hub-verified 2026-06-03).** The setup agent completed all substantive checks,
  then was killed by a stall watchdog at its FINAL (redundant) revert-verification step. Hub independently
  re-verified the state: (a) wiring proven — helion.__file__→worktree, codegen+operator edits flow, then
  reverted; (b) bare-seed mechanism sound (configs=[seed] → no autotune, normalized config used, correct,
  stable); (c) 5-way ordering OK — rms_norm 8192×4096 fp32: H-default 0.0955 / H-quick 0.0935 / H-max 0.0946
  / tc-default 0.0944 / tc-max 0.0945 ms (all within ~1.3%, bandwidth-bound shape, tc already near-optimal,
  accuracy=1 all); (d) harness-bias cross-check: hand-rolled do_bench 0.09517 ms reconciles to 0.3% of the
  TritonBench number → NO measurement bias; (e) shapes_v3 validator PASS (0 problems, 9 kernels/331 shapes/4
  transfer); (f) long_sum wired+accurate (accuracy=1 at (256,65536)+(64,1048576) with --reduce-dim 1; the
  triton_sum-accuracy=0 is the operator's OWN kernel, not the Helion path); welford fp32 footgun handled
  (operator hardcodes bf16 → setup confirmed --precision fp32 path). CRITICAL CLEANUP CHECK: ORIGINAL
  checkout git status CLEAN (the temp welford fp32+sentinel edit reverted); worktree shows only the intended
  §6 long_sum wiring + lab files. GPU idle, no orphan procs. ⇒ hill-climb numbers are trustworthy; team launch
  authorized.

- **Phase-1 FLOOR sweep DONE (worker, diagnostic — no edits, no gates).** Median-of-7, 0 correctness fails,
  0 OOMs. Commit 09353012; `_lab/run3_notebook.md` + `_lab/logs/run3/floor_sweep_merged.json`. Overall train
  geomean G(seed/tc_default)=0.977. 6/9 kernels at/above floor (long_sum 1.086, kl_div 1.078, softmax 1.053,
  sum 1.011, rms_norm 0.995, layer_norm 0.993). BELOW floor: **cross_entropy 0.744**, welford 0.946, jsd 0.936.
  27 floor losses, CLUSTERED not flat-tail. Phase-2 oracle worklist worst-first:
  - **(A) cross_entropy — the catastrophe + a likely run-2 OVER-CLAIM.** 8 of 10 worst shapes, ALL looped:
    looped CE at wide vocab is ~2× SLOWER than tc-default on the SAME standard kernel ((8192,128256)=0.520,
    (4096,128256)=0.531, (2048,256000)=0.545); persistent CE (narrow V) beats tc 1.05-1.14. Since tc is 2×
    faster on the same kernel, a 2× strategy provably EXISTS ⇒ SEEDABLE gap, NOT the "source ceiling" run-2
    claimed CLOSED via cross_entropy_online. ⚠️ CONFOUND TO RESOLVE: does `train` benchmark standard
    `cross_entropy` or `cross_entropy_online`? If run-2 "closed" the ceiling on online while train measures
    standard, that's a measuring-the-wrong-thing artifact the geomean hid. (worker tasked: ask code-investigator.)
  - **(B) long_sum (16,2097152)=0.734** — the >2^20 looped tail run-2 deferred as "synthetic-only"; it is a
    TRAIN shape, 27% below floor. Every persistent long_sum row is fine (1.0-1.5).
  - **(C) welford 0.946** — narrow-N persistent w4 (768=0.908,1024=0.942) + wide-N looped-apply (16384=0.862).
  - **(D) jsd 0.936** — narrow-vocab Band-B loss (30522=0.847), recovers to ~1.0 at wide V (opposite of CE).
  - **(E) softmax small-N** (131072,256=0.796) + one wide (4096,16384=0.908); big mid-range wins.
- **Oracle-cache KEY recipe set (ledger-keeper guardian decision).** key = sha256(read(examples/<kernel>.py)
  + to_triton_code(default_config for that shape) + config_spec knob/range dump). EXCLUDES the heuristic file
  (the oracle is what the SEARCH finds, independent of the seed — including it would defeat the cache).
  Corrected the worker's proposal (it had included the heuristic). Safety net: victory-confirm ALWAYS re-runs
  a FRESH FULL oracle, so cache under-invalidation can only mislead during iteration, never at a done-verdict.
- Worker proceeding to Phase 2a: build fresh oracle cache worst-floor-first (CE wide-vocab → long_sum tail →
  welford → jsd → softmax), reading tc-default's generated Triton (free answer key) alongside.
- **Fresh oracle cache batch-1 (13 worst-floor shapes, quick effort) DONE** (worker commit 6f28353f).
  seed/oracle table: SEEDABLE wins (oracle>tc) at CE(4096,50304)=1.576, jsd(8192,30522)=1.196,
  welford(4096,16384)=1.146, softmax(131072,256)=1.147, welford(32768,8192)=1.089; SOURCE-LIMIT candidate
  long_sum(16,2097152) seed/oracle=0.998 BOTH looped (N>2^20 structural cap), oracle/tc=0.735 (seed matches
  oracle → split-K SOURCE opportunity, not a seed miss; to be verified at full effort).
- **CE crossover PINNED** (matched-lever persist-vs-looped A/B): persistent CE wins & beats tc up to ~224KiB
  row, spills 2-4× at ≥256KiB. ⇒ CE = TWO problems: P1 the persist cap fired too early (128KiB); P2 at wide
  V (≥98304) looped is correct but the seed's fixed looped chunk=16384/w32 is ~2× off tc (looped-param +
  possible source ceiling, still open).
- **=== EDIT#1 COMMITTED (worker) → GATE PIPELINE FIRING ===** `MULTILOAD_PERSIST_MAX_BYTES` 131072→245760
  (240KiB); num_load≥2 gate unchanged. Commits dab1eea8 + 8d55a50c. Claim: 3 CE boundary shapes →oracle
  parity (4096,50304 1.576→1.000; 8192,50257 1.066→1.000; 8192,49152 1.025→0.996; oracle beats tc 1.05-1.09)
  + floor recoveries + 8 non-CE byte-identical. **Overturns run-2's "CE wide-vocab=SOURCE ceiling" at the
  boundary** (the seedable claim). Honest open residual: CE(8192,57344)=1.105 (P2 looped-chunk).
  GATES IN FLIGHT (model:opus, fresh, neutral brief, verdicts→ledger as-returned):
  - results-referee [timing] (owns GPU) — reproduce 3 parity deltas + floor + no-regression + correctness.
  - adversarial-auditor [analysis] — is 240KiB a CE-fencing magic number? overfit/wrong-thing/metric-gaming.
  - fact-integrity [analysis] — is `num_load` a hacky op-count proxy? (code-investigator flagged: CE num_load
    =3 counts OPS not operands). Whether the num_load≥2 GATE faithfully encodes the resident-byte property.
  - anti-giving-up [timing] — QUEUED behind referee (one GPU): re-litigate run-2's "wide-CE source ceiling"
    with a FULL fresh oracle on the boundary + wide-V shapes.
  Worker pulled OFF GPU (referee owns queue), tasked with non-GPU work: resolve num_load with code-investigator,
  design P2 looped-chunk lever from cached oracle configs. Oracle-cache key recipe corrected (drop heuristic
  file; key on examples/<kernel>.py + to_triton_code(default_config) + config_spec dump).
- **fact-integrity VERDICT: FAIL (recorded ledger.run3.gate_verdicts).** `num_load` is a syntactic op-count of
  hl.load FX nodes, NOT the resident-byte/re-read-pass property the cap's `num_load≥2` gate needs (CE: 1 row +
  scalar gather + 2 passes→num_load~3 over-counts; rms_norm: x+broadcast weight→num_load=2 but ~1 resident row;
  cross_entropy_online: same workload, different load-node count = style-dependent). Converges with
  code-investigator's "looped CE=2 HBM passes". Cap VALUE move (128→240KiB) endorsed; the GATE is the defect.
  - **HUB ADJUDICATION:** REJECTED the verdict's option (1) "gate on bytes alone" — I verified it forces 11/12
    long_sum train rows (>240KiB, persistent, num_load=1, currently G 1.0-1.5) → LOOPED = regression. The
    single-stream-vs-multipass distinction is PHYSICALLY REAL; num_load is the wrong ENCODING. Directed worker
    to option (2): re-key onto `num_reduction_ops` (reduction-pass count, already computed: CE=2, long_sum=1).
  - **FLAGGED softmax as the sleeper regression risk for EDIT#2:** 2-pass, wide train rows (512,131072=512KiB)
    persistent+at-floor today; a naive num_reduction_ops≥2 gate would cap them → regression. Worker must verify
    the full flip-set empirically (oracle is arbiter) + ask code-investigator whether softmax_two_pass spills
    like CE; if num_reduction_ops≥2 over-captures softmax, escalate to resident-operand-bytes via the
    _fx_trace_tensor_arg_rw_names resolver. Worker designing EDIT#2 (non-GPU) now.
  - **EDIT#1 NOT accepted standalone** — folds into EDIT#1+EDIT#2 combined re-gate. Referee's in-flight
    no-regression + CE-parity reproduction stays valuable (cap value + CE boundary parity unchanged by the gate
    re-key; softmax/long_sum persistent under both old gate and a correct EDIT#2), so let it finish, don't cancel.
- **adversarial-auditor VERDICT: PASS-with-flag (recorded).** Endorsed: diff exact, CE parity vs full verbatim
  oracle (oracle/tc 1.05-1.09 ⇒ seedable not source-ceiling), no metric-gaming, and 240KiB is a PHYSICAL
  threshold not a CE fence (no num_load≥2 curriculum shape in (196.5KiB,256KiB) except dev-only 57344 ⇒ any
  cap value in that window = identical curriculum codegen). FLAG: cap branch is SHARED with softmax — EDIT#1
  silently flips held-out softmax val(2048,49152)=192KiB + test(2048,40960)=160KiB looped→persistent (physics
  says wins, but unmeasured + on val/test).
  - **HUB RESOLUTION (no firewall breach):** softmax TRAIN brackets the flip-zone ((2048,32768)=128KiB persist
    + (1024,65536)=256KiB looped both train) ⇒ derive softmax crossover on TRAIN, held-out flips validate by
    interpolation within the trained envelope. Folds into EDIT#2 re-gate.
- **HUB SELF-CORRECTION on the faithful fact (important):** my first EDIT#2 suggestion (`num_reduction_ops`)
  is ALSO a proxy — it UNDER-counts: rms_norm/layer_norm re-read x across reduce+apply (DO spill at wide N,
  run-2 P/L=2.75@288KiB) but num_reduction_ops=1 ⇒ a ≥2 gate would EXEMPT wide rms/ln → spill → OOS regression.
  num_load over-counts (gathers/broadcasts), num_reduction_ops under-counts (apply re-read). The FAITHFUL gate
  = **HBM read-pass count of the reduction row operand** (code-investigator's "looped CE=2 passes, persistent=1"
  generalized): count distinct re-reads of the SAME host buffer across the reduction region via the roller's
  read provenance (`_fx_trace_tensor_arg_rw_names`). Unifies: CE/rms/ln/softmax = 2 passes → governed by the
  240KiB byte cap; sum/long_sum = 1 pass → exempt, persistent to structural cap (protects the 11 long_sum rows).
  Cap VALUE (240KiB) stays; only the GATE becomes faithful + style-independent. Worker directed to design EDIT#2
  on this property, predict the full train flip-set, DM before committing → then full re-gate.
- **P2 (worker, pre-redirect, commit 629dd16e):** CE looped-chunk A/B — fixed chunk=16384 too small for wide V;
  wide-V CE has a likely SOURCE ceiling even with the right chunk. ⚠️ This is a give-up claim to RE-LITIGATE
  (anti-giving-up + the standard-vs-online CE question): is wide-V CE truly oracle-bounded, or does routing to
  the single-pass cross_entropy_online (run-2's source variant) close it? Tracked separately from EDIT#1/#2.
- **results-referee VERDICT on EDIT#1: PASS (recorded).** All 4 claims reproduced (CE parity vs full verbatim
  oracle 1.000/1.000/1.002; floor G 1.056/1.094/1.056 all clear noise; non-CE byte-identical SHA-confirmed;
  correctness ≤9.5e-7). CAUSAL: monkeypatch cap→128KiB reverts CE seeds to looped floor-loser ⇒ raise is sole
  cause. **EDIT#1 cap VALUE BANKED** (gate-refix still owed). ⚠️ Referee BLOCKED ~9min for the worker's
  run3_oracle.py to exit GPU0 — near-collision. ⇒ **NEW HARD GPU-TOKEN PROTOCOL:** worker must DM
  "REQ-GPU"→wait "GPU-GRANTED"→"GPU-RELEASED"; hub holds the single timing token, never grants during a gate's
  timing. (Also: referee re-flagged the portable-lab-state hygiene items — stale /home/calebkim paths +
  startswith() assert — non-blocking, matches memory.)
- **=== MAJOR PIVOT: wide-CE 'source ceiling' REVERSED by a FULL oracle (worker commit fd50f32f) ===** This is
  the anti-giving-up #7 discipline applied self-reflexively — the worker re-litigated BOTH run-2's "wide-CE
  source ceiling CLOSED" AND its own intermediate "wide-V = 2× source ceiling" with a fresh FULL oracle and
  reversed both. CE(4096,98304) full oracle (1072s, 11 gens): oracle 588us vs tc 557us (oracle/tc=0.947, only
  ~5% off tc — NOT 2× as the quick oracle + coarse grid suggested); seed 955us ⇒ **seed/oracle=1.624, wide-CE
  is 62% SEEDABLE.** Oracle's winning config is a PERSISTENT-PID bundle OUTSIDE the current seed design space:
  reduction_loops=[4096] (SMALL chunk), pid_type='persistent_interleaved', num_sm_multiplier=32, maxnreg=64,
  num_stages=4, range_unroll/num_stages/flattens pipelining, tensor_descriptor indexing.
  - **This re-opens run-2's pid='flat' "principled constant" lock** — but doesn't simply contradict it. run-2
    locked flat via a gated A/B on grid-SATURATED narrow-forward reductions; wide-CE M=4096 is grid-LIGHT (few
    huge rows under-fill the SM grid) — a regime run-2 never probed. Resolution = a NEW workload-keyed
    grid-occupancy branch (grid-light ⇒ persistent_interleaved+sm_mult), NOT flipping the global default.
  - **Hub verified the bundle is Product-A SEEDABLE** (Config constructs with all levers present) — NOT
    inherently autotuner-only. Worker must confirm normalize()+configs=[seed] round-trip survival (run-2 trap).
  - **Sequencing decision: (a) isolate+land wide-CE pid/pipelining FIRST** (largest gap on the board 1.62×,
    generalizes). Guardrails: isolate the 7-lever bundle ONE lever at a time vs full verbatim oracle (don't
    parrot all 7 — find the carrier); grid-occupancy fact via code-investigator; pid change → fact-integrity +
    pid-specific auditor/anti-giving-up confirming no narrow-forward regression where flat genuinely wins.
  - Work renamed to avoid collision: **EDIT-GATE** = the num_load→read-pass-count gate refix (faithful fact,
    independent, lands separately); **EDIT-PID** = the grid-light pid/pipelining branch. Both DM-before-commit.
- **EDIT#2 (num_load→num_reduction_ops re-key, commit 82a0de72): REJECTED by hub (recorded), fact-integrity
  re-firing for independent confirm.** The worker swapped num_load's OVER-counting proxy for num_reduction_ops,
  which UNDER-counts (mirror defect): rms_norm/layer_norm reuse the row via reduce-THEN-apply (re-read x in the
  normalize pass) and SPILL at wide N (run-2's own committed table: rms P/L=2.91@512KiB), but their
  num_reduction_ops=1 (apply pass isn't a ReductionLowering) ⇒ nro≥2 EXEMPTS them ⇒ wide rms/ln forced
  persistent ⇒ 2.9× regression. **Hub caught it by scanning ALL splits** — rms_norm/ln ROBUSTNESS
  (1,131072)=512KiB flip looped→persistent under nro; the worker's "EXACTLY 3 CE flips" was TRAIN-ONLY (rms
  train maxes 64KiB<cap). Neither proxy is faithful (num_load got the right set by luck of over-count; nro gets
  a wrong set). **Faithful fact = `row_reread` boolean** (reduction-input host buffer read in >1 distinct
  pass/region) from the roller's read provenance (_fx_trace_tensor_arg_rw_names). Right set on ALL splits;
  ALSO generalizes the reread-eviction to CE (one fact, double duty). Worker directed to build row_reread
  (EDIT-GATE-v2), scan the FULL 4-split flip-set, DM before commit. **LESSON LOGGED: every flip-set scan must
  cover train+val+test+robustness** (byte-width cap; widest rows live in robustness). EDIT#1 cap value stays
  banked; gate stays num_load (less-wrong) until row_reread lands.
  - Hub self-note: also corrected my OWN earlier num_reduction_ops suggestion in the same motion — the hub is
    not exempt from the proxy trap; verify the fact against physics + all splits, never hand-wave it.
- **EDIT#2 REVERTED by worker (commit 1cb50a6a) — ack'd the reject cleanly.** Gate back to num_load≥2
  (less-wrong placeholder), EDIT#1 240KiB value still banked. Verified post-revert: rms/ln (1,131072)=512KiB
  correctly LOOPED again (G 1.34/1.42). Worker proceeding to row_reread (task #8), holding the edit until
  code-investigator confirms graph topology — "not going to guess a third time" (good discipline).
- **Hub design-anchor sent (pre-empting a THIRD proxy):** worker was leaning toward "count load nodes reading
  buffer B ≥2" — which is num_load again (false-positive on 2 reads in 1 pass; fusion-fragile). Corrected to
  **REGION-MEMBERSHIP**: row_reread = "is host-buffer B read in ≥2 DISTINCT regions (reduction pass(es) +
  post-reduction apply)" — set-membership across regions, not op-count; immune to fusion + single-pass-double
  -read. Eviction de-hack needs finer provenance: per-load-SLOT, 'last' iff its buffer is re-read in a region
  emitted LATER (buffer-identity + emission-order, NOT positional slot literals — de-hacks run-2's positional
  welford rule). **rms_norm is the acid separating-kernel** (num_load=2 fires, nro=1 exempts, row_reread=True
  governs — right where both proxies failed); worker to cite it as proof-of-faithfulness at DM-design time.
- Tasks restructured: #8 = row_reread fact + cap re-key + reread-eviction de-hack (in progress, worker);
  #9 = EDIT-PID grid-light cluster (blocked by #8); both block #5 PARITY. fact-integrity-on-EDIT#2 still in
  flight (independent confirm of the reject).
- **fact-integrity on EDIT#2: FAIL (independent, recorded) — and it CORRECTED THE HUB'S OWN framing.** Confirms
  nro reject AND codegen-proves the key subtlety: **rms_norm loads x with a SINGLE hl.load; the row is reused
  IN-REGISTER across reduction→apply, NOT a 2nd HBM load.** So num_load over-counts, nro under-counts, AND the
  hub's own "read in ≥2 regions" framing would ALSO miss rms_norm (one load region). Gate also found num_load
  itself is unfaithful (softmax_decomposed=num_load=1 for the same 2-pass math → would wrongly exempt) — the
  placeholder is luck-of-over-count, not faithful.
- **=== THE FAITHFUL FACT (saga resolved — one place for respawn-safety) ===** The cap gate's true property =
  **reduction-input REUSE/LIVENESS ACROSS THE REDUCTION BOUNDARY**, NOT any load/region/pass count. Definition:
  the reduction-input tile (host buffer via `_fx_trace_tensor_arg_rw_names` ~608-644) is **consumed by BOTH a
  `ReductionLowering(red_block_id)` AND a downstream apply/store — OR by ≥2 ReductionLowerings.** Captures the
  actual spill cause (the row must stay LIVE across the reduction→consumer boundary, whether re-read from HBM
  or held in-register). Right set on ALL splits: sum/long_sum=False (persist, protects 11 long_sum rows);
  rms/ln/softmax(all variants)/CE=True (governed); kl/jsd MOOT (T2 Band-B R_BLOCK cap dominates, T1 gate never
  reached). Acid separating kernel = **rms_norm** (num_load=2 fires, nro=1 exempts, reused-across-boundary=True
  is the ONLY correct one). Immune to fusion + single-pass-double-load + the softmax_decomposed style-trap.
  - Proxy progression that got us here: num_load (over-count, fusion-fragile) → num_reduction_ops (under-count,
    misses apply re-read) → hub "region-count" (misses in-register reuse) → **consumed-by-reduction-AND-apply**
    (faithful). Three wrong framings, each caught by a separated context (fact-integrity ×2 + hub self-checks).
  - Cleanup owed: delete stale triton.py:494-499 comments (assert reverted "nro>=2 GATE", contradict operative
    num_load>=2 at :523).
  - Eviction (EDIT#3) keys on a DISTINCT but related property: per-LOAD-SLOT, 'last' iff that slot's buffer is
    re-LOADED in a later slot (CE amax/sum HBM re-reads) — buffer-identity + emission-order, de-hacks run-2's
    positional welford rule. (rms_norm's in-register reuse = single load → no per-slot eviction rule, stays
    default — don't conflate cap-liveness with eviction-reload.)
- Worker building the faithful fact (holding for code-investigator's T1 downstream-consumer topology), will run
  the divergence test itself + 4-split flip-set + cite rms_norm proof, DM before commit. GPU free; codegen-only
  work. The substrate is one careful step from genuinely general.
- **code-investigator RESOLVED the topology: row_reread = "≥2 distinct non-root subgraphs LOAD the same host
  buffer" (10/10 exact, computable at fact-build, no new analysis).** This is subgraph-MEMBERSHIP, not the
  "≥2 load nodes" proxy — reconciles with fact-integrity's "single load *within* the reduction subgraph"
  finding: rms_norm's APPLY pass is a distinct rolled subgraph that re-loads x → True. (Worker had drifted
  back toward a load-node-count probe; hub re-aimed it at subgraph-membership/consumer-trace before it built
  the wrong discriminator a 3rd time.)
- **Hub posture self-correction:** I started reverse-engineering rms_norm's subgraph topology MYSELF to verify
  the fact — caught it and STOPPED. That's the fact-integrity gate's job (separated context); the hub becoming
  the fact's co-author defeats the gate. Reset: worker builds gate-ready against 4 probe anchors (rms_norm True
  via a genuine apply-subgraph re-load; sum/long_sum False; softmax-all-variants True/style-independent; keyed
  on the REDUCTION-INPUT buffer not broadcast weight), DMs the design package (fact def + per-kernel values +
  rms_norm subgraph-id proof + 4-split flip-set + eviction-slot derivation) → THEN fact-integrity adversarially
  confirms + referee no-regression + auditor on eviction. Acid test = rms_norm.
- **long_sum(16,2097152) full oracle running bg** (compile-timeout-skipping the 2M configs, slow not hung).
  Disposition: seed/oracle=0.998 = seed-DONE (matches oracle); oracle/tc=0.735 is a SEPARATE split-K
  source-rewrite opportunity, NOT a seed miss. Worker to FLAG (not self-certify) the source-ceiling → hub fires
  anti-giving-up to verify (seed≈oracle + oracle real full-effort + oracle also <tc = structural). GPU-token
  heads-up protocol extended to bg oracles.
- STATE: 5 gate verdicts recorded (EDIT#1: referee PASS + auditor PASS-w-flag; EDIT#2: fact-integrity FAIL ×
  [hub prescreen + independent]). EDIT#1 cap value BANKED, gate awaiting row_reread. Board fully mapped; worker
  on critical path (#8), productively analysis-blocked then building. Awaiting worker's row_reread design DM.
- **CE KERNEL-IDENTITY CONFOUND RESOLVED (worker, commit 0854c7bb):** the harness benchmarks the STANDARD
  `cross_entropy` (run2_measure_g.py:65), NOT cross_entropy_online. So run-2's "wide-vocab source ceiling CLOSED
  via cross_entropy_online" was on a kernel `train` does NOT measure — a measuring-the-wrong-thing artifact the
  aggregate geomean hid, exactly the suspected confound. Standard-CE wide-vocab is SEEDABLE (full oracle 588us
  vs seed 955us = 1.62×; oracle/tc 0.947), the win = reread-eviction (1.31× alone) + persistent-pid cluster.
  Oracle-cache key recipe also corrected to the hub spec (dropped heuristic/device_ir from the key).
- **=== EDIT-GATE-v2 COMMITTED (91dfd8ef) — faithful row_reread fact → GATES FIRING ===** After 3 rejected
  framings, the faithful fact landed: `ReductionFact.row_reread` = "reduction-input host buffer loaded in ≥2
  distinct LOOP graphs (ReductionLoopGraphInfo T1 / ForLoopGraphInfo T2)" — empirically divergence-tested 9/9
  (sum/long_sum=F; rms/ln/softmax/CE/welford=T; kl/jsd=F), reuses _fx_trace_tensor_arg_rw_names provenance (no
  new framework), style-independent. Gate re-keyed `num_load>=2` → `fact.row_reread` (triton.py:512); cap VALUE
  240KiB unchanged; stale nro comments deleted. Hub statically VERIFIED the commit structure (fact defined
  config_spec:174, computed device_ir:1174 wired T1:1036+T2:1243, gate:512). Worker's headline: SEED-BYTE-
  IDENTICAL to num_load curriculum-wide (only kl/jsd cap-gate flips, absorbed by Band-B R_BLOCK → identical
  seeds) ⇒ pure faithfulness fix, zero perf change. rms/ln 512KiB robustness stay LOOPED (the nro regression
  AVOIDED). Tests pass, lint clean.
  - GATES (model:opus, fresh, neutral, concurrent — no GPU contention): fact-integrity [analysis] (acid =
    rms_norm True via genuine apply-subgraph re-load, not a coincidence; style-independence on softmax_decomposed;
    keyed on reduction-input not broadcast weight) + results-referee [timing] (byte-identity curriculum-wide +
    rms/ln 512KiB-LOOPED no-regression + CE parity + 9/9 probe).
- **EDIT#3 reread-eviction (2nd row_reread consumer) = worker designing now (non-GPU).** The actual perf win
  (1.31×/1.19×/1.09× wide-CE) + de-hacks run-2's POSITIONAL welford slot[0]='last'. DISTINCT trigger from the
  cap: per-load-SLOT 'last' iff its buffer is RE-LOADED in a later slot/graph (HBM re-read), finer than
  row_reread (rms_norm in-register reuse = single load → stays default). Worker to REQ-GPU for the A/B (referee
  owns queue now). Gates: auditor (not positional refit) + referee (1.31× reproduces); no fact-integrity (no
  new fact). long_sum full-oracle KILLED (low-pri, seed already==quick oracle).
- **Ledger-keeper: oracle-cache re-key ACCEPTED (commit 02c825e8).** All 14 batch-1 entries re-stamped under
  the heuristic-independent recipe (old_source_hash->new + recipe id recorded for audit); 8 non-CE entries now
  valid+fresh. Staleness discipline correct.
- **Chunk-scaling directive WITHDRAWN (hub).** My "LOOPED_CHUNK 16384->32768" came from the worker's earlier
  quick-oracle/coarse grid; its FULL-oracle ablation (fd50f32f, 5b39b9ad) overturned it — chunk is ~INERT in
  the winning bundle; the real wide-CE levers are reread-eviction (1.31x alone) + the persistent-pid cluster.
  Worker correctly superseded the directive on better evidence and flagged it explicitly. PRINCIPLE AFFIRMED:
  the full oracle (answer key) is the arbiter, not a hub directive — better evidence always wins.
- POSTURE: hub caught up to worker's head (02c825e8); worker runs slightly ahead, disciplined + self-correcting.
  Holding for the 2 EDIT-GATE-v2 verdicts (fact-integrity + referee) + the worker's eviction design DM. GPU
  reserved for referee; worker on non-GPU (eviction impl + pid scoping). 6 verdicts recorded; nothing
  unverified in champion.
- **=== fact-integrity PASS on row_reread — SUBSTRATE IS NOW FAITHFUL (recorded) ===** The inherited champion
  keyed the persist cap on a hacky num_load proxy; it now keys on a provenance-derived property correct FOR THE
  RIGHT STRUCTURAL REASON on all 9 kernels. Graph-dump acid proof: the roller materializes rms_norm's apply pass
  as a separate ReductionLoopGraphInfo (g2, n_red=0) that RE-EMITS the x load → x in 2 loop graphs → True
  (overturned even the earlier 'in-register, no 2nd load' claim — the post-roller reality IS a 2nd load = what
  spills). Style-independent (softmax_decomposed=True where num_load failed); online-CE correctly False. 9/9.
  - **Forward caveat (gate-flagged, relayed to worker):** _compute_row_reread tests 'SOME buffer in ≥2 loop
    graphs', not 'the REDUCTION-INPUT buffer'. BENIGN for the persist cap (can't fire on a real kernel; bounded
    harm). NOT benign for the EDIT#3 eviction consumer (which equates the ≥2-graph buffer with the re-read row)
    → worker directed to tighten EDIT#3 to select the verified reduction-input buffer, not 'the count≥2 buffer'.
  - Comment cleanup owed: triton.py:225-226,532 still say 'num_load placeholder' (operative gate is :512).
  - Awaiting results-referee (byte-identity + no-regression timing) → then EDIT#1(value)+EDIT-GATE-v2(faithful
    gate) ship together as the accepted champion advance, GPU released to worker for eviction A/B.
- **=== CHAMPION ADVANCE #1 ACCEPTED + BANKED (2026-06-03): EDIT#1 + EDIT-GATE-v2 ===** results-referee PASS
  (all 5 claims): **byte-identical curriculum-wide (18/18 shapes — pure-faithfulness fix, provably ZERO perf
  risk)**; rms/ln 512KiB robustness LOOPED (nro 2.9x spill avoided); 3 CE boundary shapes (49152/50257/50304)
  persistent at oracle parity (seed/oracle 0.996/1.000/1.000, oracle/tc 1.05-1.09); correctness clean; 9/9
  row_reread. Combined with fact-integrity PASS + EDIT#1's referee+auditor PASS → ACCEPTED.
  - **Substrate now FAITHFUL**: persist cap keys on provenance-derived `row_reread` bool (consumed-by-reduction
    -AND-apply / ≥2-loop-graph buffer-load), NOT the num_load proxy. Decision: keep BOOL not int (int serves no
    consumer — eviction needs finer per-slot detail, computed separately; int = over-engineering). bool is final,
    no re-gate.
  - **First real per-shape oracle-parity WINS of the run** (3 CE shapes), on shapes the inherited geomean buried;
    overturns run-2's "CE wide-vocab = source ceiling" at the boundary.
  - 7 gate verdicts recorded; champion_advances[0] banked in ledger.run3.
- **🟢 GPU-GRANTED to worker.** Queue (one timing job at a time): (1) eviction A/B (EDIT#3, the 1.31× CE win —
  DM design+receipts before commit, esp. reduction-input-buffer verification per fact-integrity caveat +
  welford-dehack proof; gates = auditor+referee, no fact-integrity), (2) softmax-wide persist-vs-loop oracle
  (new gain), (3) pid cluster #9 (1.62×) + long_sum-tail anti-giving-up. Comment cleanup (triton.py:225-226,532)
  folds into next commit. Worker to flag long_sum "source ceiling" for anti-giving-up (don't self-certify).
- **Leapfrog churn:** worker + hub messages crossed repeatedly (fast worker, async replies) — worker re-asked
  settled items (re-gate EDIT-GATE-v2, bool-vs-int) several times. Hub sent ONE authoritative sync (settled
  list + "committed-evidence/latest-hub-state wins, proceed, don't re-summarize completed work"). Worker now
  has the GPU and a single clear next action (pid decomp). Posture: HOLD, stop preemptive messaging that crosses.
- **⚠️ EDIT-PID keying hypothesis FALSIFIED by code-investigator (load-bearing for the upcoming gate):** the
  "grid-light = few rows under-fill the SMs" fact is FALSE — CE(4096,98304) M=4096 rows = 31-62× SM count =
  heavily OVER-subscribed, not grid-light. Grid size + SM count ARE computable at seed time, but the count-based
  framing is wrong. Investigator's refined signal: **"few-but-long"** (few rows relative to width, each row very
  long). ⇒ when EDIT-PID comes, the pid='persistent_interleaved' win must be keyed on the REAL property (row
  count × row width interaction / per-program work), NOT a grid-light count threshold — and ESPECIALLY not a
  threshold that happens to fence CE (kernel-identity smuggling risk). The pid-decomposition (running now) +
  this falsification together must yield a principled keying property. Hub to scrutinize the EDIT-PID keying
  fact HARD (fact-integrity) — a falsified-then-replaced hypothesis is exactly where a CE-fitted fence sneaks in.
- **EDIT#3 reread-eviction (2nd row_reread consumer) — UNCOMMITTED, design gated by hub, A/B pending.** Adds
  `ReductionFact.reread_buffer_slots` (per-slot provenance, fact-build time — HostFunction.current() unavailable
  in heuristic). Worker's 4-split flip-set CAUGHT a broad blast radius (blanket eviction on ALL rms/ln, at-floor,
  unmeasured) → NARROWED on PHYSICS to `row_reread AND not persistent` (eviction only affects HBM-streamed
  loads; a persistent resident row has no re-stream → eviction MOOT; confirmed neutral on persistent CE boundary
  1.001). HUB APPROVED the looped-gate (it's MORE principled than a CE+welford carve-out — excluding rms/ln
  would be the ad-hoc special-case). Final scope touches ONLY: CE looped wide-V (1.31/1.19/1.09×) + welford
  de-hack + 2 wide rms/ln robustness canaries (general rule + not-catastrophic check, NOT a perf claim). At-floor
  rms/ln/softmax byte-identical.
  - **Eviction-assignment A/B (faithful-vs-faithful):** hub's a-priori "'last' iff buffer read in a LATER region"
    DISAGREES with the oracle at CE slot3 (hub rule → 'last'; oracle/measured winner → 'first': [_,_,last,first,
    last]). Both faithful (buffer-identity+region-order, no positional literal). RESOLVED BY PRINCIPLE: the oracle
    (answer key) beats the hub's a-priori rule — A/B both, ship the oracle-winner. (Reaffirms: answer key >
    hub directive.)
  - EDIT#3 gates when committed: fact-integrity (reread_buffer_slots = re-read REDUCTION-INPUT buffer's slots,
    not a coincidental ≥2-graph broadcast) + auditor (faithful rule not a CE+welford positional refit) + referee
    (CE wins reproduce + welford no-regression vs run-2 positional + rms/ln robustness + at-floor byte-identical).
  - WELFORD WATCH: faithful ['last','first','',''] vs run-2 positional ['last','first','first','first'] differ at
    weight/bias slots. faithful≥positional → ship; regresses → positional weight/bias 'first' was load-bearing,
    investigate (maybe principled refinement: evict trailing broadcast finals 'first'), don't silently lose the win.
- **🟢 GPU-GRANTED — combined session running:** pid lever-decomp (1.62× carrier; keying must avoid the falsified
  grid-light fence) + EDIT#3 A/Bs (CE eviction hub-vs-oracle assignment + welford no-regression + rms/ln
  robustness). Worker producing 2 artifacts: pid decomposition + per-kernel eviction A/B table.
- **LEAPFROG: persistent out-of-order message delivery** caused ~5 rounds of the worker re-confirming settled
  items. Hub posture: HOLD HARD — send nothing until a genuine RESULT lands; every preemptive msg risks crossing.
  Leapfrog rule restated to worker (next DM = results, not re-confirm). All decisions settled; worker has GPU +
  scope + gate plan; nothing pending hub-side.
- **STATE-SYNC RESOLVED (subgraph-membership vs consumer-trace):** what's COMMITTED + gated + banked (91dfd8ef,
  champion advance #1) is the SUBGRAPH-MEMBERSHIP row_reread (buffer loaded in ≥2 loop graphs) — fact-integrity
  PASS + referee PASS, both on THIS version, FINAL. The worker then reimplemented as CONSUMER-TRACE (value
  consumed by ≥2 ReductionLowering OR reduction+bypass-store) — UNCOMMITTED, behaviorally IDENTICAL (same 9/9,
  same seeds), strictly more robust (handles pure-in-register reuse + reduction-input-keyed-by-construction →
  closes the one non-blocking fact-integrity caveat). Decision: **(a)** subgraph-membership stays banked (don't
  re-run its gates); consumer-trace lands as a SEPARATE faithfulness-hardening commit.
- **3-COMMIT SPLIT (avoids the consumerless-fact trap the worker correctly flagged):**
  - **Commit A** = consumer-trace refinement of the CAP fact ONLY (add `reduction_input_reused`, NOT
    `reread_buffer_slots` — the latter is consumerless without eviction → fact-integrity rejects). Gate = tight:
    fact-integrity (consumer-trace predicate + rms_norm acid + 2-falsified-predicates derivation) + referee
    BYTE-IDENTITY only (consumer-trace seeds == 91dfd8ef seeds; perf inherited, no full timing). No GPU.
  - **Commit B** = EDIT#3 eviction (adds reread_buffer_slots WITH its consumer). Gate after GPU A/B =
    fact-integrity + auditor (welford de-hack not positional) + referee (CE 1.31× + no-regression).
  - **WELFORD DE-HACK PROOF (auditor evidence, banked):** run-2 positional slot[0]='last' REPRODUCES for welford
    (re-read buffer x is at slot[0]) but now DERIVED from buffer-identity (reread_buffer_slots=(0,1)); for CE the
    positional rule would wrongly mark slot[0]=labels-gather 'last' while faithful correctly marks slot[2]=logits.
    Identical-welford + correct-CE-extension from provenance-not-position = the de-hack.
  - **EDIT#3 hub-vs-oracle slot3 assignment:** A/B both faithful assignments, ship the ORACLE-winner
    [_,_,last,first,last] (oracle > hub a-priori rule).
- **PIPELINE unblocked:** Commit A's analysis gates run CONCURRENT with the EDIT#3 GPU A/B (no contention). GPU
  FREE now. Worker: Commit A → DM sha (fire A's gates) + REQ-GPU for EDIT#3 A/B + pid decomp in parallel.
- **EDIT-PID scrutiny banked:** code-investigator verified derived num_sm_multiplier=32 MATCHES oracle +
  persistent_interleaved codegens for T1 CE. ⚠️ At the EDIT-PID gate, fact-integrity MUST verify sm_multiplier/
  maxnreg are derived from PRINCIPLED hardware physics (SM count/occupancy), NOT reverse-fit to the oracle's 32
  (p-hacking — banned; derive from theory + answer-key DIFF, never the observed search winner). Grid-light
  keying hypothesis already FALSIFIED (CE M=4096 = 31-62× SMs, over-subscribed); refined signal = "few-but-long".
- TASKS: #8 = EDIT-GATE-v2 cap-gate (banked subgraph ver + consumer-trace Commit A pending); #10 = EDIT#3
  eviction (Commit B, in A/B); #9 = EDIT-PID (blocked on keying). All block #5 PARITY. 7 verdicts recorded.
- **EDIT-PID self-tempered by worker (honest, anti-over-claim):** pid cluster config is shape-INCONSISTENT
  across wide-CE (persistent_interleaved/sm32 @98304 vs persistent_blocked/sm1 @128256 vs persistent_interleaved
  /sm1 @256000) — looks autotuner-fine-tuning not a clean seedable rule. Grid-occupancy hypothesis (5c)
  FALSIFIED (flat@50304 + persistent@98304 both progs/SM=31). Worker decomposed wide-CE 1.62× = eviction(1.31×,
  ablation-isolated, SEEDABLE=EDIT#3) × pid-cluster(~1.24×, TBD). **HUB FRAMING CORRECTION:** 2 of 3 pid configs
  are QUICK oracles LOSING to tc (0.64/0.62) = SUSPECT/under-explored (like the wide-CE quick oracles the full
  oracle already reversed). Shape-inconsistency on quick oracles is a HYPOTHESIS, not a conclusion — FULL oracles
  on 128256+256000 arbitrate. "pid autotuner-only/no clean rule" is an anti-giving-up trigger (#8) → needs the
  full-effort answer key first. Both outcomes honest (consistent config = seedable EDIT-PID; genuinely-different
  = Product-B null). Eviction ~1.3× is the bankable wide-CE seed regardless.
- **-am SLIP caught+fixed by worker:** a `git commit -am` swept uncommitted consumer-trace+EDIT#3 into a notebook
  commit; worker soft-reset + re-committed notebook alone. Hub VERIFIED: tip bb646b5e clean (notebook+harness
  only, no source), orphaned 969653f1 off-branch (harmless, gc-able), EDIT#3 back uncommitted. ⚠️ Commit A is
  ENTANGLED with EDIT#3 in the working tree (triton.py spans both) → MUST hunk-stage (git add -p), NOT -am.
- **=== ALL 3 WORKER BLOCKERS CLEARED → long productive stretch authorized ===** (1) EDIT-GATE-v2 verdicts =
  both PASS/banked; (2) version = (a) consumer-trace as Commit A (hunk-staged: cap+reduction_input_reused ONLY,
  exclude reread_buffer_slots+eviction → avoid consumerless-fact reject); (3) 🟢 GPU-GRANTED (nvidia-smi
  compute-apps empty = authoritative clear; the ps|grep count was self-grep noise — lesson: use --query-compute
  -apps not ps|grep for the GPU check). GPU session order: EDIT#3 eviction A/B → softmax-wide A/B+fresh oracle
  → pid FULL oracles (128256/256000) + lever-decomp (98304). Per-result DMs, not one mega-batch.
- POSTURE: HOLD — worker heads-down on Commit A staging + multi-job GPU session (full oracles = minutes each).
  Next inputs: "committed A <sha>" (→ fire fact-integrity + byte-identity referee, no GPU) + per-job GPU results.

## 2026-05-28
- **Setup.** Created git worktree `reduction-heuristics-autotuner` at `/home/calebkim/helion-new-heuristics/wt-reduction`
  off `reduction-heuristic-plan` HEAD `1ec3193d`. GPUs 1/2/3 idle; pinned `CUDA_VISIBLE_DEVICES=2`.
- **Step 0a DONE (hub-verified).** Import wiring proven (`helion.__file__` -> worktree via PYTHONPATH);
  codegen edit flows into generated Triton (sentinel test, sha 9f3e2398 -> 18ed86ed); worktree clean.
  Canonical setup written to `_lab/SETUP.md`.
- **Architecture note.** `SendMessage` is unavailable in this harness, so the "persistent worker" is
  implemented as fresh worker invocations driven off the durable `_lab/notebook.md` + `_lab/ledger.json`
  (the wip blesses this as lossless). Hub stays in the loop, spawns all helpers, runs independent gates.
- **Step 0b DONE (measurement-harness-verifier, certified).** Bare-seed mechanism sound. rms_norm_fwd
  (2048,4096) fp32: persistent (reduction_loops=[None]), num_warps=4, median 0.03498ms, correctness PASS
  (max_abs 1.9e-6). configs=[seed] -> no autotune (no CSV); invalid seed RAISES; distinct config->distinct
  Triton (looped [512]/warps8 -> 0.0378ms). Canonical scripts `_lab/harness/{bare_seed_run,evidence_block}.py`.
  Commit 38bca573.
  - GOTCHA 1: tritonbench operators resolve to the ORIGINAL checkout (hardcoded meta-path finder); the
    Helion kernel-under-test runs from worktree. `torch_compile_<op>_default` baseline must be added in the
    original checkout (or a worktree meta-path overlay). See SETUP.md "tritonbench edit wiring".
  - GOTCHA 2: normalize() does NOT collapse reduction_loops>=size_hint to None; persistent-vs-looped is a
    CODEGEN fact. Always inspect generated Triton for the loop, not just the normalized dict.
- **Spawn-mechanism decision (per human latitude):** hybrid. Direct Agent calls for persistent worker +
  trust gates; Workflow scripts for parallel non-timing fan-outs (classification/investigation/verification,
  GPU-partitioned 1/2/3); measurement sweeps = serial scripts on one pinned GPU (parallel GPU timing corrupts
  do_bench).
- **Step 1 DONE (harness-integrity, CERTIFIED unbiased).** Hand-rolled standalone harness reconciles with
  TritonBench to <1% both sides at (4096,8192) fp32 (Helion-default +0.52%, tc-default -0.05%); 1 CUDA
  kernel/call each (no hidden host-side split). 5-way sanity (x vs eager): Helion-default 2.89/2.78,
  quick 3.76/3.71, max 3.74/3.72, tc-default 3.75/3.73, tc-max 3.77/3.75 at (4096,8192)/(8192,8192).
  Ordering as expected; no HALT. KEY: Helion default_config picks a LOOPED reduction and loses ~23-25% to
  tc-default -> that's the real seed-quality gap (oracle G_rms_norm~=1.0, no-seed baseline G~=0.77).
  `torch_compile_rms_norm_default` baseline added (orig checkout; patch in _lab/harness/patches/).
  Report `_lab/harness/step1_harness_integrity.md`; scripts sanity_5way.py, crosscheck_bias.py. Commit bbd89997.
- **Step 2 map DONE (code-investigator).** Saved to `_lab/step2_code_map.md` (ReductionFact site, populate
  point, heuristic clone target, registration). reduction_loops: value>=size_hint -> persistent at codegen;
  Triton max_reduction_threads=None; default_config = persistent for N<=4096, looped chunk 4096 for N>4096.
- **Worker invocation 1 DONE -> v1 triton_reduction_tile ACCEPTED as champion.** Implemented ReductionFact
  + populate + TritonReductionHeuristic + registration. Bare-seed G_rms_norm referee-confirmed **0.979**
  (vs no-seed default 0.908, +7.8%). Commits b20b42ea/2c1163b5/7a31a28f.
- **Gates (parallel, GPUs 2/1/3):** results-referee ACCEPT (fresh subprocess/shape; worst -6.3% within
  backstop). adversarial-auditor PASS (overfit gap ~2.1%; flagged 2 defects: PERSIST_MAX is a FENCE at
  in-sample max -> persistent wins to ~256KiB/65536 fp32 elems, costs 1.38x on held-out; (2048,2048)
  narrative wrong). harness-integrity: NO autotuner bug -- the w32 'anomaly' was a bug in our
  oracle_field_diff.py (coupled warps x block; re-benched a fabricated block=1 config). Oracle trustworthy.
  Gate verdicts recorded in ledger.gate_verdicts.
- **Self-fooling caught + corrected** (honest, not cheating): (2048,2048) is a real ~1.4% regression (not a
  tie/artifact); oracle_field_diff.py has a lever-isolation bug. Both corrected in ledger; worker to fix
  notebook + the script next.
- **Worker invocation 2 -> v2 PROPOSED, then MECHANISM REJECTED (split gate, auditor wins).** v2:
  corrections + PERSIST_MAX 64->256KiB + looped warps32 + grid-occupancy branch; widened to sum (wash,
  root-caused to num_load=1) + long_sum (claimed 3.3x win). Commits 700e2bdc/47ae968f/f5a91837/91acb0b2.
- **Gates (parallel, GPUs 2/1):** results-referee ACCEPT (long_sum G~1.03 reproduces, no regressions,
  correctness honest) BUT adversarial-auditor **FAIL on mechanism**: the long_sum win is ENTIRELY from
  num_warps=32, not the looped/grid-occupancy branches. Controlled A/B (warps held EQUAL) shows
  **persistent/w32 beats the shipped looped/w32 on 8/9 long_sum shapes** (up to 2.65x on held-out). The
  grid-occupancy branch premise was a CONFOUND (worker A/B'd persistent/w16 vs looped/w32) and effectively
  fences long_sum. Both worker+referee only compared vs the catastrophic DEFAULT strawman.
- **Decision: REJECT v2 mechanism** (the safety gate working as designed). Champion stays v1-mechanism.
  SALVAGE for v3: corrections, PERSIST_MAX raise direction, num_warps=32 (move to persistent path), kernel
  widening. KEY METHODOLOGY LESSON for worker: A/B every branch vs the best SIMPLE alternative
  (persistent/w32), never vs the default strawman.
- **Worker invocation 3 -> v3 (honest fix). Gates SPLIT again (auditor wins).** v3 deleted the v2
  grid-occ + byte-fence branches; persistent workhorse + rnumel-based w32 ramp + structural looped tail
  (only above the 2^20 compile cap). long_sum 1.018->1.10 (+8%), O=1.005 (>1.0!), no regression, and
  v3==persistent/w32 (proving v2's branches didn't earn their place). Commit 2d087e78.
- **Gates (parallel GPUs 2/1):** referee ACCEPT. auditor FAIL on a NEW subtler fence: the `num_load==1`
  condition on the w32 gate is INERT in-sample (0/27 change), FALSE physics (matched-pair A/B: w32 keyed
  on rnumel for BOTH num_load), and HARMFUL out-of-sample (rms_norm/layer_norm large-rnumel want w32 but
  the gate gives w16, -30-40%). A curriculum-split fence dressed as physics.
- **Decision: REJECT the num_load gate; ACCEPT the rest of v3.** Surgical v4 fix: gate w32 on
  rnumel>16384 ALONE (byte-identical in-sample so O~1.005 holds; recovers 30-40% on held-out large-rnumel
  multi-load). LESSON for worker: gate on the DIRECTLY-measured property (rnumel for warps) and TEST the
  gate where it actually fires (synthetic/OOS large-rnumel multi-load), not just in-sample.
- Next: WORKER invocation 4 (surgical v4) -> then a FOCUSED auditor re-check (fence gone + no regression
  + OOS recovered). If PASS, v4 becomes champion (O~1.005 over 3 kernels). Then iter 5 = Product B.
- **Worker invocation 4 -> v4 DONE (surgical, one line).** Deleted the `num_load` condition; w32 step now
  gates on `rnumel > 16384` ALONE in `_num_warps`. Verified: (1) **in-sample byte-identical** —
  AUDITOR_gate_inert_proof.py = **0/27** mismatch, so O~1.005 + per-kernel G unchanged + correctness PASS
  (seed spot-checks rms_norm(2048,16384)=w16, long_sum(8,131072)=w32, both persistent). (2) **OOS recovery**
  — live v4 emits w32 for all held-out large-rnumel rms_norm(nl=2)/layer_norm(nl=3); measured w32/w16:
  rms_norm (16,131072)=0.617 / (16,262144)=0.616; layer_norm (16,131072)=0.641 / (16,262144)=0.540
  (recovers 30-46%). (3) **matched-pair physics** re-confirmed (AUDITOR_numload_warps_ab.py): num_load=2
  ALSO wants w32 at rnumel=131072 (w32/w16=0.57) -> w32 is rnumel-driven, num_load-agnostic. num_load/
  num_store kept in ReductionFact as DATA, just not gated. notebook+ledger updated, committed. NEVER pushed.
- **v4 auditor re-check: PASS -> v4 is the ACCEPTED CHAMPION** (O~1.005 over {rms_norm,sum,long_sum};
  generalizes to multi-load large-rnumel). Gate now rnumel-only; fence gone; no new cliff.
- **Product B DONE (iter 5, GPUs 2+3).** Seeded vs unseeded quick-autotune, N=3 seeds, cold cache, 4 shapes.
  HEADLINE Slice-2 time-to-95%: seeded **1.5-1.8x** less wall-clock on 3/4 shapes (rms_norm (2048,16384)
  1.78x, (8192,8192) 1.70x, sum (2048,16384) 1.52x; long_sum ~tie at noise floor but big early-budget win).
  Slice-1 gen0/gen1 seeded clearly ahead; full-budget guardrail passes (no regression). Curve shifts
  up-and-left. Commit 057f70bd; traces in logs/productB/.
- **CRITICAL BUG found by Product B: persistent seed silently DEGRADED on autotuner injection.**
  reduction_loops=[None] flat-encodes to looped-4096 (ReductionLoopSpec._encode_flat_value maps
  None->min(next_pow2(rnumel),4096); round-trips to None only if >=size_hint -> false for rnumel>4096). So
  the injected seed loses its DOMINANT lever (persistent), keeping only num_warps. Product-A bare-seed path
  (configs=[seed]) UNAFFECTED. => Product-B 1.5-1.8x is a LOWER BOUND; fixing the round-trip should enlarge it.
- Next: code-investigator (read-only) scopes the round-trip fix blast radius BEFORE touching core autotuner
  code. Then worker fixes _encode_flat_value + re-runs Product B. Then widen Product A (layer_norm-fwd next:
  num_load>=2, benefits from v4 w32 fix; then softmax = T2).

## 2026-05-29 (cont.)
- **layer_norm-fwd ACCEPTED** (referee+auditor PASS). 4-kernel active set {rms_norm,sum,long_sum,layer_norm};
  v4 heuristic byte-identical (0 lines changed); G_layer_norm=0.99 (+12% vs default); O_4kernel=0.9997.
- **Round-trip fix ACCEPTED** (auditor PASS). config_spec `_encode_flat_value(None)` .default()->.high so a
  persistent seed round-trips into the autotuner. Tests green (87/107/24); Product A byte-identical; gen0 seed
  now persistent; autotuner not degraded. Product B GREW: Slice-2 time-to-95% rms_norm (2048,16384) 1.78->1.89x,
  (8192,8192) 1.70->1.93x; long_sum dip at M=8 is noise-floor (flips to 2.17x at M=256). Commit 664a9524.
- **T2 support fully mapped** -> `_lab/t2_code_map.md`. KEY: T2 reduction axis is NOT a reduction=True
  block_size; found via ReductionLowering.block_index filtered against grid_block_ids (load-bearing for jsd).
  Plan: new register_user_tiled_reductions() populate (guarded if not reduction_facts), gate on
  reduction_facts==1, knob=block_sizes[red_idx], reuse rnumel/warps logic, Band-B watch for jsd (num_store=2).
- Next: WORKER widens to T2 (softmax_two_pass, kl_div, jsd) per t2_code_map. Then gate (auditor esp. on the
  gate-broadening no-regression + Band B). Then Band C (welford). All gates green so far; loop healthy.

## 2026-05-29 (endgame)
- **v5 ACCEPTED** (T1+T2, Bands A+B, 7 kernels, O=0.9982); **layer_norm earlier**; **v6 ACCEPTED**
  (cross_entropy + multi-load persist cap + welford decline; 8 seeded kernels, O=0.9874; welford out-of-scope).
  All gates PASS. Forward curriculum (8 seedable) complete.
- **Product B FULL (8 kernels): median time-to-95% = 1.94x** (cross_entropy 6x, jsd 2.9x, softmax 2.4x);
  referee ADMITTED. **VALIDATION sweep: generalizes, no overfit** (4/8 kernels beat in-sample OOS; the 2
  negative gaps = documented source ceilings).
- **Codegen-knob workstream EXHAUSTED with a verified negative** (consolidation auditor): pid_type=confound
  (flat wins everywhere); indexing=no matched win; eviction=real +11-23% but AUTOTUNER-ONLY (mutually
  contradictory per-slot patterns, no seedable rule); M-block=regime-conflict fence. The oracle Gs are REAL
  (sum oracle 1.75) but the residual is NOT seedable -> it's Product-B territory. **v6 at its true
  deterministic-seed Product-A ceiling.**
- **SOFT-CONVERGENCE FLAG (non-blocking, for the human):** the 8-kernel forward Product-A seed has reached
  its seedable ceiling (O~0.99, generalizes; remaining headroom is autotuner-only). You may consider this
  milestone "done" for the forward reduction-seed. Still grinding: welford (Band C, its own treatment) is the
  last untried in-curriculum angle; then the terminal TEST read + generalization report. (Backward Band D is
  explicitly deferred / out of scope for this run.)
- Next: welford Band-C attempt (keyed on is_structured_combine, correctness-first; keep out-of-scope if not
  generalizably+correctly seedable). Then FREEZE + terminal TEST read on the final champion.

## 2026-05-29 (FORWARD CURRICULUM COMPLETE — milestone)
- **v7 ACCEPTED (both gates PASS): welford Band-C seeded** via is_structured_combine (proven-generalizable
  structural signal; built a different structured-combine -> gate fires). welford 0.526->0.894 (+70%), CORRECT
  at non-pow2 incl PRIME N. All 9 forward kernels now seeded. O_in-sample=0.9765. Commit 53ed8762.
- **TERMINAL TEST read DONE (ledger-keeper, once).** In-sample->TEST geomean gap -0.114 (O_TEST 0.8628), NOT
  broad overfit: 8-kernel gap only -0.036; cross_entropy/kl_div/jsd BEAT in-sample on TEST. Dominant driver =
  welford -0.498 at prime/poorly-factored N (correctness forces combine=largest_pow2_div(N)->1 at prime N;
  fast masked tile is numerically WRONG -> a welford-KERNEL-STRUCTURE limit, not a seedable miss).
  rms_norm/layer_norm -0.13/-0.15 = tiny-M sub-25us noise-floor edges (at-ceiling on grid-occupied shapes).
- **Fresh-oracle re-validation: seed/oracle = 1.007 geomean** (within 0-1.6% everywhere) -> oracle hasn't
  drifted, champion holds, seed at the DETERMINISTIC-SEED CEILING. `_lab/FINAL_REPORT.md` written (commit 1de57007).
- **DELIVERABLES COMPLETE:** Product A (9 forward kernels, O~0.98, at ceiling, generalizes) + Product B
  (median time-to-95% 1.94x). Adversarial gates caught+rejected 5 cheats/confounds (v2 looped/grid-occ, v3
  num_load fence, pid confound, indexing no-win, M-block regime-conflict) -- the safety mechanism worked.
- **STATUS for human (non-blocking):** the defined forward reduction-seed task is COMPLETE & at ceiling.
  Genuinely-remaining angles are out-of-scope (backward Band D deferred) or not-seedable (codegen eviction =
  Product-B/autotuner; welford prime-N = kernel-structure). Continuing per never-stop with a GENERALITY
  STRESS-TEST: does frozen v7 seed NEW (non-curriculum) forward reductions well out-of-the-box?
- **GENERALITY STRESS-TEST PASSED (strong).** Frozen v7 seeds BRAND-NEW kernels well out-of-the-box: simple
  softmax (T1 whole-row, a different code path) G=0.881 (+34%), softmax_decomposed 0.865 (+34%), both correct;
  jagged_mean/jagged_softmax correctly DECLINE (dynamic rdim). Only residual = disclosed tiny-M ceiling. NO new
  gap. helion/ unchanged. Commit 5eb23859. -> Strong evidence the heuristic GENERALIZES to kernels it wasn't built on.
- **HARNESS-INTEGRITY RE-CERT: still unbiased** (hand-rolled vs TritonBench <1% on rms_norm + cross_entropy;
  headline G's match ledger; GPU2 had a co-tenant mid-run -> GPU pinning vindicated). Whole result body trustworthy.
  Commit be8210b3.
- **CODE REVIEW: APPROVE w/ minor fixes** (design/evidence/generalization/correctness PR-ready; 3 mechanical
  blockers + 2 cleanups). **PR-READINESS FIXES applied + verified byte-identical (9/9 seeds, sha256 identical):**
  return-annotation, ruff format, PIE804, dead _m_extent removed, boolean simplified, + a new unit test
  (TestTritonReductionHeuristic). ruff/pyrefly clean; tests 24/107/46/10 green. Commit d63faba7.
- **DELIVERABLE COMPLETE + MERGEABLE.** Product A (9 forward kernels, O~0.98, at deterministic-seed ceiling,
  generalizes to TEST + brand-new kernels) + Product B (1.94x time-to-target). Harness re-certified. Code PR-ready.
  Final report `_lab/FINAL_REPORT.md`. Adversarial gates caught+rejected 5 cheats/confounds across the run.
- Next: final comprehensive adversarial audit of the COMPLETE v7 (whole-curriculum overfit/cheat/fence sweep +
  FINAL_REPORT claim check). Then the forward reduction-seed task is at its honest end (Band D backward = wip-deferred,
  out of scope; codegen-eviction = autotuner-only; welford prime-N = kernel-structure). Soft-convergence: FORWARD COMPLETE.
- **CAPSTONE AUDIT of complete v7: PASS** (honest, general, coherent routing, correct; multi-load cap generalizes to
  wide softmax = real physics). Surfaced a missed seedable lever: welford apply-tile should be capped/looped at wide N.
- **v8 ACCEPTED (both gates PASS) — FINAL CHAMPION.** welford apply-tile cap + coupled combine cap: (262144,4096)
  0.706->0.757 (block_sizes ceiling closed), welford 0.894->0.9105, O 0.9765->**0.9786**. 8 kernels + welford small-N
  byte-identical; correct at non-pow2 + PRIME N. The capstone's "27% seedable" was itself an overstatement (apply-cap
  alone +2.5%; the 0.968 oracle is autotuner-only codegen knobs, proven by knob-isolation) -> CORRECTED in report+ledger.
  Commit daca67c4 (+ verdicts ca9d76d2). FINAL_REPORT finalized to v8 (TEST carried from v7 per read-once; v8's
  welford-large-N change doesn't touch the dominant prime-N TEST driver).
- **=== RUN COMPLETE (forward reduction-seed task at its honest ceiling) ===**
  Product A: v8, 9 forward kernels, in-sample O=0.9786, generalizes (TEST -0.036 ex-welford; +34% on BRAND-NEW kernels;
  fresh-oracle seed/oracle=1.007). Product B: median time-to-95% 1.94x (~half the autotune budget). Harness re-certified
  unbiased. Code PR-ready (lint/type/tests green + unit test; byte-identical-verified fixes). Adversarial chain
  caught+corrected ~7 cheats/confounds/overstatements. All residuals are kernel-source (cross_entropy online-softmax;
  welford prime-N count-logic) or autotuner-only codegen knobs (eviction/indexing/pid) = Product-B territory, NOT
  deterministic-seedable. NO in-scope seedable Product-A angle remains.
- **SOFT-CONVERGENCE FLAG (strong, non-blocking, for human):** the defined forward reduction-seed task is COMPLETE +
  at ceiling + exhaustively verified. The next workstream (backward kernels, Band D) is EXPLICITLY wip-deferred =
  a scope decision for the human. Durable state (ledger/notebook/FINAL_REPORT/harness) is ready to resume Band-D or any
  scope expansion losslessly. Not stopping on my own judgement; surfacing the converged signal per the manager protocol.
- **GENERALITY (NEW REDUCTION OP) PASSED — final/strongest dimension.** Frozen v8 seeds max/min (a different
  accumulator) out-of-the-box: G_seed~1.0, EXACT correctness, op-agnostic levers confirmed (keys on
  size_hint/num_load/bytes, not the op). helion/ byte-identical. Commit 46ba7059. => Heuristic generalizes across
  ALL THREE dimensions: shapes (TEST -0.036 ex-welford), structures (softmax T1<->T2 +34%), reduction ops (max/min).
  No gap. This conclusively validates "general heuristics."
- **FORWARD TASK EXHAUSTIVELY COMPLETE + AT CEILING + GENERALITY-PROVEN (3 dims).** No in-scope seedable Product-A
  angle remains; all generality dimensions validated. The only further work is wip-DEFERRED (backward Band D) =
  a human scope decision. Durable state ready to resume losslessly. Holding at the converged forward milestone.

## 2026-05-30 (RUN 2 begins — new goals, same orchestration)
> Run 2 works in a SEPARATE worktree `wt-reduction-2` (branch `reduction-heuristics-run2`) off the v8 tip
> `25561778`. `_lab/run2_notebook.md` is the live run-2 reasoning trace; run-1's notebook.md/FINAL_REPORT/
> HANDOFF are historical reference. `wt-reduction` is read-only reference — do NOT edit it.
- **Setup DONE.** Worktree `wt-reduction-2` off v8 tip `25561778`; `_lab/` (165 tracked files) inherited.
  GPUs 0(1GB co-tenant)/1/2/3 — 1/2/3 idle; pin per-job, never 2 timing runs on one GPU.
- **Step 0 wiring RE-PROVEN on wt-reduction-2.** (1) helion+examples resolve to wt-reduction-2 — note
  `wt-reduction-2`.startswith(`wt-reduction`)=True, so run-1 scripts that hardcode the old path +
  `sys.path.insert(0, OLD)` would SILENTLY run old code; run-2 scripts pin `wt-reduction-2` exactly and rely on
  PYTHONPATH (no sys.path.insert). (2) codegen flows. (b) heuristic edits flow: sentinel in `_num_warps`
  changed live seed w8→w16, reverted. (c) tritonbench rms_norm operator resolves to ORIGINAL checkout w/
  `torch_compile_rms_norm_default` present. (3) bare-seed rms_norm(2048,4096) v8 persistent/w8, no-autotune,
  used, correct(1.9e-6), 35.46us/1.2% spread.
- **Step 1 sanity GREEN.** rms_norm(8192,8192) fp32 GPU1: default 336.3 / seed(v8) 252.6 / tc-default 250.3 /
  tc-max 248.4 us → G_seed=0.991, G_default=0.744, seed/tc_max=1.017. Reproduces run-1 exactly; no HALT.
- **Ledger seeded to v8 floor** (champion O=0.9785; `run2` marker added). >10% per-kernel referee-confirmed
  regression backstop in force across ALL kernels+shapes (incl in-sample-v2 + new kernels).
- Next: GOAL 1 (welford source fix + re-derive Band-C), then interleave Goal 4 (in-sample-v2) → Goal 2
  (codegen knobs) → Goal 5 (new-kernel generality). Phase II (Goal 3) after Phase I settles.
- **GOAL 1 DONE + BOTH GATES PASS (2026-05-31).** welford source bug fixed + Band-C re-derived. Commits
  ddf8fc34 (source `Tn=(tile_n.index<n).sum()`), 43492809 (band-C: independent byte caps, deleted
  divisor+coupling). welford in-sample G(orig4) 0.911→0.926; prime-N 1543 0.082(WRONG)→0.958(CORRECT+FAST);
  1536 +6.7%, 2560 +12.2%; 8 non-welford kernels BYTE-IDENTICAL. Referee ACCEPT, auditor PASS (fix real,
  gate structural+generalizes to 2 synthetic kernels; FLAG: byte-cap values curriculum-fit→Goal 5).
  Source fix ships w/ deliverable (also fixes un-seeded default at prime N). Confirmed v8 welford oracle was
  divisor-confounded. Big Goal-2 welford codegen residual: N=4096 seed 0.76 vs oracle 0.961 (TD+eviction).
- Next: GOAL 4 (in-sample-v2 shapes) → GOAL 2 (codegen knobs, big welford+small-N headroom) → GOAL 5.
- **GOAL 4 DONE.** in-sample-v2 split added to list_of_kernels.md (firewall-validated; snapshot _lab/
  list_of_kernels_run2.md) + baselined (overall G_seed 0.890). Commit 5db6d42b.
- **GOAL 2 (eviction) DONE + BOTH GATES PASS (2026-05-31).** Commit 136d187a + 5b4133dc. Found the
  separating workload property run-1 missed: per-load cache RESIDENCY. num_load==1 stream (sum/long_sum) →
  'first' (sum +29%, long_sum +16% geomean, 0 regressions); is_structured_combine re-read (welford) →
  ['last']+['first']*(n-1) (welford 4096 0.760→0.951, +6.68% geomean, small-N neutral). REJECTED (recorded):
  softmax re-read (−10% at 4096,2048), rms/ln x-only (noisy), TD-on-welford (OOM). Referee PASS (full-curric
  matched A/B + noise rechecks), auditor PASS (workload-keyed, 8 byte-identical, really-used). FLAG:
  is_structured_combine eviction welford-only → Goal 5.
- **GOAL 6 (device_ir) partial DONE.** Commit c07d9751: _count_reduction_workload int-equality caveat
  comment + symbolic-fallback symmetry at dtype site (behavior-neutral). REMAINING Goal-6: rms_norm TEST G
  regen + welford TEST re-read (consolidated TEST pass at Phase-I end).
- IN FLIGHT: in-sample O re-measurement (Goals 1+2 lift, GPU1/2); codegen-knob explorer (num_stages/TD/
  range_* on rms/ln weak shapes, GPU3). Then: finish Goal 2 remaining knobs + pid_type-explicit, Goal 5
  (new kernels), then Phase II Goal 3.
- **PRODUCT-A MILESTONE: in-sample O 0.9786 → 0.9980 (+2.0%)** after Goal1+Goal2-eviction. Seed now ≈
  torch.compile-default parity on the geomean. Drivers: sum 0.937→1.019, welford 0.911→0.975, long_sum
  1.099→1.138 (all eviction / Goal-1). rms/ln/softmax/kl/jsd/ce ~unchanged (byte-identical; ±noise).
  Residuals: CE (8192,131072) 0.539 = SOURCE ceiling (Goal-5 online-logsumexp); rms_norm (2048,2048) 0.871
  + small-N (codegen-knob explorer running). CHAMPION advanced (O improves, both gates passed, no >10% regr).
- **GOAL 2 COMPLETE (2026-05-31).** WINS: load_eviction_policies (sum/long_sum 'first', welford re-read [last,first]
  → in-sample O 0.9786→0.9980) + pid_type='flat' explicit (principled constant, behavior-neutral). NULL (honest,
  matched-lever A/B recorded w/ raw numbers in ledger.run2.codegen_knobs_other + _lab/logs/run2/knob_explore.json):
  num_stages (noisy, regresses ln 256,5120 -5%), tensor_descriptor (doesn't engage/OOMs), range_* (no win).
  KEY: rms/ln in-sample-v2 "weak" shapes (256-1024 rows) are NOISE-FLOOR (fresh default G ~1.0-1.13, not 0.75-0.88)
  — seed is at tc-default parity on reliably-measurable shapes. Genuine residuals are kernel-source (CE wide-vocab
  → Goal-5) / out-of-scope (long_sum split-K) / irreducible (welford wide-N codegen-OOM). pid lock covers new
  forward kernels (Goal 5) too.
- Next: GOAL 5 (new-kernel generality probes — structured-combine worker running on GPU2; also validates the
  is_structured_combine + eviction generality flags) → then Phase II GOAL 3 (Product B) once Phase I settles.
  Remaining Goal-6: consolidated TEST re-read (welford + rms_norm G).
- **GOAL 5 + GOAL 6 DONE → PHASE I COMPLETE (2026-05-31).** Goal 5: structured-combine generality VALIDATED
  (standardize within 1.2% of best everywhere — resolves the welford-only flag); multi-load + Band-B shown
  already-multi-kernel; cross_entropy_online (single-pass, verified) closes the wide-vocab SOURCE ceiling
  ((8192,131072) 0.539→0.956, CE in-sample 0.917→0.975, regime-best 0.995). Goal 6: device_ir robustness +
  consolidated TEST re-read DONE & RE-LOCKED — **welford TEST 0.396→0.892 (PRIME 0.082-wrong→0.905-correct+
  fast); rms_norm TEST 0.828→0.841 (0.992 excl noise-floor); TEST O 0.863→0.946.**
- **PHASE-I HEADLINE: in-sample O 0.9786→0.998 (regime-best ~1.005); TEST O 0.863→0.946; in-sample↔TEST gap
  0.115→~0.05.** Heuristic (triton.py) FROZEN for Phase I.
- **PHASE-II PREREQ VERIFIED:** the run-2 seed (load_eviction_policies + pid + reduction_loops/block_sizes)
  survives the autotuner flat-encode round-trip PRESERVED (welford/sum/long_sum) — Product-B seeded arm carries
  the eviction win (no round-trip bug; run-1's persistent fix in place).
- IN FLIGHT: capstone adversarial auditor (whole run-2 deliverable; gates Phase II). NEXT: Phase II Goal 3a
  (budget reduction) + 3b (beat max-effort, multi-seed portfolio) once capstone PASSES.
- **GOAL 3 (Product B) DONE + GATED → RUN 2 COMPLETE (2026-05-31).** 3a BUDGET REDUCTION validated on 3
  kernels (welford/softmax/cross_entropy): seeded-QUICK matches unseeded-FULL optimum within 0.1-0.9% → drop
  full→quick budget (welford ~30x wall-clock reduction to the same optimum). 3b BEAT-MAX-EFFORT = HONEST NULL
  (welford 4096 hard-coupling + sum Band-A control, N=5/arm: both arms reach optimum 5/5 at full; seed within
  1.3% = at ceiling). A preliminary incomplete-data "beat" was CAUGHT + corrected (anti-lucky-run discipline).
  Product-B auditor PASS (seed-injection genuine, 3a real, 3b null honest, no over-claim, no cherry-picking).
- **=== RUN 2 COMPLETE — all 6 goals delivered + independently gated ===** in-sample O 0.9786→0.998, TEST O
  0.863→0.946, prime-N welford 0.082(WRONG)→0.905(correct+fast). Welford fix + simplified Band-C; eviction
  (overturns run-1's "autotuner-only"); pid owned; codegen-knobs honest null; in-sample-v2; generality
  (standardize + cross_entropy_online closing the wide-vocab source ceiling); device_ir robustness; TEST
  re-locked; multi-seed plumbing. Product B: 3a budget reduction (the win) + 3b honest null. FINAL_REPORT_run2.md.
  ruff/pyrefly clean, tests pass, 8 non-touched kernels byte-identical, capstone+Product-B auditors PASS. NEVER pushed.
- **PR-READY (2026-05-31).** Full suite 187 passed/24 skipped/34 subtests; ruff+format clean; pyrefly clean
  (heuristic); diff vs v8 = 8 files +527/-123 (shippable). SOFT-CONVERGENCE FLAG (strong, non-blocking): run 2
  is COMPLETE + at ceiling + exhaustively gated. Remaining is wip-DEFERRED (bf16 expansion; backward Band-D =
  human scope decisions) or diminishing-returns (more 3a kernels — in progress as never-stop continuation).
  Durable state (ledger/run2_notebook/HUB_LOG/FINAL_REPORT_run2/harness/logs) ready to resume losslessly.

## 2026-06-03 (RUN 3 cont. — EDIT#3 gating + EDIT-PID re-opened by anti-giving-up)
- **EDIT#3 reread-eviction COMMITTED (a62e26da) → 3 GATES FIRING.** Faithful provenance eviction (reread_buffer_slots = HBM-re-read AND reduction-input buffer's slots; de-hacks run-2 positional slot0='last'). Ship table: CE 1.31/1.19/1.08× (de-hack-attributable, pos counterfactual fails), welford 1.29× (==positional BYTE-IDENTICAL, row=slot0, zero regression), **rms_norm(1,131072) NEW 1.07× win** (overturns run-2 "rms no clean eviction rule"), layer_norm tie-within-noise. UNIFORM rule (hub ruled: carving ln = identity fence; uniform faithful rule wins). slot4 uniform-'first' (passenger, perf=oracle's 'last', no special-case). Gates: fact-integrity (reread_buffer_slots/adv3) + auditor (not positional refit) + referee (wins+no-regression+tiny-shape-noise check). row_reread bool unchanged/accepted.
- **=== anti-giving-up FAILED the EDIT-PID decline — the run's key supervisory moment ===** Worker (and hub, drifting toward "lowest-leverage bound-it") were about to retire the wide-CE pid residual as "autotuner-only/no clean rule." Gate BLOCKED it:
  - #8 trap on 3 CORRELATED points: M/SM (31,62 vs 15.5) + row-bytes (512K,512K vs 1M) + chunk-count (6,8 vs 16) EACH separate interleaved@32 from blocked@4 = MISSING DATA not no-rule. Worker discarded grid-hypo on a FALSE test (progs/SM for flat-vs-persistent, not interleaved-vs-blocked-within-persistent).
  - Disposition A UNMEASURED: 1.124× decomp ran ONLY on (4096,98304); ZERO interleaved@32-vs-flat on 128256/256000. Decline-trigger was a guess.
  - Gains LARGE: 38%@98304, 19.5%@128256 (non-monotone, tracks variant not width).
  - oracle/tc<1 does NOT license stop (caps seed-vs-tc never seed-vs-oracle). [worker RIGHT that blocked@4@256000 is stable real, source-ceiling real → cross_entropy_online = separate Product-A-via-source opportunity.]
  - MANDATORY: matched-lever {flat, interleaved@32, blocked@4} on all 3 wide CE. beats-flat-everywhere → build EDIT-PID(A) coarse rule; loses@256000 → add 2-4 shapes to DE-CORRELATE the 3 properties, re-key (prior: row-bytes≥1MiB→blocked else interleaved). Decline ONLY if interleaved net-negative AND no property predicts. GPU-PRIORITY (38% stakes). sm_mult must be PHYSICS-derived not fit-to-32 (oracle said blocked@4 on widest → 32 not universally blessed).
- **Hub self-correction logged:** the gate corrected the HUB too — my "lowest-leverage, bound it" was functionally under-claiming. The separated gate caught producer+hub about to retire a 38% seed<oracle gap on a false no-rule. This is the run's thesis working.
- **Per-shape scoreboard so far:** VICTORY (seed≈oracle): 3 CE boundary (49152/50257/50304); softmax-wide 256K/512K. EVICTION-banked (pending EDIT#3 gates): CE wide-V 1.31/1.19/1.08×, welford 1.29×, rms_norm 1.07×. OPEN: CE wide-V pid residual (38%/19.5%/6.5%, EDIT-PID mandatory A/B); EDIT#4 welford apply-cap; jsd Band-B ~1.20; softmax small-N ~1.15; long_sum-tail (seed≈oracle<tc, anti-giving-up pending); BROAD at-floor→at-oracle confirmation on the 6 at-floor kernels (NOT yet confirmed at parity, only at floor). 8 gate verdicts recorded.
- Context ~50%; durable state current; respawn-safe via this log + ledger.run3.

## 2026-06-03 (EDIT#3 gates back + a WORKING-TREE-DRIFT hazard caught)
- **EDIT#3 analysis gates BOTH back (verdicts stand on the IMMUTABLE commit a62e26da):**
  - fact-integrity PASS — reread_buffer_slots faithful; gated via git-archive a62e26da (defended vs tree churn); own separating kernel adv_sep + negative control; de-hack verified at generated-Triton level (CE 'last' on logits/slot2 not labels/slot0). AND-condition (>=2 loop-graph AND reduction-input) load-bearing.
  - adversarial-auditor PASS-WITH-FLAGS — CE wins 1.31/1.19/1.08x REAL + buffer-identity-attributable (positional refit refuted, 30% gap; positional regresses boundary 0.970). welford byte-identical=layout coincidence. uniform rule no fence. slot4 real passenger. NO metric gaming. **2 FRAMING HEDGES REQUIRED:** (a) rms_norm 1.07x = sub-noise-floor + placement-non-discriminating (all evict variants tie ~21us) → "robustness canary not-slower" NOT a headline win; does NOT overturn run-2 (the not-persistent gate excludes the larger-M shapes run-2 found regressing). (b) layer_norm = within-noise but leans slightly NEG (seed slowest arm), shipped uniform to avoid a carve-out. CE wins stand as the real deliverable.
- **⚠️ WORKING-TREE DRIFT HAZARD (caught, being fixed):** despite the worker's "clean a62e26da" reports, hub verified the LIVE working tree was HALF-REVERTED (triton.py: 0 reread_buffer_slots = EDIT#3 eviction MISSING; M/MM staged). The worker flagged env instability ("edits reverted between operations / linter touched triton.py"). COMMIT a62e26da is INTACT+correct (4/4/1 reread_buffer_slots) — only the working tree drifted. KEY DE-RISK: the analysis-gate PASSes examined the IMMUTABLE commit, so they STAND regardless of tree drift. Directed worker: hard-reset helion/ to HEAD + rm __pycache__, verify it STICKS after 30s (if it re-drifts, an active reverter is a BLOCKER to diagnose). Referee MUST time from a clean `git archive a62e26da` checkout, NOT the live tree.
- **EDIT#3 ACCEPT GATED ON:** (a) worker confirms working-tree==a62e26da stable, (b) referee verdict on that clean state, (c) rms/ln hedges recorded. Then → champion advance #2 (CE wide-V eviction 1.31/1.19/1.08x real + welford byte-identical + rms/ln hedged).
- **slot4 ruling FINAL:** uniform 'first' (worker ships oracle-winner on slot3 already; slot4 = passenger, no special-case).
- **EDIT-PID: decline BLOCKED (anti-giving-up FAIL), 3×3 A/B mandatory** (38%/19.5% unmeasured); plumbing committed 4bdd8cdf (pid_type gates sm_mult/maxnreg); seed NOT to be committed until the A/B + gates. sm_mult PHYSICS-derived not fit-to-32.
- 10 gate verdicts recorded. Context ~50%. Durable/respawn-safe.

## 2026-06-03 (alignment: full-oracle clarification + pid disposition A/B running)
- **fa11264a (consumer-trace row_reread refinement) committed** on top of EDIT#3 — worker landed it as a FORWARD commit (not the literal revert-split, which thrashed the tree). HUB VERIFIED 9/9 byte-identical to the banked region-membership (ran the probe at HEAD: sum/long/kl/jsd=F, rms/ln/softmax/CE/welford=T) → same seeds → EDIT#3 perf carries forward. Tree clean at 49319efe. fact-integrity FIRING on the consumer-trace predicate (analysis-only; no re-time since byte-identical). Process note to worker: don't execute THROUGH an explicit stop (DM+ack first) — the fa11264a commit crossed my "workstream CLOSED" but outcome was fine.
- **CLARIFICATION (hub owns a misread):** the worker's 3-shape pid "shape-inconsistency" was from FULL oracles (58ebff1b), NOT quick. The anti-giving-up gate + I conflated them with the earlier quick oracles. Full answer key: (4096,98304) interleaved@32 seed/oracle 1.624 oracle/tc 0.947; (8192,128256) interleaved@32 1.243 / 0.798; (2048,256000) **blocked@4** 1.070 / 0.623. So: 2/3 CONVERGE interleaved@32 at full effort (the quick 0.64/0.62 were under-explored); the widest GENUINELY wants blocked@4 (real full-effort divergence); ALL 3 source-bound (oracle<tc).
- **ALIGNED on the decider:** the do_bench disposition A/B (NOT the oracle — the oracle says what the SEARCH picked; the A/B says whether a SEEDED interleaved@32 beats FLAT everywhere it'd fire). Worker converged on exactly this. NOT a give-up (full answer key in hand). GPU-GRANTED.
  - **LOAD-BEARING = the cross-kernel probe:** interleaved@32-vs-flat on softmax-wide/welford-wide/rms-ln-wide (ZERO pid evidence there). (A) coarse seedable rule iff interleaved@32 net-positive vs flat on ALL firing shapes (3 wide CE incl 256000-vs-FLAT + the 4 cross-kernel). (B) Product-B/honest-null iff net-negative anywhere → record wide-CE as seed<oracle pid=Product-B + seed≈oracle<tc source-bound (cross_entropy_online = separate opportunity). If interleaved helps CE but hurts softmax/welford, a regime key = CE fence (smuggling) → (B).
  - EDIT#3 NOT re-run (banked, champion advance #2).
- 11 gate verdicts. Context ~50%. Holding for: pid A/B table (A/B verdict) + fact-integrity on fa11264a.

## 2026-06-03 (eviction family generalizes to T2 + EDIT-PID decline gating)
- **fa11264a (consumer-trace row_reread) BANKED** — fact-integrity PASS (4-point FX-graph: 9/9==banked, rms_norm True via disjunct-B bypass-store, kl_div/jsd False COHERENT [accumulator not masking — softmax control proves the predicate sees reduction-feeding loads], BFS-cut real). Behaviorally identical (seeds unchanged) → EDIT#3 perf inherited. row_reread workstream CLOSED. (Gate's full run was watchdog-stalled at the final check; a focused completion rendered the verdict — 2nd watchdog stall this run, both handled via focused completion, never as a silent pass.)
- **=== SOFTMAX-WIDE: full-oracle mandate FOUND A REAL WIN the quick oracle hid (worker self-reversed) ===** quick oracle said null (1.000/1.012 'at parity'); FULL oracle = 1.359/1.097, BOTH BEAT tc (1.39/1.10 — tc-beating, NOT source ceiling). Lever-decomp: EVICTION ALONE carries it (1.306/1.076; chunk/warps passengers). → **EDIT#6: extend EDIT#3 reread-eviction to the T2 plain path (softmax_two_pass).** SAME faithful reread_buffer_slots rule (already computes softmax slots=(0,1)); T2 return just didn't consume it. ~3 lines, no new fact, no identity, tc-beating. APPROVED → commit separate, gates=auditor+referee (no fact-integrity). The eviction family now spans ALL 3 tracks: T1(CE)+Band-C(welford)+T2(softmax), one faithful rule.
- **=== EDIT-PID DISPOSITION: worker proposes (B) DECLINE, now EVIDENCED — anti-giving-up FIRING to ratify ===** The mandatory 3×3 + cross-kernel probe (the answer key the earlier anti-giving-up FAIL demanded): interleaved@32 BEATS flat on all 3 wide CE (1.05-1.24×) BUT REGRESSES welford -3.2% under the ONLY non-identity gate (row_reread AND looped, which fires on welford too); softmax neutral, rms tie. Plus CE pid is oracle-only (oracle<tc 0.62-0.95). Worker's logic: seeding CE's gain needs a gate that also regresses peer welford → separating them = kernel identity (banned) → Product-B not Product-A seed. **CLEAN CONTRAST: EDIT#6 eviction PASSES the same cross-kernel test pid FAILS** (eviction helps softmax+CE+welford regresses nothing; pid helps CE regresses welford). HUB FIRED anti-giving-up to ratify: hunting whether a FINER non-identity fact (is_structured_combine / num_tiled_accumulators / band) separates CE-helps from welford-hurts (→ premature, test it) OR the only separator is kernel identity (→ decline CORRECT, fence worse than declining). Held until the gate rules.
- **3 champion advances banked** (row_reread substrate + fa11264a hardening; EDIT#3 eviction). EDIT#6 banking. pid declined-pending-gate.
- **NEXT MAJOR FOCUS (flagged to worker):** the broad at-floor→at-oracle confirmation. The 6 "at-floor" kernels (rms/ln/sum/long_sum/kl/jsd at G≈floor vs tc) are NOT confirmed at seed≈ORACLE — only at floor. Largest unexamined surface; where remaining buried per-shape gaps hide. Sweep after CE/softmax/pid close.
- 12 gate verdicts. Context ~50%. Respawn-safe.

## 2026-06-03 (=== EDIT-PID decline OVERTURNED — 38%/19.5% gain recovered by anti-giving-up ===)
- **anti-giving-up FAILED the (B) pid decline — PREMATURE (#8). The pid gain is REAL + SEEDABLE.** The worker's "no non-identity rule" premise was FALSE: it conflated the ABSTRACT gate `row_reread AND looped` ("also fires on welford -3.2%") with the heuristic's ACTUAL branches. get_seed_config has 3 DISJOINT branches: T1(is_t1: CE/rms/ln), Band-C(is_structured_combine: welford ONLY), T2-plain(softmax/kl/jsd). A T1-scoped pid override is STRUCTURALLY UNREACHABLE by welford(Band-C)+softmax(T2). The gate ran the experiment the worker SKIPPED — layer_norm(1,131072)=1.007 tie (worker only measured rms_norm) — so the COMPLETE T1-override set {CE +5-24%, rms tie, ln tie} regresses NOTHING. welford -3.2% real but different-branch-irrelevant.
- **The finer fact existed all along: `is_structured_combine`** (welford=sole True=sole regressor, CE=False) — fact-integrity-blessed, non-identity, already a branch key. The brief's "interleaved helps scalar-accumulator-multipass but hurts structured-combine-recurrence" maps exactly onto it.
- **BUILD EDIT-PID (T1-scoped):** {pid=persistent_interleaved, num_sm_multiplier=32, maxnreg=64} in T1 branch, gated fact.row_reread AND not persistent. sm_mult/maxnreg PHYSICS-derived NOT fit-to-32 (oracle said blocked@4 on widest → 32 not universally blessed). 256000 interleaved 1.052≈blocked 1.048 immaterial → coarse-but-positive (A). Recovers 38%/19.5% seed<oracle. Ledger: wide-CE = seed<oracle SEEDABLE-via-T1-EDIT-PID (NOT Product-B); separately seed≈oracle<tc source-bound (cross_entropy_online = distinct opportunity).
- **RUN THESIS VINDICATED (under-claim side):** a separated gate caught producer+HUB (I was leaning accept-B) about to retire a 38% gain on a false #8 no-rule. The gate found BOTH the missed branch-structure AND ran the skipped measurement. This is the single clearest demonstration this run of why acceptance must be gated by a separate context.
- **Process learning for worker (relayed):** when a decline hinges on "this gate also fires on peer X," verify X's ACTUAL branch + measure EVERY shape the gate fires on before concluding. Narrow gap, cheap check.
- **EDIT#6 (softmax T2 eviction): GO given** (commit pending). Sequencing: EDIT#6 first (characterized), then EDIT-PID T1-scoped. Both champion-advancing, gated.
- 13 gate verdicts. 3 champion advances banked + EDIT#6/EDIT-PID building. Context ~50%, respawn-safe.
- **NEXT after CE/softmax/pid close: the broad at-floor→at-oracle sweep** (6 kernels) → PARITY milestone.

## 2026-06-03 (multi-track forward: triage sweep + EDIT#6/EDIT-PID/EDIT#4 building)
- **Worker fully synced + operating cleanly** (processed the EDIT-PID overturn at a15d71c3; rms_norm no-rule overturn + uniform-ship recorded; standing-autonomy posture good). Leapfrog churn resolved.
- **EDIT-PID sm_mult formula NODded (physics-derived, the model p-hacking-guard answer):** `num_sm_multiplier = clamp(np2(ceil(grid_rows/num_sm)), 1, CAP=32)`, maxnreg=DEFAULT (the +0.02 passenger, not seeded), pid_type=persistent_interleaved. Derived from the MECHANISM (persistent_interleaved launches num_sm×sm_mult programs striding M); M=1 rms/ln proves const-32 can't be principled. CRITICAL gap flagged: the formula emits sm_mult∈{16,32,64} but the 3×3 only tested flat 32 → the A/B MUST test the DERIVED per-shape value. (8192,128256)→64 + (2048,256000)→16 UNTESTED; 256000@derived-16 is AT-RISK (oracle wanted 4 for L2-thrash) → if it regresses, byte-keyed CAP becomes load-bearing.
- **🟢 GPU-GRANTED: floor-oracle triage batch** (jsd narrow-V ~1.20 + softmax small-N ~1.15 + 3 at-floor spot-checks sum/kl_div/softmax). The at-floor→at-oracle sweep = path to PARITY. Worker's cheap-first triage logic CORRECT + internalized: quick-GAP=real (full widens) → EDIT candidate; quick-PARITY=suspect → full-confirm before "at oracle" counts (the quick-undershoots lesson from softmax-wide).
- **GPU PRIORITY ORDER set:** (1) EDIT#6 softmax T2 eviction (commit+referee, simplest tc-beating win); (2) EDIT-PID derived-sm_mult A/B (build T1-scoped + the 4-shape derived-config test, the 38% recovered gain); (3) EDIT#4 welford apply-cap (designed 5aa8d140); (4) full-confirm whatever triage flags.
- **STATE: 3 champion advances banked** (row_reread substrate + consumer-trace hardening; EDIT#3 eviction). Building: EDIT#6, EDIT-PID(T1), EDIT#4. Triage running. 13 gate verdicts. Context ~50%, respawn-safe.
- **The run's spine, restated:** eviction family (EDIT#3 T1+BandC, EDIT#6 T2) = one faithful reread_buffer_slots rule across all tracks; EDIT-PID = T1 wide-reread pid regime (physics-derived). Then at-floor→at-oracle sweep closes the remaining surface → PARITY → freeze+TEST (Phase 3) → beat-oracle overtime (Phase 4).

## 2026-06-03 (EDIT-PID built + 3 gates firing — hardest workstream closing)
- **EDIT-PID committed 94851da9, 3 gates FIRING** (fact-integrity track-key+sm_mult-physics / auditor no-CE-fence / referee CE-wins+byte-identical, tree-drift-cautioned). T1 seed gated `fact.row_reread and not persistent`: pid=persistent_interleaved, num_sm_multiplier=clamp(np2(ceil(grid_rows/num_sm)),1,32), maxnreg=64. Emission 10/10 verified (CE→interleaved sm 32/32/16 mnr64; rms M=1→sm1; welford/softmax/sum/long/kl/jsd→flat byte-identical = cross-branch scope).
- **Worker self-corrections (both right, both via the derived-config A/B I mandated):** (1) OWNED the false grid-test (#8 trap the anti-giving-up gate caught); (2) caught maxnreg=64 is LOAD-BEARING not the passenger it leaned — isolation A/B: with-maxnreg 1.23/1.25/1.05 vs without 1.05/1.15/1.00 (maxnreg doubles the gain + makes 256000 net-positive). Had it shipped its maxnreg=default lean, 256000 ties. The derived-per-shape A/B (vs flat-32) earned its keep. Also flagged 2 build-bugs: circular import + env-current→silent-seed-DROP (run-2 trap, caught).
- **CE wide-V per-shape accounting:** seed/oracle 1.62 raw → 1.31 (EDIT#3 eviction) → ~1.05-1.08 (EDIT-PID, pending referee) = at/near seed VICTORY bar vs oracle. SEPARATELY seed≈oracle<tc (source-bound 2-pass CE) → cross_entropy_online = distinct Product-A-via-source opportunity (NOT this seed deliverable). Record "seed→oracle CLOSED (pending gates), source-bound vs tc."
- **Worker directed (non-GPU while gates run):** commit EDIT#6 (softmax T2 eviction, GO given) + DM sha; finalize EDIT#4 (welford apply-cap, designed 5aa8d140); DM triage table (jsd ~1.20 / softmax-smallN ~1.15 / 3 at-floor spot-checks). GPU queue after referee: EDIT#6-reproduce → EDIT#4 A/B → full-confirm triage-flagged gaps.
- **eviction family now T1+BandC+T2 (EDIT#3+#6, one faithful rule); EDIT-PID adds the T1-wide-reread pid regime.** Remaining to PARITY: EDIT#4 welford, jsd Band-B, softmax small-N, long_sum-tail, + the at-floor→at-oracle full-confirms. 13 verdicts + 3 EDIT-PID firing. Context ~50%, respawn-safe.

## 2026-06-03 (git-hygiene thread CLOSED — hub stale-HEAD misread, codified)
- **RESOLVED: the "staged half-revert" I alarmed on was MY stale-HEAD misread, NOT a dirty tree.** Verified: helion/ porcelain CLEAN, HEAD=e73ba2ea, full stack committed (persistent_interleaved=3 EDIT-PID / reread_buffer_slots=4 EDIT#3 / consumer-trace _compute_row_reread(red_block_id)). a62e26da intact for the referee (persistent_interleaved=0=pid-flat, reread_buffer_slots=4=eviction). The 45+/113− "divergence" was the COMMITTED EDIT-PID delta (I diffed vs a62e26da while HEAD had EDIT-PID stacked on top).
- **CODIFIED LESSON (for respawn-safety — I misread tree state TWICE this way):** authoritative tree check = `git status --porcelain helion/` (clean/dirty) + `git rev-parse HEAD` (actual tip). Do NOT infer "uncommitted divergence" from `git diff <remembered-sha>` — a committed stack on top reads as a diff. TRUST the worker's committed-sha reports + verify with porcelain. My earlier hard-reset-to-a62e26da was based on a transient mid-build intermediate; the worker's tree was fine.
- **Referee/gate sha-pinning discipline (worker's lesson, adopted):** gates MUST pin an immutable sha (git-archive / own clean checkout + rm __pycache__), NEVER time the worker's shared live tree — so a mid-gate worker commit can't confound. fact-integrity does this; every referee brief now instructs it. EDIT#3 gated on a62e26da (pure eviction, pid=flat); EDIT-PID referee isolates pid-vs-flat-both-evicted (baseline=a62e26da seed) = pure pid delta, not confounded eviction×pid. Do NOT `git checkout a62e26da` into the worker's tree (un-ships committed EDIT-PID).
- **rms/ln hedge committed (e73ba2ea), ACCEPTED:** rms_norm = robustness-canary-not-slower (NOT a win/overturn; placement-non-discriminating, sub-noise, not-persistent gate excludes run-2's larger-M shapes); layer_norm within-noise-leans-neg, uniform-to-avoid-carve-out. CE eviction wins stand.
- STATE: EDIT-PID 3 gates firing (immutable-sha-pinned). Worker non-GPU: commit EDIT#6, finalize EDIT#4, DM triage. Holding for EDIT-PID verdicts. Context ~50%, respawn-safe.

## 2026-06-03 (DRIFT ROOT-CAUSE found — gate-checkout-on-shared-tree; isolation codified)
- **ROOT CAUSE of the multi-hour working-tree drift saga: a GATE (referee) was `git checkout`-ing a BASELINE sha into the SHARED working tree** (its "time the parent commit" step) — NOT the worker's edits, NOT env, NOT a half-revert. That transient pre-EDIT#3 checkout is exactly the "0 reread_buffer_slots / 45+113− / CE→None" states the hub + auditor saw mid-session. Explains ALL of: my 2 "half-revert" alarms, the worker's "edits reverting between operations," the auditor's stale-state transient.
- **RESOLVED + CODIFIED RULE (enforce on ALL future gates):** a gate that needs to time/emit a baseline or parent commit MUST do so in an ISOLATED checkout (`git archive <sha>` to /tmp, or a separate `git worktree`), NEVER `git checkout` in the shared worktree (= the worker's live workspace). The referee is NOW correctly isolated in `/tmp/parent_tree` (verified: dir exists, shared tree clean+stable across 4 reads, referee not touching helion/). fact-integrity already git-archives. EVERY referee brief must mandate this.
- **Worker tree CONFIRMED settled:** helion/ porcelain CLEAN + stable (4 consistent reads, reread_buffer_slots=4), HEAD=e73ba2ea emits correct champion config (EDIT#3 eviction + EDIT-PID pid). a62e26da intact (4/4/1) for the referee's isolated pure-EDIT#3 timing.
- This consumed a large fraction of the session (async message-crossing + the shared-tree gate-checkout drift). SUBSTANTIVE state unaffected — all gated verdicts rest on immutable shas (git-archive), so the drift never corrupted an acceptance. Lesson banked: gates pin immutable shas; hub checks porcelain+HEAD not diff-vs-remembered-sha; don't build/commit on the shared tree while a gate runs.
- HOLDING for: EDIT-PID 3 gates (94851da9, isolated) + EDIT#6 sha + EDIT#4 + triage table. 13 verdicts banked. Context ~50%, respawn-safe.

## 2026-06-03 (=== EDIT-PID BANKED — champion advance #4, wide-CE workstream CLOSED ===)
- **EDIT-PID (94851da9) BANKED — all 3 gates PASS** (fact-integrity + results-referee 5/5 + adversarial-auditor), all on the immutable commit:
  - fact-integrity: track-key STRUCTURAL non-identity (T1/T2 mutually exclusive → welford/softmax unreachable by the T1 pid block, no leak); sm_mult PHYSICS-derived not oracle-fit (ships derived-16 at 256000 where oracle picked 1-or-4 → matches NONE of the oracle's 3 picks).
  - referee 5/5: CE pid wins 1.231/1.241/1.051 reproduced (clears noise ≥16×), derived-sm_mult shipped (32/32/16/1 not flat-32), maxnreg load-bearing confirmed (without it 256000 ties flat), byte-identical elsewhere (12-shape Triton-sha diff), correct ≤9.5e-7. Honored isolation (blocked on worker's concurrent proc, archived parent).
  - auditor: NO fence (row_reread excludes long_sum's real 8MiB rows in-curriculum; rms/ln firing = anti-fence evidence); reversal addresses the DISJOINT grid-starved-wide-looped regime run-2's flat-lock A/B never tested; 256000 honestly net-positive (beats both flat AND oracle's blocked@4); oracle-only disclosed (not "done"); env-bug fixed (n_seeds=1); no metric gaming.
- **THE RUN'S SIGNATURE RECOVERY:** a 38%/19.5% seed<oracle gain TWICE nearly declined as Product-B (worker leaned (B)), recovered by anti-giving-up FAILing the decline (#8: worker missed its own branch structure — welford=Band-C unreachable by T1 — AND skipped the layer_norm measurement the gate then ran=tie). Built T1-scoped + physics-derived sm_mult, triple-gated clean. The under-claim guard working decisively + the worker owning the error + building clean.
- **WIDE-CE WORKSTREAM CLOSED:** EDIT#3 eviction + EDIT-PID pid → CE wide-V seed/oracle 1.62→~1.05 = at/near victory bar. Residual seed≈oracle<tc (source-bound 2-pass logsumexp) = cross_entropy_online SEPARATE Product-A-via-source opportunity (disclosed, not this seed deliverable).
- **4 CHAMPION ADVANCES BANKED:** (1) row_reread substrate + (1b) consumer-trace hardening; (2) EDIT#3 eviction T1+BandC; (#4) EDIT-PID T1 pid. [EDIT#6 softmax T2 eviction = advance #3-in-progress, GO given.] 16 gate verdicts.
- **GPU released to worker. Forward queue → PARITY:** EDIT#6 (commit+referee) → EDIT#4 welford apply-cap → triage full-confirms (jsd ~1.20 + softmax-smallN ~1.15 + 3 at-floor spot-checks at seed≈oracle). Then === PARITY REACHED === → Phase 3 (freeze + Product B + TEST once).
- Context ~50%, respawn-safe. Infra disciplines hardened (gate sha-isolation, GPU token, porcelain+HEAD tree checks).

## 2026-06-03 (EDIT-PID cross-kernel probe = corroboration; welford Band-C candidate logged)
- **Worker's cross-kernel probe (553e8c21) CORROBORATES the banked EDIT-PID** (not a re-gate — gates already PASSed). CE reproduced 3rd time (1.233/1.244/1.053). Cross-kernel interleaved@32-vs-flat: softmax(T2) 0.998 neutral; welford(Band-C) 65536,4096 = 0.967 HURTS / 32768,8192 = 1.064 HELPS; rms/ln(T1) 1.006/1.007 tie. → (A) holds for T1-scope (net-positive on its firing set CE+rms/ln); (B) NOT triggered (interleaved net-neg only OUTSIDE the firing set = welford Band-C, structurally excluded). The probe VALIDATES the track-scope: a track-agnostic gate would self-inconsistently regress welford (and welford's pid response is shape-dependent WITHIN Band-C). Auditor already used this structural exclusion as its no-fence defense.
- **NEW BOARD CANDIDATE (logged, not chased): welford(32768,8192) +6.4% under interleaved pid.** Shape-dependent within Band-C (65536,4096 = −3.3%). OUT of EDIT-PID scope (T1 only). Fold into the welford FULL-ORACLE triage confirm: chase ONLY if welford(32768,8192) shows seed<ORACLE there ("beats flat" ≠ "below oracle"). If real, needs a Band-C pid fact with a finer key separating the +6.4% from the −3.3% shape (hard — shape-dependent within one kernel). Triage candidate, low priority vs the broad at-floor sweep.
- **LEAPFROG persisted on EDIT-PID** (worker asked to "gate" it ~4×, all crossing my "it's banked" replies — pure delivery latency, substance agreed throughout). Resolved by a maximally-explicit "BANKED, stop asking, build EDIT#6" + the worker's own don't-execute-through-a-stop protocol (which it correctly applied: DM-conflict+ack rather than re-run).
- STATE: 4 advances banked. Forward queue (GPU worker's): EDIT#6 (softmax T2 eviction, building) → EDIT#4 welford apply-cap → triage full-confirms (jsd ~1.20, softmax-smallN ~1.15, at-floor spot-checks incl welford 32768,8192 vs oracle) → PARITY. 16 verdicts. Context ~50%, respawn-safe.

## 2026-06-03 (EDIT#6 committed + gating — eviction family now spans ALL 3 tracks)
- **EDIT#6 (8577b675) COMMITTED + 2 gates FIRING** (referee timing softmax 1.36×/1.10× + byte-identical kl/jsd/narrow; auditor no-softmax-fence; NO fact-integrity — no new fact, 3rd consumer of reread_buffer_slots). triton.py-only. The reread-eviction rule now routes in all 3 branches: T1 (line 638), Band-C (750), T2-plain (807) — ONE faithful reread_buffer_slots fact, IDENTICAL gate (row_reread and not persistent), spanning cross_entropy/welford/softmax. This is the "general heuristic not kernel-fitted" payoff: a provenance-derived fact generalizing across every track.
- **EDIT-PID gate-state CLARIFIED (resolved worker's (A)/(B) confusion, ~5th cross):** NO pending anti-giving-up gate. Timeline: anti-giving-up FAILED the (B)-decline → worker built (A) T1-scoped → 3 FRESH gates (FI+referee+auditor) PASSED on 94851da9 → BANKED (advance #4). Worker's INSIGHT AFFIRMED + correct: the finer fact anti-giving-up hunted (is_structured_combine separating scalar-accum-multipass-CE from structured-combine-recurrence-welford) IS what EDIT-PID's T1-scope encodes — track/band = the principled non-identity separator. Worker + gate converged on the same separator. Cross-kernel probe (553e8c21) = corroboration.
- **eviction family COMPLETE:** EDIT#3 (T1 CE + Band-C welford) + EDIT#6 (T2 softmax) = the reread-eviction rule on every track that has a re-read row. tc-beating where not source-bound (softmax 1.36/1.10 BEATS tc; CE wide source-bound but seed→oracle closed).
- STATE: 4 advances banked (+ EDIT#6 banking = 5th). Per-shape VICTORY: 3 CE boundary, softmax-wide, CE wide-V (evict+pid), + EDIT#6 softmax-wide eviction. Forward to PARITY: EDIT#4 welford apply-cap → triage full-confirms (jsd ~1.20, softmax-smallN ~1.15, 3 at-floor spot-checks, welford 32768,8192-vs-oracle). Then === PARITY REACHED === → Phase 3. 16 verdicts + 2 EDIT#6 firing. Context ~50%, respawn-safe.

## 2026-06-03 (=== EDIT#6 BANKED #5 — EVICTION FAMILY COMPLETE across all 3 tracks ===)
- **EDIT#6 (8577b675) BANKED — champion advance #5.** referee PASS 4/4 (softmax 1.31×/1.08× eviction-sole-carrier reproduced, byte-identical kl/jsd/narrow, fp32 correct, measured exactly 8577b675) + auditor PASS (NO softmax-fence: same _eviction_policies('reread') call + same gate as T1/Band-C, fires only row_reread-AND-looped softmax, cross-track uniform, tc-beating 1.336/1.077 — distinguished from source-bound CE). Magnitude honest (1.36/1.10=oracle ceiling, 1.31/1.08=shipped lever, both disclosed).
- **=== EVICTION FAMILY COMPLETE ===** the faithful reread-eviction rule (reread_buffer_slots, gated row_reread AND not persistent) now spans ALL 3 kernel tracks: T1 (cross_entropy) + Band-C (welford) + T2-plain (softmax). ONE provenance-derived fact, ONE gate, generalizing across every track with per-kernel buffer-identity 'last' placement (CE slot2=logits, softmax slot0=x, welford slot0=x) — NOT positional. This is the run's central "general heuristic not kernel-fitted" deliverable. Surfaced/completed by the hub's full-oracle-arbitrates mandate (the softmax-wide quick-null was reversed 1.000→1.359 at full effort).
- **5 CHAMPION ADVANCES BANKED:** (1) row_reread substrate + (1b) consumer-trace hardening; (2) EDIT#3 eviction T1+BandC; (4) EDIT-PID T1 pid (38% recovery); (5) EDIT#6 softmax T2 eviction. 18 gate verdicts, all PASS or correctly-overturned-decline, all on immutable shas.
- **PER-SHAPE VICTORY (seed≈oracle) so far:** 3 CE boundary; softmax-wide (cap + eviction); CE wide-V (eviction+pid ~1.05); + the eviction wins on welford/rms (hedged). 
- **HOME STRETCH to PARITY (GPU granted to worker):** EDIT#4 (welford apply-cap, designed) → at-floor→at-oracle sweep (task #13, 15-shape batch: full-confirm rms/ln/sum/long_sum/kl/jsd at seed≈oracle; candidates: jsd narrow-V ~1.20, softmax small-N ~1.15, welford(32768,8192)-vs-oracle, long_sum(16,2097152) source-limit→anti-giving-up). PARITY = full-curriculum per-shape seed/oracle table all at-oracle-or-verified-source-ceiling → === PARITY REACHED === → Phase 3 (freeze+Product B+TEST once) → Phase 4 (beat-oracle overtime).
- Leapfrog fully reconciled (worker silent+aligned; both edits banked, no pending gates). Context ~50%, respawn-safe.

## 2026-06-03 (PARITY home stretch — triage candidates scoped, GPU freed for forward work)
- **At-floor→at-oracle TRIAGE done (e9fc8283/1a311401):** surfaced 2 real seed<oracle candidates + parity confirms. jsd narrow-V 1.21/1.13 (seedable, oracle/tc 1.01-1.03); softmax small-N 1.205 (real but run-2 warps-overfit-trap); kl_div(8192,30522)=PARITY ✓; sum(16384,2048)=1.032 ~parity suspect. Worker correctly applied quick-GAP=real / quick-PARITY=suspect + caught the softmax-warps trap.
- **EDIT#5 (jsd narrow-V) — worker pre-scoped with a SHARP finer-key catch (EDIT-PID lesson applied proactively):** kl_div(8192,30522) at PARITY while jsd wants smaller chunk at the SAME shape → "Band-B narrow-V" is NOT the separating key (peer-at-parity-same-shape = wrong key). Hypothesis: num_tiled_accumulators (jsd=2: intermediate_loss+intermediate_dX; kl_div=1: loss_sum). HUB confirmed num_tiled_accumulators is an EXISTING fact (config_spec:171, Band-B cap gate triton.py:764) → if it separates jsd-wants-smaller from kl_div-at-parity cleanly, EDIT#5 is principled (not a jsd-fence, no new fact); else finer-property or jsd-source-specific Product-B. Worker NOT building until measured (lever-decomp + the num_tiled_accumulators separation test). Tasks #14 (EDIT#5) + #15 (softmax-smallN careful) created, block #5 PARITY.
- **GPU FREED for forward work:** worker was idling thinking a referee owned the GPU — but EDIT-PID + EDIT#6 referees both FINISHED (banked). Confirmed 0 compute procs. Cleared worker to start EDIT#4 (welford apply-cap, run3_wf_tile_ab.py ready).
- **CE wide-V accounting (worker recorded, crisp):** seed/oracle 1.62→1.31(EDIT#3 evict)→~1.05-1.08(EDIT-PID) = at/near victory bar vs oracle; SEPARATELY seed≈oracle<tc source-bound (2-pass CE) → cross_entropy_online = distinct Product-A-via-source opportunity, NOT this deliverable.
- **PATH TO PARITY (concrete, finite gap-list):** EDIT#4 welford + EDIT#5 jsd (pending num_tiled_accumulators test) + softmax-smallN disposition (careful, occupancy fact or proven-runtime-only) + sum/kl full-confirms + long_sum(16,2097152) source-limit (anti-giving-up). When the full-curriculum per-shape table is all seed≈oracle OR verified-source-ceiling → === PARITY REACHED === → Phase 3 (freeze+Product B+TEST once) → Phase 4 (beat-oracle overtime).
- 18 gate verdicts, 5 champion advances banked. Worker disciplines fully internalized (caught softmax-warps trap + jsd-vs-kl_div finer-key proactively). Context ~50%, respawn-safe.

## 2026-06-03 (EDIT#4 gating + at-floor sweep done → BOUNDED PARITY worklist)
- **EDIT#4 (c3d90e8d, welford apply-cap 8192→16384) committed + 2 gates FIRING** (referee reproduce 1.068/1.047 PARTIAL + byte-identity; auditor honest-partial-not-fence; no fact-integrity). Gated AS a PARTIAL — worker honestly flags it narrows welford (214→200us) but does NOT close to oracle (still 0.93 arm/tc); full close = EDIT#4b. 
- **AT-FLOOR SWEEP COMPLETE (25bd7d6b) = the PARITY gap-list. GOOD NEWS: most of the 6 at-floor kernels ARE at oracle parity** (rms mid/wide/high-M, ln all, sum, long_sum-persist, kl_div) — the geomean did NOT hide gaps everywhere. (quick — extreme bands need full-confirm before counting.)
- **BOUNDED PARITY WORKLIST (the only remaining gaps):**
  1. **EDIT#5 jsd narrow-V** (1.214/1.127 full-confirmed, beats tc) — multi-lever (chunk 4096→2048 + warps 32→16 + stages 1→4/8); decomp which carries + nta-finer-key test (does it correctly NOT fire on kl_div-at-parity?). Worker running the decomp NOW. Task #14.
  2. **EDIT#4b welford full-close** (1.163/1.116) — coupled combine 8192→16384 + M_block 1→2 (+ EDIT#4 apply); M_block-why-2 (occupancy) + combine-cap-N-bounded (>16KiB spills N≥32768) principled-Qs. Task #16.
  3. **Narrow-N cluster HARD** (task #15): rms(8192,768)=1.136 wants warps 16→8, softmax(131072,256)=1.205 wants 4→8 — OPPOSITE dirs → occupancy/rows-per-SM-keyed NOT rnumel (run-2 trap). Hunt the occupancy fact; anti-giving-up on any no-rule claim; may be proven-runtime-only Product-B.
  4. **long_sum(16,2097152)** source-limit → anti-giving-up flag.
  5. Quick-VICTORY extreme-band full-confirms.
- **5 advances banked + EDIT#4 gating (#6).** Worker fully internalized gate-state (no more status-loop), operating forward-only with full decomposition discipline (the EDIT#5 decomp + kl_div finer-key probe is self-directed, exactly right). 18 gate verdicts. Context ~50%, respawn-safe.
- PATH: close the 5 worklist items (or verify source-ceiling/proven-runtime-only) → === PARITY REACHED === → Phase 3 (freeze+Product B+TEST once) → Phase 4 (beat-oracle overtime).

## 2026-06-03 (EDIT#4 REJECTED — referee caught a 7× regression the producer+auditor missed; reverted)
- **=== EDIT#4 (c3d90e8d) RESULTS-REFEREE REJECT — the over-claim guard's headline catch ===** The apply-cap 8192→16384 constant raises apply 2048→4096 on EVERY structured-combine shape above the persist threshold — incl large-M ones the worker's 3-shape A/B never tested. CATASTROPHIC: welford(262144,5120) [in-sample-v2 curriculum] 4612→33571us = ~7.3× SLOWER (end-to-end 8.2→36ms = 4.4×; arm/tc 0.104). Also welford(8192,4096)/(16384,4096) ~-2.3%. Reproduced 5×/3 codepaths/2 seeds, spread<0.3%. BOTH the worker's A/B (3 hand-picked shapes, clamps to min(np2(N),…)) AND the auditor's byte-id sample MISSED it — only the referee's OWN broad re-run (testing shapes BETWEEN narrow+wide) caught it. This is exactly why the referee re-runs independently, not the producer's A/B.
- **REVERTED (d09e08b7, cap→8192; HEAD 6cff9b01).** welford(262144,5120) back to apply=2048 (4.6ms). 7× regression OUT of the champion. EDIT#4 is NOT banked (only 4 advances, not 5). LESSON banked (worker memory): a config-flipping cap's no-regression A/B MUST sweep the axis it flips on at MID + EXTREME M, not endpoints/hand-picked shapes.
- **REALITY-CHECK on PARITY timeline:** welford is a HARDER gap than the constant suggested — apply/combine/M_block sizing is M-dependent + coupled; a bare per-N cap is too coarse (regresses large-M). EDIT#4+#4b likely MERGE into one principled (M,N)-grid-keyed welford-tile rule, full-curriculum A/B (incl 262144,5120). Re-opened.
- **EDIT#5 jsd — refined to a CLEAN candidate (pending World-A/B):** chunk-ALONE (footprint cap // max(1,nro)) lands jsd seed/oracle ~1.025 TIE → warps lever DROPPED (don't pollute the rnumel-only _num_warps ramp for ~2% on a closed gap). nta CANNOT discriminate (jsd=kl_div=2, docstring "kl=1" was stale); nro (jsd=2/kl=1) is the differentiator — and the rejected-proxy worry is a CATEGORY ERROR (nro rejected as a ROW_REREAD proxy; here it's the DIRECT faithful reduction-accumulator count; rule = 16KiB budget // (itemsize×nro) = per-accumulator share = resident-footprint physics). GPU-GRANTED for the World-A/B gate: wide-V jsd oracle (V-independent? clean BUILD : re-fork) + footprint-cap A/B + MANDATORY full-M-range flip-set sweep (EDIT#4 lesson).
- **STATE: 4 advances banked** (row_reread+consumer-trace; EDIT#3 evict T1+BandC; EDIT#6 T2; EDIT-PID). 20 gate verdicts (incl the EDIT#4 REJECT — the over-claim guard working). Worklist to PARITY: welford-tile redesign (M-keyed, harder), EDIT#5 jsd (World-A/B pending), narrow-N cluster (occupancy, hard), long_sum-tail (anti-giving-up), quick-VICTORY confirms. Context ~50%, respawn-safe.

## 2026-06-04 — REATTACHED #0 (fresh hub, genesis baton) at phase Phase-2 (oracle ascent)
Previous hub context manually /cleared (confirmed by human; in-process team died with it). GPU idle, no orphan procs.
Tree reconciled: helion/ clean, HEAD 264ba037, APPLY_CHUNK_BYTES==8192 (EDIT#4 reverted). Ledger: 5 advances + 20 verdicts.
EDIT#5 (jsd Band-B footprint cap) analysis-DONE + full-oracle confirmed, NOT yet committed — gating it is move #1.
SCOPE: welford excluded (another agent owns its Band-C branch); parity = the other 8 kernels.
Stood up team `reduction-heuristics`: worker + code-investigator + perf-investigator, all model:opus. Baton: _lab/HUB_BATON.md.

### 2026-06-04 ~01:50 — EDIT#5 (jsd Band-B ÷nro cap) GATED: auditor+referee PASS, fact-integrity FAIL → NOT banked, rework
Canonical SHA 46e58c73 (triton.py-only; orphaned d2ff878a ≡ it, blob d3bb8476). Gates on d2ff878a (== live, diff empty):
- adversarial-auditor af9b3c12 PASS (ledger 21): gain real (oracle picks 2048 narrow+mid+wide), nro IR-derived not name-fence (jsd sole Band-B∧nro≥2; kl_div nro=1 control), structurally INVERSE of EDIT#4 (LOWERS R_BLOCK → footprint shrinks monotonically).
- results-referee a144dd29 PASS (ledger 22): independently reproduced 34 flips, ZERO regressions full flip-set (worst (2048,256000)@1.021 still 2048-faster), EDIT#4 valley-analog jsd(262144,4096)=1.047x WIN. Caught the welford-owner (~62GB transient) mid-run, re-ran idle-gated, stable. GPU released.
- fact-integrity: original a24ad820 STALLED on 600s watchdog after analysis, before verdict (recorded nothing) → salvaged via completion gate a8f7737f → **FAIL** (ledger 23, case B lucky proxy / failure mode #11).
DECISION: NOT banked (champion_advances stays 5). fact-integrity substrate veto > 2 measurement PASSes ("curriculum-correct ≠ faithful"). num_reduction_ops counts ReductionLowering nodes = resident-accum count ONLY under jsd/kl_div 1:1 coincidence; gate built DIV-A (nro=2,resident=1→under-size) + DIV-B (nro=1,resident=2→over-size=the spill the cap prevents). SEED VALUES right on curriculum (jsd 2 carried accums, kl_div 1 — kl_loss is in-loop scratch); FACT hacky.
FIX (gate-provided): re-key divisor onto NEW faithful fact = count of [M,R] 2D accums CARRIED across the inner loop (excl in-loop scratch); = jsd:2/kl_div:1 (curriculum, byte-id to ÷nro → auditor+referee transfer on byte-id proof) AND DIV-A:1/DIV-B:2 (divergence). num_tiled_accumulators-as-shipped also wrong (=2 both, over-counts kl scratch). Worker reworking; interim ÷nro stays on live tree (no curriculum regression), superseded not reverted.
NOTE: confirmed the welford-owner is a real separate GPU process — strict-serial + nvidia-smi-before-every-bench validated.

### 2026-06-04 ~02:42 — EDIT#5-v2 BANKED as champion advance #6 (SHA 6bcfeed1)
EDIT#5 jsd Band-B footprint cap is IN. Journey: v1 ÷num_reduction_ops (46e58c73) → fact-integrity FAIL #23 (lucky-on-curriculum proxy, mode #11) → reworked onto faithful num_carried_accumulators (inner-ForLoop carry-set [M,R] tiles) v2 6bcfeed1. 3/3 gates: fact-integrity PASS #24 (native; DIV-A=1/DIV-B=2 by construction, jsd=2/kl_div=1 confirmed 70 shapes), adversarial-auditor PASS #20 + results-referee PASS #21 (TRANSFERRED via byte-identity num_carried==nro 70/70). Referee full flip-set: 34 jsd flips, ZERO regressions, EDIT#4 valley-analog jsd(262144,4096)=1.047x WIN.
GATE-INFRA: fact-integrity died 3x before landing (v1 watchdog→completion gate a8f7737f rendered the FAIL; v2 API-error + v2 watchdog both wasted time AUTHORING DIV-A/DIV-B in DSL) → rescoped 3rd v2 instance (run the worker probe, no kernel-authoring, record-first) landed in 145s. LESSON: brief analysis gates to REUSE committed probe artifacts, never re-author kernels.
SIDE-FIX: oracle-cache source_hash now deterministic+seed-independent (12f341ef) — was process-non-deterministic (BlockIdSequence addr-reprs) → guard was vacuous (no past corruption). Cache re-stamped (29 in-scope, 3 welford skipped), uncommitted by design (welford latency drift not ours).
NEXT: GPU-GRANT worker for task#2 (live seed/oracle status table, 8 in-scope kernels, welford-idle-gated) → resume climb on shapes furthest from oracle.

### 2026-06-04 16:10 — STALL DETECTED + RECOVERY (human prompt caught it)
**The run was STALLED ~13h (02:47→16:09).** Root cause: worker spawned a DETACHED background oracle batch (narrow-N: softmax(131072,256)+rms(8192,768)+ln(8192,768), bromwg7tb / timeout 2400) at ~02:47, ended its turn waiting on it. The detached GPU job DIED at the first autotune (60s compile-timeout on softmax first config; narrowN_oracle_batch.json has ONLY the input shapes, no oracle latencies — known failure mode: detached bg GPU runs get killed here). Worker was never re-invoked (its wait-condition vanished w/o notification) → worker dormant 13h; hub dormant waiting on worker. The ScheduleWakeup backstop was refused at session start → no heartbeat caught it. Human prompt ("are you sure work is going on?") surfaced it.
NO corruption: champion intact (6 advances), GPU idle/clean, worker still a team member (resumable). RECOVERY: wake worker, re-run narrow-N oracles in FOREGROUND under the GPU token (never detached). LESSON: forbid detached/backgrounded GPU jobs; long oracles run foreground-under-token (worker blocks, hub holds token), or hub polls a file the FOREGROUND job writes. Hub must not infer progress from idle notifications — verify the beacon mtime each tick.

### 2026-06-04 16:56 — RUN STOPPED BY HUMAN (clean)
Human requested full stop. Shut down team (worker + code-investigator + perf-investigator). Killed/confirmed-down the worker's in-flight foreground GPU A/B (run3_smalln_occ_contour_ab.py, PID 403189). GPU fully released (0 MiB, 0 procs). helion/ clean, HEAD 12f341ef. Champion intact: 6 advances, 24 gate_verdicts. NO uncommitted helion/ work lost (tree clean; lab artifacts on disk).
RESUME POINT: worker reached a GATE-READY candidate EDIT#7 right at stop (beacon tick 28): high-OCC small-N softmax -> M_BLOCK x2 + num_warps 16, occupancy-keyed (grid_rows//num_sm, NO new fact). Generality A/B CLEAN: all 3 hi-OCC small-N shapes want it (1.03-1.27x); lo-OCC controls confirm the gate (the bump HURTS at low OCC, so the occupancy gate is real, not overfit). NOT yet built/committed/gated. To resume: build EDIT#7 + 4-split flip-set + full-confirm the softmax(131072,256) oracle (1.170/bundle; carrier=coupled {M_BLOCK 8->16, warps 4->16}=1.107~oracle 1.110, ns/evict passengers) -> fire gate pipeline.
