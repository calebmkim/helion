# Heuristic-PR Perf Audit — `perf-repro/`

A self-contained harness for **independently re-auditing the performance claims in a Helion
autotuner-heuristic PR**. Built first for PR #2996 (the generalized reduction seed heuristic), but
designed to be the reusable template for auditing *any* seed-heuristic PR. If you're re-running this
for a new heuristic, read **§9 "Redoing this for a new PR"** — the rest is context.

---

## 1. Goal

A PR claims a config-seeding heuristic gives "reasonable perf on nearly any kernel of class X". We
independently verify that claim to a **skeptical-manager** standard:

- **Reproduce** the PR's posted numbers on the exact kernels/shapes it reported.
- **Stress generalization** — run the heuristic on kernels *and* shapes it was never tuned on, to
  answer "is this overfit to the curriculum?".
- **Vet the benchmarking itself** — make every measurement choice defensible (cold-L2, interleaving,
  device vs eager time, accuracy gating, fair baselines) so a hostile reviewer can't dismiss it.
- Produce numbers that are **honest in both directions** — surface where the heuristic loses, not
  just where it wins, and explain *why* from the emitted code.

The output is **not** "the heuristic is good/bad." It's a defensible per-(kernel, shape, dtype)
dataset + an explanation of every win and loss.

---

## 2. What we compare (the arms)

For each `(kernel, shape, dtype)` cell, up to 4 configs are timed on the **same inputs, in one
process**, by explicit config replay (never by flipping env flags mid-process):

| arm | what it is | why |
|---|---|---|
| **seed** | `compiler_seed_configs(env, device_ir)[0]` — the heuristic under test | the thing being audited |
| **default** | `config_spec._base_default_config()` — the UNSEEDED compiler default | "what the heuristic buys over doing nothing". NB: **not** `default_config()`, which can return the seed if the heuristic sets `promote_seed_to_default`. |
| **tc** | `torch.compile(reference_fn)`, default mode (no max-autotune) | what a normal PyTorch user gets for free |
| **vllm_shipped** | the Helion kernel run with vLLM's shipped pre-tuned JSON config (nearest-shape / exact-key lookup) | for vLLM kernels: the *key* comparison — is the heuristic as good as vLLM's hand-tuning? NB: this runs the **Helion** kernel with vLLM's *config*; it is NOT vLLM's native CUDA kernel. |

Ratios (>1 ⇒ seed faster): `G_def = default/seed`, `G_tc = tc/seed`, `G_vllm = vllm/seed`. Reported
as **geomean over shapes**, re-derived from raw µs by the aggregator (never hand-entered).

---

## 3. What we built off of

- **Branch:** `perf-report-repro` (worktree `/home/dev/local/helion-perf-repro`) = the **live PR
  head rebased onto current `upstream/main`** (1 commit on top of main). The heuristic source is
  byte-identical to the open PR; only the base moved forward. We rebased (rather than using the PR's
  stale base) so the numbers reflect what actually ships, incl. already-merged infra (e.g. the fast
  dispatch cache). *Verify the heuristic diff is byte-identical to the PR after rebasing.*
- **Env:** `/home/dev/helion/.venv` (triton 3.7.0). Never `pip install`; use `PYTHONPATH`.
- **Harness lineage:** the timing/replay method is a hardened version of the original
  `_lab/perf_report/perf_report_bench.py` from the reduction-redesign work, made self-contained here.
- **Vendored, verbatim:** the kernels + builders the harness needs (see §4). vLLM kernel bodies are
  **byte-identical copies** of `vllm/kernels/helion/ops/*` (verified line-count); only their
  vLLM-internal imports were stubbed by a shim so they run standalone.

---

## 4. File map

```
perf-repro/
├── README.md                  ← you are here
├── perf_report_bench.py       ← the harness: builds a cell, extracts seed/default configs,
│                                 compiles+accuracy-gates+times all arms, writes per-cell JSON.
│                                 One process per (corpus, kernel). --resume for crash recovery.
├── aggregate_report.py        ← reads results/*.json, re-derives ratios from raw µs, emits
│                                 SUMMARY.md + summary.json (reproduction vs generalization split,
│                                 per-corpus, disasters, acc-fail table, launch-overhead x-check).
├── run_all.py                 ← driver: runs every (corpus,kernel) as its own process, sequential,
│                                 per-process timeout, --smoke (1 cell each) gate, resumable.
├── overnight_supervisor.sh    ← relaunches run_all.py until complete (auto-restart on driver death).
├── shapes.json                ← the corpus definition: per-kernel shape lists per split (§5).
│
├── deps/                      ← everything vendored so the branch is self-contained
│   ├── refs.py                  pure-torch reference impls (what tc compiles; the accuracy oracle)
│   ├── ab_three_arm_transfer.py builders for transfer kernels (fused_linear_jsd, grpo)
│   ├── transfer_kernels.py      + shapes_transfer.py  (transfer builder deps)
│   ├── mreduction_styles_view_only.py  builders for the backward (M-reduction) kernels
│   ├── bench_arms.py            vLLM kernel specs + tuned-config lookup (nearest_vllm_config)
│   ├── kut/                     vendored vLLM Helion kernel bodies (byte-identical to upstream ops/)
│   ├── vllm/                    minimal shim so kut/* import without a full vLLM install
│   └── vllm_configs/            snapshot of vLLM tuned JSONs for kernels our vllm-src predates
│
├── tools/                     ← diagnostics (NOT part of the headline run)
│   ├── acc_sweep.py             fast accuracy-only sweep (no timing) — find failing cells
│   ├── triton_head_to_head.py   dump/compare emitted Triton: Helion seed vs inductor call()
│   └── probe_*.py               ad-hoc single-cell probes used during investigations
│
├── notes/                     ← methodology + findings write-ups (defensibility artifacts)
│   ├── LAUNCH_OVERHEAD_NOTE.md  why cold-L2 do_bench hides CPU launch overhead (+ the boundary)
│   ├── ACCURACY_FIXES.md        the bf16-fp32-accum + harness rsqrt-dtype fixes (benchmark-branch only)
│   └── QK_NORM_ROPE_FINDING.md  the out-of-sample kernel + the tile-size miscompile it surfaced
│
└── results/                   ← output
    ├── <corpus>__<kernel>.json  per-cell Level-2 raw data (see §6)
    ├── SUMMARY.md / summary.json machine + human report (regenerate with aggregate_report.py)
    ├── MORNING_SUMMARY.md       the running "for review" log of every correction + judgment call
    └── logs/                    per-process stdout
```

---

## 5. Corpus design — reproduction vs generalization

The single most important structural idea. Shapes are split into two families, reported
**separately**, so "I reproduced the posted numbers" and "it generalizes" are distinct claims:

**Reproduction corpora** (match the PR's posted shapes exactly):
`curriculum` (9 example kernels) · `transfer` (fused_linear_jsd, grpo) · `mreduction`
(rms_norm_bwd, layer_norm_bwd) · `vllm` (4 quant kernels).

**Generalization corpora** (`*_gen`) — shapes/kernels the heuristic **never saw**:
- `curriculum_gen` / `transfer_gen` / `mreduction_gen`: **unseen shapes** — verified absent from
  *every* train/val/test/robustness split of the PR's curriculum, chosen to be realistic (real
  model dims + interpolations between seen bands) and to have a **novel reduction width**.
- `vllm_gen`: a stratified sweep of vLLM's **exact tuned-grid keys** (every hidden dim × log-spaced
  token counts) — so `G_vllm` is an exact-config comparison, and it covers the full decode→prefill
  regime vLLM actually serves (the posted 4 shapes/kernel were a favorable subset).
- `qk_norm_rope_gen`: an **out-of-sample KERNEL** (`fused_qk_norm_rope`) — structurally unlike the
  curriculum (3D grid + inner RMS reduction + RoPE epilogue). The strongest overfit test: does the
  heuristic even fire correctly on a kernel it was never designed around? (It did — and surfaced a
  latent codegen miscompile; see `notes/QK_NORM_ROPE_FINDING.md`.)

Total: 9 corpora, 455 cells.

---

## 6. Benchmarking technique (the defensible core)

Every choice here has a reason a skeptic will ask about. Details/experiments in
`notes/LAUNCH_OVERHEAD_NOTE.md`.

- **Primitive:** CUDA-event timing on the same 256 MB **cold-L2 flush** used by triton `do_bench` /
  tritonbench. Same ingredients as `do_bench`, arranged as **interleaved round-robin** across arms
  (Helion's own `interleaved_bench` mechanism) so slow drift (thermal/clock) is common-mode and
  **cancels in the ratio** — the gold standard for A/B ratios.
- **Reps:** scaled to a **100 ms/round budget + 25 ms warmup** (matching tritonbench `do_bench`
  defaults), clamped [5, 1000]. Fast kernels get more reps (tighter median where jitter is worst),
  slow kernels fewer. Median-of-9 rounds, escalate to 15 on >5% spread. Each cell records the rep
  count it used.
- **Cold-L2, no CUDA graphs in the headline** for functional kernels: this is the deployment regime
  *and* the regime the autotuner selects in. Cold-L2 `do_bench` **hides CPU launch overhead** for
  kernels whose per-call host dispatch < the ~86 µs flush (verified: `do_bench ≈ cudagraph device
  time`); the flush's own GPU time is a CPU "run-ahead budget". Boundary + proof in the note.
- **Per-cell cudagraph cross-check:** every arm is *also* timed via cold-L2 CUDA-graph replay
  (`coldgraph_us`); the aggregator flags any cell where `do_bench` and cudagraph diverge (the canary
  that launch overhead leaked in). Mostly clean; the in-place vLLM kernels are the exception (below).
- **vLLM-family metric = cudagraph DEVICE time**, not eager. vLLM deploys these under CUDA graphs,
  so launch is amortized in production; device time is the deployment-faithful, launch-invariant
  kernel-vs-kernel number. Validated against `torch.profiler` device counters (<1 µs agreement).
  In-place kernels must be timed **without cloning** read-only args each rep (cloning a 10 MB
  cos/sin cache adds tens of µs/rep) — production declares `mutates_args` and overwrites in place.
- **Accuracy-gated before timing** vs an eager fp32-reference. Tolerances atol=rtol=1e-2 (fp8: ~1
  ULP). **Close-enough inclusion rule:** a cell joins its ratios if the seed passes accuracy **OR
  the default makes the identical mistake** (same acc_detail) — then the miss is a benign
  kernel/dtype fact the whole family shares (bf16-accumulator margin, fp8 tie-rounding), so the
  speed comparison is still apples-to-apples (marked with a †). A seed that fails while the default
  is *correct* is genuinely wrong → **excluded from every ratio** (this is how the qk miscompile
  cells are handled).
- **Fair torch.compile baseline.** Our reference *is* the tc baseline, so a badly-written reference
  handicaps tc. We fix references so tc fuses well — e.g. rewrote `.item()` graph breaks (vLLM
  quant) and `index_put`/scatter → single slice-write (qk_norm_rope: 6 kernels → 2). Principle:
  *"we wrote the torch code, so we make it fuse."* Same standard applied to Helion.
- **Per-cell isolation:** fresh process per (corpus, kernel), `torch._dynamo.reset()` per shape,
  fresh `HELION_CACHE_DIR` per process, `kfn.reset()` per shape in config extraction (see the
  per_token_group bug in §8).

**Level-2 raw data** — every `results/<corpus>__<kernel>.json` row stores, per arm: `us`,
`coldgraph_us`, per-round `round_medians`, `reps_per_round`, `spread`, the full `seed_config` /
`base_default_config`, which heuristic fired, and accuracy pass/fail + max-abs error. Ratios are
derived from this; you can re-aggregate or re-analyze without re-running the GPU.

---

## 7. Results (PR #2996, this run)

Headline geomeans (>1 ⇒ seed faster). Full tables in `results/SUMMARY.md`.

| scope | G_tc (vs torch.compile) | G_def (vs unseeded default) | G_vllm (vs vLLM tuned) |
|---|---|---|---|
| **reproduction** | 1.042 (n188) | 2.589 (n188) | 0.987 (n16) |
| **generalization** | 1.102 (n252) | 2.899 (n257) | 0.955 (n106) |

Reading: the heuristic **beats the unseeded default ~2.6–2.9×** (its actual job), is **~parity or
slightly ahead of torch.compile**, and is **within ~1–5% of vLLM's hand-tuned per-shape configs**.
Generalization holds (G_tc even higher on unseen shapes) — the overfit worry is not borne out.

Interpretation caveats worth carrying to a manager:
- G_def is large (up to 15× on backward kernels) mostly because the **untuned default is
  pathological** on those kernels (register-spilling tile), not because the seed is superhuman — the
  seed does the *obvious correct thing* the default fails to. Honest framing: "seed fixes a bad
  default."
- Losses are real and explained: `fused_linear_jsd` bf16 loses to tc (~0.76–0.87) because Inductor
  fuses the shared softmax stats across its 4 softmax-family ops (fewer HBM re-reads); the seed's
  config is fine, it's a below-the-config codegen-fusion gap. `per_token_group` is ~parity (a prior
  "regression" was a harness bug, §8).

---

## 8. Gotchas we hit (read before trusting any number)

These cost real time and each silently corrupted results until caught. Watch for them next time.

1. **`kfn.reset()` between shapes.** Config extraction reused one bound kernel across a kernel's
   shape sweep; for a `static_shapes=False` kernel with no `hl.specialize(num_tokens)`
   (per_token_group), shapes collided in the bind cache and inherited an earlier shape's seed →
   timed ~4× slow. Looked like a real regression; was a harness bug. **Always `kfn.reset()` per
   shape.**
2. **`default_config()` ≠ unseeded default.** If the heuristic sets `promote_seed_to_default`,
   `default_config()` returns the *seed* → you'd compare seed-vs-seed. Use `_base_default_config()`.
3. **Clone cost in in-place-kernel timing.** Cloning read-only args every rep inflates small-shape
   latency by 3–5×. Time in place (production does); clone only for the accuracy check.
4. **Eager vs device time for launch-bound kernels.** On small kernels a residual ~5–7 µs/call
   dispatch doesn't fully cancel in the ratio and is seed-pessimistic. Use cudagraph device time for
   the launch-amortized (vLLM) regime.
5. **Unfair torch.compile references.** `.item()` and `index_put`/scatter in the reference force
   graph breaks / extra kernels, making tc look artificially slow. Check `torch._dynamo.explain`
   (graph breaks) and `run_and_get_triton_code` (kernel count) on every tc reference.
6. **bf16 accumulation in the kernel *or* the reference.** A reduction that accumulates at input
   dtype fails accuracy at bf16; `torch.sum` upcasts internally, so make the kernel match. Also
   ensure the reference is graded against the *same-precision* intermediate the kernel receives (we
   had a ref recompute rsqrt in fp32 while the kernel got bf16 → false failure).
7. **Config drift between the PR branch and reported numbers.** Confirm the heuristic source is
   byte-identical to the open PR after rebasing (diff the specific files); base drift is fine, source
   drift invalidates reproduction.
8. **Buffered subprocess output / teardown hangs.** `... | tail` buffers everything until exit (you
   see nothing live) — write to a file. Inductor SubprocPool can hang in teardown after printing
   DONE (zombie compile-worker) — the supervisor + per-cell checkpointing make this survivable.

---

## 9. Redoing this for a new heuristic PR

1. **Branch:** rebase the PR head onto current `upstream/main`; confirm the heuristic source files
   are byte-identical to the open PR (`git diff <pr-head> <your-branch> -- <heuristic files>` →
   content-only diff empty).
2. **Corpus:** edit `shapes.json`. Keep the reproduction shapes = the PR's posted shapes exactly.
   Add `*_gen` corpora: unseen shapes (verify against the PR's shape curriculum splits) + at least
   one out-of-sample *kernel* if the class allows. Vendor any new kernel bodies into `deps/kut/` (or
   equivalent) byte-identical, with a pure-torch reference in `deps/refs.py`.
3. **Builders:** add a builder that returns `(kfn, args, ref, acc_fn, tc_ref)` and route it in
   `run_real_cell`. In-place kernels get a dedicated cell runner (see `run_qk_norm_rope_cell`).
4. **References:** write each `ref` to (a) match the kernel's math *and precision* exactly, and (b)
   fuse well under torch.compile (no `.item()`, no scatter; check kernel count). This is the fair-
   baseline step — don't skip it.
5. **Smoke gate:** `run_all.py --smoke --out-dir <d>` — one cheapest cell per (corpus,kernel).
   Must be 100% clean (builds → times → writes) before committing GPU-hours to the full run.
6. **Full run:** `overnight_supervisor.sh <out-dir>` (foreground, one process at a time — a timing
   study must own the GPU; never run GPU jobs concurrently). Resumable.
7. **Aggregate:** `aggregate_report.py <out-dir>` → `SUMMARY.md`. Re-derives everything from raw µs.
8. **Explain the losses.** For any cell where the seed loses, dump both Triton kernels
   (`tools/triton_head_to_head.py` or `HELION_PRINT_OUTPUT_CODE=1` + `run_and_get_triton_code`) and
   say *why* from the code — is it a config the seed got wrong, or a below-the-config codegen gap? A
   manager will ask, and "the config is fine, it's a fusion gap" vs "the seed mis-sized the tile"
   are very different conclusions.
9. **Log every correction + judgment call** in a `MORNING_SUMMARY.md`-style running doc. When a
   number moves, record why (which bug, which fix, old→new). That doc is what makes the audit
   trustable.

## 10. Reproduce this exact run

```bash
# from the worktree root, venv = /home/dev/helion/.venv
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  bash perf-repro/overnight_supervisor.sh $PWD/perf-repro/results
PYTHONPATH=$PWD /home/dev/helion/.venv/bin/python \
  perf-repro/aggregate_report.py $PWD/perf-repro/results
# single (corpus,kernel):  perf_report_bench.py --corpus <c> --kernel <k> --out-dir <d> [--resume]
# smoke gate:              run_all.py --smoke --out-dir <d>
```
