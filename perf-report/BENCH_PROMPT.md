# Benchmarking prompt — reduction-seed perf report (fp32 + bf16, 3-arm, no autotune)

You are producing a **performance report** for the Helion reduction-seed heuristic, to be shown to a
manager. Your job is to benchmark every tested kernel at every test shape, at **both fp32 and bf16**
where applicable, and produce a clean results table. **This is a MEASUREMENT + REPORTING task, not a
hill-climb** — you are not editing the heuristic; you are characterizing it as it stands.

Read `KERNEL_INVENTORY.md` (same dir) FIRST — it has the full kernel list, shape counts, source
locations, and which dtypes apply to each corpus. This prompt covers the how.

## 0. Environment (hard rules — from prompts-lab/method/hillclimb-method.md §1 + machines/local-setup.md)

- **Worktree:** `/home/dev/local/helion-redesign`, branch `reduction-redesign` (the heuristic under
  test lives here — NOT `helion-pr-with-lab`, which `local-setup.md` names for a different run).
  Confirm `git rev-parse --abbrev-ref HEAD` = `reduction-redesign` before benching.
- **Interpreter:** `/home/dev/helion/.venv/bin/python`. **Never pip install.**
- Run from `cwd=/tmp` with `PYTHONPATH=/home/dev/local/helion-redesign` (+ the transfer/vllm dirs as
  needed). **Assert `helion.__file__` is under the worktree at the top of every script** (the silent
  wrong-helion footgun).
- **GPU: one dedicated H100 80GB, L2 = 50 MB, 132 SMs.** Not shared unless told. Run **foreground,
  ONE kernel per process, one job at a time. NEVER detach/background a GPU job** (they SIGKILL
  silently). JSON-checkpoint after every (kernel,shape,dtype) so a kill loses nothing.
- Commit results/scripts to the branch as you go; don't `git push`.

## 1. The three arms (exactly what the manager asked for — NO autotune)

Per (kernel, shape, dtype), time three arms **in one process on the same input tensors**:
1. **seeded Helion** — the reduction seed as default:
   `HELION_PROMOTE_REDUCTION_SEED=1 HELION_AUTOTUNE_EFFORT=none`.
2. **unseeded Helion default** — heuristics disabled (the compiler's base config):
   `HELION_DISABLE_AUTOTUNER_HEURISTICS=1 HELION_AUTOTUNE_EFFORT=none`.
3. **torch.compile** — `torch.compile(reference)`, NOT max-autotune. **You choose the fairest
   apples-to-apples comparison** (see below) — do not blindly default the mode.
Report per row: `seed_us`, `default_us`, `tc_us`, and the two ratios `G_tc = tc_us/seed_us`
(>1 = seed beats tc) and `G_def = default_us/seed_us` (>1 = seed beats the unseeded default).
The seed-vs-default ratio is the headline "what the heuristic buys"; seed-vs-tc is the external yardstick.

### Measure GPU time FAIRLY — this is your call, make it symmetric
The one rule: **all arms must be measured on the SAME footing.** The goal is the fairest possible
GPU-time comparison. `mode="reduce-overhead"` in torch.compile IS CUDA-graph capture/replay (it
strips per-launch CPU overhead); Helion's arms do not self-graph. So the trap is an **asymmetric**
comparison — e.g. a CUDA-graphed tc arm (reduce-overhead) vs a non-graphed Helion arm charges CPU
launch overhead to Helion only, fabricating a tc win (footgun #10). Pick ONE consistent regime and
apply it to all three arms:
- **either** no CUDA graphs anywhere (all three arms plain, so CPU launch overhead is present in all —
  it biases every arm toward 1.0 equally, the SAFE direction, footgun #9), **or**
- **CUDA graphs on all three arms** (seed + default wrapped the same way you'd wrap tc; measures pure
  GPU time with launch overhead removed from all) — **but only if you also handle the cold-L2 trap**
  (#9): plain cudagraph replay over the same buffers keeps a ≤50 MB working set L2-HOT → a
  physically-impossible 3–5 TB/s and a distorted RATIO, so a graphed metric needs an explicit L2 flush
  before each replay for sub-L2 shapes.
Ideal = pure GPU/device time, measured identically for all arms (e.g. profiler `self_device_time`, or
a cold-L2 `do_bench`). What you must NOT do: CUDA-graph one arm but not another, or cudagraph-wrap a
tc arm that is ITSELF `reduce-overhead` (double-graphing, #10). State in the report which regime you
chose and why it's symmetric.

**Arm 3 (torch.compile) applies ONLY to the real-workload corpora** (curriculum, transfer,
m-reduction, vLLM). For the **synthetic + adversarial** kernels, run ONLY arms 1 and 2 (seeded vs
unseeded Helion default) and report just `G_def` — there is no meaningful torch reference for them (see
§2/§6).

These env flags + the promote mechanism are implemented in `_lab/bench/run_seeded.py` (read its
docstring). But that script routes through tritonbench, which does NOT know our 4 corpora and hits
footgun #15 — see §3 for the recommended harness.

## 2. What to run — LOAD `shapes.json` (machine-readable; `SHAPES.md` is the human mirror)

**Iterate `shapes.json`, do not re-parse markdown or re-import the shape modules.** Each corpus has
`kernels{name:{source, shapes{split:[...]}}}`, `required_splits`, `dtypes`, `arms`. Loop
`required_splits` (curriculum=`["test"]`, others=`["all"]`) × `dtypes` × `arms`. `_meta` carries the
arm env flags + the required-shape tally (182). `SHAPES.md` has the same values formatted for humans.


**SCOPE — read `SHAPES.md` for the exact shapes and the split rule:**
- **curriculum** (9 kernels): bench the **`test` split ONLY = 66 shapes** (fp32 AND bf16). The
  train/val/robustness shapes are listed in SHAPES.md for reference but are **NOT required** for this
  report — do not bench them unless separately asked.
- **transfer / robustness** (8 kernels, 80 shapes, no split → ALL): fp32 AND bf16.
- **m-reduction** (6 kernels, 16 shapes, no split → ALL): fp32 AND bf16.
- **vLLM** (5 kernels, 20 shapes): **NATIVE dtype only** — each kernel's builder already uses its
  intended dtype (bf16 input / fp8 output; some scales fp32). Run them AS-IS in that native dtype; do
  NOT force fp32 and do NOT add a bf16 variant sweep. 3 arms (seed / default / tc).
- **synthetic probes** (13) + **adversarial synth** (7): run LAST, report SEPARATELY, and bench them
  **seed vs unseeded Helion default ONLY — NO torch.compile arm** (report just `G_def`). They are
  generality/correctness diagnostics with no meaningful torch reference (probes have 1 fixed shape each
  and were built to stress the categorization pass; adversarial ones are persist-vs-chunk N-sweeps).
  `G_def` here is the real signal — "does the heuristic's config beat the unseeded default on these
  generality kernels" — which is exactly what we want to show for them. Frame as "heuristic-generality
  coverage," not headline perf; do not let them dominate the summary. Native dtype (fp32 as authored,
  unless a kernel specifies otherwise) — no dtype sweep needed for these.

Required run size ≈ **344 real (kernel,shape,dtype) cells** × 3 arms (curriculum-test 66×2 + transfer
80×2 + mreduction 16×2 + vLLM 20×1). Foreground GPU — checkpoint aggressively, run corpus-by-corpus.

## 3. Recommended harness — extend the config recorder's corpus iterators (do NOT hand-roll from scratch)

`helion-redesign/_lab/harness/unified_config_recorder.py` already wires ALL FOUR real corpora
(`_CORPORA = {curriculum, transfer, vllm, mreduction}`), each yielding `(corpus, kernel, shape, dtype,
fn, args, split)`. Build your 3-arm bench on top of those iterators so you inherit the exact
kernel/shape/builder wiring the heuristic work used. For the seed/default/tc arms, reuse the pattern in
`_lab/bench/bare_fwd_seed_vs_tc.py` (bare-forward, dynamo-reset-per-shape, single-process, contention-
guarded) — but generalize it to all corpora + both baselines + both dtypes.

### THE DTYPE GAP (must fix before you can do bf16 on 2 corpora)
- **transfer builders ARE dtype-parameterized** (`ab_three_arm_transfer._make` → `build(shape, dt)`),
  so fp32+bf16 is free there.
- **curriculum builders are HARDCODED fp32** (`run2_measure_g.build_rms_norm` etc. use
  `dtype=torch.float32`), and **m-reduction builders are hardcoded** (fp16/fp32). To get bf16 you must
  parameterize these builders (add a `dtype=` arg, thread it through). This is the main harness change
  needed — do it cleanly and verify the tensors actually took the dtype (footgun #6d: some ops silently
  default; assert dtype on the built tensors).
- **vLLM builders are bf16-in/fp8-out by construction** — leave as-is, bf16 only.

## 4. Benchmarking footguns (READ hillclimb-method.md §4 IN FULL — the list is authoritative)

The method file's §4 (footguns 1–15) governs. The ones most likely to bite THIS run:
- **#1 FORWARD ONLY** — `requires_grad=False`, no `.backward()`, time the BARE forward
  (`helion.kernel(fwd.fn, config=...)`) for both Helion arms and the tc reference. (The m-reduction
  kernels are *backward* kernels by name but are still benched as a single forward pass of that kernel.)
- **#2 dynamo reset per shape** (`torch._dynamo.reset()`) or tc silently recompiles into slow
  dynamic-shapes mode — unfair to tc.
- **#4 single-process head-to-head** — all 3 arms in ONE process on the SAME tensors, median-of-9–15.
- **#11 ONE FRESH PROCESS PER KERNEL** — do NOT batch many kernels/variants through one long process
  (accumulated dynamo guards fabricated a bogus 2.18× "win" once). Arms together, kernel fresh.
- **#9 + #10 COLD-L2 device time** — L2 is **50 MB** here; any working set ≤ 50 MB (e.g. rms_norm bf16
  (2048,4096) = 16 MB — most curriculum shapes) stays L2-hot under plain cudagraph → physically-
  impossible 3–5 TB/s AND a distorted ratio (a real seed LOSS read as a fake 1.27× win). Use a cold-L2
  metric: this Triton build's `do_bench` flushes L2 between reps (re-confirm if the Triton version
  changed), OR profiler `self_device_time` with an explicit ~128 MB flush before every call. Plain
  cudagraph is OK ONLY when the working set ≫ 50 MB (wide-N / large-vocab). Default to the harness's own
  `do_bench` mode; deviate only on measured evidence.
- **#6 accuracy gate BEFORE timing**, vs an eager reference built at the SAME dtype, upcast both to
  fp32 before allclose. EXCLUDE acc-fail / NaN rows from the geomean and surface them. **#12: a bf16
  acc-FAIL is usually a low-precision ACCUMULATOR, not a bad seed** — naked `.sum()`/`.mean()`
  accumulates at input dtype; if an fp32 run of the same kernel passes, it's the accumulator (a
  known kernel-source issue, not the heuristic). Report those rows as "acc-fail (accumulator)", don't
  fold their latency into perf.
- **#6c arm-equivalence** — both timed arms must compute the SAME work (jsd once timed an extra dX
  output → loss inflated ~11–18%). Check the reference computes exactly what the kernel does.
- **#7 verify the config actually ran** — record the normalized running config after
  `bound.ensure_config_exists(args)` to prove the SEEDED arm ran the seed and the DEFAULT arm ran the
  unseeded base (they must differ; if identical, the promote/disable flag didn't take).
- **#13 noise floor** — sub-~25 µs shapes swing ±25%; re-run any row whose do_bench spread > ~5%,
  take median-of-medians. Many curriculum small-M shapes are here.
- **#3 tc baseline is NOT max-autotune.** Beyond that, the mode is your fairness call (§1 "Measure
  GPU time FAIRLY"): whatever regime you pick must be applied SYMMETRICALLY to all three arms. #10:
  never cudagraph-wrap an arm that is ITSELF `reduce-overhead` (double-graphing).
- **#5 contention guard** only if the GPU turns out shared (it's dedicated by default) — a quick
  `nvidia-smi` before headline timings + confirm clocks aren't throttling is cheap insurance.

## 5. Step 0 sanity check (do ONCE before the full sweep — method §Step 0)

On ONE representative curriculum shape (e.g. rms_norm (8192,4096) fp32): run all 3 arms, confirm
(a) accuracy passes, (b) the SEEDED arm's normalized config is the seed (not the default), (c) the
number is cold-L2 (implied bandwidth well under HBM peak for a ≤L2 shape, not a 3–5 TB/s artifact),
(d) a hand-rolled single-process `do_bench` cross-check agrees within 3%. If the harness disagrees
>3%, STOP and fix the harness before sweeping — a miscalibrated harness produces plausible-but-wrong
numbers you'd chase for hours.

## 6. Deliverable

Write results to `prompts-lab/perf-report/results/` (create it), JSON per (kernel,shape,dtype,arm) +
a summary. Two clearly-separated sections:

**(A) Real workloads** (curriculum, transfer, m-reduction, vLLM) — 3 arms. Per (kernel, dtype): the
per-shape rows (seed_us / default_us / tc_us / G_tc / G_def) and a **per-(kernel,dtype) geomean of
G_tc and G_def** over accuracy-PASSING rows only (footgun #6a). Group by corpus. vLLM rows are its
native dtype (no fp32/bf16 split). Note any acc-fail rows and why (usually the bf16 accumulator, #12),
so the manager sees them as kernel-source facts, not heuristic regressions.

**(B) Generality diagnostics** (synthetic probes + adversarial synth) — **2 arms only, seed vs
unseeded default, report `G_def`** (NO tc column). Label it clearly as generality/correctness coverage,
not headline perf. For the adversarial N-sweeps, show the per-N `G_def` curve (that's the point of them).

**Failures are expected and FINE — just mark the cell, don't drop it or crash the run.** If an arm
fails to compile (or OOMs, or the config won't normalize), put **`x`** (or `n/a`) in that cell's
latency and leave any ratio that needs it blank/`n/a` too; keep going. Record the reason in the JSON
(a short `error` string: which arm, `compile-fail` / `OOM` / `acc-fail` / `normalize-fail`) so the
manager can see WHY a cell is empty. An acc-FAIL (wrong/NaN output) is also `x` for perf and excluded
from geomeans (footgun #6a/#12) — distinct from a compile-fail. A ratio is only computed when BOTH its
arms produced a valid timing; otherwise `n/a`. Never let one failed cell abort the sweep (that's what
the per-cell JSON checkpointing in §0 is for).

Also report the **headline aggregate**: overall geomean of G_tc and of G_def across all real cells
**that have valid timings for both arms**, per dtype, and call out any per-shape disaster (G_tc < 0.75
on a realistic shape — the method's floor). Note how many cells were `x`/`n/a` and why.

## 7. What you are NOT doing
- No autotuning (the manager explicitly wants a fast no-autotune characterization).
- No heuristic edits (this is reporting; if you find a disaster, RECORD it for later, don't fix it).
- Don't chase weird shapes or hill-climb — just measure faithfully and report.
