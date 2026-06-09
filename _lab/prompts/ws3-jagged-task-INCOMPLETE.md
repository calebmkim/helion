# TASK (WS3): jagged / ragged reductions

# ⚠️⚠️ INCOMPLETE DRAFT — DO NOT LAUNCH ⚠️⚠️

> **STATUS: NOT A FINISHED PROMPT.** This file exists only to preserve the WS3 *findings* and
> scope so they aren't lost. The operational sections are **stubs**. WS3 is deliberately sequenced
> to be authored **after WS2 lands its tile-accumulation fact recognizer** — this prompt must point
> at that *real* infra, and several inputs below are still unknown (see "What's MISSING"). **Do not
> hand this to an autonomous agent in its current state.** The FINDINGS section is solid and
> measurement-backed; everything below it is scaffolding to be completed later.

---

## FINDINGS (solid — measurement-backed, keep)

**Oracle-vs-tc study (Helion full autotune vs torch.compile, single H100, 2026-06-08; tritonbench
default jagged sweep, do_bench metric, all accuracy PASS):**

| kernel | Helion oracle / tc | verdict | default→oracle headroom |
|---|---|---|---|
| **jagged_sum**  | **1.08–2.33×** (Helion wins) | **oracle ≥ tc → SoTA reachable in the existing config space** | 4–26× |
| **jagged_mean** | **1.04–1.76×** (Helion wins) | **oracle ≥ tc → SoTA reachable** | 10–26× |
| **jagged_softmax** | **0.20–0.68×** (Helion loses) | **oracle < tc → DSL/codegen ceiling** (gap narrows with autotune budget but does NOT close) | — |
| **jagged_layer_norm** | unmeasured | example hardcodes `@helion.kernel(autotune_effort="none")` → autotuner disabled at source; needs a 1-line source tweak to assess | — |

**Key conclusions:**
1. **jagged_sum / jagged_mean are a NORMAL, high-ROI seed problem — NO new lever needed.** Max-
   autotune already reaches/beats tc using the *existing* config knobs (jagged-tile block_size,
   warps), so the seed only has to **pick** the right config; it does not need any new perf lever.
   (This REFUTES the earlier "jagged is the most lever-heavy expansion / needs a load-balancing
   lever" hypothesis — that was guessed from the naive `hl.tile(num_rows)` grid, not measured.)
2. **The un-autotuned default is catastrophic** (4–26× slower than its own oracle, 4–20× slower than
   tc). So for sum/mean the *entire* win is config selection — exactly what a good seed captures.
   This is the headroom WS3 exists to capture.
3. **jagged_softmax is the cautionary tail:** even full autotune is 2–5× behind tc → a codegen
   ceiling on the two-pass (max + exp-sum) ragged softmax. Per the oracle-retarget rule
   ([[hillclimb-oracle-retarget]]), a seed there targets the **oracle**, not tc — still worth
   seeding (huge default gap) but it will NOT beat tc; closing the tc gap is a separate
   kernel-authoring / codegen project. softmax + layer_norm are also multi-pass (multi-reduction),
   so they are the harder tail regardless.

**Why jagged is blocked today (the linchpin):** jagged reductions produce **no `ReductionFact`**.
`hl.jagged_tile` (helion/language/loops.py:594-612, 745-755) allocates a regular `block_sizes` tile
(`size=None`, `reduction=False`); the `.sum(dim=1)` is manual `+=` accumulation. T1 only iterates
`reduction=True` rdims (device_ir.py:889); T2 **explicitly declines** `size=None`/jagged dims
(device_ir.py:1091-1095). It is a deliberate, documented exclusion. **So the certain new work is:
build a `JaggedReductionFact` so the gate fires, then key the existing knobs to it.**

**The unification with WS2 (this is why WS3 waits for WS2):** jagged's `.sum`-over-an-inner-device-
loop-tile is the **same pattern** as WS2's backward `grad_w` (a tile-accumulation reduction over a
`reduction=False` inner tile). **WS2 builds the recognizer; WS3 reuses + extends it.** Differences:
- **jagged:** dynamic extent (`size=None`), single-fact, **existing knobs suffice (no new lever)**.
- **M-reduction (WS2):** static extent, co-exists with the N-axis fact (multi-fact plumbing), wants
  a tall→tiny lever.

**Scope:** clean targets = **jagged_sum, jagged_mean** (reachable, high ROI). Deferred tail =
jagged_softmax (codegen ceiling, oracle-retarget), jagged_layer_norm (multi-pass + unmeasured).

**Real-world relevance:** recsys — embedding-bag pooling over variable-length feature lists
(DLRM-style), ragged-sequence reductions (HSTU). High production QPS. (MoE routing reductions are
dense/tiny → not this; MoE's heavy compute is grouped-GEMM = matmul = excluded by the gate.)

---

## WHAT'S STILL MISSING (TODO before this is a launchable prompt)

- [ ] **Depends on WS2.** WS3 must reuse WS2's tile-accumulation fact recognizer — write this prompt
      only once that infra exists, and point at the actual class/builder names WS2 produced.
- [ ] **The answer-key is NOT yet extracted.** We know the oracle ≥ tc for sum/mean, but NOT *which
      config knobs* it used to win (block_size? warps? num_stages?). The crux of the seed is a
      field-diff of the oracle's winning configs vs the default — a Step-2/3 item to run (dump
      `jagged_sum`/`jagged_mean` oracle winners + field-diff). Until that's known, the seed's actual
      rule is undefined.
- [ ] **Dynamic-extent (`size=None`) handling.** How the `JaggedReductionFact` represents a dynamic
      reduction extent, and how the seed keys the existing knobs when `static_rnumel is None` (today
      the explicitly-untuned fallback path, triton.py:433-453).
- [ ] **Harness wiring.** Jagged runs via tritonbench (`benchmarks/run.py --kernel jagged_sum`) with
      NestedTensor inputs; the lab harness needs jagged input builders.
- [ ] **Curriculum.** The benchmark used tritonbench's default sweep; WS3 needs a curated shape set
      anchored to real recsys sizes (variable seqlen distributions, feature counts).
- [ ] **SHAPES vs TRANSFER decision.** Unlike the WS1 probes, we likely WANT to *tune + ship*
      jagged_sum/mean (huge headroom) → they'd be `SHAPES` (train/val/test), not held-out probes.
- [ ] **softmax / layer_norm decisions.** Defer? oracle-retarget? the layer_norm `autotune_effort`
      source tweak needed to even measure it?

## SKELETON (to be filled in after WS2 — placeholders only)

- Read-order preamble (method / local-setup / gate-prompts / this file) — *stub*
- Resume context (fresh climb; fresh notebook `ws3_notebook.md` + ledger key `ws3`) — *stub*
- Goal (beat tc on sum/mean per-shape, all 3 dtypes; oracle-retarget softmax) — *stub*
- The new machinery (reuse WS2's recognizer for the dynamic-extent jagged case) — *stub, needs WS2*
- Where everything lives (kernels, tritonbench baselines, harness, curriculum) — *stub*
- Deliverable / DoD — *stub*
