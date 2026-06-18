# WS2 curriculum extension — M-reduction seed for 4 new backward kernels

**Branch** `ws2-mreduction`. **Base** `628d7a99` (prior WS2 deliverable: rms/ln/softmax bwd).
**Commits** `f8a21c56` (lever fix) → `af969c5b` (role-based re-route + non-pow2 gw match). Additive on
top of the banked m-reduction seed; squash-ready.

## 1. Goal
Extend the m-reduction (grad_w / M-axis tile-accumulation) seed heuristic to four new backward kernels
(`_lab/curriculum_candidates/mreduction_styles.py`) without regressing the banked baseline (9 forward
kernels + rms/ln/softmax bwd). Bars (method §3): per-shape floor `G = tc/seed ≥ 0.75`, per-(kernel,
dtype) geomean ≥ 0.85.

| kernel | class | fires | M-collapse over | grad_x |
|---|---|---|---|---|
| `bias_grad` | A pure collapse | m_reduction (re-routed) | M→[N] | none |
| `dyt` | A collapse + elementwise | m_reduction (re-routed) | M→[N] | elementwise [M,N] |
| `group_norm` | B decoupled | m_reduction | N(grid)→[C] | over (Cg,S) per group |
| `instance_norm` | B decoupled | m_reduction | B(grid)→[C] | over S |

## 2. Contributions (all gated)
### (a) Lever fix — `feature_extent` = full resident footprint
`MReductionFact.feature_extent` read the gw accumulator width (**C** only), but the 3-D norms'
resident grad_x tile is `[inner, C, S]`. So the inner byte-cap undersized the row footprint by a
factor of **S** → `inner` came out S× too large → spilled *worse than the generic default*. Fix
(`device_ir.build_m_reduction_facts`): `feature_extent` = product of the input load's **materialized
feature dims** (dims where `MemoryOpFact.subscript_block_ids[d] is None`, indexed to a `reduction=True`
block ∉ block_sizes/reduction_loops), max over loads. The reshape-split (G, Cg) blocks never appear in
the un-reshaped load's `indexed_block_ids`, so no double-count. Pure derived-fact read of walker facts —
no graph walk. 2-D norms unchanged (N == N). **Result fp32:** group_norm (1024,64,128,32) G **0.22→2.01**,
instance_norm (1024,32,256) G **0.19→1.05**.

### (b) Role-based re-route — bias_grad/dyt T2 → m_reduction
A pure-collapse M-reduction registers its `.sum(dim=0)` as a single `ReductionFact` over the **inner M
tile**, so T2 fired with a persistent feature `R_BLOCK` seed `[1,8192]` that floors the grid to 1 →
~M tiny partials → expensive finalize (fp32 train: bias_grad all below floor, dyt 12/14 below).
`build_m_reduction_facts` now declines only when a `ReductionFact` sits on an axis **other** than an
inner row tile (a genuine feature reduction — every forward kernel + softmax_bwd); `_triton_reduction_
eligible` gains `and not m_reduction_facts` so T1/T2 yield. The same M_CTA-occupancy + inner-byte-cap
seed serves the M-collapse. **Result fp32 train:** dyt geo **0.67→1.37** (0 below floor), bias_grad geo
**0.39→0.83**.

### (c) Non-pow2 gw-accumulator match
At non-pow2 feature extent the gw partial buffer is padded to next_pow2, so its dim resolves to `None`
(the feature block keeps the true extent). The builder matches a 1-D `(None,)` accumulator as the gw
buffer when there is a single materialized feature block — so non-pow2-N bias_grad/dyt fire.

## 3. Gates (full stack PASS @ af969c5b)
- **Gate R** ×2: forward 739 cells **0 changed**; rms/ln/softmax bwd seeds **byte-identical**
  (config-recorder before/after). The banked baseline is provably unperturbed.
- **Gate D** 3/3 PASS: `build_m_reduction_facts` is a pure derived fact (no `device_ir.graphs` /
  `node.users` / `_classify_load_dataflow`); `feature_extent` faithfully tracks the resident footprint
  (divergence tests failed to refute); the disjointness key is structural block-id set-membership, not
  a dtype/identity fence; the `(None,)` fallback is sound documented padding-reproduction.
- **Gate H** 2/2 KEEP: both levers key on faithful workload properties (bytes/footprint budget;
  structural role of the reduction axis), catastrophe-rescue, bounded downside.
- **Gate A** PASS: 2/3 skeptics PASS (the 1 REFUTE was procedural — the driver-run independent
  reproduction). Independent fresh-authored script (`/tmp/indep_repro.py`), two timers (do_bench +
  cuda.Event) agree: group_norm (1024,64,128,32) **G 2.07 / 2.26**, dyt (8192,4096) **G 1.53 / 1.60**,
  both accuracy-passing — matching the harness numbers.
- **Gate F** PASS (re-route is a structural lever): mechanism code-verified in the generated Triton —
  T2's grid-floored seed produces ~M tiny partials + a 128 MiB finalize buffer; the M_CTA-occupancy
  lever cuts that to ~num_sm partials (64× fewer) + a 2 MiB finalize. No inert fields (the seed sets only
  block_sizes/num_warps/num_stages/pid_type). Boundary: the win degrades gracefully to the T2 seed for
  M ≤ ~2·num_sm, never regresses.
- **Gate E** PASS (no overfit), all 3 dtypes: held-out TEST tractable cells track train (dyt 1.24/0.95/
  0.97; group/instance tractable 1.6–2.2, all 0 below floor); wide-S kernel-bound on TEST too (structural,
  not memorization); bias_grad consistent (codegen-bound).

## 4. Per-kernel results (train, acc-gated geomean of G = tc/seed)
| kernel | fp32 | bf16 | fp16 | verdict |
|---|---|---|---|---|
| **dyt** | 1.37 | 1.00 | 1.00 | **CLEARS BAR all dtypes** (0–1 below floor) |
| **bias_grad** | ~0.81 | 0.66 | 0.67 | codegen-bound pure sum (below bar; noise-sensitive ±, robust verdict) |
| **group_norm** | 0.64 | 0.82 | 0.80 | full split below bar; see §5 |
| **instance_norm** | 0.70 | 0.71 | 0.86 | full split below bar; see §5 |

### group/instance: the bar is met on every KERNEL-TRACTABLE shape
Split by feature footprint F = C·S (the curriculum band axis): F ≤ 32768 = kernel-tractable, F > 32768
= wide-S (9/14 of the train split).

| kernel | tractable-F geo (fp32/bf16/fp16) | below-floor | wide-S geo |
|---|---|---|---|
| group_norm | **1.86 / 2.42 / 2.33** | 0 | 0.35 / 0.45 / 0.44 |
| instance_norm | **1.78 / 1.67 / 2.80** | 0 | 0.21 / 0.44 / 0.44 |

The seed **clears the bar handily** (geo 1.7–2.8, 0 below floor) on every shape the kernel can run; the
wide-S cells drag the full-split geomean.

## 5. Structural limits (NOT seed limitations) — see HRQ
- **group/instance WIDE-S is KERNEL-authoring-bound.** The given kernels grid *only over N/B* and
  materialize the full `[inner, C, S]` resident tile for grad_x (x_hat reused after the stats
  reduction). For wide C·S (vision GroupNorm, S = H·W = thousands) the resident tile spills to local
  memory (256 KB–4 MB) and low-N shapes launch only N CTAs (8–16) → 8–16× slower than torch.compile,
  which tiles S. `inner` is already at its floor of 1 and S is materialized with no tile knob, so **no
  seed config can help** (knob sweep over all warps/stages confirms `[1,1]w32`=0.13 is best; the full
  autotuner hangs on the un-compilable large-block trials). The seed is **optimal-given-the-kernel**
  (it beats the generic default). **Fix = re-author the kernels to tile S** (2-pass: stats over
  S-tiles, then apply) like a real GroupNorm/InstanceNorm backward + add a seed S-tile lever — a
  kernel-engineering change beyond the seed deliverable (HRQ #1).
- **group/instance_norm CRASH on non-pow2 C** (robustness canaries C=96/320). `weight[:].reshape(...)`
  plus the kernel-wide next_pow2 padding of non-pow2 C make the compile fail/hang (config-free, no seed
  involved — a curriculum-kernel bug). Realistic GroupNorm uses C=96/320 (non-pow2, ÷ groups). The
  1-line weight fix is insufficient (the kernel hangs elsewhere); needs the same kind of re-authoring as
  the wide-S fix (HRQ #3). [An earlier robustness note here claimed "no crashes" — that was a
  silent-truncation error in the sweep grep; corrected.]
- **bias_grad is a codegen-bound pure sum.** torch.compile uses a split-reduction Helion's grid+finalize
  can't match (worse in bf16). It is *not* a hard ceiling — the oracle ~ties tc — but the per-shape
  optima are scattered (`[16,16]w4` / `[128,4]w32` / `[64,64]w8ns3`) with no clean faithful single rule;
  the best trade ([16,16]w4) clears the marginals but regresses (16384,1024) 1.14→0.79. This is the
  autotuner's (Product B) territory; the Product-A seed (geo 0.83, up from a broken 0.39) is a strong
  start (HRQ #2).

## 6. Files
- `helion/_compiler/device_ir.py` — `build_m_reduction_facts` (lever fix + re-route + non-pow2).
- `helion/_compiler/autotuner_heuristics/triton.py` — `_triton_reduction_eligible`,
  `TritonMReductionHeuristic.is_eligible`.
- `helion/autotuner/config_spec.py` — `MReductionFact.feature_extent` doc.
- `_lab/curriculum_candidates/` — kernels, shapes, `mr_bench.py` (bench), `factdump_new.py`,
  `CURRICULUM_NOTEBOOK.md` (worker log), this report.
