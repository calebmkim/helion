# Matmul configs the CuTe autotuner cannot reach

Each case below is a config that **compiles, passes accuracy, and runs measurably faster**
than anything the CuTe autotuner can find on this tree — but which the search space excludes.

Baseline: **`origin/main` = `74378a6ca`** (this branch's base). B200, 148 SMs.

## How to falsify any claim here

Every unreachable winner is given in full in **`unreachable_winners.json`**, keyed by class and
shape. For each one:

1. **Run it.** `helion.Config(**unreachable_winner)` + `set_config` → it compiles and passes
   accuracy. Measure it; you should see `winner_perf`.
2. **Then challenge the autotuner.** Autotune the same shape on this tree, at any effort level
   and any budget. Try to beat `winner_perf`.

The claim is that step 2 cannot succeed — not that it is unlikely, but that the config space
the search draws from contains nothing that fast. **If you find a reachable config that
matches or beats `winner_perf` on any of these shapes, that case is refuted and should be
struck from this document.**

Each section also names the gate, so the exclusion can be checked by reading rather than
running: the winner will either fail `set_config` in *strict* mode, or survive strict mode and
then be silently rewritten when passed through the search path
(`normalize(..., _fix_invalid=True)`).

**Reading the numbers.** Every percentage in this document is
`winner_perf / best_reachable_perf - 1` on the **same shape, same tree, both measured in
Helion** — i.e. how much throughput is left on the table by the search-space exclusion. It is
never a comparison against cutlass, quack or aten; where such a ratio is quoted it is labelled
inline (e.g. "0.729× vs quack"). Units are cold-L2 CUDA-graph TFLOP/s unless noted. Shapes are
`M×K×N`; "production" shapes are real model dimensions (Llama-2/3, Qwen2.5/3, DeepSeek).

**Write your own measurement harness.** This document deliberately ships **no benchmarking
script**. If the measurement code came from the same source as the claims, a harness bug would
be baked into both the claim and its verification — and the two traps documented at the bottom
of this file were exactly that: harness artifacts, not config properties. An independently
written harness is the check. What it needs to control:

- **one cudagraph capture per process** (see trap 1 — recapture manufactured a phantom +13.4%);
- **cold vs warm L2 stated explicitly**, since several of these configs invert between the two;
- **accuracy validated on every config**, not just the fast ones (see trap 2);
- **locked clocks** if comparing across processes, or interleaved A/B within one process.

The reachability claims are a different matter — those are pure config-space queries with no
timing involved. `unsearchable_repro/reachability_probe.py` is provided as **one reference
example** of how to ask that question (it covers the two C5 shapes); the other classes are
deliberately left for you to probe, since writing your own is itself a check on that one. It
does not touch the GPU timer, so it is cheap to read and re-run.

---

## Summary

| # | class | biggest miss | production shape | winner vs reachable |
|---|---|---|---|---|
| **C1** | 16-bit `ab_stages` hard-capped at 6 | `1152×4096×4096` bf16 | `512×14336×4096` bf16 — Llama-3-8B MLP | **+18.7%** |
| **C2** | `(ab=6, c=2)` absent from the direct-entry stage table | `8192×8192×10240` bf16 — Llama-70B QKV | same | **+6.0%** |
| **C3** | `block_k` capped at 128 | `64×6144×2048` fp8 | `64×2048×4096` fp8 — Qwen3 decode | **+14.4%** |
| **C4** | on FFI-eligible 16-bit shapes the search space is 1 config | `2048×4096×32000` bf16 — Llama-2 lm_head | `2048×4096×14336` bf16 — Llama-3-8B down-proj | **+12.2%** |
| **C5** | `block_n=128` with `cluster_m=2` is rewritten to 256 | `512×2048×4096` fp8 | `512×4096×4096` bf16 | **+25.4%** |
| **C6** | 16-bit `ab_stages` **search** cap of 3 | `512×14336×4096` bf16 | `640×4096×4096` bf16 — ragged serving batch | **+37%** |
| **C7** | fp8 small-grid early return excludes a tile band | — | — | **unmeasured** |
| **C8** | a seed emits a `cluster_m=2` config the search space cannot hold | `512×8192×2048` fp8 | `256×4096×4096` fp8 | **+22%** |
| **C9** | `128 < M < 256` cannot use the 2-CTA path | `192×4096×4096` fp8 | same | **+28%** |

---

## C1. 16-bit `ab_stages` is hard-capped at 6; 8 wins

**Gate.** `helion/_compiler/cute/tcgen05_config.py:1109-1110`
```python
if dtype_bytes == 2:  # FP16/BF16
    return 6
```
The SMEM budget is enforced *separately* by `max_ab_stages_that_fit`, which independently
refuses depths that don't fit — so this literal blocks configs that **do** fit.

**Check the exclusion.** `set_config` with `tcgen05_ab_stages=8` fails on all three surfaces:
`InvalidConfig: tcgen05_ab_stages must be in [1, 6], got 8`. Running the winner therefore
requires lifting the literal (6 → 8); `max_ab_stages_that_fit` still refuses over-budget
depths independently, so nothing unsafe becomes reachable by doing so.

| | shape | best reachable (what autotune finds) | unreachable winner | winner vs reachable |
|---|---|---|---|---|
| **biggest** | `1152×4096×4096` bf16 | ab6 @ 997.5 | `[128,256,32]` cm1 **ab8** flat → **1184.3** | **+18.7%** |
| **production** | `512×14336×4096` bf16 — Llama-3-8B MLP | ab6 @ 948 | `[256,128,64]` cm2 **ab8** → **1088.7** | **+15%** |
| extra | `896×4096×4096` bf16 | ab6 @ 817.7 | `[128,256,32]` cm1 **ab8** → **921.1** | **+12.7%** |

Full configs: `unreachable_winners.json`, class `C1`.

---

## C2. `(ab_stages=6, c_stages=2)` is absent from the direct-entry stage table

**Gate.** `helion/_compiler/cute/tcgen05_constants.py:404-407`
```python
TCGEN05_DIRECT_ENTRY_STAGE_TUPLES_BY_BK = {64: ((3, 2), (6, 4)), 128: ((3, 2),)}
```
`_validate_direct_entry_ab_stage_envelope` admits `ab > 3` only when `(ab, c)` appears in this
table for the drawn `bk`. It is a **table lookup, not a budget test** —
`max_ab_stages_that_fit` reports `256×256×64 cm2 → 6`, i.e. ab6 fits the 203,776 B budget.
`(6,2)` is simply not listed.

**Check the exclusion.** These winners fail even *strict* `set_config`. The discriminator: take
the same config and set `c_stages=4` (the table-legal `(6,4)`) — accepted. Only the `(6,2)`
pairing is refused.

| | shape | best reachable (what autotune finds) | unreachable winner | winner vs reachable |
|---|---|---|---|---|
| **biggest / production** | `8192×8192×10240` bf16 — **Llama-70B QKV-proj** | seed swz1 @ 1331.6 | `[256,256,64]` cm2 ab6 **c2** swz8 l2g1 → **1408.5** | **+6.0%** |
| second | `512×5120×152064` bf16 — Qwen3 lm_head | ab3 @ 1228.4 | `[256,256,64]` cm2 ab6 **c2** swz1 l2g1 → **1289.3** | **+4.9%** |

Full configs: `unreachable_winners.json`, class `C2`.

---

## C3. `block_k = 256` is not drawable — and shipped fp8 golden configs use it

**Gate.** `helion/language/matmul_ops.py:374`
```python
max_search_k = min(128, pow2_floor_at_least(static_k, mma_k))
```
The block-size fragment can only offer `bk ∈ {32, 64, 128}`. 3000 real `random_flat()` draws
never produce `bk=256`.

The in-tree fp8 decode answer key wants exactly that.
`helion/_compiler/autotuner_heuristics/cute_matmul_formula.py:164`, verbatim:
> `fp8 decode bn=32 : bk=256/ab=8 (196 608) > bk=128/ab=12 (147 456) [pretuned]`

The `bk=256` variant beats the pretuned `bk=128` one, and `bk=256` configs ship in the AOT
table under `pretuned_kernels/` — so this tree already contains configs its own search cannot
re-derive.

| | shape | best reachable (what autotune finds) | unreachable winner | winner vs reachable |
|---|---|---|---|---|
| **biggest** | `64×6144×2048` fp8 rowwise | 16.256 µs / 0.888× aten | `[64,32,256]` cm1 ab8 l2[1] → 14.208 µs / 0.776× aten | **+14.4%** |
| **production** | `64×2048×4096` fp8 — Qwen3 decode width | — | `[64,32,256]` cm1 ab8 (shipped AOT entry) | not re-derivable |

Five more shipped AOT entries affected: `64×2048×2048`, `64×2048×12288`, `64×4096×4096`,
`64×12288×4096`.

This is a **`block_k`** limit, not an `ab_stages` limit — a deeper *K tile* at ab8, not a
deeper pipeline. The fp8 `ab_stages` cap is 12 and is not implicated.

---

## C4. On FFI-eligible 16-bit shapes the entire search space is one config

Not one config excluded — the search collapsing altogether.

**Gate.** `helion/_compiler/cute/tcgen05_config.py:2023`
```python
EnumFragment((True,)) if for_search else BooleanFragment()
```
Every drawn candidate therefore carries `tvm_ffi_launch=True`, which alone satisfies
`_target1_tvm_ffi_promotion_requested`. `_fix_target1_tvm_ffi_search_config` then pops **every**
`tcgen05_*` key (`:2060-2063`) and overwrites the config with the single direct-entry seed
(`:2091`, `config.update(seed.config)`).

**Check the exclusion** — config-space only, no timing. 1200 `random_config()`
draws per shape:

| metric | measured |
|---|---|
| distinct configs drawn | **1** |
| full population (300 requested) | **2** |
| distinct after 5 mutation rounds (4906 flat configs explored) | **8** |
| domains reached | `ab_stages {3}`, `l2_swizzle {1}`, `l2_groupings {[2]}` |

Holds on all 12 FFI-eligible shapes tested, so **every** better config on these shapes is
unreachable:

| | shape | best reachable (what autotune finds) | unreachable winner | winner vs reachable |
|---|---|---|---|---|
| **biggest** | `2048×4096×32000` bf16 — **Llama-2 lm_head** | seed l2g[2] @ 1291 | `[256,256,64]` cm2 ab6 c2 l2g1 → **1448.7** | **+12.2%** |
| **production** | `2048×4096×14336` bf16 — **Llama-3-8B down-proj** | seed bk128/ab3/l2g2 @ 1366 | `[256,256,64]` cm2 ab6 c4 l2g1 → **1416.5** | **+3.7%** |

Others on the same gate: `8192×8192×10240` +6.0%, `512×5120×152064` +4.9%,
`1024×4096×16384` +4.3%, `2048×13824×4096` +3.9%, `2048×16384×4096` +3.2%.

**Collateral from the same gate**, worth checking while here: `encode_config` raises on this
tree's *own* `ffi=False` compiler seed (`ValueError: Invalid enum value False for EnumFragment.
Valid choices: (True,)`), and `pattern_neighbors` raised **11,520×** in a single mutation run.

---

## C5. `block_n=128` with `cluster_m=2` is rewritten to 256

**Gate.** `helion/_compiler/cute/tcgen05_config.py:863-868`
```python
# Only the fully validated narrow-N CLC+aux-TMA seed may keep
# block_n=128; other candidates use the canonical block_n=256.
if is_narrow_clc_aux_tma:
    block_sizes[n_index] = TCGEN05_TWO_CTA_EDGE_K_TAIL_NARROW_BLOCK_N   # 128
else:
    block_sizes[n_index] = TCGEN05_TWO_CTA_BLOCK_N                      # 256
```
Every drawn `bn` snaps to 256 unless the candidate *is* one specific validated seed. For plain
fp8 that escape is closed — `is_narrow_clc_aux_tma` requires
`exact_shape_aux_kernel_detected ∧ allow_edge_k_tail_family`, both false — and when it does
fire it pins `bk=256`, giving `[256,128,256]`, a different config.

**Check the exclusion** — config-space only, no timing. Reproduce with
`unsearchable_repro/reachability_probe.py` (`--case c5-fp8`, and `--case c5-bf16` for the bf16
row), on `512×2048×4096` fp8:

```
compiler seeds: 2 — both [256,256,128]   →  [256,128,128] cm2 IN SEEDS: False
[256,128,*] cm2 in 400 draws: 0
cm2 tiles drawn: (128,128,64) (128,128,128) (256,256,128) (256,256,32) (256,256,64) (128,128,32)
set_config STRICT:    OK  →  [256,128,128] ab8
through SEARCH path:      →  [256,256,128]        ← silently rewritten
```

The winner is *settable* but not *findable*, and handing it to the search path destroys it.

The bf16 row fails with a **different signature** — worth running `--case c5-bf16` to see it:
`cluster_m=2` is suppressed outright (0 seeds, and `cm2 tiles drawn: {}`), and the config is
not even settable (`InvalidConfig: tcgen05_ab_stages > 3 is only supported by the validated
TVM-FFI direct-entry path (or fp8 within the SMEM budget)`), landing at
`[128,128,128] cm1 ab3` through the search path. Two gates compound there rather than one:
the wave-quantization veto below, plus the `ab>3` restriction of C6.

| | shape | best reachable (what autotune finds) | unreachable winner | winner vs reachable |
|---|---|---|---|---|
| **biggest** | `512×2048×4096` fp8 | shipped seed `[256,256,128]` @ 420.1 | `[256,128,128]` cm2 ab8 → **527.0** (warm ties cutlass 1.00×) | **+25.4%** |
| **production** | `512×4096×4096` bf16 | `[128,128,128]` cm1 ab3 @ 645.3 | `[256,128,128]` cm2 ab4 → **699.1** | **+8.3%** |

The band is the **entire fp8 `cluster_m=2`-admitted set** — 13 of 13 admitted shapes scanned,
0 exceptions. Not a medium-M special case.

On the bf16 shape a second gate compounds it: `matmul_ops.py:477-493` computes
`work_clusters = (512//256)·(4096//256) = 32 < num_sms//4 = 37`, suppressing `cluster_m=2`
search entirely — missing by 5 clusters.

---

## C6. The 16-bit `ab_stages` **search** cap is 3, and `ab>3` is admitted for fp8 only

Distinct from C1: that is the hard cap of 6; this is the search *fragment* stopping at 3.

**Gate.** `helion/_compiler/cute/tcgen05_config.py:1988` sets `ab_stages_max = 3` on the
`for_search` branch, and the `ab > 3` admission at `:1486`/`:1979` is gated
`dtype_bytes == 1` — fp8 only. The in-tree justification at `:1966-7`:
> *"SEARCH stays capped at 3 (budget-aware) since the generalized seed runs at ab=3 and
> **deeper pipelines are not worth searching**."*

That is an explicit performance claim; the cases below contradict it.

**Check the exclusion.** Draw census on this tree: `ab_stages ∈ {1,2,3}` over 2000 draws. A
config with `ab_stages=4` survives strict `set_config` but is demoted to `ab3` through the
search path.

| | shape | best reachable (what autotune finds) | unreachable winner | winner vs reachable |
|---|---|---|---|---|
| **biggest** | `512×14336×4096` bf16 — Llama-3-8B MLP | 794 (no `cluster_m=2` seeds offered at all) | cm2 tile + **ab8** → **1088.7** (0.729× → 1.001× vs quack) | **+37%** |
| **production** | `640×4096×4096` bf16 — ragged serving batch | ab3 @ 525.9 | `[128,256,64]` cm1 **ab4** → **619.1** | **+17.7%** |

Also: `384×4096×4096` **+8.2%** (486 → 526, ncu-attributed), `2048×8192×256` **1.177× cold /
1.043× warm** under locked clocks from a blind climber, `256×4096×4096` at ab6.

---

## C7. The fp8 small-grid early return excludes a tile band

**Gate.** `helion/_compiler/cute/tcgen05_config.py:851-861` — an fp8 shape admitted via the
small-grid family that draws `bm ≤ 128` is pinned to `[128,128]` and **returns early**,
skipping the `block_n` shaping and every downstream step. Within an otherwise-admitted family,
every `[128, bn≠128]` `cluster_m=2` tile is excluded.

Affects e.g. `512×2048×4096` and `4864×4096×128` fp8.

⚠ **No performance measurement exists for this band** — only its two endpoints (small-grid
`128×128` and rectangular `[256,bn]`) have been measured. The exclusion is proven by reading
the gate; the cost is unknown. Listed for completeness, not as a demonstrated miss.

---

## C8. A seed emits a `cluster_m=2` config the search space cannot hold

**Gate.** An ordering problem in `fix_search_config`: validation runs before the shaping steps,
so the seed is judged against a surface that later changes. The seed emits `[128,128,128]` cm2
ab12 (196,608 B — exactly on the SMEM isobar), but the search space cannot hold `cluster_m=2`
on these shapes, so the seed does not survive injection.

`cute_matmul_formula.py:352` records in-tree what that config should deliver:

| | shape | winner vs reachable |
|---|---|---|
| **biggest** | `512×8192×2048` fp8 | **+22%** (932 vs 762) |
| **production** | `256×4096×4096` fp8 | **+13%** (526 vs 467) |

Also `512×2048×2048` fp8 **+16%** (347 vs 298).

Affects a swept grid of 14 fp8 shapes — `512×4096×512`, `256×4096×{128..2048}`,
`1024×4096×{128,256,512}`, `2048×4096×{128,256}`, `4096×4096×128` — plus 5 milder
(`{256..4096}×4096×64`).

⚠ Mechanism reproduced live; the exact root-gate attribution is provisional.

---

## C9. `128 < M < 256` cannot use the 2-CTA path at all

**Gate.** `helion/language/matmul_ops.py:442-446` — `allow_full_tile_persistent_pid_types`
requires `static_m % max_search_m == 0`. For M=192, `192 % 128 == 64 ≠ 0` → False, which strips
both persistent pid types (`tcgen05_config.py:2015-2017`), collapses `cluster_m` to `(1,)`
(`:2027`), and leaves `cluster_n=1`. The whole band is confined to 1-CTA, flat.

| shape | strategy | perf | vs cutlass 450.0 |
|---|---|---|---|
| `192×4096×4096` fp8 — ragged serving batch | native 1-CTA best `[64,128,128]` cm1 ab8 c4 | 350.1 (warm p50 486) | 0.778× |
| same | **pad M 192→256 externally**, then run the `256×4096×4096` config | **449.4** effective (padded kernel 14.34 µs) | **0.998×** |

⚠ **Scope caveat.** For the *native* problem this is not a config-space hole: `pid_type` is the
only unreachable component of the native winner, and the persistent config and its reachable
flat twin measure **bit-identically**. The 28% gap closes only by padding M **outside** the
kernel, which no config can express. Included because the practical symptom is the same, but it
is a missing capability rather than a search restriction.

---

## Measurement caveats

- Performance numbers are re-quoted from prior measurement runs with their original qualifier
  (cold-L2 CUDA-graph unless noted). **Reachability was re-measured on this tree** for C1–C6
  and C9.
- C1's ab8 numbers required lifting the cap locally to make the config runnable at all — that
  is the finding.
- C2's winners were verified unsettable, with the `(6,4)` control verified settable.
- C7 has no performance measurement.
- C8's mechanism is reproduced; its attribution is provisional.

### Two traps that invalidate naive force-config comparisons

Both were hit while producing this document, and both produced a confidently wrong result
before being caught.

1. **Cudagraph recapture can manufacture a phantom win.** A `cluster_n=2` config measured
   **+13.4%** and was recorded as "the primary fast-mode driver". Re-measured under
   one-capture-per-process: **1.0000×**. On a shape chosen to favour it: **0.886×**. ncu shows
   kernel duration constant at ~12.3 µs across both apparent modes — the delta was pure
   dispatch artifact. Control cudagraph capture before trusting any number here.
2. **A config can benchmark fast and compute wrong answers.** That same `cluster_m=2 +
   cluster_n=2` combination silently returned **wrong results while measuring as the fastest
   config in a climb**, and hung the GPU on verification. Its exclusion from the search space
   is load-bearing for correctness, and it is deliberately **not** in this document.
