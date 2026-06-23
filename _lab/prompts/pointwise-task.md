> **⚠️ CORRECTIONS — read first; these OVERRIDE anything that disagrees below.**
> - **Base = the tip of your NEWEST reduction stack: branch `reduction-4pr-stack` @ `9bf7b1f9`**
>   (worktree `/home/calebkim/helion-new-heuristics/helion-3stage`). That stack = stage1/2a/2b
>   (PRs **#2828 / #2829 / #2830**) + a local stage3 (matmul+reduction-epilogue). **Discover + record the
>   exact tip SHA at run start** (`git -C helion-3stage rev-parse reduction-4pr-stack`); it drifts.
>   Pointwise is **independent** of the reduction fact layer (it adds a brand-new fact + heuristic and
>   touches none of the reduction/matmul code), so it composes as **the next, cleanly-separable PR** on
>   top of that stack. Make a **fresh worktree on a fresh branch** off that tip for this work.
> - **Machine = `local-setup-devserver.md`** (the 4×H100 devserver). Interpreter
>   `/home/calebkim/.conda/envs/helion/bin/python` (conda env `helion`); **L2 = 50 MB/H100** (cold-L2
>   discipline); run scripts `cwd=/tmp` with `PYTHONPATH=<your worktree>` and assert `helion.__file__`
>   resolves under it (the editable `.pth` points at `local/helion` — the #1 wiring footgun).
> - **Remotes are INVERTED:** `origin` = pytorch/helion (UPSTREAM), `fork` = calebmkim/helion.
>   **NEVER `git push` to either, never touch a PR ref.** Commit to your local branch only; the human
>   assembles/pushes the PR.

# TASK: a seed heuristic for POINTWISE / ELEMENTWISE kernels (a NEW family — SwiGLU / GeGLU / activations / quant)

**Read in order (same dir): `hillclimb-method.md` (durable method — §2 walker/derived fact doctrine,
§3 the per-iteration loop + goal hierarchy, §4 benchmarking footguns, §5 gates, §6 orchestration +
the NEVER-STOP §6.0) → `local-setup-devserver.md` (THIS box: interpreter, GPU, L2, scripts, env
knobs) → `gate-prompts.md` (fill the gate frames VERBATIM) → this file (the *what*). Curriculum:
`shapes_pointwise_draft.py` (same dir; `validate()` → PASS, 6 kernels, 168 shapes).**

This file is a **map, not a checklist** (method §2 — be aggressive). The architecture, the fact/lever
design, and the order of operations are **yours**. Where anything here disagrees with the live code or
your own fresh measurements, **those win** — every path/line was read on one checkout and will drift.

═══════════════════════════════════════════════════════════════════════════
## 0. OPERATING CONTRACT — unattended run, READ TWICE
═══════════════════════════════════════════════════════════════════════════
**NEVER STOP until the deliverable is built, gated, reported, and you have climbed past it into
overtime — then keep climbing.** This is an unattended run; the human will not answer anything.
FORBIDDEN: stopping/pausing "for review" or "to confirm"; asking the human ANY question; ending a turn
with work outstanding; treating a dead-end ("no clean rule / noisy / stuck / at a ceiling / done") as
an exit instead of the trigger for the next action (fresh oracle, different workload property, the
BROADEN/refactor queue, the completeness-critic — method §6.0). If you catch yourself composing a
"should I…?" sentence: delete it, pick the most defensible option, **log it**, and continue. **Trust
your LOG over your context** (method §6.1): keep your OWN notebook + ledger
(`_lab/pointwise/{NOTEBOOK.md,ledger.json,REPORT.md}`); the gated log is the source of truth. The run
ends ONLY when a hard external block hits with literally no non-GPU work left.

═══════════════════════════════════════════════════════════════════════════
## 1. WHAT THIS FAMILY IS — and why it's a clean, empty niche
═══════════════════════════════════════════════════════════════════════════
A **pointwise / elementwise** kernel maps a flattened tile with **no reduction, no matmul, no
loop-carried accumulator** — it reads N input elements, computes, writes N output elements. It is
**bandwidth-bound**. The canonical carriers already ship: `examples/swiglu.py`
(`_swiglu_fwd(a,b) = SiLU(a)*b`), `examples/geglu.py` (`GELU_tanh(a)*b`), `examples/add.py`. All take a
bare `@helion.kernel()` (no pinned config) → they ride the **generic default**, which is the niche.

**The geometry that drives the whole curriculum.** For a gated MLP the elementwise op runs on
**`[M = tokens, N = intermediate_size]`** — it reads `gate[M,N]` + `up[M,N]`, writes `out[M,N]`
(`~3·M·N·itemsize`). **N = intermediate_size, NOT hidden_size.** (On GPU these kernels `view(-1)` and
`hl.tile(numel)` — a 1-D flatten; `add.py`/`exp.py` keep N-D `hl.tile(x.size())`. The fact/lever must
handle BOTH — key on total tile numel + bytes, agnostic to 1-D-flat vs N-D.)

**No pointwise heuristic exists today** — `autotuner_heuristics/__init__.py` `HEURISTICS_BY_BACKEND`
registers only skinny-gemm, B200-matmul, split-join-rotate (rope), and the two reduction heuristics.
This is a genuinely empty niche, disjoint from all of them (no `MatmulFact`, no `ReductionFact`, no
`AccumulatorFact` fires on a pure elementwise body — prove it, don't assume it).

**DISJOINTNESS RULE — REDUCTION WINS TIES; NEVER CLAW A KERNEL BACK.** The fact is *defined* by the
absence of a reduction. If a `ReductionFact` (or any welford / m-reduction / amax-for-scale machinery)
fires on a kernel, that kernel is **reduction-family — full stop.** `PointwiseElementwiseFact` must NOT
fire on it, and you must **never** re-route ("nab back") a reducing kernel into the pointwise track to
chase a number — leave it to the reduction heuristic and explain why in the log. This is both
faithfulness (the field's *meaning* is "pure elementwise map, no reduction") and disjointness (no kernel
claimed twice). It cleanly excludes, with NO special-casing: **per-token/per-block fp8 activation quant**
(scale = `amax(row)`), **fused add+RMSNorm/LayerNorm** (the norm reduces), and any **backward that
reduces over rows** (dyt-bwd, bias_grad, the norm bwds). Forward-only elementwise bodies are the scope.

═══════════════════════════════════════════════════════════════════════════
## 2. THE CRUX — measured headroom (SCOUTING — verify it yourself, method §2)
═══════════════════════════════════════════════════════════════════════════
These are prior-session numbers + a code read — **treat as hints to verify, not facts.** Reproduce the
3-shape test (§7) before relying on any of it.

- **Root cause (verified in source):** `config_spec.py:223-224` set `DEFAULT_NUM_WARPS=4`,
  `DEFAULT_NUM_STAGES=1`, and `:1789-1790` set **`default = 32`** block size whenever
  `total_ndim<=2 and reduction_numel<=128` — i.e. exactly a flattened pointwise tile (no reduction ⇒
  `reduction_numel=1`). **A 32-element tile for a bandwidth-bound op ≈ 315 GB/s ≈ ~10 % of H100 HBM.**
- **Gate (a) — default→oracle gap = HUGE & config-reachable.** Measured ~**5.4–7.0×** on standalone
  elementwise SwiGLU (H100, bf16, cold-L2) over a mini grid (bs∈{256..8192}, warps{2..16}, stages{1..4}):
  large-M 16384×11008 ≈7.0×, square 4096×4096 ≈6.3×, small-M 512×11008 ≈5.4×. **Dominant lever =
  block_size (32 → ~1–4K elements); warps/stages secondary.** No new lever needed.
- **Gate (b) — competitive = YES, at PARITY.** Oracle ties `torch.compile(max-autotune-no-cudagraphs)`
  within 0–5 % at ~2.2 TB/s on the same standalone op → the oracle is best-available, not beating a weak
  baseline. **Per the oracle-retarget doctrine: retarget the seed to oracle/tc parity; do NOT chase
  >1.0× vs tc** (the ceiling is fixed HBM bandwidth — chasing it is chasing noise).
- **Honest framing of the win.** The dramatic number is **default→seed (~5–7×)**; once a sane block_size
  is set the **seed→oracle remainder is modest (~1.1–1.3×)**. Report the baseline explicitly. The
  weakest case is small-M (M≈512, ~21 µs, near the noise floor) where one fixed config may leave
  ~10–20 % — which motivates a **size_hint-aware** seed (below), not a per-shape fence.

═══════════════════════════════════════════════════════════════════════════
## 3. ARCHITECTURE — a fresh derived fact + a new heuristic (the skeleton; the rest is YOURS)
═══════════════════════════════════════════════════════════════════════════
Build a **derived** `PointwiseElementwiseFact` + a new `TritonPointwiseSeedHeuristic` (register it in
`autotuner_heuristics/__init__.py`). Per the walker/derived doctrine (method §2): the derived fact
**NEVER walks the graph** — it reads slices of the existing **walker** `MemoryOpFact`s (raw per-op
itemsize / extent provenance) + trivial structural reads (`block_sizes`, `size_hint`), and is gated by
the **absence** of `ReductionFact`/`MatmulFact`/`AccumulatorFact`. Derive the fields, the recognizer,
and faithfulness per Gate D yourself; a sketch of what it likely needs:
- **total tile numel** (∏ block extents) and the **static problem numel** (M·N) — the occupancy input.
- **per-element byte traffic** = Σ over the tiled tensors of (read+write itemsize) — the feature that
  separates **traffic-2** (unary: relu_squared / bias_gelu / dyt) from **traffic-3** (gated act / add:
  swiglu / geglu / residual_add). Derive it from `MemoryOpFact` itemsizes, **never** a dtype literal or
  op name. (Asymmetric read≠write width — a per-TENSOR dequant `fp8→bf16` or a dtype cast — is a faithful
  *optional* coverage axis, NOT a headline kernel; the realistic fp8 path is per-token = reduction, §1.)
- **broadcast structure** (an input whose extent is [N] or [M,1] broadcast over the tile) — bias_gelu,
  dyt, MoE-scale need this; it must not change *which* kernels fire, only inform the tile.

**The lever — derive it via the hill-climb (spike it; don't pick a formula up front).** It is
block_size (the flattened/N-D tile) + num_warps + num_stages. Make it **size_hint-aware** (choose
block_size so `grid = numel / block_size` comfortably exceeds `num_sm × waves`; small grids prefer
fewer warps for occupancy, large working sets prefer bigger tiles + 1–2 extra stages for latency
hiding) and **bytes-aware** (scale the tile to bytes-MOVED, not element count — the traffic-2 vs
traffic-3 split is the probe that this is real: a traffic-3 kernel moves 50 % more bytes/element, so for
the same byte budget it wants a smaller element block_size). **Keep the FACT general (it recognizes the
whole elementwise family);
scope the CLAIM to what you measure**, widen via Gate H.

═══════════════════════════════════════════════════════════════════════════
## 4. THE KERNELS — the corpus (6 kernels; author the missing ones from NAMED refs — Gate-E firewall)
═══════════════════════════════════════════════════════════════════════════
`N`-dim meaning per kernel is in the table. In-tree carriers: `swiglu.py`, `geglu.py`, `add.py`
(+ `low_mem_dropout.py`, `exp.py`). The rest you **author from a named external reference** with its own
correctness oracle + realistic shapes (Liger-Kernel is the primary ref). The faithful lever (bytes +
occupancy) is **blind to which activation it is** — that blindness is what forces generalization.

| phase | kernel | math | traffic | N = | in-tree? | real-world anchor |
|---|---|---|---|---|---|---|
| **P1 — the spike (build first)** | `swiglu` | `SiLU(gate)·up` | 3 | intermediate_size | ✅ swiglu.py | Llama-2/3, Mistral/Mixtral, Qwen2/2.5/3, DeepSeek, Phi-3/4, Yi |
| | `geglu` | `GELU_tanh(gate)·up` | 3 | intermediate_size | ✅ geglu.py | Gemma-1/2/3, PaLM |
| **P2 — bytes/breadth (REQUIRED)** | `residual_add` | `h + residual` | 3 | hidden_size | ✅ add.py | every block ×2 — pure-BW floor (your "poi add") |
| | `relu_squared` | `max(x,0)²` (**1 input**) | 2 | intermediate_size | author (Liger `relu_squared`) | Primer / modded-nanoGPT — the ungated contrast |
| | `bias_gelu` | `GELU(x + bias[N])` (**broadcast**) | 2 | 4·hidden_size | author (`F.gelu(x+bias)`, **pre-projected x**) | GPT-2 / BERT / Falcon / OPT non-gated FFN |
| **HELD OUT** (freeze) | `dyt` (**fwd only**) | `tanh(α·x)·γ[N] (+β[N])` | 2 | hidden_size | author (Liger `dyt`) | Dynamic-Tanh norm-replacement (fwd reduction-free; **bwd reduces → out of scope**) |

- **All 6 are reduction-FREE in the forward** (audited): swiglu/geglu (silu/gelu·mul), residual_add
  (add), relu_squared (max²), bias_gelu (broadcast-add+gelu), dyt-fwd (tanh·γ+β). **`bias_gelu` MUST be
  authored on a PRE-PROJECTED `x[M,N]`** (`GELU(x+bias)`), never `GELU(x@W+b)` — the matmul would fire a
  `MatmulFact`. If your authored form of ANY kernel makes a reduction fire, that is the signal it belongs
  to the reduction track (§1 disjointness rule) — do not force it into the pointwise fact.
- **FIT vs HELD-OUT (Gate E):** fit on **swiglu, geglu, residual_add, relu_squared, bias_gelu**;
  **hold out `dyt`** (distinct broadcast-vector structure + norm-replacement role) for a single read at
  freeze — passing it proves interpolation, not memorization. (If you author a 7th, a `reglu`/bilinear
  GLU is a free held-out sibling: same shape/bytes, cheaper act.)
- **Negative-recognizer members (must NOT fire as pointwise — they PROVE the fact is faithful):** a
  pure reduction (`examples/rms_norm.py`), a pure matmul (`examples/matmul.py`), and a carried-state
  kernel — dump facts and confirm `PointwiseElementwiseFact` is **absent** for all three AND that their
  existing facts are **byte-identical** to before your change (the no-regression invariant —
  config-recorder diff, [[config-diff-regression-scoping]]).
- **Per-kernel oracle audit before a kernel enters the FIT** (method §5): drop any member whose
  default→oracle win needs *codegen*, not config (expectation: all 6 are config-reachable — it's the
  block_size). A dropped member stays on the worklist with its floor retargeted, it is not exempted.
- **Author standalone, pre-projected.** Every kernel takes the already-computed `[M,N]` tensors (gate/up
  for GLU; x/residual for add; x for the unary ops) — the matmuls live in the `nn.Module` wrapper, NOT
  the kernel. The pointwise kernel IS the elementwise core.

═══════════════════════════════════════════════════════════════════════════
## 5. THE SHAPES — inline below; canonical machine-readable copy = `shapes_pointwise_draft.py`
═══════════════════════════════════════════════════════════════════════════
`(M, N)`, **M = tokens = batch·seq_len**, **dtype = bf16** (real LLM serving; the tritonbench
geglu/vector_add operators silently default to fp32 — pin bf16). Five-bucket discipline (method): train
covers every N-band val/test probe (test = interpolation); measurable splits clear the ~20 µs floor
(bf16 itemsize=2 → floor math is `TRAFFIC·M·N·2 / 2e12`; small-N bands are paired with LARGE M to stay
measurable); robustness = **correctness-only canaries** (decode M, odd/prime M, non-pow2 N), NO G claim.
`validate()` in the `.py` enforces disjointness + envelope + floor (currently PASS). Verify the
anchors are realistic for your kernels and extend if a real config is missing.

```
# swiglu  (SiLU(gate)*up, N = intermediate_size, TRAFFIC=3)
train: (32768,1536)Qwen3-235B/DSV2-expert (16384,2048)DSV3-expert (16384,2880)GPT-OSS (8192,3072)Qwen3-0.6B
       (16384,8192)Phi-3-mini (8192,11008)Llama-2-7B (8192,12288)Qwen3-8B (8192,13824)Qwen2.5-14B
       (8192,14336)Llama-3-8B/Mistral (4096,17408)Qwen3-14B (4096,17920)Phi-4 (4096,18432)DeepSeek-V3
       (4096,18944)Qwen2.5-7B (4096,25600)Qwen3-32B (4096,28672)Llama-70B (2048,29568)Qwen2.5-72B
val:   (32768,2048)DSV3-exp (16384,9728)Qwen3-4B (16384,12288)Qwen3-8B (8192,16384)Mixtral-8x22B-exp
       (8192,20480)Yi-34B (4096,27648)Qwen2.5-32B (4096,29568)Qwen2.5-72B
test:  (8192,2880)GPT-OSS (16384,8960)Qwen2-1.5B (16384,11008)Llama-2-7B (4096,15360)interp
       (8192,22016)Llama-1-65B (2048,24576)interp (8192,28672)Llama-3-70B
robust:(1,11008) (7,14336) (33,14336) (64,11008) (128,18944) (256,14336)   # decode + odd M (<20us)
       (2048,11007) (512,14337)                                            # non-pow2 N

# geglu  (GELU_tanh(gate)*up, N = intermediate_size, TRAFFIC=3)
train: (16384,2880)cov (8192,3072)cov (8192,6912)Gemma-3-1B (8192,9216)Gemma-2-2B (8192,10240)Gemma-3-4B
       (8192,14336)Gemma-2-9B (8192,15360)Gemma-3-12B (4096,16384)Gemma-1-2B (4096,21504)Gemma-3-27B
       (4096,24576)Gemma-1-7B (4096,36864)Gemma-2-27B (2048,36864)Gemma-2-27B
val:   (16384,6912)Gemma-3-1B (16384,10240)Gemma-3-4B (8192,16384)Gemma-1-2B (8192,21504)Gemma-3-27B (4096,30720)interp
test:  (16384,9216)Gemma-2-2B (4096,14336)Gemma-2-9B (2048,21504)Gemma-3-27B (2048,24576)Gemma-1-7B (8192,36864)Gemma-2-27B
robust:(1,9216) (7,16384) (33,10240) (64,9216) (128,36864) (256,14336) (2048,16383) (512,9217)

# residual_add  (h+residual, N = hidden_size, TRAFFIC=3, pure-BW floor)
train: (32768,768)GPT-2 (16384,1024)Qwen3-0.6B (16384,1536)Qwen2-1.5B (16384,2048)Llama-3.2-1B
       (16384,2560)Gemma-3-4B (8192,3072)Llama-3.2-3B (8192,3584)Qwen2-7B (8192,4096)Llama-3-8B
       (8192,5120)Llama-2-13B (8192,7168)DeepSeek-V3 (4096,8192)Llama-3-70B
val:   (32768,1024) (8192,2048) (16384,3584) (16384,5120) (8192,8192)
test:  (32768,1536) (4096,2560) (16384,4096) (4096,7168) (16384,8192)
robust:(1,4096) (7,4096) (33,8192) (64,4096) (128,8192) (256,5120) (2048,4095) (512,8191)

# relu_squared  (max(x,0)^2, ungated 1-input, N=intermediate, TRAFFIC=2 — the contrast)
train: (32768,3072)Qwen3-0.6B (16384,8192)Phi-3 (8192,11008)Llama-2-7B (8192,14336)Llama-3-8B
       (4096,16384)Gemma-1-2B (4096,18944)Qwen2.5-7B (4096,28672)70B-FFN (4096,20480)OPT-13B/Yi-34B
val:   (32768,8192) (16384,11008) (8192,16384) (8192,28672)
test:  (16384,3072) (4096,14336) (8192,18944) (8192,20480)
robust:(1,11008) (7,14336) (64,16384) (256,11008) (2048,11007)

# bias_gelu  (GELU(x+bias[N]) non-gated FFN, BROADCAST bias, N=4*hidden, TRAFFIC=2)
train: (32768,3072)GPT-2-sm (16384,4096)BERT-lg/GPT-2-md (16384,5120)GPT-2-lg (16384,6400)GPT-2-xl
       (8192,8192)proxy (8192,10240)Phi-2 (8192,16384)mid (8192,18176)Falcon-7B (4096,20480)OPT-13B (4096,32768)Falcon-40B
val:   (32768,4096) (8192,6400) (16384,10240) (16384,16384) (8192,32768)
test:  (32768,5120) (16384,8192) (4096,18176) (8192,20480) (2048,32768)
robust:(1,4096) (7,4096) (64,8192) (256,16384) (2048,4095) (512,8191)

# dyt  (HELD OUT — read once at freeze)  tanh(alpha*x)*gamma[N], fwd only, N=hidden, TRAFFIC=2
train: (32768,768) (16384,1024) (16384,1536) (16384,2048) (16384,2560) (8192,3072) (8192,4096)
       (8192,5120) (8192,7168) (4096,8192)
val:   (32768,1024) (8192,2048) (16384,4096) (16384,7168)
test:  (32768,1536) (8192,2560) (16384,5120) (8192,8192)
robust:(1,4096) (7,4096) (64,4096) (256,5120) (2048,4095) (512,8191)
```

**MoE crux:** per-expert N is `moe_intermediate_size` (1536 Qwen3-235B/DSV2, 2048 DSV3) — a distinct
SMALL-N regime, NOT the dense intermediate (Mixtral has no separate moe dim — expert FFN IS
intermediate=14336/16384). **Confidence:** all N high-confidence (HF config.json) except Gemma-3-27B
(21504) — quick-verify before final lock.

═══════════════════════════════════════════════════════════════════════════
## 6. THE HARNESS — WRITE YOUR OWN, standalone-elementwise (this is the #1 footgun)
═══════════════════════════════════════════════════════════════════════════
**🚩 DO NOT measure through the swiglu/geglu *tritonbench operators*.** They bench the FULL
matmul-dominated MLP (`down_proj(act(gate_proj(x))·up_proj(x))`); the elementwise win is a rounding
error on wall time AND the Helion arm (standalone `swiglu_fwd`) vs the baseline arm (full `LlamaMLP`)
is **not arm-equivalent** (footgun §4 #6c). Measuring there makes pointwise look like NOT
low-hanging-fruit. **Author a forward-only harness** instead — clone `_lab/transfer/ab_three_arm_transfer.py`
(forces configs via `helion.kernel(fn.fn, config=cfg, static_shapes=True)`, med-of-9 cold-L2 `do_bench`,
acc-gate BEFORE timing, dynamo-reset per shape, single-GPU pin [[one-gpu-at-a-time]]).

**Three arms, all on the SAME pre-projected `[M,N]` tensors:**
- **`helion_default`** — `config_spec.default_config()`, heuristics off
  (`HELION_DISABLE_AUTOTUNER_HEURISTICS=1`). This is the broken-block_size=32 baseline.
- **`helion_seeded`** — force `compiler_seed_configs(bound.env, bound.host_function.device_ir)` via
  `configs=[seed]` (no autotune). Headline = **`seeded_vs_default`** (close the measured ~5–7× gap).
- **`tc`** — `torch.compile(elementwise_fn, mode='max-autotune-no-cudagraphs')` of the **same standalone
  elementwise function**. Report `seeded_vs_tc` (target = **parity**, per §2 / oracle-retarget).

**Footguns that bite HARDER hand-rolled (method §4 mandatory):** cold-L2 device time — robustness
shapes (M≤512 → ≤~34 MB) FIT in the 50 MB L2 and a hot-L2 read fakes multi-TB/s ([[cudagraph-l2-residency-footgun]]),
so keep them correctness-only and time cold if at all; **accuracy-gate BEFORE timing** (the swiglu/geglu
operators define rtol=0.05/atol=0.005 for bf16 — reuse it; exclude acc-fails/NaN from the geomean — the
silu/gelu compute fp32 internally so this is mainly a fp16-overflow / wrong-output guard); **force the
config** (never autotune mid-measure); `torch._dynamo.reset()` per shape; `empty_cache()` between
multi-GB shapes; one **fresh process per kernel** (§4 #11); flag/re-run sub-25 µs rows. Sanity-check
implied BW < HBM peak (a >3.3 TB/s reading is an L2 artifact, not a win).

═══════════════════════════════════════════════════════════════════════════
## 7. THE 3-SHAPE HEADROOM TEST — run this FIRST (Step 0, before building anything)
═══════════════════════════════════════════════════════════════════════════
Prove the niche is real on ONE kernel before investing: standalone `swiglu_fwd(a,b)`, bf16, cold-L2,
**large-M (16384, 11008)**, **small-M (512, 11008)**, **square (4096, 4096)** — measure `default(bs=32)`
vs a mini block_size/warps/stages grid (the oracle proxy) vs `torch.compile(max-autotune-no-cudagraphs)`.
**Expect ~5–7× default→oracle and ~1.0× oracle-vs-tc.** If it doesn't reproduce, STOP building the
formula and re-investigate the regime (don't hill-climb a phantom). Log the numbers; they calibrate the
whole run.

═══════════════════════════════════════════════════════════════════════════
## 8. POINTWISE-SPECIFIC GATE NOTES (fill the frames verbatim from `gate-prompts.md`)
═══════════════════════════════════════════════════════════════════════════
- **Gate D (faithful field + population + threshold) + the divergence kernel:** the canonical divergence
  kernel is **per-token fp8 quant** (`scale = amax(x, dim=-1)/448; out = (x/scale).to(fp8)`) — it *looks*
  like a pointwise quant but the `amax` over the feature dim makes a reduction fire. Author it, confirm
  `PointwiseElementwiseFact` does **NOT** fire on it (it routes to the reduction track instead — the §1
  disjointness rule, working as intended), AND confirm an elementwise kernel WITH a broadcast input
  (bias[N]) DOES fire. The byte-traffic field must be **populated from `MemoryOpFact` itemsizes**, not
  inferred from the op name or a dtype literal (the lucky-proxy class): verify it reports traffic-3 for
  swiglu and traffic-2 for relu_squared from the actual load/store itemsizes.
- **Gate F (mechanism in the lowered Triton):** your story is "block_size=32 starves bandwidth; a bigger
  tile + scaled warps saturates HBM." CONFIRM it — read the generated Triton, check the load/store
  vectorization + that every field you set is doing work (drop inert knobs — the dead-knob rule); a
  plausible story you didn't read the code to verify is a shape-scoped observation, not a banked rule.
- **Gate H (breadth, BOTH directions):** the lever keys on **numel / bytes / occupancy**, NEVER on
  activation identity (silu vs gelu), bare dtype, or a band bracketing exactly the curriculum's N → that
  is a BROADEN. A `dyt`-only or traffic-2-only knob that the bytes/occupancy lever should subsume is a fence.
- **Gate R (no new disaster) + config-recorder:** run the recorder over the FULL active matrix (all 6
  kernels × splits × dtypes) before/after every edit; the **negative-recognizer kernels (rms_norm /
  matmul / carried-state / the per-token-quant divergence kernel) must be 0-changed**. Realistic shapes outside the curriculum still bind the
  floor; decode M is the only legitimately-below-floor (synthetic) regime.
- **Gate E (overfit firewall):** `dyt` (held-out kernel) + the `test` shapes read **once** at freeze.

═══════════════════════════════════════════════════════════════════════════
## 9. DELIVERABLE (a milestone to BANK, then keep climbing — method §6.0)
═══════════════════════════════════════════════════════════════════════════
1. A fresh **`PointwiseElementwiseFact`** + its populator + a new **`TritonPointwiseSeedHeuristic`**
   (registered in `__init__.py`), proven **faithful & disjoint** (Gate D: divergence kernel fires
   correctly; the 3 negative-recognizer kernels' facts are byte-identical — the no-regression invariant).
2. **Correctness** for every corpus kernel at the measured bf16 tolerance.
3. **Perf, per shape:** headline **`seeded_vs_default`** (close the ~5–7× default gap on the GLU/add
   kernels) AND **`seeded_vs_tc` ≈ parity**; method bar `un-autotuned Helion ≥ 0.75 × min(tc,
   helion-max-autotune-oracle)` per realistic shape (here oracle ≈ tc, so the live target is
   **seeded ≈ oracle**, i.e. the size_hint-aware seed lands near oracle across BOTH small-M and large-M).
   The traffic-2 unary cases (relu_squared/bias_gelu/dyt) and the traffic-3 gated/add cases must each be
   captured by the SAME bytes-aware seed, not a per-kernel knob. Held-out **`dyt`** + **`test`** shapes
   read **once** at freeze.
4. **Phase 1 (swiglu+geglu) is the first milestone to bank; the run does NOT stop there** — continue
   into Phase 2 (residual_add, relu_squared, bias_gelu) until the whole family clears the bar, then into
   overtime (broaden the lever, simplify, push small-M).
5. A short **report** (`_lab/pointwise/REPORT.md`): per-kernel/per-N G vs default and vs tc; the
   size_hint/bytes lever the hill-climb selected (with the lowered-Triton mechanism for Gate F); the
   no-regression diff on the negative kernels; the small-M behavior.

Never `git push` / touch a PR ref (the human decides that). Never stop on your own (method §6.0). Begin
by reading `hillclimb-method.md`, then `local-setup-devserver.md`, then run the §7 3-shape test on the
new worktree off `reduction-4pr-stack`'s tip, then build Phase 1.
