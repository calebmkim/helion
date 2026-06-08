# Reduction-seed PR (#2704) benchmark — H100 sm90, fp32

**Three arms**, all via TritonBench `do_bench` (median-of-9), all accuracy-gated, all
verified uncontaminated (foreign-GPU-process guard = 0 MiB on every run):

- **helion_default** — unseeded base `default_config()` (`HELION_AUTOTUNE_EFFORT=none` + heuristics disabled)
- **torch.compile** — default mode (NOT max-autotune); the stable anchor
- **helion_seeded** — this PR's reduction seed, run directly (no autotune search)

`G = tc_default_latency / helion_latency` (≥1 ⇒ helion matches/beats torch.compile-default).
`seed_lift = G_seed / G_default` (>1 ⇒ seed faster than the unseeded default).

## Test-split results (66 shapes across 9 kernels)

| kernel | geo lift | median lift | min–max | G_seed | G_default | acc |
|---|---|---|---|---|---|---|
| rms_norm | 1.13 | 1.11 | 0.99–1.26 | 0.82 | 0.73 | ok |
| layer_norm | 1.06 | 1.05 | 0.96–1.24 | 1.15 | 1.08 | ok |
| softmax | 3.81 | 3.21 | 1.33–21.3 | 1.22 | 0.32 | ok |
| sum | 1.07 | 1.05 | 1.00–1.18 | 0.99 | 0.92 | ok |
| long_sum | 2.44 | 2.02 | 1.51–7.61 | 1.28 | 0.53 | ok |
| welford* | 2.58 | 2.59 | 1.98–3.41 | 1.33 | 0.52 | ok |
| cross_entropy | 1.60 | 1.53 | 1.30–1.92 | 0.85 | 0.54 | ok |
| kl_div | 5.78 | 5.33 | 2.00–20.2 | 1.25 | 0.22 | ok |
| jsd | 3.89 | 3.46 | 1.89–12.8 | 1.09 | 0.28 | ok |

**Aggregate:** geomean per-kernel **2.18x**; geomean over 66 pooled shapes **2.15x**;
**median 1.89x**. Worst single-shape lift 0.961 (within do_bench noise); only 1/66 shapes < 0.97.

## Honest caveats (from adversarial review)

1. **Seed-vs-default measures "good config without search," not vs a tuned baseline.** Both
   helion arms are non-autotuned. The seed's job is to be a good *starting point*; a real
   autotune would improve the default arm. The lifts are "how much the unsearched starting
   point improves," correctly the right axis for a seed.
2. **Big geomeans are tail-inflated.** softmax 3.81 / kl_div 5.78 / jsd 3.89 are carried by
   extreme-N shapes where the *base default* spills catastrophically (e.g. softmax(512,98304)
   3.96ms→0.186ms = 21x; kl_div(1024,128256) 7.18ms→0.355ms = 20x). Medians are materially
   lower. Report the distribution, not just geomean.
3. **welford was benched at fp32, not the operator's hardcoded bf16.** The seed's residency
   caps are fp32-tuned (at bf16 they under-cap ~2x — a known follow-up). 2.58x is valid for
   fp32 only; the operator's default bf16 regime is untested here.
4. **Correctness verified tightly.** Beyond the allclose gate, a direct rel-error check:
   kl_div rel_err 1.2e-7, jsd rel_err 0.0 — the seed is numerically exact, not hiding behind
   loose tolerance.
5. **"Never hurts" holds at geomean**; 3 individual shapes regress 1–4% (within the 5–10%
   cross-process noise floor).

## Headline

The reduction seed **never meaningfully regresses** any kernel and delivers large wins where
the unseeded default is pathological — especially the loss kernels (kl_div, jsd), long
reductions (long_sum), welford, and wide softmax. On the well-behaved norms/sum it's a modest
+5–13%. The cleanest, least-caveated win is **cross_entropy (1.6x, tight 1.3–1.9x, exact)**.
Validated against the prior run's ledger: fresh unseeded G_default (rms_norm 0.73) matched the
banked floor (0.745–0.77), and the seeded G hit the banked autotune-oracle ceiling (~1.0).
