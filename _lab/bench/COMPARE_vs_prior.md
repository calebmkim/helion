# Seeded Helion vs torch.compile-default — NOW vs previously-recorded

Metric: **G_seed = tc_default_latency / helion_seeded_latency** (>1 ⇒ seeded Helion
beats torch.compile default-mode). Identical definition in both datasets.

- **NOW** = this run, **test** split, fp32, contention-guarded (foreign_mib=0).
- **PRIOR** = run3, **train** split, from `_lab/logs/run3/t1_train_tcdefault.json`
  (T1: rms_norm/layer_norm/sum/cross_entropy/long_sum) and
  `t2_train_3way_summary.md` ratio_tc (T2: softmax/welford/kl_div/jsd).
- Splits differ (train vs test = different shapes), so compare **distributions/medians**,
  and regime-match where a delta appears.

| kernel | NOW G_seed (test med) | PRIOR G_seed (train med) | NOW seed_lift | verdict |
|---|---|---|---|---|
| rms_norm | 0.828 | 0.991 | 1.114 | shape-mix (see below) — CONSISTENT |
| layer_norm | 1.123 | 0.987 | 1.054 | now slightly better |
| sum | 1.000 | 1.021 | 1.049 | consistent |
| cross_entropy | 0.834 | 1.058 | 1.526 | shape-mix (V-heavy) — CONSISTENT |
| long_sum | 1.191 | 1.117 | 2.016 | consistent / better |
| softmax | 1.106 | 1.079 | 3.211 | consistent |
| welford | 1.362 | 0.963 | 2.586 | now better (BUT fp32 vs prior bf16*) |
| kl_div | 1.264 | 1.051 | 5.329 | consistent / better |
| jsd | 1.057 | 1.013 | 3.464 | consistent |

## The two apparent "drops" are shape-mix, not regressions

**cross_entropy** (NOW 0.834 vs PRIOR 1.058): G_seed falls as vocab V grows — present in
BOTH datasets. Regime-matched:
- V < 98k:  PRIOR median 1.084  vs  NOW 1.056  ✓
- V ≥ 98k:  PRIOR median 0.757  vs  NOW 0.718  ✓
The test split is V-heavy (4 of 7 shapes ≥114k), so its median sits in the wide-V regime
where seed trails tc. Fully consistent with prior.

**rms_norm** (NOW 0.828 vs PRIOR 0.991): a small-shape **noise-floor** effect, not a regression.
- The big shapes agree exactly: prior (4096,8192) G=0.997; my validation of (4096,8192)=0.997.
- The low NOW values are all at **tiny absolute latency** (tc 28–45 µs: (2048,4096) 28.6µs,
  (8192,1280) 33.8µs). Below ~50µs these kernels are launch/overhead-bound (memory: sub-0.1ms
  shapes carry ±12% timing noise), and the seed's extra warps cost fixed overhead tc avoids.
  The prior train split had only one such tiny shape ((8192,768) 22µs → G=0.967).
- **seed_lift stays ≥1.0 on every rms_norm shape** (1.05–1.26): the seed still beats the
  *unseeded* default everywhere; it just doesn't catch tc at sub-50µs shapes.

## Bottom line
Where regime-matched, the NOW seed-vs-tc numbers **reproduce the previously-recorded values**.
Net, the seeded arm is **as-good-or-better vs tc than before** on 7/9 kernels; the two lower
medians (rms_norm, cross_entropy) are explained entirely by the test split sampling more of
the small-latency / wide-vocab regimes the prior train split under-sampled — the prior data
shows the identical trend in those regimes. *welford is fp32 here vs the prior bf16 operator,
so its improvement is not directly comparable.
