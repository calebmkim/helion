# Lab Notebook — Reduction Autotuner Heuristics

> The DURABLE source of truth for the hill-climb. A fresh worker reads this to continue losslessly.
> Maintained by the worker (decisions + empirical why; tried-and-rejected + why; open hypotheses;
> champion). The hub appends gate verdicts. Keep it current at every clean iteration boundary.

## Champion (current best heuristic)
- _none yet — pre-Step-2._

## Objective
- Product A: maximize `O = geomean_k G_k`, `G_k = geomean over kernel k's in-sample shapes of
  (tc_default_latency / seed_latency)`. Accept iff O improves AND gates pass (correctness; seed used;
  no active kernel's referee-confirmed G_k regresses >10% vs champion).
- Product B (every 5 iters): seeded vs unseeded quick-autotune convergence curve.

## Active kernels (curriculum)
- Start: rms_norm (fwd). Widen to: sum, layer_norm-fwd, softmax, long_sum (Band A); kl_div, jsd
  (Band B); welford (Band C). Forward only for now; defer backward (Band D).

## Track classification (T1 rolled / T2 manual / out-of-scope) — per kernel
- _to be filled by kernel-classifier per kernel._

## ReductionFact design
- _to be designed in Step 2._

## Heuristic decisions (with empirical why)
- _none yet._

## Tried and rejected (with why it failed)
- _none yet._

## Open hypotheses
- _none yet._

## Oracle cache pointers
- See `_lab/ledger.json` `oracle_cache`.
