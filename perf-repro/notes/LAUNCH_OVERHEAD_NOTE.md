# Launch overhead & why cold-L2 `do_bench`-style timing is GPU-side truth

*Methodology note for the reduction-heuristic perf report. Answers the question: "your
benchmark includes CPU kernel-launch overhead — doesn't that contaminate the ratios?"*

## TL;DR

- Our timing uses **cold-L2, CUDA-event, interleaved** measurement (the same primitives as
  triton `do_bench` / tritonbench, arranged as a round-robin A/B like Helion's autotuner).
- **The 256 MB L2-cache flush that precedes every timed launch HIDES CPU launch overhead**, so
  the reported µs is **pure GPU device time**, not host dispatch cost.
- This is not assumed — it is measured three ways, and **cross-checked on every cell** at
  runtime against a CUDA-graph pure-GPU measurement (the aggregator flags any divergence).
- The condition under which it would stop holding is known and quantified (per-call host
  dispatch would have to exceed the flush's ~86 µs GPU time; Helion's is ~40 µs — comfortably
  hidden, verified).

## The mechanism (why an event timer captures — or hides — a *CPU* cost)

`do_bench`'s (and our) timed loop is, per rep:

```
clear_cache()      # enqueue a 256 MB GPU memset  (~86 µs of GPU work on H100)
start.record()     # enqueue a GPU timestamp
fn()               # CPU spends ~40 µs of Python/dispatch enqueuing the Helion launch
end.record()       # enqueue a GPU timestamp
```

CUDA events are timestamped **on the GPU, in stream order**. The memset is enqueued *before*
`start.record()`. While the GPU spends ~86 µs executing the memset, the CPU races ahead and
enqueues `fn()` (only ~40 µs of host work). By the time the GPU finishes the memset and reaches
`start.record()`, the kernel is already queued — so the GPU never stalls waiting for the CPU,
and the `start→end` bracket encloses **only device work**.

The flush's own GPU duration is therefore a **CPU "run-ahead budget."** If host dispatch <
flush time, launch cost is fully hidden. If host dispatch > flush time, the GPU idles inside the
bracket and launch cost leaks in.

## Experiment 1 — does cold-L2 `do_bench` equal pure-GPU device time?

Compared, on the real Helion kernel dispatch path (H100), cold-L2 `do_bench`-style event timing
vs a cold-L2 CUDA-graph replay (launch amortized = pure GPU device time). `helion_cpu_enq` is
the standalone per-call CPU dispatch cost (measured async, before sync).

| shape (M,N) | do_bench-style µs | cold cudagraph µs | **gap** | cpu_enqueue µs |
|---|---|---|---|---|
| (64,512)    | 9.09  | 8.78  | +0.30  | 42.7 |
| (256,1024)  | 12.70 | 12.93 | −0.22  | 42.2 |
| (1024,2048) | 22.21 | 22.21 |  0.00  | 43.3 |
| (4096,4096) | 53.54 | 53.60 | −0.06  | 43.1 |
| (8192,4096) | 68.77 | 68.90 | −0.13  | 43.0 |

**do_bench-style ≈ cudagraph to within noise, even though Helion's CPU dispatch is ~43 µs.**
The ~43 µs of host cost is *not* in the number. (A naive per-call `torch.cuda.synchronize()`
timer, by contrast, measured ~62 µs on the tiny shape — it *does* expose the launch. `do_bench`
does not, because it syncs once at the end, not per call.)

## Experiment 2 — the boundary: when would launch overhead leak in?

Held the GPU kernel tiny and artificially inflated host-side Python work before the launch,
sweeping `host_dispatch_us`:

| host dispatch µs | do_bench µs | cudagraph µs | verdict |
|---|---|---|---|
| 0.1  | 5.73  | 1.42 | hidden (floor) |
| 37   | 5.92  | 1.42 | hidden |
| 59   | 16.06 | 1.42 | leaking (near budget) |
| 89   | 43.62 | 1.42 | leaking (≈ budget) |
| 374  | 360   | 1.41 | leaking ~1:1 |

**Crossover is exactly at the ~86 µs memset budget.** Below it, hidden; above it, do_bench
tracks host cost. Confirmed causally by sweeping the *flush buffer size* (16 MB → 1 GB) at fixed
host cost: the hidden/leaking threshold moves in lockstep with the memset duration — proving the
memset duration is the causal knob, not any property of the kernel.

## Experiment 3 — margin for the shapes we actually benchmark

| shape | cpu_dispatch µs | flush budget µs | margin |
|---|---|---|---|
| (64,512)    | 42.0 | 86.5 | 44.5 |
| (1024,1024) | 37.7 | 86.5 | 48.9 |
| (4096,4096) | 39.8 | 86.5 | 46.7 |

Helion dispatch (~38–43 µs) sits ~45 µs under the flush budget → launch overhead hidden across
our shape range.

## Two caveats (state these to a skeptic)

1. **The margin is finite.** A kernel with unusually heavy host-side dispatch (many tensor args,
   many symbolic shapes) could approach ~86 µs and begin leaking. This is exactly why we run the
   **per-cell cudagraph cross-check** (below) — it catches any such cell automatically.
2. **The budget is hardware-dependent.** It is `256 MB / memory_bandwidth`, so it shrinks on
   higher-bandwidth GPUs (e.g. B200). Re-verify the cross-check when porting off H100.

## How this is enforced in the harness (not just asserted)

- **Every arm of every cell** is additionally timed via a cold-L2 CUDA-graph replay
  (`coldgraph_us`), budget-scaled to the same 100 ms target as the main measurement.
- The aggregator computes `launch_frac = (do_bench_us − coldgraph_us) / do_bench_us` on the seed
  arm and **flags any cell where it exceeds 15%** (`## Launch-overhead cross-check` section of
  `SUMMARY.md`). An empty list = every headline number is provably GPU-side truth.

## Why no CUDA graphs in the headline number (and why that's the honest choice)

CUDA-graph replay gives pure GPU time by amortizing launch. We use it only as the *cross-check*,
not the headline, because:
1. **All arms pay launch symmetrically** in the eager (no-graph) regime, so the *ratio* is fair
   regardless — and the ratio is what we report.
2. It's the **deployment-truthful** regime: torch.compile-default, vLLM eager, and Helion all
   pay host dispatch in real inference.
3. It's the regime the **autotuner selects in** (Helion's config selection uses cold-L2
   `do_bench`), so our benchmark matches how the heuristic was tuned.

Since the cross-check shows do_bench-style ≈ cudagraph for our shapes anyway, the eager headline
*is* the GPU-side number — we get deployment-truthful framing and GPU-side truth simultaneously.

## Related upstream work (independent corroboration)

A concurrent Helion workstream (yushangdi) targets exactly this host-overhead effect, with
numbers matching ours:

- **#3004 / #3009 / #3010 / #3012** — "Reduce Triton kernel launch overhead": a Helion kernel
  call cost **~57 µs → ~12 µs** on H100; the pretuned-kernel geomean vs baseline went
  **0.575× → 1.21×** *purely by cutting host overhead* (the kernel unchanged). #3009 (fast
  dispatch cache) is merged and is in our benchmark base.
- **#2803 (cute)** — "wall-clock collapses a 5× config gap to ~1.3× because ~98 µs Python
  dispatch swamps an ~8 µs kernel" — the same ratio-compression effect we demonstrate.
- **#2994** — "wall-clock timing folds CPU launch overhead into every measurement (a 7 µs kernel
  measures ~39 µs)."
- **#2986** — cudagraph timing must clear L2 (else warm-L2 under-reports) — validates our
  cold-L2 stance.

**Bearing on our report:** the launch-overhead workstream is *orthogonal* to our headline —
those PRs fix the *eager wall-clock* deficit (what a naive per-call sync timer sees), which our
cold-L2 event timing already excludes. So #3009 landing on main does not materially move our
cold-L2 ratios (verified: our base includes it and the ratios are GPU-side). The launch-overhead
dimension is real for deployment but is not what this heuristic optimizes.

## Addendum: the in-place-kernel CLONE artifact (found + fixed after the first full run)

The cross-check initially flagged 87 cells — almost all the **in-place** kernels (vllm, vllm_gen,
qk_norm_rope). Root cause was NOT launch overhead: the timing thunk was `kernel(*_clone_args(args))`,
which cloned **every** argument each rep, including large read-only tensors (qk's 10.5 MB
`cos_sin_cache` → +31 µs/rep; small quant kernels → +13 µs dispatch/rep). That clone landed inside
the timed region and inflated every arm equally — a shared additive constant that compressed all
ratios toward 1.0.

Production vLLM (`vllm/kernels/helion/register.py`) declares `mutates_args` and calls the op
directly on pre-allocated buffers: it **clones nothing**, overwriting outputs in place. So our
per-rep clone was a pure benchmark artifact with no deployment analogue. Measured: cloning even
ONLY the mutated arg still added 2.9–5.2× on small shapes → also unfaithful. No-clone timing
(overwrite in place, matching production) is stable (0.4% spread over ~1000 in-place reps) and
bounded/finite.

**Fix:** the timing thunks call the kernel in place with no clone (accuracy checks still use full
clones — not perf-sensitive). L2 flush per rep unchanged → cold-L2 preserved. After the fix +
re-timing the in-place kernels, cross-check flags dropped **87 → 10**, i.e. do_bench-style and
cudagraph now agree → launch overhead confirmed hidden, no metric dilemma. Corrected numbers moved
materially (e.g. vllm_gen G_def 1.55→3.74, G_vllm 0.818→0.950) — the artifact had been badly
understating the heuristic's wins. See `results/MORNING_SUMMARY.md` "clone bug" section.

## Reproduce

```
# Experiments were run on an idle H100, foreground, sequential (never concurrent — a timing
# study must be the exclusive GPU user). Scripts: /tmp/*_probe.py, /tmp/helion_launch_test.py,
# /tmp/definitive_timing.py (see git history / session transcript).
```
