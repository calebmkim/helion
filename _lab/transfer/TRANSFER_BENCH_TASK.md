# Transfer benchmark task — reduction seed heuristic (PR #2762) on UNSEEN kernels

You are benchmarking whether the **merged Helion reduction seed heuristic** (`pytorch/helion`
PR #2762, "Triton reduction seed heuristic (generalizable core)") gives good perf on **8 fused
reduction+pointwise / loss kernels it was NOT tuned on**. This is a *generality* test: does the
seed's perf carry over to real-world kernels.

You are **long-running**: the harness in this directory is a useful BASE, not a finished artifact.
Run it, sanity-check the numbers, and **fix / re-bench / improve it as needed** — don't trust a
surprising number, re-measure it.

## Files (this directory, `prompts-lab/transfer/`)
- `transfer_kernels.py` — 6 new fwd-only Helion kernels + pure-torch references.
- `shapes_transfer.py` — real-model shape curriculum (`SHAPES`) + a cheap `SMOKE` set.
- `ab_three_arm_transfer.py` — the harness. Modes:
  - `verify` — correctness gate only (no timing); also reports whether the seed FIRED + its config.
  - `bench`  — single-process 3-arm timing: `helion_default` (heuristics off), `helion_seeded`
    (the PR seed), `torch_compile` (default mode, NOT max-autotune). Reports `seeded_vs_default`
    (the floor that matters — Helion defaults are bad) and `seeded_vs_tc` (the goal), median-of-9.

The 8 kernels: `fused_add_rmsnorm, fused_add_layernorm, gated_rmsnorm, scaled_masked_softmax,
cross_entropy_ls_zloss, dynamic_quant` (these 6 from `transfer_kernels.py`) + `fused_linear_jsd`,
`grpo` (wired in from the helion checkout's `examples/`).

## Step 0 — discover this box (do NOT hardcode; the setup docs describe OLD machines)
1. Find a **helion checkout that has the MERGED reduction heuristic**. Verify:
   `PYTHONPATH=<helion> python -c "import helion; from helion._compiler.autotuner_heuristics.triton import TritonStandardReductionHeuristic, TritonUserTiledReductionHeuristic; print(helion.__file__)"`
   It must also have `examples/grpo_loss.py` and `examples/fused_linear_jsd.py`. If `import helion`
   resolves elsewhere, set `PYTHONPATH=<helion checkout>` (it shadows any editable install).
2. Find the python interpreter that has torch+triton+cuda (the conda/venv env — NOT system python).
3. `nvidia-smi`: pick ONE idle GPU, pin `CUDA_VISIBLE_DEVICES=<idx>`. Note the GPU + compute
   capability (`torch.cuda.get_device_capability()` → (9,0)=H100/sm90, (10,0)=B200/sm100) and the
   L2 size (`torch.cuda.get_device_properties(0).L2_cache_size`).
4. Run scripts from `cwd=/tmp` with `PYTHONPATH=<helion checkout>`.

## Step 1 — hardware gate (READ if this is a B200 / sm100 box)
The merged reduction seed **gates on sm90**. On a B200 (sm100) it falls back: the standard track
emits a conservative `_narrow_seed`, the user-tiled track (grpo) DECLINES entirely → `seed_fired`
will be False and `helion_seeded` ≈ `helion_default`. That is correct-as-shipped but uninformative.
- **On a B200, set `HELION_TRANSFER_FORCE_SEED=1`** so the sm90 seed fires on sm100, to MEASURE
  whether the H100-tuned constants transfer to B200. The constants (240 KiB SMEM cap, 132-SM
  occupancy limits) are H100-specific, so a suboptimal B200 config IS the finding (it would motivate
  a B200-specific reduction heuristic). Confirm `seed_fired: true` in the output.
- On an H100, do NOT set it (the seed fires natively).

## Step 2 — correctness FIRST
```
cd /tmp && CUDA_VISIBLE_DEVICES=<idx> [HELION_TRANSFER_FORCE_SEED=1] PYTHONPATH=<helion> \
  <python> <this-dir>/ab_three_arm_transfer.py verify all --dtypes bf16,fp32
```
Every (kernel, shape, dtype) must PASS and the seed must FIRE. If a kernel fails correctness, FIX it
(or its reference) before benching — never bench a wrong-output kernel.

## Step 3 — bench, ONE KERNEL PER PROCESS, both dtypes
Loop the 8 kernels, each in its own fresh process (avoids cross-kernel dynamo-guard buildup):
```
for k in fused_add_rmsnorm fused_add_layernorm gated_rmsnorm scaled_masked_softmax \
         cross_entropy_ls_zloss dynamic_quant fused_linear_jsd grpo ; do
  cd /tmp && CUDA_VISIBLE_DEVICES=<idx> [HELION_TRANSFER_FORCE_SEED=1] PYTHONPATH=<helion> \
    <python> <this-dir>/ab_three_arm_transfer.py bench $k --dtypes bf16,fp32 \
    > /tmp/transfer_${k}.json 2> /tmp/transfer_${k}.log
done
```
(fp16: skip the wide-vocab loss kernels — `cross_entropy_ls_zloss`, `fused_linear_jsd`, `grpo` —
they NaN-underflow at large V. fp16's lever decisions equal bf16's anyway, so it's optional.)

## Be AWARE of the benchmarking footguns (read `../method/hillclimb-method.md` §4)
The harness already does: forward-only, single-process same-tensors across arms, median-of-9
do_bench, accuracy gate before timing (acc-fails excluded from the geomean), tc=default-mode (not
max-autotune), dynamo-reset per shape, fp32 accumulators. **You still need to WATCH for:**
- **Cold-L2** (§4 #9): do_bench is cold-L2 only if this Triton build flushes L2 between reps. For
  small working sets (≤ this box's L2; e.g. the narrow-N rows), sanity-check the implied bandwidth is
  well under HBM peak — a 3–5 TB/s number means L2-hot (fake). If suspect, switch to a profiler
  cold-L2 metric or lift M.
- **Noise floor** (§4 #13): rows flagged `noisy_sub25us` swing ±25% — re-run, or don't draw
  conclusions from them.
- **Contention**: re-check `nvidia-smi` before a trusted timing (rule out co-tenants).
- **Config actually ran**: the row prints `seed_config`; confirm it's the seed you expect (and
  `seed_fired: true`).
This is *awareness*, not a checklist to perfect — fix what you see, re-bench what looks off.

## Step 4 — report
For each (kernel, dtype): the geomean `seeded_vs_default` and `seeded_vs_tc`, plus a per-shape table.
Headline = **does the seed beat the Helion default (it should, defaults are bad) and reach/beat
torch.compile** on these unseen kernels. Flag: any acc-fail, any `seed_fired: false`, any noisy/
L2-suspect rows. Save the raw JSON. Note the box (GPU, cc, L2, force-seed on/off) in the report.

You are NOT modifying the helion source or pushing anything — this is measurement only.
