# XSA benchmark results — B200, autotuned Helion vs eager SDPA + torch.compile

Hardware: NVIDIA B200 (183 GiB), bf16, B=4, H=32. Shapes were chosen to match the
defaults of `examples/blackwell_attention.py main()` (T=8192 at D=64 and D=128),
plus a smaller T=4096 D=128 shape for context.

Helion measured under full autotune (`benchmarks/run.py --kernel xsa --metrics
speedup,accuracy,latency`) with `HELION_AUTOTUNE_RANDOM_SEED=0` and a per-shape
budget of `HELION_AUTOTUNE_BUDGET_SECONDS=8400` (140 min). All three shapes
converged before the budget ran out. All three shapes pass accuracy.

Driver and raw artifacts: `xsa_autotune/20260521T060741Z/`.

## Latencies and speedups (lower is better; speedup vs `sdpa_xsa`)

| Shape | sdpa_xsa baseline (ms) | torch.compile (max-autotune) | Helion (autotuned) |
|---:|---:|---:|---:|
| T=4096 D=128 | 1.979 (±0.32%) | **1.96× → 1.010 (±1.13%)** | 1.02× → 1.946 (±4.06%) |
| T=8192 D=64  | 3.723 (±0.22%) | **1.47× → 2.537 (±0.97%)**  | 0.96× → 3.868 (±0.14%) |
| T=8192 D=128 | 5.329 (±1.15%) | **1.52× → 3.514 (±0.12%)**  | 0.70× → 7.637 (±0.15%) |

## Autotune cost

| Shape | Wall time | Configs tested | Compile fails | Outcome |
|---:|---:|---:|---:|---|
| T=4096 D=128 | 27 min | 588 | 171 | early-stopped at gen 10 (no improvement) |
| T=8192 D=64  | 20 min | 475 | 96  | converged |
| T=8192 D=128 | 45 min | 701 | 162 | converged |

Total wall time: ~1h 32m. Each shape was a separate `benchmarks/run.py`
invocation with `--seq-len <T> --num-inputs 1` (avoids the
`flash_attention.generate_inputs` Q-only doubling bug — see XSA_RESULTS.md).

## Best Helion configs

### T=4096 D=128 (best: 1.946 ms)

```python
@helion.kernel(
    config=helion.Config(
        atomic_indexing=[],
        block_sizes=[1, 128, 64],
        indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'],
        l2_groupings=[1],
        load_eviction_policies=['last', 'first', '', 'first'],
        loop_orders=[[0, 1]],
        num_stages=3,
        num_warps=4,
        pid_type='flat',
        range_flattens=[None, True],
        range_multi_buffers=[None, None],
        range_num_stages=[0, 3],
        range_unroll_factors=[0, 4],
        range_warp_specializes=[None, None],
    ),
    static_shapes=True,
)
```

### T=8192 D=64 (best: 3.868 ms)

```python
@helion.kernel(
    config=helion.Config(
        atomic_indexing=[],
        block_sizes=[1, 128, 128],
        indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'],
        l2_groupings=[32],
        load_eviction_policies=['', 'first', 'last', 'last'],
        loop_orders=[[1, 0]],
        num_sm_multiplier=32,
        num_stages=3,
        num_warps=4,
        pid_type='persistent_blocked',
        range_flattens=[False, True],
        range_multi_buffers=[True, None],
        range_num_stages=[3, 0],
        range_unroll_factors=[0, 0],
        range_warp_specializes=[False, None],
    ),
    static_shapes=True,
)
```

### T=8192 D=128 (best: 7.637 ms)

```python
@helion.kernel(
    config=helion.Config(
        atomic_indexing=[],
        block_sizes=[1, 128, 64],
        indexing=['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'],
        l2_groupings=[1],
        load_eviction_policies=['last', '', 'last', 'first'],
        loop_orders=[[0, 1]],
        num_stages=3,
        num_warps=4,
        pid_type='flat',
        range_flattens=[None, True],
        range_multi_buffers=[None, True],
        range_num_stages=[0, 0],
        range_unroll_factors=[0, 3],
        range_warp_specializes=[None, None],
    ),
    static_shapes=True,
)
```

## Notes / interpretation

- This is a **regression vs the H100 numbers in `XSA_RESULTS.md`**, where
  autotuned Helion beat torch.compile at T=512 (3.97× vs 2.86×) and stayed
  within 17% at T=2048. On B200, autotuned Helion is **slightly slower than
  even the unfused `sdpa_xsa` baseline** at every shape (0.70×–1.02×) and
  clearly behind torch.compile.
- The XSA "win" of fusing the epilogue into the attention output is bounded by
  the cost of attention itself. On B200, the SDPA backend is significantly
  faster relative to a hand-written attention loop than on H100 (cuDNN /
  FlashAttention is taking advantage of Blackwell-specific features), so the
  fixed XSA-epilogue savings are a smaller fraction of total runtime — and the
  generic Helion attention loop loses to it.
- The XSA kernel is essentially an H100-style attention. None of the
  Blackwell-specific tricks in `examples/blackwell_attention.py` are in there:
  no TMEM/MMAv5 lowering hint, no warp specialization
  (`_triton_config_maxRegAutoWS`), no `persistent_interleaved` scheduling, no
  subtiling, no vectorized F32 PTX. Two of three best configs landed on
  `pid_type='flat'`; only T=8192 D=64 picked `persistent_blocked`. None
  picked `persistent_interleaved`.
- Best configs across all three shapes are conservative: `num_warps=4`,
  `num_stages=3`, `block_sizes` capped at 128. The autotuner did try larger
  tiles and warp counts (171/96/162 compile failures across shapes is
  significant — many were OOM or invalid TMEM allocations), but converged on
  small ones.

## Possible follow-ups

- Port Blackwell-specific patterns from `examples/blackwell_attention.py` into
  the XSA kernel (TMEM, warp specialization, persistent interleaved scheduling,
  subtiling) and re-autotune.
- Compare to a hand-rolled "plain attention" (no XSA epilogue) Helion run on
  the same shapes to separate "Blackwell attention is hard for Helion" from
  "the XSA epilogue specifically hurts on B200".
- Try `--input-types CUSTOMIZED_SHAPES` from the `blackwell_attentions`
  TritonBench operator to compare across the same sweep that Blackwell
  attention is normally tuned against.
