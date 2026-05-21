# XSA benchmark results — full Helion autotune vs eager SDPA + torch.compile

Hardware: NVIDIA H100 80GB HBM3, bf16, B=4, H=48, D=128 (matches `flash_attention`'s
default sweep in `benchmarks/run.py`). Helion measured under full autotune
(`benchmarks/run.py --kernel xsa --metrics speedup,accuracy,latency`, no
`HELION_AUTOTUNE_EFFORT=none`). All three rows pass accuracy.

## Latencies and speedups (lower is better; speedup vs `sdpa_xsa`)

| T (=T_kv) | sdpa_xsa baseline (ms) | torch.compile (max-autotune) | Helion (autotuned) |
|---:|---:|---:|---:|
| 512  | 0.431 (±0.65%) | 2.86× → 0.151 (±1.70%) | **3.97× → 0.108 (±1.95%)** |
| 1024 | 0.875 (±0.32%) | **2.53× → 0.346 (±0.76%)** | 2.44× → 0.359 (±0.50%) |
| 2048 | 1.976 (±0.51%) | **2.05× → 0.964 (±0.61%)** | 1.70× → 1.163 (±1.61%) |

## Autotune cost

| T | Wall time | Configs tested | Compile fails | Accuracy fails | Generations |
|---:|---:|---:|---:|---:|---:|
| 512  | 33 min | 801 | 100 | 1 | 15 |
| 1024 | 50 min | 970 | 141 | 0 | 15 |
| 2048 | 60 min | 903 | 237 | 0 | 20 |

Total autotune wall time across the three shapes: ~2h 23m.

## Best Helion configs

### T=512 (best: 0.108 ms)

```python
@helion.kernel(
    config=helion.Config(
        atomic_indexing=[],
        block_sizes=[1, 64, 128],
        indexing=['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'],
        l2_groupings=[16],
        load_eviction_policies=['first', 'first', '', 'first'],
        loop_orders=[[0, 1]],
        num_stages=1,
        num_warps=4,
        pid_type='flat',
        range_flattens=[None, True],
        range_multi_buffers=[None, None],
        range_num_stages=[0, 1],
        range_unroll_factors=[0, 1],
        range_warp_specializes=[],
    ),
    static_shapes=True,
)
```

### T=1024 (best: 0.359 ms)

```python
@helion.kernel(
    config=helion.Config(
        atomic_indexing=[],
        block_sizes=[1, 128, 64],
        indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'],
        l2_groupings=[1],
        load_eviction_policies=['first', 'last', 'last', 'first'],
        loop_orders=[[1, 0]],
        num_stages=1,
        num_warps=4,
        pid_type='flat',
        range_flattens=[None, False],
        range_multi_buffers=[None, False],
        range_num_stages=[0, 1],
        range_unroll_factors=[0, 2],
        range_warp_specializes=[],
    ),
    static_shapes=True,
)
```

### T=2048 (best: 1.163 ms)

```python
@helion.kernel(
    config=helion.Config(
        atomic_indexing=[],
        block_sizes=[1, 256, 128],
        indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'],
        l2_groupings=[32],
        load_eviction_policies=['last', '', 'last', 'first'],
        loop_orders=[[1, 0]],
        num_stages=2,
        num_warps=8,
        pid_type='flat',
        range_flattens=[None, False],
        range_multi_buffers=[None, False],
        range_num_stages=[0, 3],
        range_unroll_factors=[0, 4],
        range_warp_specializes=[],
    ),
    static_shapes=True,
)
```

## Notes

- Each shape autotunes from scratch: tile shape, L2 grouping, TMA vs pointer
  indexing, loop order, and software pipelining all change with T. Tile shape
  grows (`[1,64,128]` → `[1,128,64]` → `[1,256,128]`) and pipelining deepens
  (`num_stages`, `range_num_stages`, `range_unroll_factors` all increase) as T
  scales up.
- Each shape was run as its own `benchmarks/run.py` invocation with explicit
  `--seq-len`, because flash_attention's `_get_standard_shapes` only doubles
  Q's seq_len (leaving K/V fixed), which breaks XSA's self-attention assert.
  For a single-invocation sweep, drop `--seq-len` and use `--num-inputs N` —
  the fall-through branch yields square shapes.
- `sdpa_xsa` is the unfused baseline: `F.scaled_dot_product_attention` (one
  cuDNN/FlashAttention launch) + a separate `_xsa_epilogue` (normalize + dot +
  sub). `torch_compile_sdpa_xsa` is `torch.compile(..., mode="max-autotune")`
  of the same function — Inductor compiles the epilogue but does not fuse
  into the SDPA call itself.
