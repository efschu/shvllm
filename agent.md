You are implementing a feature in a fork of vLLM: heterogeneous tensor parallelism,
where each GPU receives a proportional share of tensor-parallel work instead of
equal shares.

## Goal

Add a `--tensor-parallel-weights` CLI argument that accepts a comma-separated list
of floats (e.g. "2,1,1") representing the relative compute capacity of each TP rank.
These weights define how matrix columns (ColumnParallel) and rows (RowParallel) are
partitioned across GPUs instead of equal 1/N splits.

Example: `--tensor-parallel-size 3 --tensor-parallel-weights 2,1,1`
→ GPU0 gets 50% of columns/rows, GPU1 gets 25%, GPU2 gets 25%.

The number of weights must equal tensor-parallel-size. Weights are normalized
internally to sum to 1.0. Equal weights (e.g. "1,1,1") must reproduce current behavior
exactly.

---

## Constraints and NCCL compatibility

`ncclAllGather` requires all ranks to contribute equally sized chunks.
Since our chunks are now unequal, you MUST replace the AllGather in
`gather_from_tensor_model_parallel_region` with a pad-gather-slice approach:

1. Each rank pads its chunk to `max(partition_sizes)` along the gather dimension.
2. All ranks perform `torch.distributed.all_gather` on the padded tensors.
3. Each rank slices the real content back: `gathered[i][..., :partition_sizes[i]]`
4. Concatenate slices to reconstruct the full tensor.

`ncclAllReduce` (RowParallel) does NOT need changes — it sums partial results of
identical output shape regardless of input shard size.

---

## Files to modify

### 1. `vllm/config.py`

In `ParallelConfig`:
- Add field: `tensor_parallel_weights: Optional[List[float]] = None`
- Add property: `tp_partition_sizes(hidden_size: int) -> List[int]`
  that converts normalized weights to actual column/row counts, ensuring they sum
  exactly to hidden_size (assign remainder to rank 0).
- Validate in `__post_init__`: if weights given, len must equal tensor_parallel_size,
  all values must be > 0.

### 2. `vllm/engine/arg_utils.py`

In `EngineArgs`:
- Add field: `tensor_parallel_weights: Optional[str] = None`
  with argparse entry: `--tensor-parallel-weights`
  help: "Comma-separated relative weights per TP rank, e.g. '2,1,1'. 
         Must have exactly tensor-parallel-size entries."
- In `create_engine_config()`: parse the string to `List[float]` and pass to
  `ParallelConfig`.

### 3. `vllm/model_executor/layers/linear.py`

This is the core change.

#### `ColumnParallelLinear`
- Replace `self.output_size_per_partition = divide(output_size, tp_size)` with a
  per-rank partition size lookup.
- Add: `self.partition_sizes: List[int]` from `tp_partition_sizes(output_size)`
- Add: `self.output_size_per_partition: int` = `partition_sizes[get_tensor_model_parallel_rank()]`
- Weight loading in `weight_loader` must use per-rank offsets computed from
  `partition_sizes`, not uniform strides.

#### `RowParallelLinear`
- Same pattern: `self.input_size_per_partition` becomes rank-specific.
- Weight loading uses cumulative offsets from `partition_sizes`.

Both classes must store `partition_sizes` as a class attribute so the communication
layer can access it.

### 4. `vllm/distributed/utils.py`

In `gather_from_tensor_model_parallel_region`:

Replace:
```python
torch.distributed.all_gather(tensor_list, input_)
output = torch.cat(tensor_list, dim=-1)
```

With:
```python
tp_group = get_tensor_model_parallel_group()
world_size = get_tensor_model_parallel_world_size()

# Retrieve partition sizes from the linear layer context or parallel state
partition_sizes = get_tp_partition_sizes()  # see parallel_state.py below
max_size = max(partition_sizes)

# Pad local tensor to max partition size along last dim
pad_size = max_size - input_.shape[-1]
if pad_size > 0:
    padded = F.pad(input_, (0, pad_size))
else:
    padded = input_

# All-gather equal-sized padded chunks
padded_list = [torch.empty_like(padded) for _ in range(world_size)]
torch.distributed.all_gather(padded_list, padded, group=tp_group)

# Slice real content and concatenate
slices = [padded_list[i][..., :partition_sizes[i]] for i in range(world_size)]
output = torch.cat(slices, dim=-1)
```

### 5. `vllm/distributed/parallel_state.py`

- Add module-level variable: `_TP_PARTITION_SIZES: Optional[List[int]] = None`
- Add setter: `set_tp_partition_sizes(sizes: List[int])`
- Add getter: `get_tp_partition_sizes() -> List[int]`
  → falls back to `[hidden_size // tp_size] * tp_size` if not set (preserving
  current behavior for callers that never set weights).
- Call `set_tp_partition_sizes` during worker initialization after `ParallelConfig`
  is available.

---

## Weight loading correctness

The existing `weight_loader` functions in model files (e.g.
`vllm/model_executor/models/qwen2.py`) use slicing patterns like:
```python
param_data = param_data[rank * shard_size : (rank + 1) * shard_size]
```

These must be replaced with offset-based slicing using cumulative partition sizes:
```python
offset = sum(partition_sizes[:rank])
param_data = param_data[offset : offset + partition_sizes[rank]]
```

Audit all model files for `output_size_per_partition` and `input_size_per_partition`
references in weight loaders and update them.

---

## Testing

Implement a unit test in `tests/distributed/test_heterogeneous_tp.py`:

1. Mock a 3-rank TP group with weights [2, 1, 1] and hidden_size=16.
2. Assert partition_sizes == [8, 4, 4].
3. Assert that pad-gather-slice round-trips correctly: concatenate random tensors
   of sizes [8, 4, 4], scatter them, gather them, verify reconstruction is exact.
4. Assert that weights=[1,1,1] produces identical output to the current all_gather
   path (parity test).

---

## Non-goals (do not implement)

- Dynamic per-layer weight adjustment.
- Automatic weight inference from GPU specs.
- Changes to pipeline parallelism.
- Changes to the attention backend or KV cache layout.

---

## Definition of done

- `--tensor-parallel-weights 2,1,1` with `--tensor-parallel-size 3` runs without
  error on any vLLM-supported model.
- Output logits are numerically identical to a single-GPU reference run
  (within fp16 tolerance).
- `--tensor-parallel-weights 1,1,1` is bit-exact equivalent to the current code
  path (no regression).
- All existing TP tests pass.
