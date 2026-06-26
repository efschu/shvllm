# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from abc import abstractmethod

import torch
from torch.nn.parameter import Parameter, UninitializedParameter

import vllm.envs as envs

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_partition_sizes_or_uniform,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.batch_invariant import (
    linear_batch_invariant,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.utils import (
    dispatch_unquantized_gemm,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
)
from vllm.logger import init_logger
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)

WEIGHT_LOADER_V2_SUPPORTED = [
    "UnquantizedLinearMethod",
    "CompressedTensorsLinearMethod",
    "CompressedTensorsLinearTransformMethod",
    "AWQMarlinLinearMethod",
    "AWQLinearMethod",
    "AutoGPTQLinearMethod",
    "Fp8LinearMethod",
    "FBGEMMFp8LinearMethod",
    "ModelOptFp8LinearMethod",
    "ModelOptFp8PcPtLinearMethod",
    "ModelOptFp8PbWoLinearMethod",
    "QuarkLinearMethod",
    "ModelOptNvFp4LinearMethod",
    "ModelOptNvFp4W4A16LinearMethod",
    "HummingLinearMethod",
]


def register_weight_loader_v2_supported_method(cls):
    """Decorator to register a LinearMethod as supporting weight_loader_v2."""
    WEIGHT_LOADER_V2_SUPPORTED.append(cls.__name__)
    return cls


def adjust_marlin_shard(
    param: Parameter,
    shard_size: int,
    shard_offset: int,
) -> tuple[int, int]:
    marlin_tile_size: int | None = getattr(param, "marlin_tile_size", None)
    if marlin_tile_size is None:
        return shard_size, shard_offset

    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


def adjust_block_scale_shard(
    weight_block_size: tuple[int, ...] | None,
    shard_size: int,
    shard_offset: int,
) -> tuple[int, int]:
    assert weight_block_size is not None
    block_n = weight_block_size[0]
    shard_offset = (shard_offset + block_n - 1) // block_n
    shard_size = (shard_size + block_n - 1) // block_n
    return shard_size, shard_offset


def adjust_bitsandbytes_4bit_shard(
    param: Parameter,
    shard_offsets: dict[str, tuple[int, int]],
    loaded_shard_id: str,
) -> tuple[int, int]:
    """Adjust the quantization offsets and sizes for BitsAndBytes sharding."""

    total, _ = shard_offsets["total"]
    orig_offset, orig_size = shard_offsets[loaded_shard_id]

    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total

    return quantized_size, quantized_offset


def adjust_scalar_to_fused_array(
    param_data: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: int | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For fused modules (QKV and MLP) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to
    one of the shards on disk. Here, we slice the param based on
    the shard_id for loading.
    """
    qkv_idxs = {"q": 0, "k": 1, "v": 2}

    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]

    return param_data[shard_id], loaded_weight


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError


class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # This method creates unquantized linear weights.
        # The weights are not quantized, and they are not sharded.
        # The amount of memory allocated for the weights is
        weight_loader = extra_weight_attrs.pop("weight_loader")
        # ``tp_partition_sizes_output``/``tp_partition_sizes_input`` drive
        # the parameter's heterogeneous-TP slicing. Under uniform TP they
        # collapse to the legacy ``tp_rank * shard_size`` formula.
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        if hasattr(layer, "partition_sizes"):
            weight.tp_partition_sizes_output = layer.partition_sizes
            # For column-parallel layers the input is replicated; for
            # row-parallel the input is partitioned. The linear layer
            # exposes ``partition_sizes`` for the *partitioned* dim, so we
            # also forward it as the input dim when the input dim shape
            # matches the partition sum (i.e. it really is row-parallel).
            if layer.input_size_per_partition != layer.input_size:
                weight.tp_partition_sizes_input = layer.partition_sizes

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if current_platform.is_cpu():
            from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm

            dispatch_cpu_unquantized_gemm(layer, remove_weight=True)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
            return linear_batch_invariant(x, layer.weight, bias)
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)


class LinearBase(PluggableLayer):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: Prefix for parameter names.
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, tensor parallelism will be disabled for this layer.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.has_bias = bias
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.quant_config = quant_config
        self.prefix = prefix
        self.allow_fp8_block_shape_mismatch = False
        self.quant_method: QuantizeMethodBase
        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        elif quant_method := quant_config.get_quant_method(self, prefix=prefix):
            self.quant_method = quant_method
        else:
            raise ValueError("All linear layers should support quant method.")
        self.return_bias = return_bias
        self.disable_tp = disable_tp
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1

    def update_param_tp_status(self):
        for param in self.parameters():
            if isinstance(param, BasevLLMParameter):
                param.tp_rank = self.tp_rank
                param.tp_size = self.tp_size


# --8<-- [start:replicated_linear]
@PluggableLayer.register("replicated_linear")
class ReplicatedLinear(LinearBase):
    """Replicated linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: Take no effect for replicated linear layers.
    """

    # --8<-- [end:replicated_linear]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # If MergedReplicatedLinear, use output size of each partition.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = self.output_sizes
        else:
            self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.quant_method.create_weights(
            self,
            self.input_size,
            self.output_partition_sizes,
            self.input_size,
            self.output_size,
            self.params_dtype,
            weight_loader=self.weight_loader,
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size, dtype=self.params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        # If the weight on disk does not have a shape, give it one
        # (such scales for AutoFp8).
        # Special case for GGUF

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param.size() == loaded_weight.size(), (
            f"Tried to load weights of size {loaded_weight.size()}"
            f"to a parameter of size {param.size()}"
        )
        param.data.copy_(loaded_weight)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None

        output = self.quant_method.apply(self, x, bias)

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        return s


# --8<-- [start:column_parallel_linear]
@PluggableLayer.register("column_parallel_linear")
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
    """

    # --8<-- [end:column_parallel_linear]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # Divide the weight matrix along the last dimension.
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.input_size_per_partition = input_size
        # Per-rank output partition sizes for heterogeneous TP: ``self.partition_sizes[i]``
        # is the number of output columns assigned to TP rank ``i``. ``self.output_partition_sizes``
        # is the *rank-local* view (what this rank sees); both views are needed by the
        # weight loaders (cumulative offset) and the parameter creation (rank-local shape).
        if disable_tp or self.tp_size == 1:
            self.partition_sizes = [output_size]
            self.output_size_per_partition = output_size
        else:
            self.partition_sizes = get_tp_partition_sizes_or_uniform(output_size)
            self.output_size_per_partition = self.partition_sizes[self.tp_rank]
        self.output_partition_sizes = [self.output_size_per_partition]
        # If QKV or MergedColumn, partition each segment independently and stack the
        # rank-local slices into ``output_partition_sizes`` for ``create_weights``.
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                get_tp_partition_sizes_or_uniform(s)[self.tp_rank]
                for s in self.output_sizes
            ]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self._maybe_allow_fp8_block_shape_mismatch()
        self.gather_output = gather_output

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )

        if bias:
            self.bias = Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def _maybe_allow_fp8_block_shape_mismatch(self) -> None:
        quant_config = getattr(self, "quant_config", None)
        weight_block = getattr(quant_config, "weight_block_size", None)
        if (
            weight_block is None
            or len(weight_block) < 1
            or len(self.output_partition_sizes) <= 1
        ):
            return

        try:
            block_n = int(weight_block[0])
        except (ValueError, TypeError):
            return

        if block_n <= 0:
            return

        if any(size % block_n != 0 for size in self.output_partition_sizes):
            self.allow_fp8_block_shape_mismatch = True
            logger.debug(
                "Allowing FP8 block shape mismatch for %s (block_n=%d, partitions=%s)",
                getattr(self, "prefix", "<unknown>"),
                block_n,
                self.output_partition_sizes,
            )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        output_dim = getattr(param, "output_dim", None)

        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if output_dim is not None:
                # Use the rank-local slice (heterogeneous TP may have a
                # different per-rank size than ``loaded_weight.size // tp_size``).
                final_shape[output_dim] = self.output_size_per_partition
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        param_data = param.data
        if output_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[output_dim]
            # Use the cumulative offset across per-rank partition sizes; for
            # the uniform case this collapses to ``tp_rank * shard_size``.
            start_idx = sum(self.partition_sizes[: self.tp_rank])
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)
        param.load_column_parallel_weight(loaded_weight=loaded_weight)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        output_parallel = self.quant_method.apply(self, input_, bias)

        if self.gather_output and self.tp_size > 1:
            # All-gather across the partitions. Pass the per-rank partition
            # sizes so the heterogeneous-TP pad-gather-slice path runs when
            # the output dim is itself heterogeneously partitioned (e.g.
            # LM head over vocab).
            output = tensor_model_parallel_all_gather(
            )
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size}"
        s += f", output_features={self.output_size_per_partition}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", gather_output={self.gather_output}"
        return s


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, all weights matrix won't be sharded, this layer
                    will be treated as a "Replicated" MergedLinear.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        self.output_sizes = output_sizes
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0

        # Note: no uniform-divisibility assertion here. With heterogeneous TP
        # (--tensor-parallel-weights) each segment is partitioned by a
        # per-rank weight list and individual segment sizes need not be
        # divisible by tp_size; rank-local slices are constrained to be
        # integers by the partition_sizes logic in ParallelConfig.
        super().__init__(
            input_size=input_size,
            output_size=sum(output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def validate_shard_id(self, loaded_shard_id: int | tuple[int, ...] | None):
        if loaded_shard_id is None:
            return
        if isinstance(loaded_shard_id, tuple):
            for idx in loaded_shard_id:
                if not (0 <= idx < len(self.output_sizes)):
                    raise ValueError(
                        f"Shard id index {idx} should be between 0 and "
                        f"{len(self.output_sizes) - 1}. Got shard id {loaded_shard_id}."
                    )
            if len(loaded_shard_id) > 1 and any(
                b - a != 1 for a, b in zip(loaded_shard_id[:-1], loaded_shard_id[1:])
            ):
                raise ValueError(
                    "Shard id with multiple indices should be consecutive. "
                    f"Got shard id {loaded_shard_id}."
                )
            return
        elif isinstance(loaded_shard_id, int):
            if loaded_shard_id < 0 or loaded_shard_id >= len(self.output_sizes):
                raise ValueError(
                    f"Shard id should be between 0 and {len(self.output_sizes) - 1}. "
                    f"Got shard id {loaded_shard_id}."
                )
            return
        raise ValueError("This line should not be reached")

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        self.validate_shard_id(loaded_shard_id)
        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if isinstance(loaded_shard_id, tuple) and (
            is_gguf_weight or is_gguf_weight_type
        ):
            raise NotImplementedError(
                "Shard id with multiple indices is not supported for GGUF."
            )
        if is_gguf_weight_type:
            if loaded_shard_id is not None:
                param.data[loaded_shard_id].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            else:
                param.shard_weight_type = {
                    i: loaded_weight.item() for i, _ in enumerate(self.output_sizes)
                }
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            if loaded_shard_id is not None:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            # Loaded weight is already fused on disk (mlp).
            # (e.g., Phi-3's gate_up_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return

            output_sizes = (
                self.output_sizes[loaded_shard_id[0] : loaded_shard_id[-1] + 1]
                if loaded_shard_id is not None
                else self.output_sizes
            )
            current_shard_offset = 0
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            if (
                use_bitsandbytes_4bit
                and isinstance(loaded_shard_id, tuple)
                and self.tp_size > 1
            ):
                raise NotImplementedError(
                    "Shard id with multiple indices is not supported "
                    "for BNB quantization with TP yet."
                )
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                # Add check to adjust the size/offset for FP8 block scales
                if isinstance(param, BlockQuantScaleParameter):
                    weight_block_size = getattr(self, "weight_block_size", None)
                    shard_size, shard_offset = adjust_block_scale_shard(
                        weight_block_size, shard_size, shard_offset
                    )

                if packed_dim == output_dim:
                    shard_size = shard_size // param.packed_factor
                    shard_offset = shard_offset // param.packed_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:
                    index = list(itertools.accumulate([0] + self.output_sizes))
                    orig_offsets = {
                        str(i): (index[i], size)
                        for i, size in enumerate(self.output_sizes)
                    }
                    orig_offsets["total"] = (self.output_size, 0)
                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_offsets, str(shard_id)
                    )

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        if output_dim is not None:
            # Heterogeneous TP path: each segment is partitioned independently
            # using its own per-rank partition sizes (a single shared weight
            # list applied proportionally to each segment). For the uniform
            # case this collapses to ``sum(self.output_sizes[:i]) // tp_size``
            # and ``self.output_sizes[i] // tp_size`` (matching the legacy
            # behaviour bit-exactly).
            seg_part_sizes = (
                [self.output_sizes[loaded_shard_id]]
                if self.tp_size == 1
                else get_tp_partition_sizes_or_uniform(
                    self.output_sizes[loaded_shard_id]
                )
            )
            shard_size = seg_part_sizes[self.tp_rank]
            # Offset within the rank-local param_data: concatenate the rank's
            # slices of all preceding segments.
            shard_offset = sum(
                get_tp_partition_sizes_or_uniform(self.output_sizes[j])[self.tp_rank]
                for j in range(loaded_shard_id)
            )

            if isinstance(param, BlockQuantScaleParameter):
                weight_block_size = getattr(self, "weight_block_size", None)
                shard_size, shard_offset = adjust_block_scale_shard(
                    weight_block_size, shard_size, shard_offset
                )

            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = round(shard_size // param.packed_factor)
                shard_offset = round(shard_offset // param.packed_factor)
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow
            is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

            if use_bitsandbytes_4bit:
                index = list(itertools.accumulate([0] + self.output_sizes))
                orig_offsets = {
                    str(i): (index[i], size) for i, size in enumerate(self.output_sizes)
                }
                orig_offsets["total"] = (self.output_size, 0)
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_offsets, str(loaded_shard_id)
                )
            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            # Offset within the *full* loaded weight: sum of preceding
            # segments (in full-size units) + this rank's offset within the
            # current segment (heterogeneous TP: ``sum(seg_part_sizes[:rank])``,
            # collapses to ``tp_rank * shard_size`` when uniform).
            start_idx = sum(self.output_sizes[:loaded_shard_id]) + sum(
                seg_part_sizes[: self.tp_rank]
            )
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def _load_fused_module_from_checkpoint(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        output_sizes: list[int] | None = None,
    ):
        """
        Handle special case for models where MLP layers are already
        fused on disk. In this case, we have no shard id. This function
        determines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """

        current_shard_offset = 0
        shard_offsets: list[tuple[int, int, int]] = []
        output_sizes = output_sizes or self.output_sizes
        for i, output_size in enumerate(output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ):
        self.validate_shard_id(loaded_shard_id)
        if loaded_shard_id is None or isinstance(loaded_shard_id, tuple):
            if isinstance(param, PerTensorScaleParameter):
                if isinstance(loaded_shard_id, tuple):
                    for idx in loaded_shard_id:
                        param.load_merged_column_weight(
                            loaded_weight=loaded_weight, shard_id=idx
                        )
                else:
                    # When weights are already fused on disk (e.g. Phi-3's
                    # gate_up_proj), there is only a single scale for the
                    # entire fused matrix. Fill all slots with this scale
                    # to ensure that any subsequent reduction (like .max())
                    # works correctly while preserving the parameter shape.
                    for idx in range(param.data.shape[0]):
                        param.load_merged_column_weight(
                            loaded_weight=loaded_weight, shard_id=idx
                        )
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_merged_column_weight(loaded_weight=loaded_weight)
                return
            output_sizes = (
                [self.output_sizes[idx] for idx in loaded_shard_id]
                if loaded_shard_id
                else None
            )
            if isinstance(param, BlockQuantScaleParameter):
                weight_block_size = getattr(self, "weight_block_size", None)
                output_sizes = [
                    adjust_block_scale_shard(weight_block_size, size, 0)[0]
                    for size in (output_sizes or self.output_sizes)
                ]
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(
                param, loaded_weight, output_sizes=output_sizes
            )
            return

        assert loaded_shard_id < len(self.output_sizes)

        # Heterogeneous TP per-segment partition sizes (see ``weight_loader``
        # for the rationale; uniform collapses to legacy ``//= tp_size``).
        seg_part_sizes = (
            [self.output_sizes[loaded_shard_id]]
            if self.tp_size == 1
            else get_tp_partition_sizes_or_uniform(self.output_sizes[loaded_shard_id])
        )
        shard_size = seg_part_sizes[self.tp_rank]
        # Offset within the rank-local param_data (concatenation of per-rank
        # slices of all preceding segments).
        shard_offset = sum(
            get_tp_partition_sizes_or_uniform(self.output_sizes[j])[self.tp_rank]
            for j in range(loaded_shard_id)
        )

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        # Heterogeneous TP: ``loaded_weight_offset`` is the offset within the
        # *full* loaded_weight for this rank's slice of this segment
        # (sum of previous segments in full-size units, plus the cumulative
        # sum of previous ranks in this segment's partition sizes).
        loaded_weight_offset = sum(self.output_sizes[:loaded_shard_id]) + sum(
            seg_part_sizes[: self.tp_rank]
        )
        param.load_merged_column_weight(
            loaded_weight=loaded_weight,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            loaded_weight_offset=loaded_weight_offset,
            tp_rank=self.tp_rank,
        )


class QKVParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's QKV transformation.

    Linear layers for the linear transformation of the query, key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.num_heads = divide(self.total_num_heads, tp_size)
        if tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        # Heterogeneous TP note: ``num_heads``/``num_kv_heads`` above stay
        # uniform; the actual per-rank head count is derived from
        # ``partition_sizes`` when --tensor-parallel-weights is in effect
        # (see ``num_heads_per_rank`` below). The QKV weight matrix itself is
        # partitioned by column size, so the parallel math is consistent as
        # long as the chosen weights yield an integer number of heads per rank
        # for both Q and KV (otherwise an explicit error is raised).
        input_size = self.hidden_size
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.v_head_size * tp_size,  # v_proj
        ]

        # Per-rank head counts under heterogeneous TP. When weights are unset
        # (uniform TP) this collapses to ``num_heads // tp_size`` per rank.
        if disable_tp or tp_size == 1:
            self.num_heads_per_rank = [self.num_heads]
            self.num_kv_heads_per_rank = [self.num_kv_heads]
        else:
            q_full = self.total_num_heads * self.head_size
            kv_full = self.total_num_kv_heads * self.head_size
            v_full = self.total_num_kv_heads * self.v_head_size
            q_parts = get_tp_partition_sizes_or_uniform(q_full)
            kv_parts = get_tp_partition_sizes_or_uniform(kv_full)
            v_parts = get_tp_partition_sizes_or_uniform(v_full)
            self.num_heads_per_rank = [p // self.head_size for p in q_parts]
            self.num_kv_heads_per_rank = [p // self.head_size for p in kv_parts]
            # QKV parallelism is fundamentally head-count based, so the
            # per-rank partition must align to the head grid. We allow
            # ``--tensor-parallel-weights`` only when every rank ends up
            # with an integer number of heads for Q, K, and V; otherwise
            # we fall back to uniform partitioning and warn so the user
            # can either pick weights that divide cleanly or accept the
            # uniform split.
            if any(
                p % self.head_size != 0
                for p in (*q_parts, *kv_parts, *v_parts)
            ):
                from vllm.logger import init_logger

                logger = init_logger(__name__)
                logger.warning(
                    "QKVParallelLinear: --tensor-parallel-weights %s does not "
                    "produce an integer number of heads per rank "
                    "(q_parts=%s, kv_parts=%s, v_parts=%s for head_size=%d). "
                    "Falling back to uniform partitioning for this layer.",
                    self.tp_rank,
                    q_parts,
                    kv_parts,
                    v_parts,
                    self.head_size,
                )
                self.num_heads_per_rank = [self.num_heads] * tp_size
                self.num_kv_heads_per_rank = [self.num_kv_heads] * tp_size

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def validate_shard_id(self, loaded_shard_id: str | None):
        if loaded_shard_id is None:
            return
        if isinstance(loaded_shard_id, str):
            if loaded_shard_id not in ["q", "k", "v"]:
                raise ValueError(
                    "Shard id for QKVParallelLinear should be 'q', 'k', or 'v', "
                    f"got shard id {loaded_shard_id}."
                )
            return
        raise ValueError("This line should not be reached")

    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        # Heterogeneous TP: use the rank's own head count so the loaded
        # shard slice lines up with the rank-local param_data view.
        local_num_heads = self.num_heads_per_rank[self.tp_rank]
        local_num_kv_heads = self.num_kv_heads_per_rank[self.tp_rank]
        shard_offset_mapping = {
            "q": 0,
            "k": local_num_heads * self.head_size,
            "v": (local_num_heads + local_num_kv_heads) * self.head_size,
            "total": (local_num_heads + local_num_kv_heads) * self.head_size
            + local_num_kv_heads * self.v_head_size,
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str):
        local_num_heads = self.num_heads_per_rank[self.tp_rank]
        local_num_kv_heads = self.num_kv_heads_per_rank[self.tp_rank]
        shard_size_mapping = {
            "q": local_num_heads * self.head_size,
            "k": local_num_kv_heads * self.head_size,
            "v": local_num_kv_heads * self.v_head_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ):
        """
        Handle special case for models where QKV layers are already
        fused on disk. In this case, we have no shard id. This function
        determines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.v_head_size,
            ),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if isinstance(param, BlockQuantScaleParameter):
                weight_block_size = getattr(self, "weight_block_size", None)
                shard_size, shard_offset = adjust_block_scale_shard(
                    weight_block_size, shard_size, shard_offset
                )
            elif (
                isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                and param.packed_dim == param.output_dim
            ):
                shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                    shard_size=shard_size, shard_offset=shard_offset
                )

            loaded_weight_shard = loaded_weight.narrow(
                param.output_dim, shard_offset, shard_size
            )
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        self.validate_shard_id(loaded_shard_id)
        if loaded_shard_id is None:  # special case for certain models
            if isinstance(param, PerTensorScaleParameter):
                # When weights are already fused on disk (e.g. Phi-3's
                # qkv_proj), there is only a single scale for the entire
                # fused matrix. Fill all slots (q, k, v) with this scale
                # to ensure that any subsequent reduction (like .max())
                # works correctly while preserving the parameter shape.
                for idx in range(param.data.shape[0]):
                    param.load_qkv_weight(
                        loaded_weight=loaded_weight, shard_id=idx, tp_rank=self.tp_rank
                    )
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, tp_rank=self.tp_rank)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        # Heterogeneous TP: the offset within the full loaded_weight for this
        # rank's slice of the current QKV segment (see the v1 weight_loader's
        # mirroring block for the rationale).
        if loaded_shard_id == "q":
            q_parts = (
                [self.num_heads_per_rank[0] * self.head_size]
                if self.tp_size == 1
                else get_tp_partition_sizes_or_uniform(self.total_num_heads * self.head_size)
            )
            loaded_weight_offset = sum(q_parts[: self.tp_rank])
        else:
            if self.num_kv_head_replicas > 1:
                loaded_weight_offset = (
                    (self.tp_rank // self.num_kv_head_replicas)
                    * self.num_kv_heads_per_rank[0]
                    * self.head_size
                )
            else:
                loaded_weight_offset = (
                    sum(self.num_kv_heads_per_rank[: self.tp_rank]) * self.head_size
                )

        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=self.num_kv_head_replicas,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            loaded_weight_offset=loaded_weight_offset,
            tp_rank=self.tp_rank,
        )
    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        self.validate_shard_id(loaded_shard_id)
        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            idx_map = {"q": 0, "k": 1, "v": 2}
            if loaded_shard_id is not None:
                param.data[idx_map[loaded_shard_id]].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            else:
                param.shard_weight_type = {k: loaded_weight.item() for k in idx_map}
            return

        if is_gguf_weight:
            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size

            if loaded_shard_id is not None:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)

        # Special case for per-tensor scales in fused case.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv).
            # (e.g., Phi-3's qkv_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0
                    )

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_heads * self.head_size),
                (
                    "k",
                    self.total_num_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                    self.total_num_kv_heads * self.v_head_size,
                ),
            ]
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)

            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantized Weights.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                # Add check to adjust the size/offset for FP8 block scales
                if isinstance(param, BlockQuantScaleParameter):
                    weight_block_size = getattr(self, "weight_block_size", None)
                    shard_size, shard_offset = adjust_block_scale_shard(
                        weight_block_size, shard_size, shard_offset
                    )

                if packed_dim == output_dim:
                    shard_size = round(shard_size // param.packed_factor)
                    shard_offset = round(shard_offset // param.packed_factor)

                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset
                    )

                if use_bitsandbytes_4bit:
                    orig_qkv_offsets = {
                        "q": (0, self.total_num_heads * self.head_size),
                        "k": (
                            self.total_num_heads * self.head_size,
                            self.total_num_kv_heads * self.head_size,
                        ),
                        "v": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size,
                            self.total_num_kv_heads * self.v_head_size,
                        ),
                        "total": (
                            (self.total_num_heads + self.total_num_kv_heads)
                            * self.head_size
                            + self.total_num_kv_heads * self.v_head_size,
                            0,
                        ),
                    }

                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_qkv_offsets, shard_id
                    )

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return
        # If output dim is defined, use the default loading process.
        if output_dim is not None:
            # Heterogeneous TP: use the rank's own head counts so the loaded
            # shard slice lines up with both the rank-local param_data view
            # (already partitioned when the parameter was created) and the
            # full on-disk tensor. Under uniform TP the per-rank values are
            # all equal and this collapses to the legacy math.
            local_num_heads = self.num_heads_per_rank[self.tp_rank]
            local_num_kv_heads = self.num_kv_heads_per_rank[self.tp_rank]
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = local_num_heads * self.head_size
            elif loaded_shard_id == "k":
                shard_offset = local_num_heads * self.head_size
                shard_size = local_num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = (local_num_heads + local_num_kv_heads) * self.head_size
                shard_size = local_num_kv_heads * self.v_head_size

            if isinstance(param, BlockQuantScaleParameter):
                weight_block_size = getattr(self, "weight_block_size", None)
                shard_size, shard_offset = adjust_block_scale_shard(
                    weight_block_size, shard_size, shard_offset
                )

            # Special case for Quantized Weights.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = round(shard_size // param.packed_factor)
                shard_offset = round(shard_offset // param.packed_factor)

                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow
            is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

            if use_bitsandbytes_4bit:
                orig_qkv_offsets = {
                    "q": (0, self.num_heads * self.head_size),
                    "k": (
                        self.num_heads * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "v": (
                        (self.num_heads + self.num_kv_heads) * self.head_size,
                        self.num_kv_heads * self.v_head_size,
                    ),
                    "total": (
                        (self.num_heads + self.num_kv_heads) * self.head_size
                        + self.num_kv_heads * self.v_head_size,
                        0,
                    ),
                }
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, loaded_shard_id
                )

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            # start_idx is the offset within the *full* loaded weight for this
            # rank's slice of the current segment. Q is always partitioned by
            # column. K/V is partitioned when ``tp_size < total_num_kv_heads``
            # (``num_kv_head_replicas == 1``) and replicated otherwise (each KV
            # head is shared by ``num_kv_head_replicas`` consecutive ranks).
            # The cumulative offset collapses to the legacy
            # ``shard_rank * shard_size`` under the uniform-TP path.
            if loaded_shard_id == "q":
                # Cumulative offset across previous ranks in Q's partition.
                q_parts = (
                    [self.num_heads_per_rank[0] * self.head_size]
                    if self.tp_size == 1
                    else get_tp_partition_sizes_or_uniform(
                        self.total_num_heads * self.head_size
                    )
                )
                start_idx = sum(q_parts[: self.tp_rank])
            else:
                # K/V: each rank owns ``num_kv_heads_per_rank[rank]`` heads, and
                # replica groups of size ``num_kv_head_replicas`` share the
                # same heads.
                # - Partitioned (``num_kv_head_replicas == 1``): the offset is
                #   the cumulative head count before this rank (each rank has
                #   a unique chunk of the KV segment).
                # - Replicated (``num_kv_head_replicas > 1``): the offset is
                #   the replica-group index times one head's worth (each
                #   ``num_kv_head_replicas`` consecutive ranks share the same
                #   head).
                # Under uniform TP this collapses to the legacy
                # ``(tp_rank // num_kv_head_replicas) * num_kv_heads *
                # head_size`` formula.
                if self.num_kv_head_replicas > 1:
                    start_idx = (
                        (self.tp_rank // self.num_kv_head_replicas)
                        * self.num_kv_heads_per_rank[0]
                        * self.head_size
                    )
                else:
                    start_idx = (
                        sum(self.num_kv_heads_per_rank[: self.tp_rank])
                        * self.head_size
                    )
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


# --8<-- [start:row_parallel_linear]
@PluggableLayer.register("row_parallel_linear")
class RowParallelLinear(LinearBase):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        reduce_results: If true, call all-reduce on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y = X_iA_i
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.down_proj)
        return_bias: If true, return bias together with outputs in forward pass.
        disable_tp: If true, weights matrix won't be sharded through tp rank.
    """

    # --8<-- [end:row_parallel_linear]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ):
        # Divide the weight matrix along the first dimension.
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        # Per-rank input partition sizes for heterogeneous TP (see the
        # ``ColumnParallelLinear`` ``__init__`` for the rationale).
        if disable_tp or self.tp_size == 1:
            self.partition_sizes = [input_size]
            self.input_size_per_partition = input_size
        else:
            self.partition_sizes = get_tp_partition_sizes_or_uniform(input_size)
            self.input_size_per_partition = self.partition_sizes[self.tp_rank]
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        super().__init__(
            input_size,
            output_size,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
            prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=self.input_size_per_partition,
            output_partition_sizes=self.output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=(
                self.weight_loader_v2
                if self.quant_method.__class__.__name__ in WEIGHT_LOADER_V2_SUPPORTED
                else self.weight_loader
            ),
        )
        if not reduce_results and (bias and not skip_bias_add):
            raise ValueError(
                "When not reduce the results, adding bias to the "
                "results can lead to incorrect results"
            )

        if bias:
            self.bias = Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)
        self.update_param_tp_status()

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        is_sharded_weight = getattr(param, "is_sharded_weight", False)
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow
        is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

        # Special case for GGUF
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            weight_shape = list(loaded_weight.shape)
            if input_dim:
                weight_shape[input_dim] = self.input_size_per_partition
            param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)

        param_data = param.data
        if input_dim is not None and not is_sharded_weight:
            shard_size = param_data.shape[input_dim]
            # Use the cumulative offset across per-rank partition sizes; for
            # the uniform case this collapses to ``tp_rank * shard_size``.
            start_idx = sum(self.partition_sizes[: self.tp_rank])
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def weight_loader_v2(self, param: BasevLLMParameter, loaded_weight: torch.Tensor):
        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            assert loaded_weight.numel() == 1
            loaded_weight = loaded_weight.reshape(1)

        param.load_row_parallel_weight(loaded_weight=loaded_weight)

    def forward(
        self,
        input_,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size
            )
            input_parallel = split_input[self.tp_rank].contiguous()

        # Matrix multiply.
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def extra_repr(self) -> str:
        s = f"in_features={self.input_size_per_partition}"
        s += f", output_features={self.output_size}"
        s += f", bias={self.bias is not None}"
        s += f", tp_size={self.tp_size}"
        s += f", reduce_results={self.reduce_results}"
        return s
