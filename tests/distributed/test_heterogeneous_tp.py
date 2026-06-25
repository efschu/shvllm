# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for heterogeneous tensor parallelism (--tensor-parallel-weights).

These tests focus on the *pure* logic that doesn't need a real NCCL backend:

* ``ParallelConfig.tp_partition_sizes`` (numeric correctness, including the
  ``[1, 1, 1]`` parity case and remainder handling).
* ``ParallelConfig.__post_init__`` validation (length, sign).
* ``tensor_model_parallel_all_gather`` pad-gather-slice round-trip with a
  mocked TP group.
* CLI parsing for ``--tensor-parallel-weights``.
"""

from unittest.mock import MagicMock

import pytest

from vllm.config.parallel import ParallelConfig
from vllm.distributed import parallel_state as parallel_state_mod
from vllm.distributed.communication_op import tensor_model_parallel_all_gather


# ---------- ParallelConfig.tp_partition_sizes ----------


class TestPartitionSizes:
    """Verify the numeric split derived from --tensor-parallel-weights."""

    def test_uniform_weights_match_uniform_path(self):
        cfg = ParallelConfig(tensor_parallel_size=3, tensor_parallel_weights=[1, 1, 1])
        assert cfg.tp_partition_sizes(16) == [6, 5, 5]
        assert sum(cfg.tp_partition_sizes(16)) == 16
        assert sum(cfg.tp_partition_sizes(12)) == 12
        assert sum(cfg.tp_partition_sizes(15)) == 15

    def test_heterogeneous_2_1_1(self):
        cfg = ParallelConfig(tensor_parallel_size=3, tensor_parallel_weights=[2, 1, 1])
        # 50% / 25% / 25% of hidden_size=16 -> [8, 4, 4]
        assert cfg.tp_partition_sizes(16) == [8, 4, 4]
        # Remainder goes to rank 0 (ties broken by rank)
        assert sum(cfg.tp_partition_sizes(10)) == 10
        # The 50% rank never has fewer than the others (relative share)
        for total in (10, 12, 15, 16, 24, 1024, 4096):
            sizes = cfg.tp_partition_sizes(total)
            assert len(sizes) == 3
            assert all(s >= 0 for s in sizes)
            assert sum(sizes) == total
            # Rank 0 has at least as much as any other rank (weights are 2,1,1)
            assert sizes[0] >= sizes[1]
            assert sizes[0] >= sizes[2]

    def test_tp_size_1_returns_full(self):
        cfg = ParallelConfig(tensor_parallel_size=1)
        # Weights ignored when tp_size == 1; everything stays on the one rank.
        assert cfg.tp_partition_sizes(64) == [64]

    def test_weights_default_to_uniform_when_unset(self):
        cfg = ParallelConfig(tensor_parallel_size=4)
        assert cfg.tp_partition_sizes(17) == [5, 4, 4, 4]

    def test_validation_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="tensor_parallel_weights must have exactly"):
            ParallelConfig(
                tensor_parallel_size=3, tensor_parallel_weights=[1, 1]
            )

    def test_validation_rejects_non_positive(self):
        with pytest.raises(ValueError, match="all tensor_parallel_weights must be > 0"):
            ParallelConfig(
                tensor_parallel_size=3, tensor_parallel_weights=[1, 0, 1]
            )
        with pytest.raises(ValueError, match="all tensor_parallel_weights must be > 0"):
            ParallelConfig(
                tensor_parallel_size=3, tensor_parallel_weights=[1, -1, 1]
            )


# ---------- AllGather pad-gather-slice ----------


class _MockGatherGroup:
    """Mocks just enough of the TP GroupCoordinator for the pad-gather-slice path.

    The production code calls ``group.rank_in_group``, ``group.world_size``, and
    ``group.all_gather(input_, dim)``. The real ``all_gather`` is replaced with a
    per-rank ``torch.cat`` of equal-sized padded tensors, which is what the
    NCCL collective is supposed to do. We then assert the slice reconstructs
    the original full tensor.
    """


    def __init__(self, rank: int, world_size: int):
        self.rank_in_group = rank
        self.world_size = world_size
        self.observed_inputs: list[torch.Tensor] = []
        self.observed_partition_sizes: list[list[int] | None] = []
        # Per-call chunks: stored as ``(chunks, partition_sizes)`` so each
        # ``all_gather`` invocation can carry its own partition context.
        self._pending_chunks: list[torch.Tensor] = []
        self._pending_partition_sizes: list[int] = []
        # Whether the next ``all_gather`` call should use the equal-chunk
        # (legacy) path or the pad-gather-slice path.
        self.next_use_legacy = False

    def set_chunks(
        self,
        chunks: list[torch.Tensor],
        partition_sizes: list[int],
    ) -> None:
        self._pending_chunks = chunks
        self._pending_partition_sizes = partition_sizes

    def all_gather(
        self,
        input_: torch.Tensor,
        dim: int,
        partition_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

        self.observed_inputs.append(input_)
        self.observed_partition_sizes.append(partition_sizes)

        # Mirror the production ``GroupCoordinator.all_gather`` dispatch:
        # - if ``partition_sizes`` is set and the local chunk matches the
        #   expected size, use pad-gather-slice;
        # - else fall back to the legacy equal-chunk ``all_gather`` (we
        #   approximate it by repeating the caller's input, which is fine
        #   for tests that just want to exercise the fallback branch).
        if (
            partition_sizes is not None
            and self._pending_partition_sizes
            and input_.size(dim) == partition_sizes[self.rank_in_group]
        ):
            max_size = max(partition_sizes)
            padded = [
                F.pad(c, (0, max_size - c.shape[-1]))
                if c.shape[-1] < max_size
                else c.clone()
                for c in self._pending_chunks
            ]
            return torch.cat(padded, dim=dim)
        return torch.cat([input_] * self.world_size, dim=dim)

@pytest.fixture
def mock_tp_group(monkeypatch):
    """Install a fake ``get_tp_group`` that returns our mock."""

    def _make(weights, total_size, hidden_size):
        sizes = ParallelConfig(
            tensor_parallel_size=len(weights),
            tensor_parallel_weights=list(weights),
        ).tp_partition_sizes(hidden_size)
        # Set the global flag so the production all_gather dispatches into the
        # pad-gather-slice branch.
        parallel_state_mod._TP_PARTITION_SIZES = sizes
        import torch
        full = torch.arange(total_size, dtype=torch.float32)
        chunks = [
            full[sum(sizes[:r]) : sum(sizes[: r + 1])].clone()
            for r in range(len(weights))
        ]
        groups = {
            rank: _MockGatherGroup(rank=rank, world_size=len(weights))
            for rank in range(len(weights))
        }
        for g in groups.values():
            g.set_chunks(chunks, sizes)
        active = {"rank": 0}

        def _get_tp_group():
            return groups[active["rank"]]

        monkeypatch.setattr(parallel_state_mod, "get_tp_group", _get_tp_group)
        return groups, sizes, active, chunks

    yield _make
    parallel_state_mod._TP_PARTITION_SIZES = None


def test_uniform_weights_skip_pad(mock_tp_group):
    """Weights [1,1,1] with hidden_size=12 (tp=3) should be bit-exact: no padding."""
    groups, sizes, active, chunks = mock_tp_group(
        [1, 1, 1], total_size=12, hidden_size=12
    )
    assert sizes == [4, 4, 4]

    import torch

    full = torch.arange(12, dtype=torch.float32)

    for rank in range(3):
        active["rank"] = rank
        out = tensor_model_parallel_all_gather(chunks[rank], dim=-1)
        assert torch.equal(out, full), (
            f"Rank {rank}: out={out} vs full={full}, sizes={sizes}"
        )
        # No padding: each rank's contribution to all_gather was the
        # original 4-element chunk.
        last = groups[rank].observed_inputs[-1]
        assert last.shape[-1] == sizes[rank], (
            f"Rank {rank}: padded to {last.shape[-1]} != {sizes[rank]}"
        )


def test_heterogeneous_round_trip(mock_tp_group):
    """Weights [2,1,1] with hidden_size=10 should round-trip through
    pad-gather-slice without losing or duplicating data."""
    groups, sizes, active, chunks = mock_tp_group(
        [2, 1, 1], total_size=10, hidden_size=10
    )
    # 50%/25%/25% with remainder distribution -> [5, 3, 2] summing to 10.
    assert sum(sizes) == 10
    assert all(s > 0 for s in sizes)

    import torch

    full = torch.arange(10, dtype=torch.float32)
    for rank in range(3):
        active["rank"] = rank
        out = tensor_model_parallel_all_gather(chunks[rank], dim=-1)
        assert torch.equal(out, full), (
            f"Rank {rank}: reconstruction differs. "
            f"Got {out}, expected {full}, sizes={sizes}"
        )
        # Every padded input has size == max(sizes).
        max_size = max(sizes)
        last = groups[rank].observed_inputs[-1]
        assert last.shape[-1] == max_size, (
            f"Rank {rank} padded to {last.shape[-1]} != max(sizes)={max_size}"
        )


def test_uniform_parity_with_unset_partition_sizes(monkeypatch):
    """When no weights are set, ``get_tp_partition_sizes()`` returns None and
    the all_gather path falls back to the legacy ``group.all_gather``."""
    # Patch the module attribute (and the binding imported into
    # communication_op at load time, since ``from X import Y`` binds locally).
    monkeypatch.setattr(parallel_state_mod, "_TP_PARTITION_SIZES", None)
    monkeypatch.setattr(
        "vllm.distributed.communication_op.get_tp_partition_sizes",
        lambda: None,
    )

    fallback = MagicMock()
    fallback.world_size = 2
    fallback.rank_in_group = 0
    fallback.all_gather.side_effect = lambda x, dim: x * 2

    monkeypatch.setattr(parallel_state_mod, "get_tp_group", lambda: fallback)

    import torch

    x = torch.tensor([1.0, 2.0])
    out = tensor_model_parallel_all_gather(x, dim=-1)
    # The legacy ``group.all_gather`` was invoked with our tensor and we
    # returned ``x * 2`` from the mock.
    fallback.all_gather.assert_called_once()
    assert torch.equal(out, x * 2)


# ---------- CLI parsing ----------


def test_engine_args_cli_parsing():
    """``--tensor-parallel-weights`` parses to ``list[float]`` and round-trips
    through ``ParallelConfig``."""
    from vllm.engine.arg_utils import _parse_tp_weights

    assert _parse_tp_weights("2,1,1") == [2.0, 1.0, 1.0]
    assert _parse_tp_weights(" 1.5 , 0.5 , 1 ") == [1.5, 0.5, 1.0]
    assert _parse_tp_weights(None) is None
    assert _parse_tp_weights("") is None
    assert _parse_tp_weights("   ") is None

    with pytest.raises(Exception):
        _parse_tp_weights("2,abc,1")
