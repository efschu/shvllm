# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the Qwen3.5/3.6 text-only and MTP config fixes."""

import pytest
import torch
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS


def test_text_only_architectures_registered():
    assert _TEXT_GENERATION_MODELS["Qwen3_5ForCausalLM"] == (
        "qwen3_5",
        "Qwen3_5ForCausalLM",
    )
    assert _TEXT_GENERATION_MODELS["Qwen3_5MoeForCausalLM"] == (
        "qwen3_5",
        "Qwen3_5MoeForCausalLM",
    )


def _mtp_config(model_type: str) -> PretrainedConfig:
    return PretrainedConfig(
        model_type=model_type,
        architectures=["SomeArch"],
        mtp_num_hidden_layers=1,
    )


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
        # Text-only config variants (e.g. GGUF checkpoints shipping only
        # the text_config) carry the same mtp_num_hidden_layers field.
        ("qwen3_5_text", "Qwen3_5MTP"),
        ("qwen3_5_moe_text", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_recognizes_text_only_types(model_type, expected_arch):
    cfg = SpeculativeConfig.hf_config_override(_mtp_config(model_type))
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 1


def test_mrope_positions_for_text_only_model():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    tokens = list(range(5))
    positions, delta = Qwen3_5ForCausalLMBase.get_mrope_input_positions(
        None, tokens, []
    )
    # Text-only: all three M-RoPE streams equal the 1D positions.
    assert positions.shape == (3, 5)
    assert delta == 0
    expected = torch.arange(5, dtype=torch.long)
    for stream in positions:
        assert torch.equal(stream, expected)


def test_mrope_positions_reject_multimodal_features():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    with pytest.raises(ValueError, match="text-only"):
        Qwen3_5ForCausalLMBase.get_mrope_input_positions(None, [0, 1], ["mm"])
