from __future__ import annotations

import torch

from src.model.context_adapter import ContextAdapter


def test_context_adapter_output_shape() -> None:
    adapter = ContextAdapter(in_dim_text=32, in_channels_pbr=7, out_dim=128)
    pbr = torch.rand(4, 7, 16, 16)
    text = torch.rand(4, 32)
    output = adapter(pbr, text)
    assert output.shape == (4, 128)


def test_context_adapter_handles_missing_modalities() -> None:
    adapter = ContextAdapter(in_dim_text=0, in_channels_pbr=7, out_dim=64)
    pbr = torch.rand(2, 7, 8, 8)
    output = adapter(pbr, None)
    assert output.shape == (2, 64)
