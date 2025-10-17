"""Backbone blocks with FiLM conditioning."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .film import FiLM


class ConvBNAct(nn.Sequential):
    """Convenience block used by the toy backbone."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(),
        )


class ConditionedBackbone(nn.Module):
    """Small CNN backbone that injects the context vector through FiLM."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        dim_context: int = 64,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.stem = ConvBNAct(in_channels, base_channels, kernel_size=3, stride=2)
        self.stage1 = nn.Sequential(
            ConvBNAct(base_channels, base_channels, kernel_size=3, stride=1),
            ConvBNAct(base_channels, base_channels * 2, kernel_size=3, stride=2),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(base_channels * 2, base_channels * 2, kernel_size=3, stride=1),
            ConvBNAct(base_channels * 2, base_channels * 4, kernel_size=3, stride=2),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(base_channels * 4, base_channels * 4, kernel_size=3, stride=1),
            ConvBNAct(base_channels * 4, base_channels * 8, kernel_size=3, stride=2),
        )

        self.film1 = FiLM(base_channels * 2, dim_context)
        self.film2 = FiLM(base_channels * 4, dim_context)
        self.film3 = FiLM(base_channels * 8, dim_context)

        self._out_channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

    @property
    def out_channels(self) -> List[int]:
        return list(self._out_channels)

    def forward(self, x: torch.Tensor, context_vec: torch.Tensor | None) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []

        x = self.stem(x)
        features.append(x)

        x = self.stage1(x)
        x = self.film1(x, context_vec)
        features.append(x)

        x = self.stage2(x)
        x = self.film2(x, context_vec)
        features.append(x)

        x = self.stage3(x)
        x = self.film3(x, context_vec)
        features.append(x)

        return features
