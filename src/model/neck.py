"""Neck modules that incorporate FiLM conditioning."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .film import FiLM
from .backbone import ConvBNAct


class ConditionedNeck(nn.Module):
    """Simple FPN-like neck with FiLM on each level."""

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 128,
        dim_context: int = 64,
    ) -> None:
        super().__init__()
        if not in_channels_list:
            raise ValueError("in_channels_list cannot be empty")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")

        self.in_channels_list = list(in_channels_list)
        self.out_channels = out_channels

        self.reduce_layers = nn.ModuleList(
            [ConvBNAct(cin, out_channels, kernel_size=1, stride=1) for cin in in_channels_list]
        )
        self.film_pre = nn.ModuleList([FiLM(out_channels, dim_context) for _ in in_channels_list])
        self.post_layers = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, kernel_size=3, stride=1) for _ in in_channels_list]
        )
        self.film_post = nn.ModuleList([FiLM(out_channels, dim_context) for _ in in_channels_list])

    def forward(self, features: List[torch.Tensor], context_vec: torch.Tensor | None) -> List[torch.Tensor]:
        if len(features) != len(self.reduce_layers):
            raise ValueError("features length must match the configured levels")

        conditioned = []
        for feat, reduce, film in zip(features, self.reduce_layers, self.film_pre):
            x = reduce(feat)
            x = film(x, context_vec)
            conditioned.append(x)

        outputs: List[torch.Tensor] = [None] * len(conditioned)
        top_down: torch.Tensor | None = None
        for level in reversed(range(len(conditioned))):
            x = conditioned[level]
            if top_down is not None:
                top_down = F.interpolate(top_down, size=x.shape[-2:], mode="nearest")
                x = x + top_down
            x = self.post_layers[level](x)
            x = self.film_post[level](x, context_vec)
            outputs[level] = x
            top_down = x

        return outputs

    @property
    def out_channels_list(self) -> List[int]:
        return [self.out_channels] * len(self.reduce_layers)
