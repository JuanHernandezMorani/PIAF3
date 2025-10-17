"""Context adapter that fuses textual and PBR information."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(in_features: int, hidden: int, out_features: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_features = in_features
    if hidden > 0:
        layers.append(nn.Linear(current_features, hidden))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_features = hidden
    layers.append(nn.Linear(current_features, out_features))
    return nn.Sequential(*layers)


class ContextAdapter(nn.Module):
    """Fuse PBR feature maps and text embeddings into a single context vector."""

    def __init__(
        self,
        in_dim_text: int,
        in_channels_pbr: int,
        out_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if out_dim <= 0:
            raise ValueError("out_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.in_dim_text = in_dim_text
        self.in_channels_pbr = in_channels_pbr
        self.out_dim = out_dim

        self.pbr_mlp: Optional[nn.Module]
        if in_channels_pbr > 0:
            self.pbr_mlp = _build_mlp(in_channels_pbr, hidden_dim, hidden_dim, dropout=dropout)
        else:
            self.pbr_mlp = None

        self.text_mlp: Optional[nn.Module]
        if in_dim_text > 0:
            self.text_mlp = _build_mlp(in_dim_text, hidden_dim, hidden_dim, dropout=dropout)
        else:
            self.text_mlp = None

        fusion_in = 0
        if self.pbr_mlp is not None:
            fusion_in += hidden_dim
        if self.text_mlp is not None:
            fusion_in += hidden_dim
        if fusion_in == 0:
            fusion_in = out_dim

        self.layer_norm = nn.LayerNorm(fusion_in)
        self.output = nn.Sequential(
            nn.Linear(fusion_in, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        pbr: Optional[torch.Tensor],
        text: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the conditioning vector for FiLM modules."""

        device = None
        features: list[torch.Tensor] = []

        if pbr is not None and self.pbr_mlp is not None:
            if pbr.dim() != 4:
                raise ValueError("pbr tensor must be 4D (B, C, H, W)")
            pooled = F.adaptive_avg_pool2d(pbr, 1).view(pbr.size(0), -1)
            features.append(self.pbr_mlp(pooled))
            device = pbr.device
        if text is not None and self.text_mlp is not None:
            if text.dim() != 2:
                raise ValueError("text tensor must be 2D (B, E)")
            features.append(self.text_mlp(text))
            device = text.device

        if not features:
            if device is None:
                device = torch.device("cpu")
            batch = 0
            if pbr is not None:
                batch = pbr.size(0)
            elif text is not None:
                batch = text.size(0)
            if batch <= 0:
                batch = 1
            return torch.zeros((batch, self.out_dim), device=device)

        concatenated = torch.cat(features, dim=1)
        normalized = self.layer_norm(concatenated)
        return self.output(normalized)


__all__ = ["ContextAdapter"]

