"""Minimal multimodal model tying together the backbone, neck and heads."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .backbone import ConditionedBackbone
from .neck import ConditionedNeck


class MultimodalYoloStub(nn.Module):
    """Toy segmentation model that showcases FiLM conditioning.

    This is **not** a drop-in replacement for YOLOv11 but a minimal network that
    demonstrates how the provided FiLM blocks are wired through the backbone and
    neck. The forward signature accepts both the image tensor and the context
    vector, complying with the project specification for multimodal training.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dim_context: int = 64,
        base_channels: int = 32,
        neck_channels: int = 128,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.backbone = ConditionedBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            dim_context=dim_context,
        )
        self.neck = ConditionedNeck(
            in_channels_list=self.backbone.out_channels,
            out_channels=neck_channels,
            dim_context=dim_context,
        )
        self.head = nn.Sequential(
            nn.Conv2d(neck_channels, neck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(neck_channels),
            nn.SiLU(),
            nn.Conv2d(neck_channels, num_classes, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor, context_vec: torch.Tensor | None
    ) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
        """Forward pass of the multimodal model."""

        backbone_feats = self.backbone(x, context_vec)
        neck_feats = self.neck(backbone_feats, context_vec)
        logits = self.head(neck_feats[0])
        return {
            "logits": logits,
            "neck_features": neck_feats,
            "backbone_features": backbone_feats,
        }
