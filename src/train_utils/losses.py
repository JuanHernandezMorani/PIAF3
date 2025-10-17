"""Loss helpers for multimodal YOLO training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from src.model.aux_heads import PBRReconHeads


class DetectionSegLoss:
    """Wrapper around Ultralytics' native detection/segmentation loss."""

    def __init__(self, model: nn.Module) -> None:
        if not hasattr(model, "loss"):
            raise ValueError("The provided model does not expose a 'loss' method")
        self.model = model

    def __call__(self, predictions: object, batch: Mapping[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        loss, metrics = self.model.loss(predictions, batch)
        log: Dict[str, float] = {}
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                log[f"det/{key}"] = float(torch.as_tensor(value).detach().cpu())
        elif isinstance(metrics, (list, tuple)):
            names = getattr(self.model, "loss_names", None)
            if isinstance(names, (list, tuple)):
                for name, value in zip(names, metrics):
                    log[f"det/{name}"] = float(torch.as_tensor(value).detach().cpu())
        return loss, log


@dataclass
class AuxLossConfig:
    weights: Mapping[str, float]
    use_ssim: bool = False


class AuxPBRLoss:
    """Auxiliary reconstruction loss handler."""

    def __init__(self, head: Optional[PBRReconHeads], config: AuxLossConfig) -> None:
        self.head = head
        self.config = config

    def __call__(
        self,
        preds: Mapping[str, Tensor],
        target: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        if self.head is None or not preds:
            device = target.device
            return torch.zeros((), device=device), {}
        loss, logs = self.head.compute_loss(
            preds,
            target,
            mask=mask,
            weights=self.config.weights,
            use_ssim=self.config.use_ssim,
        )
        return loss, logs


class MultiModalLoss:
    """Combine detection and auxiliary reconstruction losses."""

    def __init__(
        self,
        detection_loss: DetectionSegLoss,
        aux_loss: AuxPBRLoss,
        *,
        aux_lambda: float,
        warmup_iters: int,
    ) -> None:
        self.detection_loss = detection_loss
        self.aux_loss = aux_loss
        self.aux_lambda = float(aux_lambda)
        self.warmup_iters = int(max(0, warmup_iters))
        self.iteration = 0

    def _current_lambda(self) -> float:
        if self.aux_lambda <= 0:
            return 0.0
        if self.warmup_iters <= 0:
            return self.aux_lambda
        progress = min(self.iteration / max(1, self.warmup_iters), 1.0)
        return float(self.aux_lambda * progress)

    def __call__(
        self,
        predictions: object,
        batch: Mapping[str, Tensor],
        *,
        aux_preds: Mapping[str, Tensor],
        aux_target: Tensor,
        aux_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        det_loss, det_logs = self.detection_loss(predictions, batch)
        aux_loss_value, aux_logs = self.aux_loss(aux_preds, aux_target, mask=aux_mask)
        weight = self._current_lambda()
        total = det_loss + weight * aux_loss_value
        metrics = dict(det_logs)
        metrics.update(aux_logs)
        metrics["loss/aux_weight"] = weight
        metrics["loss/det"] = float(det_loss.detach().cpu())
        metrics["loss/aux"] = float(aux_loss_value.detach().cpu())
        metrics["loss/total"] = float(total.detach().cpu())
        self.iteration += 1
        return total, metrics


__all__ = ["DetectionSegLoss", "AuxPBRLoss", "AuxLossConfig", "MultiModalLoss"]

