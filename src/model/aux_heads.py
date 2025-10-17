"""Auxiliary heads for reconstructing PBR modalities."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_head(in_ch: int, out_ch: int) -> nn.Sequential:
    """Simple two-layer conv head used for auxiliary predictions."""

    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True),
    )


class AuxHeads(nn.Module):
    """Configurable collection of auxiliary prediction heads.

    The module exposes a unified interface that allows the caller to enable a
    subset of heads (normals, roughness, specular and emissive). Each head can
    compute its own loss term which is aggregated by :meth:`compute_loss` and
    returned alongside a dictionary of logging values (useful for TensorBoard or
    console logging).
    """

    _HEAD_CHANNELS: Mapping[str, int] = {
        "normals": 3,
        "rough": 1,
        "spec": 1,
        "emiss": 1,
    }

    def __init__(
        self,
        in_channels: int,
        *,
        enabled: Optional[Iterable[str]] = None,
        loss_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        if enabled is None:
            enabled = self._HEAD_CHANNELS.keys()

        self.heads = nn.ModuleDict()
        for name in enabled:
            if name not in self._HEAD_CHANNELS:
                raise ValueError(f"Unknown auxiliary head '{name}'")
            out_ch = self._HEAD_CHANNELS[name]
            self.heads[name] = _make_head(in_channels, out_ch)

        self.loss_weights = {
            head: float(loss_weights[head]) if loss_weights and head in loss_weights else 1.0
            for head in self.heads.keys()
        }

        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------
    @property
    def enabled_heads(self) -> Tuple[str, ...]:
        """Returns the tuple of enabled head names."""

        return tuple(self.heads.keys())

    # ------------------------------------------------------------------
    # Forward / loss API
    # ------------------------------------------------------------------
    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs every enabled head over the provided feature map."""

        outputs: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            outputs[name] = head(feat)
        return outputs

    def compute_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        *,
        reduction: str = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the aggregated auxiliary loss and logging scalars.

        Parameters
        ----------
        preds:
            Dictionary with the predictions produced by :meth:`forward`.
        targets:
            Dictionary with reference tensors (same naming convention as
            :attr:`enabled_heads`). Target tensors will be resized if needed so
            they match the spatial dimensions of the predictions.
        reduction:
            Currently supports ``"mean"`` and ``"sum"``.
        """

        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be either 'mean' or 'sum'")

        if not self.heads:
            zero = self._zero.detach().clone()
            return zero, {}

        total_loss: Optional[torch.Tensor] = None
        log_scalars: Dict[str, float] = {}

        for name, head in self.heads.items():
            if name not in preds or name not in targets:
                continue

            pred = preds[name]
            target = targets[name]
            if pred.shape != target.shape:
                target = F.interpolate(
                    target,
                    size=pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss_val = F.l1_loss(pred, target, reduction=reduction)
            weighted = loss_val * self.loss_weights[name]
            total_loss = weighted if total_loss is None else total_loss + weighted

            log_scalars[f"aux/{name}"] = float(loss_val.detach().cpu())

        if total_loss is None:
            total_loss = self._zero.detach().clone()
        log_scalars["aux/total"] = float(total_loss.detach().cpu())
        return total_loss, log_scalars

    def extra_repr(self) -> str:
        heads = ", ".join(self.enabled_heads) or "none"
        return f"heads=[{heads}]"

