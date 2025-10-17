"""Auxiliary heads for reconstructing PBR modalities."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PBR_CHANNEL_GROUPS: Dict[str, slice] = {
    "normal": slice(0, 3),
    "rough": slice(3, 4),
    "ao": slice(4, 5),
    "height": slice(5, 6),
    "metallic": slice(6, 7),
}


def _make_head(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True),
    )


def _masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    channel_slice: slice,
) -> torch.Tensor:
    pred_slice = pred[:, channel_slice, ...]
    tgt_slice = target[:, channel_slice, ...]
    if mask is not None:
        mask_slice = mask[:, channel_slice, ...]
        weight = mask_slice.sum()
        if weight <= 0:
            return pred_slice.new_zeros(())
        return torch.sum(torch.abs(pred_slice - tgt_slice) * mask_slice) / weight.clamp_min(1e-6)
    return torch.mean(torch.abs(pred_slice - tgt_slice))


def _masked_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    kernel_size = 3
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(pred, kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size, stride=1, padding=padding) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = torch.clamp(numerator / (denominator + 1e-6), 0.0, 1.0)
    if mask is not None:
        mask_pool = F.avg_pool2d(mask, kernel_size, stride=1, padding=padding)
        weight = mask_pool.sum().clamp_min(1e-6)
        return torch.sum(ssim_map * mask_pool) / weight
    return ssim_map.mean()


class PBRReconHeads(nn.Module):
    """Auxiliary reconstruction heads for PBR channels."""

    def __init__(
        self,
        channels: Mapping[str, int],
        *,
        out_channels: int,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels mapping must not be empty")
        self.heads = nn.ModuleDict({name: _make_head(in_ch, out_channels) for name, in_ch in channels.items()})
        self.out_channels = out_channels

    def forward(self, features: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            if name in features:
                outputs[name] = head(features[name])
        return outputs

    def compute_loss(
        self,
        preds: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[Mapping[str, float]] = None,
        use_ssim: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not preds:
            device = target.device
            return torch.zeros((), device=device), {}
        total_loss = target.new_tensor(0.0)
        channel_totals: Dict[str, torch.Tensor] = {key: target.new_tensor(0.0) for key in PBR_CHANNEL_GROUPS}
        logs: Dict[str, float] = {}
        for name, pred in preds.items():
            if target.shape[-2:] != pred.shape[-2:]:
                resized_target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
            else:
                resized_target = target
            if mask is not None and mask.shape[-2:] != pred.shape[-2:]:
                resized_mask = F.interpolate(mask, size=pred.shape[-2:], mode="nearest")
            else:
                resized_mask = mask
            head_loss = pred.new_tensor(0.0)
            for channel_name, channel_slice in PBR_CHANNEL_GROUPS.items():
                if channel_slice.stop > self.out_channels:
                    continue
                weight = float(weights.get(channel_name, 1.0)) if weights else 1.0
                l1_value = _masked_l1(pred, resized_target, resized_mask, channel_slice)
                channel_totals[channel_name] = channel_totals[channel_name] + l1_value
                head_loss = head_loss + weight * l1_value
            if use_ssim:
                ssim_val = _masked_ssim(pred, resized_target, resized_mask)
                head_loss = 0.8 * head_loss + 0.2 * (1.0 - ssim_val)
                logs[f"aux/{name}_ssim"] = float(ssim_val.detach().cpu())
            logs[f"aux/{name}_loss"] = float(head_loss.detach().cpu())
            total_loss = total_loss + head_loss
        total_loss = total_loss / len(preds)
        for channel_name, value in channel_totals.items():
            logs[f"aux/l1_{channel_name}"] = float((value / len(preds)).detach().cpu())
        return total_loss, logs


__all__ = ["PBRReconHeads", "PBR_CHANNEL_GROUPS"]

