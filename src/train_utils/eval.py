"""Reusable evaluation helpers for multimodal YOLO training and validation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .text import SimpleTextEncoder

try:  # pragma: no cover
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
except Exception:  # pragma: no cover
    ops = None
    SegmentMetrics = None
    box_iou = None
    mask_iou = None


def prepare_detection_batch(
    images: Tensor,
    targets: Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
    device: Optional[torch.device],
) -> Dict[str, Tensor]:
    """Convert YOLO targets into the format expected by Ultralytics losses."""

    if isinstance(targets, Mapping):
        sample_count = len(next(iter(targets.values()))) if targets else 0
        target_list: List[Mapping[str, Any]] = []
        for idx in range(sample_count):
            sample: Dict[str, Any] = {}
            for key, values in targets.items():
                value = values[idx]
                if value is not None:
                    sample[key] = value
            target_list.append(sample)
    else:
        target_list = list(targets)

    cls_tensors: List[Tensor] = []
    box_tensors: List[Tensor] = []
    batch_indices: List[Tensor] = []
    mask_tensors: List[Tensor] = []
    for batch_idx, target in enumerate(target_list):
        boxes = target.get("boxes")
        classes = target.get("cls") or target.get("classes")
        if boxes is None or classes is None or boxes.numel() == 0:
            continue
        boxes_tensor = boxes.clone()
        cls_tensor = classes.view(-1, 1).float()
        if device is not None:
            boxes_tensor = boxes_tensor.to(device)
            cls_tensor = cls_tensor.to(device)
        cls_tensors.append(cls_tensor)
        box_tensors.append(boxes_tensor)
        idx_tensor = torch.full((boxes_tensor.size(0),), batch_idx, dtype=torch.long, device=device)
        if device is None:
            idx_tensor = idx_tensor.cpu()
        batch_indices.append(idx_tensor)
        masks = target.get("masks")
        if masks is not None and masks.numel() > 0:
            mask_tensor = masks.clone()
            if device is not None:
                mask_tensor = mask_tensor.to(device)
            mask_tensors.append(mask_tensor)
    device_out = device if device is not None else images.device
    cls = torch.cat(cls_tensors, dim=0) if cls_tensors else torch.zeros((0, 1), device=device_out)
    bboxes = torch.cat(box_tensors, dim=0) if box_tensors else torch.zeros((0, 4), device=device_out)
    batch_idx_tensor = (
        torch.cat(batch_indices, dim=0)
        if batch_indices
        else torch.zeros((0,), device=device_out, dtype=torch.long)
    )
    mask_shape = (0, images.shape[2], images.shape[3])
    masks = torch.cat(mask_tensors, dim=0) if mask_tensors else torch.zeros(mask_shape, device=device_out)
    return {
        "img": images,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx_tensor,
        "masks": masks,
    }


def _postprocess_predictions(
    predictions: Any,
    *,
    conf: float,
    iou: float,
    max_det: int,
    agnostic: bool,
) -> tuple[List[Tensor], Tensor]:
    if ops is None:  # pragma: no cover
        raise ImportError("Ultralytics ops utilities are required for post-processing predictions")
    pred_logits, proto = predictions
    detections = ops.non_max_suppression(
        pred_logits,
        conf,
        iou,
        multi_label=True,
        agnostic=agnostic,
        max_det=max_det,
    )
    if isinstance(proto, (list, tuple)) and len(proto) == 3:
        proto = proto[-1]
    return detections, proto


def _match_predictions(
    pred_classes: Tensor,
    true_classes: Tensor,
    iou: Tensor,
    iouv: Tensor,
) -> Tensor:
    if pred_classes.numel() == 0 or true_classes.numel() == 0:
        return torch.zeros((pred_classes.shape[0], iouv.numel()), dtype=torch.bool, device=pred_classes.device)

    correct = torch.zeros((pred_classes.shape[0], iouv.numel()), dtype=torch.bool, device=pred_classes.device)
    correct_class = true_classes[:, None] == pred_classes
    iou = iou * correct_class
    iou_np = iou.detach().cpu().numpy()
    thresholds = iouv.detach().cpu().tolist()
    for idx, thr in enumerate(thresholds):
        matches = np.argwhere(iou_np >= thr)
        if matches.size == 0:
            continue
        if matches.shape[0] > 1:
            match_scores = iou_np[matches[:, 0], matches[:, 1]]
            order = match_scores.argsort()[::-1]
            matches = matches[order]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1], idx] = True
    return correct


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Mapping[str, Any],
    aux_loss: Any,
    text_encoder: SimpleTextEncoder,
    *,
    disable_text: bool = False,
    disable_pbr: bool = False,
) -> Dict[str, float]:
    if SegmentMetrics is None or box_iou is None or mask_iou is None:  # pragma: no cover
        raise ImportError("Ultralytics metrics utilities are required for validation")

    model.eval()
    stats: Dict[str, List[Tensor]] = {"tp": [], "tp_m": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    metrics = SegmentMetrics()
    count = 0
    aux_sums: Dict[str, float] = {}
    aux_batches = 0
    train_cfg = cfg.get("train", {})
    conf_th = float(train_cfg.get("val_conf", 0.25))
    iou_th = float(train_cfg.get("val_iou", 0.7))
    max_det = int(train_cfg.get("val_max_det", 300))
    agnostic = bool(train_cfg.get("val_agnostic", False))

    use_text_flag = bool(train_cfg.get("use_text", True))
    use_pbr_flag = bool(train_cfg.get("use_pbr", True))

    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            pbr_tensor = batch.get("pbr")
            if isinstance(pbr_tensor, Tensor) and use_pbr_flag:
                pbr = pbr_tensor.to(device)
                if disable_pbr:
                    pbr = torch.zeros_like(pbr)
            else:
                pbr = None

            text_inputs = batch.get("context_txt", [])
            text = None
            if use_text_flag and text_encoder.dim > 0:
                text = text_encoder.batch(text_inputs).to(device)
                if disable_text:
                    text = torch.zeros_like(text)

            detections_batch = prepare_detection_batch(images, batch["labels"], device)
            outputs = model(images, pbr=pbr, text=text)
            detections, proto = _postprocess_predictions(
                outputs["pred"], conf=conf_th, iou=iou_th, max_det=max_det, agnostic=agnostic
            )

            for sample_idx in range(images.size(0)):
                mask = detections_batch["batch_idx"] == sample_idx
                gt_boxes = detections_batch["bboxes"][mask]
                gt_cls = detections_batch["cls"][mask].view(-1).to(device)
                if gt_boxes.numel():
                    gain = torch.tensor(
                        [images.shape[3], images.shape[2], images.shape[3], images.shape[2]],
                        device=device,
                        dtype=gt_boxes.dtype,
                    )
                    gt_xyxy = torch.empty_like(gt_boxes)
                    gt_xyxy[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
                    gt_xyxy[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
                    gt_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
                    gt_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2
                    gt_xyxy = torch.clamp(gt_xyxy, 0.0, 1.0) * gain
                else:
                    gt_xyxy = gt_boxes.new_zeros((0, 4))
                label_mapping = batch["labels"]
                target = {
                    key: values[sample_idx]
                    for key, values in label_mapping.items()
                    if len(values) > sample_idx
                }
                gt_masks = target.get("masks")
                if gt_masks is not None:
                    gt_masks = gt_masks.to(device)

                det = detections[sample_idx]
                proto_sample = proto[sample_idx]
                box_tp, mask_tp, conf, pred_cls = _match_sample(
                    det,
                    proto_sample,
                    gt_boxes=gt_xyxy,
                    gt_cls=gt_cls,
                    gt_masks=gt_masks,
                    image_shape=(images.shape[2], images.shape[3]),
                    iouv=iouv,
                )
                stats["tp"].append(box_tp.cpu())
                stats["tp_m"].append(mask_tp.cpu())
                stats["conf"].append(conf.detach().cpu())
                stats["pred_cls"].append(pred_cls.detach().cpu())
                stats["target_cls"].append(gt_cls.detach().cpu())
                stats["target_img"].append(gt_cls.detach().cpu().unique())
            count += images.size(0)

            if model.aux_head is not None and outputs.get("aux"):
                pbr_dim = len(cfg.get("context", {}).get("pbr_channels", []))
                aux_target = pbr if pbr is not None else torch.zeros(
                    (images.size(0), pbr_dim, images.size(2), images.size(3)),
                    device=device,
                )
                aux_loss_value, aux_logs = aux_loss(outputs["aux"], aux_target, mask=None)
                for key, value in aux_logs.items():
                    aux_sums[key] = aux_sums.get(key, 0.0) + float(value)
                aux_sums["loss/aux_eval"] = aux_sums.get("loss/aux_eval", 0.0) + float(aux_loss_value.detach().cpu())
                aux_batches += 1

    results: Dict[str, float] = {"count": float(count)}
    if stats["tp"]:
        stats_np = {key: torch.cat(values, dim=0).numpy() if values else np.zeros((0,)) for key, values in stats.items()}
        metrics.process(**stats_np)
        results.update({k: float(v) for k, v in metrics.results_dict.items()})
    if aux_batches:
        for key, value in aux_sums.items():
            results[key] = value / aux_batches
    return results


def _match_sample(
    detections: Tensor,
    proto: Tensor,
    *,
    gt_boxes: Tensor,
    gt_cls: Tensor,
    gt_masks: Optional[Tensor],
    image_shape: Sequence[int],
    iouv: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if ops is None or box_iou is None or mask_iou is None:  # pragma: no cover
        raise ImportError("Ultralytics must be installed to compute metrics")

    pred_cls = detections[:, 5] if detections.numel() else torch.zeros(0, device=gt_cls.device)
    conf = detections[:, 4] if detections.numel() else torch.zeros(0, device=gt_cls.device)
    if detections.numel():
        pred_boxes = detections[:, :4]
        box_tp = _match_predictions(pred_cls, gt_cls, box_iou(gt_boxes, pred_boxes), iouv)
    else:
        box_tp = torch.zeros((0, iouv.numel()), dtype=torch.bool, device=gt_cls.device)

    mask_tp = torch.zeros_like(box_tp)
    if detections.numel() and gt_masks is not None and gt_masks.numel():
        pred_masks = ops.process_mask(proto, detections[:, 6:], detections[:, :4], shape=image_shape)
        pred_masks = pred_masks.gt(0.5)
        masks_gt = gt_masks.float()
        if masks_gt.shape[1:] != pred_masks.shape[1:]:
            masks_gt = F.interpolate(masks_gt.unsqueeze(0), size=pred_masks.shape[1:], mode="bilinear", align_corners=False)[0]
        masks_gt = masks_gt.gt(0.5)
        mask_tp = _match_predictions(
            pred_cls,
            gt_cls,
            mask_iou(masks_gt.view(masks_gt.shape[0], -1), pred_masks.view(pred_masks.shape[0], -1)),
            iouv,
        )
    return box_tp, mask_tp, conf, pred_cls


__all__ = ["prepare_detection_batch", "evaluate"]
