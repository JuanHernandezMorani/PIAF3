"""Training script for the multimodal YOLOv11 model."""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.multimodal_loader import (
    AugmentationConfig,
    MultiModalYOLODataset,
    multimodal_collate,
)
from src.model.build import YoloMultiModel, build_yolo11_multi
from src.train_utils.losses import AuxLossConfig, AuxPBRLoss, DetectionSegLoss, MultiModalLoss

try:  # pragma: no cover - ultralytics is an optional runtime dependency for testing
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
except Exception:  # pragma: no cover
    ops = None
    SegmentMetrics = None
    box_iou = None
    mask_iou = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""

    parser = argparse.ArgumentParser(description="Train the multimodal YOLO model")
    parser.add_argument("--config", type=Path, required=True, help="Path to the training YAML configuration")
    parser.add_argument("--pretrained", type=str, required=True, help="Path to the pretrained YOLO checkpoint")
    parser.add_argument("--project", type=Path, default=None, help="Override the project directory")
    parser.add_argument("--name", type=str, default=None, help="Override the experiment name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--deterministic", type=lambda v: v.lower() == "true", default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--val-only", action="store_true", help="Run validation only")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Configuration file must contain a mapping")
    return data


def merge_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge command-line overrides into the configuration dictionary."""

    cfg = dict(cfg)
    cfg["pretrained"] = args.pretrained
    cfg.setdefault("save", {})
    if args.project is not None:
        cfg["save"]["project"] = str(args.project)
    if args.name is not None:
        cfg["save"]["name"] = args.name
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.deterministic is not None:
        cfg["deterministic"] = bool(args.deterministic)
    if args.resume is not None:
        cfg.setdefault("train", {})["resume"] = str(args.resume)
    return cfg


def set_seed(seed: Optional[int], deterministic: bool = False) -> None:
    """Set seeds for reproducible experiments."""

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_dataloader(
    cfg: Mapping[str, Any],
    *,
    split: str,
    use_text: bool,
    use_pbr: bool,
    seed: Optional[int],
) -> DataLoader:
    """Instantiate a dataloader for the requested split."""

    data_cfg = cfg.get("data", {})
    imgsz = int(data_cfg.get("imgsz", 640))
    text_dim = int(cfg.get("context", {}).get("text_dim", 0))
    mosaic = bool(data_cfg.get("mosaic", False)) if split == "train" else False
    aug_cfg = AugmentationConfig(
        mosaic=mosaic,
        mosaic_prob=float(data_cfg.get("mosaic_prob", 0.5)),
        hsv_jitter=float(data_cfg.get("hsv_jitter", 0.05)),
        block_mix_prob=float(data_cfg.get("block_mix_prob", 0.0)),
    )
    dataset_path_key = "train" if split == "train" else "val"
    data_yaml = data_cfg.get(dataset_path_key)
    if data_yaml is None:
        raise ValueError(f"Missing data.{dataset_path_key} entry in configuration")
    dataset = MultiModalYOLODataset(
        data_cfg=Path(data_yaml),
        split="train" if split == "train" else "val",
        imgsz=imgsz,
        text_dim=text_dim,
        augmentation=aug_cfg if split == "train" else AugmentationConfig(),
        seed=seed,
        use_text=use_text,
        use_pbr=use_pbr,
    )
    workers = int(data_cfg.get("workers", 4))
    batch_size = int(data_cfg.get("batch", 8))
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        collate_fn=multimodal_collate,
    )


def freeze_backbone(model: YoloMultiModel, freeze: bool) -> None:
    """Enable or disable gradient computation for the YOLO backbone."""

    for name, param in model.model.named_parameters():
        param.requires_grad = not freeze
    for block in model.film_blocks:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.context_adapter.parameters():
        param.requires_grad = True
    if model.aux_head is not None:
        for param in model.aux_head.parameters():
            param.requires_grad = True


def configure_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """Create the AdamW optimizer with configuration defaults."""

    train_cfg = cfg.get("train", {})
    lr = float(train_cfg.get("lr", 2e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Construct a cosine scheduler with linear warmup."""

    warmup_steps = max(0, warmup_steps)
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEMA:
    """Maintain an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            ema_params = dict(self.ema.named_parameters())
            model_params = dict(model.named_parameters())
            for name, param in model_params.items():
                if name in ema_params:
                    ema_params[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.ema.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.ema.load_state_dict(state)


class CSVLogger:
    """Simple CSV logger for per-epoch metrics."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = path.open("w", newline="")
        self.writer: Optional[csv.DictWriter] = None

    def log(self, data: Mapping[str, Any]) -> None:
        if self.writer is None:
            fieldnames = sorted(data.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
            self.writer.writeheader()
        self.writer.writerow({k: data.get(k, "") for k in self.writer.fieldnames})
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def prepare_detection_batch(
    images: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    device: Optional[torch.device],
) -> Dict[str, torch.Tensor]:
    """Convert YOLO targets into the format expected by Ultralytics loss functions."""

    cls_tensors: List[torch.Tensor] = []
    box_tensors: List[torch.Tensor] = []
    batch_indices: List[torch.Tensor] = []
    mask_tensors: List[torch.Tensor] = []
    for batch_idx, target in enumerate(targets):
        boxes = target.get("boxes")
        classes = target.get("classes")
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
    batch_idx_tensor = torch.cat(batch_indices, dim=0) if batch_indices else torch.zeros((0,), device=device_out, dtype=torch.long)
    mask_shape = (0, images.shape[2], images.shape[3])
    masks = torch.cat(mask_tensors, dim=0) if mask_tensors else torch.zeros(mask_shape, device=device_out)
    return {
        "img": images,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx_tensor,
        "masks": masks,
    }


def save_checkpoint(
    path: Path,
    model: YoloMultiModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_metric: float,
    ema: Optional[ModelEMA] = None,
) -> None:
    """Persist a training checkpoint to disk."""

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if ema is not None:
        state["ema"] = ema.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: YoloMultiModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
    ema: Optional[ModelEMA] = None,
) -> tuple[int, float]:
    """Load a checkpoint returning the starting epoch and best metric."""

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    epoch = int(checkpoint.get("epoch", 0))
    best_metric = float(checkpoint.get("best_metric", 0.0))
    return epoch, best_metric


def _postprocess_predictions(
    predictions: Any,
    *,
    conf: float,
    iou: float,
    max_det: int,
    agnostic: bool,
) -> tuple[List[torch.Tensor], torch.Tensor]:
    """Run non-max suppression on raw model outputs."""

    if ops is None:  # pragma: no cover - handled during runtime
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
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    iou: torch.Tensor,
    iouv: torch.Tensor,
) -> torch.Tensor:
    """Match predictions against ground truth across IoU thresholds."""

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


def _compute_metrics_for_sample(
    detections: torch.Tensor,
    proto: torch.Tensor,
    *,
    gt_boxes: torch.Tensor,
    gt_cls: torch.Tensor,
    gt_masks: Optional[torch.Tensor],
    image_shape: Sequence[int],
    iouv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute box and mask matches for a single validation sample."""

    if SegmentMetrics is None or box_iou is None or mask_iou is None:  # pragma: no cover
        raise ImportError("Ultralytics metrics utilities are required for validation")

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


def evaluate(
    model: YoloMultiModel,
    loader: DataLoader,
    device: torch.device,
    cfg: Mapping[str, Any],
    aux_loss: AuxPBRLoss,
    *,
    disable_text: bool = False,
    disable_pbr: bool = False,
) -> Dict[str, float]:
    """Run validation and return detection/segmentation metrics along with auxiliary losses."""

    if SegmentMetrics is None:  # pragma: no cover
        raise ImportError("Ultralytics must be installed to compute validation metrics")

    model.eval()
    stats: Dict[str, List[torch.Tensor]] = {"tp": [], "tp_m": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    metrics = SegmentMetrics()
    count = 0
    aux_sums: Dict[str, float] = defaultdict(float)
    aux_batches = 0
    train_cfg = cfg.get("train", {})
    conf_th = float(train_cfg.get("val_conf", 0.25))
    iou_th = float(train_cfg.get("val_iou", 0.7))
    max_det = int(train_cfg.get("val_max_det", 300))
    agnostic = bool(train_cfg.get("val_agnostic", False))

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            pbr = batch.get("pbr")
            pbr_mask = batch.get("pbr_mask")
            text = batch.get("text")
            if pbr is not None:
                pbr = pbr.to(device)
                if disable_pbr:
                    pbr = torch.zeros_like(pbr)
            if pbr_mask is not None:
                pbr_mask = pbr_mask.to(device)
            if text is not None:
                text = text.to(device)
                if disable_text:
                    text = torch.zeros_like(text)

            detections_batch = prepare_detection_batch(images, batch["targets"], device)
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
                target = batch["targets"][sample_idx]
                gt_masks = target.get("masks")
                if gt_masks is not None:
                    gt_masks = gt_masks.to(device)

                det = detections[sample_idx]
                proto_sample = proto[sample_idx]
                box_tp, mask_tp, conf, pred_cls = _compute_metrics_for_sample(
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
                aux_target = pbr if pbr is not None else torch.zeros(
                    (images.size(0), model.context_adapter.in_channels_pbr, images.size(2), images.size(3)),
                    device=device,
                )
                aux_loss_value, aux_logs = aux_loss(outputs["aux"], aux_target, mask=pbr_mask)
                for key, value in aux_logs.items():
                    aux_sums[key] += float(value)
                aux_sums["loss/aux_eval"] += float(aux_loss_value.detach().cpu())
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


def run_ablations(
    model: YoloMultiModel,
    loader: DataLoader,
    device: torch.device,
    cfg: Mapping[str, Any],
    aux_loss: AuxPBRLoss,
) -> Dict[str, Dict[str, float]]:
    """Run optional ablation evaluations based on configuration flags."""

    ablations_cfg = cfg.get("train", {}).get("ablations", {}) or {}
    results: Dict[str, Dict[str, float]] = {}
    if not ablations_cfg:
        return results
    if ablations_cfg.get("use_text", False):
        results["no_text"] = evaluate(model, loader, device, cfg, aux_loss, disable_text=True, disable_pbr=False)
    if ablations_cfg.get("use_pbr", False):
        results["no_pbr"] = evaluate(model, loader, device, cfg, aux_loss, disable_text=False, disable_pbr=True)
    if ablations_cfg.get("use_text_pbr", False):
        results["no_text_pbr"] = evaluate(model, loader, device, cfg, aux_loss, disable_text=True, disable_pbr=True)
    return results


def main() -> None:
    """Entrypoint for training the multimodal YOLO model."""

    args = parse_args()
    cfg = merge_cli(load_config(args.config), args)
    set_seed(cfg.get("seed"), cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_cfg = cfg.get("save", {})
    project_dir = Path(save_cfg.get("project", "runs/multi"))
    name = save_cfg.get("name", "exp")
    save_dir = project_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir / "metrics.csv")

    train_cfg = cfg.get("train", {})
    use_text = bool(train_cfg.get("use_text", True))
    use_pbr = bool(train_cfg.get("use_pbr", True))

    train_loader = build_dataloader(cfg, split="train", use_text=use_text, use_pbr=use_pbr, seed=cfg.get("seed"))
    val_loader = build_dataloader(cfg, split="val", use_text=use_text, use_pbr=use_pbr, seed=cfg.get("seed"))

    cfg["pretrained"] = args.pretrained
    model = build_yolo11_multi(cfg).to(device)

    detection_loss = DetectionSegLoss(model.model)
    aux_config = AuxLossConfig(weights=cfg.get("loss", {}).get("pbr_weights", {}), use_ssim=cfg.get("loss", {}).get("use_ssim", False))
    aux_loss = AuxPBRLoss(model.aux_head, aux_config)
    multimodal_loss = MultiModalLoss(
        detection_loss,
        aux_loss,
        aux_lambda=float(cfg.get("loss", {}).get("aux_lambda", 0.0)),
        warmup_iters=int(cfg.get("loss", {}).get("aux_warmup_iters", 0)),
    )

    freeze_epochs = int(train_cfg.get("freeze_epochs", 0))
    if freeze_epochs > 0:
        freeze_backbone(model, True)

    optimizer = configure_optimizer(model, cfg)
    epochs = int(train_cfg.get("epochs", 1))
    total_steps = len(train_loader) * max(1, epochs)
    scheduler = build_scheduler(optimizer, total_steps, int(train_cfg.get("warmup_steps", 1000)))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    ema: Optional[ModelEMA] = ModelEMA(model) if train_cfg.get("ema", False) else None

    start_epoch = 0
    best_metric = 0.0
    resume_path = train_cfg.get("resume")
    if resume_path:
        checkpoint_path = Path(resume_path)
        if checkpoint_path.exists():
            start_epoch, best_metric = load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, ema)

    if args.val_only:
        metrics = evaluate(model, val_loader, device, cfg, aux_loss)
        print("Validation metrics:", metrics)
        logger.close()
        return

    grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))
    clip_grad = float(train_cfg.get("clip_grad", 0.0))
    val_interval = max(1, int(train_cfg.get("val_interval", 1)))

    total_images = 0
    total_batches = 0
    start_time = time.perf_counter()
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, epochs):
        model.train()
        if epoch == freeze_epochs:
            freeze_backbone(model, False)
            for group in optimizer.param_groups:
                group["lr"] *= 0.5

        epoch_metrics: Dict[str, float] = {}
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress):
            images = batch["image"].to(device)
            pbr = batch.get("pbr")
            pbr_mask = batch.get("pbr_mask")
            text = batch.get("text")
            if pbr is not None:
                pbr = pbr.to(device)
            if pbr_mask is not None:
                pbr_mask = pbr_mask.to(device)
            if text is not None:
                text = text.to(device)
            det_batch = prepare_detection_batch(images, batch["targets"], device)
            aux_target = pbr if pbr is not None else torch.zeros(
                (images.size(0), model.context_adapter.in_channels_pbr, images.size(2), images.size(3)),
                device=device,
            )

            with autocast(enabled=scaler.is_enabled()):
                outputs = model(images, pbr=pbr, text=text)
                aux_preds = outputs.get("aux", {})
                loss, metrics = multimodal_loss(
                    outputs.get("pred"),
                    det_batch,
                    aux_preds=aux_preds,
                    aux_target=aux_target,
                    aux_mask=pbr_mask,
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if ema is not None:
                    ema.update(model)

            for key, value in metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + float(value)
            progress.set_postfix({"loss": float(loss.detach().cpu())})
            total_images += images.size(0)
            total_batches += 1

        val_metrics: Dict[str, float] = {}
        if (epoch + 1) % val_interval == 0:
            active_model = model
            if ema is not None:
                current_state = model.state_dict()
                model.load_state_dict(ema.state_dict(), strict=False)
                active_model = model
            val_metrics = evaluate(active_model, val_loader, device, cfg, aux_loss)
            if ema is not None:
                model.load_state_dict(current_state)

        averaged_metrics = {k: v / max(1, len(train_loader)) for k, v in epoch_metrics.items()}
        log_row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **averaged_metrics, **val_metrics}
        logger.log(log_row)

        metric_key = "metrics/mAP50-95(B)"
        metric = val_metrics.get(metric_key, val_metrics.get("metrics/mAP50(B)", val_metrics.get("count", 0.0)))
        if metric >= best_metric:
            best_metric = metric
            save_checkpoint(save_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_metric, ema)
        save_checkpoint(save_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_metric, ema)

        ablation_results = run_ablations(model, val_loader, device, cfg, aux_loss)
        for name, metrics_dict in ablation_results.items():
            print(f"Ablation {name}: {metrics_dict}")

    if ema is not None:
        torch.save(ema.state_dict(), save_dir / "ema.pt")
    best_path = save_dir / "best.pt"
    if best_path.exists():
        shutil.copyfile(best_path, save_dir / "yolo-multi.pt")
    logger.close()

    elapsed = time.perf_counter() - start_time
    time_per_image = elapsed / max(1, total_images)
    print(
        f"Imágenes entrenadas: {total_images}, Batches: {total_batches}, "
        f"Tiempo/imagen: {time_per_image:.4f}s, Tiempo total: {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()

