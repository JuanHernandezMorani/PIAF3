"""
Training loop skeleton para YOLOv11 multimodal con FiLM y heads auxiliares.
Codex Plus debe completar: integración real del modelo YOLOv11-seg, pérdidas y métricas.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from src.data.multimodal_loader import CLASSES, MultimodalYoloDataset, collate_fn
from src.model.model import MultimodalYoloStub


class _ContextAwareHead(torch.nn.Module):
    """Thin wrapper so exporters receive logits directly."""

    def __init__(
        self,
        model: MultimodalYoloStub,
        projector: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.projector = projector

    def forward(self, images: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        ctx = context
        if self.projector is not None:
            ctx = self.projector(context)
        outputs = self.model(images, ctx)
        return outputs["logits"]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=str)
    ap.add_argument("--split-train", default="splits/train.txt")
    ap.add_argument("--split-val", default="splits/val.txt")
    ap.add_argument("--imgsz", default=512, type=int)
    ap.add_argument("--batch", default=8, type=int)
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--weight-decay", default=1e-4, type=float)
    ap.add_argument("--num-workers", default=2, type=int)
    ap.add_argument("--device", default=None, type=str)
    ap.add_argument("--out-dir", default="runs/train", type=str)
    ap.add_argument("--use-metalness", action="store_true")
    ap.add_argument("--use-coordconv", action="store_true")
    ap.add_argument("--film-dim", default=64, type=int)
    ap.add_argument("--aux-on", action="store_true")
    return ap.parse_args()


def _make_save_dir(root: str, out_dir: str) -> Path:
    save_dir = Path(root).joinpath(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _infer_device(choice: Optional[str] = None) -> torch.device:
    if choice:
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_model(
    sample: Tuple[torch.Tensor, torch.Tensor, Mapping[str, object]],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[MultimodalYoloStub, Optional[torch.nn.Module]]:
    image, context, _ = sample
    in_channels = image.shape[0]
    context_dim = context.numel()
    film_dim = max(int(args.film_dim), 1)

    model = MultimodalYoloStub(
        in_channels=in_channels,
        num_classes=len(CLASSES),
        dim_context=film_dim,
        base_channels=32,
        neck_channels=128,
        aux_heads=args.aux_on,
    ).to(device)

    projector: Optional[torch.nn.Module]
    if film_dim != context_dim:
        projector = torch.nn.Linear(context_dim, film_dim).to(device)
    else:
        projector = None

    return model, projector


def _build_optimizer(
    model: MultimodalYoloStub,
    projector: Optional[torch.nn.Module],
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    params: Iterable[torch.nn.Parameter] = model.parameters()
    if projector is not None:
        params = list(params) + list(projector.parameters())
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


def _targets_to_masks(
    targets: List[Mapping[str, object]],
    num_classes: int,
    spatial_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    height, width = spatial_size
    batch_size = len(targets)
    masks = torch.zeros((batch_size, num_classes, height, width), device=device)

    for b, tgt in enumerate(targets):
        classes: Iterable[int] = tgt.get("classes", [])  # type: ignore[assignment]
        segments: Iterable[torch.Tensor] = tgt.get("segments", [])  # type: ignore[assignment]
        for cls_idx, segment in zip(classes, segments):
            cls_int = int(cls_idx)
            if cls_int < 0 or cls_int >= num_classes:
                continue
            if segment.numel() < 6:
                continue
            poly = segment.detach().cpu().numpy()
            if poly.shape[0] < 3:
                continue
            xy = [(float(x * width), float(y * height)) for x, y in poly]
            img = Image.new("L", (width, height), 0)
            drawer = ImageDraw.Draw(img)
            drawer.polygon(xy, outline=1, fill=1)
            mask_arr = torch.from_numpy(np.array(img, dtype=np.float32))
            masks[b, cls_int] = torch.maximum(masks[b, cls_int], mask_arr.to(device))

    return masks


def _prepare_aux_targets(
    target: Mapping[str, object],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    pbr_dict = target.get("pbr", {})
    out: Dict[str, torch.Tensor] = {}
    if isinstance(pbr_dict, Mapping):
        for key, value in pbr_dict.items():
            if torch.is_tensor(value):
                out[key] = value.to(device)
    return out


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels.sum()
    if positives <= 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    true_pos = np.cumsum(sorted_labels)
    false_pos = np.cumsum(1 - sorted_labels)

    precision = true_pos / np.maximum(true_pos + false_pos, 1e-8)
    recall = true_pos / np.maximum(positives, 1e-8)

    # Append sentinel values for interpolation.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def _compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = torch.sigmoid(logits.detach()).cpu()
    target_cpu = targets.detach().cpu()
    b, c, h, w = probs.shape
    preds = probs.view(b, c, -1)
    labels = target_cpu.view(b, c, -1)
    preds_np = preds.numpy()
    labels_np = labels.numpy()
    preds_flat = preds_np.transpose(1, 0, 2).reshape(c, -1)
    labels_flat = labels_np.transpose(1, 0, 2).reshape(c, -1)
    return preds_flat, labels_flat


def _evaluate(
    model: MultimodalYoloStub,
    projector: Optional[torch.nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> Dict[str, object]:
    model.eval()
    if projector is not None:
        projector.eval()

    running_loss = 0.0
    batches = 0
    preds_per_class: List[List[np.ndarray]] = [[] for _ in CLASSES]
    labels_per_class: List[List[np.ndarray]] = [[] for _ in CLASSES]

    with torch.no_grad():
        for images, context, batch_targets in dataloader:
            images = images.to(device)
            context = context.to(device)
            if projector is not None:
                context = projector(context)

            outputs = model(images, context)
            logits = outputs["logits"]
            masks = _targets_to_masks(
                batch_targets,
                num_classes=len(CLASSES),
                spatial_size=logits.shape[-2:],
                device=device,
            )
            loss = criterion(logits, masks)
            running_loss += float(loss.detach().cpu())
            batches += 1

            preds_flat, labels_flat = _compute_metrics(logits, masks)
            for idx, (preds, labels) in enumerate(zip(preds_flat, labels_flat)):
                preds_per_class[idx].append(preds)
                labels_per_class[idx].append(labels)

    per_class_ap: Dict[str, float] = {}
    ap_values: List[float] = []
    for idx, name in enumerate(CLASSES):
        if not preds_per_class[idx]:
            per_class_ap[name] = float("nan")
            continue
        preds = np.concatenate(preds_per_class[idx], axis=0)
        labels = np.concatenate(labels_per_class[idx], axis=0)
        ap = _average_precision(preds, labels)
        per_class_ap[name] = ap
        if not np.isnan(ap):
            ap_values.append(ap)

    mean_ap = float(np.mean(ap_values)) if ap_values else 0.0
    avg_loss = running_loss / max(batches, 1)

    return {
        "loss": avg_loss,
        "mAP": mean_ap,
        "per_class_ap": per_class_ap,
    }


def _train_one_epoch(
    model: MultimodalYoloStub,
    projector: Optional[torch.nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
) -> Dict[str, float]:
    model.train()
    if projector is not None:
        projector.train()

    running_loss = 0.0
    running_aux = 0.0
    batches = 0

    for images, context, batch_targets in dataloader:
        images = images.to(device)
        context = context.to(device)
        if projector is not None:
            context = projector(context)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, context)
        logits = outputs["logits"]
        masks = _targets_to_masks(
            batch_targets,
            num_classes=len(CLASSES),
            spatial_size=logits.shape[-2:],
            device=device,
        )
        loss = criterion(logits, masks)

        aux_loss_value = torch.zeros((), device=device)
        if "aux_outputs" in outputs and isinstance(outputs["aux_outputs"], Mapping):
            aux_targets_list = [
                _prepare_aux_targets(tgt, device) for tgt in batch_targets
            ]
            merged_aux_lists: Dict[str, List[torch.Tensor]] = {}
            for aux_dict in aux_targets_list:
                for key, value in aux_dict.items():
                    merged_aux_lists.setdefault(key, []).append(value)
            merged_aux_targets = {
                key: torch.stack(values, dim=0) for key, values in merged_aux_lists.items()
            }
            aux_outputs = outputs["aux_outputs"]
            if isinstance(aux_outputs, Mapping):
                aux_loss_value, _ = model.compute_auxiliary_loss(
                    aux_outputs,
                    merged_aux_targets,
                )
                loss = loss + aux_loss_value

        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu())
        running_aux += float(aux_loss_value.detach().cpu())
        batches += 1

    return {
        "loss": running_loss / max(batches, 1),
        "aux_loss": running_aux / max(batches, 1),
    }


def _save_checkpoint(
    save_path: Path,
    model: MultimodalYoloStub,
    projector: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_map: float,
    metrics: Mapping[str, object],
) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_mAP": best_map,
        "metrics": dict(metrics),
        "context_projector": projector.state_dict() if projector is not None else None,
    }
    torch.save(state, save_path)


def _export_artifacts(
    model: MultimodalYoloStub,
    projector: Optional[torch.nn.Module],
    device: torch.device,
    save_dir: Path,
    image_shape: Tuple[int, int, int],
    context_dim: int,
) -> None:
    wrapper = _ContextAwareHead(model, projector).to(device)
    wrapper.eval()

    dummy_input = torch.randn((1, *image_shape), device=device)
    dummy_context = torch.randn((1, context_dim), device=device)

    onnx_path = save_dir / "model_best.onnx"
    torchscript_path = save_dir / "model_best.ts"

    try:
        torch.onnx.export(
            wrapper,
            (dummy_input, dummy_context),
            onnx_path.as_posix(),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["images", "context"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "context": {0: "batch"}, "logits": {0: "batch"}},
        )
    except Exception as exc:  # noqa: BLE001 - surface exporter issues to the console
        print(f"[export] ONNX export failed: {exc}")

    try:
        scripted = torch.jit.trace(wrapper, (dummy_input, dummy_context))
        scripted.save(torchscript_path.as_posix())
    except RuntimeError as exc:
        print(f"[export] TorchScript export failed: {exc}")

def main():
    args = parse_args()
    device = _infer_device(args.device)
    save_dir = _make_save_dir(args.root, args.out_dir)

    train_ds = MultimodalYoloDataset(
        args.root,
        args.split_train,
        imgsz=args.imgsz,
        use_metalness=args.use_metalness,
        use_coordconv=args.use_coordconv,
        augment=True,
        pbr_dropout_prob=0.2,
    )
    val_ds = MultimodalYoloDataset(
        args.root,
        args.split_val,
        imgsz=args.imgsz,
        use_metalness=args.use_metalness,
        use_coordconv=args.use_coordconv,
        augment=False,
        pbr_dropout_prob=0.0,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty")

    sample = train_ds[0]
    image_shape = sample[0].shape
    context_dim = sample[1].numel()

    model, projector = _prepare_model(sample, args, device)
    optimizer = _build_optimizer(model, projector, args)
    scheduler = _build_scheduler(optimizer, args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_map = 0.0
    best_epoch = -1

    for epoch in range(args.epochs):
        epoch_num = epoch + 1
        train_stats = _train_one_epoch(
            model,
            projector,
            train_dl,
            device,
            optimizer,
            criterion,
        )

        val_metrics = _evaluate(model, projector, val_dl, device, criterion)
        scheduler.step()

        current_map = float(val_metrics.get("mAP", 0.0))
        if np.isnan(current_map):
            current_map = 0.0
        if current_map >= best_map:
            best_map = current_map
            best_epoch = epoch_num
            _save_checkpoint(
                save_dir / "checkpoint_best.pt",
                model,
                projector,
                optimizer,
                scheduler,
                epoch_num,
                best_map,
                val_metrics,
            )

        _save_checkpoint(
            save_dir / "checkpoint_last.pt",
            model,
            projector,
            optimizer,
            scheduler,
            epoch_num,
            best_map,
            val_metrics,
        )

        val_loss = float(val_metrics.get("loss", 0.0))
        val_map = float(val_metrics.get("mAP", 0.0))
        log_msg = (
            f"[epoch {epoch_num:03d}/{args.epochs}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"aux_loss={train_stats['aux_loss']:.4f} "
            f"val_loss={val_loss:.4f} "
            f"mAP={val_map:.4f}"
        )
        print(log_msg)

    print(
        f"[train] Finished training. Best mAP={best_map:.4f} at epoch {best_epoch if best_epoch > 0 else 'N/A'}."
    )

    best_ckpt = save_dir / "checkpoint_best.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        if projector is not None and state.get("context_projector") is not None:
            projector.load_state_dict(state["context_projector"])

    _export_artifacts(
        model,
        projector,
        device,
        save_dir,
        image_shape,
        context_dim,
    )

    print("[train] DONE")

if __name__ == "__main__":
    main()
