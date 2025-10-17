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
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.multimodal_loader import build_dataloaders
from src.model.build import YoloMultiModel, build_yolo11_multi
from src.train_utils.eval import evaluate, prepare_detection_batch
from src.train_utils.losses import AuxLossConfig, AuxPBRLoss, DetectionSegLoss, MultiModalLoss
from src.train_utils.pbr import expand_pbr_channels
from src.train_utils.text import SimpleTextEncoder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""

    parser = argparse.ArgumentParser(description="Train the multimodal YOLOv11 model")
    parser.add_argument("--cfg", type=Path, default=Path("configs/train_multi.yaml"), help="Ruta al YAML de entrenamiento")
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"), help="Ruta al YAML de dataset")
    parser.add_argument("--pretrained", type=Path, default=None, help="Checkpoint YOLO base (yolo.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescribe epochs")
    parser.add_argument("--imgsz", type=int, default=None, help="Sobrescribe tamaño de imagen")
    parser.add_argument("--batch", type=int, default=None, help="Sobrescribe batch size")
    parser.add_argument("--workers", type=int, default=None, help="Sobrescribe número de workers")
    parser.add_argument("--pbr", dest="pbr", action="store_true", help="Forzar uso de PBR")
    parser.add_argument("--text", dest="text", action="store_true", help="Forzar uso de texto")
    parser.add_argument("--seed", type=int, default=None, help="Semilla global")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo CUDA o cpu")
    parser.add_argument("--resume", type=Path, default=None, help="Ruta a checkpoint para reanudar")
    parser.add_argument("--project", type=Path, default=None, help="Directorio raíz de logs")
    parser.add_argument("--name", type=str, default=None, help="Nombre de experimento")
    parser.add_argument("--val-only", action="store_true", help="Ejecuta solo validación")
    parser.set_defaults(pbr=None, text=None)
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
    data_section = cfg.setdefault("data", {})
    data_section["yaml"] = str(args.data)
    if args.imgsz is not None:
        data_section["imgsz"] = int(args.imgsz)
    if args.batch is not None:
        data_section["batch"] = int(args.batch)
    if args.workers is not None:
        data_section["workers"] = int(args.workers)

    train_section = cfg.setdefault("train", {})
    if args.epochs is not None:
        train_section["epochs"] = int(args.epochs)
    if args.pbr is not None:
        train_section["use_pbr"] = bool(args.pbr)
    if args.text is not None:
        train_section["use_text"] = bool(args.text)
    if args.resume is not None:
        train_section["resume"] = str(args.resume)

    logging_section = cfg.setdefault("logging", {})
    if args.project is not None:
        logging_section["project"] = str(args.project)
    if args.name is not None:
        logging_section["name"] = args.name

    if args.pretrained is not None:
        cfg["pretrained"] = str(args.pretrained)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.device is not None:
        cfg["device"] = args.device
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


def run_ablations(
    model: YoloMultiModel,
    loader: DataLoader,
    device: torch.device,
    cfg: Mapping[str, Any],
    aux_loss: AuxPBRLoss,
    text_encoder: SimpleTextEncoder,
) -> Dict[str, Dict[str, float]]:
    """Run optional ablation evaluations based on configuration flags."""

    ablations_cfg = cfg.get("train", {}).get("ablations", {}) or {}
    results: Dict[str, Dict[str, float]] = {}
    if not ablations_cfg:
        return results
    if ablations_cfg.get("use_text", False):
        results["no_text"] = evaluate(
            model,
            loader,
            device,
            cfg,
            aux_loss,
            text_encoder,
            disable_text=True,
            disable_pbr=False,
        )
    if ablations_cfg.get("use_pbr", False):
        results["no_pbr"] = evaluate(
            model,
            loader,
            device,
            cfg,
            aux_loss,
            text_encoder,
            disable_text=False,
            disable_pbr=True,
        )
    if ablations_cfg.get("use_text_pbr", False):
        results["no_text_pbr"] = evaluate(
            model,
            loader,
            device,
            cfg,
            aux_loss,
            text_encoder,
            disable_text=True,
            disable_pbr=True,
        )
    return results


def main() -> None:
    """Entrypoint for training the multimodal YOLO model."""

    args = parse_args()
    cfg = merge_cli(load_config(args.cfg), args)
    set_seed(cfg.get("seed"), cfg.get("deterministic", False))

    device_str = cfg.get("device")
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging_cfg = cfg.get("logging", {})
    project_dir = Path(logging_cfg.get("project", "runs/multi"))
    name = logging_cfg.get("name", "exp")
    save_dir = project_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir / "metrics.csv")

    train_cfg = cfg.get("train", {})
    use_text = bool(train_cfg.get("use_text", True))
    use_pbr = bool(train_cfg.get("use_pbr", True))

    dataloaders = build_dataloaders(cfg)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders.get("test")

    train_dataset = getattr(train_loader, "dataset", None)
    context_cfg = cfg.setdefault("context", {})
    if train_dataset is not None:
        pbr_order = getattr(getattr(train_dataset, "spec", None), "pbr", None)
        if pbr_order is not None and not context_cfg.get("pbr_channels"):
            context_cfg["pbr_channels"] = expand_pbr_channels(pbr_order.order)
    if not use_pbr:
        context_cfg["pbr_channels"] = []
    text_dim = int(context_cfg.get("text_dim", 0))
    text_encoder = SimpleTextEncoder(text_dim, seed=cfg.get("seed", 0))

    sample_batch = next(iter(train_loader))
    img_shape = tuple(sample_batch["img"].shape)
    pbr_tensor = sample_batch.get("pbr")
    pbr_shape = tuple(pbr_tensor.shape) if isinstance(pbr_tensor, torch.Tensor) else None
    label_info = sample_batch["labels"]
    boxes_per_sample = [int(t.size(0)) if isinstance(t, torch.Tensor) else 0 for t in label_info.get("boxes", [])]
    mask_list = label_info.get("masks", [])
    mask_counts = [int(m.sum().item()) if isinstance(m, torch.Tensor) else 0 for m in mask_list]
    print(
        f"[dataloader] img_shape={img_shape} pbr_shape={pbr_shape} "
        f"boxes/sample={boxes_per_sample} mask_pixels={mask_counts}"
    )
    del sample_batch

    if args.pretrained is not None:
        cfg["pretrained"] = str(args.pretrained)
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
        metrics = evaluate(model, val_loader, device, cfg, aux_loss, text_encoder)
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
        pbr_dim = len(context_cfg.get("pbr_channels", [])) if use_pbr else 0
        for step, batch in enumerate(progress):
            images = batch["img"].to(device)
            pbr_tensor = batch.get("pbr")
            pbr = pbr_tensor.to(device) if isinstance(pbr_tensor, torch.Tensor) and use_pbr else None
            text_inputs = batch.get("context_txt", [])
            text = (
                text_encoder.batch(text_inputs).to(device)
                if use_text and text_dim > 0
                else None
            )
            det_batch = prepare_detection_batch(images, batch["labels"], device)
            if pbr is not None:
                aux_target = pbr
            else:
                aux_target = torch.zeros(
                    (images.size(0), pbr_dim, images.size(2), images.size(3)),
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
                    aux_mask=None,
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
            val_metrics = evaluate(active_model, val_loader, device, cfg, aux_loss, text_encoder)
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

        ablation_results = run_ablations(model, val_loader, device, cfg, aux_loss, text_encoder)
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

