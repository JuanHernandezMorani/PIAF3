"""Validation script for multimodal YOLO models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from src.data.multimodal_loader import build_dataloaders
from src.model.build import build_yolo11_multi
from src.train_utils.eval import evaluate
from src.train_utils.losses import AuxLossConfig, AuxPBRLoss
from src.train_utils.pbr import expand_pbr_channels
from src.train_utils.text import SimpleTextEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validación para YOLOv11 multimodal")
    parser.add_argument("--cfg", type=Path, default=Path("configs/train_multi.yaml"), help="Config de entrenamiento")
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"), help="Config de dataset")
    parser.add_argument("--weights", type=Path, default=Path("runs/multi/exp/last.pt"), help="Checkpoint a evaluar")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo CUDA o cpu")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Split a evaluar")
    parser.add_argument("--output", type=Path, default=Path("runs/multi/exp/metrics_val.json"), help="Archivo JSON de salida")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"La config {path} debe ser un mapping")
    return data


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)
    cfg.setdefault("data", {})["yaml"] = str(args.data)

    dataloaders = build_dataloaders(cfg)
    if args.split not in dataloaders:
        raise ValueError(f"Split {args.split} no disponible en dataloaders")
    loader = dataloaders[args.split]

    context_cfg = cfg.setdefault("context", {})
    train_cfg = cfg.get("train", {})
    use_pbr = bool(train_cfg.get("use_pbr", True))
    use_text = bool(train_cfg.get("use_text", True))

    dataset = loader.dataset
    if getattr(dataset, "use_pbr", False) and dataset.spec.pbr.order:
        context_cfg["pbr_channels"] = expand_pbr_channels(dataset.spec.pbr.order)
    elif not use_pbr:
        context_cfg["pbr_channels"] = []

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_cfg = dict(cfg)
    model_cfg["pretrained"] = None
    model = build_yolo11_multi(model_cfg).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    aux_config = AuxLossConfig(weights=cfg.get("loss", {}).get("pbr_weights", {}), use_ssim=cfg.get("loss", {}).get("use_ssim", False))
    aux_loss = AuxPBRLoss(model.aux_head, aux_config)
    text_dim = int(context_cfg.get("text_dim", 0))
    text_encoder = SimpleTextEncoder(text_dim, seed=cfg.get("seed", 0))

    metrics_main = evaluate(model, loader, device, cfg, aux_loss, text_encoder)
    print("Metrics (full):", json.dumps(metrics_main, indent=2))

    ablations: Dict[str, Dict[str, float]] = {}
    if use_pbr or use_text:
        ablations["rgb_only"] = evaluate(
            model,
            loader,
            device,
            cfg,
            aux_loss,
            text_encoder,
            disable_text=True,
            disable_pbr=True,
        )
        if use_pbr:
            ablations["rgb_pbr"] = evaluate(
                model,
                loader,
                device,
                cfg,
                aux_loss,
                text_encoder,
                disable_text=True,
                disable_pbr=False,
            )
        if use_text:
            ablations["rgb_text"] = evaluate(
                model,
                loader,
                device,
                cfg,
                aux_loss,
                text_encoder,
                disable_text=False,
                disable_pbr=True,
            )
        ablations["rgb_pbr_text"] = metrics_main
        print("Ablations:", json.dumps(ablations, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump({"main": metrics_main, "ablations": ablations}, handle, indent=2)


if __name__ == "__main__":
    main()
