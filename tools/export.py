"""Export utilities for multimodal YOLO models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from src.data.multimodal_loader import build_dataloaders
from src.model.build import build_yolo11_multi
from src.train_utils.pbr import expand_pbr_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exportar YOLOv11 multimodal")
    parser.add_argument("--cfg", type=Path, default=Path("configs/train_multi.yaml"), help="Config de entrenamiento")
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"), help="Config de dataset")
    parser.add_argument("--weights", type=Path, required=True, help="Checkpoint entrenado")
    parser.add_argument("--out", type=Path, required=True, help="Archivo .pt destino")
    parser.add_argument("--onnx", type=Path, default=None, help="Ruta opcional para exportar ONNX")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo para el export")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("La configuración debe ser un mapping")
    return data


def build_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_cfg = dict(cfg)
    model_cfg["pretrained"] = None
    model = build_yolo11_multi(model_cfg).to(device)
    return model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg)
    cfg.setdefault("data", {})["yaml"] = str(args.data)

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)

    checkpoint = torch.load(args.weights, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg}, args.out)
    print(f"Modelo exportado a {args.out}")

    if args.onnx is not None:
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        dataloaders = build_dataloaders(cfg)
        dataset = dataloaders["train"].dataset
        context_cfg = cfg.setdefault("context", {})
        if getattr(dataset, "use_pbr", False) and dataset.spec.pbr.order:
            context_cfg["pbr_channels"] = expand_pbr_channels(dataset.spec.pbr.order)
        elif not getattr(dataset, "use_pbr", False):
            context_cfg["pbr_channels"] = []
        imgsz = int(cfg.get("data", {}).get("imgsz", 640))
        batch_size = 1
        img = torch.zeros((batch_size, 3, imgsz, imgsz), device=device)
        pbr_channels = len(context_cfg.get("pbr_channels", []))
        pbr = torch.zeros((batch_size, pbr_channels, imgsz, imgsz), device=device) if pbr_channels > 0 else None
        text_dim = int(context_cfg.get("text_dim", 0))
        text = torch.zeros((batch_size, text_dim), device=device) if text_dim > 0 else None

        input_names = ["images"]
        dynamic_axes: Dict[str, Dict[int, str]] = {"images": {0: "batch"}}
        kwargs: Dict[str, torch.Tensor] = {}
        if pbr is not None:
            input_names.append("pbr")
            dynamic_axes["pbr"] = {0: "batch"}
            kwargs["pbr"] = pbr
        if text is not None:
            input_names.append("text")
            dynamic_axes["text"] = {0: "batch"}
            kwargs["text"] = text

        torch.onnx.export(
            model,
            args=(img,),
            f=args.onnx,
            input_names=input_names,
            output_names=["pred", "aux"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            kwargs=kwargs,
        )
        print(f"Modelo ONNX exportado a {args.onnx}")


if __name__ == "__main__":
    main()
