"""Inference utilities for the multimodal YOLOv11 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from tqdm.auto import tqdm

from src.data.multimodal_loader import AugmentationConfig, MultiModalYOLODataset, multimodal_collate
from src.model.build import build_yolo11_multi
from src.train_utils.pbr import expand_pbr_channels
from src.train_utils.text import SimpleTextEncoder

try:  # pragma: no cover - optional at test time
    from ultralytics.utils import ops
except Exception:  # pragma: no cover
    ops = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal YOLOv11 inference")
    parser.add_argument("--weights", type=Path, required=True, help="Checkpoint a evaluar")
    parser.add_argument("--cfg", type=Path, default=Path("configs/train_multi.yaml"), help="YAML de configuración del modelo")
    parser.add_argument("--data", type=Path, default=Path("configs/data.yaml"), help="YAML del dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Split a evaluar")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de entrada")
    parser.add_argument("--batch", type=int, default=1, help="Batch size de inferencia")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo CUDA o cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    parser.add_argument("--iou", type=float, default=0.7, help="Umbral IoU para NMS")
    parser.add_argument("--pbr", dest="pbr", action="store_true", help="Forzar uso de PBR")
    parser.add_argument("--text", dest="text", action="store_true", help="Forzar uso de texto")
    parser.add_argument("--output", type=Path, default=Path("runs/multi/infer"), help="Directorio de resultados")
    parser.set_defaults(pbr=None, text=None)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"El archivo YAML {path} debe contener un mapping")
    return data


def build_dataset(args: argparse.Namespace, cfg: Dict[str, Any]) -> MultiModalYOLODataset:
    train_cfg = cfg.get("train", {})
    use_text = bool(train_cfg.get("use_text", True)) if args.text is None else bool(args.text)
    use_pbr = bool(train_cfg.get("use_pbr", True)) if args.pbr is None else bool(args.pbr)
    augment = AugmentationConfig(mosaic=False, hsv_jitter=0.0, block_mix_prob=0.0)
    dataset = MultiModalYOLODataset(
        data_cfg=args.data,
        split=args.split,
        imgsz=args.imgsz,
        augmentation=augment,
        seed=cfg.get("seed"),
        use_text=use_text,
        use_pbr=use_pbr,
    )
    return dataset


def load_model(cfg: Dict[str, Any], weights: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = dict(cfg)
    model_cfg["pretrained"] = None
    model = build_yolo11_multi(model_cfg).to(device)
    checkpoint = torch.load(weights, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def run_inference() -> None:
    args = parse_args()
    cfg = load_yaml(args.cfg)
    cfg = dict(cfg)
    cfg.setdefault("data", {})["yaml"] = str(args.data)

    dataset = build_dataset(args, cfg)
    context_cfg = cfg.setdefault("context", {})
    if dataset.use_pbr and dataset.spec.pbr.order:
        context_cfg["pbr_channels"] = expand_pbr_channels(dataset.spec.pbr.order)
    elif not dataset.use_pbr:
        context_cfg["pbr_channels"] = []

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.weights, device)

    text_dim = int(context_cfg.get("text_dim", 0))
    text_encoder = SimpleTextEncoder(text_dim, seed=cfg.get("seed", 0))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=multimodal_collate,
    )

    if ops is None:  # pragma: no cover
        raise ImportError("Ultralytics no está instalado: requerido para NMS")

    args.output.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Inferencia"):
        images = batch["img"].to(device)
        pbr_tensor = batch.get("pbr")
        if isinstance(pbr_tensor, torch.Tensor) and dataset.use_pbr:
            pbr = pbr_tensor.to(device)
        else:
            pbr = None
        text_inputs = batch.get("context_txt", [])
        if dataset.use_text and text_dim > 0:
            text_tensor = text_encoder.batch(text_inputs).to(device)
        else:
            text_tensor = None

        with torch.no_grad():
            outputs = model(images, pbr=pbr, text=text_tensor)
        predictions = outputs.get("pred")
        proto = predictions[1] if isinstance(predictions, (list, tuple)) else None
        logits = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        dets = ops.non_max_suppression(logits, args.conf, args.iou, multi_label=True, agnostic=False)

        for sample_idx, det in enumerate(dets):
            meta = batch["meta"][sample_idx]
            orig_h, orig_w = meta.get("orig_size", (args.imgsz, args.imgsz))
            boxes: List[Dict[str, Any]] = []
            masks: List[List[List[float]]] = []

            if det is not None and det.numel():
                det_cpu = det.to("cpu")
                xyxy = det_cpu[:, :4]
                conf = det_cpu[:, 4].tolist()
                cls = det_cpu[:, 5].tolist()
                scale_w = orig_w / images.shape[3]
                scale_h = orig_h / images.shape[2]
                for box_tensor, score, cls_idx in zip(xyxy, conf, cls):
                    x0, y0, x1, y1 = box_tensor.tolist()
                    boxes.append(
                        {
                            "xyxy": [x0 * scale_w, y0 * scale_h, x1 * scale_w, y1 * scale_h],
                            "confidence": score,
                            "class_id": int(cls_idx),
                        }
                    )
                if proto is not None and det_cpu.shape[1] > 6:
                    proto_sample = proto[sample_idx]
                    mask_coeffs = det_cpu[:, 6:]
                    masks_tensor = ops.process_mask(proto_sample, mask_coeffs, xyxy, shape=(orig_h, orig_w))
                    for mask in masks_tensor:
                        contour: List[List[float]] = []
                        ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
                        contour = [[float(x), float(y)] for x, y in zip(xs.tolist(), ys.tolist())]
                        masks.append(contour)

            result = {"path": meta.get("path"), "boxes": boxes, "masks": masks}
            stem = Path(meta.get("path", f"sample_{sample_idx}"))
            out_file = args.output / f"{stem.stem}.json"
            with out_file.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_inference()
