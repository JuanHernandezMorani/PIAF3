"""Inference helpers for the multimodal YOLOv11 skeleton."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data.multimodal_loader import add_coordconv, load_img


def _infer_device(device_like: Optional[str | torch.device]) -> torch.device:
    """Resolve a device string/handle to a :class:`torch.device`."""

    if device_like is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device_like, torch.device):
        return device_like
    return torch.device(device_like)


@dataclass
class _PredictorConfig:
    """Typed view over the configuration dictionary."""

    imgsz: int = 512
    use_metalness: bool = False
    use_coordconv: bool = False
    mask_threshold: float = 0.5
    score_threshold: float = 0.3
    device: Optional[str | torch.device] = None
    projector: Optional[torch.nn.Module] = None


class MultimodalPredictor:
    """Utility wrapper that mirrors the dataset preprocessing for inference."""

    def __init__(self, model: torch.nn.Module, cfg: Optional[Mapping[str, Any]] = None):
        self.cfg = _PredictorConfig(**(cfg or {}))
        self.device = _infer_device(self.cfg.device)
        self.model = model.to(self.device).eval()

        self.projector: Optional[torch.nn.Module]
        self.projector = self.cfg.projector
        if self.projector is not None:
            self.projector = self.projector.to(self.device).eval()

    # ------------------------------------------------------------------
    # Loading helpers
    def _paths(self, root: str, name: str) -> Dict[str, str]:
        base = os.path.join(root, "data")
        return {
            "rgb": os.path.join(base, "images", f"{name}.png"),
            "n": os.path.join(base, "maps", "normal", f"{name}_n.png"),
            "r": os.path.join(base, "maps", "roughness", f"{name}_r.png"),
            "s": os.path.join(base, "maps", "specular", f"{name}_s.png"),
            "e": os.path.join(base, "maps", "emissive", f"{name}_e.png"),
            "m": os.path.join(base, "maps", "metalness", f"{name}_m.png"),
            "meta": os.path.join(base, "meta", f"{name}.json"),
        }

    def _load_modalities(self, paths: Mapping[str, str]) -> np.ndarray:
        rgb = load_img(paths["rgb"], "RGB").astype(np.float32) / 255.0
        nrm = load_img(paths["n"], "RGB").astype(np.float32) / 255.0
        rough = load_img(paths["r"], "L").astype(np.float32) / 255.0
        spec = load_img(paths["s"], "L").astype(np.float32) / 255.0
        emiss = load_img(paths["e"], "L").astype(np.float32) / 255.0
        rough = rough[..., None]
        spec = spec[..., None]
        emiss = emiss[..., None]

        chans: List[np.ndarray] = [rgb, nrm, rough, spec, emiss]

        if self.cfg.use_metalness and os.path.exists(paths["m"]):
            metal = load_img(paths["m"], "L").astype(np.float32) / 255.0
            metal = metal[..., None]
            chans.append(metal)

        h, w = rgb.shape[:2]
        if self.cfg.use_coordconv:
            xx, yy = add_coordconv(h, w)
            chans.extend([xx, yy])

        stacked = np.concatenate(chans, axis=-1)
        return stacked

    def _load_context(self, meta_path: str) -> np.ndarray:
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}

        lum = float(meta.get("luminance_lab", meta.get("lum", 50.0)))
        sat = float(meta.get("saturation", meta.get("sat", 0.5)))
        contrast = float(meta.get("contrast", 0.5))

        def _to_unit(value: float, low: float, high: float) -> float:
            if high == low:
                return 0.0
            scaled = (value - low) / (high - low)
            return float(np.clip(scaled, 0.0, 1.0))

        lum_unit = _to_unit(lum, 0.0, 100.0)
        sat_unit = _to_unit(sat, 0.0, 1.0)
        contrast_unit = _to_unit(contrast, 0.0, 100.0)

        vec = [lum_unit, sat_unit, contrast_unit]

        colors = meta.get("dominant_colors", [])[:5]
        flattened: List[float] = []
        for color in colors:
            if isinstance(color, list) and len(color) == 3:
                flattened.extend([float(c) / 255.0 for c in color])
        while len(flattened) < 15:
            flattened.append(0.0)
        vec.extend(flattened)

        arr = np.asarray(vec, dtype=np.float32)
        arr = arr * 2.0 - 1.0
        return arr

    # ------------------------------------------------------------------
    # Tensor helpers
    def _to_tensor(self, np_arr: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np_arr.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(self.cfg.imgsz, self.cfg.imgsz),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.squeeze(0)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, root: str, name: str) -> Dict[str, Any]:
        """Run inference on a single sample."""

        paths = self._paths(root, name)
        arr = self._load_modalities(paths)
        orig_h, orig_w = arr.shape[0], arr.shape[1]

        image_tensor = self._to_tensor(arr).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)

        ctx_vec = torch.from_numpy(self._load_context(paths["meta"])).to(self.device)
        ctx_vec = ctx_vec.unsqueeze(0)

        ctx_input = ctx_vec
        if self.projector is not None:
            ctx_input = self.projector(ctx_vec)

        outputs = self.model(image_tensor, ctx_input)
        logits = outputs.get("logits")
        if logits is None:
            raise RuntimeError("Model output does not contain 'logits'.")

        probs = torch.sigmoid(logits)
        resized = F.interpolate(
            probs,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        result_masks: List[np.ndarray] = []
        result_boxes: List[List[float]] = []
        result_classes: List[int] = []
        result_scores: List[float] = []

        threshold = float(self.cfg.mask_threshold)
        score_thr = float(self.cfg.score_threshold)

        for cls_idx in range(resized.shape[1]):
            cls_mask = resized[0, cls_idx]
            score = float(cls_mask.max().item())
            if score < score_thr:
                continue

            binary = (cls_mask >= threshold).float()
            if binary.sum() <= 0:
                continue

            mask_np = binary.cpu().numpy()
            ys, xs = np.where(mask_np > 0.0)
            if xs.size == 0 or ys.size == 0:
                continue

            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            result_masks.append(mask_np)
            result_boxes.append([x0, y0, x1, y1])
            result_classes.append(cls_idx)
            result_scores.append(score)

        return {
            "name": name,
            "boxes": result_boxes,
            "masks": result_masks,
            "classes": result_classes,
            "scores": result_scores,
            "logits": logits.detach().cpu(),
            "context": ctx_vec.detach().cpu(),
        }
