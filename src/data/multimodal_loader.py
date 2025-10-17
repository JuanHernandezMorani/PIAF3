"""Multimodal dataset loader for RGB, PBR and textual context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset

try:  # pragma: no cover - optional dependency guard
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    msg = "PyYAML is required to load dataset configuration files"
    raise ModuleNotFoundError(msg) from exc


DEFAULT_PBR_VALUES: Dict[str, float | Sequence[float]] = {
    "normal": (0.5, 0.5, 1.0),
    "rough": 0.5,
    "ao": 1.0,
    "height": 0.5,
    "metallic": 0.0,
}

PBR_CHANNEL_ORDER: Sequence[str] = (
    "normal_x",
    "normal_y",
    "normal_z",
    "rough",
    "ao",
    "height",
    "metallic",
)

PBR_SUFFIXES: Dict[str, Sequence[str]] = {
    "normal": ("normal",),
    "rough": ("rough", "roughness"),
    "ao": ("ao", "ambient_occlusion"),
    "height": ("height", "displacement"),
    "metallic": ("metallic", "metalness"),
}


@dataclass
class AugmentationConfig:
    """Configuration container for data augmentations."""

    mosaic: bool = False
    mosaic_prob: float = 0.5
    hsv_jitter: float = 0.05
    block_mix_prob: float = 0.0
    block_size: int = 8


class TextEmbeddingCache:
    """Caches deterministic text embeddings computed as averaged token vectors."""

    def __init__(self, dim: int, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.dim = dim
        self.seed = seed
        self._cache: Dict[Path, Tensor] = {}

    def __call__(self, path: Path, content: str) -> Tensor:
        """Return a cached embedding for ``content`` associated with ``path``."""

        cached = self._cache.get(path)
        if cached is not None:
            return cached.clone()
        tokens = [token for token in content.replace("\n", " ").split(" ") if token]
        if not tokens:
            embedding = torch.zeros(self.dim, dtype=torch.float32)
        else:
            vectors: List[Tensor] = []
            for token in tokens:
                token_seed = (hash(token) ^ self.seed) & 0xFFFFFFFF
                rng = np.random.default_rng(token_seed)
                vec = torch.from_numpy(rng.standard_normal(self.dim).astype(np.float32))
                vectors.append(vec)
            embedding = torch.stack(vectors, dim=0).mean(dim=0)
        norm = torch.linalg.norm(embedding)
        if torch.isfinite(norm) and norm > 0:
            embedding = embedding / norm
        self._cache[path] = embedding.clone()
        return embedding


def _load_image(path: Path, mode: str = "RGB") -> np.ndarray:
    """Load an image from ``path`` in ``mode`` and normalise it to ``[0, 1]``."""

    with Image.open(path) as img:
        arr = np.asarray(img.convert(mode), dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    return np.clip(arr / 255.0, 0.0, 1.0)


def _xywhn_to_xywh(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] *= width
    out[:, 1] *= height
    out[:, 2] *= width
    out[:, 3] *= height
    return out


def _xywh_to_xywhn(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] /= width
    out[:, 1] /= height
    out[:, 2] /= width
    out[:, 3] /= height
    return out


class MultiModalYOLODataset(Dataset):
    """Dataset that returns RGB images, PBR maps, text embeddings and YOLO targets."""

    def __init__(
        self,
        data_cfg: Path | str,
        split: str,
        imgsz: int,
        *,
        text_dim: int,
        augmentation: Optional[AugmentationConfig] = None,
        seed: Optional[int] = None,
        use_text: bool = True,
        use_pbr: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val' or 'test'")
        self.cfg_path = Path(data_cfg).expanduser().resolve()
        self.split = split
        self.imgsz = int(imgsz)
        self.augment = augmentation or AugmentationConfig()
        self.rng = np.random.default_rng(seed)
        self.use_text = use_text
        self.use_pbr = use_pbr
        self.text_cache = TextEmbeddingCache(text_dim, seed=seed or 0)

        raw_cfg = self._load_yaml(self.cfg_path)
        root = (self.cfg_path.parent / raw_cfg.get("path", "")).resolve()
        split_entry = raw_cfg.get(split, split)
        self.images = self._collect_images(root, split_entry)
        if not self.images:
            raise FileNotFoundError(f"No images found for split '{split}' in {split_entry}")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if (
            self.split == "train"
            and self.augment.mosaic
            and self.rng.random() < self.augment.mosaic_prob
        ):
            indices = [index]
            while len(indices) < 4:
                indices.append(int(self.rng.integers(0, len(self.images))))
            sample = self._load_mosaic(indices)
        else:
            sample = self._prepare(self._load_raw(index))
        if self.split == "train":
            sample = self._apply_aug(sample)
        return sample

    # YAML helpers -----------------------------------------------------------------
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise TypeError("Dataset YAML must contain a mapping")
        return data

    def _collect_images(self, root: Path, entry: Any) -> List[Path]:
        entries: Iterable[str]
        if isinstance(entry, (list, tuple)):
            entries = [str(item) for item in entry]
        else:
            entries = [str(entry)]
        paths: List[Path] = []
        for item in entries:
            resolved = (root / item).expanduser()
            if resolved.is_dir():
                for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                    paths.extend(sorted(resolved.rglob(ext)))
            elif resolved.is_file():
                with resolved.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        candidate = line.strip()
                        if candidate:
                            paths.append((root / candidate).resolve())
        return sorted(set(paths))

    # Raw loading ------------------------------------------------------------------
    def _load_raw(self, index: int) -> Dict[str, Any]:
        image_path = self.images[index]
        rgb = _load_image(image_path, "RGB")
        height, width = rgb.shape[:2]
        labels, segments = self._load_labels(self._label_path(image_path))
        pbr, pbr_mask = self._load_pbr(image_path, (height, width))
        text = self._load_text(image_path)
        masks = self._segments_to_masks(segments, height, width) if segments else None
        return {
            "path": image_path,
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1)),
            "pbr": pbr,
            "pbr_mask": pbr_mask,
            "boxes": torch.from_numpy(labels[:, 1:5]) if labels.size else torch.zeros((0, 4), dtype=torch.float32),
            "classes": torch.from_numpy(labels[:, :1]).squeeze(-1).long() if labels.size else torch.zeros((0,), dtype=torch.long),
            "segments": segments,
            "masks": masks,
            "text": text,
            "orig_size": (height, width),
        }

    def _prepare(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        image = self._resize(raw["rgb"], mode="bilinear")
        pbr = self._resize(raw["pbr"], mode="bilinear") if raw["pbr"].numel() else raw["pbr"]
        pbr_mask = self._resize(raw["pbr_mask"], mode="nearest") if raw["pbr_mask"].numel() else raw["pbr_mask"]
        boxes = raw["boxes"].clone()
        classes = raw["classes"].clone()
        masks = raw["masks"]
        if masks is not None and masks.numel():
            masks = self._resize(masks, mode="nearest")

        target: Dict[str, Any] = {
            "boxes": boxes,
            "classes": classes,
        }
        if masks is not None:
            target["masks"] = masks

        sample = {
            "image": image,
            "pbr": pbr,
            "pbr_mask": pbr_mask,
            "text": raw["text"],
            "targets": target,
            "meta": {"path": str(raw["path"]), "orig_size": raw["orig_size"]},
        }
        return sample

    # Utility loaders ---------------------------------------------------------------
    def _label_path(self, image_path: Path) -> Path:
        labels_dir = image_path.parents[1] / "labels" / image_path.parent.name
        candidate = labels_dir / f"{image_path.stem}.txt"
        return candidate if candidate.exists() else image_path.with_suffix(".txt")

    def _load_labels(self, path: Path) -> tuple[np.ndarray, List[np.ndarray]]:
        if not path.exists():
            return np.zeros((0, 5), dtype=np.float32), []
        entries: List[List[float]] = []
        segments: List[np.ndarray] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = [float(x) for x in line.strip().split()]
                if len(parts) < 5:
                    continue
                entries.append(parts[:5])
                if len(parts) > 5:
                    seg = np.array(parts[5:], dtype=np.float32).reshape(-1, 2)
                    segments.append(seg)
        return np.array(entries, dtype=np.float32) if entries else np.zeros((0, 5), dtype=np.float32), segments

    def _load_pbr(self, image_path: Path, size: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """Load PBR maps, filling missing channels with sensible defaults."""

        height, width = size
        channels: List[np.ndarray] = []
        availability = np.zeros(len(PBR_CHANNEL_ORDER), dtype=np.float32)

        normal = self._load_normal_map(image_path, size)
        if normal is not None:
            availability[:3] = 1.0
        normal_defaults = DEFAULT_PBR_VALUES["normal"]
        assert isinstance(normal_defaults, Sequence)
        for idx in range(3):
            channel = normal[..., idx : idx + 1] if normal is not None else np.full(
                (height, width, 1), float(normal_defaults[idx]), dtype=np.float32
            )
            channels.append(channel)

        for key in ("rough", "ao", "height", "metallic"):
            channel = self._load_scalar_map(image_path, key, size)
            if channel is None:
                channel = np.full((height, width, 1), float(DEFAULT_PBR_VALUES[key]), dtype=np.float32)
            else:
                availability[PBR_CHANNEL_ORDER.index(key)] = 1.0
            channels.append(channel)

        stacked = np.concatenate(channels, axis=-1)
        tensor = torch.from_numpy(stacked.transpose(2, 0, 1)) if self.use_pbr else torch.zeros(
            (len(PBR_CHANNEL_ORDER), height, width), dtype=torch.float32
        )
        mask_np = availability.reshape(-1, 1, 1)
        mask_np = np.repeat(np.repeat(mask_np, height, axis=1), width, axis=2)
        mask = (
            torch.from_numpy(mask_np)
            if self.use_pbr
            else torch.zeros((len(PBR_CHANNEL_ORDER), height, width), dtype=torch.float32)
        )
        return tensor.float(), mask.float()

    def _load_normal_map(self, image_path: Path, size: tuple[int, int]) -> Optional[np.ndarray]:
        """Load a normal map and ensure it is clamped to ``[0, 1]``."""

        for suffix in PBR_SUFFIXES["normal"]:
            candidate = image_path.with_name(f"{image_path.stem}_{suffix}{image_path.suffix}")
            if candidate.exists():
                normal = _load_image(candidate, "RGB")
                if normal.min() < 0.0 or normal.max() > 1.0:
                    normal = (normal + 1.0) * 0.5
                return np.clip(normal, 0.0, 1.0)
        return None

    def _load_scalar_map(
        self,
        image_path: Path,
        key: str,
        size: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Load a scalar PBR map or return ``None`` if not present."""

        height, width = size
        for suffix in PBR_SUFFIXES[key]:
            candidate = image_path.with_name(f"{image_path.stem}_{suffix}{image_path.suffix}")
            if candidate.exists():
                return _load_image(candidate, "L")
        return None

    def _load_text(self, image_path: Path) -> Tensor:
        if not self.use_text:
            return torch.zeros(self.text_cache.dim, dtype=torch.float32)
        candidates = [image_path.with_suffix(".txt"), image_path.parents[1] / "texts" / f"{image_path.stem}.txt"]
        for candidate in candidates:
            if candidate.exists():
                content = candidate.read_text(encoding="utf-8")
                return self.text_cache(candidate, content)
        return torch.zeros(self.text_cache.dim, dtype=torch.float32)

    def _segments_to_masks(self, segments: Sequence[np.ndarray], height: int, width: int) -> Tensor:
        masks = torch.zeros((len(segments), height, width), dtype=torch.float32)
        for idx, polygon in enumerate(segments):
            poly = polygon.copy()
            poly[:, 0] *= width
            poly[:, 1] *= height
            img = Image.new("L", (width, height), 0)
            ImageDraw.Draw(img).polygon([(float(x), float(y)) for x, y in poly], fill=1, outline=1)
            masks[idx] = torch.from_numpy(np.array(img, dtype=np.float32))
        return masks

    # Augmentations ----------------------------------------------------------------
    def _apply_aug(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["image"]
        pbr = sample["pbr"]
        pbr_mask = sample["pbr_mask"]
        boxes = sample["targets"]["boxes"]

        masks = sample["targets"].get("masks")
        if self.rng.random() < 0.5:
            image = torch.flip(image, dims=(2,))
            pbr = torch.flip(pbr, dims=(2,))
            pbr_mask = torch.flip(pbr_mask, dims=(2,))
            boxes[:, 0] = 1.0 - boxes[:, 0]
            if masks is not None:
                masks = torch.flip(masks, dims=(2,))
        if self.rng.random() < 0.5:
            image = torch.flip(image, dims=(1,))
            pbr = torch.flip(pbr, dims=(1,))
            pbr_mask = torch.flip(pbr_mask, dims=(1,))
            boxes[:, 1] = 1.0 - boxes[:, 1]
            if masks is not None:
                masks = torch.flip(masks, dims=(1,))
        rotation = self.rng.integers(0, 4)
        if rotation:
            image = torch.rot90(image, k=rotation, dims=(1, 2))
            pbr = torch.rot90(pbr, k=rotation, dims=(1, 2))
            pbr_mask = torch.rot90(pbr_mask, k=rotation, dims=(1, 2))
            boxes = self._rotate_boxes(boxes, int(rotation))
            if masks is not None:
                masks = torch.rot90(masks, k=rotation, dims=(1, 2))

        image = self._hsv_jitter(image)
        image = self._block_mix(image)

        sample["image"] = torch.clamp(image, 0.0, 1.0)
        sample["pbr"] = pbr
        sample["pbr_mask"] = pbr_mask
        sample["targets"]["boxes"] = boxes
        if masks is not None:
            sample["targets"]["masks"] = masks
        if sample["targets"]["boxes"].numel():
            boxes = sample["targets"]["boxes"]
            boxes[:, 0].clamp_(0.0, 1.0)
            boxes[:, 1].clamp_(0.0, 1.0)
            boxes[:, 2].clamp_(0.0, 1.0)
            boxes[:, 3].clamp_(0.0, 1.0)
        return sample

    def _rotate_boxes(self, boxes: Tensor, k: int) -> Tensor:
        rotated = boxes.clone()
        for _ in range(k % 4):
            x, y, w, h = rotated[:, 0], rotated[:, 1], rotated[:, 2], rotated[:, 3]
            rotated[:, 0] = 1.0 - y
            rotated[:, 1] = x
            rotated[:, 2] = h
            rotated[:, 3] = w
        return rotated

    def _hsv_jitter(self, image: Tensor) -> Tensor:
        strength = self.augment.hsv_jitter
        if strength <= 0:
            return image
        perturb = torch.from_numpy(self.rng.normal(0.0, strength, size=3).astype(np.float32))
        hsv = rgb_to_hsv(image)
        hsv[0] = (hsv[0] + perturb[0]) % 1.0
        hsv[1] = torch.clamp(hsv[1] * (1.0 + perturb[1]), 0.0, 1.0)
        hsv[2] = torch.clamp(hsv[2] * (1.0 + perturb[2]), 0.0, 1.0)
        return hsv_to_rgb(hsv)

    def _block_mix(self, image: Tensor) -> Tensor:
        prob = self.augment.block_mix_prob
        if prob <= 0 or self.rng.random() >= prob:
            return image
        block = max(1, int(self.augment.block_size))
        donor_count = int(self.rng.integers(0, 3))
        if donor_count == 0:
            return image
        donors = [self._prepare(self._load_raw(int(self.rng.integers(0, len(self.images))))) for _ in range(donor_count)]
        for donor in donors:
            donor_img = donor["image"]
            alpha = float(self.rng.uniform(0.2, 0.5))
            for y in range(0, image.shape[1], block):
                for x in range(0, image.shape[2], block):
                    bottom = min(y + block, image.shape[1])
                    right = min(x + block, image.shape[2])
                    patch = donor_img[:, y:bottom, x:right]
                    palette = patch.mean(dim=(1, 2), keepdim=True)
                    image[:, y:bottom, x:right] = (
                        (1.0 - alpha) * image[:, y:bottom, x:right] + alpha * palette
                    )
        return image

    # Mosaic ----------------------------------------------------------------------
    def _load_mosaic(self, indices: Sequence[int]) -> Dict[str, Any]:
        raws = [self._load_raw(idx) for idx in indices]
        canvas_img = torch.zeros((3, self.imgsz, self.imgsz), dtype=torch.float32)
        canvas_pbr = torch.zeros((len(PBR_CHANNEL_ORDER), self.imgsz, self.imgsz), dtype=torch.float32)
        canvas_mask = torch.zeros_like(canvas_pbr)
        canvas_boxes: List[Tensor] = []
        canvas_classes: List[Tensor] = []
        canvas_masks: List[Tensor] = []
        offsets = [(0, 0), (0, self.imgsz // 2), (self.imgsz // 2, 0), (self.imgsz // 2, self.imgsz // 2)]
        tile = self.imgsz // 2
        for raw, (top, left) in zip(raws, offsets):
            img = self._resize(raw["rgb"], mode="bilinear")
            pbr = self._resize(raw["pbr"], mode="bilinear")
            mask = self._resize(raw["pbr_mask"], mode="nearest")
            bottom, right = top + tile, left + tile
            canvas_img[:, top:bottom, left:right] = img[:, :tile, :tile]
            canvas_pbr[:, top:bottom, left:right] = pbr[:, :tile, :tile]
            canvas_mask[:, top:bottom, left:right] = mask[:, :tile, :tile]

            orig_h, orig_w = raw["orig_size"]
            boxes = raw["boxes"].numpy() if raw["boxes"].numel() else np.zeros((0, 4), dtype=np.float32)
            boxes_abs = _xywhn_to_xywh(boxes, orig_w, orig_h)
            boxes_abs[:, 0] = boxes_abs[:, 0] * (tile / orig_w) + left
            boxes_abs[:, 1] = boxes_abs[:, 1] * (tile / orig_h) + top
            boxes_abs[:, 2] *= tile / orig_w
            boxes_abs[:, 3] *= tile / orig_h
            boxes_norm = _xywh_to_xywhn(boxes_abs, self.imgsz, self.imgsz)
            if boxes_norm.size:
                canvas_boxes.append(torch.from_numpy(boxes_norm.astype(np.float32)))
                canvas_classes.append(raw["classes"].clone())
            if raw["masks"] is not None:
                resized = self._resize(raw["masks"], mode="nearest")
                pad = torch.zeros((resized.shape[0], self.imgsz, self.imgsz), dtype=torch.float32)
                pad[:, top:bottom, left:right] = resized[:, :tile, :tile]
                canvas_masks.append(pad)

        target_boxes = torch.cat(canvas_boxes, dim=0) if canvas_boxes else torch.zeros((0, 4), dtype=torch.float32)
        target_classes = torch.cat(canvas_classes, dim=0) if canvas_classes else torch.zeros((0,), dtype=torch.long)
        target_masks = torch.cat(canvas_masks, dim=0) if canvas_masks else None

        target: Dict[str, Any] = {"boxes": target_boxes, "classes": target_classes}
        if target_masks is not None:
            target["masks"] = target_masks
        return {
            "image": canvas_img,
            "pbr": canvas_pbr,
            "pbr_mask": canvas_mask,
            "text": raws[0]["text"],
            "targets": target,
            "meta": {"path": str(raws[0]["path"]), "orig_size": raws[0]["orig_size"]},
        }

    # Resize helper ----------------------------------------------------------------
    def _resize(self, tensor: Tensor, *, mode: str) -> Tensor:
        align = {"bilinear", "bicubic", "trilinear"}
        args = {"align_corners": False} if mode in align else {}
        if tensor.dim() == 4:
            return F.interpolate(tensor, size=(self.imgsz, self.imgsz), mode=mode, **args)
        tensor4d = tensor.unsqueeze(0)
        return F.interpolate(tensor4d, size=(self.imgsz, self.imgsz), mode=mode, **args).squeeze(0)


def multimodal_collate(batch: Sequence[MutableMapping[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    pbrs = torch.stack([item["pbr"] for item in batch]) if batch[0].get("pbr") is not None else None
    pbr_masks = (
        torch.stack([item["pbr_mask"] for item in batch])
        if batch[0].get("pbr_mask") is not None
        else None
    )
    texts = torch.stack([item["text"] for item in batch])
    targets = [item["targets"] for item in batch]
    meta = [item.get("meta", {}) for item in batch]
    return {
        "image": images,
        "pbr": pbrs,
        "pbr_mask": pbr_masks,
        "text": texts,
        "targets": targets,
        "meta": meta,
    }


def rgb_to_hsv(image: Tensor) -> Tensor:
    r, g, b = image[0], image[1], image[2]
    maxc = torch.max(image, dim=0).values
    minc = torch.min(image, dim=0).values
    v = maxc
    delta = maxc - minc
    s = torch.zeros_like(maxc)
    mask = maxc > 0
    s[mask] = delta[mask] / maxc[mask]
    h = torch.zeros_like(maxc)
    mask_delta = delta > 0
    rc = torch.zeros_like(maxc)
    gc = torch.zeros_like(maxc)
    bc = torch.zeros_like(maxc)
    rc[mask_delta] = (maxc - r)[mask_delta] / (6 * delta[mask_delta])
    gc[mask_delta] = (maxc - g)[mask_delta] / (6 * delta[mask_delta])
    bc[mask_delta] = (maxc - b)[mask_delta] / (6 * delta[mask_delta])
    h[r == maxc] = bc[r == maxc] - gc[r == maxc]
    h[g == maxc] = 1 / 3 + rc[g == maxc] - bc[g == maxc]
    h[b == maxc] = 2 / 3 + gc[b == maxc] - rc[b == maxc]
    h = h % 1.0
    return torch.stack((h, s, v), dim=0)


def hsv_to_rgb(image: Tensor) -> Tensor:
    h, s, v = image[0], image[1], image[2]
    i = torch.floor(h * 6.0).to(torch.int64)
    f = h * 6.0 - i.float()
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    rgb = torch.zeros_like(image)
    masks = [(i_mod == idx) for idx in range(6)]
    rgb[0][masks[0]] = v[masks[0]]
    rgb[1][masks[0]] = t[masks[0]]
    rgb[2][masks[0]] = p[masks[0]]
    rgb[0][masks[1]] = q[masks[1]]
    rgb[1][masks[1]] = v[masks[1]]
    rgb[2][masks[1]] = p[masks[1]]
    rgb[0][masks[2]] = p[masks[2]]
    rgb[1][masks[2]] = v[masks[2]]
    rgb[2][masks[2]] = t[masks[2]]
    rgb[0][masks[3]] = p[masks[3]]
    rgb[1][masks[3]] = q[masks[3]]
    rgb[2][masks[3]] = v[masks[3]]
    rgb[0][masks[4]] = t[masks[4]]
    rgb[1][masks[4]] = p[masks[4]]
    rgb[2][masks[4]] = v[masks[4]]
    rgb[0][masks[5]] = v[masks[5]]
    rgb[1][masks[5]] = p[masks[5]]
    rgb[2][masks[5]] = q[masks[5]]
    return rgb


__all__ = [
    "AugmentationConfig",
    "MultiModalYOLODataset",
    "multimodal_collate",
    "PBR_CHANNEL_ORDER",
    "DEFAULT_PBR_VALUES",
]
