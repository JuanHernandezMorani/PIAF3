"""Multimodal dataset loader for RGB, PBR maps and optional textual context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

try:  # pragma: no cover - dependency guard for packaging
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyYAML is required to load dataset configuration files") from exc


# --------------------------------------------------------------------------------------
# Configuration containers


@dataclass(frozen=True)
class PBRSpec:
    """Description of the PBR side-channel configuration."""

    enabled: bool
    suffix: Mapping[str, str]
    order: Sequence[str]


@dataclass(frozen=True)
class TextSpec:
    """Description of the textual context side-channel."""

    enabled: bool
    root: str


@dataclass(frozen=True)
class FormatsSpec:
    """Dataset formats supported by the loader."""

    images: Sequence[str]
    labels: Sequence[str]
    masks_ok: bool


@dataclass(frozen=True)
class DataSpec:
    """Representation of ``configs/data.yaml``."""

    path: Path
    train: str
    val: str
    test: Optional[str]
    nc: int
    names: Sequence[str]
    pbr: PBRSpec
    text: TextSpec
    formats: FormatsSpec


DEFAULT_PBR_VALUES: Dict[str, float | Sequence[float]] = {
    "albedo": (0.0, 0.0, 0.0),
    "normal": (0.0, 0.0, 0.0),
    "roughness": 0.0,
    "metallic": 0.0,
    "ao": 0.0,
    "height": 0.0,
    "curvature": 0.0,
}

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - Pillow < 9.1 compatibility
    RESAMPLE_BILINEAR = Image.BILINEAR


PBR_CHANNELS: Dict[str, int] = {
    "albedo": 3,
    "normal": 3,
    "roughness": 1,
    "metallic": 1,
    "ao": 1,
    "height": 1,
    "curvature": 1,
}


# --------------------------------------------------------------------------------------
# YAML helpers


def _as_sequence(value: Any, *, key: str) -> Sequence[str]:
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    msg = f"'{key}' must be a list"
    raise TypeError(msg)


def _load_data_spec(path: Path) -> DataSpec:
    """Load and validate the dataset specification file."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise TypeError("Dataset YAML must contain a mapping")

    root = Path(str(data.get("path", "."))).expanduser()
    train = str(data.get("train", "")).strip()
    val = str(data.get("val", "")).strip()
    test_raw = data.get("test")
    test = str(test_raw).strip() if test_raw else None
    if not train or not val:
        raise ValueError("'train' and 'val' entries must be provided in the dataset YAML")

    nc = int(data.get("nc", 0))
    names = data.get("names", [])
    if nc < 0:
        raise ValueError("'nc' must be non-negative")
    if not isinstance(names, Sequence):
        raise TypeError("'names' must be a list")

    pbr_cfg = data.get("pbr", {})
    if not isinstance(pbr_cfg, Mapping):
        raise TypeError("'pbr' section must be a mapping")
    pbr_enabled = bool(pbr_cfg.get("enabled", False))
    suffix_map = pbr_cfg.get("suffix", {})
    if not isinstance(suffix_map, Mapping):
        raise TypeError("'pbr.suffix' must be a mapping")
    order = _as_sequence(pbr_cfg.get("order", []), key="pbr.order")
    pbr = PBRSpec(enabled=pbr_enabled, suffix={str(k): str(v) for k, v in suffix_map.items()}, order=order)

    text_cfg = data.get("text", {})
    if not isinstance(text_cfg, Mapping):
        raise TypeError("'text' section must be a mapping")
    text = TextSpec(enabled=bool(text_cfg.get("enabled", False)), root=str(text_cfg.get("root", "texts")))

    formats_cfg = data.get("formats", {})
    if not isinstance(formats_cfg, Mapping):
        raise TypeError("'formats' must be a mapping")
    image_exts = tuple(str(ext).lower() for ext in formats_cfg.get("images", [".png", ".jpg"]))
    label_exts = tuple(str(ext).lower() for ext in formats_cfg.get("labels", [".txt"]))
    masks_ok = bool(formats_cfg.get("masks_ok", False))
    formats = FormatsSpec(images=image_exts, labels=label_exts, masks_ok=masks_ok)

    return DataSpec(
        path=root,
        train=train,
        val=val,
        test=test,
        nc=nc,
        names=tuple(str(x) for x in names),
        pbr=pbr,
        text=text,
        formats=formats,
    )


# --------------------------------------------------------------------------------------
# Utility functions


def _load_image(path: Path, mode: Optional[str] = None) -> np.ndarray:
    """Load an image from ``path`` as a float32 array in ``[0, 1]``."""

    if not path.exists():
        raise ValueError(f"Image file not found: {path}")
    with Image.open(path) as img:
        if mode is None:
            img = img.convert("RGBA" if img.mode == "RGBA" else "RGB")
        else:
            img = img.convert(mode)
        arr = np.asarray(img, dtype=np.float32)
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


def _segments_to_masks(segments: Sequence[np.ndarray], height: int, width: int) -> Tensor:
    masks = torch.zeros((len(segments), height, width), dtype=torch.bool)
    for idx, polygon in enumerate(segments):
        poly = polygon.copy()
        poly[:, 0] *= width
        poly[:, 1] *= height
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in poly], fill=1, outline=1)
        masks[idx] = torch.from_numpy(np.array(canvas, dtype=np.uint8)).bool()
    return masks


# --------------------------------------------------------------------------------------
# Augmentations


@dataclass
class AugmentationConfig:
    """Configuration container for data augmentations."""

    mosaic: bool = False
    mosaic_prob: float = 0.5
    hsv_jitter: float = 0.05
    block_mix_prob: float = 0.0
    block_size: int = 8


# --------------------------------------------------------------------------------------
# Dataset implementation


class MultiModalYOLODataset(Dataset):
    """Dataset returning RGB tensors, optional PBR stacks and raw text snippets."""

    def __init__(
        self,
        data_cfg: Path | str,
        split: str,
        imgsz: int,
        *,
        augmentation: Optional[AugmentationConfig] = None,
        seed: Optional[int] = None,
        use_text: bool = True,
        use_pbr: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val' or 'test'")
        self.cfg_path = Path(data_cfg).expanduser().resolve()
        self.spec = _load_data_spec(self.cfg_path)
        self.split = split
        self.imgsz = int(imgsz)
        self.augment = augmentation or AugmentationConfig()
        self.rng = np.random.default_rng(seed)
        self.use_text = bool(use_text and self.spec.text.enabled)
        self.use_pbr = bool(use_pbr and self.spec.pbr.enabled)

        split_key = {"train": self.spec.train, "val": self.spec.val, "test": self.spec.test}[split]
        if split_key is None:
            raise ValueError(f"Split '{split}' not defined in dataset specification")

        self.root = self.spec.path.expanduser().resolve()
        self.image_base = Path(split_key)
        self.images = self._collect_images(self.root / self.image_base)
        if not self.images:
            raise FileNotFoundError(f"No images found for split '{split}' at {self.root / self.image_base}")

        self._images_prefix = self.image_base.parts[0] if self.image_base.parts else "images"

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial proxy
        return len(self.images)

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if (
            self.split == "train"
            and self.augment.mosaic
            and self.rng.random() < float(self.augment.mosaic_prob)
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

    # ------------------------------------------------------------------
    def _collect_images(self, directory: Path) -> List[Path]:
        if not directory.exists():
            raise ValueError(f"Image directory does not exist: {directory}")
        image_paths: List[Path] = []
        for ext in self.spec.formats.images:
            image_paths.extend(sorted(directory.rglob(f"*{ext}")))
        return sorted({path.resolve() for path in image_paths})

    # ------------------------------------------------------------------
    def _label_path(self, image_path: Path) -> Path:
        rel = image_path.relative_to(self.root)
        if not rel.parts or rel.parts[0] != self._images_prefix:
            raise ValueError(
                f"Image '{image_path}' is expected to live under '{self._images_prefix}' relative to {self.root}"
            )
        suffix = rel.suffix
        rel_without_prefix = Path(*rel.parts[1:]).with_suffix(".txt")
        label_path = self.root / "labels" / rel_without_prefix
        if label_path.suffix.lower() not in self.spec.formats.labels:
            label_path = label_path.with_suffix(self.spec.formats.labels[0])
        if not label_path.exists():
            raise ValueError(f"Missing label file for image '{image_path}': expected {label_path}")
        return label_path

    # ------------------------------------------------------------------
    def _text_path(self, image_path: Path) -> Path:
        rel = image_path.relative_to(self.root)
        rel_without_prefix = Path(*rel.parts[1:]).with_suffix(".txt")
        return self.root / self.spec.text.root / rel_without_prefix

    # ------------------------------------------------------------------
    def _load_labels(self, path: Path) -> tuple[np.ndarray, List[np.ndarray]]:
        entries: List[List[float]] = []
        segments: List[np.ndarray] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = [float(x) for x in stripped.split()]
                if len(parts) < 5:
                    continue
                entries.append(parts[:5])
                if len(parts) > 5 and self.spec.formats.masks_ok:
                    seg = np.array(parts[5:], dtype=np.float32).reshape(-1, 2)
                    segments.append(seg)
        return (
            np.array(entries, dtype=np.float32) if entries else np.zeros((0, 5), dtype=np.float32),
            segments,
        )

    # ------------------------------------------------------------------
    def _load_text(self, image_path: Path) -> str:
        if not self.use_text:
            return ""
        candidate = self._text_path(image_path)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
        return ""

    # ------------------------------------------------------------------
    def _load_pbr(self, image_path: Path, size: tuple[int, int]) -> Tensor:
        if not self.use_pbr:
            return torch.zeros((0, *size), dtype=torch.float32)
        height, width = size
        channels: List[np.ndarray] = []
        for key in self.spec.pbr.order:
            suffix = self.spec.pbr.suffix.get(key)
            if suffix is None:
                raise ValueError(f"PBR suffix missing for key '{key}'")
            candidate = image_path.with_name(f"{image_path.stem}{suffix}{image_path.suffix}")
            ch = PBR_CHANNELS.get(key, 1)
            if candidate.exists():
                mode = "RGB" if ch == 3 else "L"
                arr = _load_image(candidate, mode=mode)
            else:
                default = DEFAULT_PBR_VALUES.get(key, 0.0)
                if isinstance(default, Sequence):
                    arr = np.zeros((height, width, len(default)), dtype=np.float32)
                    arr[...] = np.asarray(default, dtype=np.float32)
                else:
                    arr = np.full((height, width, ch), float(default), dtype=np.float32)
            if arr.shape[:2] != size:
                resized_channels: List[np.ndarray] = []
                for channel_idx in range(arr.shape[2]):
                    channel_img = Image.fromarray((arr[..., channel_idx] * 255).astype(np.uint8), mode="L")
                    resized = channel_img.resize(size[::-1], resample=RESAMPLE_BILINEAR)
                    resized_channels.append(np.asarray(resized, dtype=np.float32) / 255.0)
                arr = np.stack(resized_channels, axis=-1)
            channels.append(arr if arr.ndim == 3 else arr[..., None])
        if not channels:
            return torch.zeros((0, height, width), dtype=torch.float32)
        stacked = np.concatenate(channels, axis=-1)
        return torch.from_numpy(stacked.transpose(2, 0, 1)).float()

    # ------------------------------------------------------------------
    def _load_raw(self, index: int) -> Dict[str, Any]:
        image_path = self.images[index]
        rgb = _load_image(image_path, mode="RGB")
        height, width = rgb.shape[:2]
        labels, segments = self._load_labels(self._label_path(image_path))
        pbr = self._load_pbr(image_path, (height, width))
        text = self._load_text(image_path)
        masks = _segments_to_masks(segments, height, width) if segments else None
        boxes = labels[:, 1:5] if labels.size else np.zeros((0, 4), dtype=np.float32)
        classes = labels[:, 0] if labels.size else np.zeros((0,), dtype=np.float32)
        boxes_tensor = torch.from_numpy(boxes.astype(np.float32))
        cls_tensor = torch.from_numpy(classes.astype(np.float32)).long()
        return {
            "path": image_path,
            "rgb": torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
            "pbr": pbr,
            "boxes": boxes_tensor,
            "classes": cls_tensor,
            "masks": masks,
            "text": text,
            "orig_size": (height, width),
        }

    # ------------------------------------------------------------------
    def _prepare(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        image = self._resize(raw["rgb"], mode="bilinear")
        pbr = self._resize(raw["pbr"], mode="bilinear") if raw["pbr"].numel() else raw["pbr"]
        boxes = raw["boxes"].clone()
        classes = raw["classes"].clone()
        masks = raw["masks"]
        if masks is not None and masks.numel():
            masks = self._resize(masks.float(), mode="nearest").bool()

        target: Dict[str, Any] = {
            "boxes": boxes,
            "cls": classes,
        }
        if masks is not None:
            target["masks"] = masks
        return {
            "img": torch.clamp(image, 0.0, 1.0),
            "pbr": pbr if pbr.numel() else None,
            "context_txt": raw["text"],
            "labels": target,
            "meta": {"path": str(raw["path"]), "orig_size": raw["orig_size"]},
        }

    # ------------------------------------------------------------------
    def _apply_aug(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample["img"]
        pbr = sample["pbr"]
        boxes = sample["labels"]["boxes"]
        masks = sample["labels"].get("masks")

        if self.rng.random() < 0.5:
            image = torch.flip(image, dims=(2,))
            if pbr is not None:
                pbr = torch.flip(pbr, dims=(2,))
            boxes[:, 0] = 1.0 - boxes[:, 0]
            if masks is not None:
                masks = torch.flip(masks, dims=(2,))
        if self.rng.random() < 0.5:
            image = torch.flip(image, dims=(1,))
            if pbr is not None:
                pbr = torch.flip(pbr, dims=(1,))
            boxes[:, 1] = 1.0 - boxes[:, 1]
            if masks is not None:
                masks = torch.flip(masks, dims=(1,))

        rotation = int(self.rng.integers(0, 4))
        if rotation:
            image = torch.rot90(image, k=rotation, dims=(1, 2))
            if pbr is not None:
                pbr = torch.rot90(pbr, k=rotation, dims=(1, 2))
            boxes = self._rotate_boxes(boxes, rotation)
            if masks is not None:
                masks = torch.rot90(masks, k=rotation, dims=(1, 2))

        image = self._hsv_jitter(image)
        image = self._block_mix(image)

        boxes[:, 0:4].clamp_(0.0, 1.0)
        sample["img"] = torch.clamp(image, 0.0, 1.0)
        sample["pbr"] = pbr
        sample["labels"]["boxes"] = boxes
        if masks is not None:
            sample["labels"]["masks"] = masks
        return sample

    # ------------------------------------------------------------------
    def _rotate_boxes(self, boxes: Tensor, k: int) -> Tensor:
        if boxes.numel() == 0:
            return boxes
        rotated = boxes.clone()
        for _ in range(k % 4):
            x, y, w, h = rotated[:, 0], rotated[:, 1], rotated[:, 2], rotated[:, 3]
            rotated[:, 0] = 1.0 - y
            rotated[:, 1] = x
            rotated[:, 2] = h
            rotated[:, 3] = w
        return rotated

    # ------------------------------------------------------------------
    def _hsv_jitter(self, image: Tensor) -> Tensor:
        strength = float(self.augment.hsv_jitter)
        if strength <= 0:
            return image
        perturb = torch.from_numpy(self.rng.normal(0.0, strength, size=3).astype(np.float32))
        hsv = rgb_to_hsv(image)
        hsv[0] = (hsv[0] + perturb[0]) % 1.0
        hsv[1] = torch.clamp(hsv[1] * (1.0 + perturb[1]), 0.0, 1.0)
        hsv[2] = torch.clamp(hsv[2] * (1.0 + perturb[2]), 0.0, 1.0)
        return hsv_to_rgb(hsv)

    # ------------------------------------------------------------------
    def _block_mix(self, image: Tensor) -> Tensor:
        prob = float(self.augment.block_mix_prob)
        if prob <= 0 or self.rng.random() >= prob:
            return image
        block = max(1, int(self.augment.block_size))
        donor_idx = int(self.rng.integers(0, len(self.images)))
        donor = self._prepare(self._load_raw(donor_idx))
        donor_img = donor["img"]
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

    # ------------------------------------------------------------------
    def _load_mosaic(self, indices: Sequence[int]) -> Dict[str, Any]:
        raws = [self._load_raw(idx) for idx in indices]
        canvas_img = torch.zeros((3, self.imgsz, self.imgsz), dtype=torch.float32)
        canvas_pbr = None
        if self.use_pbr and self.spec.pbr.order:
            total_channels = sum(PBR_CHANNELS.get(k, 1) for k in self.spec.pbr.order)
            canvas_pbr = torch.zeros((total_channels, self.imgsz, self.imgsz), dtype=torch.float32)
        canvas_boxes: List[Tensor] = []
        canvas_classes: List[Tensor] = []
        canvas_masks: List[Tensor] = []
        offsets = [(0, 0), (0, self.imgsz // 2), (self.imgsz // 2, 0), (self.imgsz // 2, self.imgsz // 2)]
        tile = self.imgsz // 2
        for raw, (top, left) in zip(raws, offsets):
            img = self._resize(raw["rgb"], mode="bilinear")
            bottom, right = top + tile, left + tile
            canvas_img[:, top:bottom, left:right] = img[:, :tile, :tile]

            if canvas_pbr is not None and raw["pbr"].numel():
                resized_pbr = self._resize(raw["pbr"], mode="bilinear")
                canvas_pbr[:, top:bottom, left:right] = resized_pbr[:, :tile, :tile]

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
            if raw["masks"] is not None and raw["masks"].numel():
                resized = self._resize(raw["masks"].float(), mode="nearest").bool()
                pad = torch.zeros((resized.shape[0], self.imgsz, self.imgsz), dtype=torch.bool)
                pad[:, top:bottom, left:right] = resized[:, :tile, :tile]
                canvas_masks.append(pad)

        target_boxes = torch.cat(canvas_boxes, dim=0) if canvas_boxes else torch.zeros((0, 4), dtype=torch.float32)
        target_classes = torch.cat(canvas_classes, dim=0) if canvas_classes else torch.zeros((0,), dtype=torch.long)
        target_masks = torch.cat(canvas_masks, dim=0) if canvas_masks else None

        labels: Dict[str, Any] = {"boxes": target_boxes, "cls": target_classes}
        if target_masks is not None:
            labels["masks"] = target_masks
        return {
            "img": canvas_img,
            "pbr": canvas_pbr,
            "context_txt": raws[0]["text"],
            "labels": labels,
            "meta": {"path": str(raws[0]["path"]), "orig_size": raws[0]["orig_size"]},
        }

    # ------------------------------------------------------------------
    def _resize(self, tensor: Tensor, *, mode: str) -> Tensor:
        if tensor.numel() == 0:
            return tensor
        align = {"bilinear", "bicubic", "trilinear"}
        args = {"align_corners": False} if mode in align else {}
        if tensor.dim() == 4:
            return F.interpolate(tensor, size=(self.imgsz, self.imgsz), mode=mode, **args)
        tensor4d = tensor.unsqueeze(0)
        resized = F.interpolate(tensor4d, size=(self.imgsz, self.imgsz), mode=mode, **args)
        return resized.squeeze(0)


# --------------------------------------------------------------------------------------
# Collate & dataloader helpers


def multimodal_collate(batch: Sequence[MutableMapping[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch passed to collate function")
    imgs = torch.stack([item["img"] for item in batch])
    pbr_values = [item["pbr"] for item in batch]
    pbr_tensor: Optional[Tensor]
    if all(val is None for val in pbr_values):
        pbr_tensor = None
    else:
        template = next(val for val in pbr_values if val is not None)
        filled = [val if val is not None else torch.zeros_like(template) for val in pbr_values]
        pbr_tensor = torch.stack(filled)
    labels_keys = set().union(*(item["labels"].keys() for item in batch))
    labels: Dict[str, List[Any]] = {key: [] for key in labels_keys}
    for item in batch:
        for key in labels_keys:
            labels[key].append(item["labels"].get(key))
    return {
        "img": imgs,
        "pbr": pbr_tensor,
        "context_txt": [item["context_txt"] for item in batch],
        "labels": labels,
        "meta": [item.get("meta", {}) for item in batch],
    }


def build_dataloaders(cfg: Mapping[str, Any]) -> Dict[str, DataLoader]:
    """Instantiate dataloaders for the splits declared in ``cfg``."""

    data_section = cfg.get("data", {})
    if not isinstance(data_section, Mapping):
        raise TypeError("cfg['data'] must be a mapping")
    data_yaml = data_section.get("yaml")
    if data_yaml is None:
        raise ValueError("cfg['data']['yaml'] must point to configs/data.yaml")
    imgsz = int(data_section.get("imgsz", 640))
    batch = int(data_section.get("batch", 16))
    workers = int(data_section.get("workers", 4))
    shuffle = bool(data_section.get("shuffle", True))
    augment_section = data_section.get("augment", {})
    if not isinstance(augment_section, Mapping):
        raise TypeError("cfg['data']['augment'] must be a mapping")
    aug_cfg = AugmentationConfig(
        mosaic=bool(augment_section.get("mosaic", False)),
        mosaic_prob=float(augment_section.get("mosaic_prob", 0.5)),
        hsv_jitter=float(augment_section.get("hsv_jitter", 0.05)),
        block_mix_prob=float(augment_section.get("block_mix_prob", 0.0)),
        block_size=int(augment_section.get("block_size", 8)),
    )

    train_section = cfg.get("train", {})
    if not isinstance(train_section, Mapping):
        raise TypeError("cfg['train'] must be a mapping")
    use_text = bool(train_section.get("use_text", True))
    use_pbr = bool(train_section.get("use_pbr", True))

    dataset_args = dict(data_cfg=data_yaml, imgsz=imgsz, use_text=use_text, use_pbr=use_pbr)
    train_dataset = MultiModalYOLODataset(split="train", augmentation=aug_cfg, **dataset_args)
    val_dataset = MultiModalYOLODataset(split="val", augmentation=AugmentationConfig(), **dataset_args)
    dataloaders: Dict[str, DataLoader] = {}
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=multimodal_collate,
    )
    dataloaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=multimodal_collate,
    )

    try:
        test_dataset = MultiModalYOLODataset(split="test", augmentation=AugmentationConfig(), **dataset_args)
    except ValueError:
        return dataloaders
    except FileNotFoundError:
        return dataloaders
    dataloaders["test"] = DataLoader(
        test_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=multimodal_collate,
    )
    return dataloaders


# --------------------------------------------------------------------------------------
# Colour space helpers


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
    "build_dataloaders",
    "multimodal_collate",
]
