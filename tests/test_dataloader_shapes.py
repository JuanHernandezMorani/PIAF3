from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

pytest.importorskip("PIL")
from PIL import Image

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from src.data.multimodal_loader import (
    AugmentationConfig,
    MultiModalYOLODataset,
    multimodal_collate,
)


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def _write_gray(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((32, 32), int(value * 255), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


@pytest.fixture()
def sample_dataset(tmp_path: Path) -> MultiModalYOLODataset:
    data_root = tmp_path / "dataset"
    (data_root / "images" / "train").mkdir(parents=True)
    (data_root / "images" / "val").mkdir(parents=True)
    (data_root / "labels" / "train").mkdir(parents=True)
    (data_root / "labels" / "val").mkdir(parents=True)
    (data_root / "texts" / "train").mkdir(parents=True)
    (data_root / "texts" / "val").mkdir(parents=True)

    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        """
path: {root}
train: images/train
val: images/val
nc: 1
names: [class_0]

pbr:
  enabled: true
  suffix:
    normal: "_normal"
    roughness: "_rough"
    metallic: "_metal"
    ao: "_ao"
    height: "_height"
    curvature: "_curv"
  order: ["normal", "roughness", "metallic", "ao", "height", "curvature"]

text:
  enabled: true
  root: texts

formats:
  images: [".png"]
  labels: [".txt"]
  masks_ok: true
        """.strip().format(root=data_root.as_posix()),
        encoding="utf-8",
    )

    img_train = data_root / "images" / "train" / "sample1.png"
    _write_rgb(img_train, (255, 0, 0))
    _write_rgb(img_train.with_name("sample1_normal.png"), (128, 128, 255))
    _write_gray(img_train.with_name("sample1_rough.png"), 0.4)
    _write_gray(img_train.with_name("sample1_metal.png"), 0.1)
    _write_gray(img_train.with_name("sample1_ao.png"), 1.0)
    _write_gray(img_train.with_name("sample1_height.png"), 0.6)
    _write_gray(img_train.with_name("sample1_curv.png"), 0.3)

    img_train2 = data_root / "images" / "train" / "sample2.png"
    _write_rgb(img_train2, (0, 255, 0))
    _write_rgb(img_train2.with_name("sample2_normal.png"), (100, 150, 255))
    _write_gray(img_train2.with_name("sample2_rough.png"), 0.7)
    # Missing metallic map on purpose
    _write_gray(img_train2.with_name("sample2_ao.png"), 0.8)
    _write_gray(img_train2.with_name("sample2_height.png"), 0.2)
    _write_gray(img_train2.with_name("sample2_curv.png"), 0.5)

    seg_points = "0.5 0.5 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75"
    label_line = f"0 0.5 0.5 0.25 0.25 {seg_points}\n"
    (data_root / "labels" / "train" / "sample1.txt").write_text(label_line, encoding="utf-8")
    (data_root / "labels" / "train" / "sample2.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    (data_root / "texts" / "train" / "sample1.txt").write_text("red creature", encoding="utf-8")
    (data_root / "texts" / "train" / "sample2.txt").write_text("", encoding="utf-8")

    # Mirror validation split with a single sample
    img_val = data_root / "images" / "val" / "sample3.png"
    _write_rgb(img_val, (0, 0, 255))
    (data_root / "labels" / "val" / "sample3.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (data_root / "texts" / "val" / "sample3.txt").write_text("blue creature", encoding="utf-8")
    _write_rgb(img_val.with_name("sample3_normal.png"), (128, 128, 255))
    _write_gray(img_val.with_name("sample3_rough.png"), 0.4)
    _write_gray(img_val.with_name("sample3_metal.png"), 0.1)
    _write_gray(img_val.with_name("sample3_ao.png"), 1.0)
    _write_gray(img_val.with_name("sample3_height.png"), 0.6)
    _write_gray(img_val.with_name("sample3_curv.png"), 0.3)

    dataset = MultiModalYOLODataset(
        data_cfg=yaml_path,
        split="train",
        imgsz=64,
        augmentation=AugmentationConfig(mosaic=False, hsv_jitter=0.0, block_mix_prob=0.0),
        seed=0,
        use_text=True,
        use_pbr=True,
    )
    return dataset


def test_dataset_shapes(sample_dataset: MultiModalYOLODataset) -> None:
    sample = sample_dataset[0]
    assert sample["img"].shape == (3, 64, 64)
    assert isinstance(sample["context_txt"], str)
    assert sample["pbr"] is not None
    assert sample["pbr"].shape[0] == 8  # 3 normales + 5 canales escalares
    assert sample["labels"]["boxes"].shape == (1, 4)
    assert "masks" in sample["labels"]
    assert sample_dataset[1]["pbr"].shape[0] == 8


def test_collate_and_missing_modalities(sample_dataset: MultiModalYOLODataset) -> None:
    batch = [sample_dataset[0], sample_dataset[1]]
    collated = multimodal_collate(batch)
    assert collated["img"].shape == (2, 3, 64, 64)
    assert collated["pbr"].shape == (2, 8, 64, 64)
    assert collated["context_txt"] == ["red creature", ""]
    assert len(collated["labels"]["boxes"]) == 2
    metallic_mask = collated["pbr"][1, 4]  # metallic channel index after normals+roughness
    assert torch.allclose(metallic_mask, torch.zeros_like(metallic_mask))


class _DeterministicRNG:
    def __init__(self) -> None:
        self._random_calls = 0

    def random(self) -> float:
        self._random_calls += 1
        return 0.0 if self._random_calls == 1 else 1.0

    def integers(self, low: int, high: Optional[int] = None, size: Optional[int] = None) -> int:
        del low, high, size
        return 0

    def normal(self, loc: float, scale: float, size: tuple[int, ...] | int | None = None) -> np.ndarray:
        shape = size if isinstance(size, tuple) else (size,) if isinstance(size, int) else ()
        return np.zeros(shape, dtype=np.float32)

    def uniform(self, low: float, high: float, size: Optional[int] = None) -> float:
        del low, high, size
        return 0.3


def test_box_mask_alignment_after_flip(sample_dataset: MultiModalYOLODataset) -> None:
    raw = sample_dataset._prepare(sample_dataset._load_raw(0))
    raw["labels"]["boxes"] = torch.tensor([[0.25, 0.5, 0.5, 0.5]], dtype=torch.float32)
    mask = torch.zeros((1, 64, 64), dtype=torch.bool)
    mask[:, 16:48, 8:40] = True
    raw["labels"]["masks"] = mask

    sample_dataset.rng = _DeterministicRNG()  # force first call < 0.5 (horizontal flip)
    augmented = sample_dataset._apply_aug(raw)

    boxes = augmented["labels"]["boxes"]
    assert torch.allclose(boxes[0], torch.tensor([0.75, 0.5, 0.5, 0.5]))
    flipped_mask = augmented["labels"]["masks"]
    assert flipped_mask is not None
    assert torch.equal(flipped_mask[:, :, 64 - 40 : 64 - 8], mask[:, :, 8:40])
