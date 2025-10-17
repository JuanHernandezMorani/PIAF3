from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.multimodal_loader import (
    AugmentationConfig,
    MultiModalYOLODataset,
    multimodal_collate,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (32, 32), color).save(path)


def _write_map(path: Path, value: float) -> None:
    arr = np.full((32, 32), value, dtype=np.float32)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


@pytest.fixture()
def sample_dataset(tmp_path: Path) -> MultiModalYOLODataset:
    data_dir = tmp_path
    (data_dir / "images" / "train").mkdir(parents=True)
    (data_dir / "labels" / "train").mkdir(parents=True)
    (data_dir / "texts").mkdir(parents=True)

    yaml_path = data_dir / "data.yaml"
    yaml_path.write_text("path: .\ntrain: images/train\nval: images/train\n", encoding="utf-8")

    img1 = data_dir / "images" / "train" / "sample1.png"
    _write_image(img1, (255, 0, 0))
    _write_map(img1.with_name("sample1_normal.png"), 0.5)
    _write_map(img1.with_name("sample1_roughness.png"), 0.4)
    _write_map(img1.with_name("sample1_ao.png"), 1.0)
    _write_map(img1.with_name("sample1_height.png"), 0.6)
    _write_map(img1.with_name("sample1_metallic.png"), 0.1)

    img2 = data_dir / "images" / "train" / "sample2.png"
    _write_image(img2, (0, 255, 0))
    _write_map(img2.with_name("sample2_normal.png"), 0.7)
    _write_map(img2.with_name("sample2_roughness.png"), 0.3)
    _write_map(img2.with_name("sample2_ao.png"), 0.8)
    _write_map(img2.with_name("sample2_height.png"), 0.2)

    label = "0 0.5 0.5 0.2 0.2\n"
    (data_dir / "labels" / "train" / "sample1.txt").write_text(label, encoding="utf-8")
    (data_dir / "labels" / "train" / "sample2.txt").write_text(label, encoding="utf-8")

    (data_dir / "texts" / "sample1.txt").write_text("red creature", encoding="utf-8")
    (data_dir / "texts" / "sample2.txt").write_text("green creature", encoding="utf-8")

    dataset = MultiModalYOLODataset(
        data_cfg=yaml_path,
        split="train",
        imgsz=64,
        text_dim=32,
        augmentation=AugmentationConfig(mosaic=False, block_mix_prob=0.0),
        seed=0,
    )
    return dataset


def test_dataset_shapes(sample_dataset: MultiModalYOLODataset) -> None:
    sample = sample_dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["pbr"].shape == (7, 64, 64)
    assert sample["pbr_mask"].shape == (7, 64, 64)
    assert sample["text"].shape[0] == 32
    assert sample["targets"]["boxes"].shape[-1] == 4

    sample2 = sample_dataset[1]
    assert sample2["pbr_mask"][6].sum() == 0


def test_collate(sample_dataset: MultiModalYOLODataset) -> None:
    batch = [sample_dataset[0], sample_dataset[1]]
    collated = multimodal_collate(batch)
    assert collated["image"].shape == (2, 3, 64, 64)
    assert collated["pbr"].shape == (2, 7, 64, 64)
    assert collated["pbr_mask"].shape == (2, 7, 64, 64)
    assert len(collated["targets"]) == 2
