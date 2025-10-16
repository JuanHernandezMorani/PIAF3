import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from src.data.multimodal_loader import MultimodalYoloDataset, collate_fn


def _ensure_dirs(root: Path) -> None:
    (root / "data/images").mkdir(parents=True, exist_ok=True)
    (root / "data/maps/normal").mkdir(parents=True, exist_ok=True)
    (root / "data/maps/roughness").mkdir(parents=True, exist_ok=True)
    (root / "data/maps/specular").mkdir(parents=True, exist_ok=True)
    (root / "data/maps/emissive").mkdir(parents=True, exist_ok=True)
    (root / "data/maps/metalness").mkdir(parents=True, exist_ok=True)
    (root / "data/meta").mkdir(parents=True, exist_ok=True)
    (root / "data/ann").mkdir(parents=True, exist_ok=True)
    (root / "splits").mkdir(parents=True, exist_ok=True)


def _save_modalities(root: Path, name: str, size=(64, 64), pattern: str = "grad") -> None:
    from PIL import Image

    h, w = size
    if pattern == "grad":
        xv = np.linspace(0, 255, w, dtype=np.uint8)
        rgb = np.stack([np.tile(xv, (h, 1)) for _ in range(3)], axis=-1)
    else:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

    normal = np.zeros((h, w, 3), dtype=np.uint8)
    roughness = np.zeros((h, w), dtype=np.uint8)
    specular = np.zeros((h, w), dtype=np.uint8)
    emissive = np.zeros((h, w), dtype=np.uint8)

    Image.fromarray(rgb).save(root / "data/images" / f"{name}.png")
    Image.fromarray(normal).save(root / "data/maps/normal" / f"{name}_n.png")
    Image.fromarray(roughness).save(root / "data/maps/roughness" / f"{name}_r.png")
    Image.fromarray(specular).save(root / "data/maps/specular" / f"{name}_s.png")
    Image.fromarray(emissive).save(root / "data/maps/emissive" / f"{name}_e.png")


def _write_meta(root: Path, name: str, meta: dict) -> None:
    (root / "data/meta" / f"{name}.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def _write_ann(root: Path, name: str, ann_line: str) -> None:
    (root / "data/ann" / f"{name}.txt").write_text(ann_line, encoding="utf-8")


def _write_split(root: Path, names: list[str]) -> None:
    lines = [f"images/{name}.png" for name in names]
    (root / "splits/train.txt").write_text("\n".join(lines), encoding="utf-8")


def _build_basic_dataset(tmp_path: Path, names: list[str], *, pattern: str = "grad") -> None:
    _ensure_dirs(tmp_path)
    for name in names:
        _save_modalities(tmp_path, name, pattern=pattern)
def test_dataset_channels_context_and_targets(tmp_path):
    root = Path(tmp_path)
    _build_basic_dataset(root, ["foo"], pattern="zeros")

    meta = {
        "luminance_lab": 100.0,
        "saturation": 0.25,
        "contrast": 25.0,
        "dominant_colors": [[255, 0, 0], [0, 128, 0]],
    }
    _write_meta(root, "foo", meta)

    ann_line = "0 0.25 0.25 0.75 0.25 0.75 0.75"
    _write_ann(root, "foo", ann_line)
    _write_split(root, ["foo"])

    ds = MultimodalYoloDataset(
        str(root), str(root / "splits/train.txt"), imgsz=64, use_coordconv=True, augment=False
    )

    x, ctx, target = ds[0]

    expected_channels = 3 + 3 + 1 + 1 + 1 + 2
    assert x.shape == (expected_channels, 64, 64)

    # Contexto normalizado a [-1, 1]
    assert ctx.shape[0] == 18
    assert torch.all(ctx <= 1.0 + 1e-6)
    assert torch.all(ctx >= -1.0 - 1e-6)
    assert torch.isclose(ctx[0], torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(ctx[1], torch.tensor(-0.5), atol=1e-5)
    assert torch.isclose(ctx[2], torch.tensor(-0.5), atol=1e-5)

    assert target["name"] == "foo"
    assert target["classes"].dtype == torch.long
    assert target["classes"].tolist() == [0]
    assert len(target["segments"]) == 1
    seg = target["segments"][0]
    expected = torch.tensor(
        [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75]], dtype=torch.float32
    )
    assert torch.allclose(seg, expected, atol=1e-5)


def test_horizontal_flip_sync(tmp_path):
    root = Path(tmp_path)
    _build_basic_dataset(root, ["foo"], pattern="grad")

    meta = {"luminance_lab": 50.0, "saturation": 0.5, "contrast": 10.0, "dominant_colors": []}
    _write_meta(root, "foo", meta)
    ann_line = "1 0.25 0.25 0.75 0.25 0.75 0.75"
    _write_ann(root, "foo", ann_line)
    _write_split(root, ["foo"])

    ds_no_aug = MultimodalYoloDataset(
        str(root), str(root / "splits/train.txt"), imgsz=64, augment=False
    )
    base_x, _, _ = ds_no_aug[0]

    ds = MultimodalYoloDataset(
        str(root), str(root / "splits/train.txt"), imgsz=64, augment=True, seed=1
    )
    x, _, target = ds[0]

    # Verifica flip horizontal sobre primer canal
    assert torch.allclose(x[0, :, 0], base_x[0, :, -1])
    assert torch.allclose(x[0, :, -1], base_x[0, :, 0])

    seg = target["segments"][0]
    expected = torch.tensor(
        [[0.75, 0.25], [0.25, 0.25], [0.25, 0.75]], dtype=torch.float32
    )
    assert torch.allclose(seg, expected, atol=1e-5)


def test_collate_fn_batches(tmp_path):
    root = Path(tmp_path)
    _build_basic_dataset(root, ["foo", "bar"], pattern="zeros")

    meta = {"luminance_lab": 75.0, "saturation": 0.5, "contrast": 50.0, "dominant_colors": []}
    _write_meta(root, "foo", meta)
    _write_meta(root, "bar", meta)
    ann_line = "0 0.1 0.1 0.2 0.1 0.2 0.2"
    _write_ann(root, "foo", ann_line)
    _write_ann(root, "bar", ann_line)
    _write_split(root, ["foo", "bar"])

    ds = MultimodalYoloDataset(
        str(root), str(root / "splits/train.txt"), imgsz=32, augment=False
    )

    batch = [ds[0], ds[1]]
    xs, ctxs, targets = collate_fn(batch)

    assert xs.shape[0] == 2
    assert ctxs.shape[0] == 2
    assert len(targets) == 2
    assert targets[0]["classes"].shape[0] == 1
    assert isinstance(targets[0]["segments"], list)
