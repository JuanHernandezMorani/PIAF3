#!/usr/bin/env python3
"""Escaneo de estadísticas visuales para assets RGB.

Calcula luminancia en espacio Lab, saturación media en HSV, contraste global y
colores dominantes mediante KMeans. Los resultados se escriben en
``data/meta/{name}.json`` fusionando con la metadata existente.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

try:  # pragma: no cover - dependencia opcional
    from sklearn.cluster import KMeans

    SK_OK = True
except Exception:  # pragma: no cover - fallback sin sklearn
    SK_OK = False


@dataclass
class ImageStats:
    luminance_lab: float
    saturation: float
    contrast: float
    dominant_colors: List[List[int]]
    dominant_hex: List[str]
    height: int
    width: int


def _srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    channel = channel / 255.0
    mask = channel <= 0.04045
    channel_lin = np.empty_like(channel, dtype=np.float64)
    channel_lin[mask] = channel[mask] / 12.92
    channel_lin[~mask] = ((channel[~mask] + 0.055) / 1.055) ** 2.4
    return channel_lin


def lab_luminance(img: np.ndarray) -> Tuple[np.ndarray, float]:
    r_lin = _srgb_to_linear(img[..., 0])
    g_lin = _srgb_to_linear(img[..., 1])
    b_lin = _srgb_to_linear(img[..., 2])

    # Conversión a XYZ (D65)
    X = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
    Y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
    Z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin

    # Normalización con el punto blanco D65
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    y_norm = Y / Yn

    epsilon = 0.008856
    kappa = 903.3

    f_y = np.where(y_norm > epsilon, np.cbrt(y_norm), (kappa * y_norm + 16.0) / 116.0)
    L = 116.0 * f_y - 16.0
    return L, float(np.mean(L))


def hsv_saturation_mean(img: np.ndarray) -> float:
    img01 = img.astype(np.float64) / 255.0
    cmax = img01.max(axis=-1)
    cmin = img01.min(axis=-1)
    delta = cmax - cmin
    saturation = np.where(cmax == 0, 0.0, delta / (cmax + 1e-8))
    return float(np.mean(saturation))


def image_contrast(l_channel: np.ndarray) -> float:
    return float(np.std(l_channel))


def kmeans_colors(img: np.ndarray, k: int, max_samples: int = 25000) -> List[List[int]]:
    if not SK_OK or k <= 0:
        return []

    pixels = img.reshape(-1, 3).astype(np.float32)
    if len(pixels) > max_samples:
        idx = np.random.default_rng(42).choice(len(pixels), size=max_samples, replace=False)
        pixels = pixels[idx]

    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    km.fit(pixels)
    centers = km.cluster_centers_.clip(0, 255).astype(np.uint8)
    return centers.tolist()


def rgb_list_to_hex(colors: Iterable[Iterable[int]]) -> List[str]:
    result: List[str] = []
    for rgb in colors:
        r, g, b = (int(v) for v in rgb)
        result.append(f"#{r:02x}{g:02x}{b:02x}")
    return result


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def compute_stats(path: Path, k: int) -> ImageStats:
    rgb = load_image(path)
    l_channel, luminance_mean = lab_luminance(rgb)
    saturation = hsv_saturation_mean(rgb)
    contrast = image_contrast(l_channel)
    colors = kmeans_colors(rgb, k)
    stats = ImageStats(
        luminance_lab=float(luminance_mean),
        saturation=float(saturation),
        contrast=float(contrast),
        dominant_colors=colors,
        dominant_hex=rgb_list_to_hex(colors),
        height=int(rgb.shape[0]),
        width=int(rgb.shape[1]),
    )
    return stats


def merge_meta(meta_path: Path, stats: ImageStats) -> Dict[str, object]:
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - validación I/O
            data = {}
    else:
        data = {}

    data.update(asdict(stats))
    return data


def resolve_items(split_file: Path) -> List[str]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split no encontrado: {split_file}")
    return [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula estadísticas visuales y actualiza meta/*.json")
    parser.add_argument("--root", default=".", help="Directorio raíz del dataset")
    parser.add_argument("--split", default="splits/train.txt", help="Archivo con rutas relativas (images/*.png)")
    parser.add_argument("--k", type=int, default=3, help="Número de clusters KMeans para colores dominantes")
    parser.add_argument("--dry-run", action="store_true", help="No escribe archivos, solo muestra resultados")
    parser.add_argument("--only", nargs="*", default=None, help="Subset opcional de nombres a procesar")
    args = parser.parse_args()

    split_file = Path(args.split)
    items = resolve_items(split_file)
    if args.only:
        targets = {name.lower() for name in args.only}
        items = [item for item in items if rel_to_name(item).lower() in targets]

    if not items:
        print("[scan] Split vacío, nada que procesar.")
        return

    if not SK_OK:
        print("[scan] Advertencia: scikit-learn no disponible, se omiten colores dominantes.")

    for rel in items:
        name = rel_to_name(rel)
        rgb_path = Path(args.root) / "data" / "images" / f"{name}.png"
        if not rgb_path.exists():
            print(f"[scan] ⚠️ RGB ausente: {rgb_path}")
            continue

        stats = compute_stats(rgb_path, args.k)
        meta_path = Path(args.root) / "data" / "meta" / f"{name}.json"
        updated_meta = merge_meta(meta_path, stats)

        if args.dry_run:
            print(f"[scan] {name}: {json.dumps(updated_meta, ensure_ascii=False)}")
            continue

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(updated_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[scan] ✅ meta actualizada -> {meta_path}")


def rel_to_name(rel_path: str) -> str:
    base = os.path.basename(rel_path)
    name, _ = os.path.splitext(base)
    return name


if __name__ == "__main__":
    main()
