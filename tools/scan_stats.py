#!/usr/bin/env python3
"""
Escanea imágenes RGB para calcular estadísticas globales (luminancia, saturación,
contraste, colores dominantes) y escribe/actualiza meta/{name}.json.
"""
import os, json, argparse
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import numpy as np

try:
    from sklearn.cluster import KMeans
    SK_OK = True
except Exception:
    SK_OK = False

def rgb_to_lab_luminance(img: np.ndarray) -> float:
    # Aproximación rápida usando BT.709 -> Y (proxy de luminancia)
    r, g, b = img[...,0], img[...,1], img[...,2]
    y = 0.2126*r + 0.7152*g + 0.0722*b
    return float(y.mean())

def hsv_saturation_mean(img: np.ndarray) -> float:
    # Convert simple RGB->HSV (0..1) para estimar saturación media
    img01 = img.astype(np.float32) / 255.0
    r, g, b = img01[...,0], img01[...,1], img01[...,2]
    cmax = np.max(img01, axis=-1)
    cmin = np.min(img01, axis=-1)
    delta = cmax - cmin
    sat = np.where(cmax==0, 0, delta / (cmax + 1e-8))
    return float(sat.mean())

def img_contrast_std(img: np.ndarray) -> float:
    gray = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2])
    return float(np.std(gray))

def kmeans_colors(img: np.ndarray, k: int=3) -> list:
    if not SK_OK:
        return []
    h, w, _ = img.shape
    sample = img.reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(sample)
    centers = km.cluster_centers_.clip(0,255).astype(np.uint8).tolist()
    return centers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=str)
    ap.add_argument("--split", default="splits/train.txt", type=str)
    ap.add_argument("--k", default=3, type=int)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    lines = [l.strip() for l in Path(args.split).read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]

    for rel in lines:
        name = os.path.splitext(os.path.basename(rel))[0]
        rgb_path = os.path.join(args.root, f"data/images/{name}.png")
        if not os.path.exists(rgb_path):
            print(f"[skip] no RGB: {rgb_path}")
            continue
        img = np.array(Image.open(rgb_path).convert("RGB"))
        meta = {
            "dominant_colors": kmeans_colors(img, args.k),
            "lum": rgb_to_lab_luminance(img),
            "sat": hsv_saturation_mean(img),
            "contrast": img_contrast_std(img),
            "style": "pixel-art",
            "source": "unknown",
            "h": int(img.shape[0]),
            "w": int(img.shape[1]),
        }
        meta_path = os.path.join(args.root, f"data/meta/{name}.json")
        if args.write:
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[write] {meta_path}")
        else:
            print(f"[dry-run] {name}: {meta}")

if __name__ == "__main__":
    main()
