#!/usr/bin/env python3
"""Herramienta de validación integral para datasets multimodales.

Comprueba la presencia de todos los assets RGB + PBR, valida tamaños entre
modalidades, inspecciona anotaciones YOLO-seg y genera un reporte agregado por
clase. Los hallazgos se imprimen en consola y, opcionalmente, se guardan como
JSON para integrarlos en pipelines automáticos.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

from PIL import Image


CLASSES = [
    "eyes",
    "wings",
    "body",
    "extremities",
    "fangs",
    "claws",
    "extra",
    "head",
    "mouth",
    "heart",
    "cracks",
    "cristal",
    "flower",
    "zombie_zone",
    "armor",
    "sky",
    "stars",
    "aletas",
]


ANN_REQUIRED_POINTS = 3  # número mínimo de vértices por polígono


@dataclass
class SampleIssues:
    """Aglutina problemas detectados para un asset individual."""

    missing: Dict[str, str] = field(default_factory=dict)
    size_mismatch: Dict[str, str] = field(default_factory=dict)
    ann_errors: List[str] = field(default_factory=list)
    meta_missing_keys: List[str] = field(default_factory=list)

    def extend(self, other: "SampleIssues") -> None:
        self.missing.update(other.missing)
        self.size_mismatch.update(other.size_mismatch)
        self.ann_errors.extend(other.ann_errors)
        self.meta_missing_keys.extend(other.meta_missing_keys)

    def has_issues(self) -> bool:
        return bool(self.missing or self.size_mismatch or self.ann_errors or self.meta_missing_keys)


def read_split_list(split_file: str) -> List[str]:
    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"Split no encontrado: {split_file}")

    items: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    return items


def rel_to_name(rel_path: str) -> str:
    base = os.path.basename(rel_path)
    name, _ = os.path.splitext(base)
    return name


def item_paths(root: str, name: str) -> Mapping[str, Path]:
    base = Path(root)
    return {
        "rgb": base / "data" / "images" / f"{name}.png",
        "normal": base / "data" / "maps" / "normal" / f"{name}_n.png",
        "roughness": base / "data" / "maps" / "roughness" / f"{name}_r.png",
        "specular": base / "data" / "maps" / "specular" / f"{name}_s.png",
        "emissive": base / "data" / "maps" / "emissive" / f"{name}_e.png",
        "metalness": base / "data" / "maps" / "metalness" / f"{name}_m.png",
        "meta": base / "data" / "meta" / f"{name}.json",
        "ann": base / "data" / "ann" / f"{name}.txt",
    }


def collect_existing_assets(paths: Mapping[str, Path], require_metalness: bool) -> SampleIssues:
    issues = SampleIssues()
    for key, path in paths.items():
        if key == "metalness" and not require_metalness:
            continue
        if not path.exists():
            issues.missing[key] = str(path)
    return issues


def validate_image_sizes(paths: Mapping[str, Path]) -> SampleIssues:
    issues = SampleIssues()
    ref_size: Optional[tuple[int, int]] = None
    ref_key = ""
    for key in ("rgb", "normal", "roughness", "specular", "emissive", "metalness"):
        path = paths[key]
        if not path.exists():
            continue
        try:
            with Image.open(path) as im:
                size = im.size
        except Exception as exc:  # pragma: no cover - errores por I/O
            issues.size_mismatch[key] = f"Error al abrir ({exc})"
            continue

        if ref_size is None:
            ref_size = size
            ref_key = key
        elif size != ref_size:
            issues.size_mismatch[key] = f"{size} != {ref_size} ({ref_key})"
    return issues


def _parse_float(value: str, line_no: int, ann_path: Path) -> float:
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - validación I/O
        raise ValueError(f"{ann_path.name}: línea {line_no}: valor no numérico '{value}'") from exc


def validate_annotation(ann_path: Path, class_counts: MutableMapping[int, int]) -> List[str]:
    errors: List[str] = []
    if not ann_path.exists():
        errors.append("anotación ausente")
        return errors

    lines = ann_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        errors.append("archivo vacío")
        return errors

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 1 + 2 * ANN_REQUIRED_POINTS:
            errors.append(f"línea {idx}: polígono requiere >= {ANN_REQUIRED_POINTS} puntos")
            continue

        if (len(parts) - 1) % 2 != 0:
            errors.append(f"línea {idx}: coordenadas no vienen en pares x/y")
            continue

        try:
            class_id = int(float(parts[0]))
        except ValueError:
            errors.append(f"línea {idx}: class_id inválido '{parts[0]}'")
            continue

        if not (0 <= class_id < len(CLASSES)):
            errors.append(f"línea {idx}: class_id fuera de rango ({class_id})")
            continue

        coords = []
        for value in parts[1:]:
            coord = _parse_float(value, idx, ann_path)
            if not 0.0 <= coord <= 1.0:
                errors.append(f"línea {idx}: coordenada fuera de [0,1] ({coord:.3f})")
            coords.append(coord)

        # Si hay errores de coordenadas seguimos contando la clase, evita falsos negativos
        class_counts[class_id] += 1
    return errors


def validate_meta(meta_path: Path, required_keys: Iterable[str]) -> List[str]:
    if not meta_path.exists():
        return ["meta ausente"]

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - validación I/O
        return [f"json inválido ({exc})"]

    missing = [key for key in required_keys if key not in data]
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida assets multimodales y anotaciones YOLO-seg.")
    parser.add_argument("--root", default=".", help="Directorio raíz que contiene /data y /splits")
    parser.add_argument("--split", default="splits/train.txt", help="Archivo con rutas relativas de imágenes")
    parser.add_argument("--require-metalness", action="store_true", help="Fuerza la presencia del mapa metalness")
    parser.add_argument(
        "--report-json",
        default=None,
        help="Ruta opcional para escribir un reporte JSON con los hallazgos",
    )
    parser.add_argument(
        "--meta-required",
        nargs="*",
        default=["dominant_colors", "luminance_lab", "saturation", "contrast"],
        help="Claves esperadas en meta/{name}.json",
    )
    args = parser.parse_args()

    items = read_split_list(args.split)
    if not items:
        print("[validate] Split vacío, nada que validar.")
        return

    total = len(items)
    problematic = 0
    aggregate_missing = defaultdict(int)
    aggregate_size = defaultdict(int)
    aggregate_ann_errors = 0
    aggregate_meta_missing = Counter()
    class_counts: Counter[int] = Counter()
    sample_reports: Dict[str, Dict[str, object]] = {}

    for rel in items:
        name = rel_to_name(rel)
        paths = item_paths(args.root, name)
        sample_issue = SampleIssues()

        sample_issue.extend(collect_existing_assets(paths, args.require_metalness))
        sample_issue.extend(validate_image_sizes(paths))
        ann_errors = validate_annotation(paths["ann"], class_counts)
        if ann_errors:
            sample_issue.ann_errors.extend(ann_errors)
        if args.meta_required:
            missing_meta = validate_meta(paths["meta"], args.meta_required)
            if missing_meta:
                sample_issue.meta_missing_keys.extend(missing_meta)

        if sample_issue.has_issues():
            problematic += 1
            for key in sample_issue.missing:
                aggregate_missing[key] += 1
            for key in sample_issue.size_mismatch:
                aggregate_size[key] += 1
            aggregate_ann_errors += len(sample_issue.ann_errors)
            for key in sample_issue.meta_missing_keys:
                aggregate_meta_missing[key] += 1
            sample_reports[name] = {
                "missing": sample_issue.missing,
                "size_mismatch": sample_issue.size_mismatch,
                "ann_errors": sample_issue.ann_errors,
                "meta_missing_keys": sample_issue.meta_missing_keys,
            }
    class_report = {CLASSES[idx]: class_counts[idx] for idx in range(len(CLASSES)) if class_counts[idx]}

    print(f"[validate] assets evaluados: {total}")
    print(f"[validate] muestras con incidencias: {problematic}")
    if aggregate_missing:
        print("[validate] archivos faltantes por tipo:")
        for key, count in sorted(aggregate_missing.items(), key=lambda kv: kv[0]):
            print(f"  - {key}: {count}")
    else:
        print("[validate] ✅ Todos los assets requeridos están presentes.")

    if aggregate_size:
        print("[validate] discrepancias de tamaño detectadas:")
        for key, count in sorted(aggregate_size.items(), key=lambda kv: kv[0]):
            print(f"  - {key}: {count}")

    if aggregate_ann_errors:
        print(f"[validate] anotaciones con errores: {aggregate_ann_errors}")
    else:
        print("[validate] ✅ Anotaciones pasan validaciones básicas.")

    if aggregate_meta_missing:
        print("[validate] llaves ausentes en meta:")
        for key, count in sorted(aggregate_meta_missing.items(), key=lambda kv: kv[0]):
            print(f"  - {key}: {count}")

    if class_report:
        print("[validate] polígonos por clase:")
        for name, count in sorted(class_report.items(), key=lambda kv: kv[0]):
            print(f"  - {name}: {count}")
    else:
        print("[validate] ⚠️ No se encontraron polígonos válidos en las anotaciones.")

    if args.report_json:
        payload = {
            "split": args.split,
            "root": args.root,
            "missing": aggregate_missing,
            "size_mismatch": aggregate_size,
            "annotation_errors": aggregate_ann_errors,
            "meta_missing": aggregate_meta_missing,
            "class_report": class_report,
            "samples": sample_reports,
        }
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"[validate] Reporte JSON escrito en {args.report_json}")


if __name__ == "__main__":
    main()
