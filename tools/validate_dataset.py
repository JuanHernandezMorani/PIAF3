#!/usr/bin/env python3
"""
Valida la integridad del dataset multimodal (RGB + PBR + meta + anotaciones).
Genera un reporte con conteos por clase, faltantes y estadísticas básicas.
"""
import os, sys, json, argparse
from collections import defaultdict
from pathlib import Path
from typing import List

CLASSES = ["eyes","wings","body","extremities","fangs","claws","extra","head","mouth","heart",
           "cracks","cristal","flower","zombie_zone","armor","sky","stars","aletas"]

def read_split_list(split_file: str) -> List[str]:
    items = []
    for line in Path(split_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): 
            continue
        items.append(line)
    return items

def rel_to_name(rel_path: str) -> str:
    # Ej: images/foo.png -> foo
    base = os.path.basename(rel_path)
    name, _ = os.path.splitext(base)
    return name

def exists_all(root: str, name: str, require_metal: bool = False) -> dict:
    paths = {
        "rgb": f"data/images/{name}.png",
        "n":   f"data/maps/normal/{name}_n.png",
        "r":   f"data/maps/roughness/{name}_r.png",
        "s":   f"data/maps/specular/{name}_s.png",
        "e":   f"data/maps/emissive/{name}_e.png",
        "m":   f"data/maps/metalness/{name}_m.png",
        "meta":f"data/meta/{name}.json",
        "ann": f"data/ann/{name}.txt",
    }
    out = {k: os.path.exists(os.path.join(root, v)) for k,v in paths.items()}
    if not require_metal:
        out["m"] = True  # no exigir metalness si no es obligatorio
    return out, paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=str)
    ap.add_argument("--split", default="splits/train.txt")
    ap.add_argument("--require-metalness", action="store_true")
    args = ap.parse_args()

    items = read_split_list(args.split)
    missing_counts = defaultdict(int)
    total = 0

    for rel in items:
        name = rel_to_name(rel)
        ok, paths = exists_all(args.root, name, args.require_metalness)
        total += 1
        for k, v in ok.items():
            if not v:
                missing_counts[k] += 1

    print(f"[validate] total items in split: {total}")
    if missing_counts:
        print("[validate] missing by key:")
        for k, c in missing_counts.items():
            print(f"  - {k}: {c}")
    else:
        print("[validate] OK: no missing files.")

    # TODO: parsear anotaciones y sumar por clase (skeleton para Codex Plus)
    print("[validate] TODO: conteo por clase y verificación de polígonos YOLO-seg.")

if __name__ == "__main__":
    main()
