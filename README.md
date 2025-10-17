# PIAF3 — Entrenamiento multimodal YOLOv11

Este repositorio contiene un esqueleto completo para entrenar un modelo YOLOv11
(segmentación) condicionado por entradas RGB, mapas PBR y contexto textual.
Incluye scripts de entrenamiento, validación, exportación y herramientas para la
sanidad del dataset.

## Requisitos rápidos

- Ubuntu 20.04 o superior.
- Python 3.10+.
- PyTorch 2.2–2.3 con CUDA 11/12 (opcional, pero recomendado para GPU).

Instalación mínima:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configura tu dataset

Declara la estructura en `configs/data.yaml`:

```yaml
# rutas
path: /ABSOLUTE/PATH/TO/DATASET
train: images/train
val: images/val

# clases
nc: 3
names: [class_0, class_1, class_2]

# PBR y texto
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
```

Estructura esperada:

```
dataset/
├── images/
│   ├── train/foo.png
│   └── val/bar.png
├── labels/
│   ├── train/foo.txt       # YOLO-seg (cls x y w h [polígonos])
│   └── val/bar.txt
├── texts/
│   └── train/foo.txt       # opcional, string libre ("", si no hay contexto)
└── images/train/
    ├── foo_normal.png
    ├── foo_rough.png
    ├── foo_metal.png
    ├── foo_ao.png
    ├── foo_height.png
    └── foo_curv.png
```

Los mapas PBR que falten se rellenan con ceros. Si un PNG trae canal alfa se
mantiene internamente y la red recibe únicamente RGB + PBR.

## Entrena

Ajusta `configs/train_multi.yaml` y ejecuta:

```bash
python -m src.train \
  --cfg configs/train_multi.yaml \
  --data configs/data.yaml \
  --pretrained /ruta/a/yolo.pt \
  --epochs 100 --batch 16 --imgsz 640 \
  --pbr --text --project runs/multi --name exp
```

Argumentos destacados:

- `--pbr` / `--text`: habilita explícitamente cada modalidad.
- `--resume runs/multi/exp/last.pt`: reanuda entrenamiento (scheduler + scaler incluidos).
- `--device cuda:0` o `cpu`.

El script registra las shapes de entrada, estadísticas de boxes/masks y guarda
los checkpoints en `runs/multi/<name>/` (`last.pt`, `best.pt`, `ema.pt`).

## Valida

```bash
python tools/validate.py \
  --cfg configs/train_multi.yaml \
  --data configs/data.yaml \
  --weights runs/multi/exp/last.pt \
  --output runs/multi/exp/metrics_val.json
```

Se reportan mAP50 y mAP50-95 por clase. Si las modalidades PBR/TXT están
activadas se imprime un bloque de ablations:

- `rgb_only`
- `rgb_pbr`
- `rgb_text`
- `rgb_pbr_text`

## Exporta

```bash
python tools/export.py \
  --cfg configs/train_multi.yaml \
  --data configs/data.yaml \
  --weights runs/multi/exp/best.pt \
  --out exports/yolo-multi.pt \
  --onnx exports/yolo-multi.onnx
```

`--onnx` es opcional y genera un grafo con entradas dinámicas para RGB, PBR y
texto.

## Herramientas de dataset

Antes de entrenar, audita el dataset con:

```bash
python tools/validate_dataset.py --root /ruta/dataset --split splits/train.txt
python tools/scan_stats.py --root /ruta/dataset --split splits/train.txt --k 4
```

`validate_dataset.py` comprueba existencia y consistencia de RGB/PBR/labels,
segmentaciones y metadatos. `scan_stats.py` extrae estadísticas de color y
luminancia (KMeans opcional) y puede integrarse en tu pipeline de CI/CD.
