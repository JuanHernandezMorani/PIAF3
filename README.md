# PIAF3 — Herramientas de Dataset

Este repositorio contiene un esqueleto para entrenar un modelo YOLOv11 multimodal
(RGB + PBR + contexto). Las utilidades en `tools/` facilitan la auditoría y el
procesamiento de los assets para garantizar su consistencia antes de
entrenamiento.

## Entrenamiento multimodal

### Requisitos

- Ubuntu 20.04 o superior.
- NVIDIA CUDA 12.x (opcional pero recomendado para acelerar el entrenamiento).
- Python 3.10+ con PyTorch y torchvision compatibles con la versión de CUDA instalada.

Instala las dependencias necesarias mediante:

```bash
pip install -r requirements.txt
```

### Ejecución

```bash
python -m src.train \
  --config configs/train_multi.yaml \
  --pretrained yolo.pt \
  --project runs/multi \
  --name exp1
```

### Estructura del dataset

```
dataset/
├── data.yaml          # describe `path`, `train` y `val`
├── images/
│   ├── train/
│   │   └── *.png | *.jpg
│   └── val/
│       └── *.png | *.jpg
├── labels/
│   ├── train/
│   │   └── *.txt
│   └── val/
│       └── *.txt
├── texts/             # opcional, descripciones por asset
│   └── *.txt
└── PBR maps junto a cada imagen con sufijos:
    *_normal.png, *_roughness.png, *_ao.png,
    *_height.png, *_metallic.png
```

Los mapas PBR deben estar normalizados en el rango `[0, 1]`. Las normales en espacio tangente aceptan valores en `[-1, 1]`, pero el *loader* los remapeará automáticamente a `[0, 1]` si detecta valores fuera del rango esperado.

Los textos pueden residir junto a la imagen (`image.png` + `image.txt`) o en `texts/` con el mismo nombre base.

## Validación de dataset (`tools/validate_dataset.py`)

La validación inspecciona cada elemento del *split* y verifica:

- Presencia de todos los archivos esperados (RGB, mapas PBR, `meta/*.json`, `ann/*.txt`).
- Consistencia dimensional entre RGB y mapas PBR.
- Integridad básica de las anotaciones YOLO-seg (IDs válidos, puntos suficientes, coordenadas normalizadas).
- Existencia de claves críticas en los metadatos (por defecto: `dominant_colors`, `luminance_lab`, `saturation`, `contrast`).
- Conteo agregado de polígonos por clase.

Uso básico:

```bash
python tools/validate_dataset.py \
    --root /ruta/al/dataset \
    --split splits/train.txt \
    --report-json reports/train_report.json
```

Argumentos relevantes:

- `--require-metalness`: obliga a que el mapa `metalness` exista para cada asset.
- `--report-json`: guarda un reporte estructurado para su posterior análisis.
- `--meta-required`: lista opcional de claves que deben aparecer en `meta/{name}.json`.

## Estadísticas visuales (`tools/scan_stats.py`)

Genera estadísticas de color y luminancia para cada RGB del split y las fusiona
en `data/meta/{name}.json`. Calcula:

- Luminancia promedio en espacio Lab.
- Saturación media en espacio HSV.
- Contraste (desviación estándar del canal L*).
- Colores dominantes vía KMeans (`dominant_colors` en RGB y `dominant_hex` en formato hexadecimal).
- Altura y anchura del asset.

Ejemplo de ejecución en modo escritura:

```bash
python tools/scan_stats.py \
    --root /ruta/al/dataset \
    --split splits/train.txt \
    --k 4
```

Argumentos útiles:

- `--dry-run`: muestra el JSON resultante sin modificar archivos.
- `--only nombre1 nombre2`: limita el procesamiento a un subconjunto.
- `--k`: número de clusters para KMeans (requiere `scikit-learn`).

## Flujo recomendado

1. Ejecuta `scan_stats.py --dry-run` para comprobar las estadísticas y,
   posteriormente, ejecútalo sin `--dry-run` para persistirlas en `meta/`.
2. Corre `validate_dataset.py` para confirmar que los assets y metadatos están
   completos antes de iniciar entrenamiento o exportar el dataset.
3. Añade los reportes generados a tu pipeline de CI/CD o documentación.

Ambas herramientas están pensadas para trabajar con splits en formato texto
(`splits/*.txt`) con rutas relativas como `images/example.png`.
