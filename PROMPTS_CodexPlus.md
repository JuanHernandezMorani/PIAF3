# Prompts para Codex Plus — Orden recomendado

## P0 — Bootstrap datasets y validadores
[Título: bootstrap_datasets_and_validators]
```
Actúa como arquitecto ML senior. En el repositorio actual crea/valida:
- tools/validate_dataset.py: validación integral de assets RGB+PBR, tamaños, anotaciones seg, reporte por clase.
- tools/scan_stats.py: calcula luminancia (Lab), saturación (HSV), contraste, y colores dominantes (KMeans).
- Actualiza meta/{name}.json con los stats.
- Completa README con instrucciones de uso de tools.
```

## P1 — Dataloader multimodal
[Título: build_multimodal_yolo_dataloader]
```
Implementa src/data/multimodal_loader.py:
- Dataset que concatena canales RGB(3)+N(3)+R(1)+S(1)+E(1)[+M(1)][+Coord(2)].
- Augmentations sincronizadas para imágenes y polígonos.
- context_vec desde meta/*.json normalizado a [-1,1].
- collate_fn y tests en src/tests/test_multimodal_loader.py.
```

## P2 — Patch in_channels YOLOv11-seg
[Título: patch_yolov11_seg_inchannels]
```
Crea y ajusta src/model/yolo11m_multi.yaml y las partes del modelo para aceptar in_channels variables.
Inicializa nuevos canales (Xavier o copia/average). Mantén compatibilidad con 3ch.
```

## P3 — FiLM (gamma/beta) con contexto
[Título: add_film_conditioning]
```
Implementa src/model/film.py y úsalo en backbone/neck con dim_context configurable.
Modifica el forward del modelo para recibir context_vec.
```

## P4 — Heads auxiliares + Input Dropout
[Título: aux_heads_pbr_and_input_dropout]
```
Implementa src/model/aux_heads.py con heads opcionales para normals(3), rough(1), spec(1), emiss(1).
Añade dropout de canales PBR p=0.2 en entrenamiento. Integra pérdidas y logging.
```

## P5 — Training loop
[Título: training_loop_multimodal]
```
Completa src/train.py con loop, optim, scheduler, métricas (mAP, AP por clase),
y guardado de checkpoints (best/last) + export onnx/torchscript.
```

## P6 — Inference API + demo
[Título: inference_api_and_demo]
```
Implementa src/infer.py (MultimodalPredictor) y un notebook demo.
```

## P7 — Ablations
[Título: ablation_studies_context_pbr]
```
Completa scripts/ablation.py para correr RGB-only vs RGB+PBR vs RGB+PBR+FiLM (+/- aux).
Exporta tablas y resumen.
```
