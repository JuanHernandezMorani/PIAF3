# YOLOv11 Multimodal (RGB + PBR + Context) — Project Skeleton

**Fecha:** 2025-10-16

Este esqueleto implementa la estructura propuesta para entrenar un modelo YOLOv11 (seg) modificado
con entradas multi-canal (RGB + normales + roughness + specular + emissive [+ metalness] [+ coordconv])
y condicionamiento por **contexto** (metadatos/“texto”) usando **FiLM**.

> Nota: Son **plantillas** listas para que Codex Plus complete funciones y lógica específica.
> Incluye tests, scripts y YAML base.

## Estructura

```
yolo11_multimodal_skeleton/
  ├─ data/
  │   ├─ images/            # RGB
  │   ├─ maps/
  │   │   ├─ normal/
  │   │   ├─ roughness/
  │   │   ├─ specular/
  │   │   ├─ emissive/
  │   │   └─ metalness/     # opcional
  │   ├─ ann/               # yolo-seg txt o coco json
  │   └─ meta/              # contexto por imagen (.json)
  ├─ splits/
  │   ├─ train.txt
  │   ├─ val.txt
  │   └─ test.txt
  ├─ tools/
  │   ├─ validate_dataset.py
  │   └─ scan_stats.py
  ├─ src/
  │   ├─ data/
  │   │   └─ multimodal_loader.py
  │   ├─ model/
  │   │   ├─ __init__.py
  │   │   ├─ film.py
  │   │   ├─ aux_heads.py
  │   │   └─ yolo11m_multi.yaml
  │   ├─ train.py
  │   ├─ infer.py
  │   └─ tests/
  │       └─ test_multimodal_loader.py
  └─ scripts/
      └─ ablation.py
```

## Clases
```
nc: 18
names: [eyes, wings, body, extremities, fangs, claws, extra, head, mouth, heart,
        cracks, cristal, flower, zombie_zone, armor, sky, stars, aletas]
```

## Prompts sugeridos para Codex Plus
Revisa `PROMPTS_CodexPlus.md` para ejecutar tareas P0..P7.
