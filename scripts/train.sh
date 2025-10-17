#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --cfg configs/train_multi.yaml \
  --data configs/data.yaml \
  --pretrained /ruta/a/yolo.pt \
  --epochs 100 --batch 16 --imgsz 640 \
  --pbr --text --project runs/multi --name exp
