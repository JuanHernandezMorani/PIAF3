from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.model.build import build_yolo11_multi


def test_model_forward_smoke() -> None:
    pytest.importorskip("ultralytics")
    weights_path = os.environ.get("YOLO_PRETRAINED")
    if not weights_path or not Path(weights_path).exists():
        pytest.skip("YOLO_PRETRAINED environment variable must point to a valid checkpoint")

    cfg = {
        "pretrained": weights_path,
        "context": {
            "text_dim": 256,
            "pbr_channels": [
                "normal_x",
                "normal_y",
                "normal_z",
                "rough",
                "ao",
                "height",
                "metallic",
            ],
            "adapter_dim": 128,
            "hidden_dim": 128,
        },
        "film": {"targets": [], "hidden": 128, "dropout": 0.0},
        "aux": {"enable": False},
    }
    model = build_yolo11_multi(cfg).eval()
    images = torch.rand(1, 3, 64, 64)
    pbr = torch.rand(1, 7, 64, 64)
    text = torch.rand(1, 256)
    outputs = model(images, pbr=pbr, text=text)
    assert "pred" in outputs
    assert "aux" in outputs
