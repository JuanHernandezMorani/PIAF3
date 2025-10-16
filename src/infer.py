"""
Inferencia multimodal: carga RGB + PBR + meta, arma context_vec y ejecuta el modelo.
"""
import os, json, torch
import numpy as np
from typing import Dict, Any
from PIL import Image

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

class MultimodalPredictor:
    def __init__(self, model, cfg=None):
        self.model = model
        self.cfg = cfg or {}

    def _load_modalities(self, root, name, use_metalness=False):
        def p(rel): return os.path.join(root, rel)
        rgb = p(f"data/images/{name}.png")
        n   = p(f"data/maps/normal/{name}_n.png")
        r   = p(f"data/maps/roughness/{name}_r.png")
        s   = p(f"data/maps/specular/{name}_s.png")
        e   = p(f"data/maps/emissive/{name}_e.png")
        m   = p(f"data/maps/metalness/{name}_m.png")
        meta= p(f"data/meta/{name}.json")
        return {"rgb": rgb, "n":n, "r":r, "s":s, "e":e, "m":m, "meta":meta}

    def predict(self, root: str, name: str) -> Dict[str, Any]:
        # TODO: preparar tensores igual que en el dataloader y hacer forward
        return {"boxes": [], "masks": [], "classes": [], "scores": []}
