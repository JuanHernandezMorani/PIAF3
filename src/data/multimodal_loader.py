import os
import json
import random
from typing import Dict, Any, Tuple, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

CLASSES = ["eyes","wings","body","extremities","fangs","claws","extra","head","mouth","heart",
           "cracks","cristal","flower","zombie_zone","armor","sky","stars","aletas"]

def load_img(path: str, mode="RGB") -> np.ndarray:
    img = Image.open(path).convert(mode)
    return np.array(img, dtype=np.uint8)

def add_coordconv(h, w):
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, h, dtype=np.float32),
        np.linspace(-1, 1, w, dtype=np.float32),
        indexing='ij'
    )
    return xx[...,None], yy[...,None]  # (H,W,1)

class MultimodalYoloDataset(Dataset):
    """Dataset multimodal para YOLO-seg con canales PBR sincronizados."""

    def __init__(
        self,
        root: str,
        split_file: str,
        imgsz: int = 512,
        use_metalness: bool = False,
        use_coordconv: bool = False,
        augment: bool = True,
        seed: Optional[int] = None,
        pbr_dropout_prob: float = 0.0,
    ):
        self.root = root
        self.imgsz = imgsz
        self.use_metalness = use_metalness
        self.use_coordconv = use_coordconv
        self.augment = augment
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.pbr_dropout_prob = float(max(0.0, pbr_dropout_prob))

        with open(split_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
        self.items = [os.path.splitext(os.path.basename(p))[0] for p in lines]

    def __len__(self) -> int:
        return len(self.items)

    def _paths(self, name: str) -> Dict[str, str]:
        base = "data"
        return {
            "rgb": os.path.join(self.root, f"{base}/images/{name}.png"),
            "n": os.path.join(self.root, f"{base}/maps/normal/{name}_n.png"),
            "r": os.path.join(self.root, f"{base}/maps/roughness/{name}_r.png"),
            "s": os.path.join(self.root, f"{base}/maps/specular/{name}_s.png"),
            "e": os.path.join(self.root, f"{base}/maps/emissive/{name}_e.png"),
            "m": os.path.join(self.root, f"{base}/maps/metalness/{name}_m.png"),
            "meta": os.path.join(self.root, f"{base}/meta/{name}.json"),
            "ann": os.path.join(self.root, f"{base}/ann/{name}.txt"),
        }

    def _context_vec(self, meta_path: str) -> np.ndarray:
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}

        lum = float(meta.get("luminance_lab", meta.get("lum", 50.0)))
        sat = float(meta.get("saturation", meta.get("sat", 0.5)))
        contrast = float(meta.get("contrast", 0.5))

        def _to_unit(value: float, low: float, high: float) -> float:
            if high == low:
                return 0.0
            scaled = (value - low) / (high - low)
            return float(np.clip(scaled, 0.0, 1.0))

        lum_unit = _to_unit(lum, 0.0, 100.0)
        sat_unit = _to_unit(sat, 0.0, 1.0)
        contrast_unit = _to_unit(contrast, 0.0, 100.0)

        vec = [lum_unit, sat_unit, contrast_unit]

        cols = meta.get("dominant_colors", [])[:5]
        flat: List[float] = []
        for c in cols:
            if isinstance(c, list) and len(c) == 3:
                flat.extend([float(v) / 255.0 for v in c])
        while len(flat) < 15:
            flat.append(0.0)
        vec.extend(flat)

        arr = np.array(vec, dtype=np.float32)
        arr = arr * 2.0 - 1.0
        return arr

    def _load_modalities(self, paths: Dict[str, str]) -> np.ndarray:
        rgb = load_img(paths["rgb"], "RGB").astype(np.float32) / 255.0
        nrm = load_img(paths["n"], "RGB").astype(np.float32) / 255.0
        rgh = load_img(paths["r"], "L").astype(np.float32) / 255.0[..., None]
        spc = load_img(paths["s"], "L").astype(np.float32) / 255.0[..., None]
        ems = load_img(paths["e"], "L").astype(np.float32) / 255.0[..., None]
        chans = [rgb, nrm, rgh, spc, ems]
        if self.use_metalness and os.path.exists(paths["m"]):
            mtl = load_img(paths["m"], "L").astype(np.float32) / 255.0[..., None]
            chans.append(mtl)
        h, w = rgb.shape[:2]
        if self.use_coordconv:
            xx, yy = add_coordconv(h, w)
            chans.extend([xx, yy])
        x = np.concatenate(chans, axis=-1)
        return x

    def _load_polygons(self, ann_path: str, width: int, height: int) -> Tuple[List[np.ndarray], List[int]]:
        polys: List[np.ndarray] = []
        classes: List[int] = []
        if not os.path.exists(ann_path):
            return polys, classes

        with open(ann_path, "r", encoding="utf-8") as fh:
            for raw in fh.readlines():
                stripped = raw.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                coords = np.array([float(p) for p in parts[1:]], dtype=np.float32)
                pts = coords.reshape(-1, 2)
                pts[:, 0] *= float(width)
                pts[:, 1] *= float(height)
                polys.append(pts)
                classes.append(class_id)
        return polys, classes

    def _apply_augmentations(self, arr: np.ndarray, polygons: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        h, w = arr.shape[0], arr.shape[1]
        polys = [poly.copy() for poly in polygons]

        if self.augment and self.rng.random() < 0.5:
            arr = arr[:, ::-1, :]
            for poly in polys:
                poly[:, 0] = float(w) - poly[:, 0]

        if self.augment and self.rng.random() < 0.5:
            arr = arr[::-1, :, :]
            for poly in polys:
                poly[:, 1] = float(h) - poly[:, 1]

        return arr, polys

    def _apply_pbr_dropout(self, arr: np.ndarray) -> np.ndarray:
        if self.pbr_dropout_prob <= 0.0:
            return arr

        dropped = arr.copy()
        slices = {
            "normals": slice(3, 6),
            "rough": slice(6, 7),
            "spec": slice(7, 8),
            "emiss": slice(8, 9),
        }

        for key, slc in slices.items():
            if slc.stop > dropped.shape[-1]:
                continue
            if self.rng.random() < self.pbr_dropout_prob:
                dropped[..., slc] = 0.0

        return dropped

    def _extract_pbr_targets(self, arr: np.ndarray) -> Dict[str, np.ndarray]:
        slices = {
            "normals": slice(3, 6),
            "rough": slice(6, 7),
            "spec": slice(7, 8),
            "emiss": slice(8, 9),
        }

        targets: Dict[str, np.ndarray] = {}
        for key, slc in slices.items():
            if slc.stop <= arr.shape[-1]:
                targets[key] = arr[..., slc].copy()

        return targets

    def _to_chw_tensor(self, np_arr: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np_arr.transpose(2, 0, 1)).float()
        return self._resize(tensor)

    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unsqueeze(0)
        resized = F.interpolate(tensor, size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False)
        return resized.squeeze(0)

    def __getitem__(self, idx: int):
        name = self.items[idx]
        paths = self._paths(name)
        x_np = self._load_modalities(paths)
        height, width = x_np.shape[0], x_np.shape[1]

        polys_px, classes = self._load_polygons(paths["ann"], width, height)
        x_np, polys_px = self._apply_augmentations(x_np, polys_px)

        pbr_targets_np = self._extract_pbr_targets(x_np)

        x_np = self._apply_pbr_dropout(x_np)

        x_tensor = self._to_chw_tensor(x_np)

        pbr_targets = {
            key: self._to_chw_tensor(value) for key, value in pbr_targets_np.items()
        }

        scale_x = self.imgsz / float(width)
        scale_y = self.imgsz / float(height)
        scaled_polys: List[torch.Tensor] = []
        for poly in polys_px:
            scaled = poly.copy()
            scaled[:, 0] *= scale_x
            scaled[:, 1] *= scale_y
            norm = np.clip(scaled / float(self.imgsz), 0.0, 1.0)
            scaled_polys.append(torch.from_numpy(norm.astype(np.float32)))

        ctx = torch.from_numpy(self._context_vec(paths["meta"]))

        targets = {
            "name": name,
            "classes": torch.as_tensor(classes, dtype=torch.long),
            "segments": scaled_polys,
            "orig_size": torch.tensor([height, width], dtype=torch.float32),
            "img_size": torch.tensor([self.imgsz, self.imgsz], dtype=torch.float32),
            "pbr": pbr_targets,
        }

        return x_tensor, ctx, targets

def collate_fn(batch):
    xs, ctxs, tgts = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ctxs = torch.stack(ctxs, dim=0)
    return xs, ctxs, tgts
