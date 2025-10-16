import os, json, random
from typing import Dict, Any, Tuple, List
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
    def __init__(self, root: str, split_file: str, imgsz: int = 512,
                 use_metalness: bool = False, use_coordconv: bool = False):
        self.root = root
        self.imgsz = imgsz
        self.use_metalness = use_metalness
        self.use_coordconv = use_coordconv

        with open(split_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
        self.items = [os.path.splitext(os.path.basename(p))[0] for p in lines]

    def __len__(self): return len(self.items)

    def _paths(self, name: str) -> Dict[str,str]:
        base = f"data"
        return {
            "rgb": os.path.join(self.root, f"{base}/images/{name}.png"),
            "n":   os.path.join(self.root, f"{base}/maps/normal/{name}_n.png"),
            "r":   os.path.join(self.root, f"{base}/maps/roughness/{name}_r.png"),
            "s":   os.path.join(self.root, f"{base}/maps/specular/{name}_s.png"),
            "e":   os.path.join(self.root, f"{base}/maps/emissive/{name}_e.png"),
            "m":   os.path.join(self.root, f"{base}/maps/metalness/{name}_m.png"),
            "meta":os.path.join(self.root, f"{base}/meta/{name}.json"),
            "ann": os.path.join(self.root, f"{base}/ann/{name}.txt"),
        }

    def _context_vec(self, meta_path: str) -> np.ndarray:
        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
        except Exception:
            meta = {"lum":0.5,"sat":0.5,"contrast":0.5,"dominant_colors":[]}
        vec = [
            float(meta.get("lum", 0.5)),
            float(meta.get("sat", 0.5)),
            float(meta.get("contrast", 0.5)),
        ]
        cols = meta.get("dominant_colors", [])[:5]
        flat = []
        for c in cols:
            if isinstance(c, list) and len(c)==3:
                flat.extend([v/255.0 for v in c])
        while len(flat) < 15:
            flat.append(0.0)
        vec.extend(flat)
        return np.array(vec, dtype=np.float32)

    def _load_modalities(self, paths: Dict[str,str]) -> np.ndarray:
        rgb = load_img(paths["rgb"], "RGB").astype(np.float32)/255.0
        nrm = load_img(paths["n"], "RGB").astype(np.float32)/255.0
        rgh = load_img(paths["r"], "L").astype(np.float32)/255.0[...,None]
        spc = load_img(paths["s"], "L").astype(np.float32)/255.0[...,None]
        ems = load_img(paths["e"], "L").astype(np.float32)/255.0[...,None]
        chans = [rgb, nrm, rgh, spc, ems]
        if self.use_metalness and os.path.exists(paths["m"]):
            mtl = load_img(paths["m"], "L").astype(np.float32)/255.0[...,None]
            chans.append(mtl)
        h, w = rgb.shape[:2]
        if self.use_coordconv:
            xx, yy = add_coordconv(h, w)
            chans.extend([xx, yy])
        x = np.concatenate(chans, axis=-1)  # (H,W,C)
        return x

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        pil = Image.fromarray((arr*255.0).clip(0,255).astype(np.uint8))
        pil = pil.resize((self.imgsz, self.imgsz), Image.NEAREST)
        out = np.array(pil).astype(np.float32)/255.0
        return out

    def __getitem__(self, idx: int):
        name = self.items[idx]
        paths = self._paths(name)
        x = self._load_modalities(paths)
        x = self._resize(x)

        ctx = self._context_vec(paths["meta"])
        targets = {"name": name, "polygons": [], "classes": []}  # placeholder

        x = torch.from_numpy(x.transpose(2,0,1))  # (C,H,W)
        ctx = torch.from_numpy(ctx)               # (D,)
        return x, ctx, targets

def collate_fn(batch):
    xs, ctxs, tgts = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ctxs = torch.stack(ctxs, dim=0)
    return xs, ctxs, tgts
