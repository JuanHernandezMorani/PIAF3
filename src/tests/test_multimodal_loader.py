import os, numpy as np, torch
from src.data.multimodal_loader import MultimodalYoloDataset, collate_fn

def test_shapes_smoke(tmp_path):
    # Crear estructura mínima falsa
    root = tmp_path
    os.makedirs(root/"data/images", exist_ok=True)
    os.makedirs(root/"data/maps/normal", exist_ok=True)
    os.makedirs(root/"data/maps/roughness", exist_ok=True)
    os.makedirs(root/"data/maps/specular", exist_ok=True)
    os.makedirs(root/"data/maps/emissive", exist_ok=True)
    os.makedirs(root/"data/meta", exist_ok=True)
    os.makedirs(root/"data/ann", exist_ok=True)
    os.makedirs(root/"splits", exist_ok=True)

    import PIL.Image as Image
    def save_rgb(p, size=(64,64)):
        Image.fromarray(np.zeros((*size,3), dtype=np.uint8)).save(p)
    def save_gray(p, size=(64,64)):
        Image.fromarray(np.zeros((*size,), dtype=np.uint8)).save(p)

    # sample foo
    (root/"splits/train.txt").write_text("images/foo.png\n")
    save_rgb(root/"data/images/foo.png")
    save_rgb(root/"data/maps/normal/foo_n.png")
    save_gray(root/"data/maps/roughness/foo_r.png")
    save_gray(root/"data/maps/specular/foo_s.png")
    save_gray(root/"data/maps/emissive/foo_e.png")
    (root/"data/meta/foo.json").write_text('{"lum":0.5,"sat":0.5,"contrast":0.5}')
    (root/"data/ann/foo.txt").write_text("")

    ds = MultimodalYoloDataset(str(root), str(root/"splits/train.txt"), imgsz=128, use_coordconv=True)
    x, ctx, tgt = ds[0]
    assert x.ndim == 3 and x.shape[1] == 128 and x.shape[2] == 128
    assert ctx.ndim == 1
    assert isinstance(tgt, dict)
