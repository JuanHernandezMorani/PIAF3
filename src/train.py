"""
Training loop skeleton para YOLOv11 multimodal con FiLM y heads auxiliares.
Codex Plus debe completar: integración real del modelo YOLOv11-seg, pérdidas y métricas.
"""
import argparse, os, torch
from torch.utils.data import DataLoader
from src.data.multimodal_loader import MultimodalYoloDataset, collate_fn

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=str)
    ap.add_argument("--split-train", default="splits/train.txt")
    ap.add_argument("--split-val", default="splits/val.txt")
    ap.add_argument("--imgsz", default=512, type=int)
    ap.add_argument("--batch", default=8, type=int)
    ap.add_argument("--epochs", default=100, type=int)
    ap.add_argument("--use-metalness", action="store_true")
    ap.add_argument("--use-coordconv", action="store_true")
    ap.add_argument("--film-dim", default=64, type=int)
    ap.add_argument("--aux-on", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    train_ds = MultimodalYoloDataset(
        args.root,
        args.split_train,
        imgsz=args.imgsz,
        use_metalness=args.use_metalness,
        use_coordconv=args.use_coordconv,
        augment=True,
        pbr_dropout_prob=0.2,
    )
    val_ds = MultimodalYoloDataset(
        args.root,
        args.split_val,
        imgsz=args.imgsz,
        use_metalness=args.use_metalness,
        use_coordconv=args.use_coordconv,
        augment=False,
        pbr_dropout_prob=0.0,
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # TODO: construir modelo y pérdidas
    print("[train] TODO: construir modelo, optimizer, scheduler, pérdidas y métricas.")

    for epoch in range(args.epochs):
        # TODO: epoch loop
        print(f"[train] epoch {epoch+1}/{args.epochs} ... (skeleton)")

    print("[train] DONE (skeleton)")

if __name__ == "__main__":
    main()
