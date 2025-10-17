"""Orquesta experimentos de ablation para el modelo multimodal.

El script corre las siguientes configuraciones:

1. **RGB-only**: solo canales RGB, sin FiLM ni cabezales auxiliares.
2. **RGB+PBR**: canales RGB+PBR sin condicionamiento (FiLM apagado).
3. **RGB+PBR+FiLM**: entradas completas con condicionamiento FiLM.
4. **RGB+PBR+FiLM+Aux**: igual al anterior pero habilitando cabezales auxiliares.

Luego consolida los resultados en tablas CSV y un resumen en Markdown.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch

from src import train as train_module
from src.data.multimodal_loader import MultimodalYoloDataset


@dataclass
class AblationArgs:
    """Argumentos comunes que se propagan a cada experimento."""

    root: str
    split_train: str
    split_val: str
    imgsz: int
    batch: int
    epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    device: Optional[str]
    base_out: str
    film_dim: int
    use_metalness: bool
    use_coordconv: bool
    skip_existing: bool
    clean: bool
    dry_run: bool


class _ZeroContextDataset(MultimodalYoloDataset):
    """Dataset que devuelve vectores de contexto nulos."""

    context_dim: int = 1

    def _context_vec(self, meta_path: str) -> np.ndarray:  # noqa: D401 - implementación concreta
        dim = max(int(self.context_dim), 1)
        return np.zeros(dim, dtype=np.float32)


class _RGBOnlyDataset(_ZeroContextDataset):
    """Mantiene solo canales RGB y desactiva PBR/FiLM."""

    def _load_modalities(self, paths: Mapping[str, str]) -> np.ndarray:
        arr = super()._load_modalities(paths)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr

    def _extract_pbr_targets(self, arr: np.ndarray) -> Dict[str, np.ndarray]:  # noqa: D401 - no targets
        return {}


class _RGBPBRNoContextDataset(_ZeroContextDataset):
    """Preserva canales PBR pero ignora contexto (FiLM off)."""

    context_dim: int = 1


@dataclass
class Experiment:
    """Configura un experimento individual."""

    name: str
    description: str
    modalities: str
    use_film: bool
    aux_on: bool
    dataset_factory: Callable[[], type[MultimodalYoloDataset]]
    arg_overrides: MutableMapping[str, object]


def _temporary_attr(obj: object, attr: str, value: object):
    """Context manager para parchear atributos temporariamente."""

    @contextlib.contextmanager
    def _manager():
        sentinel = object()
        original = getattr(obj, attr, sentinel)
        setattr(obj, attr, value)
        try:
            yield
        finally:
            if original is sentinel:
                delattr(obj, attr)
            else:
                setattr(obj, attr, original)

    return _manager()


def _format_command(namespace: argparse.Namespace) -> str:
    """Genera una representación reproducible del comando de training."""

    parts: List[str] = ["python -m src.train"]
    for key, value in sorted(vars(namespace).items()):
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                parts.append(flag)
            continue
        if value is None:
            continue
        parts.append(f"{flag} {value}")
    return " ".join(parts)


def _make_train_args(base: AblationArgs, overrides: Mapping[str, object]) -> argparse.Namespace:
    data: Dict[str, object] = {
        "root": base.root,
        "split_train": base.split_train,
        "split_val": base.split_val,
        "imgsz": base.imgsz,
        "batch": base.batch,
        "epochs": base.epochs,
        "lr": base.lr,
        "weight_decay": base.weight_decay,
        "num_workers": base.num_workers,
        "device": base.device,
        "out_dir": "runs/train",
        "use_metalness": base.use_metalness,
        "use_coordconv": base.use_coordconv,
        "film_dim": base.film_dim,
        "aux_on": False,
    }
    data.update(dict(overrides))
    return argparse.Namespace(**data)


def _load_metrics(run_dir: Path) -> Tuple[Dict[str, object], Dict[str, float], Optional[int], Optional[float]]:
    best_ckpt = run_dir / "checkpoint_best.pt"
    if not best_ckpt.exists():
        return {}, {}, None, None
    state = torch.load(best_ckpt, map_location="cpu")
    metrics = state.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    per_class = metrics.get("per_class_ap", {}) if isinstance(metrics, Mapping) else {}
    if not isinstance(per_class, Mapping):
        per_class = {}
    epoch = state.get("epoch")
    best_map = state.get("best_mAP")
    return dict(metrics), dict(per_class), epoch if isinstance(epoch, int) else None, float(best_map) if isinstance(best_map, (float, int)) else None


def _invoke_training(args: argparse.Namespace, dataset_cls: type[MultimodalYoloDataset]) -> None:
    with contextlib.ExitStack() as stack:
        stack.enter_context(_temporary_attr(train_module, "parse_args", lambda: args))
        stack.enter_context(_temporary_attr(train_module, "MultimodalYoloDataset", dataset_cls))
        train_module.main()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_experiments(base: AblationArgs) -> List[Experiment]:
    base_out_rel = base.base_out.rstrip("/")
    if not base_out_rel:
        base_out_rel = "runs/ablation"
    experiments: List[Experiment] = [
        Experiment(
            name="rgb_only",
            description="Solo imagen RGB (sin PBR ni FiLM).",
            modalities="RGB",
            use_film=False,
            aux_on=False,
            dataset_factory=lambda: _RGBOnlyDataset,
            arg_overrides={
                "out_dir": f"{base_out_rel}/rgb_only",
                "film_dim": 1,
                "aux_on": False,
            },
        ),
        Experiment(
            name="rgb_pbr",
            description="RGB + mapas PBR, sin condicionamiento.",
            modalities="RGB+PBR",
            use_film=False,
            aux_on=False,
            dataset_factory=lambda: _RGBPBRNoContextDataset,
            arg_overrides={
                "out_dir": f"{base_out_rel}/rgb_pbr",
                "film_dim": 1,
                "aux_on": False,
            },
        ),
        Experiment(
            name="rgb_pbr_film",
            description="RGB + PBR con FiLM activado.",
            modalities="RGB+PBR",
            use_film=True,
            aux_on=False,
            dataset_factory=lambda: MultimodalYoloDataset,
            arg_overrides={
                "out_dir": f"{base_out_rel}/rgb_pbr_film",
                "film_dim": base.film_dim,
                "aux_on": False,
            },
        ),
        Experiment(
            name="rgb_pbr_film_aux",
            description="RGB + PBR con FiLM y cabezales auxiliares.",
            modalities="RGB+PBR",
            use_film=True,
            aux_on=True,
            dataset_factory=lambda: MultimodalYoloDataset,
            arg_overrides={
                "out_dir": f"{base_out_rel}/rgb_pbr_film_aux",
                "film_dim": base.film_dim,
                "aux_on": True,
            },
        ),
    ]
    return experiments


def _export_csv(rows: Iterable[Mapping[str, object]], header: Iterable[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})


def _render_markdown_summary(results: List[Dict[str, object]], args: AblationArgs) -> str:
    header = [
        "Experimento",
        "Descripción",
        "Modalidades",
        "FiLM",
        "Aux",
        "mAP",
        "Val loss",
        "Época",
        "Duración (s)",
    ]
    lines = ["# Resumen de ablation", ""]
    lines.append(f"* Dataset root: `{Path(args.root).resolve()}`")
    lines.append(f"* Épocas: `{args.epochs}` | Tamaño de imagen: `{args.imgsz}` | Batch: `{args.batch}`")
    lines.append("")
    lines.append("## Métricas agregadas")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    def _fmt(value: object) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "NaN"
            return f"{value:.4f}"
        if value is None:
            return "-"
        return str(value)

    for row in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("name", "-"),
                    row.get("description", "-"),
                    row.get("modalities", "-"),
                    "Sí" if row.get("use_film") else "No",
                    "Sí" if row.get("aux_on") else "No",
                    _fmt(row.get("best_map", float("nan"))),
                    _fmt(row.get("val_loss", float("nan"))),
                    _fmt(row.get("best_epoch")),
                    _fmt(row.get("duration_sec", float("nan"))),
                ]
            )
            + " |"
        )

    ok_runs: List[Dict[str, object]] = []
    for r in results:
        if r.get("status") != "ok":
            continue
        best_map = r.get("best_map")
        if not isinstance(best_map, (float, int)):
            continue
        if math.isnan(float(best_map)):
            continue
        ok_runs.append(r)
    best_block = ""
    if ok_runs:
        best_run = max(ok_runs, key=lambda r: float(r.get("best_map", float("-inf"))))
        best_map_val = float(best_run.get("best_map", 0.0))
        baseline = next((r for r in results if r.get("name") == "rgb_only"), None)
        improvement = None
        if baseline and isinstance(baseline.get("best_map"), (float, int)):
            base_map = float(baseline["best_map"])
            if not math.isnan(base_map):
                improvement = best_map_val - base_map
        lines.append("")
        lines.append("## Highlights")
        lines.append("")
        lines.append(f"* Mejor configuración: **{best_run.get('name')}** (mAP={best_map_val:.4f})")
        if improvement is not None:
            lines.append(f"* Mejora sobre RGB-only: `{improvement:.4f}` puntos de mAP.")
    failed = [r for r in results if r.get("status") not in {"ok", "skipped"}]
    if failed:
        lines.append("")
        lines.append("## Errores")
        lines.append("")
        for row in failed:
            lines.append(f"* {row.get('name')}: {row.get('error', 'motivo desconocido')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation RGB/PBR/FiLM")
    ap.add_argument("--root", default=".", type=str, help="Directorio raíz del proyecto/dataset")
    ap.add_argument("--split-train", default="splits/train.txt", type=str)
    ap.add_argument("--split-val", default="splits/val.txt", type=str)
    ap.add_argument("--imgsz", default=512, type=int)
    ap.add_argument("--epochs", default=50, type=int)
    ap.add_argument("--batch", default=8, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--weight-decay", default=1e-4, type=float)
    ap.add_argument("--num-workers", default=2, type=int)
    ap.add_argument("--device", default=None, type=str)
    ap.add_argument("--film-dim", default=64, type=int)
    ap.add_argument("--base-out", default="runs/ablation", type=str, help="Directorio base para los resultados")
    ap.add_argument("--use-metalness", action="store_true")
    ap.add_argument("--use-coordconv", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", help="No relanza si existe checkpoint_best")
    ap.add_argument("--clean", action="store_true", help="Elimina carpetas previas del experimento")
    ap.add_argument("--dry-run", action="store_true", help="Solo recopila métricas existentes, sin entrenar")
    args = ap.parse_args()

    base = AblationArgs(
        root=args.root,
        split_train=args.split_train,
        split_val=args.split_val,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        base_out=args.base_out,
        film_dim=args.film_dim,
        use_metalness=args.use_metalness,
        use_coordconv=args.use_coordconv,
        skip_existing=args.skip_existing or args.dry_run,
        clean=args.clean and not args.dry_run,
        dry_run=args.dry_run,
    )

    experiments = _build_experiments(base)

    results_dir = Path(base.root) / base.base_out
    _ensure_dir(results_dir)

    aggregate_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []

    for exp in experiments:
        print(f"[ablation] >>> {exp.name}: {exp.description}")
        dataset_cls = exp.dataset_factory()
        train_args = _make_train_args(base, exp.arg_overrides)
        run_dir = Path(base.root) / train_args.out_dir
        if base.clean and run_dir.exists():
            print(f"[ablation] Limpiando directorio previo: {run_dir}")
            shutil.rmtree(run_dir)
        _ensure_dir(run_dir)

        command_repr = _format_command(train_args)
        print(f"[ablation] Comando: {command_repr}")

        status = "skipped" if base.dry_run else "pending"
        duration = 0.0
        error_msg: Optional[str] = None

        if base.skip_existing and (run_dir / "checkpoint_best.pt").exists():
            print("[ablation] Checkpoint existente, se omite el entrenamiento.")
            status = "skipped"
        elif base.dry_run:
            print("[ablation] Modo dry-run: no se ejecuta entrenamiento.")
            status = "skipped"
        else:
            status = "running"
            start = time.time()
            try:
                _invoke_training(train_args, dataset_cls)
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - propagamos el error al resumen
                status = "failed"
                error_msg = str(exc)
                print(f"[ablation] Error en {exp.name}: {exc}")
            finally:
                duration = time.time() - start
                torch.cuda.empty_cache()

        metrics, per_class, epoch, best_map = _load_metrics(run_dir)
        val_loss = metrics.get("loss") if isinstance(metrics.get("loss"), (float, int)) else float("nan")
        map_value = metrics.get("mAP") if isinstance(metrics.get("mAP"), (float, int)) else float("nan")

        aggregate_row = {
            "name": exp.name,
            "description": exp.description,
            "modalities": exp.modalities,
            "use_film": exp.use_film,
            "aux_on": exp.aux_on,
            "best_map": float(map_value) if isinstance(map_value, (float, int)) else float("nan"),
            "val_loss": float(val_loss) if isinstance(val_loss, (float, int)) else float("nan"),
            "best_epoch": epoch,
            "duration_sec": duration,
            "status": status,
            "error": error_msg,
            "command": command_repr,
            "out_dir": run_dir.as_posix(),
        }
        if best_map is not None and math.isnan(aggregate_row["best_map"]):
            aggregate_row["best_map"] = best_map
        aggregate_rows.append(aggregate_row)

        for cls_name, ap_value in per_class.items():
            per_class_rows.append(
                {
                    "experiment": exp.name,
                    "class": cls_name,
                    "AP": ap_value,
                }
            )

    overview_csv = results_dir / "ablation_overview.csv"
    per_class_csv = results_dir / "ablation_per_class_ap.csv"
    summary_md = results_dir / "ablation_summary.md"

    _export_csv(
        aggregate_rows,
        [
            "name",
            "description",
            "modalities",
            "use_film",
            "aux_on",
            "best_map",
            "val_loss",
            "best_epoch",
            "duration_sec",
            "status",
            "error",
            "command",
            "out_dir",
        ],
        overview_csv,
    )
    _export_csv(per_class_rows, ["experiment", "class", "AP"], per_class_csv)
    summary_md.write_text(_render_markdown_summary(aggregate_rows, base), encoding="utf-8")

    print(f"[ablation] Resultados exportados en {overview_csv}")
    print(f"[ablation] AP por clase exportado en {per_class_csv}")
    print(f"[ablation] Resumen Markdown disponible en {summary_md}")


if __name__ == "__main__":
    main()
