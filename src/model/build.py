"""Factory utilities to assemble the multimodal YOLOv11 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .aux_heads import PBRReconHeads
from .context_adapter import ContextAdapter
from .integrations import FiLMBlock, inject_film


@dataclass
class ModelComponents:
    """Container object holding the assembled sub-modules."""

    base: nn.Module
    context_adapter: ContextAdapter
    film_blocks: Iterable[FiLMBlock]
    aux_head: Optional[PBRReconHeads]
    aux_scales: Sequence[str]


class YoloMultiModel(nn.Module):
    """Wrapper around Ultralytics YOLO that augments it with multimodal context."""

    def __init__(self, components: ModelComponents) -> None:
        super().__init__()
        self.model = components.base
        self.context_adapter = components.context_adapter
        self.film_blocks = nn.ModuleList(list(components.film_blocks))
        self.aux_head = components.aux_head
        self.aux_scales = tuple(components.aux_scales)
        self._feature_cache: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Attach hooks to collect intermediate features for auxiliary heads."""

        if self.aux_head is None:
            return
        detect_module: Optional[nn.Module] = None
        if hasattr(self.model, "model"):
            module = getattr(self.model, "model")
            if isinstance(module, (list, tuple)) and module:
                detect_module = module[-1]
            elif isinstance(module, nn.ModuleList) and len(module) > 0:
                detect_module = module[-1]
        if detect_module is None:
            return

        def _cache_features(_: nn.Module, inputs: tuple[object, ...]) -> None:
            if not inputs:
                return
            features = inputs[0]
            if not isinstance(features, (list, tuple)):
                return
            for idx, feat in enumerate(features):
                if isinstance(feat, torch.Tensor):
                    self._feature_cache[f"p{3 + idx}"] = feat

        detect_module.register_forward_pre_hook(_cache_features)

    def forward(
        self,
        images: torch.Tensor,
        *,
        pbr: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        targets: Optional[Mapping[str, Any]] = None,
        mode: str = "train",
    ) -> Dict[str, Any]:
        """Perform a forward pass with FiLM conditioning."""

        del targets, mode  # Targets are consumed by the external loss wrapper.
        ctx_vec = self.context_adapter(pbr, text)
        for block in self.film_blocks:
            block.set_context(ctx_vec)
        outputs = self.model(images)
        for block in self.film_blocks:
            block.clear_context()

        aux_outputs: Dict[str, torch.Tensor] = {}
        if self.aux_head is not None and self._feature_cache:
            features = {
                name: self._feature_cache[name]
                for name in self.aux_scales
                if name in self._feature_cache
            }
            if features:
                aux_outputs = self.aux_head(features)
        self._feature_cache.clear()
        return {"pred": outputs, "aux": aux_outputs}


def _resolve_aux_channels(model: nn.Module) -> Dict[str, int]:
    """Inspect the detection head to determine available feature map channels."""

    detect_module: Optional[nn.Module] = None
    if hasattr(model, "model"):
        module = getattr(model, "model")
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            detect_module = module[-1]
        elif isinstance(module, (list, tuple)) and module:
            detect_module = module[-1]
    if detect_module is None or not hasattr(detect_module, "ch"):
        return {}
    channels = list(getattr(detect_module, "ch"))
    return {f"p{3 + idx}": ch for idx, ch in enumerate(channels)}


def build_yolo11_multi(cfg: Mapping[str, Any]) -> YoloMultiModel:
    """Instantiate the multimodal YOLOv11 model according to ``cfg``."""

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency guard
        message = (
            "Ultralytics (and its runtime dependencies such as opencv-python-headless) "
            "is required to build the YOLOv11 multimodal model"
        )
        raise ImportError(message) from exc

    model_section = cfg.get("model", {})
    model_yaml = model_section.get("cfg")
    pretrained = cfg.get("pretrained")

    if model_yaml:
        base = YOLO(model_yaml)
        if pretrained:
            base.load(pretrained)
    elif pretrained:
        base = YOLO(pretrained)
    else:
        raise ValueError("Either 'pretrained' or 'model.cfg' must be provided in the config")
    base_model = base.model

    context_cfg = cfg.get("context", {})
    text_dim = int(context_cfg.get("text_dim", 0) or 0)
    pbr_channels = context_cfg.get("pbr_channels", []) or []
    pbr_dim = len(pbr_channels)
    adapter_dim = int(context_cfg.get("adapter_dim", 256))
    hidden_dim = int(context_cfg.get("hidden_dim", adapter_dim))
    context_adapter = ContextAdapter(
        in_dim_text=text_dim,
        in_channels_pbr=pbr_dim,
        out_dim=adapter_dim,
        hidden_dim=hidden_dim,
        dropout=float(context_cfg.get("dropout", 0.0)),
    )

    film_cfg = cfg.get("film", {})
    film_blocks = inject_film(base_model, adapter_dim, film_cfg)

    aux_cfg = cfg.get("aux", {})
    aux_head: Optional[PBRReconHeads] = None
    aux_scales: Sequence[str] = ()
    if aux_cfg.get("enable", False) and pbr_dim > 0:
        channel_map = _resolve_aux_channels(base_model)
        selected_channels = {
            name: channel_map[name]
            for name in aux_cfg.get("out_scales", [])
            if name in channel_map
        }
        if selected_channels:
            aux_head = PBRReconHeads(selected_channels, out_channels=pbr_dim)
            aux_scales = tuple(selected_channels.keys())

    components = ModelComponents(
        base=base_model,
        context_adapter=context_adapter,
        film_blocks=film_blocks,
        aux_head=aux_head,
        aux_scales=aux_scales,
    )
    return YoloMultiModel(components)


__all__ = ["build_yolo11_multi", "YoloMultiModel"]

