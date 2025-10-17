"""Helpers to integrate FiLM modules within YOLO architectures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .film import FiLM


@dataclass
class FiLMConfig:
    targets: Sequence[str]
    hidden: int | Sequence[int] = 256
    dropout: float = 0.0


class FiLMBlock(nn.Module):
    """Wrapper that applies FiLM modulation to the output of a module."""

    def __init__(
        self,
        module: nn.Module,
        ctx_dim: int,
        *,
        hidden: int | Sequence[int] = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.module = module
        self.ctx_dim = ctx_dim
        self.hidden = hidden
        self.dropout = dropout
        self.context: Optional[torch.Tensor] = None
        self.film: Optional[FiLM] = None

    def set_context(self, ctx: Optional[torch.Tensor]) -> None:
        self.context = ctx

    def clear_context(self) -> None:
        self.context = None

    def _ensure_film(self, channels: int, device: torch.device) -> None:
        if self.film is not None:
            if self.film.in_channels == channels:
                self.film = self.film.to(device)
                return
        film = FiLM(
            in_channels=channels,
            dim_context=self.ctx_dim,
            hidden_dims=self.hidden,
            dropout=self.dropout,
        )
        with torch.no_grad():
            last = film.mlp[-1]
            if isinstance(last, nn.Linear):
                last.weight.zero_()
                if last.bias is not None:
                    half = last.bias.numel() // 2
                    last.bias.zero_()
                    last.bias[:half] = 1.0
        self.film = film.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.module(x)
        if self.context is None or output.dim() != 4:
            return output
        self._ensure_film(output.size(1), output.device)
        assert self.film is not None
        return self.film(output, self.context)

    def extra_repr(self) -> str:
        return f"ctx_dim={self.ctx_dim}, hidden={self.hidden}, dropout={self.dropout}"

def _locate_module(model: nn.Module, path: str) -> tuple[nn.Module, str, nn.Module]:
    parts = path.split(".")
    parent = model
    for idx, part in enumerate(parts):
        last = idx == len(parts) - 1
        if part.isdigit():
            parent_list = parent
            if not isinstance(parent_list, (nn.Sequential, nn.ModuleList)):
                raise AttributeError(f"Cannot index into module '{type(parent).__name__}'")
            index = int(part)
            if index >= len(parent_list):
                raise IndexError(f"Index {index} out of range for module '{path}'")
            if last:
                return parent_list, part, parent_list[index]
            parent = parent_list[index]
        else:
            if not hasattr(parent, part):
                raise AttributeError(f"Module '{type(parent).__name__}' has no attribute '{part}'")
            child = getattr(parent, part)
            if last:
                return parent, part, child
            parent = child
    raise RuntimeError(f"Invalid module path '{path}'")


def _replace_module(parent: nn.Module, name: str, module: nn.Module) -> None:
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and name.isdigit():
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def inject_film(model: nn.Module, ctx_dim: int, film_config: dict) -> List[FiLMBlock]:
    """Replace target modules with FiLM-wrapped versions."""

    config = FiLMConfig(
        targets=tuple(film_config.get("targets", ())),
        hidden=film_config.get("hidden", 256),
        dropout=float(film_config.get("dropout", 0.0)),
    )
    if not config.targets:
        return []

    blocks: List[FiLMBlock] = []
    for target in config.targets:
        try:
            parent, name, child = _locate_module(model, target)
        except (AttributeError, IndexError) as err:
            raise ValueError(f"Unable to locate module '{target}' for FiLM injection") from err
        block = FiLMBlock(child, ctx_dim, hidden=config.hidden, dropout=config.dropout)
        _replace_module(parent, name, block)
        blocks.append(block)
    return blocks


__all__ = ["FiLMBlock", "inject_film"]

