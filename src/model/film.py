"""Feature-wise Linear Modulation (FiLM) blocks used across the project."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def _to_sequence(hidden_dims: Iterable[int] | int | None) -> Sequence[int]:
    if hidden_dims is None:
        return ()
    if isinstance(hidden_dims, int):
        return (hidden_dims,)
    return tuple(hidden_dims)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation module.

    The module projects a context vector into a pair of ``gamma`` and ``beta``
    tensors that modulate a spatial feature map using the transformation
    ``gamma * x + beta``. It is lightweight but flexible enough for different
    context dimensionalities thanks to the configurable multilayer perceptron
    (MLP).
    """

    def __init__(
        self,
        in_channels: int,
        dim_context: int = 64,
        hidden_dims: Iterable[int] | int | None = 128,
        activation: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer")
        if dim_context <= 0:
            raise ValueError("dim_context must be a positive integer")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        hidden_sequence = _to_sequence(hidden_dims)
        layers: list[nn.Module] = []
        in_features = dim_context
        for hidden in hidden_sequence:
            if hidden <= 0:
                raise ValueError("hidden dimensions must be positive integers")
            layers.append(nn.Linear(in_features, hidden, bias=bias))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, 2 * in_channels, bias=bias))

        self.in_channels = in_channels
        self.dim_context = dim_context
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, context_vec: torch.Tensor | None) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            x: Spatial feature map of shape ``(B, C, H, W)``.
            context_vec: Context embedding of shape ``(B, D)`` or ``(B, T, D)``.
                A ``None`` context disables the modulation and returns ``x``.

        Returns:
            The modulated tensor with the same shape as ``x``.
        """

        if context_vec is None:
            return x

        if context_vec.dim() == 3:
            # Pool temporal tokens or sequences.
            context = context_vec.mean(dim=1)
        elif context_vec.dim() == 2:
            context = context_vec
        else:
            raise ValueError("context_vec must have 2 or 3 dimensions")

        if context.size(0) != x.size(0):
            raise ValueError("Batch size of context_vec and x must match")

        modulation = self.mlp(context)
        gamma, beta = modulation.chunk(2, dim=1)
        gamma = gamma.view(-1, self.in_channels, 1, 1)
        beta = beta.view(-1, self.in_channels, 1, 1)
        return gamma * x + beta
