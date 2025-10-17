"""Utility helpers for encoding textual context."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import Tensor


class SimpleTextEncoder:
    """Deterministic encoder that maps strings to fixed-size embeddings."""

    def __init__(self, dim: int, seed: int = 0) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        self._cache: Dict[str, Tensor] = {}

    def _encode_token(self, token: str) -> Tensor:
        token_seed = (hash(token) ^ self.seed) & 0xFFFFFFFF
        rng = np.random.default_rng(token_seed)
        vec = torch.from_numpy(rng.standard_normal(self.dim).astype(np.float32))
        return vec

    def encode(self, text: str) -> Tensor:
        cached = self._cache.get(text)
        if cached is not None:
            return cached.clone()
        tokens = [tok for tok in text.replace("\n", " ").split(" ") if tok]
        if not tokens or self.dim <= 0:
            embedding = torch.zeros(self.dim, dtype=torch.float32)
        else:
            vectors = torch.stack([self._encode_token(token) for token in tokens])
            embedding = vectors.mean(dim=0)
            norm = embedding.norm()
            if torch.isfinite(norm) and norm > 0:
                embedding = embedding / norm
        self._cache[text] = embedding.clone()
        return embedding

    def batch(self, texts: Sequence[str]) -> Tensor:
        if self.dim <= 0:
            return torch.zeros((len(texts), 0), dtype=torch.float32)
        embeddings = [self.encode(text) for text in texts]
        return torch.stack(embeddings, dim=0)


__all__ = ["SimpleTextEncoder"]
