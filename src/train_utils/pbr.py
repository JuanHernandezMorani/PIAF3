"""Helpers for handling PBR channel configuration."""

from __future__ import annotations

from typing import Dict, List, Sequence


def expand_pbr_channels(order: Sequence[str]) -> List[str]:
    """Expand map names (e.g. ``normal``) into explicit channel identifiers."""

    channel_map: Dict[str, List[str]] = {
        "normal": ["normal_x", "normal_y", "normal_z"],
        "albedo": ["albedo_r", "albedo_g", "albedo_b"],
    }
    expanded: List[str] = []
    for key in order:
        names = channel_map.get(key)
        if names is not None:
            expanded.extend(names)
        else:
            expanded.append(key)
    return expanded


__all__ = ["expand_pbr_channels"]
