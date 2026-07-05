"""Common utilities for image conversion, configuration, and filesystem IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def to_float01(image: np.ndarray) -> np.ndarray:
    """Convert an image to float64 in [0, 1] without changing shape."""
    arr = np.asarray(image)
    if arr.dtype.kind in {"u", "i"}:
        info = np.iinfo(arr.dtype)
        return arr.astype(np.float64) / float(info.max)
    arr = arr.astype(np.float64, copy=False)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float64)
    min_v = float(finite.min())
    max_v = float(finite.max())
    if min_v >= 0.0 and max_v <= 1.0:
        return np.clip(arr, 0.0, 1.0)
    if max_v <= 255.0 and min_v >= 0.0:
        return np.clip(arr / 255.0, 0.0, 1.0)
    return normalize01(arr)


def normalize01(image: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize arbitrary numeric data to [0, 1]."""
    arr = np.asarray(image, dtype=np.float64)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if mx - mn < eps:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - mn) / (mx - mn), 0.0, 1.0)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB/RGBA image to luminance in [0, 1]."""
    arr = to_float01(image)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape}")
    if arr.shape[2] == 1:
        return arr[..., 0]
    rgb = arr[..., :3]
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def as_channels(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return image as HxWxC and whether original image was 2D."""
    arr = to_float01(image)
    if arr.ndim == 2:
        return arr[..., None], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError(f"Expected image with 2 or 3 dims, got {arr.shape}")


def restore_channels(image: np.ndarray, was_gray: bool) -> np.ndarray:
    """Undo as_channels for grayscale input."""
    if was_gray:
        return image[..., 0]
    return image


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    """Save JSON with stable formatting."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def percentile_scale(values: np.ndarray, percentile: float = 75.0, eps: float = 1e-12) -> float:
    """Robust positive scale from a nonnegative map."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    scale = float(np.percentile(finite, percentile))
    return max(scale, eps)


def method_slug(name: str) -> str:
    """Stable file-name friendly method identifier."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("+", "plus")
        .replace("-", "_")
    )
