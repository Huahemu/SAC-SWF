"""Visualization helpers for saving report-ready figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from .io import save_image
from .utils import ensure_dir, normalize01, rgb_to_gray


def save_heatmap(path: str | Path, data: np.ndarray, title: str | None = None, cmap: str = "magma") -> None:
    """Save a scalar map as a heatmap image."""
    p = Path(path)
    ensure_dir(p.parent)
    plt.figure(figsize=(5, 4))
    plt.imshow(data, cmap=cmap)
    plt.colorbar(fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()


def save_comparison_grid(
    path: str | Path,
    images: Mapping[str, np.ndarray],
    cols: int = 4,
    figsize_per_image: float = 3.0,
) -> None:
    """Save a grid of named images."""
    p = Path(path)
    ensure_dir(p.parent)
    names = list(images.keys())
    rows = int(np.ceil(len(names) / cols))
    plt.figure(figsize=(cols * figsize_per_image, rows * figsize_per_image))
    for idx, name in enumerate(names, start=1):
        plt.subplot(rows, cols, idx)
        img = images[name]
        if np.asarray(img).ndim == 2:
            plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        else:
            plt.imshow(np.clip(img, 0, 1))
        plt.title(name, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()


def save_intermediate_maps(out_dir: str | Path, intermediates: dict) -> None:
    """Save SAC-SWF intermediate maps needed for report figures."""
    root = ensure_dir(out_dir)
    scalar_maps = {
        "beta": intermediates.get("beta"),
        "structure_conf": intermediates.get("structure_conf"),
        "coherence": intermediates.get("coherence"),
        "texture_energy": intermediates.get("texture_energy"),
        "oscillation": intermediates.get("oscillation"),
        "high_residual": intermediates.get("high_residual"),
        "direction_index": intermediates.get("direction_index"),
    }
    for name, data in scalar_maps.items():
        if data is None:
            continue
        save_heatmap(root / f"{name}.png", np.asarray(data), title=name)
    for name in ["q_full", "q_side"]:
        if name in intermediates:
            save_image(root / f"{name}.png", intermediates[name])


def residual_image(input_image: np.ndarray, output_image: np.ndarray) -> np.ndarray:
    """Return normalized absolute residual for visualization."""
    return normalize01(np.abs(rgb_to_gray(input_image) - rgb_to_gray(output_image)))
