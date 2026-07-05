"""Image and tabular IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Any

import cv2
import numpy as np
import pandas as pd
from skimage import io as skio

from .utils import ensure_dir, to_float01


def read_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """Read image as RGB or grayscale float in [0, 1]."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    image = skio.imread(str(p))
    if grayscale and image.ndim == 3:
        image = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    return to_float01(image)


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save float image, clipping to [0, 1]."""
    p = Path(path)
    ensure_dir(p.parent)
    arr = np.clip(np.asarray(image), 0.0, 1.0)
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    skio.imsave(str(p), arr8, check_contrast=False)


def list_images(root: str | Path) -> list[Path]:
    """List common image files recursively."""
    r = Path(root)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in r.rglob("*") if p.suffix.lower() in exts)


def save_rows_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Save row dictionaries as a CSV file."""
    p = Path(path)
    ensure_dir(p.parent)
    pd.DataFrame(list(rows)).to_csv(p, index=False, encoding="utf-8-sig")
