"""Metrics for structure-texture separation experiments."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .utils import rgb_to_gray, to_float01


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(to_float01(target), to_float01(pred), data_range=1.0))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    p = to_float01(pred)
    t = to_float01(target)
    if p.ndim == 3:
        return float(structural_similarity(t, p, data_range=1.0, channel_axis=-1))
    return float(structural_similarity(t, p, data_range=1.0))


def edge_fscore(pred: np.ndarray, target: np.ndarray, sigma: float = 1.5, tolerance: int = 2) -> float:
    """Boundary F-score with a small matching tolerance."""
    ep = canny(rgb_to_gray(pred), sigma=sigma)
    et = canny(rgb_to_gray(target), sigma=sigma)
    if not ep.any() and not et.any():
        return 1.0
    if not ep.any() or not et.any():
        return 0.0
    structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    tp_p = ep & ndi.binary_dilation(et, structure=structure)
    tp_t = et & ndi.binary_dilation(ep, structure=structure)
    precision = tp_p.sum() / max(ep.sum(), 1)
    recall = tp_t.sum() / max(et.sum(), 1)
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def texture_residual_energy(pred: np.ndarray, inp: np.ndarray, sigma: float = 2.0) -> float:
    """Ratio of high-frequency energy retained in output."""
    pg = rgb_to_gray(pred)
    ig = rgb_to_gray(inp)
    hp = pg - ndi.gaussian_filter(pg, sigma=sigma, mode="reflect")
    hi = ig - ndi.gaussian_filter(ig, sigma=sigma, mode="reflect")
    return float(np.linalg.norm(hp) / max(np.linalg.norm(hi), 1e-12))


def edge_leakage(pred: np.ndarray, target: np.ndarray, sigma: float = 1.5, dilation: int = 3) -> float:
    """Mean absolute error near target edges."""
    tg = rgb_to_gray(target)
    pg = rgb_to_gray(pred)
    edge = canny(tg, sigma=sigma)
    if dilation > 0:
        edge = ndi.binary_dilation(edge, iterations=dilation)
    if not edge.any():
        return float(np.mean(np.abs(pg - tg)))
    return float(np.mean(np.abs(pg[edge] - tg[edge])))


def evaluate_structure(pred: np.ndarray, target: np.ndarray, inp: np.ndarray | None = None) -> dict[str, float]:
    """Compute the core metrics used in synthetic experiments."""
    out = {
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "edge_fscore": edge_fscore(pred, target),
        "edge_leakage": edge_leakage(pred, target),
    }
    if inp is not None:
        out["texture_residual_energy"] = texture_residual_energy(pred, inp)
    return out


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager returning elapsed seconds in result['seconds']."""
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = float(time.perf_counter() - start)
