"""Structure tensor features for SAC-SWF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .utils import normalize01, percentile_scale, rgb_to_gray


@dataclass
class StructureTensorResult:
    """Maps derived from the smoothed structure tensor."""

    gray: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    lambda1: np.ndarray
    lambda2: np.ndarray
    strength: np.ndarray
    coherence: np.ndarray
    theta: np.ndarray
    strength_norm: np.ndarray


def structure_tensor(
    image: np.ndarray,
    rho: float = 1.5,
    pre_sigma: float = 0.6,
    eta: float = 1e-12,
    kappa_e: float | None = None,
) -> StructureTensorResult:
    """Compute structure strength and orientation coherence."""
    gray = rgb_to_gray(image)
    smooth = ndi.gaussian_filter(gray, sigma=float(pre_sigma), mode="reflect") if pre_sigma > 0 else gray
    iy = ndi.sobel(smooth, axis=0, mode="reflect") / 8.0
    ix = ndi.sobel(smooth, axis=1, mode="reflect") / 8.0

    j11 = ndi.gaussian_filter(ix * ix, sigma=float(rho), mode="reflect")
    j22 = ndi.gaussian_filter(iy * iy, sigma=float(rho), mode="reflect")
    j12 = ndi.gaussian_filter(ix * iy, sigma=float(rho), mode="reflect")

    trace = j11 + j22
    delta = np.sqrt(np.maximum((j11 - j22) ** 2 + 4.0 * j12 * j12, 0.0))
    lambda1 = 0.5 * (trace + delta)
    lambda2 = 0.5 * (trace - delta)
    strength = np.maximum(lambda1 + lambda2, 0.0)
    coherence = (lambda1 - lambda2) / (strength + float(eta))
    coherence = np.clip(coherence, 0.0, 1.0)
    theta = 0.5 * np.arctan2(2.0 * j12, j11 - j22)

    if kappa_e is None:
        kappa_e = percentile_scale(strength, 75.0)
    strength_norm = strength / (strength + float(kappa_e))

    return StructureTensorResult(
        gray=gray,
        ix=ix,
        iy=iy,
        lambda1=lambda1,
        lambda2=lambda2,
        strength=strength,
        coherence=coherence,
        theta=theta,
        strength_norm=np.clip(strength_norm, 0.0, 1.0),
    )


def texture_indicator(
    image: np.ndarray,
    coherence: np.ndarray,
    sigma_t: float = 2.0,
    kappa_h: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute high-frequency energy and non-coherent texture indicator.

    Returns:
        h_norm: normalized high-frequency energy.
        oscillation: h_norm * (1 - coherence).
        high_residual: gray - Gaussian(gray).
    """
    gray = rgb_to_gray(image)
    low = ndi.gaussian_filter(gray, sigma=float(sigma_t), mode="reflect")
    high = gray - low
    local_energy = ndi.uniform_filter(high * high, size=max(3, int(round(2 * sigma_t + 1))), mode="reflect")
    if kappa_h is None:
        kappa_h = percentile_scale(local_energy, 75.0)
    h_norm = local_energy / (local_energy + float(kappa_h))
    h_norm = np.clip(h_norm, 0.0, 1.0)
    oscillation = np.clip(h_norm * (1.0 - coherence), 0.0, 1.0)
    return h_norm, oscillation, normalize01(np.abs(high))
