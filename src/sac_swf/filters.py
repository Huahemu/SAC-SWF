"""Classical filters used as SAC-SWF building blocks."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.restoration import denoise_bilateral

from .utils import as_channels, restore_channels, rgb_to_gray, to_float01


def local_mean(image: np.ndarray, kernel: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """Normalized local weighted mean using correlation, preserving kernel orientation."""
    img = np.asarray(image, dtype=np.float64)
    ker = np.asarray(kernel, dtype=np.float64)
    denom = float(ker.sum())
    if denom <= 0:
        raise ValueError("Kernel must contain at least one positive weight.")
    ker = ker / denom
    if img.ndim == 2:
        return ndi.correlate(img, ker, mode=mode)
    if img.ndim == 3:
        out = np.empty_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            out[..., c] = ndi.correlate(img[..., c], ker, mode=mode)
        return out
    raise ValueError(f"Expected 2D or 3D input, got {img.shape}")


def box_filter(image: np.ndarray, radius: int, mode: str = "reflect") -> np.ndarray:
    """Box filter with a square window of size 2r+1."""
    size = 2 * int(radius) + 1
    arr, was_gray = as_channels(image)
    out = ndi.uniform_filter(arr, size=(size, size, 1), mode=mode)
    return restore_channels(out, was_gray)


def gaussian_filter(image: np.ndarray, sigma: float, mode: str = "reflect") -> np.ndarray:
    """Gaussian smoothing for gray/RGB images."""
    arr, was_gray = as_channels(image)
    out = ndi.gaussian_filter(arr, sigma=(float(sigma), float(sigma), 0), mode=mode)
    return restore_channels(out, was_gray)


def bilateral_filter(
    image: np.ndarray,
    sigma_color: float = 0.08,
    sigma_spatial: float = 4.0,
    channel_axis: int | None = -1,
) -> np.ndarray:
    """Bilateral filter wrapper from scikit-image."""
    arr = to_float01(image)
    if arr.ndim == 2:
        channel_axis = None
    return denoise_bilateral(
        arr,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        channel_axis=channel_axis,
    )


def guided_filter_with_kernel(
    image: np.ndarray,
    guide: np.ndarray | None,
    kernel: np.ndarray,
    eps: float = 1e-3,
    mode: str = "reflect",
) -> np.ndarray:
    """Guided filter where local statistics are computed by an arbitrary support kernel.

    The guide is scalar. For RGB guidance, luminance is used. The input can be
    grayscale or RGB; the same scalar guide controls every channel.
    """
    p, was_gray = as_channels(image)
    g = rgb_to_gray(image if guide is None else guide)

    mean_g = local_mean(g, kernel, mode=mode)
    mean_gg = local_mean(g * g, kernel, mode=mode)
    var_g = mean_gg - mean_g * mean_g

    q = np.empty_like(p, dtype=np.float64)
    for c in range(p.shape[2]):
        pc = p[..., c]
        mean_p = local_mean(pc, kernel, mode=mode)
        mean_gp = local_mean(g * pc, kernel, mode=mode)
        cov_gp = mean_gp - mean_g * mean_p
        a = cov_gp / (var_g + float(eps))
        b = mean_p - a * mean_g
        mean_a = local_mean(a, kernel, mode=mode)
        mean_b = local_mean(b, kernel, mode=mode)
        q[..., c] = mean_a * g + mean_b

    return np.clip(restore_channels(q, was_gray), 0.0, 1.0)


def guided_filter(
    image: np.ndarray,
    guide: np.ndarray | None = None,
    radius: int = 5,
    eps: float = 1e-3,
    mode: str = "reflect",
) -> np.ndarray:
    """Standard square-window guided filter."""
    size = 2 * int(radius) + 1
    kernel = np.ones((size, size), dtype=np.float64)
    return guided_filter_with_kernel(image, guide, kernel, eps=eps, mode=mode)
