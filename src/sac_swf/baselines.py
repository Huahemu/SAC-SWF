"""Optional baselines used by extended experiments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .filters import bilateral_filter, gaussian_filter, guided_filter
from .sac_swf import SACSWFConfig, sac_swf
from .side_window import aggregate_side_windows, side_window_guided_candidates


MethodFn = Callable[[np.ndarray, SACSWFConfig], tuple[np.ndarray, dict]]


def gaussian_method(image: np.ndarray, cfg: SACSWFConfig) -> tuple[np.ndarray, dict]:
    """Gaussian low-pass baseline."""
    return gaussian_filter(image, sigma=max(1.0, cfg.radius / 2)), {}


def bilateral_method(image: np.ndarray, cfg: SACSWFConfig) -> tuple[np.ndarray, dict]:
    """Bilateral filter baseline."""
    return bilateral_filter(image), {}


def guided_method(image: np.ndarray, cfg: SACSWFConfig) -> tuple[np.ndarray, dict]:
    """Guided filter baseline."""
    return guided_filter(image, radius=cfg.radius, eps=cfg.eps), {}


def soft_swgf_method(image: np.ndarray, cfg: SACSWFConfig) -> tuple[np.ndarray, dict]:
    """Soft side-window guided filter baseline."""
    candidates = side_window_guided_candidates(image, radius=cfg.radius, eps=cfg.eps)
    q, alpha, direction_index, direction_names = aggregate_side_windows(
        image,
        candidates,
        tau=cfg.tau,
        lambda_grad=cfg.lambda_grad,
        hard=False,
    )
    return q, {
        "alpha": alpha,
        "direction_index": direction_index,
        "direction_names": direction_names,
    }


def fixed_mix_method(image: np.ndarray, cfg: SACSWFConfig) -> tuple[np.ndarray, dict]:
    """A non-adaptive half-and-half full/side window mixture."""
    q_full = guided_filter(image, radius=cfg.radius, eps=cfg.eps)
    q_side, inter = soft_swgf_method(image, cfg)
    q = np.clip(0.5 * q_full + 0.5 * q_side, 0.0, 1.0)
    inter = dict(inter)
    inter["q_full"] = q_full
    inter["q_side"] = q_side
    return q, inter


def hard_swgf_method(image: np.ndarray, cfg: SACSWFConfig) -> tuple[np.ndarray, dict]:
    """Hard-minimum side-window guided filter baseline."""
    candidates = side_window_guided_candidates(image, radius=cfg.radius, eps=cfg.eps)
    q, alpha, direction_index, direction_names = aggregate_side_windows(image, candidates, hard=True)
    return q, {
        "alpha": alpha,
        "direction_index": direction_index,
        "direction_names": direction_names,
    }


def sac_swf_method(image: np.ndarray, cfg: SACSWFConfig) -> tuple[np.ndarray, dict]:
    """Proposed SAC-SWF method."""
    return sac_swf(image, config=cfg, return_intermediates=True)


def tv_chambolle_method(image: np.ndarray, cfg: SACSWFConfig) -> tuple[np.ndarray, dict]:
    """Total variation denoising baseline from scikit-image."""
    try:
        from skimage.restoration import denoise_tv_chambolle
    except Exception as exc:  # pragma: no cover - optional dependency diagnostics
        raise RuntimeError("scikit-image TV Chambolle baseline is unavailable") from exc
    channel_axis = -1 if np.asarray(image).ndim == 3 else None
    return denoise_tv_chambolle(image, weight=0.08, channel_axis=channel_axis), {}


def available_extended_methods(include_tv: bool = True) -> dict[str, MethodFn]:
    """Return deterministic method order for extended comparisons."""
    methods: dict[str, MethodFn] = {
        "gaussian": gaussian_method,
        "bilateral": bilateral_method,
        "guided": guided_method,
        "hard_swgf": hard_swgf_method,
        "soft_swgf": soft_swgf_method,
        "fixed_mix": fixed_mix_method,
        "sac_swf": sac_swf_method,
    }
    if include_tv:
        methods["tv_chambolle"] = tv_chambolle_method
    return methods
