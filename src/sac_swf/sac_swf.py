"""Structure-Aware Combined Side-Window Filtering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .side_window import (
    aggregate_side_windows,
    full_window_guided_candidate,
    side_window_guided_candidates,
)
from .structure_tensor import structure_tensor, texture_indicator
from .utils import as_channels, restore_channels, to_float01


@dataclass
class SACSWFConfig:
    """Configuration for SAC-SWF."""

    radius: int = 5
    eps: float = 1e-3
    rho: float = 1.5
    pre_sigma: float = 0.6
    sigma_t: float = 2.0
    gamma: float = 0.5
    tau: float | None = None
    lambda_grad: float = 0.0
    mode: str = "reflect"
    hard_side: bool = False
    beta_mode: str = "clip"  # "clip" or "sigmoid"
    theta0: float = 0.0
    theta1: float = 4.0
    theta2: float = 4.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def compute_beta(
    strength_norm: np.ndarray,
    coherence: np.ndarray,
    oscillation: np.ndarray,
    gamma: float = 0.5,
    beta_mode: str = "clip",
    theta0: float = 0.0,
    theta1: float = 4.0,
    theta2: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute full-window weight beta and structure confidence R."""
    structure_conf = np.clip(strength_norm * coherence, 0.0, 1.0)
    if beta_mode == "clip":
        beta = np.clip(1.0 - structure_conf + float(gamma) * oscillation, 0.0, 1.0)
    elif beta_mode == "sigmoid":
        beta = _sigmoid(float(theta0) + float(theta1) * oscillation - float(theta2) * structure_conf)
    else:
        raise ValueError(f"Unknown beta_mode: {beta_mode}")
    return beta, structure_conf


def sac_swf(
    image: np.ndarray,
    guide: np.ndarray | None = None,
    config: SACSWFConfig | None = None,
    return_intermediates: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run SAC-SWF and return output plus intermediate maps."""
    cfg = config or SACSWFConfig()
    img = to_float01(image)
    g = img if guide is None else to_float01(guide)

    tensor = structure_tensor(img, rho=cfg.rho, pre_sigma=cfg.pre_sigma)
    h_norm, oscillation, high_residual = texture_indicator(
        img,
        tensor.coherence,
        sigma_t=cfg.sigma_t,
    )
    beta, structure_conf = compute_beta(
        tensor.strength_norm,
        tensor.coherence,
        oscillation,
        gamma=cfg.gamma,
        beta_mode=cfg.beta_mode,
        theta0=cfg.theta0,
        theta1=cfg.theta1,
        theta2=cfg.theta2,
    )

    q_full = full_window_guided_candidate(img, g, radius=cfg.radius, eps=cfg.eps, mode=cfg.mode)
    candidates = side_window_guided_candidates(img, g, radius=cfg.radius, eps=cfg.eps, mode=cfg.mode)
    q_side, alpha, direction_index, direction_names = aggregate_side_windows(
        img,
        candidates,
        tau=cfg.tau,
        lambda_grad=cfg.lambda_grad,
        hard=cfg.hard_side,
    )

    qf, was_gray = as_channels(q_full)
    qs, _ = as_channels(q_side)
    b = beta[..., None]
    out = np.clip(b * qf + (1.0 - b) * qs, 0.0, 1.0)
    output = restore_channels(out, was_gray)

    if not return_intermediates:
        return output, {}

    intermediates: dict[str, Any] = {
        "config": cfg.to_dict(),
        "q_full": q_full,
        "q_side": q_side,
        "beta": beta,
        "structure_conf": structure_conf,
        "strength": tensor.strength,
        "strength_norm": tensor.strength_norm,
        "coherence": tensor.coherence,
        "theta": tensor.theta,
        "texture_energy": h_norm,
        "oscillation": oscillation,
        "high_residual": high_residual,
        "alpha": alpha,
        "direction_index": direction_index,
        "direction_names": direction_names,
        "candidates": candidates,
    }
    return output, intermediates
