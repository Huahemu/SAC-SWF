"""Side-window support generation and aggregation."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .filters import guided_filter_with_kernel
from .utils import as_channels, restore_channels, rgb_to_gray

DIRECTIONS = ("L", "R", "U", "D", "NW", "NE", "SW", "SE")


def side_window_kernels(radius: int, include_full: bool = False) -> OrderedDict[str, np.ndarray]:
    """Return oriented binary kernels for side-window filtering."""
    r = int(radius)
    if r < 1:
        raise ValueError("radius must be >= 1")
    size = 2 * r + 1
    kernels: OrderedDict[str, np.ndarray] = OrderedDict()
    if include_full:
        kernels["F"] = np.ones((size, size), dtype=np.float64)

    kernels["L"] = np.pad(np.ones((size, r + 1)), ((0, 0), (0, r)))
    kernels["R"] = np.pad(np.ones((size, r + 1)), ((0, 0), (r, 0)))
    kernels["U"] = np.pad(np.ones((r + 1, size)), ((0, r), (0, 0)))
    kernels["D"] = np.pad(np.ones((r + 1, size)), ((r, 0), (0, 0)))
    kernels["NW"] = np.pad(np.ones((r + 1, r + 1)), ((0, r), (0, r)))
    kernels["NE"] = np.pad(np.ones((r + 1, r + 1)), ((0, r), (r, 0)))
    kernels["SW"] = np.pad(np.ones((r + 1, r + 1)), ((r, 0), (0, r)))
    kernels["SE"] = np.pad(np.ones((r + 1, r + 1)), ((r, 0), (r, 0)))
    return kernels


def side_window_guided_candidates(
    image: np.ndarray,
    guide: np.ndarray | None = None,
    radius: int = 5,
    eps: float = 1e-3,
    mode: str = "reflect",
) -> OrderedDict[str, np.ndarray]:
    """Compute side-window guided-filter candidates for 8 directions."""
    kernels = side_window_kernels(radius, include_full=False)
    candidates: OrderedDict[str, np.ndarray] = OrderedDict()
    for name, kernel in kernels.items():
        candidates[name] = guided_filter_with_kernel(image, guide, kernel, eps=eps, mode=mode)
    return candidates


def full_window_guided_candidate(
    image: np.ndarray,
    guide: np.ndarray | None = None,
    radius: int = 5,
    eps: float = 1e-3,
    mode: str = "reflect",
) -> np.ndarray:
    """Compute a full-window guided-filter candidate using the same kernel logic."""
    size = 2 * int(radius) + 1
    kernel = np.ones((size, size), dtype=np.float64)
    return guided_filter_with_kernel(image, guide, kernel, eps=eps, mode=mode)


def _candidate_cost(
    candidate: np.ndarray,
    image: np.ndarray,
    lambda_grad: float = 0.0,
) -> np.ndarray:
    """Compute per-pixel side-window candidate cost."""
    cand, _ = as_channels(candidate)
    inp, _ = as_channels(image)
    diff = np.mean((cand - inp) ** 2, axis=2)
    if lambda_grad <= 0:
        return diff
    cg = rgb_to_gray(candidate)
    ig = rgb_to_gray(image)
    cgy, cgx = np.gradient(cg)
    igy, igx = np.gradient(ig)
    grad_diff = (cgx - igx) ** 2 + (cgy - igy) ** 2
    return diff + float(lambda_grad) * grad_diff


def aggregate_side_windows(
    image: np.ndarray,
    candidates: OrderedDict[str, np.ndarray],
    tau: float | None = None,
    lambda_grad: float = 0.0,
    hard: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Aggregate side-window candidates by hard min or softmax weights.

    Returns:
        q_side: aggregated side-window output.
        alpha: HxWxK direction weights.
        direction_index: HxW argmax/argmin direction index.
        names: direction names in alpha order.
    """
    names = list(candidates.keys())
    if not names:
        raise ValueError("At least one candidate is required.")

    costs = np.stack([_candidate_cost(candidates[n], image, lambda_grad) for n in names], axis=-1)
    direction_index = np.argmin(costs, axis=-1).astype(np.int32)

    img_channels, was_gray = as_channels(image)
    cand_stack = np.stack([as_channels(candidates[n])[0] for n in names], axis=-1)  # H W C K

    if hard:
        alpha = np.zeros_like(costs, dtype=np.float64)
        np.put_along_axis(alpha, direction_index[..., None], 1.0, axis=-1)
    else:
        if tau is None or tau <= 0:
            finite = costs[np.isfinite(costs)]
            tau = max(float(np.median(finite)) * 0.1, 1e-8) if finite.size else 1e-4
        logits = -costs / float(tau)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        alpha = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    q = np.sum(cand_stack * alpha[..., None, :], axis=-1)
    q = np.clip(restore_channels(q, was_gray), 0.0, 1.0)
    return q, alpha, direction_index, names
