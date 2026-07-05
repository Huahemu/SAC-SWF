"""Lightweight data-driven baselines for Experiment 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .io import read_image
from .metrics import evaluate_structure
from .utils import ensure_dir, rgb_to_gray


@dataclass
class PatchRidgeConfig:
    """Configuration for the patch ridge regression baseline."""

    patch_radius: int = 2
    ridge_lambda: float = 1e-2
    samples_per_image: int = 2500
    seed: int = 1234


def _sample_patch_matrix(
    image: np.ndarray,
    target: np.ndarray,
    cfg: PatchRidgeConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rgb_to_gray(image)
    y = rgb_to_gray(target)
    r = int(cfg.patch_radius)
    padded = np.pad(x, r, mode="reflect")
    h, w = x.shape
    count = min(int(cfg.samples_per_image), h * w)
    flat = rng.choice(h * w, size=count, replace=False)
    yy = flat // w
    xx = flat % w
    patch_size = 2 * r + 1
    features = np.empty((count, patch_size * patch_size + 1), dtype=np.float64)
    for idx, (row, col) in enumerate(zip(yy, xx, strict=False)):
        patch = padded[row : row + patch_size, col : col + patch_size]
        features[idx, :-1] = patch.reshape(-1)
        features[idx, -1] = 1.0
    labels = y[yy, xx]
    return features, labels


def fit_patch_ridge(
    sample_dirs: list[Path],
    cfg: PatchRidgeConfig,
    model_path: str | Path,
    log_path: str | Path,
) -> dict:
    """Fit a patch-to-center ridge regressor from synthetic input/structure pairs."""
    rng = np.random.default_rng(cfg.seed)
    xs = []
    ys = []
    train_rows = []
    for sample_dir in sample_dirs:
        image = read_image(sample_dir / "input.png", grayscale=True)
        target = read_image(sample_dir / "structure_gt.png", grayscale=True)
        x_i, y_i = _sample_patch_matrix(image, target, cfg, rng)
        xs.append(x_i)
        ys.append(y_i)
        train_rows.append({"sample_id": sample_dir.name, "pixels": len(y_i)})

    if not xs:
        raise ValueError("No training samples for patch ridge baseline.")

    x = np.vstack(xs)
    y = np.concatenate(ys)
    reg = float(cfg.ridge_lambda) * np.eye(x.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + reg, x.T @ y)
    pred = np.clip(x @ weights, 0.0, 1.0)
    train_mse = float(np.mean((pred - y) ** 2))

    model = {
        "weights": weights,
        "patch_radius": cfg.patch_radius,
        "ridge_lambda": cfg.ridge_lambda,
        "samples_per_image": cfg.samples_per_image,
        "seed": cfg.seed,
        "train_mse": train_mse,
        "train_sample_ids": [p.name for p in sample_dirs],
    }
    ensure_dir(Path(model_path).parent)
    np.savez_compressed(model_path, **model)
    pd.DataFrame(train_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
    return model


def load_patch_ridge(model_path: str | Path) -> dict:
    """Load a patch ridge model saved by fit_patch_ridge."""
    data = np.load(model_path, allow_pickle=True)
    return {key: data[key].tolist() if data[key].shape == () else data[key] for key in data.files}


def predict_patch_ridge(image: np.ndarray, model: dict) -> np.ndarray:
    """Apply the learned local linear filter to an image."""
    weights = np.asarray(model["weights"], dtype=np.float64)
    r = int(model["patch_radius"])
    kernel = weights[:-1].reshape((2 * r + 1, 2 * r + 1))
    bias = float(weights[-1])
    x = rgb_to_gray(image)
    pred = ndi.correlate(x, kernel, mode="reflect") + bias
    return np.clip(pred, 0.0, 1.0)


def evaluate_patch_ridge_model(model: dict, sample_dirs: list[Path], out_csv: str | Path) -> pd.DataFrame:
    """Evaluate the learned model on held-out synthetic samples."""
    rows = []
    for sample_dir in sample_dirs:
        image = read_image(sample_dir / "input.png", grayscale=True)
        target = read_image(sample_dir / "structure_gt.png", grayscale=True)
        pred = predict_patch_ridge(image, model)
        row = {"sample_id": sample_dir.name, "method": "patch_ridge"}
        row.update(evaluate_structure(pred, target, inp=image))
        rows.append(row)
    df = pd.DataFrame(rows)
    ensure_dir(Path(out_csv).parent)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def model_metadata(model: dict) -> dict:
    """Return JSON/Markdown-friendly metadata for a fitted model."""
    return {
        "patch_radius": int(model["patch_radius"]),
        "ridge_lambda": float(model["ridge_lambda"]),
        "samples_per_image": int(model["samples_per_image"]),
        "seed": int(model["seed"]),
        "train_mse": float(model["train_mse"]),
        "train_sample_ids": list(model["train_sample_ids"]),
    }
