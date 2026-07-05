"""Synthetic structure-texture dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy import ndimage as ndi

from .io import save_image
from .utils import ensure_dir, normalize01, save_json


@dataclass
class SyntheticConfig:
    size: int = 256
    seed: int = 0
    texture_type: str = "sine"  # sine, checkerboard, stripes, random
    texture_frequency: float = 12.0
    texture_amplitude: float = 0.18
    noise_sigma: float = 0.02
    structure_type: str = "mixed"  # mixed, step, circle, lines


def make_structure(size: int, kind: str = "mixed") -> np.ndarray:
    """Create a clean structure layer S."""
    h = w = int(size)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.ones((h, w), dtype=np.float64) * 0.35
    if kind in {"mixed", "step"}:
        img[:, w // 2 :] += 0.35
    if kind in {"mixed", "circle"}:
        mask = (xx - 0.35 * w) ** 2 + (yy - 0.35 * h) ** 2 < (0.17 * w) ** 2
        img[mask] = 0.85
    if kind in {"mixed", "lines"}:
        img[np.abs(yy - (0.72 * h + 0.30 * (xx - w / 2))) < 2.2] = 0.08
        img[(yy > 0.15 * h) & (yy < 0.24 * h) & (xx > 0.12 * w) & (xx < 0.78 * w)] = 0.72
    if kind == "corner":
        img[(xx > 0.45 * w) & (yy > 0.45 * h)] = 0.82
    return np.clip(img, 0.0, 1.0)


def make_texture(
    size: int,
    texture_type: str = "sine",
    frequency: float = 12.0,
    amplitude: float = 0.18,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create a zero-mean texture layer T."""
    rng = rng or np.random.default_rng(0)
    h = w = int(size)
    yy, xx = np.mgrid[0:h, 0:w]
    if texture_type == "sine":
        angle = np.deg2rad(35.0)
        coord = np.cos(angle) * xx + np.sin(angle) * yy
        tex = np.sin(2.0 * np.pi * frequency * coord / w)
    elif texture_type == "checkerboard":
        period = max(2, int(round(w / frequency)))
        tex = (((xx // period + yy // period) % 2) * 2 - 1).astype(np.float64)
    elif texture_type == "stripes":
        period = max(2, int(round(w / frequency)))
        tex = (((xx // period) % 2) * 2 - 1).astype(np.float64)
    elif texture_type == "random":
        noise = rng.normal(size=(h, w))
        tex = ndi.gaussian_filter(noise, sigma=max(1.0, w / (frequency * 8.0)), mode="reflect")
        tex = normalize01(tex) * 2.0 - 1.0
    else:
        raise ValueError(f"Unknown texture_type: {texture_type}")
    tex = tex - float(tex.mean())
    denom = max(float(np.max(np.abs(tex))), 1e-12)
    return float(amplitude) * tex / denom


def generate_sample(cfg: SyntheticConfig) -> dict[str, np.ndarray | dict]:
    """Generate I, S, T, N maps for one synthetic sample."""
    rng = np.random.default_rng(cfg.seed)
    s = make_structure(cfg.size, cfg.structure_type)
    t = make_texture(
        cfg.size,
        cfg.texture_type,
        cfg.texture_frequency,
        cfg.texture_amplitude,
        rng=rng,
    )
    n = rng.normal(scale=cfg.noise_sigma, size=s.shape)
    i = np.clip(s + t + n, 0.0, 1.0)
    return {"I": i, "S": s, "T": t, "N": n, "config": asdict(cfg)}


def config_grid(
    size: int = 256,
    seeds: list[int] | None = None,
) -> Iterator[SyntheticConfig]:
    """Small deterministic grid for experiments."""
    seeds = seeds or [0, 1, 2]
    textures = ["sine", "checkerboard", "stripes", "random"]
    freqs = [8.0, 16.0, 28.0]
    amps = [0.10, 0.18, 0.28]
    for seed in seeds:
        for texture in textures:
            for freq in freqs:
                for amp in amps:
                    yield SyntheticConfig(
                        size=size,
                        seed=seed,
                        texture_type=texture,
                        texture_frequency=freq,
                        texture_amplitude=amp,
                        noise_sigma=0.02,
                        structure_type="mixed",
                    )


def save_synthetic_sample(sample: dict[str, np.ndarray | dict], out_dir: str | Path, sample_id: str) -> None:
    """Save one sample images and metadata."""
    root = ensure_dir(Path(out_dir) / sample_id)
    save_image(root / "input.png", sample["I"])  # type: ignore[arg-type]
    save_image(root / "structure_gt.png", sample["S"])  # type: ignore[arg-type]
    save_image(root / "texture_gt_vis.png", normalize01(sample["T"]))  # type: ignore[arg-type]
    save_image(root / "noise_gt_vis.png", normalize01(sample["N"]))  # type: ignore[arg-type]
    save_json(sample["config"], root / "config.json")  # type: ignore[arg-type]
