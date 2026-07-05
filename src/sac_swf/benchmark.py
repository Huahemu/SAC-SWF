"""Benchmark-suite preparation for Experiment 5."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import data
from skimage.transform import resize

from .io import save_image
from .synthetic import SyntheticConfig, generate_sample, save_synthetic_sample
from .utils import ensure_dir, save_json


REAL_SOURCES = [
    ("camera", "grayscale natural image with smooth regions and edges"),
    ("coins", "object boundaries with weak texture"),
    ("coffee", "color image with edges and smooth areas"),
    ("astronaut", "color portrait with mixed details"),
    ("chelsea", "fine animal-fur texture and semantic boundaries"),
    ("rocket", "strong edges and color changes"),
    ("brick", "regular high-frequency texture"),
    ("grass", "irregular high-frequency texture"),
]


SYNTHETIC_STRESS = [
    ("mixed", "sine", 10.0, 0.14, 201),
    ("mixed", "sine", 26.0, 0.26, 202),
    ("mixed", "checkerboard", 18.0, 0.18, 203),
    ("mixed", "checkerboard", 34.0, 0.28, 204),
    ("corner", "stripes", 16.0, 0.22, 205),
    ("lines", "stripes", 30.0, 0.18, 206),
    ("mixed", "random", 14.0, 0.24, 207),
    ("corner", "random", 24.0, 0.20, 208),
]


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def _resize01(image: np.ndarray, size: int) -> np.ndarray:
    image = _center_crop_square(image)
    shape = (size, size) if image.ndim == 2 else (size, size, image.shape[2])
    return resize(image, shape, anti_aliasing=True, preserve_range=False)


def prepare_benchmark_suite(
    out_root: str | Path = "data/benchmark",
    size: int = 256,
    real_limit: int = 6,
    synthetic_limit: int = 8,
) -> pd.DataFrame:
    """Create a deterministic benchmark suite with real and synthetic-stress subsets."""
    root = ensure_dir(out_root)
    rows: list[dict] = []

    real_root = ensure_dir(root / "real_public")
    for name, note in REAL_SOURCES[:real_limit]:
        if not hasattr(data, name):
            continue
        sample_id = f"real_{name}"
        sample_root = ensure_dir(real_root / sample_id)
        image = _resize01(getattr(data, name)(), size=size)
        save_image(sample_root / "input.png", image)
        save_json({"source": f"skimage.data.{name}", "note": note, "size": size}, sample_root / "config.json")
        rows.append(
            {
                "sample_id": sample_id,
                "subset": "real_public",
                "input_path": str(sample_root / "input.png"),
                "target_path": "",
                "has_target": False,
                "source": f"skimage.data.{name}",
                "note": note,
            }
        )

    syn_root = ensure_dir(root / "synthetic_stress")
    for idx, (structure, texture, freq, amp, seed) in enumerate(SYNTHETIC_STRESS[:synthetic_limit]):
        cfg = SyntheticConfig(
            size=size,
            seed=seed,
            texture_type=texture,
            texture_frequency=freq,
            texture_amplitude=amp,
            noise_sigma=0.025,
            structure_type=structure,
        )
        sample_id = f"stress_{idx:02d}_{structure}_{texture}_f{int(freq)}_a{int(amp * 100)}"
        sample = generate_sample(cfg)
        save_synthetic_sample(sample, syn_root, sample_id)
        sample_root = syn_root / sample_id
        rows.append(
            {
                "sample_id": sample_id,
                "subset": "synthetic_stress",
                "input_path": str(sample_root / "input.png"),
                "target_path": str(sample_root / "structure_gt.png"),
                "has_target": True,
                "source": "generated_synthetic_stress",
                "note": str(asdict(cfg)),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(root / "benchmark_index.csv", index=False, encoding="utf-8-sig")
    manifest = [
        "# Course Benchmark Suite",
        "",
        f"- size: {size}",
        f"- real_public_count: {sum(df['subset'] == 'real_public') if not df.empty else 0}",
        f"- synthetic_stress_count: {sum(df['subset'] == 'synthetic_stress') if not df.empty else 0}",
        "- official_eps: false",
        "",
        "This suite is deterministic and runnable without external downloads.",
        "It is not the official EPS Benchmark. If official EPS data is later placed in data/raw/eps,",
        "Experiment 5 will switch to official_eps_user_provided mode.",
    ]
    (root / "BENCHMARK_MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")
    return df
