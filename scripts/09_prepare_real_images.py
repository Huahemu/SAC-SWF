"""Prepare a small public real-image set from scikit-image examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from skimage import data
from skimage.transform import resize

import _bootstrap  # noqa: F401
from sac_swf.io import save_image
from sac_swf.utils import ensure_dir


SOURCES = [
    ("camera", "classic grayscale natural image"),
    ("coins", "object boundaries and mild texture"),
    ("coffee", "color image with smooth regions and object edges"),
    ("astronaut", "color portrait with edges and texture"),
    ("chelsea", "fine fur texture and semantic edges"),
    ("rocket", "color image with strong edges"),
    ("brick", "regular high-frequency texture"),
    ("grass", "irregular high-frequency texture"),
]


def center_crop_square(image):
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/real")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    out = ensure_dir(args.out)
    rows = []
    for name, note in SOURCES[: args.limit]:
        if not hasattr(data, name):
            continue
        image = getattr(data, name)()
        image = center_crop_square(image)
        resized = resize(
            image,
            (args.size, args.size) if image.ndim == 2 else (args.size, args.size, image.shape[2]),
            anti_aliasing=True,
            preserve_range=False,
        )
        path = out / f"skimage_{name}.png"
        save_image(path, resized)
        rows.append({"sample_id": path.stem, "source": f"skimage.data.{name}", "note": note})

    if not rows:
        raise SystemExit("No scikit-image sample images were available.")
    pd.DataFrame(rows).to_csv(out / "real_image_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} real images to {Path(out).resolve()}")


if __name__ == "__main__":
    main()
