"""Run SAC-SWF on real images placed in data/raw/real."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

import _bootstrap  # noqa: F401
from sac_swf.experiment import process_one_sample
from sac_swf.io import list_images, read_image, save_rows_csv
from sac_swf.sac_swf import SACSWFConfig
from sac_swf.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/real")
    parser.add_argument("--out", default="results/exp03_real_images")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    images = list_images(args.data)
    if not images:
        raise SystemExit("No images found. Put images under data/raw/real first.")
    out = ensure_dir(args.out)
    cfg = SACSWFConfig(radius=5, eps=1e-3, gamma=0.5)
    rows = []
    for path in tqdm(images[: args.limit], desc="real images"):
        image = read_image(path)
        sample_id = path.stem
        rows.extend(process_one_sample(image, out, cfg, target=None, sample_id=sample_id))
    save_rows_csv(rows, out / "runtime_all.csv")
    print(f"Saved real-image experiment to {out}")


if __name__ == "__main__":
    main()
