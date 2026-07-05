"""Run SAC-SWF and baselines on generated synthetic samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from sac_swf.experiment import process_one_sample
from sac_swf.io import read_image, save_rows_csv
from sac_swf.sac_swf import SACSWFConfig
from sac_swf.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--out", default="results/exp02_synthetic")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.5)
    args = parser.parse_args()

    data_root = Path(args.data)
    out = ensure_dir(args.out)
    cfg = SACSWFConfig(radius=args.radius, eps=args.eps, gamma=args.gamma)

    sample_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    rows = []
    for sample_dir in tqdm(sample_dirs[: args.limit], desc="synthetic eval"):
        image = read_image(sample_dir / "input.png", grayscale=True)
        target = read_image(sample_dir / "structure_gt.png", grayscale=True)
        rows.extend(process_one_sample(image, out, cfg, target=target, sample_id=sample_dir.name))
    save_rows_csv(rows, out / "metrics_all.csv")
    if rows:
        summary = pd.DataFrame(rows).groupby("method").mean(numeric_only=True).reset_index()
        summary.to_csv(out / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Saved synthetic experiment to {out}")


if __name__ == "__main__":
    main()
