"""Generate synthetic structure-texture samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

import _bootstrap  # noqa: F401
from sac_swf.synthetic import config_grid, generate_sample, save_synthetic_sample
from sac_swf.io import save_rows_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic", help="Output directory.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=24)
    args = parser.parse_args()

    rows = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for idx, cfg in enumerate(tqdm(config_grid(size=args.size), desc="synthetic")):
        if idx >= args.max_samples:
            break
        sample_id = f"syn_{idx:04d}_{cfg.texture_type}_f{int(cfg.texture_frequency)}_a{int(cfg.texture_amplitude*100)}"
        sample = generate_sample(cfg)
        save_synthetic_sample(sample, out, sample_id)
        row = cfg.__dict__.copy()
        row["sample_id"] = sample_id
        rows.append(row)
    save_rows_csv(rows, out / "synthetic_index.csv")
    print(f"Saved {len(rows)} samples to {out}")


if __name__ == "__main__":
    main()
