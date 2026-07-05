"""Run core SAC-SWF ablation on synthetic samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from sac_swf.experiment import process_one_sample
from sac_swf.io import read_image
from sac_swf.sac_swf import SACSWFConfig
from sac_swf.utils import ensure_dir


ABLATIONS = {
    "base": SACSWFConfig(radius=5, eps=1e-3, gamma=0.5, lambda_grad=0.0),
    "hard_side": SACSWFConfig(radius=5, eps=1e-3, gamma=0.5, hard_side=True),
    "no_texture": SACSWFConfig(radius=5, eps=1e-3, gamma=0.0),
    "grad_cost": SACSWFConfig(radius=5, eps=1e-3, gamma=0.5, lambda_grad=0.2),
    "large_radius": SACSWFConfig(radius=9, eps=1e-3, gamma=0.5),
    "small_radius": SACSWFConfig(radius=3, eps=1e-3, gamma=0.5),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--out", default="results/exp04_ablation")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    sample_dirs = sorted(p for p in Path(args.data).iterdir() if p.is_dir())[: args.limit]
    out = ensure_dir(args.out)
    all_rows = []
    for name, cfg in ABLATIONS.items():
        exp_out = ensure_dir(out / name)
        for sample_dir in tqdm(sample_dirs, desc=f"ablation {name}"):
            image = read_image(sample_dir / "input.png", grayscale=True)
            target = read_image(sample_dir / "structure_gt.png", grayscale=True)
            rows = process_one_sample(image, exp_out, cfg, target=target, sample_id=sample_dir.name)
            for row in rows:
                row["ablation"] = name
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / "metrics_ablation_all.csv", index=False, encoding="utf-8-sig")
    df.groupby(["ablation", "method"]).mean(numeric_only=True).reset_index().to_csv(
        out / "metrics_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Saved ablation experiment to {out}")


if __name__ == "__main__":
    main()
