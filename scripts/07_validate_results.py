"""Validate result folders for required output files."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default="results", help="Result root to validate.")
    args = parser.parse_args()
    root = Path(args.result)
    if not root.exists():
        raise SystemExit(f"Missing result directory: {root}")

    comparison_grids = list(root.rglob("comparison_grid.png"))
    metrics_files = list(root.rglob("metrics.csv")) + list(root.rglob("metrics_all.csv"))
    sac_beta = list(root.rglob("sac_swf/intermediates/beta.png"))
    expected_dirs = [
        "exp01_window_principle",
        "exp02_synthetic",
        "exp03_real_images",
        "exp04_ablation",
        "exp05_benchmark",
        "exp06_learning_comparison",
        "exp07_paper_reproduction",
    ]

    print(f"Result root: {root.resolve()}")
    print(f"comparison_grid.png files: {len(comparison_grids)}")
    print(f"metrics files: {len(metrics_files)}")
    print(f"SAC-SWF beta maps: {len(sac_beta)}")
    for name in expected_dirs:
        print(f"{name}: {'OK' if (root / name).exists() else 'MISSING'}")

    problems = []
    if not comparison_grids:
        problems.append("No comparison_grid.png found.")
    if not metrics_files:
        problems.append("No metrics CSV found.")
    if not sac_beta:
        problems.append("No sac_swf/intermediates/beta.png found.")
    for name in expected_dirs:
        if not (root / name).exists():
            problems.append(f"Missing result directory: {name}")

    if problems:
        print("VALIDATION FAILED")
        for p in problems:
            print(f"- {p}")
        raise SystemExit(1)

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
