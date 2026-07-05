"""Collect already generated experiment images into report figure folders."""

from __future__ import annotations

import shutil
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    report_dir = project / "reports" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in [
        "results/exp01_window_principle/*/comparison_grid.png",
        "results/exp02_synthetic/*/comparison_grid.png",
        "results/exp03_real_images/*/comparison_grid.png",
        "results/exp04_ablation/*/*/comparison_grid.png",
        "results/exp05_benchmark/*/comparison_grid.png",
        "results/exp06_learning_comparison/*/*/comparison_grid.png",
        "results/exp07_paper_reproduction/*/comparison_grid.png",
        "results/exp07_paper_reproduction/*/line_profile_y*.png",
        "results/**/sac_swf/intermediates/beta.png",
        "results/**/sac_swf/intermediates/direction_index.png",
    ]:
        for src in project.glob(pattern):
            rel_name = "_".join(src.relative_to(project).parts)
            dst = report_dir / rel_name
            shutil.copy2(src, dst)
            copied += 1
    print(f"Copied {copied} figures to {report_dir}")


if __name__ == "__main__":
    main()
