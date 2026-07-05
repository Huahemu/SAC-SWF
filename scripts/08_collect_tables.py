"""Collect metrics CSV files into summary tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401


def main() -> None:
    root = Path("results")
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for csv_path in root.rglob("metrics*.csv"):
        if "tables" in csv_path.parts:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - diagnostics script
            print(f"Skip {csv_path}: {exc}")
            continue
        df["source_csv"] = str(csv_path)
        frames.append(df)

    if not frames:
        raise SystemExit("No metrics CSV files found under results.")

    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(table_dir / "all_metrics.csv", index=False, encoding="utf-8-sig")

    numeric = all_metrics.select_dtypes(include="number").columns.tolist()
    if "method" in all_metrics.columns and numeric:
        summary = all_metrics.groupby("method")[numeric].mean().reset_index()
        summary.to_csv(table_dir / "summary_by_method.csv", index=False, encoding="utf-8-sig")
        try:
            summary.to_latex(table_dir / "summary_by_method.tex", index=False, float_format="%.4f")
        except Exception as exc:
            print(f"LaTeX table export skipped: {exc}")
    print(f"Saved tables to {table_dir}")


if __name__ == "__main__":
    main()
