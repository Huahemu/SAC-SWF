"""Run Experiment 5: official EPS or deterministic course benchmark suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from sac_swf.baselines import available_extended_methods
from sac_swf.benchmark import prepare_benchmark_suite
from sac_swf.experiment import process_method_set
from sac_swf.io import list_images, read_image, save_rows_csv
from sac_swf.sac_swf import SACSWFConfig
from sac_swf.utils import ensure_dir


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _write_manifest(out: Path, mode: str, count: int, benchmark_root: Path, eps_root: Path) -> None:
    text = [
        "# Experiment 5 Benchmark Manifest",
        "",
        f"- dataset_mode: {mode}",
        f"- sample_count: {count}",
        f"- official_eps_root: {eps_root}",
        f"- course_benchmark_root: {benchmark_root}",
        f"- output_root: {out}",
        "",
        "Mode meanings:",
        "- official_eps_user_provided: images were found under data/raw/eps.",
        "- course_benchmark_suite: official EPS data was absent; data/benchmark was used.",
        "",
        "The course benchmark suite is deterministic and contains public real images plus synthetic stress cases.",
        "It is a complete runnable benchmark for this course project, but it is not the official EPS Benchmark.",
        "Do not write 'completed EPS Benchmark' unless dataset_mode is official_eps_user_provided and the dataset source is verified.",
    ]
    (out / "benchmark_manifest.md").write_text("\n".join(text), encoding="utf-8")


def _load_course_benchmark(root: Path, args: argparse.Namespace) -> pd.DataFrame:
    index = root / "benchmark_index.csv"
    if not index.exists():
        prepare_benchmark_suite(
            out_root=root,
            size=args.benchmark_size,
            real_limit=args.real_limit,
            synthetic_limit=args.synthetic_limit,
        )
    if not index.exists():
        raise SystemExit(f"Missing benchmark index: {index}")
    return pd.read_csv(index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", default="data/raw/eps")
    parser.add_argument("--benchmark", default="data/benchmark")
    parser.add_argument("--out", default="results/exp05_benchmark")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--benchmark-size", type=int, default=256)
    parser.add_argument("--real-limit", type=int, default=6)
    parser.add_argument("--synthetic-limit", type=int, default=8)
    args = parser.parse_args()

    out = ensure_dir(args.out)
    eps_root = Path(args.eps)
    benchmark_root = Path(args.benchmark)
    cfg = SACSWFConfig(radius=args.radius, eps=1e-3, gamma=0.5)
    methods = available_extended_methods(include_tv=True)
    rows: list[dict] = []

    eps_images = list_images(eps_root) if eps_root.exists() else []
    if eps_images:
        mode = "official_eps_user_provided"
        items = [
            {
                "sample_id": p.stem,
                "subset": "official_eps",
                "input_path": str(p),
                "target_path": "",
                "has_target": False,
                "source": str(p),
                "note": "User-provided EPS image; no target parsing is assumed by this script.",
            }
            for p in eps_images[: args.limit]
        ]
    else:
        mode = "course_benchmark_suite"
        df = _load_course_benchmark(benchmark_root, args)
        items = df.head(args.limit).to_dict("records")

    for item in tqdm(items, desc=f"exp05 {mode}"):
        input_path = Path(str(item["input_path"]))
        has_target = _truthy(item.get("has_target", False))
        target_path = Path(str(item.get("target_path", ""))) if has_target else None
        target = read_image(target_path, grayscale=True) if target_path and target_path.exists() else None
        image = read_image(input_path, grayscale=target is not None)
        sample_rows = process_method_set(
            image,
            out,
            cfg,
            methods,
            target=target,
            sample_id=str(item["sample_id"]),
            experiment="exp05_benchmark",
        )
        for row in sample_rows:
            row["dataset_mode"] = mode
            row["subset"] = item.get("subset", "")
            row["source"] = item.get("source", "")
            row["has_ground_truth"] = target is not None
        rows.extend(sample_rows)

    save_rows_csv(rows, out / "metrics_all.csv")
    run_index = pd.DataFrame(items)
    run_index.to_csv(out / "benchmark_run_index.csv", index=False, encoding="utf-8-sig")
    df_rows = pd.DataFrame(rows)
    if not df_rows.empty:
        df_rows.groupby(["dataset_mode", "subset", "method"]).mean(numeric_only=True).reset_index().to_csv(
            out / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    _write_manifest(out, mode, len(items), benchmark_root, eps_root)
    print(f"Saved Experiment 5 benchmark results to {out}")


if __name__ == "__main__":
    main()
