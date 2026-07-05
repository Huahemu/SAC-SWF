"""Run Experiment 6: data-driven learning baseline plus optional external outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

import _bootstrap  # noqa: F401
from sac_swf.baselines import available_extended_methods
from sac_swf.experiment import process_method_set
from sac_swf.io import read_image, save_image, save_rows_csv
from sac_swf.learning_baseline import (
    PatchRidgeConfig,
    fit_patch_ridge,
    model_metadata,
    predict_patch_ridge,
)
from sac_swf.metrics import evaluate_structure, timer
from sac_swf.official_deep import inspect_official_deep_specs, official_deep_methods
from sac_swf.sac_swf import SACSWFConfig
from sac_swf.utils import ensure_dir, save_json
from sac_swf.visualization import residual_image, save_comparison_grid, save_heatmap


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_external_output(method_dir: Path, sample_id: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = method_dir / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate
    matches = list(method_dir.glob(f"{sample_id}.*"))
    return next((p for p in matches if p.suffix.lower() in IMAGE_EXTS), None)


def split_samples(sample_dirs: list[Path], train_limit: int, test_limit: int) -> tuple[list[Path], list[Path]]:
    if len(sample_dirs) < 2:
        raise SystemExit("Experiment 6 needs at least two synthetic samples. Run scripts/01_generate_synthetic.py first.")
    train_count = min(max(1, train_limit), len(sample_dirs) - 1)
    train = sample_dirs[:train_count]
    test = sample_dirs[train_count : train_count + test_limit]
    if not test:
        test = sample_dirs[-1:]
    return train, test


def write_status(
    out: Path,
    external_root: Path,
    external_method_count: int,
    model_meta: dict,
    train_samples: list[Path],
    test_samples: list[Path],
    official_specs: list[dict],
) -> None:
    text = [
        "# Experiment 6 Learning-Method Comparison Status",
        "",
        "This experiment now contains a project-internal data-driven baseline:",
        "",
        "- method: patch_ridge",
        "- type: supervised patch ridge regression",
        "- training target: synthetic structure_gt center pixel",
        "- input feature: local grayscale patch plus bias",
        "",
        "External deep-learning outputs are still supported, but none are synthesized by this project.",
        "",
        f"- external_output_root: {external_root}",
        f"- detected_external_method_count: {external_method_count}",
        f"- train_samples: {', '.join(p.name for p in train_samples)}",
        f"- test_samples: {', '.join(p.name for p in test_samples)}",
        f"- patch_radius: {model_meta['patch_radius']}",
        f"- ridge_lambda: {model_meta['ridge_lambda']}",
        f"- samples_per_image: {model_meta['samples_per_image']}",
        f"- train_mse: {model_meta['train_mse']:.8f}",
        "",
        "Official deep-network reproduction status:",
        *[
            f"- {item['method']}: code_ready={item['code_ready']}, weight_ready={item['weight_ready']}, ready_for_inference={item['ready_for_inference']}, weight_bytes={item['weight_actual_bytes']}"
            for item in official_specs
        ],
        "",
        "If detected_external_method_count is zero, the report may claim a lightweight data-driven baseline comparison,",
        "and may claim only the official deep methods whose ready_for_inference is true and whose sample outputs exist.",
        "Place external outputs as data/raw/learning_outputs/<method>/<sample_id>.png to enable that comparison.",
    ]
    (out / "learning_baseline_status.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic")
    parser.add_argument("--external", default="data/raw/learning_outputs")
    parser.add_argument("--out", default="results/exp06_learning_comparison")
    parser.add_argument("--train-limit", type=int, default=4)
    parser.add_argument("--test-limit", "--limit", dest="test_limit", type=int, default=4)
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--samples-per-image", type=int, default=2500)
    args = parser.parse_args()

    out = ensure_dir(args.out)
    cfg = SACSWFConfig(radius=5, eps=1e-3, gamma=0.5)
    sample_dirs = sorted(p for p in Path(args.data).iterdir() if p.is_dir())
    train_samples, test_samples = split_samples(sample_dirs, args.train_limit, args.test_limit)

    model_dir = ensure_dir(out / "learned_models")
    patch_cfg = PatchRidgeConfig(
        patch_radius=args.patch_radius,
        ridge_lambda=args.ridge_lambda,
        samples_per_image=args.samples_per_image,
        seed=1234,
    )
    model = fit_patch_ridge(
        train_samples,
        patch_cfg,
        model_dir / "patch_ridge_model.npz",
        model_dir / "patch_ridge_train_log.csv",
    )
    meta = model_metadata(model)
    save_json(meta, model_dir / "patch_ridge_metadata.json")
    weights = np.asarray(model["weights"], dtype=np.float64)
    kernel = weights[:-1].reshape((2 * args.patch_radius + 1, 2 * args.patch_radius + 1))
    pd.DataFrame(kernel).to_csv(model_dir / "patch_ridge_kernel.csv", index=False, header=False, encoding="utf-8-sig")
    save_heatmap(model_dir / "patch_ridge_kernel.png", kernel, title="patch_ridge_kernel", cmap="coolwarm")

    split_rows = (
        [{"sample_id": p.name, "split": "train"} for p in train_samples]
        + [{"sample_id": p.name, "split": "test"} for p in test_samples]
    )
    pd.DataFrame(split_rows).to_csv(out / "train_test_split.csv", index=False, encoding="utf-8-sig")

    methods = available_extended_methods(include_tv=True)

    def patch_ridge_method(image, _cfg):
        return predict_patch_ridge(image, model), {}

    methods["patch_ridge"] = patch_ridge_method
    for method_name, fn in official_deep_methods().items():
        methods[method_name] = (lambda image, _cfg, fn=fn: (fn(image), {}))

    official_specs = inspect_official_deep_specs()
    (out / "official_deep_status.json").write_text(json.dumps(official_specs, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "method": item["method"],
                "display_name": item["display_name"],
                "author_repo": item["author_repo"],
                "execution_repo": item["execution_repo"],
                "code_ready": item["code_ready"],
                "weight_ready": item["weight_ready"],
                "ready_for_inference": item["ready_for_inference"],
                "weight_path": item["weight_path"],
                "weight_url": item["weight_url"],
                "weight_actual_bytes": item["weight_actual_bytes"],
                "expected_min_bytes": item["expected_min_bytes"],
            }
            for item in official_specs
        ]
    ).to_csv(out / "official_deep_status.csv", index=False, encoding="utf-8-sig")

    rows = []
    internal_out = ensure_dir(out / "internal_learning")
    for sample_dir in tqdm(test_samples, desc="exp06 internal learning"):
        image = read_image(sample_dir / "input.png", grayscale=True)
        target = read_image(sample_dir / "structure_gt.png", grayscale=True)
        sample_rows = process_method_set(
            image,
            internal_out,
            cfg,
            methods,
            target=target,
            sample_id=sample_dir.name,
            experiment="exp06_internal_learning",
        )
        for row in sample_rows:
            row["split"] = "test"
            row["learning_baseline"] = row["method"] == "patch_ridge" or str(row["method"]).startswith("official_")
        rows.extend(sample_rows)

    external_root = Path(args.external)
    external_methods = [p for p in external_root.iterdir() if p.is_dir()] if external_root.exists() else []
    external_rows = []
    if external_methods:
        external_out = ensure_dir(out / "external_learning")
        for sample_dir in tqdm(test_samples, desc="exp06 external learning"):
            image = read_image(sample_dir / "input.png", grayscale=True)
            target = read_image(sample_dir / "structure_gt.png", grayscale=True)
            grid = {"input": image, "structure_gt": target}
            sample_out = ensure_dir(external_out / sample_dir.name)
            save_image(sample_out / "input.png", image)
            save_image(sample_out / "structure_gt.png", target)
            for method_dir in external_methods:
                found = find_external_output(method_dir, sample_dir.name)
                if found is None:
                    continue
                method_out = ensure_dir(sample_out / method_dir.name)
                with timer() as elapsed:
                    pred = read_image(found, grayscale=True)
                save_image(method_out / "output.png", pred)
                save_image(method_out / "residual_vis.png", residual_image(image, pred))
                shutil.copy2(found, method_out / f"source_{found.name}")
                row = {
                    "sample_id": sample_dir.name,
                    "method": method_dir.name,
                    "experiment": "exp06_external_learning",
                    "runtime_sec": elapsed["seconds"],
                    "status": "ok_external_output",
                    "split": "test",
                    "learning_baseline": True,
                }
                row.update(evaluate_structure(pred, target, inp=image))
                external_rows.append(row)
                grid[method_dir.name] = pred
            save_comparison_grid(sample_out / "comparison_grid.png", grid, cols=4)

    rows.extend(external_rows)
    save_rows_csv(rows, out / "metrics_all.csv")
    df = pd.DataFrame(rows)
    if not df.empty:
        df.groupby("method").mean(numeric_only=True).reset_index().to_csv(
            out / "metrics_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    write_status(out, external_root, len(external_methods), meta, train_samples, test_samples, official_specs)
    print(f"Saved Experiment 6 learning-comparison results to {out}")


if __name__ == "__main__":
    main()
