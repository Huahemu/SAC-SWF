"""Reusable experiment runners."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import numpy as np

from .filters import bilateral_filter, gaussian_filter, guided_filter
from .io import save_image, save_rows_csv
from .metrics import evaluate_structure, timer
from .sac_swf import SACSWFConfig, sac_swf
from .side_window import aggregate_side_windows, side_window_guided_candidates
from .utils import ensure_dir
from .visualization import residual_image, save_comparison_grid, save_intermediate_maps

MethodFn = Callable[[np.ndarray, SACSWFConfig], tuple[np.ndarray, dict]]


def process_one_sample(
    image: np.ndarray,
    out_dir: str | Path,
    cfg: SACSWFConfig,
    target: np.ndarray | None = None,
    sample_id: str = "sample",
) -> list[dict]:
    """Run all methods on one sample and save outputs."""
    root = ensure_dir(Path(out_dir) / sample_id)
    save_image(root / "input.png", image)
    if target is not None:
        save_image(root / "structure_gt.png", target)

    rows = []
    grid_images = {"input": image}
    if target is not None:
        grid_images["structure_gt"] = target

    method_results: dict[str, tuple[np.ndarray, dict, float]] = {}

    with timer() as elapsed:
        method_results["gaussian"] = (
            gaussian_filter(image, sigma=max(1.0, cfg.radius / 2)),
            {},
            0.0,
        )
    method_results["gaussian"] = (
        method_results["gaussian"][0],
        method_results["gaussian"][1],
        elapsed["seconds"],
    )

    with timer() as elapsed:
        method_results["bilateral"] = (bilateral_filter(image), {}, 0.0)
    method_results["bilateral"] = (
        method_results["bilateral"][0],
        method_results["bilateral"][1],
        elapsed["seconds"],
    )

    with timer() as elapsed:
        q_guided = guided_filter(image, radius=cfg.radius, eps=cfg.eps)
    method_results["guided"] = (q_guided, {}, elapsed["seconds"])

    with timer() as elapsed:
        candidates = side_window_guided_candidates(image, radius=cfg.radius, eps=cfg.eps)
        q_hard, alpha_hard, dir_hard, names = aggregate_side_windows(image, candidates, hard=True)
    method_results["hard_swgf"] = (
        q_hard,
        {"alpha": alpha_hard, "direction_index": dir_hard, "direction_names": names},
        elapsed["seconds"],
    )

    with timer() as elapsed:
        q_soft, alpha_soft, dir_soft, names = aggregate_side_windows(
            image,
            candidates,
            tau=cfg.tau,
            lambda_grad=cfg.lambda_grad,
            hard=False,
        )
    method_results["soft_swgf"] = (
        q_soft,
        {"alpha": alpha_soft, "direction_index": dir_soft, "direction_names": names},
        elapsed["seconds"],
    )

    with timer() as elapsed:
        q_fixed = np.clip(0.5 * q_guided + 0.5 * q_soft, 0.0, 1.0)
    method_results["fixed_mix"] = (q_fixed, {}, elapsed["seconds"])

    with timer() as elapsed:
        q_sac, inter = sac_swf(image, config=cfg, return_intermediates=True)
    method_results["sac_swf"] = (q_sac, inter, elapsed["seconds"])

    for name, (output, intermediates, runtime_sec) in method_results.items():
        method_dir = ensure_dir(root / name)
        save_image(method_dir / "output.png", output)
        save_image(method_dir / "residual_vis.png", residual_image(image, output))
        if intermediates:
            save_intermediate_maps(method_dir / "intermediates", intermediates)
        row = {"sample_id": sample_id, "method": name, "runtime_sec": runtime_sec}
        if target is not None:
            row.update(evaluate_structure(output, target, inp=image))
        rows.append(row)
        grid_images[name] = output

    save_rows_csv(rows, root / "metrics.csv")
    save_comparison_grid(root / "comparison_grid.png", grid_images, cols=4)
    return rows


def process_method_set(
    image: np.ndarray,
    out_dir: str | Path,
    cfg: SACSWFConfig,
    methods: dict[str, MethodFn],
    target: np.ndarray | None = None,
    sample_id: str = "sample",
    experiment: str = "",
) -> list[dict]:
    """Run a named method set and save outputs, residuals, intermediates and metrics."""
    root = ensure_dir(Path(out_dir) / sample_id)
    save_image(root / "input.png", image)
    if target is not None:
        save_image(root / "structure_gt.png", target)

    rows: list[dict] = []
    grid_images = {"input": image}
    if target is not None:
        grid_images["structure_gt"] = target

    for name, method in methods.items():
        method_dir = ensure_dir(root / name)
        row = {"sample_id": sample_id, "method": name, "experiment": experiment}
        try:
            with timer() as elapsed:
                output, intermediates = method(image, cfg)
            row["runtime_sec"] = elapsed["seconds"]
            save_image(method_dir / "output.png", output)
            save_image(method_dir / "residual_vis.png", residual_image(image, output))
            if intermediates:
                save_intermediate_maps(method_dir / "intermediates", intermediates)
            if target is not None:
                row.update(evaluate_structure(output, target, inp=image))
            grid_images[name] = output
            row["status"] = "ok"
            error_path = method_dir / "error.txt"
            if error_path.exists():
                error_path.unlink()
        except Exception as exc:  # pragma: no cover - experiment diagnostics
            row["runtime_sec"] = np.nan
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            with (method_dir / "error.txt").open("w", encoding="utf-8") as f:
                f.write(row["error"])
        rows.append(row)

    save_rows_csv(rows, root / "metrics.csv")
    save_comparison_grid(root / "comparison_grid.png", grid_images, cols=4)
    return rows
