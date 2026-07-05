"""Run Experiment 7: paper-style reproduction evidence for used methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from sac_swf.baselines import available_extended_methods
from sac_swf.experiment import process_method_set
from sac_swf.io import read_image, save_rows_csv
from sac_swf.sac_swf import SACSWFConfig
from sac_swf.synthetic import SyntheticConfig, generate_sample, make_structure, save_synthetic_sample
from sac_swf.utils import ensure_dir


def make_case(case_id: str, out: Path, size: int) -> tuple[np.ndarray, np.ndarray, str]:
    if case_id == "swf_step_edge":
        s = make_structure(size, "step")
        yy, xx = np.mgrid[0:size, 0:size]
        texture = 0.12 * np.sin(2 * np.pi * xx / 12.0)
        image = np.clip(s + texture, 0.0, 1.0)
        return image, s, "SWF edge-window principle: centered support crosses a step edge."
    if case_id == "swf_corner":
        s = make_structure(size, "corner")
        yy, xx = np.mgrid[0:size, 0:size]
        texture = 0.10 * np.sin(2 * np.pi * (xx + yy) / 18.0)
        image = np.clip(s + texture, 0.0, 1.0)
        return image, s, "SWF corner case: support selection must avoid two-sided leakage."
    if case_id == "guided_filter_edge":
        sample = generate_sample(SyntheticConfig(size=size, seed=71, texture_type="sine", texture_frequency=10.0, texture_amplitude=0.16))
        return sample["I"], sample["S"], "Guided filter comparison on structure-texture separation."
    if case_id == "bilateral_range_kernel":
        sample = generate_sample(SyntheticConfig(size=size, seed=72, texture_type="checkerboard", texture_frequency=20.0, texture_amplitude=0.16))
        return sample["I"], sample["S"], "Bilateral range-kernel behavior on high-contrast repetitive texture."
    if case_id == "regularized_texture":
        sample = generate_sample(SyntheticConfig(size=size, seed=73, texture_type="random", texture_frequency=18.0, texture_amplitude=0.22))
        return sample["I"], sample["S"], "Regularization-style smoothing comparison using TV Chambolle."
    raise ValueError(case_id)


def save_line_profile(sample_root: Path, method_names: list[str], y: int) -> None:
    rows = []
    input_img = read_image(sample_root / "input.png", grayscale=True)
    rows.append(pd.DataFrame({"x": np.arange(input_img.shape[1]), "value": input_img[y], "series": "input"}))
    target_path = sample_root / "structure_gt.png"
    if target_path.exists():
        target = read_image(target_path, grayscale=True)
        rows.append(pd.DataFrame({"x": np.arange(target.shape[1]), "value": target[y], "series": "structure_gt"}))
    for name in method_names:
        output_path = sample_root / name / "output.png"
        if output_path.exists():
            output = read_image(output_path, grayscale=True)
            rows.append(pd.DataFrame({"x": np.arange(output.shape[1]), "value": output[y], "series": name}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(sample_root / f"line_profile_y{y}.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4))
    for series, sub in df.groupby("series"):
        plt.plot(sub["x"], sub["value"], label=series, linewidth=1.3)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("x")
    plt.ylabel("intensity")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(sample_root / f"line_profile_y{y}.png", dpi=180)
    plt.close()


def write_scope(out: Path) -> None:
    text = [
        "# Experiment 7 Reproduction Scope",
        "",
        "This folder is for appendix evidence. It reproduces the experimental logic used by the papers/methods,",
        "using the implementations in this course project.",
        "",
        "- bilateral: bilateral filtering behavior after Tomasi and Manduchi.",
        "- guided: guided filter local-linear model after He, Sun and Tang.",
        "- hard_swgf / soft_swgf: side-window support-selection principle after the SWF paper.",
        "- fixed_mix: non-adaptive full/side mixture used as an ablation baseline.",
        "- sac_swf: proposed adaptive full/side combination, not an external paper result.",
        "- tv_chambolle: TV denoising baseline available in scikit-image, included only as a regularization reference.",
        "",
        "This is not a claim of bit-exact reproduction of original author code.",
        "If original code is later supplied, put its outputs under a separate appendix folder and keep this scope file.",
    ]
    (out / "REPRODUCTION_SCOPE.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/exp07_paper_reproduction")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--radius", type=int, default=7)
    args = parser.parse_args()

    out = ensure_dir(args.out)
    case_root = ensure_dir(out / "generated_cases")
    cfg = SACSWFConfig(radius=args.radius, eps=1e-3, gamma=0.5, lambda_grad=0.0)
    methods = available_extended_methods(include_tv=True)
    cases = [
        "swf_step_edge",
        "swf_corner",
        "guided_filter_edge",
        "bilateral_range_kernel",
        "regularized_texture",
    ]
    manifest_rows = []
    metric_rows = []
    for case_id in cases:
        image, target, note = make_case(case_id, case_root, args.size)
        sample = {"I": image, "S": target, "T": image - target, "N": np.zeros_like(target), "config": {"case_id": case_id, "note": note}}
        save_synthetic_sample(sample, case_root, case_id)
        metric_rows.extend(
            process_method_set(
                image,
                out,
                cfg,
                methods,
                target=target,
                sample_id=case_id,
                experiment="exp07_paper_reproduction",
            )
        )
        save_line_profile(out / case_id, list(methods.keys()), y=args.size // 2)
        manifest_rows.append({"sample_id": case_id, "note": note})

    save_rows_csv(metric_rows, out / "metrics_all.csv")
    pd.DataFrame(metric_rows).groupby("method").mean(numeric_only=True).reset_index().to_csv(
        out / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(manifest_rows).to_csv(out / "appendix_case_manifest.csv", index=False, encoding="utf-8-sig")
    write_scope(out)
    print(f"Saved Experiment 7 paper-style reproduction results to {out}")


if __name__ == "__main__":
    main()
