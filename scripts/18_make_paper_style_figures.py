"""Build paper-style figures from saved experiment outputs.

The figures generated here are intentionally selective: each one supports one
experimental claim in the report instead of displaying every saved result.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "paper_figures"


def read_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def show_image(ax: plt.Axes, path: Path, title: str, cmap: str | None = None) -> None:
    img = read_image(path)
    if cmap == "gray":
        ax.imshow(img[..., 0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_window_mechanism(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.add_patch(Rectangle((0, 0), 5, 5, color="0.18"))
    ax.add_patch(Rectangle((5, 0), 5, 5, color="0.82"))
    ax.plot([5, 5], [0, 5], "k-", lw=2)
    ax.add_patch(Rectangle((3.2, 1.1), 3.6, 2.8, fill=False, ec="#d62728", lw=2.2))
    ax.plot(5, 2.5, "o", color="#d62728", ms=6)
    ax.text(5, 4.35, "center window crosses edge", ha="center", fontsize=9)
    ax.text(1.1, 0.35, "region A", color="white", fontsize=9)
    ax.text(7.0, 0.35, "region B", color="black", fontsize=9)
    ax.axis("off")


def plot_side_window(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.add_patch(Rectangle((0, 0), 5, 5, color="0.18"))
    ax.add_patch(Rectangle((5, 0), 5, 5, color="0.82"))
    ax.plot([5, 5], [0, 5], "k-", lw=2)
    ax.add_patch(Rectangle((1.4, 1.1), 3.6, 2.8, fill=False, ec="#1f77b4", lw=2.2))
    ax.plot(5, 2.5, "o", color="#1f77b4", ms=6)
    ax.text(5, 4.35, "side window stays on one side", ha="center", fontsize=9)
    ax.text(1.1, 0.35, "region A", color="white", fontsize=9)
    ax.text(7.0, 0.35, "region B", color="black", fontsize=9)
    ax.axis("off")


def make_mechanism_figure() -> None:
    csv_path = ROOT / "results" / "exp07_paper_reproduction" / "swf_step_edge" / "line_profile_y96.csv"
    df = pd.read_csv(csv_path)
    keep = ["input", "structure_gt", "gaussian", "hard_swgf", "sac_swf"]
    colors = {
        "input": "0.45",
        "structure_gt": "black",
        "gaussian": "#d62728",
        "hard_swgf": "#1f77b4",
        "sac_swf": "#2ca02c",
    }

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15])
    plot_window_mechanism(fig.add_subplot(gs[0, 0]))
    plot_side_window(fig.add_subplot(gs[0, 1]))
    ax = fig.add_subplot(gs[1, :])
    for series in keep:
        sub = df[df["series"] == series]
        if not sub.empty:
            ax.plot(sub["x"], sub["value"], lw=2, label=series.replace("_", " "), color=colors[series])
    ax.set_title("Line profile on a step edge", fontsize=11)
    ax.set_xlabel("pixel position")
    ax.set_ylabel("intensity")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=8, loc="upper center")
    save(fig, "fig01_window_principle_profile.png")


def make_texture_tradeoff_figure() -> None:
    base = ROOT / "results" / "exp02_synthetic" / "syn_0000_sine_f8_a10"
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    items = [
        ("input.png", "Input"),
        ("structure_gt.png", "Structure GT"),
        ("sac_swf/intermediates/q_full.png", "Full-window candidate"),
        ("sac_swf/intermediates/q_side.png", "Side-window candidate"),
        ("guided/output.png", "Guided"),
        ("hard_swgf/output.png", "Hard SWGF"),
        ("sac_swf/output.png", "SAC-SWF"),
        ("sac_swf/intermediates/beta.png", "Full-window weight"),
    ]
    for ax, (rel, title) in zip(axes.flat, items, strict=True):
        show_image(ax, base / rel, title)
    save(fig, "fig02_structure_texture_tradeoff.png")


def make_weight_diagnostics_figure() -> None:
    base = ROOT / "results" / "exp02_synthetic" / "syn_0000_sine_f8_a10" / "sac_swf" / "intermediates"
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    items = [
        ("structure_conf.png", "Structure confidence"),
        ("coherence.png", "Coherence"),
        ("oscillation.png", "Non-coherent texture"),
        ("direction_index.png", "Selected direction"),
    ]
    for ax, (rel, title) in zip(axes, items, strict=True):
        show_image(ax, base / rel, title)
    save(fig, "fig03_weight_diagnostics.png")


def make_stress_case_figure() -> None:
    base = ROOT / "results" / "exp05_benchmark" / "stress_00_mixed_sine_f10_a14"
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    items = [
        ("input.png", "Input"),
        ("structure_gt.png", "Structure GT"),
        ("gaussian/output.png", "Gaussian"),
        ("bilateral/output.png", "Bilateral"),
        ("tv_chambolle/output.png", "TV"),
        ("hard_swgf/output.png", "Hard SWGF"),
        ("sac_swf/output.png", "SAC-SWF"),
        ("sac_swf/intermediates/beta.png", "Full-window weight"),
    ]
    for ax, (rel, title) in zip(axes.flat, items, strict=True):
        show_image(ax, base / rel, title)
    save(fig, "fig04_stress_case.png")


def make_learning_comparison_figure() -> None:
    base = ROOT / "results" / "exp06_learning_comparison" / "internal_learning" / "syn_0004_sine_f16_a18"
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    items = [
        ("input.png", "Input"),
        ("structure_gt.png", "Structure GT"),
        ("sac_swf/output.png", "SAC-SWF"),
        ("patch_ridge/output.png", "Patch ridge"),
        ("official_dncnn/output.png", "DnCNN"),
        ("official_drunet/output.png", "DRUNet"),
        ("official_restormer/output.png", "Restormer"),
        ("sac_swf/intermediates/beta.png", "SAC-SWF weight"),
    ]
    for ax, (rel, title) in zip(axes.flat, items, strict=True):
        show_image(ax, base / rel, title)
    save(fig, "fig05_learning_comparison.png")


def make_real_image_figure() -> None:
    base = ROOT / "results" / "exp03_real_images" / "skimage_camera"
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    items = [
        ("input.png", "Input"),
        ("guided/output.png", "Guided"),
        ("hard_swgf/output.png", "Hard SWGF"),
        ("sac_swf/output.png", "SAC-SWF"),
        ("sac_swf/intermediates/beta.png", "Full-window weight"),
        ("sac_swf/intermediates/direction_index.png", "Direction index"),
    ]
    for ax, (rel, title) in zip(axes.flat, items, strict=True):
        show_image(ax, base / rel, title)
    save(fig, "fig06_real_image_diagnostics.png")


def main() -> None:
    make_mechanism_figure()
    make_texture_tradeoff_figure()
    make_weight_diagnostics_figure()
    make_stress_case_figure()
    make_learning_comparison_figure()
    make_real_image_figure()


if __name__ == "__main__":
    main()
