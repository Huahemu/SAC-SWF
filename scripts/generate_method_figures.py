"""Generate method-section illustrative figures (SWF Fig.1/Fig.2 style)."""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(r"C:/Users/15001/WorkBuddy/2026-07-05-18-21-50/reports/paper_figures_new")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 220})


def fig_window_definition():
    """SWF Fig.2 style: full-window vs 8 side-windows on a pixel near edge."""
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2))
    titles = ["Full", "L", "R", "U", "D", "NW", "NE", "SW", "SE", ""]
    r = 3
    h = 0.5  # half pixel
    x0 = y0 = -r - h
    x1 = y1 = r + h
    full = x1 - x0  # full window side length
    half = full / 2
    for idx, ax in enumerate(axes.flat):
        if idx >= 9:
            ax.axis("off")
            continue
        name = titles[idx]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        # draw edge (a diagonal line for illustration)
        ax.plot([-r, r], [r, -r], "k-", lw=1.0, alpha=0.4)
        # target pixel at center
        ax.plot(0, 0, "ro", ms=7, zorder=5)
        # draw side-window support; every support touches the shared square frame
        # and any side/quarter boundary passes through the target pixel.
        if name == "Full":
            rect = Rectangle((x0, y0), full, full, fill=True, facecolor="#4C72B0", alpha=0.35, edgecolor="k", lw=1.2)
        elif name == "L":  # left half of full window
            rect = Rectangle((x0, y0), half, full, fill=True, facecolor="#55A868", alpha=0.35, edgecolor="k", lw=1.2)
        elif name == "R":  # right half
            rect = Rectangle((0, y0), half, full, fill=True, facecolor="#55A868", alpha=0.35, edgecolor="k", lw=1.2)
        elif name == "U":  # upper half
            rect = Rectangle((x0, 0), full, half, fill=True, facecolor="#C44E52", alpha=0.35, edgecolor="k", lw=1.2)
        elif name == "D":  # lower half
            rect = Rectangle((x0, y0), full, half, fill=True, facecolor="#C44E52", alpha=0.35, edgecolor="k", lw=1.2)
        elif name == "NW":  # upper-left quarter
            rect = Rectangle((x0, 0), half, half, fill=True, facecolor="#8172B2", alpha=0.35, edgecolor="k", lw=1.2)
        elif name == "NE":  # upper-right quarter
            rect = Rectangle((0, 0), half, half, fill=True, facecolor="#8172B2", alpha=0.35, edgecolor="k", lw=1.2)
        elif name == "SW":  # lower-left quarter
            rect = Rectangle((x0, y0), half, half, fill=True, facecolor="#8172B2", alpha=0.35, edgecolor="k", lw=1.2)
        elif name == "SE":  # lower-right quarter
            rect = Rectangle((0, y0), half, half, fill=True, facecolor="#8172B2", alpha=0.35, edgecolor="k", lw=1.2)
        ax.add_patch(rect)
        ax.set_title(name, fontsize=10)
    plt.tight_layout()
    fig.savefig(OUT / "method_fig_window_definition.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("  [saved] method_fig_window_definition.png")


def fig_beta_behavior():
    """Illustrate beta behavior on different local region types."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    # three region types: structure edge, texture, flat
    types = [
        ("Structure edge", "edge"),
        ("Texture region", "texture"),
        ("Flat region", "flat"),
    ]
    for c, (title, kind) in enumerate(types):
        ax = axes[0, c]
        s = 64
        if kind == "edge":
            img = np.zeros((s, s), dtype=np.float32)
            img[:, s // 2:] = 1.0
        elif kind == "texture":
            yy, xx = np.mgrid[0:s, 0:s]
            img = 0.5 + 0.3 * np.sin(xx / 3.0) * np.cos(yy / 3.0)
            img = np.stack([img] * 3, axis=-1).astype(np.float32)
        else:
            img = np.full((s, s, 3), 0.5, dtype=np.float32)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        # beta map
        ax = axes[1, c]
        if kind == "edge":
            beta = np.zeros((s, s), dtype=np.float32)
            beta[:, :s // 2] = 0.1
            beta[:, s // 2:] = 0.1
            beta[:, s // 2 - 3:s // 2 + 3] = 0.05
        elif kind == "texture":
            beta = np.full((s, s), 0.85, dtype=np.float32)
        else:
            beta = np.full((s, s), 0.7, dtype=np.float32)
        im = ax.imshow(beta, cmap="RdBu_r", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"β (low=side, high=full)", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "method_fig_beta_behavior.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("  [saved] method_fig_beta_behavior.png")


def fig_center_vs_side():
    """SWF Fig.1 style: why center window crosses edge."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    # two regions
    ax.add_patch(Rectangle((0, 0), 6, 5, facecolor="#3a3a3a", edgecolor="k"))
    ax.add_patch(Rectangle((6, 0), 6, 5, facecolor="#d0d0d0", edgecolor="k"))
    ax.text(3, 4.5, "region A", color="white", fontsize=11, ha="center")
    ax.text(9, 4.5, "region B", color="black", fontsize=11, ha="center")
    # center window on edge pixel (at x=6)
    ax.add_patch(Rectangle((3, 1), 6, 3, fill=False, edgecolor="#d62728", lw=2.2))
    ax.plot(6, 2.5, "o", color="#d62728", ms=8, zorder=5)
    ax.text(6, 0.4, "center window\n(crosses both regions)", color="#d62728", fontsize=9, ha="center")
    # side window (R) on same pixel
    ax.add_patch(Rectangle((6, 1), 3, 3, fill=False, edgecolor="#2ca02c", lw=2.2, linestyle="--"))
    ax.text(7.5, 3.5, "side window R\n(one region only)", color="#2ca02c", fontsize=9, ha="center")
    ax.set_title("Why center window blurs edges: it covers both regions", fontsize=11)
    fig.savefig(OUT / "method_fig_center_vs_side.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("  [saved] method_fig_center_vs_side.png")


def main():
    print("Output dir:", OUT)
    fig_window_definition()
    fig_beta_behavior()
    fig_center_vs_side()
    print("[done] method figures generated")


if __name__ == "__main__":
    main()
