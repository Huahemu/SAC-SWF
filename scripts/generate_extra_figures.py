"""Generate additional figures for the expanded experiment section.

Adds:
- frequency response of side-window kernels (OSBF Fig.10 style)
- parameter analysis curves (radius / iteration vs PSNR, OSBF Fig.7,8 style)
- denoising comparison (SWF Fig.5 style)
- checkerboard edge preservation (OSBF Fig.11 style)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SRC = Path(r"E:/数字图像处理/src")
sys.path.insert(0, str(SRC))

from sac_swf import SACSWFConfig, sac_swf
from sac_swf.baselines import (
    bilateral_method,
    gaussian_method,
    guided_method,
    soft_swgf_method,
    hard_swgf_method,
)
from sac_swf.filters import box_filter, gaussian_filter
from sac_swf.side_window import DIRECTIONS, side_window_kernels

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "paper_figures_new"
OUT.mkdir(parents=True, exist_ok=True)
REAL_DIR = Path(r"E:/数字图像处理/data/raw/real")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "image.cmap": "gray",
})


def to_float01(arr):
    return np.clip(np.asarray(arr, dtype=np.float32) / 255.0, 0.0, 1.0)


def load_real(name):
    p = REAL_DIR / f"{name}.png"
    return to_float01(np.asarray(Image.open(p).convert("RGB")))


def save_fig(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [saved] {name}")


def make_checkerboard(size=129, n_sq=4):
    img = np.zeros((size, size, 3), dtype=np.float32)
    sq = size // n_sq
    for i in range(n_sq):
        for j in range(n_sq):
            if (i + j) % 2 == 0:
                img[i*sq:(i+1)*sq, j*sq:(j+1)*sq, :] = 1.0
    return img


# ============================================================
# Frequency response of side-window kernels (OSBF Fig.10 style)
# ============================================================

def fig_frequency_response():
    print("[freq] frequency response of side-window kernels")
    r = 7
    kernels = side_window_kernels(r, include_full=True)
    n = len(kernels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for ax, (name, ker) in zip(axes.flat, kernels.items()):
        # pad kernel to 64x64 for finer spectrum
        sz = 64
        padded = np.zeros((sz, sz), dtype=np.float64)
        kh, kw = ker.shape
        padded[:kh, :kw] = ker
        spec = np.fft.fftshift(np.abs(np.fft.fft2(padded)))
        spec = np.log(spec + 1)
        ax.imshow(spec, cmap="jet", extent=[-1, 1, -1, 1])
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    # hide unused
    for ax in axes.flat[n:]:
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "fig09_frequency_response.png")


# ============================================================
# Parameter analysis (OSBF Fig.7,8,11 style)
# ============================================================

def fig_parameter_analysis():
    print("[param] parameter analysis curves")
    img = load_real("skimage_camera")
    h, w = img.shape[:2]
    if max(h, w) > 256:
        scale = 256 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((nw, nh))) / 255.0

    # --- radius sweep with PSNR (input vs output, no noise) ---
    radii = list(range(1, 11))
    methods_psnr = {"BOX": [], "GF": [], "SWGF": [], "SAC-SWF": []}
    for r in radii:
        cfg = SACSWFConfig(radius=r, eps=1e-2)
        outs = {
            "BOX": box_filter(img, radius=r),
            "GF": guided_method(img, cfg)[0],
            "SWGF": soft_swgf_method(img, cfg)[0],
            "SAC-SWF": sac_swf(img, config=cfg)[0],
        }
        for m, o in outs.items():
            mse = np.mean((img - o) ** 2)
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            methods_psnr[m].append(psnr)

    # --- iteration sweep with checkerboard PSNR (OSBF Fig.11 style) ---
    cb = make_checkerboard(129, 4)
    iters = [1, 2, 5, 10, 20, 50]
    iter_psnr = {"BOX": [], "GF": [], "SWGF": [], "SAC-SWF": []}
    for it in iters:
        # iteratively apply each method
        outs = {}
        cur_box = cb.copy()
        cur_gf = cb.copy()
        cur_swgf = cb.copy()
        cur_sac = cb.copy()
        for _ in range(it):
            cfg = SACSWFConfig(radius=2, eps=1e-2)
            cur_box = box_filter(cur_box, radius=2)
            cur_gf = guided_method(cur_gf, cfg)[0]
            cur_swgf = soft_swgf_method(cur_swgf, cfg)[0]
            cur_sac = sac_swf(cur_sac, config=cfg)[0]
        for m, o in [("BOX", cur_box), ("GF", cur_gf), ("SWGF", cur_swgf), ("SAC-SWF", cur_sac)]:
            mse = np.mean((cb - o) ** 2)
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            iter_psnr[m].append(psnr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    for m, vals in methods_psnr.items():
        ax.plot(radii, vals, marker="o", lw=1.2, label=m)
    ax.set_xlabel("window radius r")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(a) PSNR vs radius (camera)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for m, vals in iter_psnr.items():
        ax.plot(iters, vals, marker="s", lw=1.2, label=m)
    ax.set_xlabel("iteration number")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(b) PSNR vs iteration (checkerboard, r=2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    plt.tight_layout()
    save_fig(fig, "fig10_parameter_analysis.png")


# ============================================================
# Denoising comparison (SWF Fig.5 style)
# ============================================================

def fig_denoising():
    print("[denoise] denoising comparison")
    img = load_real("skimage_astronaut")
    h, w = img.shape[:2]
    if max(h, w) > 256:
        scale = 256 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((nw, nh))) / 255.0

    rng = np.random.default_rng(42)
    sigma = 25.0 / 255.0
    noisy = np.clip(img + rng.normal(0, sigma, img.shape), 0, 1)

    cfg = SACSWFConfig(radius=5, eps=1e-2)
    methods = [
        ("Noisy", noisy),
        ("BOX", box_filter(noisy, radius=5)),
        ("GF", guided_method(noisy, cfg)[0]),
        ("BF", bilateral_method(noisy, cfg)[0]),
        ("SWGF", soft_swgf_method(noisy, cfg)[0]),
        ("SAC-SWF", sac_swf(noisy, config=cfg)[0]),
    ]
    zy, zx, zs = img.shape[0] // 4, img.shape[1] // 4, min(img.shape[:2]) // 3

    fig, axes = plt.subplots(2, len(methods), figsize=(15, 6),
                              gridspec_kw={"height_ratios": [3, 2]})
    from matplotlib.patches import Rectangle
    for c, (name, out) in enumerate(methods):
        # PSNR vs clean
        mse = np.mean((img - out) ** 2)
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        ax = axes[0, c]
        ax.imshow(np.clip(out, 0, 1))
        rect = Rectangle((zx, zy), zs, zs, fill=False, ec="red", lw=1.5)
        ax.add_patch(rect)
        ax.set_title(f"{name}\nPSNR={psnr:.2f}", fontsize=9)
        ax.axis("off")
        ax = axes[1, c]
        ax.imshow(np.clip(out[zy:zy + zs, zx:zx + zs], 0, 1))
        ax.set_title("zoom", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "fig11_denoising.png")


# ============================================================
# Checkerboard edge preservation (OSBF Fig.11 style)
# ============================================================

def fig_checkerboard():
    print("[checker] checkerboard edge preservation")
    cb = make_checkerboard(129, 4)
    cfg = SACSWFConfig(radius=5, eps=1e-2)
    methods = [
        ("Input", cb),
        ("BOX", box_filter(cb, radius=5)),
        ("GF", guided_method(cb, cfg)[0]),
        ("SWGF", soft_swgf_method(cb, cfg)[0]),
        ("SAC-SWF", sac_swf(cb, config=cfg)[0]),
    ]
    fig, axes = plt.subplots(2, len(methods), figsize=(13, 6),
                              gridspec_kw={"height_ratios": [3, 2]})
    for c, (name, out) in enumerate(methods):
        ax = axes[0, c]
        ax.imshow(np.clip(out, 0, 1))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        # profile line through middle row
        ax = axes[1, c]
        mid = cb.shape[0] // 2
        ax.plot(cb[mid, :, 0], "k-", lw=1.0, label="input")
        ax.plot(out[mid, :, 0], "r-", lw=0.9, label=name if name != "Input" else "input")
        ax.set_title("profile", fontsize=8)
        ax.tick_params(labelsize=7)
        if c == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    save_fig(fig, "fig12_checkerboard.png")


def main():
    print(f"Output dir: {OUT}")
    fig_frequency_response()
    fig_parameter_analysis()
    fig_denoising()
    fig_checkerboard()
    print("\n[done] additional figures generated")


if __name__ == "__main__":
    main()
