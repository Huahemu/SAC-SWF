"""Generate paper-style figures for the redesigned SAC-SWF experiments.

Follows SWF (CVPR2019) / Quarter Laplacian / OSBF experiment style:
- figure-dominant, tables only for runtime
- mechanism figures first (line profiles, candidate visualizations)
- application figures with zoomed patches
- no ML benchmark tables, no custom metrics, no deep-learning comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --- make sac_swf importable ---
SRC = Path(r"E:/数字图像处理/src")
sys.path.insert(0, str(SRC))

from sac_swf import SACSWFConfig, sac_swf
from sac_swf.baselines import (
    bilateral_method,
    fixed_mix_method,
    gaussian_method,
    guided_method,
    hard_swgf_method,
    soft_swgf_method,
)
from sac_swf.filters import box_filter, gaussian_filter
from sac_swf.side_window import (
    DIRECTIONS,
    full_window_guided_candidate,
    side_window_guided_candidates,
)

# --- paths ---
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "paper_figures_new"
OUT.mkdir(parents=True, exist_ok=True)
REAL_DIR = Path(r"E:/数字图像处理/data/raw/real")

# --- matplotlib style ---
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "image.cmap": "gray",
})


def to_float01(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float32) / 255.0, 0.0, 1.0)


def load_real(name: str) -> np.ndarray:
    p = REAL_DIR / f"{name}.png"
    return to_float01(np.asarray(Image.open(p).convert("RGB")))


def skimage_to_float(arr: np.ndarray) -> np.ndarray:
    """Convert skimage data (uint8 or float) to [0,1] RGB array."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return to_float01(arr)
    # assume float in [0,1] or other range
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  [saved] {name}")


def ssim_free_label(value: float) -> str:
    return f"SSIM={value:.3f}"


# ============================================================
# Experiment 1: typical edge mechanism figures
# ============================================================

def make_vertical_edge(size: int = 129, u: float = 0.0, v: float = 1.0) -> np.ndarray:
    img = np.full((size, size, 3), u, dtype=np.float32)
    img[:, size // 2:, :] = v
    return img


def make_horizontal_edge(size: int = 129, u: float = 0.0, v: float = 1.0) -> np.ndarray:
    img = np.full((size, size, 3), u, dtype=np.float32)
    img[size // 2:, :, :] = v
    return img


def make_diagonal_edge(size: int = 129, u: float = 0.0, v: float = 1.0) -> np.ndarray:
    img = np.full((size, size, 3), u, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    img[xx + yy > size, :] = v
    return img


def make_corner_edge(size: int = 129, u: float = 0.0, v: float = 1.0) -> np.ndarray:
    img = np.full((size, size, 3), u, dtype=np.float32)
    c = size // 2
    img[c:, c:, :] = v
    return img


def make_ramp_edge(size: int = 129, u: float = 0.0, v: float = 1.0) -> np.ndarray:
    img = np.full((size, size, 3), u, dtype=np.float32)
    c = size // 2
    ramp = np.linspace(u, v, size - c)
    img[:, c:, 0] = ramp[None, :]
    img[:, c:, 1] = ramp[None, :]
    img[:, c:, 2] = ramp[None, :]
    return img


def make_roof_edge(size: int = 129, u: float = 0.0, v: float = 1.0) -> np.ndarray:
    img = np.full((size, size, 3), u, dtype=np.float32)
    c = size // 2
    w = size // 6
    up = np.linspace(u, v, w)
    down = np.linspace(v, u, w)
    roof = np.concatenate([up, down])
    img[:, c - w:c + w, 0] = roof[None, :]
    img[:, c - w:c + w, 1] = roof[None, :]
    img[:, c - w:c + w, 2] = roof[None, :]
    return img


def exp1_mechanism() -> None:
    print("[exp1] typical edge mechanism figures")
    cfg = SACSWFConfig(radius=7, eps=1e-2)
    edges = [
        ("vertical", make_vertical_edge),
        ("horizontal", make_horizontal_edge),
        ("diagonal", make_diagonal_edge),
        ("corner", make_corner_edge),
        ("ramp", make_ramp_edge),
        ("roof", make_roof_edge),
    ]
    methods = [
        ("Input", None),
        ("BOX", lambda im: (box_filter(im, radius=cfg.radius), {})),
        ("GF", lambda im: guided_method(im, cfg)),
        ("SWGF", lambda im: soft_swgf_method(im, cfg)),
        ("SAC-SWF", lambda im: sac_swf(im, config=cfg)),
    ]
    def get_profile(im: np.ndarray, ename: str) -> np.ndarray:
        s = im.shape[0]
        c = s // 2
        if ename == "vertical":
            return im[c, :, 0]
        if ename == "horizontal":
            return im[:, c, 0]
        if ename == "diagonal":
            yy, xx = np.mgrid[0:s, 0:s]
            mask = (xx == yy)
            return im[:, :, 0][mask]
        if ename == "corner":
            return im[c, :, 0]
        if ename == "ramp":
            return im[c, :, 0]
        if ename == "roof":
            return im[c, :, 0]
        return im[c, :, 0]

    # Fig.1: 6 edges x 5 methods grid + line profiles below each edge row
    n_edges = len(edges)
    n_methods = len(methods)
    fig, axes = plt.subplots(n_edges, n_methods + 1, figsize=(13, 14))
    for r, (ename, make_fn) in enumerate(edges):
        img = make_fn()
        row = axes[r]
        # input
        ax = row[0]
        ax.imshow(img)
        ax.set_title(ename if r == 0 else ename, fontsize=9)
        ax.axis("off")
        # methods
        for c, (mname, mfn) in enumerate(methods[1:], start=1):
            ax = row[c]
            if mfn is None:
                out = img
            else:
                res = mfn(img)
                out = res[0] if isinstance(res, tuple) else res
            ax.imshow(np.clip(out, 0, 1))
            if r == 0:
                ax.set_title(mname, fontsize=9)
            ax.axis("off")
        # line profile column (last)
        ax = row[-1]
        ax.plot(get_profile(img, ename), "k-", lw=1.0, label="input")
        for c, (mname, mfn) in enumerate(methods[1:], start=0):
            if mfn is None:
                continue
            res = mfn(img)
            out = res[0] if isinstance(res, tuple) else res
            ax.plot(get_profile(out, ename), lw=0.9, label=mname)
        ax.set_title("profile" if r == 0 else "", fontsize=9)
        ax.tick_params(labelsize=7)
        if r == 0:
            ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    save_fig(fig, "fig01_edge_mechanism.png")


# ============================================================
# Experiment 2: candidate outputs + intermediate weights
# ============================================================

def make_structure_texture(size: int = 128) -> np.ndarray:
    """Synthetic image: step edges + sinusoidal texture, ideal for candidate visualization."""
    img = np.full((size, size, 3), 0.5, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    img[:, size // 2:, :] = 0.85
    img[size // 2:, :, :] *= 0.6
    tex = 0.15 * np.sin(xx / 4.0) * np.cos(yy / 3.5)
    img[..., 0] += tex
    img[..., 1] += tex
    img[..., 2] += tex
    return np.clip(img, 0.0, 1.0)


def exp2_candidates() -> None:
    print("[exp2] candidate outputs and intermediate weights")
    cfg = SACSWFConfig(radius=7, eps=1e-2)
    img = make_structure_texture(128)

    out, inter = sac_swf(img, config=cfg)
    q_full = inter["q_full"]
    q_side = inter["q_side"]
    candidates = inter["candidates"]
    beta = inter["beta"]
    direction_index = inter["direction_index"]
    structure_conf = inter["structure_conf"]
    coherence = inter["coherence"]
    oscillation = inter["oscillation"]

    # Fig.2: 3x4 grid: input + full + 8 side + side-agg + output
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    items = [
        ("Input", img),
        ("Full-window", q_full),
    ] + [(f"Side-{d}", candidates[d]) for d in DIRECTIONS] + [
        ("Side-agg", q_side),
        ("SAC-SWF", out),
    ]
    for ax, (title, im) in zip(axes.flat, items):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "fig02_candidates_grid.png")

    # Fig.3: intermediate weights
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    panels = [
        ("Input", img, None),
        ("Structure confidence R", structure_conf, "viridis"),
        ("Coherence C", coherence, "magma"),
        ("Oscillation O", oscillation, "inferno"),
        ("Full-window weight β", beta, "RdBu_r"),
        ("Direction index", direction_index, "tab20"),
    ]
    for ax, (title, im, cmap) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(np.clip(im, 0, 1))
        else:
            im_plot = ax.imshow(im, cmap=cmap)
            plt.colorbar(im_plot, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "fig03_intermediate_weights.png")


# ============================================================
# Experiment 3: image smoothing (SWF Fig.4 style)
# ============================================================

def exp3_smoothing() -> None:
    print("[exp3] image smoothing application")
    cfg = SACSWFConfig(radius=5, eps=1e-2)
    for img_name in ["skimage_astronaut", "skimage_camera"]:
        img = load_real(img_name)
        # downsize for speed
        h, w = img.shape[:2]
        if max(h, w) > 320:
            scale = 320 / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((nw, nh))) / 255.0

        methods = [
            ("Input", img),
            ("BOX", gaussian_method(img, cfg)[0]),
            ("BF", bilateral_method(img, cfg)[0]),
            ("GF", guided_method(img, cfg)[0]),
            ("SWGF", soft_swgf_method(img, cfg)[0]),
            ("SAC-SWF", sac_swf(img, config=cfg)[0]),
        ]
        # choose a zoom region
        zh, zw = img.shape[:2]
        zy, zx = zh // 4, zw // 4
        zs = min(zh, zw) // 3

        fig, axes = plt.subplots(2, len(methods), figsize=(14, 6),
                                  gridspec_kw={"height_ratios": [3, 2]})
        for c, (name, out) in enumerate(methods):
            # full
            ax = axes[0, c]
            ax.imshow(np.clip(out, 0, 1))
            # draw zoom rectangle
            from matplotlib.patches import Rectangle
            rect = Rectangle((zx, zy), zs, zs, fill=False, ec="red", lw=1.5)
            ax.add_patch(rect)
            ax.set_title(name, fontsize=10)
            ax.axis("off")
            # zoom
            ax = axes[1, c]
            ax.imshow(np.clip(out[zy:zy + zs, zx:zx + zs], 0, 1))
            ax.set_title(f"zoom", fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        save_fig(fig, f"fig04_smoothing_{img_name}.png")


# ============================================================
# Experiment 4: structure-texture separation
# ============================================================

def exp4_texture_separation() -> None:
    print("[exp4] structure-texture separation")
    cfg = SACSWFConfig(radius=5, eps=1e-2)
    img = load_real("skimage_coins")
    h, w = img.shape[:2]
    if max(h, w) > 320:
        scale = 320 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((nw, nh))) / 255.0

    methods = [
        ("Input", img),
        ("GF", guided_method(img, cfg)[0]),
        ("SWGF", soft_swgf_method(img, cfg)[0]),
        ("SAC-SWF", sac_swf(img, config=cfg)[0]),
    ]
    zy, zx, zs = img.shape[0] // 4, img.shape[1] // 4, min(img.shape[:2]) // 3

    fig, axes = plt.subplots(3, len(methods), figsize=(13, 9),
                              gridspec_kw={"height_ratios": [3, 2, 2]})
    for c, (name, out) in enumerate(methods):
        # structure
        ax = axes[0, c]
        ax.imshow(np.clip(out, 0, 1))
        from matplotlib.patches import Rectangle
        rect = Rectangle((zx, zy), zs, zs, fill=False, ec="red", lw=1.5)
        ax.add_patch(rect)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        # zoom structure
        ax = axes[1, c]
        ax.imshow(np.clip(out[zy:zy + zs, zx:zx + zs], 0, 1))
        ax.set_title(f"zoom", fontsize=8)
        ax.axis("off")
        # texture residual
        ax = axes[2, c]
        if name == "Input":
            ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=12)
            ax.set_title("", fontsize=8)
            ax.axis("off")
        else:
            residual = img - out
            ax.imshow(np.clip((residual - residual.min()) / max(residual.max() - residual.min(), 1e-6), 0, 1))
            ax.set_title(f"texture (I-S)", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "fig05_texture_separation.png")


# ============================================================
# Experiment 5: image enhancement
# ============================================================

def exp5_enhancement() -> None:
    print("[exp5] image enhancement")
    cfg = SACSWFConfig(radius=5, eps=1e-2)
    img = load_real("skimage_coffee")
    h, w = img.shape[:2]
    if max(h, w) > 320:
        scale = 320 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((nw, nh))) / 255.0

    alpha_enh = 1.5  # enhancement factor
    methods = [
        ("Input", img),
        ("GF", guided_method(img, cfg)[0]),
        ("BF", bilateral_method(img, cfg)[0]),
        ("SWGF", soft_swgf_method(img, cfg)[0]),
        ("SAC-SWF", sac_swf(img, config=cfg)[0]),
    ]
    zy, zx, zs = img.shape[0] // 4, img.shape[1] // 4, min(img.shape[:2]) // 3

    fig, axes = plt.subplots(2, len(methods), figsize=(14, 6),
                              gridspec_kw={"height_ratios": [3, 2]})
    for c, (name, out) in enumerate(methods):
        enhanced = np.clip(out + alpha_enh * (out - img), 0, 1) if name != "Input" else img
        ax = axes[0, c]
        ax.imshow(enhanced)
        from matplotlib.patches import Rectangle
        rect = Rectangle((zx, zy), zs, zs, fill=False, ec="red", lw=1.5)
        ax.add_patch(rect)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        ax = axes[1, c]
        ax.imshow(enhanced[zy:zy + zs, zx:zx + zs])
        ax.set_title("zoom", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "fig06_enhancement.png")


# ============================================================
# Experiment 6: ablation mechanism figures
# ============================================================

def exp6_ablation() -> None:
    print("[exp6] ablation mechanism figures")
    # use a synthetic structure+texture image so candidate / beta differences are visible
    img = make_structure_texture(128)

    # full SAC-SWF (R and O both active)
    cfg_full = SACSWFConfig(radius=7, eps=1e-2, gamma=0.5)
    out_full, inter_full = sac_swf(img, config=cfg_full)

    # w/o texture term: gamma=0 -> O_i contribution zero, beta = clip(1-R, 0, 1)
    cfg_no_tex = SACSWFConfig(radius=7, eps=1e-2, gamma=0.0)
    out_no_tex, inter_no_tex = sac_swf(img, config=cfg_no_tex)

    # w/o structure term: force R=0 by setting rho very small so structure tensor degenerates
    # equivalently, set strength to zero via large kappa; here we approximate by disabling R
    # we modify beta computation by using a config that makes structure_conf ~ 0
    cfg_no_str = SACSWFConfig(radius=7, eps=1e-2, gamma=0.5, rho=0.1, pre_sigma=0.1)
    out_no_str, inter_no_str = sac_swf(img, config=cfg_no_str)

    # hard side selection (tau -> 0 effectively)
    cfg_hard = SACSWFConfig(radius=7, eps=1e-2, gamma=0.5, hard_side=True)
    out_hard, inter_hard = sac_swf(img, config=cfg_hard)

    panels = [
        ("Input", img, None),
        ("Full SAC-SWF", out_full, inter_full["beta"]),
        ("w/o texture (γ=0)", out_no_tex, inter_no_tex["beta"]),
        ("w/o structure (ρ→0)", out_no_str, inter_no_str["beta"]),
        ("Hard side", out_hard, inter_hard["beta"]),
    ]
    fig, axes = plt.subplots(2, len(panels), figsize=(14, 6),
                              gridspec_kw={"height_ratios": [3, 2]})
    for c, (name, out, beta) in enumerate(panels):
        ax = axes[0, c]
        ax.imshow(np.clip(out, 0, 1))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        ax = axes[1, c]
        if beta is not None:
            im = ax.imshow(beta, cmap="RdBu_r", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("β map", fontsize=8)
        else:
            ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("β map", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "fig07_ablation.png")


# ============================================================
# Experiment 7: runtime table (as PNG)
# ============================================================

def exp7_runtime() -> None:
    print("[exp7] runtime table")
    import time
    img = load_real("skimage_astronaut")
    h, w = img.shape[:2]
    if max(h, w) > 256:
        scale = 256 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((nw, nh))) / 255.0
    cfg = SACSWFConfig(radius=5, eps=1e-2)
    methods = [
        ("Gaussian", lambda: gaussian_method(img, cfg)),
        ("Bilateral", lambda: bilateral_method(img, cfg)),
        ("Guided", lambda: guided_method(img, cfg)),
        ("SWGF (soft)", lambda: soft_swgf_method(img, cfg)),
        ("SWGF (hard)", lambda: hard_swgf_method(img, cfg)),
        ("SAC-SWF", lambda: sac_swf(img, config=cfg)),
    ]
    rows = []
    for name, fn in methods:
        # warm up
        fn()
        t0 = time.perf_counter()
        for _ in range(3):
            fn()
        dt = (time.perf_counter() - t0) / 3
        rows.append((name, dt))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=[[n, f"{t:.4f}"] for n, t in rows],
        colLabels=["Method", "Runtime (s)"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.5)
    plt.tight_layout()
    save_fig(fig, "fig08_runtime_table.png")

    # also save as csv
    import csv
    with open(OUT / "runtime_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "runtime_sec"])
        for n, t in rows:
            w.writerow([n, f"{t:.6f}"])
    print("  [saved] runtime_table.csv")


# ============================================================
# Appendix figures (more application cases + parameter analysis)
# ============================================================

def appendix_figures() -> None:
    print("[appendix] additional application cases and parameter analysis")

    # --- Appendix A: multi-image OSBF-style smoothing comparison grid ---
    from skimage import data as skdata
    from matplotlib.patches import Rectangle

    cfg = SACSWFConfig(radius=5, eps=1e-2)
    images = [
        ("chelsea", skimage_to_float(skdata.chelsea())),
        ("cat", skimage_to_float(skdata.cat())),
        ("eagle", skimage_to_float(skdata.eagle())),
        ("grass", skimage_to_float(skdata.grass())),
        ("lily", skimage_to_float(skdata.lily())),
        ("rocket", skimage_to_float(skdata.rocket())),
    ]
    methods = ["Input", "BOX", "GF", "BF", "SWGF", "SAC-SWF"]
    n_imgs = len(images)
    n_meth = len(methods)
    fig, axes = plt.subplots(2 * n_imgs, n_meth, figsize=(2.2 * n_meth, 1.6 * n_imgs * 2),
                              gridspec_kw={"hspace": 0.02, "wspace": 0.02})
    for r, (iname, img) in enumerate(images):
        # resize to common max size for compact display
        h, w = img.shape[:2]
        if max(h, w) > 256:
            scale = 256 / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((nw, nh))) / 255.0
        # choose a zoom region
        zy, zx, zs = img.shape[0] // 3, img.shape[1] // 3, min(img.shape[:2]) // 4
        outs = {
            "Input": img,
            "BOX": box_filter(img, radius=cfg.radius),
            "GF": guided_method(img, cfg)[0],
            "BF": bilateral_method(img, cfg)[0],
            "SWGF": soft_swgf_method(img, cfg)[0],
            "SAC-SWF": sac_swf(img, config=cfg)[0],
        }
        for c, mname in enumerate(methods):
            out = outs[mname]
            ax_full = axes[2 * r, c]
            ax_full.imshow(np.clip(out, 0, 1))
            if c == 0:
                ax_full.set_ylabel(iname, fontsize=9, rotation=0, labelpad=20, va="center")
            if r == 0:
                ax_full.set_title(mname, fontsize=10)
            ax_full.axis("off")
            rect = Rectangle((zx, zy), zs, zs, fill=False, ec="red", lw=1.0)
            ax_full.add_patch(rect)
            ax_zoom = axes[2 * r + 1, c]
            ax_zoom.imshow(np.clip(out[zy:zy + zs, zx:zx + zs], 0, 1))
            ax_zoom.axis("off")
    plt.tight_layout(pad=0.2)
    save_fig(fig, "appA_multi_image_smoothing.png")

    # --- Appendix B: parameter analysis - radius and gamma effect ---
    img = load_real("skimage_camera")
    h, w = img.shape[:2]
    if max(h, w) > 256:
        scale = 256 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((nw, nh))) / 255.0

    # radius sweep
    radii = [3, 5, 7, 9]
    fig, axes = plt.subplots(2, len(radii) + 1, figsize=(14, 6),
                              gridspec_kw={"height_ratios": [3, 2]})
    axes[0, 0].imshow(img); axes[0, 0].set_title("Input", fontsize=10); axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    zy, zx, zs = img.shape[0] // 4, img.shape[1] // 4, min(img.shape[:2]) // 3
    for c, r in enumerate(radii, start=1):
        cfg_r = SACSWFConfig(radius=r, eps=1e-2)
        out = sac_swf(img, config=cfg_r)[0]
        ax = axes[0, c]
        ax.imshow(np.clip(out, 0, 1))
        rect = Rectangle((zx, zy), zs, zs, fill=False, ec="red", lw=1.5)
        ax.add_patch(rect)
        ax.set_title(f"r={r}", fontsize=10)
        ax.axis("off")
        ax = axes[1, c]
        ax.imshow(np.clip(out[zy:zy + zs, zx:zx + zs], 0, 1))
        ax.set_title("zoom", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "appB_radius_sweep.png")

    # gamma sweep
    gammas = [0.0, 0.25, 0.5, 1.0]
    fig, axes = plt.subplots(2, len(gammas) + 1, figsize=(14, 6),
                              gridspec_kw={"height_ratios": [3, 2]})
    axes[0, 0].imshow(img); axes[0, 0].set_title("Input", fontsize=10); axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    for c, g in enumerate(gammas, start=1):
        cfg_g = SACSWFConfig(radius=5, eps=1e-2, gamma=g)
        out, inter = sac_swf(img, config=cfg_g)
        ax = axes[0, c]
        ax.imshow(np.clip(out, 0, 1))
        rect = Rectangle((zx, zy), zs, zs, fill=False, ec="red", lw=1.5)
        ax.add_patch(rect)
        ax.set_title(f"γ={g}", fontsize=10)
        ax.axis("off")
        ax = axes[1, c]
        im = ax.imshow(inter["beta"], cmap="RdBu_r", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("β map", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "appB_gamma_sweep.png")

    # --- Appendix C: texture separation on a textured natural image ---
    img = load_real("skimage_astronaut")
    h, w = img.shape[:2]
    if max(h, w) > 320:
        scale = 320 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((nw, nh))) / 255.0
    methods = [
        ("Input", img),
        ("GF", guided_method(img, cfg)[0]),
        ("SWGF", soft_swgf_method(img, cfg)[0]),
        ("SAC-SWF", sac_swf(img, config=cfg)[0]),
    ]
    zy, zx, zs = img.shape[0] // 3, img.shape[1] // 3, min(img.shape[:2]) // 4
    fig, axes = plt.subplots(2, len(methods), figsize=(13, 7),
                              gridspec_kw={"height_ratios": [3, 2]})
    for c, (name, out) in enumerate(methods):
        ax = axes[0, c]
        ax.imshow(np.clip(out, 0, 1))
        rect = Rectangle((zx, zy), zs, zs, fill=False, ec="red", lw=1.5)
        ax.add_patch(rect)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        ax = axes[1, c]
        residual = img - out if name != "Input" else np.zeros_like(img)
        if name == "Input":
            ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=12)
            ax.axis("off")
        else:
            ax.imshow(np.clip((residual - residual.min()) / max(residual.max() - residual.min(), 1e-6), 0, 1))
            ax.set_title("texture (I-S)", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    save_fig(fig, "appC_texture_astronaut.png")


def main() -> None:
    print(f"Output dir: {OUT}")
    exp1_mechanism()
    exp2_candidates()
    exp3_smoothing()
    exp4_texture_separation()
    exp5_enhancement()
    exp6_ablation()
    exp7_runtime()
    appendix_figures()
    print("\n[done] all figures generated")


if __name__ == "__main__":
    main()
