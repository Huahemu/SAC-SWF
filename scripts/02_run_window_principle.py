"""Run minimal window-principle demo on controlled synthetic edges."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from sac_swf.experiment import process_one_sample
from sac_swf.sac_swf import SACSWFConfig
from sac_swf.synthetic import make_structure
from sac_swf.utils import ensure_dir


def main() -> None:
    out = ensure_dir("results/exp01_window_principle")
    cfg = SACSWFConfig(radius=7, eps=1e-3, gamma=0.5, lambda_grad=0.0)
    for kind in ["step", "corner", "lines", "mixed"]:
        s = make_structure(256, kind=kind)
        # Slight texture/noise makes the side-window choice visible but controlled.
        yy, xx = np.mgrid[0:256, 0:256]
        texture = 0.08 * np.sin(2 * np.pi * xx / 16.0)
        image = np.clip(s + texture, 0.0, 1.0)
        process_one_sample(image, Path(out), cfg, target=s, sample_id=f"principle_{kind}")
    print(f"Saved principle experiment to {out}")


if __name__ == "__main__":
    main()
