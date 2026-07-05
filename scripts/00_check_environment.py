"""Check Python environment and create required folders."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED = ["numpy", "scipy", "skimage", "cv2", "matplotlib", "pandas", "yaml", "tqdm"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(f"Project root: {root}")
    print(f"Python: {sys.version}")
    missing = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            print(f"MISSING {name}")
            missing.append(name)
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"OK {name}: {version}")
    for rel in [
        "data/raw/real",
        "data/raw/eps",
        "data/synthetic",
        "results",
        "logs",
        "reports/figures",
    ]:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        print(f"DIR {path}")
    if missing:
        print("\nEnvironment check failed. Install dependencies first:")
        print("  pip install -r requirements.txt")
        print("Missing packages: " + ", ".join(missing))
        raise SystemExit(1)
    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
