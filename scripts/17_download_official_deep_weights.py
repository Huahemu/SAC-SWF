"""Download author-released deep-denoiser weights with resume and size checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import _bootstrap  # noqa: F401
from sac_swf.official_deep import PROJECT_ROOT, SPECS


def download_one(method: str, max_time: int) -> int:
    spec = SPECS[method]
    out = PROJECT_ROOT / spec.weight_path
    out.parent.mkdir(parents=True, exist_ok=True)
    before = out.stat().st_size if out.exists() else 0
    cmd = [
        "curl.exe",
        "-L",
        "-C",
        "-",
        "--retry",
        "3",
        "--connect-timeout",
        "20",
        "--max-time",
        str(max_time),
        "-o",
        str(out),
        spec.weight_url,
    ]
    print(f"\n>>> {method}")
    print(f"url: {spec.weight_url}")
    print(f"out: {out}")
    print(f"before_bytes: {before}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    after = out.stat().st_size if out.exists() else 0
    ready = after >= spec.expected_min_bytes
    print(f"returncode: {result.returncode}")
    print(f"after_bytes: {after}")
    print(f"expected_min_bytes: {spec.expected_min_bytes}")
    print(f"ready: {ready}")
    return 0 if ready else result.returncode or 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["all", *SPECS.keys()],
        default="all",
        help="Which official method weight to download.",
    )
    parser.add_argument("--max-time", type=int, default=600, help="Per-file curl max time in seconds.")
    args = parser.parse_args()

    methods = list(SPECS.keys()) if args.method == "all" else [args.method]
    failures = []
    for method in methods:
        code = download_one(method, args.max_time)
        if code != 0:
            failures.append(method)
    if failures:
        print("\nIncomplete downloads: " + ", ".join(failures))
        raise SystemExit(1)
    print("\nAll requested official weights are ready.")


if __name__ == "__main__":
    main()
