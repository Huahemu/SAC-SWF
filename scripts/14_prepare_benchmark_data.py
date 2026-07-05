"""Prepare the deterministic benchmark suite used by Experiment 5."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from sac_swf.benchmark import prepare_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/benchmark")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--real-limit", type=int, default=6)
    parser.add_argument("--synthetic-limit", type=int, default=8)
    args = parser.parse_args()

    df = prepare_benchmark_suite(
        out_root=args.out,
        size=args.size,
        real_limit=args.real_limit,
        synthetic_limit=args.synthetic_limit,
    )
    print(f"Prepared {len(df)} benchmark samples under {args.out}")


if __name__ == "__main__":
    main()
