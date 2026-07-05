"""Run a minimal end-to-end smoke workflow."""

from __future__ import annotations

import subprocess
import sys


COMMANDS = [
    [sys.executable, "scripts/00_check_environment.py"],
    [sys.executable, "scripts/01_generate_synthetic.py", "--max-samples", "2"],
    [sys.executable, "scripts/09_prepare_real_images.py", "--limit", "2"],
    [sys.executable, "scripts/02_run_window_principle.py"],
    [sys.executable, "scripts/03_run_synthetic_experiment.py", "--limit", "1"],
    [sys.executable, "scripts/04_run_real_images.py", "--limit", "1"],
    [sys.executable, "scripts/05_run_ablation.py", "--limit", "1"],
    [sys.executable, "scripts/14_prepare_benchmark_data.py", "--real-limit", "1", "--synthetic-limit", "1", "--size", "128"],
    [sys.executable, "scripts/10_run_exp05_benchmark.py", "--limit", "1"],
    [sys.executable, "scripts/11_run_exp06_learning_comparison.py", "--train-limit", "1", "--test-limit", "1"],
    [sys.executable, "scripts/12_run_exp07_paper_reproduction.py", "--size", "128"],
    [sys.executable, "scripts/08_collect_tables.py"],
    [sys.executable, "scripts/06_make_report_figures.py"],
    [sys.executable, "scripts/07_validate_results.py", "--result", "results"],
]


def main() -> None:
    for cmd in COMMANDS:
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
