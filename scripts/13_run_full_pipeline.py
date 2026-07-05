"""Run the official experiment sequence and save one log per step."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-count", type=int, default=12)
    parser.add_argument("--eval-limit", type=int, default=6)
    parser.add_argument("--real-limit", type=int, default=4)
    parser.add_argument("--repro-size", type=int, default=192)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "logs" / f"full_pipeline_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        ("00_environment", [sys.executable, "scripts/00_check_environment.py"]),
        ("01_generate_synthetic", [sys.executable, "scripts/01_generate_synthetic.py", "--max-samples", str(args.synthetic_count)]),
        ("prepare_real_images", [sys.executable, "scripts/09_prepare_real_images.py", "--limit", str(args.real_limit)]),
        ("exp01_window_principle", [sys.executable, "scripts/02_run_window_principle.py"]),
        ("exp02_synthetic", [sys.executable, "scripts/03_run_synthetic_experiment.py", "--limit", str(args.eval_limit)]),
        ("exp03_real_images", [sys.executable, "scripts/04_run_real_images.py", "--limit", str(args.real_limit)]),
        ("exp04_ablation", [sys.executable, "scripts/05_run_ablation.py", "--limit", str(args.eval_limit)]),
        ("prepare_benchmark", [sys.executable, "scripts/14_prepare_benchmark_data.py", "--real-limit", str(args.real_limit), "--synthetic-limit", str(args.eval_limit)]),
        ("exp05_benchmark", [sys.executable, "scripts/10_run_exp05_benchmark.py", "--limit", str(args.eval_limit)]),
        ("exp06_learning_comparison", [sys.executable, "scripts/11_run_exp06_learning_comparison.py", "--train-limit", str(args.eval_limit), "--test-limit", str(args.eval_limit)]),
        ("exp07_paper_reproduction", [sys.executable, "scripts/12_run_exp07_paper_reproduction.py", "--size", str(args.repro_size)]),
        ("collect_tables", [sys.executable, "scripts/08_collect_tables.py"]),
        ("make_report_figures", [sys.executable, "scripts/06_make_report_figures.py"]),
        ("validate_results", [sys.executable, "scripts/07_validate_results.py", "--result", "results"]),
    ]

    summary_rows = []
    for idx, (name, cmd) in enumerate(commands, start=1):
        log_path = log_dir / f"{idx:02d}_{name}.log"
        print(f"\n>>> {name}")
        print(" ".join(cmd))
        with log_path.open("w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n\n")
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        summary_rows.append(f"{idx:02d},{name},{result.returncode},{log_path}")
        if result.returncode != 0:
            (log_dir / "pipeline_summary.csv").write_text(
                "step,name,returncode,log\n" + "\n".join(summary_rows),
                encoding="utf-8",
            )
            raise SystemExit(f"Step failed: {name}. See {log_path}")

    (log_dir / "pipeline_summary.csv").write_text(
        "step,name,returncode,log\n" + "\n".join(summary_rows),
        encoding="utf-8",
    )
    print(f"\nFull pipeline completed. Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
