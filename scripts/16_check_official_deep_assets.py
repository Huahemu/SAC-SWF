"""Check author-released code and weights for Experiment 6 deep-network reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from sac_swf.official_deep import inspect_official_deep_specs
from sac_swf.utils import ensure_dir


def main() -> None:
    out = ensure_dir("results/exp06_learning_comparison")
    specs = inspect_official_deep_specs()
    (out / "official_deep_status.json").write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for item in specs:
        rows.append(
            {
                "method": item["method"],
                "display_name": item["display_name"],
                "author_repo": item["author_repo"],
                "execution_repo": item["execution_repo"],
                "code_ready": item["code_ready"],
                "weight_ready": item["weight_ready"],
                "ready_for_inference": item["ready_for_inference"],
                "weight_path": item["weight_path"],
                "weight_url": item["weight_url"],
                "weight_actual_bytes": item["weight_actual_bytes"],
                "expected_min_bytes": item["expected_min_bytes"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "official_deep_status.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"Saved official deep status to {Path(out).resolve()}")


if __name__ == "__main__":
    main()
