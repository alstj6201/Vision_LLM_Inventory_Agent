from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.demand_forecasting import run_demand_forecasting  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate demand forecasting models.")
    parser.add_argument("--input-csv", type=Path, default=PROJECT_ROOT / "data" / "products" / "merge_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "demand_forecasting")
    parser.add_argument("--sqlite-path", type=Path, default=PROJECT_ROOT / "synthetic_retail_company_dataset" / "retail_inventory.sqlite")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_demand_forecasting(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        sqlite_path=args.sqlite_path,
    )
    print(
        json.dumps(
            {
                "best_model": result.best_model,
                "validation_MAE": result.validation_metric,
                "test_metrics": result.test_metrics,
                "output_dir": str(result.output_dir),
                "figures": [str(path) for path in result.figure_paths],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
