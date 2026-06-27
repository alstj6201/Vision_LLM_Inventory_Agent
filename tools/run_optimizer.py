from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.optimizer import run_optimizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PuLP MILP optimizer for safe retail replenishment.")
    parser.add_argument("--sqlite-db", type=Path, default=PROJECT_ROOT / "synthetic_retail_company_dataset" / "retail_inventory.sqlite")
    parser.add_argument("--date", default="latest", help="latest, all, or YYYY-MM-DD")
    parser.add_argument("--mode", choices=["production", "simulation"], default="simulation")
    parser.add_argument("--daily-budget", type=float, default=500_000.0)
    parser.add_argument("--storage-capacity", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "optimizer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_optimizer(
        sqlite_db=args.sqlite_db,
        output_dir=args.output_dir,
        date=args.date,
        mode=args.mode,
        daily_budget=args.daily_budget,
        storage_capacity=args.storage_capacity,
    )
    print(
        json.dumps(
            {
                "mode": summary.mode,
                "solver_status": summary.solver_status,
                "target_sku_count": summary.target_sku_count,
                "optimized_sku_count": summary.optimized_sku_count,
                "total_optimized_quantity": summary.total_optimized_quantity,
                "total_cost": summary.total_cost,
                "budget_usage": summary.budget_usage,
                "storage_usage": summary.storage_usage,
                "output_files": [str(path) for path in summary.output_files],
                "figures": [str(path) for path in summary.figure_paths],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
