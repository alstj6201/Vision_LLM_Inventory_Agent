from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.severity import run_triage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run severity scoring and triage routing.")
    parser.add_argument("--sqlite-db", type=Path, default=PROJECT_ROOT / "synthetic_retail_company_dataset" / "retail_inventory.sqlite")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "triage")
    parser.add_argument("--date", default="latest", help="latest, all, or YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_triage(sqlite_db=args.sqlite_db, output_dir=args.output_dir, date=args.date)
    print(
        json.dumps(
            {
                "processed_sku_count": summary.processed_sku_count,
                "normal_count": summary.normal_count,
                "requires_review_count": summary.requires_review_count,
                "freeze_and_alert_count": summary.freeze_and_alert_count,
                "theft_suspected_count": summary.theft_suspected_count,
                "average_severity": summary.average_severity,
                "figures": [str(path) for path in summary.figure_paths],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
