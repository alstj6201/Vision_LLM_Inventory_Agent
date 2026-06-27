from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.harness import run_harness  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic harness validation for agent order drafts.")
    parser.add_argument("--sqlite-db", type=Path, default=PROJECT_ROOT / "synthetic_retail_company_dataset" / "retail_inventory.sqlite")
    parser.add_argument("--date", default="latest", help="latest, all, or YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "harness")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate and write output files without updating SQLite tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_harness(
        sqlite_db=args.sqlite_db,
        output_dir=args.output_dir,
        date=args.date,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "processed_drafts": summary.processed_drafts,
                "approved": summary.approved,
                "blocked": summary.blocked,
                "requires_manual_review": summary.requires_manual_review,
                "average_retry_count": summary.average_retry_count,
                "failure_reason_counts": summary.failure_reason_counts,
                "output_files": [str(path) for path in summary.output_files],
                "figures": [str(path) for path in summary.figure_paths],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
