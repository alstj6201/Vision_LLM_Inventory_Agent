from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.demo_runner import run_end_to_end_demo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retail AI end-to-end demo orchestration.")
    parser.add_argument("--date", default="2026-05-21")
    parser.add_argument("--mode", choices=["simulation", "production"], default="simulation")
    parser.add_argument("--morning-image", type=Path, default=PROJECT_ROOT / "data" / "simulation" / "1.png")
    parser.add_argument("--evening-image", type=Path, default=PROJECT_ROOT / "data" / "simulation" / "2.png")
    parser.add_argument("--sqlite-db", type=Path, default=PROJECT_ROOT / "synthetic_retail_company_dataset" / "retail_inventory.sqlite")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "demo")
    parser.add_argument("--agent-dry-run", action="store_true", help="Force rule-based agent responses without calling OpenAI.")
    parser.add_argument("--detector", choices=["auto", "yolo", "owlvit"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_end_to_end_demo(
        date=args.date,
        mode=args.mode,
        morning_image=args.morning_image,
        evening_image=args.evening_image,
        sqlite_db=args.sqlite_db,
        output_dir=args.output_dir,
        force_agent_dry_run=args.agent_dry_run,
        detector=args.detector,
    )
    print("=" * 38)
    print("END TO END DEMO")
    print("=" * 38)
    print(f"Simulation Date: {summary.date}")
    print(f"Processed SKU: {summary.processed_sku}")
    print(f"Morning Count: {summary.morning_count}")
    print(f"Evening Count: {summary.evening_count}")
    print(f"POS Sales: {summary.pos_sales}")
    print(f"Total Shrinkage: {summary.total_shrinkage:.2f}")
    print(f"Alert Count: {summary.alert_count}")
    print(f"Review Count: {summary.review_count}")
    print(f"Freeze Count: {summary.freeze_count}")
    print(f"Optimized Orders: {summary.optimized_orders}")
    print(f"Execution Time: {summary.execution_time:.2f}s")
    print(f"LLM Dry-run Fallback: {summary.llm_dry_run_fallback}")
    print("Generated Files:")
    for path in summary.generated_files:
        print(f"- {path}")
    print("=" * 38)


if __name__ == "__main__":
    main()
