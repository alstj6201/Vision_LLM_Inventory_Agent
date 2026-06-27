from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.agents import run_cognition_layer  # noqa: E402
from retail_ai.llm_client import OpenAIJSONClient, RuleBasedLLMClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cognition layer agents for triage exceptions.")
    parser.add_argument("--sqlite-db", type=Path, default=PROJECT_ROOT / "synthetic_retail_company_dataset" / "retail_inventory.sqlite")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "agents")
    parser.add_argument("--date", default="latest")
    parser.add_argument("--provider", choices=["openai"], default="openai")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        client = RuleBasedLLMClient()
    else:
        client = OpenAIJSONClient(env_path=PROJECT_ROOT / ".env")
    summary = run_cognition_layer(
        sqlite_db=args.sqlite_db,
        output_dir=args.output_dir,
        date=args.date,
        llm_client=client,
    )
    print(
        json.dumps(
            {
                "exception_sku_count": summary.exception_sku_count,
                "agent_call_count": summary.agent_call_count,
                "order_draft_count": summary.order_draft_count,
                "blocked_count": summary.blocked_count,
                "requires_review_count": summary.requires_review_count,
                "output_files": [str(path) for path in summary.output_files],
                "provider": "dry-run" if args.dry_run else args.provider,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
