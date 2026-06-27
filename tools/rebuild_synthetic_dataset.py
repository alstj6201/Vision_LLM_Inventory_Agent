from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from synthetic_data import rebuild_synthetic_dataset  # noqa: E402


def main() -> None:
    summary = rebuild_synthetic_dataset()
    payload = {
        "used_sku_count": summary.used_sku_count,
        "excluded_products_sku_count": summary.excluded_products_sku_count,
        "excluded_merge_sku_count": summary.excluded_merge_sku_count,
        "date_range": [summary.date_start, summary.date_end],
        "transaction_count": summary.transaction_count,
        "inventory_snapshot_count": summary.inventory_snapshot_count,
        "order_count": summary.order_count,
        "sqlite_row_counts": summary.sqlite_row_counts,
        "anomaly_counts": summary.anomaly_counts,
        "foreign_key_check": "PASS" if summary.foreign_key_ok else "FAIL",
        "integrity_check": "PASS" if summary.integrity_ok else "FAIL",
        "output_dir": str(summary.output_dir),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not summary.foreign_key_ok or not summary.integrity_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
