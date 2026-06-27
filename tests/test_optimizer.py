from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.optimizer import OptimizerConfig, optimize_orders  # noqa: E402


def base_frame(rows: list[dict] | None = None) -> pd.DataFrame:
    default = [
        {
            "draft_id": "DR001",
            "date": "2026-06-28",
            "sku_id": "SKU001",
            "agent_suggested_qty": 30,
            "order_draft_status": "drafted",
            "product_name": "Snack",
            "supplier_id": "SUP001",
            "unit_cost": 10.0,
            "pack_size": 6,
            "min_order_qty": 12,
            "max_order_qty": 60,
            "storage_volume": 0.1,
            "lead_time_days": 3,
            "expected_stock": 0,
            "closing_stock": 0,
            "forecast_quantity": 10,
            "decision_id": "DC001",
            "severity": 0.2,
            "harness_result": "approved",
            "final_status": "approved",
            "final_decision": "normal",
        }
    ]
    return pd.DataFrame(rows if rows is not None else default)


def test_pack_size_multiple_constraint():
    result, status = optimize_orders(base_frame(), config=OptimizerConfig(daily_budget=10_000, storage_capacity=10))
    qty = int(result.loc[0, "optimized_qty"])
    assert status == "Optimal"
    assert qty % 6 == 0


def test_min_order_quantity_constraint():
    result, _ = optimize_orders(base_frame(), config=OptimizerConfig(daily_budget=1_000, storage_capacity=10))
    qty = int(result.loc[0, "optimized_qty"])
    assert qty == 0 or qty >= 12


def test_max_order_quantity_constraint():
    result, _ = optimize_orders(base_frame(), config=OptimizerConfig(daily_budget=10_000, storage_capacity=10))
    assert int(result.loc[0, "optimized_qty"]) <= 60


def test_budget_constraint():
    result, _ = optimize_orders(base_frame(), config=OptimizerConfig(daily_budget=119, storage_capacity=10))
    assert int(result.loc[0, "optimized_qty"]) == 0


def test_storage_constraint():
    result, _ = optimize_orders(base_frame(), config=OptimizerConfig(daily_budget=10_000, storage_capacity=1.1))
    assert int(result.loc[0, "optimized_qty"]) == 0


def test_blocked_sku_excluded():
    frame = base_frame()
    frame.loc[0, "harness_result"] = "blocked"
    result, status = optimize_orders(frame, mode="simulation", config=OptimizerConfig(daily_budget=10_000, storage_capacity=10))
    assert status == "NoTargets"
    assert int(result.loc[0, "optimized_qty"]) == 0
    assert result.loc[0, "optimizer_status"] == "skipped_blocked"


def test_requires_manual_review_excluded_in_production():
    frame = base_frame()
    frame.loc[0, "harness_result"] = "requires_manual_review"
    result, status = optimize_orders(frame, mode="production", config=OptimizerConfig(daily_budget=10_000, storage_capacity=10))
    assert status == "NoTargets"
    assert int(result.loc[0, "optimized_qty"]) == 0
    assert result.loc[0, "optimizer_status"] == "skipped_requires_review"


def test_requires_manual_review_included_in_simulation():
    frame = base_frame()
    frame.loc[0, "harness_result"] = "requires_manual_review"
    result, status = optimize_orders(frame, mode="simulation", config=OptimizerConfig(daily_budget=10_000, storage_capacity=10))
    assert status == "Optimal"
    assert int(result.loc[0, "optimized_qty"]) > 0
    assert result.loc[0, "optimizer_status"] == "simulated_optimized"


def test_order_qty_is_non_negative():
    rows = [
        {
            **base_frame().iloc[0].to_dict(),
            "sku_id": "SKU001",
            "draft_id": "DR001",
            "harness_result": "approved",
        },
        {
            **base_frame().iloc[0].to_dict(),
            "sku_id": "SKU002",
            "draft_id": "DR002",
            "harness_result": "requires_manual_review",
            "unit_cost": 20.0,
        },
    ]
    result, _ = optimize_orders(base_frame(rows), mode="simulation", config=OptimizerConfig(daily_budget=10_000, storage_capacity=10))
    assert (result["optimized_qty"] >= 0).all()
