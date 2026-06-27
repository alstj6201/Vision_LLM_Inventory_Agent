from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.harness import HarnessConfig, evaluate_draft, run_harness  # noqa: E402


def valid_row(**overrides):
    row = {
        "draft_id": "DR001",
        "date": "2026-06-28",
        "sku_id": "SKU001",
        "suggested_qty": 20,
        "order_draft_status": "drafted",
        "product_name": "Snack",
        "supplier_id": "SUP001",
        "valid_supplier_id": "SUP001",
        "unit_cost": 1000.0,
        "pack_size": 10,
        "min_order_qty": 10,
        "max_order_qty": 100,
        "storage_volume": 1.0,
        "actual_sales": 8,
        "forecast_quantity": 10,
        "inventory_expected_stock": 18,
        "cv_expected_stock": 18,
        "cv_count": 16,
        "count_confidence": 0.9,
        "demand_anomaly_score": 0.2,
        "shrinkage_score": 0.1,
        "severity": 0.2,
        "triage_decision": "normal",
        "existing_retry_count": 0,
    }
    row.update(overrides)
    return row


def test_invalid_sku_is_blocked():
    decision = evaluate_draft(valid_row(product_name=pd.NA), HarnessConfig())
    assert decision.final_status == "blocked"
    assert decision.semantic_status == "semantic_failed"
    assert "SKU does not exist in sku_master." in decision.failure_reasons


def test_negative_quantity_is_blocked():
    decision = evaluate_draft(valid_row(suggested_qty=-1), HarnessConfig())
    assert decision.final_status == "blocked"
    assert "Suggested quantity is negative." in decision.failure_reasons


def test_over_budget_is_blocked():
    config = HarnessConfig(daily_budget=1000)
    decision = evaluate_draft(valid_row(suggested_qty=20, unit_cost=1000), config)
    assert decision.final_status == "blocked"
    assert "Budget exceeded." in decision.failure_reasons


def test_over_max_order_is_blocked():
    decision = evaluate_draft(valid_row(suggested_qty=101, max_order_qty=100), HarnessConfig())
    assert decision.final_status == "blocked"
    assert "Suggested quantity exceeds max order." in decision.failure_reasons


def test_freeze_flag_blocks_automatic_approval():
    decision = evaluate_draft(valid_row(severity=0.75, triage_decision="freeze_and_alert"), HarnessConfig())
    assert decision.final_status == "blocked"
    assert any("Freeze flag" in reason for reason in decision.failure_reasons)


def test_retry_increases_on_failure():
    decision = evaluate_draft(valid_row(suggested_qty=-1, existing_retry_count=2), HarnessConfig(max_retry=3))
    assert decision.retry_count == 3


def test_approval_for_valid_normal_draft():
    decision = evaluate_draft(valid_row(), HarnessConfig())
    assert decision.final_status == "approved"
    assert decision.approved_qty == 20


def test_blocked_status_for_failing_draft():
    decision = evaluate_draft(valid_row(cv_count=-2), HarnessConfig())
    assert decision.final_status == "blocked"
    assert "CV count is negative." in decision.failure_reasons


def test_run_harness_persists_order_and_decision_card(tmp_path: Path):
    db_path = tmp_path / "retail.sqlite"
    create_minimal_db(db_path)
    summary = run_harness(db_path, tmp_path / "out", date="latest")
    assert summary.processed_drafts == 1
    assert summary.approved == 1

    with sqlite3.connect(db_path) as conn:
        order = conn.execute("SELECT status, approved_qty FROM order_history").fetchone()
        card = conn.execute("SELECT harness_result, final_status FROM decision_cards").fetchone()
    assert order == ("approved", 20)
    assert card == ("approved", "approved")


def create_minimal_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sku_master (
                sku_id TEXT PRIMARY KEY,
                product_name TEXT,
                category TEXT,
                supplier_id TEXT,
                representative_image_path TEXT,
                unit_cost REAL,
                selling_price REAL,
                pack_size INTEGER,
                reorder_point INTEGER,
                reorder_quantity INTEGER,
                min_order_qty INTEGER,
                max_order_qty INTEGER,
                storage_volume REAL
            );
            CREATE TABLE suppliers (
                supplier_id TEXT PRIMARY KEY,
                supplier_name TEXT,
                lead_time_days INTEGER
            );
            CREATE TABLE inventory_snapshot (
                date TEXT,
                sku_id TEXT,
                opening_stock INTEGER,
                units_sold INTEGER,
                restock_qty INTEGER,
                closing_stock INTEGER,
                expected_stock INTEGER,
                PRIMARY KEY (date, sku_id)
            );
            CREATE TABLE demand_forecasts (
                date TEXT,
                sku_id TEXT,
                forecast_quantity INTEGER,
                PRIMARY KEY (date, sku_id)
            );
            CREATE TABLE cv_count_log (
                snapshot_id TEXT PRIMARY KEY,
                date TEXT,
                timestamp TEXT,
                sku_id TEXT,
                expected_stock INTEGER,
                cv_count INTEGER,
                count_confidence REAL
            );
            CREATE TABLE anomaly_cases (
                anomaly_id TEXT PRIMARY KEY,
                date TEXT,
                sku_id TEXT,
                anomaly_type TEXT,
                demand_anomaly_score REAL,
                shrinkage_score REAL,
                severity REAL,
                reason TEXT,
                status TEXT
            );
            CREATE TABLE decision_cards (
                decision_id TEXT PRIMARY KEY,
                date TEXT,
                sku_id TEXT,
                trigger_source TEXT,
                severity REAL,
                final_decision TEXT,
                retry_count INTEGER,
                token_cost REAL,
                agent_summary TEXT
            );
            CREATE TABLE order_drafts (
                draft_id TEXT PRIMARY KEY,
                date TEXT,
                sku_id TEXT,
                suggested_qty INTEGER,
                reasoning TEXT,
                confidence REAL,
                status TEXT
            );
            """
        )
        conn.execute("INSERT INTO suppliers VALUES ('SUP001', 'Supplier', 2)")
        conn.execute(
            """
            INSERT INTO sku_master VALUES (
                'SKU001', 'Snack', 'snack', 'SUP001', '', 1000, 1300, 10, 10, 20, 10, 100, 1
            )
            """
        )
        conn.execute("INSERT INTO inventory_snapshot VALUES ('2026-06-28', 'SKU001', 40, 8, 0, 32, 32)")
        conn.execute("INSERT INTO demand_forecasts VALUES ('2026-06-28', 'SKU001', 10)")
        conn.execute("INSERT INTO cv_count_log VALUES ('CV001', '2026-06-28', '2026-06-28T00:00:00', 'SKU001', 32, 30, 0.9)")
        conn.execute("INSERT INTO anomaly_cases VALUES ('AN001', '2026-06-28', 'SKU001', 'normal', 0.1, 0.1, 0.2, 'ok', 'normal')")
        conn.execute("INSERT INTO decision_cards VALUES ('DC001', '2026-06-28', 'SKU001', 'normal', 0.2, 'normal', 0, 0, '{}')")
        conn.execute("INSERT INTO order_drafts VALUES ('DR001', '2026-06-28', 'SKU001', 20, '{}', 0.9, 'drafted')")
