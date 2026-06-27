from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.agents import (  # noqa: E402
    OrderDraftingAgent,
    SelfCritiqueAgent,
    load_exception_contexts,
    run_cognition_layer,
)
from retail_ai.llm_client import RuleBasedLLMClient  # noqa: E402


def create_test_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE anomaly_cases (
            anomaly_id TEXT PRIMARY KEY, date TEXT, sku_id TEXT, anomaly_type TEXT,
            demand_anomaly_score REAL, shrinkage_score REAL, severity REAL, reason TEXT, status TEXT
        );
        CREATE TABLE decision_cards (
            decision_id TEXT PRIMARY KEY, date TEXT, sku_id TEXT, trigger_source TEXT,
            severity REAL, final_decision TEXT, retry_count INTEGER, token_cost REAL
        );
        CREATE TABLE sku_master (
            sku_id TEXT PRIMARY KEY, product_name TEXT, category TEXT, supplier_id TEXT,
            representative_image_path TEXT, unit_cost REAL, selling_price REAL, pack_size INTEGER,
            reorder_point INTEGER, reorder_quantity INTEGER, min_order_qty INTEGER,
            max_order_qty INTEGER, storage_volume REAL
        );
        CREATE TABLE inventory_snapshot (
            date TEXT, sku_id TEXT, opening_stock INTEGER, units_sold INTEGER,
            restock_qty INTEGER, closing_stock INTEGER, expected_stock INTEGER
        );
        CREATE TABLE demand_forecasts (date TEXT, sku_id TEXT, forecast_quantity INTEGER);
        CREATE TABLE cv_count_log (
            snapshot_id TEXT, date TEXT, timestamp TEXT, sku_id TEXT,
            expected_stock INTEGER, cv_count INTEGER, count_confidence REAL
        );
        CREATE TABLE rag_case_library (
            case_id TEXT, anomaly_type TEXT, summary TEXT, evidence TEXT, resolution TEXT, tags TEXT
        );
        CREATE TABLE vision_detections_sample (snapshot_id TEXT, sku_id TEXT, detections TEXT);
        CREATE TABLE order_drafts (
            draft_id TEXT PRIMARY KEY, date TEXT, sku_id TEXT, suggested_qty INTEGER,
            reasoning TEXT, confidence REAL
        );
        """
    )
    conn.executemany(
        "INSERT INTO sku_master VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("A", "product-a", "snack", "S1", "", 1, 2, 6, 5, 12, 6, 48, 0.1),
            ("B", "product-b", "snack", "S1", "", 1, 2, 6, 5, 12, 6, 48, 0.1),
        ],
    )
    conn.executemany(
        "INSERT INTO anomaly_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("AN1", "2026-01-01", "A", "demand_spike", 0.8, 0.0, 0.35, "review", "requires_review"),
            ("AN2", "2026-01-01", "B", "normal", 0.1, 0.0, 0.1, "normal", "normal"),
        ],
    )
    conn.executemany(
        "INSERT INTO decision_cards VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("DC1", "2026-01-01", "A", "demand_trigger", 0.35, "requires_review", 0, 0.0),
            ("DC2", "2026-01-01", "B", "normal", 0.1, "normal", 0, 0.0),
        ],
    )
    conn.execute("INSERT INTO inventory_snapshot VALUES ('2026-01-01','A',10,8,0,2,2)")
    conn.execute("INSERT INTO demand_forecasts VALUES ('2026-01-01','A',4)")
    conn.execute("INSERT INTO cv_count_log VALUES ('CV1','2026-01-01','2026-01-01 23:00:00','A',2,2,0.9)")
    conn.execute("INSERT INTO rag_case_library VALUES ('CASE1','demand_spike','summary','{}','resolved','[]')")
    conn.execute("INSERT INTO vision_detections_sample VALUES ('CV1','A','[]')")
    conn.commit()
    conn.close()


def test_requires_review_only_selected(tmp_path: Path):
    db = tmp_path / "test.sqlite"
    create_test_db(db)
    contexts = load_exception_contexts(db, "latest")

    assert len(contexts) == 1
    assert contexts[0]["sku"]["sku_id"] == "A"


def test_agent_output_schema_valid():
    client = RuleBasedLLMClient()
    payload = {
        "sku": {"sku_id": "A", "reorder_quantity": 12},
        "anomaly": {"date": "2026-01-01", "severity": 0.4, "anomaly_type": "demand_spike", "status": "requires_review"},
        "context": {"actual_sales": 8, "forecast_quantity": 4, "cv_count": 2, "expected_stock": 2, "count_confidence": 0.9},
        "similar_cases": [],
    }
    draft = OrderDraftingAgent(client).run(payload)
    critique = SelfCritiqueAgent(client).run({**payload, "order_draft": draft})

    assert draft["agent_name"] == "OrderDraftingAgent"
    assert critique["agent_name"] == "SelfCritiqueAgent"


def test_run_cognition_layer_creates_order_draft(tmp_path: Path):
    db = tmp_path / "test.sqlite"
    create_test_db(db)

    summary = run_cognition_layer(db, tmp_path / "out", "latest", RuleBasedLLMClient())

    assert summary.exception_sku_count == 1
    assert summary.order_draft_count == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM order_drafts WHERE draft_id LIKE 'AGENT_DR_%'").fetchone()[0] == 1
        assert conn.execute("SELECT agent_summary FROM decision_cards WHERE sku_id='A'").fetchone()[0][0] is not None


def test_blocked_status_when_self_critique_blocks(tmp_path: Path):
    db = tmp_path / "test.sqlite"
    create_test_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE anomaly_cases SET anomaly_type='theft_suspected', severity=0.8 WHERE sku_id='A'")
        conn.execute("UPDATE decision_cards SET final_decision='freeze_and_alert', severity=0.8 WHERE sku_id='A'")

    summary = run_cognition_layer(db, tmp_path / "out", "latest", RuleBasedLLMClient())

    assert summary.blocked_count == 1
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT status FROM order_drafts WHERE draft_id LIKE 'AGENT_DR_%'").fetchone()[0] == "blocked"


def test_dry_run_client_no_api_needed(tmp_path: Path):
    db = tmp_path / "test.sqlite"
    create_test_db(db)

    summary = run_cognition_layer(db, tmp_path / "out", "latest", RuleBasedLLMClient())

    assert summary.agent_call_count == 4
