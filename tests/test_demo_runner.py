from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai import demo_runner  # noqa: E402


def test_end_to_end_demo_runner_generates_outputs_without_api(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "demo.sqlite"
    out_dir = tmp_path / "demo"
    morning = tmp_path / "morning.png"
    evening = tmp_path / "evening.png"
    Image.new("RGB", (160, 100), color=(230, 230, 230)).save(morning)
    Image.new("RGB", (160, 100), color=(220, 220, 220)).save(evening)
    create_demo_db(db_path)

    def fake_vision(image_path: Path, vision_dir: Path, prefix: str, detector_config=None):
        detection = vision_dir / f"{prefix}_detection.jpg"
        crops = vision_dir / f"{prefix}_crops"
        crops.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 100), color=(210, 210, 210)).save(detection)
        return {
            "image_path": str(image_path),
            "status": "success",
            "error": "",
            "total_detections": 1,
            "sku_counts": {"SKU001": 5 if prefix == "morning" else 3},
            "predictions": [],
            "low_confidence_predictions": [],
            "detection_image": str(detection),
            "crops_dir": str(crops),
        }

    monkeypatch.setattr(demo_runner, "run_demo_vision_count", fake_vision)
    monkeypatch.setattr(
        demo_runner,
        "run_triage",
        lambda **kwargs: SimpleNamespace(requires_review_count=1, freeze_and_alert_count=0),
    )
    monkeypatch.setattr(
        demo_runner,
        "run_cognition_layer",
        lambda **kwargs: SimpleNamespace(exception_sku_count=1),
    )
    monkeypatch.setattr(
        demo_runner,
        "run_harness",
        lambda **kwargs: SimpleNamespace(processed_drafts=1),
    )
    monkeypatch.setattr(
        demo_runner,
        "run_optimizer",
        lambda **kwargs: update_optimization_and_return(kwargs["sqlite_db"]),
    )

    summary = demo_runner.run_end_to_end_demo(
        date="2026-05-21",
        mode="simulation",
        morning_image=morning,
        evening_image=evening,
        sqlite_db=db_path,
        output_dir=out_dir,
        force_agent_dry_run=True,
    )

    assert summary.llm_dry_run_fallback is True
    assert summary.processed_sku == 1
    assert summary.morning_count == 5
    assert summary.evening_count == 3
    assert (out_dir / "demo_dashboard.html").exists()
    assert (out_dir / "demo_summary.json").exists()
    assert (out_dir / "decision_cards_demo.csv").exists()
    assert (out_dir / "alerts.csv").exists()
    assert (out_dir / "figures" / "pipeline_flow.png").exists()
    cards = pd.read_csv(out_dir / "decision_cards_demo.csv")
    assert cards.loc[0, "SKU"] == "SKU001"
    assert int(cards.loc[0, "Optimized Qty"]) == 12


def create_demo_db(db_path: Path) -> None:
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
            CREATE TABLE inventory_snapshot (
                date TEXT,
                sku_id TEXT,
                opening_stock INTEGER,
                units_sold INTEGER,
                restock_qty INTEGER,
                closing_stock INTEGER,
                expected_stock INTEGER
            );
            CREATE TABLE demand_forecasts (
                date TEXT,
                sku_id TEXT,
                forecast_quantity INTEGER
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
                agent_summary TEXT,
                harness_result TEXT,
                harness_reason TEXT,
                final_status TEXT,
                optimized_qty INTEGER,
                optimization_summary TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO sku_master VALUES (
                'SKU001', 'Demo Snack', 'snack', 'SUP001', '', 1000, 1200, 6, 5, 12, 6, 60, 0.1
            )
            """
        )
        conn.execute("INSERT INTO inventory_snapshot VALUES ('2026-05-21', 'SKU001', 10, 2, 0, 8, 8)")
        conn.execute("INSERT INTO demand_forecasts VALUES ('2026-05-21', 'SKU001', 4)")
        conn.execute("INSERT INTO anomaly_cases VALUES ('AN001', '2026-05-21', 'SKU001', 'shrinkage', 0.2, 0.4, 0.45, 'review', 'requires_review')")
        conn.execute(
            """
            INSERT INTO decision_cards VALUES (
                'DC001', '2026-05-21', 'SKU001', 'combined_trigger', 0.45, 'requires_review',
                0, 0.0, '{"summary":"dry run"}', 'requires_manual_review',
                'manual review', 'requires_manual_review', 0, '{}'
            )
            """
        )


def update_optimization_and_return(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE decision_cards SET optimized_qty = 12, optimization_summary = ? WHERE sku_id = 'SKU001'",
            ('{"mode":"simulation"}',),
        )
    return SimpleNamespace(optimized_sku_count=1)
