from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path("synthetic_retail_company_dataset/retail_inventory.sqlite")


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def get_sku(sku_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM sku_master WHERE sku_id = ?", (sku_id,)).fetchone()
    return _row_to_dict(row)


def list_skus(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM sku_master ORDER BY sku_id").fetchall()
    return [dict(row) for row in rows]


def list_sku_images(sku_id: str | None = None, db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        if sku_id is None:
            rows = conn.execute("SELECT * FROM sku_images ORDER BY sku_id, image_id").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sku_images WHERE sku_id = ? ORDER BY image_id",
                (sku_id,),
            ).fetchall()
    return [dict(row) for row in rows]


def get_inventory(sku_id: str, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM inventory_snapshot
            WHERE sku_id = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (sku_id,),
        ).fetchone()
    return _row_to_dict(row)


def insert_cv_count(record: dict[str, Any], db_path: str | Path = DEFAULT_DB_PATH) -> None:
    insert_record("cv_count_log", record, db_path)


def insert_order_draft(record: dict[str, Any], db_path: str | Path = DEFAULT_DB_PATH) -> None:
    insert_record("order_drafts", record, db_path)


def insert_order_history(record: dict[str, Any], db_path: str | Path = DEFAULT_DB_PATH) -> None:
    insert_record("order_history", record, db_path)


def insert_harness_result(record: dict[str, Any], db_path: str | Path = DEFAULT_DB_PATH) -> None:
    insert_record("harness_results", record, db_path)


def insert_decision_card(record: dict[str, Any], db_path: str | Path = DEFAULT_DB_PATH) -> None:
    insert_record("decision_cards", record, db_path)


def insert_record(table_name: str, record: dict[str, Any], db_path: str | Path = DEFAULT_DB_PATH) -> None:
    normalized = {
        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        for key, value in record.items()
    }
    columns = list(normalized)
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    sql = f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders})"
    with get_connection(db_path) as conn:
        conn.execute(sql, [normalized[column] for column in columns])
