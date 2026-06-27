from __future__ import annotations

import json
import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd


DEFAULT_DB_PATH = Path("synthetic_retail_company_dataset/retail_inventory.sqlite")
DEFAULT_OUTPUT_DIR = Path("results/harness")
FinalStatus = Literal["approved", "requires_manual_review", "blocked"]


@dataclass(frozen=True)
class HarnessConfig:
    daily_budget: float = 500_000.0
    storage_capacity: float = 10_000.0
    supplier_max_supply: int = 500
    max_retry: int = 3


@dataclass(frozen=True)
class HarnessSummary:
    processed_drafts: int
    approved: int
    blocked: int
    requires_manual_review: int
    average_retry_count: float
    failure_reason_counts: dict[str, int]
    output_files: list[Path]
    figure_paths: list[Path]


@dataclass(frozen=True)
class HarnessDecision:
    draft_id: str
    date: str
    sku_id: str
    final_status: FinalStatus
    approved_qty: int
    semantic_status: str
    stock_audit_status: str
    constraint_status: str
    rbac_status: str
    harness_reason: str
    retry_count: int
    failure_reasons: list[str]


def run_harness(
    sqlite_db: Path = DEFAULT_DB_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    date: str = "latest",
    dry_run: bool = False,
    config: HarnessConfig | None = None,
) -> HarnessSummary:
    config = config or HarnessConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_harness_schema(sqlite_db)
    inputs = load_harness_inputs(sqlite_db, date)
    results = evaluate_harness_frame(inputs, config)

    if not dry_run:
        persist_harness_results(sqlite_db, results)

    output_files = write_harness_outputs(output_dir, results)
    figure_paths = create_harness_figures(results, output_dir)
    counts = results["final_status"].value_counts().to_dict() if not results.empty else {}
    failure_reason_counts = count_failure_reasons(results)
    summary = HarnessSummary(
        processed_drafts=int(len(results)),
        approved=int(counts.get("approved", 0)),
        blocked=int(counts.get("blocked", 0)),
        requires_manual_review=int(counts.get("requires_manual_review", 0)),
        average_retry_count=float(results["retry_count"].mean()) if len(results) else 0.0,
        failure_reason_counts=failure_reason_counts,
        output_files=output_files,
        figure_paths=figure_paths,
    )
    summary_path = output_dir / "harness_summary.json"
    summary_payload = {
        "processed_drafts": summary.processed_drafts,
        "approved": summary.approved,
        "blocked": summary.blocked,
        "requires_manual_review": summary.requires_manual_review,
        "average_retry_count": summary.average_retry_count,
        "failure_reason_counts": summary.failure_reason_counts,
        "output_files": [str(path) for path in summary.output_files],
        "figures": [str(path) for path in summary.figure_paths],
        "dry_run": dry_run,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary_path not in output_files:
        output_files.append(summary_path)
    return summary


def ensure_harness_schema(sqlite_db: Path) -> None:
    with sqlite3.connect(sqlite_db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS order_history (
                order_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                sku_id TEXT NOT NULL,
                supplier_id TEXT,
                ordered_qty INTEGER,
                status TEXT NOT NULL,
                reason TEXT
            )
            """
        )
        add_columns(
            conn,
            "order_history",
            {
                "draft_id": "TEXT",
                "approved_qty": "INTEGER DEFAULT 0",
                "approval_reason": "TEXT",
                "harness_time": "TEXT",
            },
        )
        add_columns(
            conn,
            "decision_cards",
            {
                "harness_result": "TEXT",
                "harness_reason": "TEXT",
                "final_status": "TEXT",
            },
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS harness_results (
                validation_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                sku_id TEXT NOT NULL,
                semantic_check TEXT NOT NULL,
                stock_audit TEXT NOT NULL,
                constraint_check TEXT NOT NULL,
                final_result TEXT NOT NULL,
                retry_count INTEGER NOT NULL
            )
            """
        )
        add_columns(
            conn,
            "harness_results",
            {
                "draft_id": "TEXT",
                "rbac_check": "TEXT",
                "failure_reason": "TEXT",
                "harness_time": "TEXT",
            },
        )


def add_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def load_harness_inputs(sqlite_db: Path, date: str) -> pd.DataFrame:
    with sqlite3.connect(sqlite_db) as conn:
        if date == "latest":
            selected_date = conn.execute("SELECT MAX(date) FROM order_drafts").fetchone()[0]
        elif date == "all":
            selected_date = None
        else:
            selected_date = date
        if selected_date is None:
            raise ValueError("No order_drafts found for harness processing.")

        agent_filter = should_filter_agent_drafts(conn, selected_date)
        clauses = []
        params: list[Any] = []
        if selected_date is not None:
            clauses.append("od.date = ?")
            params.append(selected_date)
        if agent_filter:
            clauses.append("od.draft_id LIKE 'AGENT_DR_%'")
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        frame = pd.read_sql_query(
            f"""
            SELECT
                od.draft_id,
                od.date,
                od.sku_id,
                od.suggested_qty,
                od.reasoning AS order_draft_reasoning,
                od.confidence AS order_draft_confidence,
                od.status AS order_draft_status,
                sm.product_name,
                sm.supplier_id,
                sm.unit_cost,
                sm.pack_size,
                sm.reorder_point,
                sm.reorder_quantity,
                sm.min_order_qty,
                sm.max_order_qty,
                sm.storage_volume,
                sup.supplier_id AS valid_supplier_id,
                inv.units_sold AS actual_sales,
                inv.closing_stock,
                inv.expected_stock AS inventory_expected_stock,
                df.forecast_quantity,
                cv.expected_stock AS cv_expected_stock,
                cv.cv_count,
                cv.count_confidence,
                ac.demand_anomaly_score,
                ac.shrinkage_score,
                ac.anomaly_type,
                dc.decision_id,
                dc.trigger_source,
                dc.severity,
                dc.final_decision AS triage_decision,
                dc.retry_count AS existing_retry_count,
                dc.agent_summary
            FROM order_drafts od
            LEFT JOIN sku_master sm ON sm.sku_id = od.sku_id
            LEFT JOIN suppliers sup ON sup.supplier_id = sm.supplier_id
            LEFT JOIN inventory_snapshot inv ON inv.date = od.date AND inv.sku_id = od.sku_id
            LEFT JOIN demand_forecasts df ON df.date = od.date AND df.sku_id = od.sku_id
            LEFT JOIN cv_count_log cv ON cv.date = od.date AND cv.sku_id = od.sku_id
            LEFT JOIN anomaly_cases ac ON ac.date = od.date AND ac.sku_id = od.sku_id
            LEFT JOIN decision_cards dc ON dc.date = od.date AND dc.sku_id = od.sku_id
            {where}
            ORDER BY od.date, od.sku_id, od.draft_id
            """,
            conn,
            params=params,
        )
    if frame.empty:
        raise ValueError(f"No order drafts found for date={date}")
    return frame


def should_filter_agent_drafts(conn: sqlite3.Connection, selected_date: str | None) -> bool:
    if selected_date is None:
        count = conn.execute("SELECT COUNT(*) FROM order_drafts WHERE draft_id LIKE 'AGENT_DR_%'").fetchone()[0]
    else:
        count = conn.execute(
            "SELECT COUNT(*) FROM order_drafts WHERE date = ? AND draft_id LIKE 'AGENT_DR_%'",
            (selected_date,),
        ).fetchone()[0]
    return int(count) > 0


def evaluate_harness_frame(frame: pd.DataFrame, config: HarnessConfig | None = None) -> pd.DataFrame:
    config = config or HarnessConfig()
    rows = []
    for row in frame.to_dict(orient="records"):
        decision = evaluate_draft(row, config)
        rows.append(build_result_row(row, decision))
    return pd.DataFrame(rows)


def evaluate_draft(row: dict[str, Any], config: HarnessConfig | None = None) -> HarnessDecision:
    config = config or HarnessConfig()
    failures: list[str] = []
    rbac_status = "pass"
    suggested_qty = to_int(row.get("suggested_qty"), default=0)
    severity = to_float(row.get("severity"), default=0.0)
    old_retry = to_int(row.get("existing_retry_count"), default=0)

    semantic_failures = semantic_validation(row, suggested_qty)
    failures.extend(semantic_failures)
    semantic_status = "pass" if not semantic_failures else "semantic_failed"

    stock_failures = stock_audit(row, suggested_qty)
    failures.extend(stock_failures)
    stock_status = "pass" if not stock_failures else "stock_audit_failed"

    constraint_failures = constraint_check(row, suggested_qty, config)
    failures.extend(constraint_failures)
    constraint_status = "pass" if not constraint_failures else "constraint_failed"

    retry_count = old_retry
    if failures:
        retry_count = min(config.max_retry, old_retry + 1)

    if failures:
        final_status: FinalStatus = "blocked"
        approved_qty = 0
    elif severity >= 0.7 or row.get("triage_decision") == "freeze_and_alert":
        final_status = "blocked"
        approved_qty = 0
        failures.append("Severity is freeze-level; automatic approval is forbidden.")
        constraint_status = "constraint_failed"
        retry_count = min(config.max_retry, old_retry + 1)
    elif row.get("order_draft_status") in {"requires_review", "blocked"} or row.get("triage_decision") == "requires_review":
        final_status = "requires_manual_review"
        approved_qty = 0
    else:
        final_status = "approved"
        approved_qty = suggested_qty

    reason = "Approved by deterministic harness." if final_status == "approved" else "; ".join(failures) or "Manual review required by triage or agent critique."
    return HarnessDecision(
        draft_id=str(row.get("draft_id")),
        date=str(row.get("date")),
        sku_id=str(row.get("sku_id")),
        final_status=final_status,
        approved_qty=int(approved_qty),
        semantic_status=semantic_status,
        stock_audit_status=stock_status,
        constraint_status=constraint_status,
        rbac_status=rbac_status,
        harness_reason=reason,
        retry_count=int(retry_count),
        failure_reasons=failures,
    )


def semantic_validation(row: dict[str, Any], suggested_qty: int) -> list[str]:
    failures = []
    if pd.isna(row.get("product_name")):
        failures.append("SKU does not exist in sku_master.")
    if pd.isna(row.get("supplier_id")) or pd.isna(row.get("valid_supplier_id")):
        failures.append("Supplier does not exist.")
    if suggested_qty < 0:
        failures.append("Suggested quantity is negative.")
    if to_int(row.get("pack_size"), default=0) <= 0:
        failures.append("Invalid pack_size.")
    if to_int(row.get("min_order_qty"), default=-1) < 0:
        failures.append("Invalid min_order_qty.")
    max_order = to_int(row.get("max_order_qty"), default=-1)
    if max_order <= 0:
        failures.append("Invalid max_order_qty.")
    if suggested_qty > max_order > 0:
        failures.append("Suggested quantity exceeds max order.")
    return failures


def stock_audit(row: dict[str, Any], suggested_qty: int) -> list[str]:
    failures = []
    cv_count = to_float(row.get("cv_count"), default=0.0)
    expected_stock = to_float(
        row.get("cv_expected_stock") if not pd.isna(row.get("cv_expected_stock")) else row.get("inventory_expected_stock"),
        default=0.0,
    )
    forecast = to_float(row.get("forecast_quantity"), default=0.0)
    if cv_count < 0:
        failures.append("CV count is negative.")
    if expected_stock < 0:
        failures.append("Expected stock is negative.")
    if suggested_qty > max(expected_stock + forecast, 1) * 4 and suggested_qty > 50:
        failures.append("Suggested quantity is inconsistent with stock and forecast.")
    return failures


def constraint_check(row: dict[str, Any], suggested_qty: int, config: HarnessConfig) -> list[str]:
    failures = []
    unit_cost = to_float(row.get("unit_cost"), default=0.0)
    storage_volume = to_float(row.get("storage_volume"), default=0.0)
    max_order = to_int(row.get("max_order_qty"), default=0)
    severity = to_float(row.get("severity"), default=0.0)
    if suggested_qty * unit_cost > config.daily_budget:
        failures.append("Budget exceeded.")
    if suggested_qty * storage_volume > config.storage_capacity:
        failures.append("Storage capacity exceeded.")
    if suggested_qty > config.supplier_max_supply:
        failures.append("Supplier max supply exceeded.")
    if max_order > 0 and suggested_qty > max_order:
        failures.append("Max order quantity constraint exceeded.")
    if severity >= 0.7 or row.get("triage_decision") == "freeze_and_alert":
        failures.append("Freeze flag active because severity is high.")
    return failures


def build_result_row(row: dict[str, Any], decision: HarnessDecision) -> dict[str, Any]:
    return {
        "draft_id": decision.draft_id,
        "date": decision.date,
        "sku_id": decision.sku_id,
        "product_name": row.get("product_name"),
        "supplier_id": row.get("supplier_id"),
        "suggested_qty": to_int(row.get("suggested_qty"), default=0),
        "approved_qty": decision.approved_qty,
        "order_draft_status": row.get("order_draft_status"),
        "order_draft_reasoning": row.get("order_draft_reasoning"),
        "demand_forecast": row.get("forecast_quantity"),
        "actual_sales": row.get("actual_sales"),
        "expected_stock": row.get("cv_expected_stock") if not pd.isna(row.get("cv_expected_stock")) else row.get("inventory_expected_stock"),
        "cv_count": row.get("cv_count"),
        "count_confidence": row.get("count_confidence"),
        "shrinkage_score": row.get("shrinkage_score"),
        "demand_anomaly_score": row.get("demand_anomaly_score"),
        "severity": row.get("severity"),
        "trigger_source": row.get("trigger_source"),
        "agent_summary": row.get("agent_summary"),
        "rbac_status": decision.rbac_status,
        "semantic_status": decision.semantic_status,
        "stock_audit_status": decision.stock_audit_status,
        "constraint_status": decision.constraint_status,
        "harness_validation": json.dumps(
            {
                "rbac": decision.rbac_status,
                "semantic": decision.semantic_status,
                "stock_audit": decision.stock_audit_status,
                "constraint": decision.constraint_status,
                "failure_reasons": decision.failure_reasons,
            },
            ensure_ascii=False,
        ),
        "retry_count": decision.retry_count,
        "final_status": decision.final_status,
        "final_decision": decision.final_status,
        "harness_reason": decision.harness_reason,
        "timestamp": now_utc(),
    }


def persist_harness_results(sqlite_db: Path, results: pd.DataFrame) -> None:
    if results.empty:
        return
    ensure_harness_schema(sqlite_db)
    with sqlite3.connect(sqlite_db) as conn:
        for row in results.to_dict(orient="records"):
            suffix = stable_suffix(str(row["draft_id"]))
            validation_id = f"HARNESS_{str(row['date']).replace('-', '')}_{row['sku_id']}_{suffix}"
            conn.execute(
                "DELETE FROM harness_results WHERE date = ? AND sku_id = ? AND draft_id = ?",
                (row["date"], row["sku_id"], row["draft_id"]),
            )
            conn.execute(
                """
                INSERT INTO harness_results (
                    validation_id, date, sku_id, semantic_check, stock_audit, constraint_check,
                    final_result, retry_count, draft_id, rbac_check, failure_reason, harness_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    validation_id,
                    row["date"],
                    row["sku_id"],
                    row["semantic_status"],
                    row["stock_audit_status"],
                    row["constraint_status"],
                    row["final_status"],
                    int(row["retry_count"]),
                    row["draft_id"],
                    row["rbac_status"],
                    row["harness_reason"],
                    row["timestamp"],
                ),
            )
            order_id = f"HARNESS_ORDER_{str(row['date']).replace('-', '')}_{row['sku_id']}_{suffix}"
            conn.execute(
                """
                DELETE FROM order_history
                WHERE date = ? AND sku_id = ? AND substr(order_id, 1, 14) = 'HARNESS_ORDER_'
                """,
                (row["date"], row["sku_id"]),
            )
            conn.execute(
                """
                INSERT INTO order_history (
                    order_id, date, sku_id, supplier_id, ordered_qty, status, reason,
                    draft_id, approved_qty, approval_reason, harness_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    row["date"],
                    row["sku_id"],
                    row.get("supplier_id"),
                    int(row["approved_qty"]),
                    row["final_status"],
                    row["harness_reason"],
                    row["draft_id"],
                    int(row["approved_qty"]),
                    row["harness_reason"],
                    row["timestamp"],
                ),
            )
            conn.execute(
                """
                UPDATE decision_cards
                SET harness_result = ?,
                    harness_reason = ?,
                    retry_count = ?,
                    final_status = ?
                WHERE date = ? AND sku_id = ?
                """,
                (
                    row["final_status"],
                    row["harness_reason"],
                    int(row["retry_count"]),
                    row["final_status"],
                    row["date"],
                    row["sku_id"],
                ),
            )


def write_harness_outputs(output_dir: Path, results: pd.DataFrame) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    harness_path = output_dir / "harness_results.csv"
    cards_csv_path = output_dir / "decision_cards_final.csv"
    cards_md_path = output_dir / "decision_cards_final.md"
    results.to_csv(harness_path, index=False, encoding="utf-8-sig")
    decision_cols = [
        "trigger_source",
        "sku_id",
        "demand_forecast",
        "actual_sales",
        "expected_stock",
        "cv_count",
        "shrinkage_score",
        "demand_anomaly_score",
        "severity",
        "agent_summary",
        "order_draft_reasoning",
        "harness_validation",
        "retry_count",
        "final_decision",
        "timestamp",
    ]
    existing_cols = [col for col in decision_cols if col in results.columns]
    results[existing_cols].to_csv(cards_csv_path, index=False, encoding="utf-8-sig")
    cards_md_path.write_text(render_decision_cards_markdown(results), encoding="utf-8")
    return [harness_path, cards_csv_path, cards_md_path]


def render_decision_cards_markdown(results: pd.DataFrame) -> str:
    lines = ["# Final Decision Cards", ""]
    for row in results.to_dict(orient="records"):
        lines.extend(
            [
                f"## {row.get('date')} / {row.get('sku_id')}",
                f"- Trigger Source: {row.get('trigger_source')}",
                f"- Demand Forecast: {row.get('demand_forecast')}",
                f"- Actual Sales: {row.get('actual_sales')}",
                f"- Expected Stock: {row.get('expected_stock')}",
                f"- CV Count: {row.get('cv_count')}",
                f"- Shrinkage Score: {row.get('shrinkage_score')}",
                f"- Demand Anomaly Score: {row.get('demand_anomaly_score')}",
                f"- Severity: {row.get('severity')}",
                f"- Order Draft: qty={row.get('suggested_qty')}, status={row.get('order_draft_status')}",
                f"- Harness Validation: {row.get('harness_validation')}",
                f"- Retry Count: {row.get('retry_count')}",
                f"- Final Decision: {row.get('final_decision')}",
                f"- Timestamp: {row.get('timestamp')}",
                "",
            ]
        )
    return "\n".join(lines)


def create_harness_figures(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    if results.empty:
        return paths

    counts = results["final_status"].value_counts().reindex(["approved", "blocked", "requires_manual_review"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Harness Approval Result")
    ax.set_ylabel("Count")
    path = output_dir / "harness_approval_result.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = {name: idx for idx, name in enumerate(["approved", "requires_manual_review", "blocked"])}
    ax.scatter(results["severity"].fillna(0), results["final_status"].map(categories), alpha=0.8)
    ax.set_title("Severity vs Final Decision")
    ax.set_xlabel("Severity")
    ax.set_yticks(list(categories.values()), list(categories.keys()))
    path = output_dir / "severity_vs_final_decision.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(results["retry_count"], bins=range(0, int(results["retry_count"].max()) + 3))
    ax.set_title("Retry Count Distribution")
    ax.set_xlabel("Retry Count")
    ax.set_ylabel("Count")
    path = output_dir / "retry_count_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    reason_counts = count_failure_reasons(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    if reason_counts:
        labels = list(reason_counts)
        values = list(reason_counts.values())
        ax.barh(labels, values)
    ax.set_title("Harness Failure Reason")
    ax.set_xlabel("Count")
    path = output_dir / "harness_failure_reason.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def count_failure_reasons(results: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in results.get("harness_validation", pd.Series(dtype=str)).fillna("{}"):
        try:
            reasons = json.loads(value).get("failure_reasons", [])
        except json.JSONDecodeError:
            reasons = []
        if not reasons:
            continue
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def to_int(value: Any, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_suffix(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
