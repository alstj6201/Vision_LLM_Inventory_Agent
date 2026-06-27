from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


Route = Literal["normal", "requires_review", "freeze_and_alert"]
DEFAULT_DB_PATH = Path("synthetic_retail_company_dataset/retail_inventory.sqlite")
DEFAULT_OUTPUT_DIR = Path("results/triage")


@dataclass(frozen=True)
class TriageSummary:
    processed_sku_count: int
    normal_count: int
    requires_review_count: int
    freeze_and_alert_count: int
    theft_suspected_count: int
    average_severity: float
    figure_paths: list[Path]
    output_dir: Path


def compute_shrinkage_score(expected_stock: float, cv_count: float) -> float:
    shrinkage_gap = expected_stock - cv_count
    if shrinkage_gap <= 0:
        return 0.0
    return float(min(1.0, shrinkage_gap / max(expected_stock, 1)))


def compute_demand_anomaly_score(actual_sales: float, forecast_quantity: float) -> float:
    gap = abs(actual_sales - forecast_quantity)
    return float(min(1.0, gap / max(abs(forecast_quantity), abs(actual_sales), 1.0)))


def compute_low_confidence_penalty(count_confidence: float) -> float:
    return float(max(0.0, 0.85 - count_confidence))


def compute_theft_suspicion_indicator(
    demand_anomaly_score: float,
    shrinkage_score: float,
    count_confidence: float,
) -> int:
    return int(demand_anomaly_score < 0.3 and shrinkage_score > 0.3 and count_confidence >= 0.85)


def compute_severity(
    demand_anomaly_score: float,
    shrinkage_score: float,
    theft_suspicion_indicator: int,
    low_confidence_penalty: float,
) -> float:
    severity = (
        0.35 * demand_anomaly_score
        + 0.40 * shrinkage_score
        + 0.15 * theft_suspicion_indicator
        + 0.10 * low_confidence_penalty
    )
    return float(np.clip(severity, 0.0, 1.0))


def route_severity(severity: float, theft_suspicion_indicator: int = 0) -> Route:
    if theft_suspicion_indicator == 1:
        return "freeze_and_alert"
    if severity < 0.3:
        return "normal"
    if severity < 0.7:
        return "requires_review"
    return "freeze_and_alert"


def infer_anomaly_type(
    demand_anomaly_score: float,
    shrinkage_score: float,
    count_confidence: float,
    theft_suspicion_indicator: int,
) -> str:
    if theft_suspicion_indicator == 1:
        return "theft_suspected"
    if shrinkage_score > 0.3 and count_confidence < 0.85:
        return "cv_uncertain"
    if shrinkage_score > 0.3:
        return "shrinkage"
    if demand_anomaly_score > 0.7:
        return "demand_spike"
    return "normal"


def infer_trigger_source(demand_anomaly_score: float, shrinkage_score: float) -> str:
    demand_high = demand_anomaly_score >= 0.3
    shrinkage_high = shrinkage_score >= 0.3
    if demand_high and shrinkage_high:
        return "combined_trigger"
    if demand_high:
        return "demand_trigger"
    if shrinkage_high:
        return "supply_trigger"
    return "normal"


def run_triage(
    sqlite_db: Path = DEFAULT_DB_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    date: str = "latest",
) -> TriageSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    triage_df = load_triage_inputs(sqlite_db, date)
    results = score_triage_frame(triage_df)
    results.to_csv(output_dir / "triage_results.csv", index=False, encoding="utf-8-sig")
    persist_triage_results(sqlite_db, results)
    figure_paths = create_triage_figures(results, output_dir)

    counts = results["route"].value_counts().to_dict()
    summary = TriageSummary(
        processed_sku_count=int(results["sku_id"].nunique() if date == "latest" else len(results)),
        normal_count=int(counts.get("normal", 0)),
        requires_review_count=int(counts.get("requires_review", 0)),
        freeze_and_alert_count=int(counts.get("freeze_and_alert", 0)),
        theft_suspected_count=int(results["theft_suspicion_indicator"].sum()),
        average_severity=float(results["severity"].mean()) if len(results) else 0.0,
        figure_paths=figure_paths,
        output_dir=output_dir,
    )
    with (output_dir / "triage_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "processed_sku_count": summary.processed_sku_count,
                "normal_count": summary.normal_count,
                "requires_review_count": summary.requires_review_count,
                "freeze_and_alert_count": summary.freeze_and_alert_count,
                "theft_suspected_count": summary.theft_suspected_count,
                "average_severity": summary.average_severity,
                "figures": [str(path) for path in figure_paths],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return summary


def load_triage_inputs(sqlite_db: Path, date: str) -> pd.DataFrame:
    with sqlite3.connect(sqlite_db) as conn:
        if date == "latest":
            selected_date = conn.execute("SELECT MAX(date) FROM cv_count_log").fetchone()[0]
        elif date == "all":
            selected_date = None
        else:
            selected_date = date

        where = "" if selected_date is None else "WHERE cv.date = ?"
        params = [] if selected_date is None else [selected_date]
        query = f"""
            SELECT
                cv.date,
                cv.timestamp,
                cv.sku_id,
                sm.product_name,
                inv.units_sold AS actual_sales,
                df.forecast_quantity,
                cv.expected_stock,
                cv.cv_count,
                cv.count_confidence
            FROM cv_count_log cv
            JOIN inventory_snapshot inv
              ON inv.date = cv.date AND inv.sku_id = cv.sku_id
            JOIN sku_master sm
              ON sm.sku_id = cv.sku_id
            LEFT JOIN demand_forecasts df
              ON df.date = cv.date AND df.sku_id = cv.sku_id
            {where}
            ORDER BY cv.date, cv.sku_id
        """
        frame = pd.read_sql_query(query, conn, params=params)
    if frame.empty:
        raise ValueError(f"No triage input rows found for date={date}")
    frame["forecast_quantity"] = frame["forecast_quantity"].fillna(frame["actual_sales"])
    return frame


def score_triage_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.itertuples(index=False):
        shrinkage_score = compute_shrinkage_score(float(row.expected_stock), float(row.cv_count))
        demand_anomaly_score = compute_demand_anomaly_score(float(row.actual_sales), float(row.forecast_quantity))
        low_confidence_penalty = compute_low_confidence_penalty(float(row.count_confidence))
        theft = compute_theft_suspicion_indicator(
            demand_anomaly_score,
            shrinkage_score,
            float(row.count_confidence),
        )
        severity = compute_severity(demand_anomaly_score, shrinkage_score, theft, low_confidence_penalty)
        route = route_severity(severity, theft)
        anomaly_type = infer_anomaly_type(demand_anomaly_score, shrinkage_score, float(row.count_confidence), theft)
        trigger_source = infer_trigger_source(demand_anomaly_score, shrinkage_score)
        rows.append(
            {
                "date": row.date,
                "sku_id": row.sku_id,
                "product_name": row.product_name,
                "actual_sales": float(row.actual_sales),
                "forecast_quantity": float(row.forecast_quantity),
                "expected_stock": float(row.expected_stock),
                "cv_count": float(row.cv_count),
                "count_confidence": float(row.count_confidence),
                "demand_anomaly_score": demand_anomaly_score,
                "shrinkage_score": shrinkage_score,
                "low_confidence_penalty": low_confidence_penalty,
                "theft_suspicion_indicator": theft,
                "severity": severity,
                "route": route,
                "anomaly_type": anomaly_type,
                "trigger_source": trigger_source,
                "reason": build_reason(anomaly_type, demand_anomaly_score, shrinkage_score, row.count_confidence),
            }
        )
    return pd.DataFrame(rows)


def build_reason(anomaly_type: str, demand_score: float, shrinkage_score: float, confidence: float) -> str:
    return (
        f"{anomaly_type}: demand_anomaly_score={demand_score:.3f}, "
        f"shrinkage_score={shrinkage_score:.3f}, count_confidence={confidence:.3f}"
    )


def persist_triage_results(sqlite_db: Path, results: pd.DataFrame) -> None:
    actionable = results[results["route"].isin(["requires_review", "freeze_and_alert"])].copy()
    with sqlite3.connect(sqlite_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        if not actionable.empty:
            dates = sorted(actionable["date"].unique())
            for date in dates:
                conn.execute("DELETE FROM anomaly_cases WHERE date = ?", (date,))
                conn.execute("DELETE FROM decision_cards WHERE date = ?", (date,))
            anomaly_rows = []
            decision_rows = []
            for idx, row in enumerate(actionable.itertuples(index=False), start=1):
                suffix = f"{str(row.date).replace('-', '')}_{idx:05d}_{row.sku_id}"
                anomaly_rows.append(
                    (
                        f"TRIAGE_AN_{suffix}",
                        row.date,
                        row.sku_id,
                        row.anomaly_type,
                        row.demand_anomaly_score,
                        row.shrinkage_score,
                        row.severity,
                        row.reason,
                        row.route,
                    )
                )
                decision_rows.append(
                    (
                        f"TRIAGE_DC_{suffix}",
                        row.date,
                        row.sku_id,
                        row.trigger_source,
                        row.severity,
                        row.route,
                        0,
                        0.0,
                    )
                )
            conn.executemany(
                """
                INSERT INTO anomaly_cases (
                    anomaly_id, date, sku_id, anomaly_type, demand_anomaly_score,
                    shrinkage_score, severity, reason, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                anomaly_rows,
            )
            conn.executemany(
                """
                INSERT INTO decision_cards (
                    decision_id, date, sku_id, trigger_source, severity,
                    final_decision, retry_count, token_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                decision_rows,
            )


def create_triage_figures(results: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(results["severity"], bins=30)
    ax.set_title("Severity Score Distribution")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Count")
    path = output_dir / "severity_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    route_counts = results["route"].value_counts().reindex(["normal", "requires_review", "freeze_and_alert"], fill_value=0)
    ax.bar(route_counts.index, route_counts.values)
    ax.set_title("Routing Counts")
    ax.set_ylabel("Count")
    path = output_dir / "routing_counts.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    top = results.sort_values("severity", ascending=False).head(20).sort_values("severity")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["sku_id"], top["severity"])
    ax.set_title("Severity Top 20 SKU")
    ax.set_xlabel("Severity")
    path = output_dir / "severity_by_sku_top20.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        results["demand_anomaly_score"],
        results["shrinkage_score"],
        c=results["severity"],
        cmap="viridis",
        alpha=0.75,
    )
    ax.set_title("Demand Anomaly vs Shrinkage")
    ax.set_xlabel("Demand Anomaly Score")
    ax.set_ylabel("Shrinkage Score")
    fig.colorbar(scatter, ax=ax, label="Severity")
    path = output_dir / "demand_vs_shrinkage_scatter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths
