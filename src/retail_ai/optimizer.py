from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

try:
    import pulp
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    raise RuntimeError(
        "PuLP is required for the MILP optimizer. Install it with `python -m pip install pulp`."
    ) from exc


DEFAULT_DB_PATH = Path("synthetic_retail_company_dataset/retail_inventory.sqlite")
DEFAULT_OUTPUT_DIR = Path("results/optimizer")
OptimizerMode = Literal["production", "simulation"]


@dataclass(frozen=True)
class OptimizerConfig:
    daily_budget: float = 500_000.0
    storage_capacity: float = 10.0


@dataclass(frozen=True)
class OptimizerSummary:
    mode: str
    solver_status: str
    target_sku_count: int
    optimized_sku_count: int
    total_optimized_quantity: int
    total_cost: float
    budget_usage: float
    storage_usage: float
    output_files: list[Path]
    figure_paths: list[Path]


def run_optimizer(
    sqlite_db: Path = DEFAULT_DB_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    date: str = "latest",
    mode: OptimizerMode = "simulation",
    daily_budget: float = 500_000.0,
    storage_capacity: float = 10.0,
) -> OptimizerSummary:
    if mode not in {"production", "simulation"}:
        raise ValueError("mode must be 'production' or 'simulation'")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    ensure_optimizer_schema(sqlite_db)
    inputs = load_optimizer_inputs(sqlite_db, date)
    results, solver_status = optimize_orders(
        inputs,
        mode=mode,
        config=OptimizerConfig(daily_budget=daily_budget, storage_capacity=storage_capacity),
    )
    persist_optimizer_results(sqlite_db, results, mode)
    output_files = write_optimizer_outputs(output_dir, results, solver_status, mode, daily_budget, storage_capacity)
    figure_paths = create_optimizer_figures(results, figures_dir, daily_budget, storage_capacity)

    target = results[results["is_optimization_target"]]
    optimized = results[results["optimized_qty"] > 0]
    total_cost = float(results["optimized_cost"].sum()) if len(results) else 0.0
    total_storage = float(results["optimized_storage"].sum()) if len(results) else 0.0
    summary = OptimizerSummary(
        mode=mode,
        solver_status=solver_status,
        target_sku_count=int(target["sku_id"].nunique()) if len(target) else 0,
        optimized_sku_count=int(optimized["sku_id"].nunique()) if len(optimized) else 0,
        total_optimized_quantity=int(results["optimized_qty"].sum()) if len(results) else 0,
        total_cost=total_cost,
        budget_usage=float(total_cost / daily_budget) if daily_budget > 0 else 0.0,
        storage_usage=float(total_storage / storage_capacity) if storage_capacity > 0 else 0.0,
        output_files=output_files,
        figure_paths=figure_paths,
    )
    summary_payload = {
        "mode": summary.mode,
        "solver_status": summary.solver_status,
        "target_sku_count": summary.target_sku_count,
        "optimized_sku_count": summary.optimized_sku_count,
        "total_optimized_quantity": summary.total_optimized_quantity,
        "total_cost": summary.total_cost,
        "budget_usage": summary.budget_usage,
        "storage_usage": summary.storage_usage,
        "output_files": [str(path) for path in summary.output_files],
        "figures": [str(path) for path in summary.figure_paths],
    }
    summary_path = output_dir / "optimizer_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary_path not in output_files:
        output_files.append(summary_path)
    return summary


def ensure_optimizer_schema(sqlite_db: Path) -> None:
    with sqlite3.connect(sqlite_db) as conn:
        add_columns(
            conn,
            "decision_cards",
            {
                "optimized_qty": "INTEGER DEFAULT 0",
                "optimization_summary": "TEXT",
            },
        )
        add_columns(
            conn,
            "order_history",
            {
                "optimized_qty": "INTEGER DEFAULT 0",
                "optimization_summary": "TEXT",
                "optimizer_time": "TEXT",
            },
        )


def add_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def load_optimizer_inputs(sqlite_db: Path, date: str) -> pd.DataFrame:
    with sqlite3.connect(sqlite_db) as conn:
        if date == "latest":
            selected_date = conn.execute("SELECT MAX(date) FROM order_drafts").fetchone()[0]
        elif date == "all":
            selected_date = None
        else:
            selected_date = date
        if selected_date is None:
            raise ValueError("No order_drafts found for optimizer.")

        clauses = []
        params: list[Any] = []
        if selected_date is not None:
            clauses.append("od.date = ?")
            params.append(selected_date)
        if has_agent_drafts(conn, selected_date):
            clauses.append("od.draft_id LIKE 'AGENT_DR_%'")
        where = "" if not clauses else "WHERE " + " AND ".join(clauses)
        frame = pd.read_sql_query(
            f"""
            SELECT
                od.draft_id,
                od.date,
                od.sku_id,
                od.suggested_qty AS agent_suggested_qty,
                od.status AS order_draft_status,
                sm.product_name,
                sm.supplier_id,
                sm.unit_cost,
                sm.pack_size,
                sm.min_order_qty,
                sm.max_order_qty,
                sm.storage_volume,
                sup.lead_time_days,
                inv.expected_stock,
                inv.closing_stock,
                df.forecast_quantity,
                dc.decision_id,
                dc.severity,
                dc.harness_result,
                dc.final_status,
                dc.final_decision,
                hr.final_result AS harness_final_result
            FROM order_drafts od
            JOIN sku_master sm ON sm.sku_id = od.sku_id
            LEFT JOIN suppliers sup ON sup.supplier_id = sm.supplier_id
            LEFT JOIN inventory_snapshot inv ON inv.date = od.date AND inv.sku_id = od.sku_id
            LEFT JOIN demand_forecasts df ON df.date = od.date AND df.sku_id = od.sku_id
            LEFT JOIN decision_cards dc ON dc.date = od.date AND dc.sku_id = od.sku_id
            LEFT JOIN harness_results hr ON hr.date = od.date AND hr.sku_id = od.sku_id AND hr.draft_id = od.draft_id
            {where}
            ORDER BY od.date, od.sku_id, od.draft_id
            """,
            conn,
            params=params,
        )
    if frame.empty:
        raise ValueError(f"No optimizer input rows found for date={date}")
    frame["harness_result"] = frame["harness_result"].fillna(frame["harness_final_result"]).fillna(frame["final_status"])
    frame["harness_result"] = frame["harness_result"].map(normalize_harness_result)
    return frame


def has_agent_drafts(conn: sqlite3.Connection, selected_date: str | None) -> bool:
    if selected_date is None:
        count = conn.execute("SELECT COUNT(*) FROM order_drafts WHERE draft_id LIKE 'AGENT_DR_%'").fetchone()[0]
    else:
        count = conn.execute(
            "SELECT COUNT(*) FROM order_drafts WHERE date = ? AND draft_id LIKE 'AGENT_DR_%'",
            (selected_date,),
        ).fetchone()[0]
    return int(count) > 0


def normalize_harness_result(value: Any) -> str:
    if value is None or pd.isna(value):
        return "blocked"
    value = str(value)
    if value in {"pass", "executed", "approved", "optimized"}:
        return "approved"
    if value in {"review", "requires_review", "requires_manual_review"}:
        return "requires_manual_review"
    return "blocked"


def optimize_orders(
    frame: pd.DataFrame,
    mode: OptimizerMode = "simulation",
    config: OptimizerConfig | None = None,
) -> tuple[pd.DataFrame, str]:
    config = config or OptimizerConfig()
    prepared = prepare_optimizer_frame(frame, mode)
    candidates = prepared[prepared["is_optimization_target"]].copy()
    if candidates.empty:
        prepared["optimized_qty"] = 0
        prepared["optimized_cost"] = 0.0
        prepared["optimized_storage"] = 0.0
        prepared["optimizer_status"] = prepared["harness_result"].map(skipped_status)
        prepared["optimizer_reason"] = prepared["harness_result"].map(skipped_reason)
        return prepared, "NoTargets"

    problem = pulp.LpProblem("RetailOrderOptimization", pulp.LpMaximize)
    pack_vars: dict[int, pulp.LpVariable] = {}
    order_flags: dict[int, pulp.LpVariable] = {}
    for idx, row in candidates.iterrows():
        max_packs = int(math.floor(row["max_order_qty"] / row["pack_size"]))
        min_packs = int(math.ceil(row["min_order_qty"] / row["pack_size"]))
        pack_var = pulp.LpVariable(f"pack_count_{idx}", lowBound=0, upBound=max_packs, cat="Integer")
        flag = pulp.LpVariable(f"order_flag_{idx}", lowBound=0, upBound=1, cat="Binary")
        pack_vars[idx] = pack_var
        order_flags[idx] = flag
        problem += pack_var >= min_packs * flag
        problem += pack_var <= max_packs * flag

    problem += pulp.lpSum(
        float(row["priority"]) * float(row["pack_size"]) * pack_vars[idx]
        for idx, row in candidates.iterrows()
    )
    problem += pulp.lpSum(
        float(row["unit_cost"]) * float(row["pack_size"]) * pack_vars[idx]
        for idx, row in candidates.iterrows()
    ) <= config.daily_budget
    problem += pulp.lpSum(
        float(row["storage_volume"]) * float(row["pack_size"]) * pack_vars[idx]
        for idx, row in candidates.iterrows()
    ) <= config.storage_capacity

    solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)
    solver_status = pulp.LpStatus.get(problem.status, "Unknown")

    optimized_qty = {idx: int(round((pulp.value(var) or 0) * float(candidates.loc[idx, "pack_size"]))) for idx, var in pack_vars.items()}
    prepared["optimized_qty"] = 0
    for idx, qty in optimized_qty.items():
        prepared.loc[idx, "optimized_qty"] = qty
    prepared["optimized_cost"] = prepared["optimized_qty"] * prepared["unit_cost"]
    prepared["optimized_storage"] = prepared["optimized_qty"] * prepared["storage_volume"]
    total_cost = float(prepared["optimized_cost"].sum())
    total_storage = float(prepared["optimized_storage"].sum())
    prepared["optimizer_status"] = prepared.apply(
        lambda row: infer_optimizer_status(row, mode, total_cost, total_storage, config),
        axis=1,
    )
    prepared["optimizer_reason"] = prepared.apply(
        lambda row: infer_optimizer_reason(row, total_cost, total_storage, config),
        axis=1,
    )
    return prepared, solver_status


def prepare_optimizer_frame(frame: pd.DataFrame, mode: OptimizerMode) -> pd.DataFrame:
    prepared = frame.copy()
    for col in ["unit_cost", "pack_size", "min_order_qty", "max_order_qty", "storage_volume", "lead_time_days", "expected_stock", "closing_stock", "forecast_quantity", "severity", "agent_suggested_qty"]:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    prepared["pack_size"] = prepared["pack_size"].fillna(1).clip(lower=1)
    prepared["min_order_qty"] = prepared["min_order_qty"].fillna(0).clip(lower=0)
    prepared["max_order_qty"] = prepared["max_order_qty"].fillna(0).clip(lower=0)
    prepared["unit_cost"] = prepared["unit_cost"].fillna(0).clip(lower=0)
    prepared["storage_volume"] = prepared["storage_volume"].fillna(0).clip(lower=0)
    prepared["lead_time_days"] = prepared["lead_time_days"].fillna(1).clip(lower=1)
    prepared["forecast_quantity"] = prepared["forecast_quantity"].fillna(0).clip(lower=0)
    prepared["expected_stock"] = prepared["expected_stock"].fillna(prepared["closing_stock"]).fillna(0).clip(lower=0)
    prepared["severity"] = prepared["severity"].fillna(0).clip(lower=0)
    prepared["safety_stock"] = prepared["forecast_quantity"].clip(lower=1)
    prepared["target_stock"] = prepared["forecast_quantity"] * prepared["lead_time_days"] + prepared["safety_stock"]
    forecast_need = (prepared["target_stock"] - prepared["expected_stock"]).clip(lower=0)
    agent_need = prepared["agent_suggested_qty"].fillna(0).clip(lower=0)
    prepared["needed_qty"] = pd.concat([forecast_need, agent_need], axis=1).max(axis=1)
    prepared["priority"] = prepared["needed_qty"] * (1.0 + prepared["severity"])
    valid = (prepared["max_order_qty"] >= prepared["pack_size"]) & (prepared["unit_cost"] > 0)
    if mode == "production":
        prepared["is_optimization_target"] = valid & (prepared["harness_result"] == "approved")
    else:
        prepared["is_optimization_target"] = valid & prepared["harness_result"].isin(["approved", "requires_manual_review"])
    prepared.loc[prepared["harness_result"] == "blocked", "is_optimization_target"] = False
    return prepared


def infer_optimizer_status(
    row: pd.Series,
    mode: OptimizerMode,
    total_cost: float,
    total_storage: float,
    config: OptimizerConfig,
) -> str:
    if row["harness_result"] == "blocked":
        return "skipped_blocked"
    if row["harness_result"] == "requires_manual_review" and mode == "production":
        return "skipped_requires_review"
    if not bool(row["is_optimization_target"]):
        return skipped_status(row["harness_result"])
    if row["optimized_qty"] <= 0 and row["needed_qty"] > 0:
        if config.daily_budget > 0 and total_cost >= config.daily_budget * 0.999:
            return "budget_limited"
        if config.storage_capacity > 0 and total_storage >= config.storage_capacity * 0.999:
            return "capacity_limited"
    return "simulated_optimized" if mode == "simulation" else "optimized"


def infer_optimizer_reason(row: pd.Series, total_cost: float, total_storage: float, config: OptimizerConfig) -> str:
    if row["optimizer_status"] == "skipped_blocked":
        return "Harness blocked this SKU; optimizer must keep order_qty=0."
    if row["optimizer_status"] == "skipped_requires_review":
        return "Manual review SKU is excluded in production mode."
    if row["optimizer_status"] == "budget_limited":
        return "Budget was fully used before this SKU could receive an order."
    if row["optimizer_status"] == "capacity_limited":
        return "Storage capacity was fully used before this SKU could receive an order."
    return (
        f"MILP optimized with priority={row['priority']:.3f}, "
        f"needed_qty={row['needed_qty']:.3f}, budget_used={total_cost:.2f}, storage_used={total_storage:.3f}."
    )


def skipped_status(harness_result: Any) -> str:
    if harness_result == "requires_manual_review":
        return "skipped_requires_review"
    if harness_result == "blocked":
        return "skipped_blocked"
    return "optimized"


def skipped_reason(harness_result: Any) -> str:
    if harness_result == "requires_manual_review":
        return "Manual review SKU is not eligible for automatic production optimization."
    if harness_result == "blocked":
        return "Harness blocked this SKU."
    return "No eligible target."


def persist_optimizer_results(sqlite_db: Path, results: pd.DataFrame, mode: OptimizerMode) -> None:
    if results.empty:
        return
    ensure_optimizer_schema(sqlite_db)
    timestamp = now_utc()
    with sqlite3.connect(sqlite_db) as conn:
        for row in results.to_dict(orient="records"):
            order_id = f"OPT_{mode.upper()}_{str(row['date']).replace('-', '')}_{row['sku_id']}_{stable_suffix(str(row.get('draft_id')))}"
            conn.execute("DELETE FROM order_history WHERE order_id = ?", (order_id,))
            conn.execute(
                """
                INSERT INTO order_history (
                    order_id, date, sku_id, supplier_id, ordered_qty, status, reason,
                    draft_id, approved_qty, approval_reason, harness_time,
                    optimized_qty, optimization_summary, optimizer_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id,
                    row["date"],
                    row["sku_id"],
                    row.get("supplier_id"),
                    int(row["optimized_qty"]),
                    row["optimizer_status"],
                    row["optimizer_reason"],
                    row.get("draft_id"),
                    0,
                    row.get("optimizer_reason"),
                    row.get("harness_time"),
                    int(row["optimized_qty"]),
                    optimizer_summary_json(row, mode),
                    timestamp,
                ),
            )
            conn.execute(
                """
                UPDATE decision_cards
                SET optimized_qty = ?,
                    optimization_summary = ?
                WHERE date = ? AND sku_id = ?
                """,
                (
                    int(row["optimized_qty"]),
                    optimizer_summary_json(row, mode),
                    row["date"],
                    row["sku_id"],
                ),
            )


def optimizer_summary_json(row: dict[str, Any] | pd.Series, mode: OptimizerMode) -> str:
    return json.dumps(
        {
            "mode": mode,
            "optimized_qty": int(row["optimized_qty"]),
            "optimizer_status": row["optimizer_status"],
            "priority": float(row["priority"]),
            "needed_qty": float(row["needed_qty"]),
            "reason": row["optimizer_reason"],
            "not_executed_by_optimizer": True,
        },
        ensure_ascii=False,
    )


def write_optimizer_outputs(
    output_dir: Path,
    results: pd.DataFrame,
    solver_status: str,
    mode: OptimizerMode,
    daily_budget: float,
    storage_capacity: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "optimized_orders.csv"
    report_path = output_dir / "optimization_report.md"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        render_report(results, solver_status, mode, daily_budget, storage_capacity),
        encoding="utf-8",
    )
    return [csv_path, report_path]


def render_report(
    results: pd.DataFrame,
    solver_status: str,
    mode: OptimizerMode,
    daily_budget: float,
    storage_capacity: float,
) -> str:
    total_cost = float(results["optimized_cost"].sum()) if len(results) else 0.0
    total_storage = float(results["optimized_storage"].sum()) if len(results) else 0.0
    lines = [
        "# Optimization Report",
        "",
        f"- Mode: {mode}",
        f"- Solver Status: {solver_status}",
        f"- Target SKU Count: {int(results['is_optimization_target'].sum()) if len(results) else 0}",
        f"- Optimized SKU Count: {int((results['optimized_qty'] > 0).sum()) if len(results) else 0}",
        f"- Total Optimized Quantity: {int(results['optimized_qty'].sum()) if len(results) else 0}",
        f"- Total Cost: {total_cost:.2f}",
        f"- Budget Usage: {(total_cost / daily_budget if daily_budget else 0):.3f}",
        f"- Storage Usage: {(total_storage / storage_capacity if storage_capacity else 0):.3f}",
        "",
        "## Orders",
        "",
    ]
    for row in results.to_dict(orient="records"):
        lines.append(
            f"- {row['sku_id']}: agent={row['agent_suggested_qty']}, optimized={row['optimized_qty']}, "
            f"status={row['optimizer_status']}, priority={row['priority']:.3f}"
        )
    return "\n".join(lines)


def create_optimizer_figures(
    results: pd.DataFrame,
    figures_dir: Path,
    daily_budget: float,
    storage_capacity: float,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if results.empty:
        return paths

    x = range(len(results))
    labels = results["sku_id"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 0.2 for i in x], results["agent_suggested_qty"].fillna(0), width=0.4, label="Agent")
    ax.bar([i + 0.2 for i in x], results["optimized_qty"], width=0.4, label="Optimized")
    ax.set_xticks(list(x), labels, rotation=45, ha="right")
    ax.set_title("Agent Suggested Quantity vs Optimized Quantity")
    ax.legend()
    path = figures_dir / "optimized_vs_agent.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, results["optimized_cost"])
    ax.set_xticks(list(range(len(labels))), labels, rotation=45, ha="right")
    ax.set_title("Budget Allocation by SKU")
    ax.set_ylabel("Cost")
    path = figures_dir / "budget_allocation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 5))
    total_cost = float(results["optimized_cost"].sum())
    total_storage = float(results["optimized_storage"].sum())
    ax.bar(["Budget", "Storage"], [total_cost / daily_budget if daily_budget else 0, total_storage / storage_capacity if storage_capacity else 0])
    ax.set_ylim(0, max(1.0, total_cost / daily_budget if daily_budget else 0, total_storage / storage_capacity if storage_capacity else 0))
    ax.set_title("Constraint Utilization")
    path = figures_dir / "constraint_utilization.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, results["priority"])
    ax.set_xticks(list(range(len(labels))), labels, rotation=45, ha="right")
    ax.set_title("Priority Distribution")
    path = figures_dir / "priority_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_suffix(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
