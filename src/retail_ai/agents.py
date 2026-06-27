from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from retail_ai.llm_client import LLMClient, RuleBasedLLMClient


DEFAULT_DB_PATH = Path("synthetic_retail_company_dataset/retail_inventory.sqlite")
DEFAULT_OUTPUT_DIR = Path("results/agents")


@dataclass(frozen=True)
class AgentRunSummary:
    exception_sku_count: int
    agent_call_count: int
    order_draft_count: int
    blocked_count: int
    requires_review_count: int
    output_files: list[Path]


AGENT_SCHEMAS = {
    "CausalContextAgent": {"agent_name", "sku_id", "likely_causes", "similar_cases", "evidence", "confidence", "summary"},
    "VisionGroundedAgent": {"agent_name", "sku_id", "visual_assessment", "possible_issue", "evidence", "confidence"},
    "OrderDraftingAgent": {"agent_name", "sku_id", "recommended_action", "suggested_qty", "reasoning", "evidence", "uncertainty"},
    "SelfCritiqueAgent": {"agent_name", "sku_id", "critique_result", "issues", "final_recommendation", "summary"},
}


class BaseAgent:
    agent_name = "BaseAgent"
    system_prompt = "You are a retail inventory reasoning agent."

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        output = self.llm_client.complete_json(self.agent_name, self.system_prompt, payload)
        validate_agent_output(self.agent_name, output)
        return output


class CausalContextAgent(BaseAgent):
    agent_name = "CausalContextAgent"
    system_prompt = (
        "Summarize likely causes using historical incidents, demand forecast, sales, "
        "inventory, events, and triage evidence. Return structured JSON only."
    )


class VisionGroundedAgent(BaseAgent):
    agent_name = "VisionGroundedAgent"
    system_prompt = (
        "Assess visual/CV evidence using cv_count, expected_stock, count_confidence, "
        "and detection metadata. Use metadata if raw images are unavailable."
    )


class OrderDraftingAgent(BaseAgent):
    agent_name = "OrderDraftingAgent"
    system_prompt = (
        "Draft a safe order recommendation JSON. Do not execute orders or call APIs. "
        "Any order must later pass the deterministic harness."
    )


class SelfCritiqueAgent(BaseAgent):
    agent_name = "SelfCritiqueAgent"
    system_prompt = (
        "Critique the order draft against evidence and safety constraints. Return pass, revise, or block."
    )


def validate_agent_output(agent_name: str, output: dict[str, Any]) -> None:
    required = AGENT_SCHEMAS[agent_name]
    missing = required - set(output)
    if missing:
        raise ValueError(f"{agent_name} output missing fields: {sorted(missing)}")
    if output.get("agent_name") != agent_name:
        raise ValueError(f"{agent_name} output has wrong agent_name={output.get('agent_name')}")


def run_cognition_layer(
    sqlite_db: Path = DEFAULT_DB_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    date: str = "latest",
    llm_client: LLMClient | None = None,
) -> AgentRunSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_agent_columns(sqlite_db)
    llm_client = llm_client or RuleBasedLLMClient()
    contexts = load_exception_contexts(sqlite_db, date)

    causal_agent = CausalContextAgent(llm_client)
    vision_agent = VisionGroundedAgent(llm_client)
    drafting_agent = OrderDraftingAgent(llm_client)
    critique_agent = SelfCritiqueAgent(llm_client)

    outputs = []
    draft_rows = []
    for context in contexts:
        causal = causal_agent.run(context)
        vision = vision_agent.run({**context, "causal_context": causal})
        draft = drafting_agent.run({**context, "causal_context": causal, "vision_context": vision})
        critique = critique_agent.run({**context, "causal_context": causal, "vision_context": vision, "order_draft": draft})
        status = order_draft_status(draft, critique)
        draft_row = make_order_draft_row(context, draft, critique, status)
        draft_rows.append(draft_row)
        outputs.append(
            {
                "sku_id": context["sku"]["sku_id"],
                "date": context["anomaly"]["date"],
                "causal_context": causal,
                "vision_grounded": vision,
                "order_draft": draft,
                "self_critique": critique,
                "order_draft_status": status,
            }
        )

    persist_agent_results(sqlite_db, outputs, draft_rows)
    output_files = write_agent_outputs(output_dir, outputs, draft_rows)
    blocked = sum(1 for row in draft_rows if row["status"] == "blocked")
    requires_review = sum(1 for row in draft_rows if row["status"] == "requires_review")
    return AgentRunSummary(
        exception_sku_count=len(contexts),
        agent_call_count=len(contexts) * 4,
        order_draft_count=len(draft_rows),
        blocked_count=blocked,
        requires_review_count=requires_review,
        output_files=output_files,
    )


def ensure_agent_columns(sqlite_db: Path) -> None:
    with sqlite3.connect(sqlite_db) as conn:
        order_cols = {row[1] for row in conn.execute("PRAGMA table_info(order_drafts)").fetchall()}
        if "status" not in order_cols:
            conn.execute("ALTER TABLE order_drafts ADD COLUMN status TEXT DEFAULT 'drafted'")
        decision_cols = {row[1] for row in conn.execute("PRAGMA table_info(decision_cards)").fetchall()}
        if "agent_summary" not in decision_cols:
            conn.execute("ALTER TABLE decision_cards ADD COLUMN agent_summary TEXT")


def load_exception_contexts(sqlite_db: Path, date: str) -> list[dict[str, Any]]:
    with sqlite3.connect(sqlite_db) as conn:
        conn.row_factory = sqlite3.Row
        if date == "latest":
            target_date = conn.execute("SELECT MAX(date) FROM decision_cards").fetchone()[0]
        elif date == "all":
            target_date = None
        else:
            target_date = date
        where = "WHERE dc.final_decision IN ('requires_review', 'freeze_and_alert')"
        params: list[Any] = []
        if target_date is not None:
            where += " AND dc.date = ?"
            params.append(target_date)
        rows = conn.execute(
            f"""
            SELECT
                dc.decision_id,
                dc.date,
                dc.sku_id,
                dc.trigger_source,
                dc.severity,
                dc.final_decision,
                ac.anomaly_type,
                ac.demand_anomaly_score,
                ac.shrinkage_score,
                ac.reason,
                sm.*,
                inv.units_sold AS actual_sales,
                inv.expected_stock AS inventory_expected_stock,
                inv.closing_stock,
                df.forecast_quantity,
                cv.cv_count,
                cv.count_confidence,
                cv.expected_stock AS cv_expected_stock
            FROM decision_cards dc
            JOIN anomaly_cases ac ON ac.date = dc.date AND ac.sku_id = dc.sku_id
            JOIN sku_master sm ON sm.sku_id = dc.sku_id
            LEFT JOIN inventory_snapshot inv ON inv.date = dc.date AND inv.sku_id = dc.sku_id
            LEFT JOIN demand_forecasts df ON df.date = dc.date AND df.sku_id = dc.sku_id
            LEFT JOIN cv_count_log cv ON cv.date = dc.date AND cv.sku_id = dc.sku_id
            {where}
            ORDER BY dc.date, dc.sku_id
            """,
            params,
        ).fetchall()

        contexts = []
        for row in rows:
            sku_id = row["sku_id"]
            similar_cases = [
                dict(case)
                for case in conn.execute(
                    "SELECT * FROM rag_case_library WHERE anomaly_type = ? LIMIT 5",
                    (row["anomaly_type"],),
                ).fetchall()
            ]
            vision_rows = [
                dict(v)
                for v in conn.execute(
                    "SELECT * FROM vision_detections_sample WHERE sku_id = ? LIMIT 3",
                    (sku_id,),
                ).fetchall()
            ]
            contexts.append(
                {
                    "sku": {
                        "sku_id": sku_id,
                        "product_name": row["product_name"],
                        "category": row["category"],
                        "supplier_id": row["supplier_id"],
                        "reorder_point": row["reorder_point"],
                        "reorder_quantity": row["reorder_quantity"],
                        "min_order_qty": row["min_order_qty"],
                        "max_order_qty": row["max_order_qty"],
                    },
                    "anomaly": {
                        "date": row["date"],
                        "anomaly_type": row["anomaly_type"],
                        "demand_anomaly_score": row["demand_anomaly_score"],
                        "shrinkage_score": row["shrinkage_score"],
                        "severity": row["severity"],
                        "reason": row["reason"],
                        "status": row["final_decision"],
                        "trigger_source": row["trigger_source"],
                        "decision_id": row["decision_id"],
                    },
                    "context": {
                        "actual_sales": row["actual_sales"],
                        "forecast_quantity": row["forecast_quantity"],
                        "expected_stock": row["cv_expected_stock"] if row["cv_expected_stock"] is not None else row["inventory_expected_stock"],
                        "closing_stock": row["closing_stock"],
                        "cv_count": row["cv_count"],
                        "count_confidence": row["count_confidence"],
                    },
                    "similar_cases": similar_cases,
                    "vision_detections": vision_rows,
                }
            )
    return contexts


def order_draft_status(draft: dict[str, Any], critique: dict[str, Any]) -> str:
    if critique["critique_result"] == "block" or draft["recommended_action"] == "block":
        return "blocked"
    if draft["recommended_action"] == "review" or critique["final_recommendation"] == "require_human_review":
        return "requires_review"
    return "drafted"


def make_order_draft_row(
    context: dict[str, Any],
    draft: dict[str, Any],
    critique: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    confidence = {"low": 0.55, "medium": 0.75, "high": 0.9}.get(str(draft.get("uncertainty")), 0.65)
    return {
        "draft_id": f"AGENT_DR_{context['anomaly']['date'].replace('-', '')}_{context['sku']['sku_id']}",
        "date": context["anomaly"]["date"],
        "sku_id": context["sku"]["sku_id"],
        "suggested_qty": int(draft["suggested_qty"]),
        "reasoning": json.dumps(
            {
                "draft_reasoning": draft["reasoning"],
                "draft_evidence": draft["evidence"],
                "critique": critique,
                "harness_status": "not_run_yet",
                "agent_reasoning_only": True,
            },
            ensure_ascii=False,
        ),
        "confidence": confidence,
        "status": status,
    }


def persist_agent_results(
    sqlite_db: Path,
    outputs: list[dict[str, Any]],
    draft_rows: list[dict[str, Any]],
) -> None:
    if not outputs:
        return
    with sqlite3.connect(sqlite_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        for row in draft_rows:
            conn.execute("DELETE FROM order_drafts WHERE draft_id = ?", (row["draft_id"],))
            conn.execute(
                """
                INSERT INTO order_drafts (
                    draft_id, date, sku_id, suggested_qty, reasoning, confidence, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["draft_id"],
                    row["date"],
                    row["sku_id"],
                    row["suggested_qty"],
                    row["reasoning"],
                    row["confidence"],
                    row["status"],
                ),
            )
        for output in outputs:
            summary = {
                "causal_summary": output["causal_context"]["summary"],
                "vision_issue": output["vision_grounded"]["possible_issue"],
                "recommended_action": output["order_draft"]["recommended_action"],
                "self_critique": output["self_critique"]["critique_result"],
                "final_recommendation": output["self_critique"]["final_recommendation"],
                "harness_status": "not_run_yet",
            }
            conn.execute(
                """
                UPDATE decision_cards
                SET agent_summary = ?
                WHERE date = ? AND sku_id = ?
                """,
                (
                    json.dumps(summary, ensure_ascii=False),
                    output["date"],
                    output["sku_id"],
                ),
            )


def write_agent_outputs(
    output_dir: Path,
    outputs: list[dict[str, Any]],
    draft_rows: list[dict[str, Any]],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "agent_outputs.json"
    csv_path = output_dir / "order_drafts_from_agents.csv"
    md_path = output_dir / "agent_summary.md"
    json_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(draft_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    lines = ["# Agent Summary", ""]
    for output in outputs:
        lines.extend(
            [
                f"## {output['date']} / {output['sku_id']}",
                f"- Causal: {output['causal_context']['summary']}",
                f"- Vision issue: {output['vision_grounded']['possible_issue']}",
                f"- Draft action: {output['order_draft']['recommended_action']} qty={output['order_draft']['suggested_qty']}",
                f"- Critique: {output['self_critique']['critique_result']} -> {output['self_critique']['final_recommendation']}",
                f"- Draft status: {output['order_draft_status']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return [json_path, csv_path, md_path]
