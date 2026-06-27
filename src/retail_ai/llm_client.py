from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol


class LLMClient(Protocol):
    def complete_json(self, agent_name: str, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        ...


class OpenAIJSONClient:
    def __init__(self, model: str | None = None, env_path: str | Path = ".env") -> None:
        try:
            from dotenv import load_dotenv
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI provider requires `openai` and `python-dotenv`. "
                "Install them with `pip install openai python-dotenv`."
            ) from exc

        load_dotenv(env_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or run with --dry-run.")

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key)

    def complete_json(self, agent_name: str, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                        + "\nReturn only valid JSON. Do not call tools or external APIs.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError(f"{agent_name} returned an empty response.")
            return json.loads(content)
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI API call failed for {agent_name}. Check network access, API key, "
                "model availability, or run with --dry-run."
            ) from exc


class RuleBasedLLMClient:
    """Deterministic local client for tests and --dry-run."""

    def complete_json(self, agent_name: str, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        sku_id = str(user_payload["sku"]["sku_id"])
        anomaly = user_payload.get("anomaly", {})
        context = user_payload.get("context", {})
        severity = float(anomaly.get("severity", 0.0))
        anomaly_type = str(anomaly.get("anomaly_type", "normal"))
        shrinkage = float(anomaly.get("shrinkage_score", 0.0))
        confidence = float(context.get("count_confidence", 1.0))

        if agent_name == "CausalContextAgent":
            causes = []
            if anomaly_type == "demand_spike":
                causes.append("recent demand spike")
            if shrinkage > 0.3:
                causes.append("physical stock lower than expected")
            if confidence < 0.85:
                causes.append("low CV count confidence")
            if not causes:
                causes.append("triage severity exceeded review threshold")
            return {
                "agent_name": agent_name,
                "sku_id": sku_id,
                "likely_causes": causes,
                "similar_cases": [case.get("case_id", "synthetic_case") for case in user_payload.get("similar_cases", [])[:3]],
                "evidence": [
                    f"severity={severity:.3f}",
                    f"anomaly_type={anomaly_type}",
                    f"actual_sales={context.get('actual_sales')}",
                    f"forecast_quantity={context.get('forecast_quantity')}",
                ],
                "confidence": min(0.95, 0.55 + severity),
                "summary": "Rule-based causal summary for triage exception.",
            }

        if agent_name == "VisionGroundedAgent":
            if confidence < 0.85 and shrinkage > 0.3:
                possible_issue = "cv_error"
            elif anomaly_type == "theft_suspected":
                possible_issue = "theft_suspected"
            elif shrinkage > 0.3:
                possible_issue = "shrinkage"
            else:
                possible_issue = "unknown"
            return {
                "agent_name": agent_name,
                "sku_id": sku_id,
                "visual_assessment": "Metadata-based assessment; no raw CCTV image was provided.",
                "possible_issue": possible_issue,
                "evidence": [
                    f"cv_count={context.get('cv_count')}",
                    f"expected_stock={context.get('expected_stock')}",
                    f"count_confidence={confidence:.3f}",
                ],
                "confidence": min(0.95, max(0.3, confidence)),
            }

        if agent_name == "OrderDraftingAgent":
            reorder_qty = int(user_payload.get("sku", {}).get("reorder_quantity", 0) or 0)
            if anomaly_type == "theft_suspected" or severity >= 0.7:
                action = "block"
                qty = 0
                uncertainty = "high"
            elif severity >= 0.3:
                action = "review"
                qty = reorder_qty
                uncertainty = "medium"
            else:
                action = "order"
                qty = reorder_qty
                uncertainty = "low"
            return {
                "agent_name": agent_name,
                "sku_id": sku_id,
                "recommended_action": action,
                "suggested_qty": qty,
                "reasoning": "Draft only; deterministic harness must validate before any execution.",
                "evidence": [
                    f"severity={severity:.3f}",
                    f"route={anomaly.get('status')}",
                    f"reorder_quantity={reorder_qty}",
                ],
                "uncertainty": uncertainty,
            }

        if agent_name == "SelfCritiqueAgent":
            draft = user_payload.get("order_draft", {})
            action = draft.get("recommended_action")
            if action == "block":
                result = "block"
                final = "freeze"
                issues = ["Draft recommends blocking automatic order."]
            elif action == "review":
                result = "pass"
                final = "require_human_review"
                issues = []
            else:
                result = "pass"
                final = "send_to_harness"
                issues = []
            return {
                "agent_name": agent_name,
                "sku_id": sku_id,
                "critique_result": result,
                "issues": issues,
                "final_recommendation": final,
                "summary": "Rule-based critique completed; no write action was executed.",
            }

        raise ValueError(f"Unknown agent_name: {agent_name}")
