from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from PIL import Image, ImageDraw

from retail_ai.agents import run_cognition_layer
from retail_ai.detector_benchmark import build_detector
from retail_ai.harness import run_harness
from retail_ai.llm_client import OpenAIJSONClient, RuleBasedLLMClient
from retail_ai.optimizer import run_optimizer
from retail_ai.severity import compute_shrinkage_score, run_triage
from retail_ai.vision_counting import (
    DINOEmbeddingExtractor,
    FAISSSkuRetriever,
    OpenVocabularyProductDetector,
    VisionCountingPipeline,
)


DEFAULT_DB_PATH = Path("synthetic_retail_company_dataset/retail_inventory.sqlite")
DEFAULT_OUTPUT_DIR = Path("results/demo")
DEFAULT_MORNING_IMAGE = Path("data/simulation/Morning.png")
DEFAULT_EVENING_IMAGE = Path("data/simulation/Evening.png")
DemoMode = Literal["production", "simulation"]


@dataclass(frozen=True)
class DemoSummary:
    date: str
    mode: str
    processed_sku: int
    morning_count: int
    evening_count: int
    pos_sales: int
    total_shrinkage: float
    alert_count: int
    review_count: int
    freeze_count: int
    optimized_orders: int
    execution_time: float
    llm_dry_run_fallback: bool
    generated_files: list[Path]


def run_end_to_end_demo(
    date: str = "2026-05-21",
    mode: DemoMode = "simulation",
    morning_image: Path = DEFAULT_MORNING_IMAGE,
    evening_image: Path = DEFAULT_EVENING_IMAGE,
    sqlite_db: Path = DEFAULT_DB_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force_agent_dry_run: bool = False,
    detector: str = "auto",
) -> DemoSummary:
    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    vision_dir = output_dir / "vision"
    figures_dir = output_dir / "figures"
    vision_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    detector_config = resolve_demo_detector(detector)
    morning_vision = run_demo_vision_count(morning_image, vision_dir, "morning", detector_config)
    evening_vision = run_demo_vision_count(evening_image, vision_dir, "evening", detector_config)
    vision_files = write_vision_outputs(vision_dir, morning_vision, evening_vision)

    ensure_forecast_for_date(sqlite_db, date)
    demand_frame = load_demand_context(sqlite_db, date)
    shrinkage_frame = build_shrinkage_frame(demand_frame, morning_vision, evening_vision)

    triage_summary = run_triage(sqlite_db=sqlite_db, output_dir=output_dir / "triage", date=date)
    agent_summary, llm_fallback = run_agents_with_fallback(
        sqlite_db=sqlite_db,
        output_dir=output_dir / "agents",
        date=date,
        force_dry_run=force_agent_dry_run,
    )
    harness_summary = run_harness(sqlite_db=sqlite_db, output_dir=output_dir / "harness", date=date)
    optimizer_summary = run_optimizer(
        sqlite_db=sqlite_db,
        output_dir=output_dir / "optimizer",
        date=date,
        mode=mode,
    )

    decision_cards = build_decision_cards(sqlite_db, date, shrinkage_frame)
    alerts = build_alerts(decision_cards, date)
    files = []
    files.extend(vision_files)
    decision_path = output_dir / "decision_cards_demo.csv"
    alerts_path = output_dir / "alerts.csv"
    decision_cards.to_csv(decision_path, index=False, encoding="utf-8-sig")
    alerts.to_csv(alerts_path, index=False, encoding="utf-8-sig")
    files.extend([decision_path, alerts_path])
    figure_paths = create_demo_figures(figures_dir, decision_cards, alerts)
    files.extend(figure_paths)

    execution_time = time.perf_counter() - start
    summary = DemoSummary(
        date=date,
        mode=mode,
        processed_sku=int(decision_cards["SKU"].nunique()) if len(decision_cards) else int(demand_frame["sku_id"].nunique()),
        morning_count=int(sum(morning_vision["sku_counts"].values())),
        evening_count=int(sum(evening_vision["sku_counts"].values())),
        pos_sales=int(demand_frame["actual_sales"].sum()) if len(demand_frame) else 0,
        total_shrinkage=float(decision_cards["Shrinkage"].sum()) if len(decision_cards) else 0.0,
        alert_count=int(len(alerts)),
        review_count=int((decision_cards["Final Decision"] == "requires_manual_review").sum()) if len(decision_cards) else triage_summary.requires_review_count,
        freeze_count=int((decision_cards["Final Decision"] == "blocked").sum()) if len(decision_cards) else triage_summary.freeze_and_alert_count,
        optimized_orders=int((decision_cards["Optimized Qty"] > 0).sum()) if len(decision_cards) else optimizer_summary.optimized_sku_count,
        execution_time=execution_time,
        llm_dry_run_fallback=llm_fallback,
        generated_files=[],
    )
    dashboard_path = write_dashboard(output_dir, summary, morning_image, evening_image, decision_cards, alerts)
    report_path = write_demo_report(output_dir, summary, morning_vision, evening_vision, decision_cards)
    all_files = files + [dashboard_path, report_path]
    final_summary = DemoSummary(**{**summary.__dict__, "generated_files": all_files})
    summary_files = write_demo_summary(output_dir, final_summary, morning_vision, evening_vision, decision_cards, alerts)
    return DemoSummary(**{**final_summary.__dict__, "generated_files": all_files + summary_files})


def resolve_demo_detector(detector: str) -> dict[str, Any]:
    requested = detector.strip().lower()
    if requested != "auto":
        return {"detector_name": requested, "confidence_threshold": 0.05, "source": "cli"}
    summary_path = Path("results/detector_benchmark/benchmark_summary.json")
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            recommended = payload.get("recommended_detector")
            if recommended and recommended.get("detector_name"):
                return {
                    "detector_name": recommended["detector_name"],
                    "confidence_threshold": float(recommended.get("confidence_threshold", 0.05)),
                    "source": str(summary_path),
                }
        except Exception:
            pass
    return {"detector_name": "owlvit", "confidence_threshold": 0.05, "source": "fallback"}


def run_demo_vision_count(
    image_path: Path,
    vision_dir: Path,
    prefix: str,
    detector_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    crops_dir = vision_dir / f"{prefix}_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    detection_path = vision_dir / f"{prefix}_detection.jpg"
    result: dict[str, Any] = {
        "image_path": str(image_path),
        "status": "not_run",
        "error": "",
        "total_detections": 0,
        "sku_counts": {},
        "predictions": [],
        "low_confidence_predictions": [],
        "detection_image": str(detection_path),
        "crops_dir": str(crops_dir),
        "detector_config": detector_config or {},
    }
    if not image_path.exists():
        result["status"] = "missing_image"
        result["error"] = f"Image not found: {image_path}"
        create_placeholder_detection(image_path, detection_path, result["error"])
        return result
    try:
        detector_config = detector_config or {"detector_name": "owlvit", "confidence_threshold": 0.05}
        detector = build_detector(
            detector_config.get("detector_name", "owlvit"),
            confidence_threshold=float(detector_config.get("confidence_threshold", 0.05)),
        )
        embedding_extractor = DINOEmbeddingExtractor()
        retriever = FAISSSkuRetriever()
        pipeline = VisionCountingPipeline(detector, embedding_extractor, retriever)
        count_result = pipeline.count(image_path)
        result.update(count_result.to_dict())
        result["status"] = "success"
        save_detection_artifacts(image_path, result["predictions"], detection_path, crops_dir)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        create_placeholder_detection(image_path, detection_path, f"Vision pipeline unavailable: {exc}")
    return result


def save_detection_artifacts(image_path: Path, predictions: list[dict[str, Any]], detection_path: Path, crops_dir: Path) -> None:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        draw = ImageDraw.Draw(rgb)
        for idx, prediction in enumerate(predictions, start=1):
            bbox = prediction["bbox"]
            box = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
            draw.rectangle(box, outline="red", width=4)
            draw.text((bbox["x1"], max(0, bbox["y1"] - 14)), f"{prediction['sku_id']} {prediction['similarity']:.2f}", fill="red")
            crop = rgb.crop((int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])))
            crop.save(crops_dir / f"crop_{idx:03d}_{prediction['sku_id']}.jpg")
        rgb.save(detection_path)


def create_placeholder_detection(image_path: Path, detection_path: Path, message: str) -> None:
    if image_path.exists():
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
    else:
        rgb = Image.new("RGB", (900, 500), color=(245, 245, 245))
    draw = ImageDraw.Draw(rgb)
    draw.rectangle((10, 10, min(rgb.width - 10, 880), 80), fill=(255, 255, 255), outline=(180, 180, 180))
    draw.text((20, 30), message[:160], fill=(180, 0, 0))
    rgb.save(detection_path)


def write_vision_outputs(vision_dir: Path, morning: dict[str, Any], evening: dict[str, Any]) -> list[Path]:
    retrieval_rows = []
    count_rows = []
    for label, result in [("morning", morning), ("evening", evening)]:
        for prediction in result.get("predictions", []):
            for candidate in prediction.get("topk_candidates", []):
                retrieval_rows.append({"snapshot": label, "query_image": result["image_path"], **candidate})
        for sku_id, count in result.get("sku_counts", {}).items():
            count_rows.append({"snapshot": label, "sku_id": sku_id, "count": count, "status": result["status"]})
        if not result.get("sku_counts"):
            count_rows.append({"snapshot": label, "sku_id": "", "count": 0, "status": result["status"], "error": result.get("error", "")})
    retrieval_path = vision_dir / "retrieval_results.csv"
    count_path = vision_dir / "count_results.csv"
    pd.DataFrame(retrieval_rows).to_csv(retrieval_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(count_rows).to_csv(count_path, index=False, encoding="utf-8-sig")
    return [
        Path(morning["detection_image"]),
        Path(evening["detection_image"]),
        retrieval_path,
        count_path,
    ]


def ensure_forecast_for_date(sqlite_db: Path, date: str) -> None:
    with sqlite3.connect(sqlite_db) as conn:
        forecast_count = conn.execute("SELECT COUNT(*) FROM demand_forecasts WHERE date = ?", (date,)).fetchone()[0]
        if int(forecast_count) == 0:
            raise ValueError(
                f"No demand_forecasts rows found for {date}. Run tools/train_demand_forecast.py before the demo."
            )


def load_demand_context(sqlite_db: Path, date: str) -> pd.DataFrame:
    with sqlite3.connect(sqlite_db) as conn:
        frame = pd.read_sql_query(
            """
            SELECT
                inv.date,
                inv.sku_id,
                sm.product_name,
                inv.units_sold AS actual_sales,
                inv.expected_stock,
                inv.closing_stock,
                df.forecast_quantity,
                ac.demand_anomaly_score,
                ac.shrinkage_score,
                ac.severity
            FROM inventory_snapshot inv
            JOIN sku_master sm ON sm.sku_id = inv.sku_id
            LEFT JOIN demand_forecasts df ON df.date = inv.date AND df.sku_id = inv.sku_id
            LEFT JOIN anomaly_cases ac ON ac.date = inv.date AND ac.sku_id = inv.sku_id
            WHERE inv.date = ?
            ORDER BY inv.sku_id
            """,
            conn,
            params=[date],
        )
    if frame.empty:
        raise ValueError(f"No inventory rows found for demo date={date}")
    return frame


def build_shrinkage_frame(demand_frame: pd.DataFrame, morning: dict[str, Any], evening: dict[str, Any]) -> pd.DataFrame:
    morning_counts = {str(k): int(v) for k, v in morning.get("sku_counts", {}).items()}
    evening_counts = {str(k): int(v) for k, v in evening.get("sku_counts", {}).items()}
    rows = []
    for row in demand_frame.to_dict(orient="records"):
        sku_id = str(row["sku_id"])
        morning_count = morning_counts.get(sku_id, int(row.get("expected_stock") or 0))
        evening_count = evening_counts.get(sku_id, int(row.get("closing_stock") or 0))
        expected_closing = max(0, morning_count - int(row.get("actual_sales") or 0))
        shrinkage_gap = expected_closing - evening_count
        rows.append(
            {
                **row,
                "morning_count": morning_count,
                "evening_count": evening_count,
                "expected_closing": expected_closing,
                "shrinkage_gap": shrinkage_gap,
                "demo_shrinkage_score": compute_shrinkage_score(expected_closing, evening_count),
            }
        )
    return pd.DataFrame(rows)


def run_agents_with_fallback(
    sqlite_db: Path,
    output_dir: Path,
    date: str,
    force_dry_run: bool = False,
) -> tuple[Any, bool]:
    if force_dry_run:
        return run_cognition_layer(sqlite_db=sqlite_db, output_dir=output_dir, date=date, llm_client=RuleBasedLLMClient()), True
    try:
        return run_cognition_layer(sqlite_db=sqlite_db, output_dir=output_dir, date=date, llm_client=OpenAIJSONClient()), False
    except Exception:
        return run_cognition_layer(sqlite_db=sqlite_db, output_dir=output_dir, date=date, llm_client=RuleBasedLLMClient()), True


def build_decision_cards(sqlite_db: Path, date: str, shrinkage_frame: pd.DataFrame) -> pd.DataFrame:
    with sqlite3.connect(sqlite_db) as conn:
        frame = pd.read_sql_query(
            """
            SELECT
                dc.date,
                dc.sku_id,
                sm.product_name,
                df.forecast_quantity,
                inv.units_sold AS actual_sales,
                ac.demand_anomaly_score,
                ac.shrinkage_score,
                dc.severity,
                dc.agent_summary,
                dc.harness_result,
                dc.harness_reason,
                dc.retry_count,
                dc.final_status,
                dc.optimized_qty,
                dc.optimization_summary
            FROM decision_cards dc
            JOIN sku_master sm ON sm.sku_id = dc.sku_id
            LEFT JOIN demand_forecasts df ON df.date = dc.date AND df.sku_id = dc.sku_id
            LEFT JOIN inventory_snapshot inv ON inv.date = dc.date AND inv.sku_id = dc.sku_id
            LEFT JOIN anomaly_cases ac ON ac.date = dc.date AND ac.sku_id = dc.sku_id
            WHERE dc.date = ?
            ORDER BY dc.severity DESC, dc.sku_id
            """,
            conn,
            params=[date],
        )
    merged = frame.merge(
        shrinkage_frame[
            [
                "sku_id",
                "morning_count",
                "actual_sales",
                "expected_closing",
                "evening_count",
                "shrinkage_gap",
                "demo_shrinkage_score",
            ]
        ],
        on="sku_id",
        how="left",
        suffixes=("", "_demo"),
    )
    if "actual_sales_demo" in merged:
        merged["actual_sales"] = merged["actual_sales"].fillna(merged["actual_sales_demo"])
    return pd.DataFrame(
        {
            "SKU": merged["sku_id"],
            "Product Name": merged["product_name"],
            "Forecast": merged["forecast_quantity"].fillna(0),
            "Morning Count": merged["morning_count"].fillna(0),
            "POS Sales": merged["actual_sales"].fillna(0),
            "Expected Closing": merged["expected_closing"].fillna(0),
            "Evening Count": merged["evening_count"].fillna(0),
            "Shrinkage": merged["shrinkage_gap"].fillna(0),
            "Demand Anomaly Score": merged["demand_anomaly_score"].fillna(0),
            "Shrinkage Score": merged["shrinkage_score"].fillna(merged["demo_shrinkage_score"]).fillna(0),
            "Severity": merged["severity"].fillna(0),
            "Agent Summary": merged["agent_summary"].fillna(""),
            "Harness Result": merged["harness_result"].fillna("not_run"),
            "Harness Reason": merged["harness_reason"].fillna(""),
            "Retry Count": merged["retry_count"].fillna(0).astype(int),
            "Optimized Qty": merged["optimized_qty"].fillna(0).astype(int),
            "Final Decision": merged["final_status"].fillna(merged["harness_result"]).fillna("not_run"),
        }
    )


def build_alerts(decision_cards: pd.DataFrame, date: str) -> pd.DataFrame:
    rows = []
    for idx, row in enumerate(decision_cards.to_dict(orient="records"), start=1):
        if float(row["Severity"]) < 0.3 and row["Final Decision"] not in {"blocked", "requires_manual_review"}:
            continue
        alert_type = "freeze" if row["Final Decision"] == "blocked" else "review"
        rows.append(
            {
                "alert_id": f"DEMO_ALERT_{date.replace('-', '')}_{idx:04d}",
                "date": date,
                "sku_id": row["SKU"],
                "product_name": row["Product Name"],
                "severity": row["Severity"],
                "alert_type": alert_type,
                "message": f"{alert_type}: severity={float(row['Severity']):.3f}, shrinkage={float(row['Shrinkage']):.1f}",
                "status": "open" if alert_type == "review" else "frozen",
            }
        )
    return pd.DataFrame(rows)


def create_demo_figures(figures_dir: Path, decision_cards: pd.DataFrame, alerts: pd.DataFrame) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    pipeline_path = figures_dir / "pipeline_flow.png"
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    steps = ["Forecast", "Vision", "Severity", "Agents", "Harness", "MILP", "Decision Card"]
    for idx, step in enumerate(steps):
        x = idx / max(len(steps) - 1, 1)
        ax.text(x, 0.55, step, ha="center", va="center", bbox={"boxstyle": "round,pad=0.4", "fc": "#eef2ff", "ec": "#475569"})
        if idx < len(steps) - 1:
            ax.annotate("", xy=((idx + 0.74) / (len(steps) - 1), 0.55), xytext=((idx + 0.26) / (len(steps) - 1), 0.55), arrowprops={"arrowstyle": "->"})
    fig.tight_layout()
    fig.savefig(pipeline_path, dpi=160)
    plt.close(fig)
    paths.append(pipeline_path)

    if decision_cards.empty:
        decision_cards = pd.DataFrame({"SKU": [], "Morning Count": [], "Evening Count": [], "Expected Closing": [], "Severity": [], "Final Decision": [], "Optimized Qty": [], "Shrinkage": []})

    paths.append(bar_figure(figures_dir / "morning_vs_evening_count.png", decision_cards, "SKU", ["Morning Count", "Evening Count"], "Morning vs Evening Count"))
    paths.append(bar_figure(figures_dir / "expected_vs_actual_stock.png", decision_cards, "SKU", ["Expected Closing", "Evening Count"], "Expected vs Actual Closing Stock"))
    paths.append(hist_figure(figures_dir / "severity_distribution.png", decision_cards["Severity"], "Severity Distribution", "Severity"))
    paths.append(count_figure(figures_dir / "decision_distribution.png", decision_cards["Final Decision"], "Decision Distribution"))
    paths.append(bar_figure(figures_dir / "optimizer_summary.png", decision_cards, "SKU", ["Optimized Qty"], "Optimized Quantity by SKU"))
    alert_series = alerts["alert_type"] if not alerts.empty else pd.Series(dtype=str)
    paths.append(count_figure(figures_dir / "alert_distribution.png", alert_series, "Alert Distribution"))
    return paths


def bar_figure(path: Path, frame: pd.DataFrame, label_col: str, value_cols: list[str], title: str) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = frame[label_col].astype(str).tolist() if label_col in frame else []
    x = list(range(len(labels)))
    width = 0.8 / max(len(value_cols), 1)
    for idx, col in enumerate(value_cols):
        values = frame[col].fillna(0).tolist() if col in frame else []
        offsets = [i + (idx - (len(value_cols) - 1) / 2) * width for i in x]
        ax.bar(offsets, values, width=width, label=col)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_title(title)
    if value_cols:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def hist_figure(path: Path, values: pd.Series, title: str, xlabel: str) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pd.to_numeric(values, errors="coerce").fillna(0), bins=20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def count_figure(path: Path, values: pd.Series, title: str) -> Path:
    import matplotlib.pyplot as plt

    counts = values.fillna("none").astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index.tolist(), counts.values.tolist())
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_demo_summary(
    output_dir: Path,
    summary: DemoSummary,
    morning: dict[str, Any],
    evening: dict[str, Any],
    decision_cards: pd.DataFrame,
    alerts: pd.DataFrame,
) -> list[Path]:
    summary_path = output_dir / "demo_summary.json"
    payload = {
        **summary.__dict__,
        "generated_files": [str(path) for path in summary.generated_files],
        "vision": {"morning": morning, "evening": evening},
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return [summary_path]


def write_dashboard(
    output_dir: Path,
    summary: DemoSummary,
    morning_image: Path,
    evening_image: Path,
    decision_cards: pd.DataFrame,
    alerts: pd.DataFrame,
) -> Path:
    dashboard_path = output_dir / "demo_dashboard.html"
    rows = decision_cards.to_html(index=False, escape=False)
    alerts_html = alerts.to_html(index=False, escape=False) if not alerts.empty else "<p>No alerts.</p>"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Retail AI End-to-End Demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    .kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .tile {{ border: 1px solid #d1d5db; padding: 12px; border-radius: 6px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    section {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>Retail AI End-to-End Demo</h1>
  <div class="kpi">
    <div class="tile"><b>Simulation Date</b><br>{summary.date}</div>
    <div class="tile"><b>Processed SKU</b><br>{summary.processed_sku}</div>
    <div class="tile"><b>Forecast Count</b><br>{summary.processed_sku}</div>
    <div class="tile"><b>Morning Count</b><br>{summary.morning_count}</div>
    <div class="tile"><b>Evening Count</b><br>{summary.evening_count}</div>
    <div class="tile"><b>Total Shrinkage</b><br>{summary.total_shrinkage:.2f}</div>
    <div class="tile"><b>Alert Count</b><br>{summary.alert_count}</div>
    <div class="tile"><b>Review Count</b><br>{summary.review_count}</div>
    <div class="tile"><b>Freeze Count</b><br>{summary.freeze_count}</div>
    <div class="tile"><b>Optimized Orders</b><br>{summary.optimized_orders}</div>
  </div>
  <section><h2>Vision</h2>
    <h3>Morning CCTV</h3><img src="../../{morning_image.as_posix()}" alt="Morning CCTV">
    <h3>Evening CCTV</h3><img src="../../{evening_image.as_posix()}" alt="Evening CCTV">
    <h3>Morning Detection</h3><img src="vision/morning_detection.jpg" alt="Morning detection">
    <h3>Evening Detection</h3><img src="vision/evening_detection.jpg" alt="Evening detection">
  </section>
  <section><h2>Decision Cards</h2>{rows}</section>
  <section><h2>Alerts</h2>{alerts_html}</section>
  <section><h2>Figures</h2>
    <img src="figures/pipeline_flow.png" alt="Pipeline">
    <img src="figures/morning_vs_evening_count.png" alt="Counts">
    <img src="figures/expected_vs_actual_stock.png" alt="Stock">
    <img src="figures/severity_distribution.png" alt="Severity">
    <img src="figures/decision_distribution.png" alt="Decision">
    <img src="figures/optimizer_summary.png" alt="Optimizer">
    <img src="figures/alert_distribution.png" alt="Alerts">
  </section>
</body>
</html>"""
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path


def write_demo_report(
    output_dir: Path,
    summary: DemoSummary,
    morning: dict[str, Any],
    evening: dict[str, Any],
    decision_cards: pd.DataFrame,
) -> Path:
    report_path = output_dir / "demo_report.md"
    lines = [
        "# End-to-End Demo Report",
        "",
        f"- Simulation Date: {summary.date}",
        f"- Mode: {summary.mode}",
        f"- Processed SKU: {summary.processed_sku}",
        f"- Morning Count: {summary.morning_count}",
        f"- Evening Count: {summary.evening_count}",
        f"- POS Sales: {summary.pos_sales}",
        f"- Total Shrinkage: {summary.total_shrinkage:.2f}",
        f"- Alert Count: {summary.alert_count}",
        f"- Review Count: {summary.review_count}",
        f"- Freeze Count: {summary.freeze_count}",
        f"- Optimized Orders: {summary.optimized_orders}",
        f"- LLM Dry-run Fallback: {summary.llm_dry_run_fallback}",
        "",
        "## Vision Status",
        "",
        f"- Morning: {morning['status']} {morning.get('error', '')}",
        f"- Evening: {evening['status']} {evening.get('error', '')}",
        "",
        "## Decision Cards",
        "",
        "```text",
        decision_cards.to_string(index=False) if not decision_cards.empty else "No decision cards.",
        "```",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
