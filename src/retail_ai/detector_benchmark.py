from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from retail_ai.vision_counting import (
    DEFAULT_PRODUCT_PROMPTS,
    BaseDetector,
    DetectionBox,
    OpenVocabularyProductDetector,
)


DEFAULT_OUTPUT_DIR = Path("results/detector_benchmark")
SUPPORTED_DETECTORS = {"yolo", "owlvit", "yolo-world", "groundingdino", "florence2", "grounded-sam"}


@dataclass(frozen=True)
class BenchmarkRunSummary:
    summary_csv: Path
    summary_json: Path
    recommended_detector: dict[str, Any] | None
    visualization_files: list[Path]
    rows: list[dict[str, Any]]


class YOLOProductDetector(BaseDetector):
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.05,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._model = None

    def detect(self, image_path: str | Path) -> list[DetectionBox]:
        self._load_model()
        results = self._model.predict(str(image_path), conf=self.confidence_threshold, verbose=False)
        detections: list[DetectionBox] = []
        names = getattr(self._model, "names", {})
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].detach().cpu().tolist()]
                confidence = float(box.conf[0].detach().cpu().item())
                class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
                label = str(names.get(class_id, class_id))
                detections.append(
                    DetectionBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence,
                        label=label,
                        detector_name="yolo",
                    )
                )
        return detections

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO detector requires `ultralytics`. Install it with `pip install ultralytics`.") from exc
        self._model = YOLO(self.model_name)


class UnavailableDetector(BaseDetector):
    def __init__(self, detector_name: str, reason: str) -> None:
        self.detector_name = detector_name
        self.reason = reason

    def detect(self, image_path: str | Path) -> list[DetectionBox]:
        raise RuntimeError(self.reason)


def build_detector(
    detector_name: str,
    confidence_threshold: float = 0.05,
    yolo_model: str = "yolov8n.pt",
    prompts: list[str] | None = None,
) -> BaseDetector:
    normalized = normalize_detector_name(detector_name)
    if normalized == "yolo":
        return YOLOProductDetector(model_name=yolo_model, confidence_threshold=confidence_threshold)
    if normalized == "owlvit":
        return OpenVocabularyProductDetector(
            prompts=prompts or DEFAULT_PRODUCT_PROMPTS,
            confidence_threshold=confidence_threshold,
        )
    return UnavailableDetector(
        detector_name=normalized,
        reason=(
            f"{normalized} is registered as a future detector candidate, but its runtime integration "
            "is not installed in this project yet."
        ),
    )


def normalize_detector_name(detector_name: str) -> str:
    name = detector_name.strip().lower()
    aliases = {
        "owl-vit": "owlvit",
        "owl_vit": "owlvit",
        "yoloworld": "yolo-world",
        "grounding-dino": "groundingdino",
        "florence-2": "florence2",
        "groundedsam": "grounded-sam",
    }
    return aliases.get(name, name)


def run_detector_benchmark(
    images: list[Path],
    detectors: list[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    confidence_thresholds: list[float] | None = None,
    yolo_model: str = "yolov8n.pt",
    prompts: list[str] | None = None,
) -> BenchmarkRunSummary:
    thresholds = confidence_thresholds or [0.05, 0.1, 0.2, 0.3]
    prompts = prompts or DEFAULT_PRODUCT_PROMPTS
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    visualization_files: list[Path] = []
    for image_path in images:
        for detector_name in detectors:
            normalized_detector = normalize_detector_name(detector_name)
            for threshold in thresholds:
                row = benchmark_one(
                    image_path=Path(image_path),
                    detector_name=normalized_detector,
                    threshold=float(threshold),
                    output_dir=output_dir,
                    yolo_model=yolo_model,
                    prompts=prompts,
                )
                rows.append(row)
                if row.get("output_image_path"):
                    visualization_files.append(Path(row["output_image_path"]))

    frame = pd.DataFrame(rows)
    summary_csv = output_dir / "benchmark_summary.csv"
    summary_json = output_dir / "benchmark_summary.json"
    frame.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    recommended = recommend_detector(rows)
    summary_payload = {
        "recommended_detector": recommended,
        "rows": rows,
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return BenchmarkRunSummary(
        summary_csv=summary_csv,
        summary_json=summary_json,
        recommended_detector=recommended,
        visualization_files=visualization_files,
        rows=rows,
    )


def benchmark_one(
    image_path: Path,
    detector_name: str,
    threshold: float,
    output_dir: Path,
    yolo_model: str,
    prompts: list[str],
) -> dict[str, Any]:
    safe_stem = safe_name(image_path.stem)
    base = f"{safe_stem}_{detector_name}_conf{threshold:g}".replace(".", "p")
    output_image_path = output_dir / f"{base}.jpg"
    output_json_path = output_dir / f"{base}.json"
    detections: list[DetectionBox] = []
    error_message = ""
    try:
        detector = build_detector(
            detector_name,
            confidence_threshold=threshold,
            yolo_model=yolo_model,
            prompts=prompts,
        )
        detections = detector.detect(image_path)
    except Exception as exc:
        error_message = str(exc)

    save_detection_json(output_json_path, image_path, detector_name, threshold, detections, error_message)
    save_detection_visualization(output_image_path, image_path, detections, detector_name, threshold, error_message)
    stats = detection_stats(detections)
    return {
        "image_path": str(image_path),
        "detector_name": detector_name,
        "confidence_threshold": threshold,
        "detection_count": len(detections),
        "avg_confidence": stats["avg_confidence"],
        "min_box_area": stats["min_box_area"],
        "max_box_area": stats["max_box_area"],
        "output_image_path": str(output_image_path),
        "output_json_path": str(output_json_path),
        "error_message": error_message,
    }


def save_detection_json(
    path: Path,
    image_path: Path,
    detector_name: str,
    threshold: float,
    detections: list[DetectionBox],
    error_message: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": str(image_path),
        "detector_name": detector_name,
        "confidence_threshold": threshold,
        "detection_count": len(detections),
        "detections": [asdict(detection) for detection in detections],
        "error_message": error_message,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_detection_visualization(
    path: Path,
    image_path: Path,
    detections: list[DetectionBox],
    detector_name: str,
    threshold: float,
    error_message: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_path.exists():
        with Image.open(image_path) as image:
            canvas = image.convert("RGB")
    else:
        canvas = Image.new("RGB", (900, 500), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    header = f"{detector_name} conf={threshold:g} detections={len(detections)}"
    draw.rectangle((8, 8, min(canvas.width - 8, 600), 34), fill=(255, 255, 255), outline=(80, 80, 80))
    draw.text((16, 14), header, fill=(0, 0, 0), font=font)
    for index, detection in enumerate(detections, start=1):
        label = detection.label or "object"
        text = f"{index} {label} {detection.confidence:.2f}"
        draw.rectangle((detection.x1, detection.y1, detection.x2, detection.y2), outline=(255, 0, 0), width=3)
        draw.rectangle((detection.x1, max(0, detection.y1 - 18), detection.x1 + min(420, max(140, 8 * len(text))), detection.y1), fill=(255, 0, 0))
        draw.text((detection.x1 + 3, max(0, detection.y1 - 16)), text, fill=(255, 255, 255), font=font)
    if error_message:
        draw.rectangle((8, 40, min(canvas.width - 8, 860), 90), fill=(255, 245, 245), outline=(180, 0, 0))
        draw.text((16, 50), error_message[:180], fill=(180, 0, 0), font=font)
    canvas.save(path)


def detection_stats(detections: list[DetectionBox]) -> dict[str, float]:
    if not detections:
        return {"avg_confidence": 0.0, "min_box_area": 0.0, "max_box_area": 0.0}
    confidences = [float(detection.confidence) for detection in detections]
    areas = [box_area(detection) for detection in detections]
    return {
        "avg_confidence": float(sum(confidences) / len(confidences)),
        "min_box_area": float(min(areas)),
        "max_box_area": float(max(areas)),
    }


def box_area(detection: DetectionBox) -> float:
    return max(0.0, float(detection.x2) - float(detection.x1)) * max(0.0, float(detection.y2) - float(detection.y1))


def recommend_detector(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid_rows = [
        row
        for row in rows
        if not row.get("error_message")
        and int(row.get("detection_count", 0)) > 0
        and 5 <= int(row.get("detection_count", 0)) <= 80
    ]
    if not valid_rows:
        return None
    frame = pd.DataFrame(valid_rows)
    grouped = (
        frame.groupby(["detector_name", "confidence_threshold"], as_index=False)
        .agg(
            image_count=("image_path", "nunique"),
            total_detections=("detection_count", "sum"),
            avg_detection_count=("detection_count", "mean"),
            avg_confidence=("avg_confidence", "mean"),
        )
    )
    grouped["score"] = grouped["avg_confidence"] * 100.0 + grouped["avg_detection_count"].clip(upper=30)
    best = grouped.sort_values(["image_count", "score"], ascending=[False, False]).iloc[0]
    return {
        "detector_name": str(best["detector_name"]),
        "confidence_threshold": float(best["confidence_threshold"]),
        "image_count": int(best["image_count"]),
        "total_detections": int(best["total_detections"]),
        "avg_detection_count": float(best["avg_detection_count"]),
        "avg_confidence": float(best["avg_confidence"]),
        "reason": "Selected among non-empty detectors with 5-80 boxes and highest confidence-weighted score.",
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
