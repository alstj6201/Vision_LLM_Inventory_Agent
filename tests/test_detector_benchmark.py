from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.detector_benchmark import (  # noqa: E402
    BenchmarkRunSummary,
    recommend_detector,
    run_detector_benchmark,
)
from retail_ai.vision_counting import DetectionBox, MockDetector  # noqa: E402


def test_detection_results_are_saved_in_standard_format(tmp_path: Path, monkeypatch):
    image_path = make_image(tmp_path / "shelf.png")

    def fake_build_detector(detector_name, confidence_threshold, yolo_model="yolov8n.pt", prompts=None):
        return MockDetector(
            [
                DetectionBox(
                    1,
                    2,
                    30,
                    40,
                    0.8,
                    label="package",
                    detector_name=detector_name,
                )
            ]
        )

    monkeypatch.setattr("retail_ai.detector_benchmark.build_detector", fake_build_detector)
    summary = run_detector_benchmark(
        images=[image_path],
        detectors=["mock"],
        output_dir=tmp_path / "out",
        confidence_thresholds=[0.1],
    )

    assert summary.summary_csv.exists()
    assert summary.summary_json.exists()
    row = pd.read_csv(summary.summary_csv).iloc[0]
    assert row["detection_count"] == 1
    payload = json.loads(Path(row["output_json_path"]).read_text(encoding="utf-8"))
    assert payload["detections"][0]["label"] == "package"
    assert payload["detections"][0]["detector_name"] == "mock"


def test_detector_failure_records_error_message(tmp_path: Path, monkeypatch):
    image_path = make_image(tmp_path / "shelf.png")

    class FailingDetector:
        def detect(self, image_path):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr("retail_ai.detector_benchmark.build_detector", lambda *args, **kwargs: FailingDetector())
    summary = run_detector_benchmark(
        images=[image_path],
        detectors=["broken"],
        output_dir=tmp_path / "out",
        confidence_thresholds=[0.1],
    )
    row = pd.read_csv(summary.summary_csv).iloc[0]
    assert row["detection_count"] == 0
    assert "model unavailable" in row["error_message"]


def test_recommended_detector_prefers_valid_high_confidence_candidate():
    rows = [
        {
            "image_path": "a.png",
            "detector_name": "bad",
            "confidence_threshold": 0.1,
            "detection_count": 0,
            "avg_confidence": 0.0,
            "error_message": "",
        },
        {
            "image_path": "a.png",
            "detector_name": "owlvit",
            "confidence_threshold": 0.05,
            "detection_count": 8,
            "avg_confidence": 0.2,
            "error_message": "",
        },
        {
            "image_path": "a.png",
            "detector_name": "yolo",
            "confidence_threshold": 0.05,
            "detection_count": 7,
            "avg_confidence": 0.6,
            "error_message": "",
        },
    ]
    recommended = recommend_detector(rows)
    assert recommended is not None
    assert recommended["detector_name"] == "yolo"


def test_benchmark_summary_csv_is_created(tmp_path: Path, monkeypatch):
    image_path = make_image(tmp_path / "shelf.png")
    monkeypatch.setattr("retail_ai.detector_benchmark.build_detector", lambda *args, **kwargs: MockDetector([]))
    summary = run_detector_benchmark(
        images=[image_path],
        detectors=["mock"],
        output_dir=tmp_path / "out",
        confidence_thresholds=[0.05, 0.1],
    )
    frame = pd.read_csv(summary.summary_csv)
    assert len(frame) == 2
    assert set(frame["confidence_threshold"]) == {0.05, 0.1}


def make_image(path: Path) -> Path:
    Image.new("RGB", (100, 80), color=(240, 240, 240)).save(path)
    return path
