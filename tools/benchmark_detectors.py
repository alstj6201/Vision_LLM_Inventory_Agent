from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.detector_benchmark import DEFAULT_OUTPUT_DIR, DEFAULT_PRODUCT_PROMPTS, run_detector_benchmark  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark product candidate detectors on CCTV shelf images.")
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--detectors", nargs="+", default=["yolo", "owlvit"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conf-thresholds", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PRODUCT_PROMPTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_detector_benchmark(
        images=args.images,
        detectors=args.detectors,
        output_dir=args.output_dir,
        confidence_thresholds=args.conf_thresholds,
        yolo_model=args.yolo_model,
        prompts=args.prompts,
    )
    counts = [
        {
            "image_path": row["image_path"],
            "detector_name": row["detector_name"],
            "confidence_threshold": row["confidence_threshold"],
            "detection_count": row["detection_count"],
            "error": bool(row["error_message"]),
            "error_message": row["error_message"],
        }
        for row in summary.rows
    ]
    print(
        json.dumps(
            {
                "summary_csv": str(summary.summary_csv),
                "summary_json": str(summary.summary_json),
                "recommended_detector": summary.recommended_detector,
                "detector_counts": counts,
                "visualization_files": [str(path) for path in summary.visualization_files],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
