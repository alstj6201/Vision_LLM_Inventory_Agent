from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import transformers
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.vision_counting import OpenVocabularyProductDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test OpenVocabularyProductDetector on one image.")
    parser.add_argument("--image-path", default=PROJECT_ROOT / "data" / "simulation" / "Morning.png", type=Path)
    parser.add_argument("--output-json", default=PROJECT_ROOT / "results" / "detection_preview.json", type=Path)
    parser.add_argument("--output-image", default=PROJECT_ROOT / "results" / "detection_preview.jpg", type=Path)
    parser.add_argument("--det-conf", default=0.05, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = OpenVocabularyProductDetector(confidence_threshold=args.det_conf)
    detections = detector.detect(args.image_path)

    result = {
        "image_path": str(args.image_path),
        "detector": "open-vocab",
        "model_name": detector.model_name,
        "transformers_version": transformers.__version__,
        "post_process_method": detector.post_process_method,
        "prompts": detector.prompts,
        "detection_count": len(detections),
        "detections": [
            {
                "x1": detection.x1,
                "y1": detection.y1,
                "x2": detection.x2,
                "y2": detection.y2,
                "label": detection.label,
                "confidence": detection.confidence,
            }
            for detection in detections
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    save_visualization(args.image_path, detections, args.output_image)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"output_json: {args.output_json}")
    print(f"output_image: {args.output_image}")


def save_visualization(image_path: Path, detections, output_image: Path) -> None:
    output_image.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, detection in enumerate(detections, start=1):
        draw.rectangle(
            [detection.x1, detection.y1, detection.x2, detection.y2],
            outline=(255, 0, 0),
            width=3,
        )
        label_text = detection.label or "product"
        label = f"{index}: {label_text} {detection.confidence:.2f}"
        label_width = max(120, min(420, 8 * len(label)))
        draw.rectangle(
            [detection.x1, max(0, detection.y1 - 16), detection.x1 + label_width, detection.y1],
            fill=(255, 0, 0),
        )
        draw.text((detection.x1 + 3, max(0, detection.y1 - 15)), label, fill=(255, 255, 255), font=font)

    if not detections:
        draw.text((10, 10), "No open-vocab detections", fill=(255, 0, 0), font=font)

    canvas.save(output_image)


if __name__ == "__main__":
    main()
