from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.vision_counting import (  # noqa: E402
    DINOEmbeddingExtractor,
    DetectionBox,
    FAISSSkuRetriever,
    OpenVocabularyProductDetector,
    VisionCountingPipeline,
)


RESULTS_DIR = PROJECT_ROOT / "results"
METADATA_PATH = PROJECT_ROOT / "data" / "embeddings" / "metadata.csv"
INDEX_PATH = PROJECT_ROOT / "data" / "embeddings" / "faiss.index"
PRODUCTS_IMAGE_DIR = PROJECT_ROOT / "products_image"


class StaticDetector:
    def __init__(self, boxes: list[DetectionBox]) -> None:
        self.boxes = boxes

    def detect(self, image_path: str | Path) -> list[DetectionBox]:
        return self.boxes


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    query_info = select_query_image()
    query_path = query_info["absolute_path"]

    print("Reference Gallery query image:")
    print(f"- path: {query_path}")
    print(f"- sku_id: {query_info['sku_id']}")
    print(f"- product_name: {query_info['product_name']}")

    detector = OpenVocabularyProductDetector()
    detections = run_detection_test(detector, query_path)

    extractor = DINOEmbeddingExtractor()
    retriever = FAISSSkuRetriever(index_path=INDEX_PATH, metadata_path=METADATA_PATH)

    retrieval_summary = run_embedding_retrieval_test(
        extractor=extractor,
        retriever=retriever,
        query_info=query_info,
    )

    count_result = run_single_image_counting_test(
        detector=detector,
        extractor=extractor,
        retriever=retriever,
        query_path=query_path,
        detections=detections,
    )

    print_final_summary(
        detections=detections,
        retrieval_summary=retrieval_summary,
        count_result=count_result,
    )


def select_query_image() -> dict[str, Any]:
    metadata = pd.read_csv(METADATA_PATH, encoding="utf-8-sig")
    if metadata.empty:
        raise ValueError(f"metadata.csv has no rows: {METADATA_PATH}")

    row = metadata.iloc[0]
    relative_image_path = Path(str(row["image_path"]))
    absolute_path = PRODUCTS_IMAGE_DIR / relative_image_path
    if not absolute_path.exists():
        raise FileNotFoundError(f"Reference image does not exist: {absolute_path}")

    return {
        "absolute_path": absolute_path,
        "relative_path": relative_image_path.as_posix(),
        "filename": str(row["filename"]),
        "sku_id": str(row["sku_id"]),
        "product_name": str(row["product_name"]),
    }


def run_detection_test(detector: OpenVocabularyProductDetector, query_path: Path) -> list[DetectionBox]:
    print("\n[Test 1] Open Vocabulary Product Detection Test")
    detections = detector.detect(query_path)

    print(f"detection_count: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        print(
            f"- #{index}: bbox=({detection.x1:.1f}, {detection.y1:.1f}, "
            f"{detection.x2:.1f}, {detection.y2:.1f}), confidence={detection.confidence:.4f}"
        )

    if not detections:
        print(
            "Open-vocabulary detector found no product candidate boxes. This script uses a reference "
            "gallery image only as a smoke test input; detector quality should be checked on real CCTV "
            "shelf images."
        )

    output_path = RESULTS_DIR / "open_vocab_detection_test.jpg"
    legacy_output_path = RESULTS_DIR / "yolo_detection_test.jpg"
    save_detection_visualization(query_path, detections, output_path)
    save_detection_visualization(query_path, detections, legacy_output_path)
    print(f"visualization_saved: {output_path}")
    print(f"legacy_visualization_saved: {legacy_output_path}")
    return detections


def save_detection_visualization(
    image_path: Path,
    detections: list[DetectionBox],
    output_path: Path,
) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    if detections:
        for index, detection in enumerate(detections, start=1):
            draw.rectangle(
                [detection.x1, detection.y1, detection.x2, detection.y2],
                outline=(255, 0, 0),
                width=3,
            )
            label = f"{index}: {detection.confidence:.2f}"
            draw.rectangle(
                [detection.x1, max(0, detection.y1 - 16), detection.x1 + 78, detection.y1],
                fill=(255, 0, 0),
            )
            draw.text(
                (detection.x1 + 3, max(0, detection.y1 - 15)),
                label,
                fill=(255, 255, 255),
                font=ImageFont.load_default(),
            )
    else:
        draw.text(
            (10, 10),
            "No YOLO detections",
            fill=(255, 0, 0),
            font=ImageFont.load_default(),
        )

    canvas.save(output_path)


def run_embedding_retrieval_test(
    extractor: DINOEmbeddingExtractor,
    retriever: FAISSSkuRetriever,
    query_info: dict[str, Any],
) -> dict[str, Any]:
    print("\n[Test 2] Embedding Retrieval Test")
    with Image.open(query_info["absolute_path"]) as image:
        embedding = extractor.embed(image.convert("RGB"))

    retrieval = retriever.search(embedding, top_k=5)
    top5 = retrieval["topk_candidates"]
    top1 = top5[0]
    top1_same_image = top1.get("image_path") == query_info["relative_path"]
    top1_same_sku = str(top1.get("sku_id")) == query_info["sku_id"]
    status = "PASS" if top1_same_sku else "FAIL"

    result = {
        "query_image": query_info["relative_path"],
        "query_sku_id": query_info["sku_id"],
        "query_product_name": query_info["product_name"],
        "top5": top5,
        "top1_same_image": top1_same_image,
        "top1_same_sku": top1_same_sku,
        "status": status,
    }

    output_path = RESULTS_DIR / "retrieval_result.json"
    write_json(output_path, result)

    print(f"query image: {result['query_image']}")
    print(f"query sku_id: {result['query_sku_id']}")
    for candidate in top5:
        print(
            f"- rank={candidate['rank']}, sku={candidate['sku_id']}, "
            f"similarity={candidate['similarity']:.6f}, image={candidate['image_path']}"
        )
    print(f"top1_same_image: {top1_same_image}")
    print(f"top1_same_sku: {top1_same_sku}")
    print(f"retrieval_status: {status}")
    print(f"retrieval_result_saved: {output_path}")
    return result


def run_single_image_counting_test(
    detector: OpenVocabularyProductDetector,
    extractor: DINOEmbeddingExtractor,
    retriever: FAISSSkuRetriever,
    query_path: Path,
    detections: list[DetectionBox],
):
    print("\n[Test 3] Single Image Counting Test")
    if detections:
        counting_detector = StaticDetector(detections)
        print("Counting mode: open-vocabulary detection crop(s)")
    else:
        with Image.open(query_path) as image:
            width, height = image.size
        detections = [DetectionBox(0, 0, width, height, 1.0)]
        counting_detector = StaticDetector(detections)
        print("Counting mode: open-vocabulary detection failed, using whole image as one crop")

    pipeline = VisionCountingPipeline(
        detector=counting_detector,
        embedding_extractor=extractor,
        retriever=retriever,
        top_k=5,
        similarity_threshold=0.70,
    )
    result = pipeline.count(query_path)
    output_path = RESULTS_DIR / "count_result.json"
    write_json(output_path, result.to_dict())

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"count_result_saved: {output_path}")
    return result


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def print_final_summary(
    detections: list[DetectionBox],
    retrieval_summary: dict[str, Any],
    count_result,
) -> None:
    top1 = retrieval_summary["top5"][0]
    print("\nFinal Summary")
    print(f"- open-vocabulary detection success: {bool(detections)}")
    print(f"- detection count: {len(detections)}")
    print(
        f"- retrieval Top1: sku={top1['sku_id']}, product={top1['product_name']}, "
        f"similarity={top1['similarity']:.6f}"
    )
    print(f"- retrieval status: {retrieval_summary['status']}")
    print(f"- counting result: {count_result.sku_counts}")


if __name__ == "__main__":
    main()
