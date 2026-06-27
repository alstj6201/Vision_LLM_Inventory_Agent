from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.vision_counting import (  # noqa: E402
    DetectionBox,
    MockDetector,
    VisionCountingPipeline,
    aggregate_candidates_by_sku,
)


class MockEmbeddingExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, crop) -> np.ndarray:
        self.calls += 1
        embedding = np.zeros(4, dtype="float32")
        embedding[0] = 1.0
        return embedding


class MockRetriever:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.calls = 0

    def search(self, embedding: np.ndarray, top_k: int = 5) -> dict:
        result = self.results[self.calls]
        self.calls += 1
        return result


def make_retrieval(sku_id: str, product_name: str, similarity: float) -> dict:
    return {
        "sku_id": sku_id,
        "product_name": product_name,
        "similarity": similarity,
        "topk_candidates": [
            {
                "rank": 1,
                "row_id": 0,
                "sku_id": sku_id,
                "product_name": product_name,
                "similarity": similarity,
                "image_path": f"{sku_id}/ref.jpg",
                "filename": "ref.jpg",
                "height": 0,
                "angle": 1,
            }
        ],
    }


def create_test_image(path: Path) -> None:
    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    image.save(path)


def test_pipeline_counts_three_detections_and_aggregates_sku_counts(tmp_path: Path):
    image_path = tmp_path / "shelf.jpg"
    create_test_image(image_path)
    boxes = [
        DetectionBox(0, 0, 20, 20, 0.9),
        DetectionBox(20, 0, 40, 20, 0.8),
        DetectionBox(40, 0, 60, 20, 0.7),
    ]
    retriever = MockRetriever(
        [
            make_retrieval("10060", "product-a", 0.92),
            make_retrieval("10060", "product-a", 0.88),
            make_retrieval("10091", "product-b", 0.86),
        ]
    )
    pipeline = VisionCountingPipeline(
        detector=MockDetector(boxes),
        embedding_extractor=MockEmbeddingExtractor(),
        retriever=retriever,
        similarity_threshold=0.70,
    )

    result = pipeline.count(image_path)

    assert result.total_detections == 3
    assert result.sku_counts["10060"] == 2
    assert result.sku_counts["10091"] == 1
    assert len(result.predictions) == 3


def test_pipeline_flags_low_confidence_prediction(tmp_path: Path):
    image_path = tmp_path / "shelf.jpg"
    create_test_image(image_path)
    boxes = [DetectionBox(0, 0, 20, 20, 0.9)]
    pipeline = VisionCountingPipeline(
        detector=MockDetector(boxes),
        embedding_extractor=MockEmbeddingExtractor(),
        retriever=MockRetriever([make_retrieval("10060", "product-a", 0.69)]),
        similarity_threshold=0.70,
    )

    result = pipeline.count(image_path)

    assert result.sku_counts["10060"] == 1
    assert len(result.low_confidence_predictions) == 1
    assert result.low_confidence_predictions[0].similarity == 0.69


def test_pipeline_returns_empty_result_when_detector_finds_no_boxes(tmp_path: Path):
    image_path = tmp_path / "empty_shelf.jpg"
    create_test_image(image_path)
    pipeline = VisionCountingPipeline(
        detector=MockDetector([]),
        embedding_extractor=MockEmbeddingExtractor(),
        retriever=MockRetriever([]),
        similarity_threshold=0.70,
    )

    result = pipeline.count(image_path)

    assert result.total_detections == 0
    assert result.sku_counts == {}
    assert result.predictions == []
    assert result.low_confidence_predictions == []


def test_aggregate_candidates_by_sku_uses_max_similarity():
    candidates = [
        {"sku_id": "A", "product_name": "product-a", "similarity": 0.72},
        {"sku_id": "B", "product_name": "product-b", "similarity": 0.81},
        {"sku_id": "A", "product_name": "product-a", "similarity": 0.94},
    ]

    result = aggregate_candidates_by_sku(candidates, method="max")

    assert result["sku_id"] == "A"
    assert result["product_name"] == "product-a"
    assert result["similarity"] == 0.94
    assert result["sku_scores"] == {"A": 0.94, "B": 0.81}


def test_aggregate_candidates_by_sku_can_use_mean_similarity():
    candidates = [
        {"sku_id": "A", "product_name": "product-a", "similarity": 0.70},
        {"sku_id": "A", "product_name": "product-a", "similarity": 0.72},
        {"sku_id": "B", "product_name": "product-b", "similarity": 0.90},
    ]

    result = aggregate_candidates_by_sku(candidates, method="mean")

    assert result["sku_id"] == "B"
    assert result["similarity"] == 0.90
