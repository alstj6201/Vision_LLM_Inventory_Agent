from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from PIL import Image


AggregationMethod = Literal["max", "mean"]
DEFAULT_PRODUCT_PROMPTS = [
    "snack package",
    "chip bag",
    "cracker package",
    "cookie package",
    "candy package",
    "chocolate package",
    "instant noodle",
    "cup noodle",
    "ramen package",
    "packaged food",
    "food package",
    "pouch package",
]


@dataclass(frozen=True)
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = ""
    detector_name: str = ""


@dataclass(frozen=True)
class CropPrediction:
    sku_id: str
    product_name: str
    similarity: float
    topk_candidates: list[dict[str, Any]]
    bbox: DetectionBox


@dataclass(frozen=True)
class CountResult:
    image_path: str
    total_detections: int
    sku_counts: dict[str, int]
    predictions: list[CropPrediction]
    low_confidence_predictions: list[CropPrediction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "total_detections": self.total_detections,
            "sku_counts": self.sku_counts,
            "predictions": [prediction_to_dict(prediction) for prediction in self.predictions],
            "low_confidence_predictions": [
                prediction_to_dict(prediction) for prediction in self.low_confidence_predictions
            ],
        }


def prediction_to_dict(prediction: CropPrediction) -> dict[str, Any]:
    data = asdict(prediction)
    data["bbox"] = asdict(prediction.bbox)
    return data


class BaseDetector:
    def detect(self, image_path: str | Path) -> list[DetectionBox]:
        raise RuntimeError("Detector implementations must provide detect(image_path).")


class OpenVocabularyProductDetector(BaseDetector):
    """Find product candidate boxes on CCTV shelf images.

    The detector only proposes "there is a product here" regions. Detector labels
    are intentionally discarded and are never used as final SKU predictions.
    """

    def __init__(
        self,
        prompts: list[str] | None = None,
        confidence_threshold: float = 0.05,
        model_name: str = "google/owlvit-base-patch32",
        device: str | None = None,
    ) -> None:
        self.prompts = prompts or DEFAULT_PRODUCT_PROMPTS
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self.device_name = device
        self._torch = None
        self._processor = None
        self._model = None
        self._device = None
        self.post_process_method: str | None = None

    def detect(self, image_path: str | Path) -> list[DetectionBox]:
        self._load_model()
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

        torch = self._torch
        processor = self._processor
        model = self._model
        device = self._device
        inputs = processor(text=[self.prompts], images=rgb_image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        width, height = rgb_image.size
        target_sizes = torch.tensor([[height, width]], device=device)
        results, method_name = post_process_owlvit_outputs(
            processor=processor,
            outputs=outputs,
            target_sizes=target_sizes,
            prompts=self.prompts,
            threshold=self.confidence_threshold,
        )
        self.post_process_method = method_name

        detections: list[DetectionBox] = []
        labels = results.get("labels", results.get("text_labels", []))
        for index, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
            if hasattr(box, "detach"):
                box_values = box.detach().cpu().tolist()
            else:
                box_values = list(box)
            x1, y1, x2, y2 = [float(value) for value in box_values]
            label = format_owlvit_label(labels, index=index, prompts=self.prompts)
            confidence = float(score.detach().cpu().item() if hasattr(score, "detach") else score)
            detections.append(
                DetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    label=label,
                    detector_name="owlvit",
                )
            )
        return detections

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
        except ImportError as exc:
            raise RuntimeError(
                "OpenVocabularyProductDetector requires torch and transformers. "
                "Install them with `pip install torch transformers`."
            ) from exc

        self._torch = torch
        self._device = torch.device(
            self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            self._processor = OwlViTProcessor.from_pretrained(self.model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(self.model_name)
        except OSError as exc:
            raise RuntimeError(
                "Could not load the open-vocabulary detection model "
                f"`{self.model_name}`. If this is the first run, allow network access so "
                "Hugging Face can download the model, or pre-download it into the local cache. "
                "This detector is intended for real CCTV shelf images; reference gallery images "
                "are only for the DINOv2 + FAISS SKU embedding database."
            ) from exc
        self._model.to(self._device)
        self._model.eval()


def post_process_owlvit_outputs(
    processor: Any,
    outputs: Any,
    target_sizes: Any,
    prompts: list[str],
    threshold: float,
) -> tuple[dict[str, Any], str]:
    """Run OWL-ViT post-processing across transformers API versions."""
    if hasattr(processor, "post_process_object_detection"):
        processed = processor.post_process_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )
        return processed[0], "post_process_object_detection"

    if hasattr(processor, "post_process_grounded_object_detection"):
        processed = processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            text_labels=[prompts],
        )
        return processed[0], "post_process_grounded_object_detection"

    if hasattr(processor, "post_process"):
        processed = processor.post_process(outputs=outputs, target_sizes=target_sizes)
        result = processed[0]
        scores = result.get("scores", [])
        keep_indices = [
            index
            for index, score in enumerate(scores)
            if float(score.detach().cpu().item() if hasattr(score, "detach") else score) >= threshold
        ]
        filtered: dict[str, Any] = {}
        for key, value in result.items():
            try:
                filtered[key] = value[keep_indices]
            except Exception:
                filtered[key] = [value[index] for index in keep_indices]
        return filtered, "post_process"

    raise RuntimeError(
        "OwlViTProcessor does not provide a supported post-process method. "
        "Expected one of post_process_object_detection, "
        "post_process_grounded_object_detection, or post_process."
    )


def format_owlvit_label(labels: Any, index: int, prompts: list[str]) -> str:
    if labels is None or len(labels) <= index:
        return ""
    label = labels[index]
    if hasattr(label, "detach"):
        label = int(label.detach().cpu().item())
    if isinstance(label, (int, np.integer)):
        if 0 <= int(label) < len(prompts):
            return prompts[int(label)]
        return str(int(label))
    return str(label)


class MockDetector(BaseDetector):
    def __init__(self, boxes: list[DetectionBox] | None = None) -> None:
        self.boxes = boxes or []

    def detect(self, image_path: str | Path) -> list[DetectionBox]:
        return self.boxes


class DINOEmbeddingExtractor:
    def __init__(self, model_name: str = "facebook/dinov2-small", device: str | None = None) -> None:
        try:
            import torch
            from torch.nn import functional as F
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "torch and transformers are required for DINOEmbeddingExtractor."
            ) from exc

        self.torch = torch
        self.functional = F
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, crop: Image.Image) -> np.ndarray:
        image = crop.convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0]
            normalized = self.functional.normalize(cls_embedding, p=2, dim=1)

        return normalized[0].detach().cpu().numpy().astype("float32")


class FAISSSkuRetriever:
    def __init__(
        self,
        index_path: str | Path = "data/embeddings/faiss.index",
        metadata_path: str | Path = "data/embeddings/metadata.csv",
        aggregation: AggregationMethod = "max",
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss is required for FAISSSkuRetriever.") from exc

        self.faiss = faiss
        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
        self.aggregation = aggregation

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"Metadata rows ({len(self.metadata)}) must match FAISS vectors ({self.index.ntotal})."
            )

    def search(self, embedding: np.ndarray, top_k: int = 5) -> dict[str, Any]:
        query = np.asarray(embedding, dtype="float32")
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError(f"Expected one query embedding, got shape={query.shape}")

        norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / np.maximum(norm, 1e-12)

        scores, indices = self.index.search(np.ascontiguousarray(query), top_k)
        candidates = self._format_candidates(scores[0], indices[0])
        return aggregate_candidates_by_sku(candidates, method=self.aggregation)

    def _format_candidates(self, scores: np.ndarray, indices: np.ndarray) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for rank, (score, index) in enumerate(zip(scores, indices), start=1):
            if int(index) < 0:
                continue
            row = self.metadata.iloc[int(index)]
            candidates.append(
                {
                    "rank": rank,
                    "row_id": int(index),
                    "sku_id": str(row["sku_id"]),
                    "product_name": str(row["product_name"]),
                    "similarity": float(score),
                    "image_path": str(row.get("image_path", "")),
                    "filename": str(row.get("filename", "")),
                    "height": _safe_int(row.get("height")),
                    "angle": _safe_int(row.get("angle")),
                }
            )
        return candidates


def _safe_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def aggregate_candidates_by_sku(
    candidates: list[dict[str, Any]],
    method: AggregationMethod = "max",
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No FAISS candidates returned.")
    if method not in {"max", "mean"}:
        raise ValueError("aggregation method must be 'max' or 'mean'")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(str(candidate["sku_id"]), []).append(candidate)

    sku_scores: dict[str, float] = {}
    for sku_id, sku_candidates in grouped.items():
        similarities = [float(candidate["similarity"]) for candidate in sku_candidates]
        sku_scores[sku_id] = max(similarities) if method == "max" else float(np.mean(similarities))

    best_sku = max(sku_scores, key=sku_scores.get)
    best_group = grouped[best_sku]
    representative = max(best_group, key=lambda candidate: float(candidate["similarity"]))

    return {
        "sku_id": best_sku,
        "product_name": str(representative["product_name"]),
        "similarity": float(sku_scores[best_sku]),
        "topk_candidates": candidates,
        "sku_scores": sku_scores,
    }


class VisionCountingPipeline:
    def __init__(
        self,
        detector: Any,
        embedding_extractor: Any,
        retriever: Any,
        top_k: int = 5,
        similarity_threshold: float = 0.70,
    ) -> None:
        self.detector = detector
        self.embedding_extractor = embedding_extractor
        self.retriever = retriever
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def count(self, image_path: str | Path) -> CountResult:
        image_path = Path(image_path)
        detections = self.detector.detect(image_path)
        predictions: list[CropPrediction] = []
        low_confidence_predictions: list[CropPrediction] = []
        sku_counts: dict[str, int] = {}

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size

            for bbox in detections:
                crop = rgb_image.crop(_clamp_bbox(bbox, width=width, height=height))
                embedding = self.embedding_extractor.embed(crop)
                retrieval = self.retriever.search(embedding, top_k=self.top_k)

                prediction = CropPrediction(
                    sku_id=str(retrieval["sku_id"]),
                    product_name=str(retrieval["product_name"]),
                    similarity=float(retrieval["similarity"]),
                    topk_candidates=list(retrieval["topk_candidates"]),
                    bbox=bbox,
                )
                predictions.append(prediction)
                sku_counts[prediction.sku_id] = sku_counts.get(prediction.sku_id, 0) + 1

                if prediction.similarity < self.similarity_threshold:
                    low_confidence_predictions.append(prediction)

        return CountResult(
            image_path=str(image_path),
            total_detections=len(detections),
            sku_counts=sku_counts,
            predictions=predictions,
            low_confidence_predictions=low_confidence_predictions,
        )


def _clamp_bbox(bbox: DetectionBox, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width, int(round(bbox.x1))))
    y1 = max(0, min(height, int(round(bbox.y1))))
    x2 = max(0, min(width, int(round(bbox.x2))))
    y2 = max(0, min(height, int(round(bbox.y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid or empty detection box after clamping: {bbox}")
    return x1, y1, x2, y2
