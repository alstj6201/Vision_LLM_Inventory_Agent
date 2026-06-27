from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.vision_counting import (  # noqa: E402
    DINOEmbeddingExtractor,
    FAISSSkuRetriever,
    OpenVocabularyProductDetector,
    VisionCountingPipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count products in a CCTV shelf image using open-vocabulary detection + DINOv2 + FAISS."
    )
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--index-path", default=PROJECT_ROOT / "data" / "embeddings" / "faiss.index", type=Path)
    parser.add_argument("--metadata-path", default=PROJECT_ROOT / "data" / "embeddings" / "metadata.csv", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--detector", choices=["open-vocab"], default="open-vocab")
    parser.add_argument("--det-conf", default=0.25, type=float)
    parser.add_argument("--sim-threshold", default=0.70, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        detector = OpenVocabularyProductDetector(confidence_threshold=args.det_conf)
        embedding_extractor = DINOEmbeddingExtractor()
        retriever = FAISSSkuRetriever(index_path=args.index_path, metadata_path=args.metadata_path)
        pipeline = VisionCountingPipeline(
            detector=detector,
            embedding_extractor=embedding_extractor,
            retriever=retriever,
            top_k=args.top_k,
            similarity_threshold=args.sim_threshold,
        )
        result = pipeline.count(args.image_path)
    except RuntimeError as exc:
        print(f"Vision counting could not start: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, ensure_ascii=False, indent=2)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
