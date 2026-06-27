from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedding import DEFAULT_MODEL_NAME, build_reference_gallery_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DINOv2 embeddings for reference gallery images.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "products_image")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "embeddings")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_reference_gallery_embeddings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )

    print(f"총 상품 수: {result.product_count}")
    print(f"총 이미지 수: {result.image_count}")
    print(f"embedding dimension: {result.embedding_dim}")
    print(f"embeddings.npy shape: {result.embeddings_shape}")
    print(f"faiss index vector count: {result.faiss_vector_count}")
    print("metadata.csv 앞 5줄:")
    print(result.metadata.head(5).to_string(index=False))

    if result.skipped_files:
        print(f"건너뛴 파일/폴더 수: {len(result.skipped_files)}")
        for skipped in result.skipped_files[:10]:
            print(f"- {skipped}")


if __name__ == "__main__":
    main()
