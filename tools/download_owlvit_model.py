from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL_NAME = "google/owlvit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OWL-ViT model files into Hugging Face cache.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from huggingface_hub import scan_cache_dir
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install them with `pip install transformers huggingface_hub torch`."
        ) from exc

    print(f"Downloading OWL-ViT model to Hugging Face cache: {args.model_name}")
    processor = OwlViTProcessor.from_pretrained(args.model_name)
    model = OwlViTForObjectDetection.from_pretrained(args.model_name)

    cache_info = scan_cache_dir()
    repo_cache = next(
        (repo for repo in cache_info.repos if repo.repo_id == args.model_name),
        None,
    )
    cache_path = repo_cache.repo_path if repo_cache is not None else "unknown"

    print("Download complete.")
    print(f"model_name: {args.model_name}")
    print(f"processor_class: {processor.__class__.__name__}")
    print(f"model_class: {model.__class__.__name__}")
    print(f"cache_path: {cache_path}")
    print(f"parameter_count: {sum(parameter.numel() for parameter in model.parameters())}")


if __name__ == "__main__":
    main()
