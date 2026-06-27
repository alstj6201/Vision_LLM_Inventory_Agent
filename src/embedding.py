from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModel


DEFAULT_MODEL_NAME = "facebook/dinov2-small"
IMAGE_FILENAME_RE = re.compile(
    r"^(?P<sku_id>.+)_(?P<height>\d+)_s_(?P<angle>\d+)\.(?:jpg|jpeg)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImageMetadata:
    row_id: int
    sku_id: str
    product_name: str
    image_path: str
    filename: str
    height: int
    angle: int


@dataclass(frozen=True)
class BuildResult:
    product_count: int
    image_count: int
    embedding_dim: int
    embeddings_shape: tuple[int, int]
    faiss_vector_count: int
    metadata: pd.DataFrame
    skipped_files: list[str]


def parse_product_folder(folder_name: str) -> tuple[str, str]:
    if "_" not in folder_name:
        raise ValueError(f"Product folder must match '{{sku_id}}_{{product_name}}': {folder_name}")
    sku_id, product_name = folder_name.split("_", 1)
    if not sku_id or not product_name:
        raise ValueError(f"Product folder has empty sku_id or product_name: {folder_name}")
    return sku_id, product_name


def parse_image_filename(filename: str) -> tuple[str, int, int]:
    match = IMAGE_FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Image filename must match '{{sku_id}}_{{height}}_s_{{angle}}.jpg': {filename}")
    return (
        match.group("sku_id"),
        int(match.group("height")),
        int(match.group("angle")),
    )


def iter_product_dirs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    return sorted(path for path in input_dir.iterdir() if path.is_dir())


def collect_image_metadata(input_dir: Path) -> tuple[list[ImageMetadata], int, list[str]]:
    rows: list[ImageMetadata] = []
    skipped_files: list[str] = []
    input_dir = input_dir.resolve()
    product_dirs = iter_product_dirs(input_dir)

    for product_dir in product_dirs:
        try:
            folder_sku_id, product_name = parse_product_folder(product_dir.name)
        except ValueError as exc:
            skipped_files.append(str(exc))
            continue

        image_paths = sorted(
            path
            for path in product_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        )
        for image_path in image_paths:
            try:
                file_sku_id, height, angle = parse_image_filename(image_path.name)
            except ValueError as exc:
                skipped_files.append(str(exc))
                continue

            if file_sku_id != folder_sku_id:
                skipped_files.append(
                    f"SKU mismatch: folder={folder_sku_id}, file={file_sku_id}, path={image_path}"
                )
                continue

            rows.append(
                ImageMetadata(
                    row_id=len(rows),
                    sku_id=folder_sku_id,
                    product_name=product_name,
                    image_path=image_path.relative_to(input_dir).as_posix(),
                    filename=image_path.name,
                    height=height,
                    angle=angle,
                )
            )

    return rows, len(product_dirs), skipped_files


def metadata_to_frame(rows: Iterable[ImageMetadata]) -> pd.DataFrame:
    columns = ["row_id", "sku_id", "product_name", "image_path", "filename", "height", "angle"]
    return pd.DataFrame([row.__dict__ for row in rows], columns=columns)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dinov2(model_name: str = DEFAULT_MODEL_NAME, device: torch.device | None = None):
    device = device or select_device()
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model, device


def _load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def embed_images(
    image_paths: list[str],
    image_root: Path | None = None,
    batch_size: int = 16,
    model_name: str = DEFAULT_MODEL_NAME,
    device: torch.device | None = None,
) -> np.ndarray:
    if not image_paths:
        raise ValueError("No valid JPG images found to embed.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    processor, model, device = load_dinov2(model_name=model_name, device=device)
    image_root = image_root.resolve() if image_root is not None else None
    batches: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            images = [
                _load_rgb_image(str((image_root / path) if image_root is not None else path))
                for path in batch_paths
            ]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0]
            normalized = F.normalize(cls_embeddings, p=2, dim=1)
            batches.append(normalized.cpu().numpy().astype("float32"))

    return np.vstack(batches)


def save_faiss_index(embeddings: np.ndarray, output_path: Path) -> faiss.IndexFlatIP:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={embeddings.shape}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings.astype("float32")))
    faiss.write_index(index, str(output_path))
    return index


def build_reference_gallery_embeddings(
    input_dir: Path = Path("products_image"),
    output_dir: Path = Path("data/embeddings"),
    batch_size: int = 16,
    model_name: str = DEFAULT_MODEL_NAME,
) -> BuildResult:
    rows, product_count, skipped_files = collect_image_metadata(input_dir)
    metadata = metadata_to_frame(rows)
    image_paths = metadata["image_path"].tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = embed_images(
        image_paths,
        image_root=input_dir,
        batch_size=batch_size,
        model_name=model_name,
    )

    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.csv"
    faiss_path = output_dir / "faiss.index"

    np.save(embeddings_path, embeddings)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    index = save_faiss_index(embeddings, faiss_path)

    return BuildResult(
        product_count=product_count,
        image_count=len(rows),
        embedding_dim=int(embeddings.shape[1]),
        embeddings_shape=tuple(embeddings.shape),
        faiss_vector_count=int(index.ntotal),
        metadata=metadata,
        skipped_files=skipped_files,
    )
