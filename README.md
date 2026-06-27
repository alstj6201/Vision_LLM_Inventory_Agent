# Retail Inventory AI Prototype

This project prototypes an unmanned-store inventory intelligence system.

It combines:
- POS-based virtual inventory
- demand forecasting
- CCTV-based physical stock counting
- shrinkage detection
- severity triage
- safe order generation through a deterministic harness

## Why This Architecture
The main business problem is not just forecasting demand. The harder problem is detecting when POS inventory and real shelf inventory diverge, especially when the error is only one item.

Therefore, the system separates:
- probabilistic AI reasoning
- deterministic validation
- auditable execution

## MVP Flow
1. Load POS sales and physical stock counts.
2. Forecast next-day demand.
3. Compare expected stock with physical stock.
4. Compute shrinkage and severity.
5. Route normal cases to optimizer.
6. Route exception cases to agents.
7. Validate all order drafts through harness.
8. Generate decision card.

## Run Demo
```bash
pip install -r requirements.txt
python scripts/run_demo.py
```

## Vision Counting
Reference Gallery images under `products_image/` are used to build the SKU embedding database only. They are not detector validation data.

Build the reference gallery embeddings first:

```bash
python tools/build_embeddings.py --batch-size 8
```

Detection is performed on real CCTV shelf images. This project does not fine-tune YOLO and does not use COCO-pretrained YOLO as the default detector. Instead, open-vocabulary detection finds product candidate bounding boxes only. Final SKU classification is done by DINOv2 crop embeddings plus FAISS retrieval.

Pipeline:

```text
CCTV Shelf Image
-> OpenVocabularyProductDetector
-> Bounding Boxes
-> Crop
-> DINOv2 Embedding
-> FAISS Retrieval
-> SKU Prediction
-> SKU Count Aggregation
```

Count products in a CCTV shelf image:

```bash
python tools/count_products_in_image.py \
  --image-path path/to/shelf.jpg \
  --index-path data/embeddings/faiss.index \
  --metadata-path data/embeddings/metadata.csv \
  --output-json data/vision_counts/shelf_counts.json \
  --detector open-vocab \
  --top-k 5 \
  --det-conf 0.25 \
  --sim-threshold 0.70
```

Pre-download the OWL-ViT open-vocabulary detector model into the Hugging Face cache:

```bash
python tools/download_owlvit_model.py
```

Smoke test the real detector on an image:

```bash
python tools/test_open_vocab_detector.py --image-path path/to/image.jpg
```

The default open-vocabulary prompts are scoped to the current product domain: snack packages, chip bags, cracker/cookie/candy/chocolate packages, instant noodles, cup noodles, ramen packages, packaged food, food packages, and pouch packages. Detector labels are never used as SKU labels.

The JSON output includes `image_path`, `sku_counts`, `predictions`, and `low_confidence_predictions`.
