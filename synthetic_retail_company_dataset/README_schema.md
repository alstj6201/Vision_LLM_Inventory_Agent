# Synthetic Retail Dataset Schema

This dataset is generated from `data/products/merge_dataset.csv` as the sales source of truth and `products_image/` as the runtime SKU catalog.

## Generation Rules
- Only SKUs present in both `merge_dataset.csv` and `products_image/` are used.
- Sales quantities and date range are derived from `merge_dataset.csv` without modifying the source file.
- `sku_master` stores product-level records; `sku_images` stores all reference-gallery images as a 1:N SKU-to-image relation.
- DINOv2 embedding and FAISS retrieval should join against `sku_images`, not use `sku_master` as the image-level table.
- Missing operational fields are generated deterministically with random seed 42.
- SQLite enables `PRAGMA foreign_keys = ON` and validates foreign keys after loading.

## SKU Mapping
- Runtime SKU count: 20
- products_image-only SKU count: 0
- merge_dataset-only SKU count: 0
- Date range: 2021-01-29 to 2026-05-22

## CSV Tables
### suppliers.csv
- Rows: 3
- Columns: supplier_id, supplier_name, lead_time_days

### sku_master.csv
- Rows: 20
- Columns: sku_id, product_name, category, supplier_id, representative_image_path, unit_cost, selling_price, pack_size, reorder_point, reorder_quantity, min_order_qty, max_order_qty, storage_volume

### sku_images.csv
- Rows: 162
- Columns: image_id, sku_id, image_path, filename, height, angle

### planogram_slots.csv
- Rows: 20
- Columns: slot_id, shelf_id, shelf_level, position, sku_id

### pos_transactions.csv
- Rows: 79426
- Columns: transaction_id, date, timestamp, sku_id, quantity, unit_price, promotion_flag

### weather_holiday.csv
- Rows: 1939
- Columns: date, temperature, weather, is_holiday, is_weekend

### promotion_calendar.csv
- Rows: 38780
- Columns: date, sku_id, promotion_flag, promotion_type, discount_rate, event_name_1, event_type_1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI

### demand_forecasts.csv
- Rows: 38780
- Columns: date, sku_id, forecast_quantity

### inventory_snapshot.csv
- Rows: 38780
- Columns: date, sku_id, opening_stock, units_sold, restock_qty, closing_stock, expected_stock

### cv_count_log.csv
- Rows: 38780
- Columns: snapshot_id, date, timestamp, sku_id, expected_stock, cv_count, count_confidence

### anomaly_cases.csv
- Rows: 38780
- Columns: anomaly_id, date, sku_id, anomaly_type, demand_anomaly_score, shrinkage_score, severity, reason, status

### order_drafts.csv
- Rows: 1450
- Columns: draft_id, date, sku_id, suggested_qty, reasoning, confidence

### order_history.csv
- Rows: 1450
- Columns: order_id, date, sku_id, supplier_id, ordered_qty, status, reason

### harness_results.csv
- Rows: 1450
- Columns: validation_id, date, sku_id, semantic_check, stock_audit, constraint_check, final_result, retry_count

### decision_cards.csv
- Rows: 292
- Columns: decision_id, date, sku_id, trigger_source, severity, final_decision, retry_count, token_cost

## JSONL Files
### rag_case_library.jsonl
- Rows: 200
- Fields: case_id, anomaly_type, summary, evidence, resolution, tags

### vision_detections_sample.jsonl
- Rows: 200
- Fields: snapshot_id, sku_id, detections

## SQLite Runtime Database
- File: `retail_inventory.sqlite`
- Runtime tables mirror the CSV/JSONL exports.
- Core relations: `sku_master.supplier_id -> suppliers.supplier_id`; all SKU-bearing runtime tables reference `sku_master.sku_id`.

## SQLite Row Counts
- suppliers: 3
- sku_master: 20
- sku_images: 162
- planogram_slots: 20
- pos_transactions: 79426
- weather_holiday: 1939
- promotion_calendar: 38780
- demand_forecasts: 38780
- inventory_snapshot: 38780
- cv_count_log: 38780
- anomaly_cases: 38780
- order_drafts: 1450
- order_history: 1450
- harness_results: 1450
- decision_cards: 292
- rag_case_library: 200
- vision_detections_sample: 200

## Integrity Validation
- Foreign key check: PASS
- Data integrity check: PASS
