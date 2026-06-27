from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RANDOM_SEED = 42
MERGE_DATASET_PATH = Path("data/products/merge_dataset.csv")
PRODUCTS_IMAGE_DIR = Path("products_image")
OUTPUT_DIR = Path("synthetic_retail_company_dataset")
BACKUP_DIR = Path("synthetic_retail_company_dataset_backup")
MERGE_DATASET_COLUMNS = [
    "date",
    "wm_yr_wk",
    "wday",
    "d",
    "sku_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "product_name",
    "image_path",
    "filename",
    "height",
    "angle",
    "sales",
    "sell_price",
]
IMAGE_FILENAME_RE = re.compile(r"^(?P<sku_id>.+)_(?P<height>\d+)_[A-Za-z]_(?P<angle>\d+)\.(?:jpg|jpeg)$")


@dataclass(frozen=True)
class BuildSummary:
    used_sku_count: int
    excluded_products_sku_count: int
    excluded_merge_sku_count: int
    date_start: str
    date_end: str
    transaction_count: int
    inventory_snapshot_count: int
    order_count: int
    sqlite_row_counts: dict[str, int]
    anomaly_counts: dict[str, int]
    foreign_key_ok: bool
    integrity_ok: bool
    output_dir: Path


def rebuild_synthetic_dataset(
    merge_dataset_path: Path = MERGE_DATASET_PATH,
    products_image_dir: Path = PRODUCTS_IMAGE_DIR,
    output_dir: Path = OUTPUT_DIR,
    backup_dir: Path = BACKUP_DIR,
) -> BuildSummary:
    rng = np.random.default_rng(RANDOM_SEED)
    merge_df = load_merge_dataset(merge_dataset_path)
    product_map = parse_products_image(products_image_dir)
    mapping = map_skus(merge_df, product_map)
    sku_images = collect_sku_images(products_image_dir, mapping["common_skus"])
    runtime_df = prepare_runtime_sales(merge_df, mapping["common_skus"], product_map)

    backup_existing_dataset(output_dir, backup_dir)
    csv_dir = output_dir / "csv"
    jsonl_dir = output_dir / "jsonl"
    csv_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    tables = generate_tables(runtime_df, product_map, sku_images, rng)
    jsonl_tables = generate_jsonl_tables(tables, rng)

    for name, frame in tables.items():
        frame.to_csv(csv_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_jsonl(jsonl_dir / "rag_case_library.jsonl", jsonl_tables["rag_case_library"])
    write_jsonl(jsonl_dir / "vision_detections_sample.jsonl", jsonl_tables["vision_detections_sample"])

    sqlite_path = output_dir / "retail_inventory.sqlite"
    sqlite_row_counts = build_sqlite(sqlite_path, tables, jsonl_tables)
    foreign_key_ok = validate_foreign_keys(sqlite_path)
    integrity_ok = validate_integrity(tables, runtime_df)

    write_schema_readme(
        output_dir / "README_schema.md",
        tables=tables,
        jsonl_tables=jsonl_tables,
        mapping=mapping,
        runtime_df=runtime_df,
        sqlite_row_counts=sqlite_row_counts,
        foreign_key_ok=foreign_key_ok,
        integrity_ok=integrity_ok,
    )

    anomaly_counts = tables["anomaly_cases"]["anomaly_type"].value_counts().sort_index().to_dict()
    return BuildSummary(
        used_sku_count=len(mapping["common_skus"]),
        excluded_products_sku_count=len(mapping["products_only_skus"]),
        excluded_merge_sku_count=len(mapping["merge_only_skus"]),
        date_start=str(runtime_df["date"].min().date()),
        date_end=str(runtime_df["date"].max().date()),
        transaction_count=len(tables["pos_transactions"]),
        inventory_snapshot_count=len(tables["inventory_snapshot"]),
        order_count=len(tables["order_history"]),
        sqlite_row_counts=sqlite_row_counts,
        anomaly_counts={str(key): int(value) for key, value in anomaly_counts.items()},
        foreign_key_ok=foreign_key_ok,
        integrity_ok=integrity_ok,
        output_dir=output_dir,
    )


def load_merge_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"merge_dataset.csv not found: {path}")
    df = pd.read_csv(path, dtype={"sku_id": str}, low_memory=False)
    missing = set(MERGE_DATASET_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"merge_dataset.csv missing required columns: {sorted(missing)}")
    df = df[MERGE_DATASET_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["sku_id"] = df["sku_id"].astype(str)
    df["sales"] = pd.to_numeric(df["sales"], errors="raise").astype(int)
    df["sell_price"] = pd.to_numeric(df["sell_price"], errors="raise").astype(float)
    if df["sales"].isna().any() or (df["sales"] < 0).any():
        raise ValueError("merge_dataset.csv has missing or negative sales values")
    if df["sell_price"].isna().any() or (df["sell_price"] < 0).any():
        raise ValueError("merge_dataset.csv has missing or negative sell_price values")
    return df


def parse_products_image(products_dir: Path) -> dict[str, str]:
    if not products_dir.exists():
        raise FileNotFoundError(f"products_image folder not found: {products_dir}")
    product_map: dict[str, str] = {}
    for folder in sorted(path for path in products_dir.iterdir() if path.is_dir()):
        if "_" not in folder.name:
            continue
        sku_id, product_name = folder.name.split("_", 1)
        product_map[sku_id] = product_name
    if not product_map:
        raise ValueError(f"No product folders found in {products_dir}")
    return product_map


def collect_sku_images(products_dir: Path, common_skus: list[str]) -> pd.DataFrame:
    rows = []
    image_id = 1
    sku_set = set(common_skus)
    for folder in sorted(path for path in products_dir.iterdir() if path.is_dir()):
        if "_" not in folder.name:
            continue
        folder_sku, _ = folder.name.split("_", 1)
        if folder_sku not in sku_set:
            continue
        for image_path in sorted(folder.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            match = IMAGE_FILENAME_RE.match(image_path.name)
            if not match or match.group("sku_id") != folder_sku:
                continue
            rows.append(
                {
                    "image_id": f"IMG{image_id:08d}",
                    "sku_id": folder_sku,
                    "image_path": image_path.relative_to(products_dir).as_posix(),
                    "filename": image_path.name,
                    "height": int(match.group("height")),
                    "angle": int(match.group("angle")),
                }
            )
            image_id += 1
    if not rows:
        raise ValueError("No reference gallery images found for common SKUs")
    return pd.DataFrame(rows)


def map_skus(merge_df: pd.DataFrame, product_map: dict[str, str]) -> dict[str, list[str]]:
    merge_skus = set(merge_df["sku_id"].astype(str).unique())
    product_skus = set(product_map)
    common_skus = sorted(merge_skus & product_skus)
    if not common_skus:
        raise ValueError("No common SKU exists between merge_dataset.csv and products_image")
    return {
        "common_skus": common_skus,
        "products_only_skus": sorted(product_skus - merge_skus),
        "merge_only_skus": sorted(merge_skus - product_skus),
    }


def prepare_runtime_sales(
    merge_df: pd.DataFrame,
    common_skus: list[str],
    product_map: dict[str, str],
) -> pd.DataFrame:
    df = merge_df[merge_df["sku_id"].isin(common_skus)].copy()
    df["product_name"] = df["sku_id"].map(product_map)
    df = (
        df.groupby(["date", "sku_id", "product_name"], as_index=False)
        .agg(
            wm_yr_wk=("wm_yr_wk", "first"),
            wday=("wday", "first"),
            d=("d", "first"),
            event_name_1=("event_name_1", "first"),
            event_type_1=("event_type_1", "first"),
            event_name_2=("event_name_2", "first"),
            event_type_2=("event_type_2", "first"),
            snap_CA=("snap_CA", "max"),
            snap_TX=("snap_TX", "max"),
            snap_WI=("snap_WI", "max"),
            representative_image_path=("image_path", "first"),
            representative_filename=("filename", "first"),
            sales=("sales", "sum"),
            sell_price=("sell_price", "mean"),
        )
        .sort_values(["date", "sku_id"])
        .reset_index(drop=True)
    )
    return df


def backup_existing_dataset(output_dir: Path, backup_dir: Path) -> None:
    if not output_dir.exists():
        return
    if backup_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = backup_dir.with_name(f"{backup_dir.name}_{timestamp}")
    shutil.move(str(output_dir), str(backup_dir))


def generate_tables(
    sales_df: pd.DataFrame,
    product_map: dict[str, str],
    sku_images: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    skus = sorted(sales_df["sku_id"].unique())
    dates = pd.Series(sorted(sales_df["date"].unique()))
    suppliers = make_suppliers()
    sku_master = make_sku_master(sales_df, product_map, suppliers, sku_images, rng)
    planogram_slots = make_planogram_slots(skus)
    weather_holiday = make_weather_holiday(dates, rng)
    promotion_calendar = make_promotion_calendar(sales_df, rng)
    pos_transactions = make_pos_transactions(sales_df, promotion_calendar, rng)
    demand_forecasts = make_demand_forecasts(sales_df, rng)
    inventory_snapshot = make_inventory_snapshot(sales_df, sku_master, rng)
    cv_count_log = make_cv_count_log(inventory_snapshot, rng)
    anomaly_cases = make_anomaly_cases(sales_df, inventory_snapshot, cv_count_log, rng)
    order_drafts = make_order_drafts(sales_df, inventory_snapshot, sku_master, rng)
    order_history = make_order_history(order_drafts, sku_master, rng)
    harness_results = make_harness_results(order_history, rng)
    decision_cards = make_decision_cards(anomaly_cases, harness_results, rng)

    return {
        "suppliers": suppliers,
        "sku_master": sku_master,
        "sku_images": sku_images,
        "planogram_slots": planogram_slots,
        "pos_transactions": pos_transactions,
        "weather_holiday": weather_holiday,
        "promotion_calendar": promotion_calendar,
        "demand_forecasts": demand_forecasts,
        "inventory_snapshot": inventory_snapshot,
        "cv_count_log": cv_count_log,
        "anomaly_cases": anomaly_cases,
        "order_drafts": order_drafts,
        "order_history": order_history,
        "harness_results": harness_results,
        "decision_cards": decision_cards,
    }


def make_suppliers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"supplier_id": "SUP001", "supplier_name": "Snack Supplier", "lead_time_days": 2},
            {"supplier_id": "SUP002", "supplier_name": "Noodle Supplier", "lead_time_days": 3},
            {"supplier_id": "SUP003", "supplier_name": "General Food Supplier", "lead_time_days": 4},
        ]
    )


def infer_category(product_name: str) -> str:
    snack_keywords = ["포키", "꼬깔콘", "새우깡", "콘초", "바나나킥", "쿠키", "초코", "도넛", "젤리", "캔디"]
    noodle_keywords = ["라면", "사발", "짜파게티", "왕뚜껑", "면"]
    if any(keyword in product_name for keyword in noodle_keywords):
        return "noodles"
    if any(keyword in product_name for keyword in snack_keywords):
        return "snack"
    return "misc"


def make_sku_master(
    sales_df: pd.DataFrame,
    product_map: dict[str, str],
    suppliers: pd.DataFrame,
    sku_images: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    supplier_by_category = {"snack": "SUP001", "noodles": "SUP002", "misc": "SUP003"}
    for sku_id in sorted(sales_df["sku_id"].unique()):
        product_name = product_map[sku_id]
        category = infer_category(product_name)
        sku_sales = sales_df[sales_df["sku_id"] == sku_id]
        avg_daily_sales = max(1.0, float(sku_sales["sales"].mean()))
        selling_price = round(float(sku_sales["sell_price"].mean()), 2)
        representative_image_path = sku_images[sku_images["sku_id"] == sku_id]["image_path"].iloc[0]
        unit_cost = round(selling_price * float(rng.uniform(0.48, 0.68)), 2)
        pack_size = int(rng.choice([6, 8, 10, 12, 16, 20]))
        reorder_point = int(math.ceil(avg_daily_sales * 3))
        reorder_quantity = int(math.ceil(avg_daily_sales * 5 / pack_size) * pack_size)
        rows.append(
            {
                "sku_id": sku_id,
                "product_name": product_name,
                "category": category,
                "supplier_id": supplier_by_category[category],
                "representative_image_path": representative_image_path,
                "unit_cost": unit_cost,
                "selling_price": selling_price,
                "pack_size": pack_size,
                "reorder_point": reorder_point,
                "reorder_quantity": max(pack_size, reorder_quantity),
                "min_order_qty": pack_size,
                "max_order_qty": max(pack_size * 4, reorder_quantity * 4),
                "storage_volume": round(float(rng.uniform(0.002, 0.018)), 4),
            }
        )
    return pd.DataFrame(rows)


def make_planogram_slots(skus: list[str]) -> pd.DataFrame:
    rows = []
    for idx, sku_id in enumerate(skus, start=1):
        shelf_id = f"S{((idx - 1) // 5) + 1:02d}"
        rows.append(
            {
                "slot_id": f"SL{idx:03d}",
                "shelf_id": shelf_id,
                "shelf_level": ((idx - 1) % 5) + 1,
                "position": idx,
                "sku_id": sku_id,
            }
        )
    return pd.DataFrame(rows)


def make_weather_holiday(dates: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    weather_options = ["sunny", "cloudy", "rainy", "snowy"]
    for date in pd.to_datetime(dates):
        day_of_year = int(date.dayofyear)
        temp = 14 + 12 * math.sin(2 * math.pi * (day_of_year / 365.0)) + rng.normal(0, 3)
        is_weekend = int(date.weekday() >= 5)
        is_holiday = int(is_weekend or rng.random() < 0.025)
        rows.append(
            {
                "date": date.date().isoformat(),
                "temperature": round(float(temp), 1),
                "weather": str(rng.choice(weather_options, p=[0.55, 0.25, 0.15, 0.05])),
                "is_holiday": is_holiday,
                "is_weekend": is_weekend,
            }
        )
    return pd.DataFrame(rows)


def make_promotion_calendar(sales_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    promo_types = ["discount", "bundle", "endcap"]
    for idx, row in sales_df.reset_index(drop=True).iterrows():
        event_name = first_non_empty(row.get("event_name_1"), row.get("event_name_2"))
        event_type = first_non_empty(row.get("event_type_1"), row.get("event_type_2"))
        snap_active = int(row.get("snap_CA", 0) or row.get("snap_TX", 0) or row.get("snap_WI", 0))
        source_event = bool(event_name or event_type or snap_active)
        synthetic_promo = bool(not source_event and rng.random() < 0.05)
        promo = int(source_event or synthetic_promo)
        rows.append(
            {
                "date": row["date"].date().isoformat(),
                "sku_id": row["sku_id"],
                "promotion_flag": promo,
                "promotion_type": event_type or ("snap" if snap_active else str(rng.choice(promo_types)) if synthetic_promo else "none"),
                "discount_rate": round(float(rng.uniform(0.05, 0.25)), 2) if promo else 0.0,
                "event_name_1": first_non_empty(row.get("event_name_1")),
                "event_type_1": first_non_empty(row.get("event_type_1")),
                "event_name_2": first_non_empty(row.get("event_name_2")),
                "event_type_2": first_non_empty(row.get("event_type_2")),
                "snap_CA": int(row.get("snap_CA", 0)),
                "snap_TX": int(row.get("snap_TX", 0)),
                "snap_WI": int(row.get("snap_WI", 0)),
            }
        )
    return pd.DataFrame(rows)


def first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def make_pos_transactions(
    sales_df: pd.DataFrame,
    promotion_calendar: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    promo_lookup = {
        (row.date, row.sku_id): int(row.promotion_flag)
        for row in promotion_calendar.itertuples(index=False)
    }
    rows = []
    transaction_id = 1
    for row in sales_df.itertuples(index=False):
        quantity = int(row.sales)
        if quantity <= 0:
            continue
        split_count = int(min(max(1, math.ceil(quantity / 12)), 4))
        pieces = split_quantity(quantity, split_count, rng)
        for piece in pieces:
            hour = int(rng.choice(np.arange(8, 23)))
            minute = int(rng.integers(0, 60))
            second = int(rng.integers(0, 60))
            timestamp = datetime.combine(row.date.date(), datetime.min.time()) + timedelta(
                hours=hour, minutes=minute, seconds=second
            )
            rows.append(
                {
                    "transaction_id": f"TX{transaction_id:08d}",
                    "date": row.date.date().isoformat(),
                    "timestamp": timestamp.isoformat(sep=" "),
                    "sku_id": row.sku_id,
                    "quantity": int(piece),
                    "unit_price": round(float(row.sell_price), 2),
                    "promotion_flag": promo_lookup.get((row.date.date().isoformat(), row.sku_id), 0),
                }
            )
            transaction_id += 1
    return pd.DataFrame(rows)


def split_quantity(quantity: int, split_count: int, rng: np.random.Generator) -> list[int]:
    if split_count <= 1:
        return [quantity]
    weights = rng.dirichlet(np.ones(split_count))
    raw = np.maximum(1, np.floor(weights * quantity).astype(int))
    while raw.sum() > quantity:
        raw[int(np.argmax(raw))] -= 1
    while raw.sum() < quantity:
        raw[int(rng.integers(0, split_count))] += 1
    return [int(value) for value in raw if value > 0]


def make_demand_forecasts(sales_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for sku_id, sku_df in sales_df.groupby("sku_id"):
        sku_df = sku_df.sort_values("date")
        noise_scale = max(1.0, float(sku_df["sales"].std()) * 0.08)
        forecast_values = sku_df["sales"].to_numpy(dtype=float) + rng.normal(0, noise_scale, len(sku_df))
        for row, forecast in zip(sku_df.itertuples(index=False), forecast_values):
            rows.append(
                {
                    "date": row.date.date().isoformat(),
                    "sku_id": sku_id,
                    "forecast_quantity": max(0, int(round(float(forecast)))),
                }
            )
    return pd.DataFrame(rows)


def make_inventory_snapshot(
    sales_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    master = sku_master.set_index("sku_id")
    rows = []
    for sku_id, sku_df in sales_df.groupby("sku_id"):
        sku_df = sku_df.sort_values("date")
        reorder_point = int(master.loc[sku_id, "reorder_point"])
        reorder_quantity = int(master.loc[sku_id, "reorder_quantity"])
        stock = int(max(reorder_point * 2, sku_df["sales"].quantile(0.95) * 5))
        for row in sku_df.itertuples(index=False):
            opening_stock = stock
            units_sold = min(int(row.sales), opening_stock)
            expected_stock = opening_stock - units_sold
            restock_qty = 0
            if expected_stock <= reorder_point:
                restock_qty = reorder_quantity
            closing_stock = expected_stock + restock_qty
            rows.append(
                {
                    "date": row.date.date().isoformat(),
                    "sku_id": sku_id,
                    "opening_stock": int(opening_stock),
                    "units_sold": int(units_sold),
                    "restock_qty": int(restock_qty),
                    "closing_stock": int(closing_stock),
                    "expected_stock": int(expected_stock),
                }
            )
            stock = closing_stock
    return pd.DataFrame(rows)


def make_cv_count_log(inventory_snapshot: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for idx, row in inventory_snapshot.reset_index(drop=True).iterrows():
        anomaly_roll = rng.random()
        if anomaly_roll < 0.03:
            delta = -int(rng.integers(2, 5))
            confidence = float(rng.uniform(0.72, 0.9))
        elif anomaly_roll < 0.08:
            delta = int(rng.choice([-1, 1]))
            confidence = float(rng.uniform(0.55, 0.78))
        else:
            delta = int(rng.choice([-1, 0, 0, 0, 1]))
            confidence = float(rng.uniform(0.86, 0.99))
        cv_count = max(0, int(row["closing_stock"]) + delta)
        timestamp = datetime.fromisoformat(row["date"]) + timedelta(hours=23, minutes=30)
        rows.append(
            {
                "snapshot_id": f"CV{idx + 1:08d}",
                "date": row["date"],
                "timestamp": timestamp.isoformat(sep=" "),
                "sku_id": row["sku_id"],
                "expected_stock": int(row["closing_stock"]),
                "cv_count": int(cv_count),
                "count_confidence": round(confidence, 3),
            }
        )
    return pd.DataFrame(rows)


def make_anomaly_cases(
    sales_df: pd.DataFrame,
    inventory_snapshot: pd.DataFrame,
    cv_count_log: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sales_key = sales_df.copy()
    sales_key["date"] = sales_key["date"].dt.date.astype(str)
    merged = sales_key.merge(
        inventory_snapshot[["date", "sku_id", "closing_stock"]],
        on=["date", "sku_id"],
    ).merge(cv_count_log[["date", "sku_id", "cv_count", "count_confidence"]], on=["date", "sku_id"])
    rows = []
    anomaly_types = ["normal", "demand_spike", "shrinkage", "theft_suspected", "cv_error", "restocking_delay", "planogram_mismatch"]
    for idx, row in merged.reset_index(drop=True).iterrows():
        demand_score = min(1.0, float(row["sales"]) / max(1.0, merged[merged["sku_id"] == row["sku_id"]]["sales"].quantile(0.95)))
        shrinkage_score = min(1.0, max(0, int(row["closing_stock"]) - int(row["cv_count"])) / max(1, int(row["closing_stock"])))
        low_conf = 1.0 - float(row["count_confidence"])
        severity = round(0.35 * demand_score + 0.4 * shrinkage_score + 0.25 * low_conf, 3)
        if rng.random() < 0.075 or severity > 0.55:
            if shrinkage_score > 0.15 and demand_score < 0.75:
                anomaly_type = "theft_suspected"
            elif shrinkage_score > 0.05:
                anomaly_type = "shrinkage"
            elif low_conf > 0.35:
                anomaly_type = "cv_error"
            elif demand_score > 0.9:
                anomaly_type = "demand_spike"
            else:
                anomaly_type = str(rng.choice(anomaly_types[4:]))
        else:
            anomaly_type = "normal"
        rows.append(
            {
                "anomaly_id": f"AN{idx + 1:08d}",
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "sku_id": row["sku_id"],
                "anomaly_type": anomaly_type,
                "demand_anomaly_score": round(float(demand_score), 3),
                "shrinkage_score": round(float(shrinkage_score), 3),
                "severity": severity,
                "reason": make_anomaly_reason(anomaly_type),
                "status": "open" if anomaly_type != "normal" and severity >= 0.55 else "closed",
            }
        )
    return pd.DataFrame(rows)


def make_anomaly_reason(anomaly_type: str) -> str:
    reasons = {
        "normal": "No material exception detected.",
        "demand_spike": "Sales exceeded recent baseline.",
        "shrinkage": "CV count lower than expected stock.",
        "theft_suspected": "Demand normal while physical stock is abnormally low.",
        "cv_error": "Low CV confidence may explain stock mismatch.",
        "restocking_delay": "Expected restock may not have reached shelf.",
        "planogram_mismatch": "Detected SKU may not match assigned slot.",
    }
    return reasons[anomaly_type]


def make_order_drafts(
    sales_df: pd.DataFrame,
    inventory_snapshot: pd.DataFrame,
    sku_master: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    master = sku_master.set_index("sku_id")
    reorder_points = inventory_snapshot["sku_id"].map(master["reorder_point"].to_dict())
    latest_snapshots = inventory_snapshot[inventory_snapshot["expected_stock"] <= reorder_points].copy()
    latest_snapshots = latest_snapshots.iloc[::5].reset_index(drop=True)
    rows = []
    for idx, row in latest_snapshots.iterrows():
        sku_id = row["sku_id"]
        suggested_qty = int(master.loc[sku_id, "reorder_quantity"])
        rows.append(
            {
                "draft_id": f"DR{idx + 1:08d}",
                "date": row["date"],
                "sku_id": sku_id,
                "suggested_qty": suggested_qty,
                "reasoning": "Stock is at or below deterministic reorder point.",
                "confidence": round(float(rng.uniform(0.78, 0.96)), 3),
            }
        )
    return pd.DataFrame(rows)


def make_order_history(
    order_drafts: pd.DataFrame,
    sku_master: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    master = sku_master.set_index("sku_id")
    rows = []
    for idx, row in order_drafts.iterrows():
        sku_id = row["sku_id"]
        ordered_qty = min(int(row["suggested_qty"]), int(master.loc[sku_id, "max_order_qty"]))
        status = str(rng.choice(["executed", "executed", "executed", "requires_review", "blocked"]))
        rows.append(
            {
                "order_id": f"OR{idx + 1:08d}",
                "date": row["date"],
                "sku_id": sku_id,
                "supplier_id": master.loc[sku_id, "supplier_id"],
                "ordered_qty": int(ordered_qty),
                "status": status,
                "reason": "Harness approved." if status == "executed" else "Harness routed for safety review.",
            }
        )
    return pd.DataFrame(rows)


def make_harness_results(order_history: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for idx, row in order_history.iterrows():
        final_result = "pass" if row["status"] == "executed" else "block" if row["status"] == "blocked" else "review"
        rows.append(
            {
                "validation_id": f"VA{idx + 1:08d}",
                "date": row["date"],
                "sku_id": row["sku_id"],
                "semantic_check": "pass",
                "stock_audit": "pass" if final_result == "pass" else "review",
                "constraint_check": "pass" if final_result != "block" else "fail",
                "final_result": final_result,
                "retry_count": int(rng.integers(0, 3)) if final_result != "pass" else 0,
            }
        )
    return pd.DataFrame(rows)


def make_decision_cards(
    anomaly_cases: pd.DataFrame,
    harness_results: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    selected = anomaly_cases[anomaly_cases["anomaly_type"] != "normal"].iloc[::10].copy()
    rows = []
    for idx, row in selected.reset_index(drop=True).iterrows():
        severity = float(row["severity"])
        if severity >= 0.7:
            decision = "freeze_and_review"
        elif severity >= 0.45:
            decision = "human_review"
        else:
            decision = "automatic_order_allowed"
        rows.append(
            {
                "decision_id": f"DC{idx + 1:08d}",
                "date": row["date"],
                "sku_id": row["sku_id"],
                "trigger_source": row["anomaly_type"],
                "severity": severity,
                "final_decision": decision,
                "retry_count": int(rng.integers(0, 3)),
                "token_cost": round(float(rng.uniform(0.01, 0.08)), 4),
            }
        )
    return pd.DataFrame(rows)


def generate_jsonl_tables(tables: dict[str, pd.DataFrame], rng: np.random.Generator) -> dict[str, list[dict[str, Any]]]:
    anomaly_cases = tables["anomaly_cases"]
    cases = []
    for idx, row in anomaly_cases[anomaly_cases["anomaly_type"] != "normal"].head(200).iterrows():
        cases.append(
            {
                "case_id": f"CASE{idx + 1:08d}",
                "anomaly_type": row["anomaly_type"],
                "summary": row["reason"],
                "evidence": {
                    "sku_id": row["sku_id"],
                    "date": row["date"],
                    "severity": float(row["severity"]),
                },
                "resolution": "Use deterministic harness outcome before any order execution.",
                "tags": [row["anomaly_type"], "synthetic", "inventory"],
            }
        )

    detections = []
    cv_sample = tables["cv_count_log"].iloc[:: max(1, len(tables["cv_count_log"]) // 200)].head(200)
    sku_master = tables["sku_master"].set_index("sku_id")
    for row in cv_sample.itertuples(index=False):
        sku_id = row.sku_id
        detections.append(
            {
                "snapshot_id": row.snapshot_id,
                "sku_id": sku_id,
                "detections": [
                    {
                        "bbox": [int(rng.integers(0, 300)), int(rng.integers(0, 200)), int(rng.integers(320, 640)), int(rng.integers(220, 480))],
                        "confidence": round(float(row.count_confidence), 3),
                        "topk_candidates": [
                            {
                                "sku_id": sku_id,
                                "product_name": sku_master.loc[sku_id, "product_name"],
                                "similarity": round(float(rng.uniform(0.72, 0.98)), 3),
                            }
                        ],
                    }
                ],
            }
        )
    return {"rag_case_library": cases, "vision_detections_sample": detections}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sqlite(
    sqlite_path: Path,
    tables: dict[str, pd.DataFrame],
    jsonl_tables: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    if sqlite_path.exists():
        sqlite_path.unlink()
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    create_schema(conn)
    for table_name, df in tables.items():
        df.to_sql(table_name, conn, if_exists="append", index=False, chunksize=500)
    pd.DataFrame(jsonl_tables["rag_case_library"]).assign(
        evidence=lambda frame: frame["evidence"].map(json.dumps),
        tags=lambda frame: frame["tags"].map(json.dumps),
    ).to_sql("rag_case_library", conn, if_exists="append", index=False, chunksize=500)
    pd.DataFrame(jsonl_tables["vision_detections_sample"]).assign(
        detections=lambda frame: frame["detections"].map(json.dumps)
    ).to_sql("vision_detections_sample", conn, if_exists="append", index=False, chunksize=500)
    row_counts = {
        name: int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
        for name in list(tables) + ["rag_case_library", "vision_detections_sample"]
    }
    conn.commit()
    conn.close()
    return row_counts


def create_schema(conn: sqlite3.Connection) -> None:
    statements = [
        "CREATE TABLE suppliers (supplier_id TEXT PRIMARY KEY, supplier_name TEXT NOT NULL, lead_time_days INTEGER NOT NULL)",
        """
        CREATE TABLE sku_master (
            sku_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            supplier_id TEXT NOT NULL REFERENCES suppliers(supplier_id),
            representative_image_path TEXT NOT NULL,
            unit_cost REAL NOT NULL,
            selling_price REAL NOT NULL,
            pack_size INTEGER NOT NULL,
            reorder_point INTEGER NOT NULL,
            reorder_quantity INTEGER NOT NULL,
            min_order_qty INTEGER NOT NULL,
            max_order_qty INTEGER NOT NULL,
            storage_volume REAL NOT NULL
        )
        """,
        """
        CREATE TABLE sku_images (
            image_id TEXT PRIMARY KEY,
            sku_id TEXT NOT NULL REFERENCES sku_master(sku_id),
            image_path TEXT NOT NULL,
            filename TEXT NOT NULL,
            height INTEGER NOT NULL,
            angle INTEGER NOT NULL
        )
        """,
        "CREATE TABLE planogram_slots (slot_id TEXT PRIMARY KEY, shelf_id TEXT NOT NULL, shelf_level INTEGER NOT NULL, position INTEGER NOT NULL UNIQUE, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id))",
        "CREATE TABLE pos_transactions (transaction_id TEXT PRIMARY KEY, date TEXT NOT NULL, timestamp TEXT NOT NULL, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), quantity INTEGER NOT NULL, unit_price REAL NOT NULL, promotion_flag INTEGER NOT NULL)",
        "CREATE TABLE weather_holiday (date TEXT PRIMARY KEY, temperature REAL NOT NULL, weather TEXT NOT NULL, is_holiday INTEGER NOT NULL, is_weekend INTEGER NOT NULL)",
        "CREATE TABLE promotion_calendar (date TEXT NOT NULL, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), promotion_flag INTEGER NOT NULL, promotion_type TEXT NOT NULL, discount_rate REAL NOT NULL, event_name_1 TEXT NOT NULL, event_type_1 TEXT NOT NULL, event_name_2 TEXT NOT NULL, event_type_2 TEXT NOT NULL, snap_CA INTEGER NOT NULL, snap_TX INTEGER NOT NULL, snap_WI INTEGER NOT NULL, PRIMARY KEY(date, sku_id))",
        "CREATE TABLE demand_forecasts (date TEXT NOT NULL, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), forecast_quantity INTEGER NOT NULL, PRIMARY KEY(date, sku_id))",
        "CREATE TABLE inventory_snapshot (date TEXT NOT NULL, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), opening_stock INTEGER NOT NULL, units_sold INTEGER NOT NULL, restock_qty INTEGER NOT NULL, closing_stock INTEGER NOT NULL, expected_stock INTEGER NOT NULL, PRIMARY KEY(date, sku_id))",
        "CREATE TABLE cv_count_log (snapshot_id TEXT PRIMARY KEY, date TEXT NOT NULL, timestamp TEXT NOT NULL, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), expected_stock INTEGER NOT NULL, cv_count INTEGER NOT NULL, count_confidence REAL NOT NULL)",
        "CREATE TABLE anomaly_cases (anomaly_id TEXT PRIMARY KEY, date TEXT NOT NULL, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), anomaly_type TEXT NOT NULL, demand_anomaly_score REAL NOT NULL, shrinkage_score REAL NOT NULL, severity REAL NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL)",
        "CREATE TABLE order_drafts (draft_id TEXT PRIMARY KEY, date TEXT NOT NULL, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), suggested_qty INTEGER NOT NULL, reasoning TEXT NOT NULL, confidence REAL NOT NULL)",
        "CREATE TABLE order_history (order_id TEXT PRIMARY KEY, date TEXT NOT NULL, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), supplier_id TEXT NOT NULL REFERENCES suppliers(supplier_id), ordered_qty INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT NOT NULL)",
        "CREATE TABLE harness_results (validation_id TEXT PRIMARY KEY, date TEXT NOT NULL, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), semantic_check TEXT NOT NULL, stock_audit TEXT NOT NULL, constraint_check TEXT NOT NULL, final_result TEXT NOT NULL, retry_count INTEGER NOT NULL)",
        "CREATE TABLE decision_cards (decision_id TEXT PRIMARY KEY, date TEXT NOT NULL, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), trigger_source TEXT NOT NULL, severity REAL NOT NULL, final_decision TEXT NOT NULL, retry_count INTEGER NOT NULL, token_cost REAL NOT NULL)",
        "CREATE TABLE rag_case_library (case_id TEXT PRIMARY KEY, anomaly_type TEXT NOT NULL, summary TEXT NOT NULL, evidence TEXT NOT NULL, resolution TEXT NOT NULL, tags TEXT NOT NULL)",
        "CREATE TABLE vision_detections_sample (snapshot_id TEXT PRIMARY KEY, sku_id TEXT NOT NULL REFERENCES sku_master(sku_id), detections TEXT NOT NULL)",
    ]
    for statement in statements:
        conn.execute(statement)


def validate_foreign_keys(sqlite_path: Path) -> bool:
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    rows = conn.execute("PRAGMA foreign_key_check;").fetchall()
    conn.close()
    return len(rows) == 0


def validate_integrity(tables: dict[str, pd.DataFrame], runtime_df: pd.DataFrame) -> bool:
    sku_set = set(tables["sku_master"]["sku_id"])
    date_set = set(pd.to_datetime(runtime_df["date"]).dt.date.astype(str))
    checks = [
        set(tables["pos_transactions"]["sku_id"]).issubset(sku_set),
        set(tables["sku_images"]["sku_id"]).issubset(sku_set),
        set(tables["inventory_snapshot"]["sku_id"]).issubset(sku_set),
        set(tables["cv_count_log"]["sku_id"]).issubset(sku_set),
        (tables["inventory_snapshot"]["closing_stock"] >= 0).all(),
        (tables["cv_count_log"]["cv_count"] >= 0).all(),
        set(tables["inventory_snapshot"]["date"]) == date_set,
        not tables["planogram_slots"]["position"].duplicated().any(),
        set(tables["sku_master"]["supplier_id"]).issubset(set(tables["suppliers"]["supplier_id"])),
        not tables["sku_images"]["image_id"].duplicated().any(),
    ]
    return bool(all(checks))


def write_schema_readme(
    path: Path,
    tables: dict[str, pd.DataFrame],
    jsonl_tables: dict[str, list[dict[str, Any]]],
    mapping: dict[str, list[str]],
    runtime_df: pd.DataFrame,
    sqlite_row_counts: dict[str, int],
    foreign_key_ok: bool,
    integrity_ok: bool,
) -> None:
    lines = [
        "# Synthetic Retail Dataset Schema",
        "",
        "This dataset is generated from `data/products/merge_dataset.csv` as the sales source of truth and `products_image/` as the runtime SKU catalog.",
        "",
        "## Generation Rules",
        "- Only SKUs present in both `merge_dataset.csv` and `products_image/` are used.",
        "- Sales quantities and date range are derived from `merge_dataset.csv` without modifying the source file.",
        "- `sku_master` stores product-level records; `sku_images` stores all reference-gallery images as a 1:N SKU-to-image relation.",
        "- DINOv2 embedding and FAISS retrieval should join against `sku_images`, not use `sku_master` as the image-level table.",
        "- Missing operational fields are generated deterministically with random seed 42.",
        "- SQLite enables `PRAGMA foreign_keys = ON` and validates foreign keys after loading.",
        "",
        "## SKU Mapping",
        f"- Runtime SKU count: {len(mapping['common_skus'])}",
        f"- products_image-only SKU count: {len(mapping['products_only_skus'])}",
        f"- merge_dataset-only SKU count: {len(mapping['merge_only_skus'])}",
        f"- Date range: {runtime_df['date'].min().date()} to {runtime_df['date'].max().date()}",
        "",
        "## CSV Tables",
    ]
    for name, frame in tables.items():
        lines.append(f"### {name}.csv")
        lines.append(f"- Rows: {len(frame)}")
        lines.append(f"- Columns: {', '.join(frame.columns)}")
        lines.append("")
    lines.extend(
        [
            "## JSONL Files",
            "### rag_case_library.jsonl",
            f"- Rows: {len(jsonl_tables['rag_case_library'])}",
            "- Fields: case_id, anomaly_type, summary, evidence, resolution, tags",
            "",
            "### vision_detections_sample.jsonl",
            f"- Rows: {len(jsonl_tables['vision_detections_sample'])}",
            "- Fields: snapshot_id, sku_id, detections",
            "",
            "## SQLite Runtime Database",
            "- File: `retail_inventory.sqlite`",
            "- Runtime tables mirror the CSV/JSONL exports.",
            "- Core relations: `sku_master.supplier_id -> suppliers.supplier_id`; all SKU-bearing runtime tables reference `sku_master.sku_id`.",
            "",
            "## SQLite Row Counts",
        ]
    )
    for name, count in sqlite_row_counts.items():
        lines.append(f"- {name}: {count}")
    lines.extend(
        [
            "",
            "## Integrity Validation",
            f"- Foreign key check: {'PASS' if foreign_key_ok else 'FAIL'}",
            f"- Data integrity check: {'PASS' if integrity_ok else 'FAIL'}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
