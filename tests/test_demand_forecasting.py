from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_ai.demand_forecasting import build_feature_frame, compute_metrics, time_series_split  # noqa: E402


def make_sample_frame() -> pd.DataFrame:
    rows = []
    for sku_id in ["A", "B"]:
        for i, date in enumerate(pd.date_range("2024-01-01", periods=40, freq="D")):
            rows.append(
                {
                    "date": date,
                    "wm_yr_wk": 1,
                    "wday": date.weekday() + 1,
                    "d": i + 1,
                    "sku_id": sku_id,
                    "event_name_1": None,
                    "event_type_1": None,
                    "event_name_2": None,
                    "event_type_2": None,
                    "snap_CA": 0,
                    "snap_TX": 0,
                    "snap_WI": 0,
                    "product_name": f"product-{sku_id}",
                    "image_path": "",
                    "filename": "",
                    "height": 0,
                    "angle": 0,
                    "sales": i + (1 if sku_id == "A" else 2),
                    "sell_price": 1.5,
                }
            )
    return pd.DataFrame(rows)


def test_feature_frame_uses_next_day_target_without_future_leakage():
    features = build_feature_frame(make_sample_frame())
    row = features[(features["sku_id"] == "A")].iloc[0]

    assert row["lag_1"] == 1
    assert row["target_sales"] == 2
    assert row["target_date"] == pd.Timestamp("2024-01-02")


def test_time_series_split_is_chronological():
    features = build_feature_frame(make_sample_frame())
    train_df, val_df, test_df = time_series_split(features, train_ratio=0.6, val_ratio=0.2)

    assert train_df["target_date"].max() < val_df["target_date"].min()
    assert val_df["target_date"].max() < test_df["target_date"].min()


def test_metrics_are_finite():
    metrics = compute_metrics([10, 20, 0], [12, 18, 1])

    assert metrics["MAE"] > 0
    assert metrics["RMSE"] > 0
    assert metrics["sMAPE"] >= 0
    assert metrics["WAPE"] >= 0
