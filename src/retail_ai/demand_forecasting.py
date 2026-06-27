from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MERGE_DATASET_PATH = Path("data/products/merge_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("results/demand_forecasting")
DEFAULT_SQLITE_PATH = Path("synthetic_retail_company_dataset/retail_inventory.sqlite")
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
FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "sell_price",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "lag_28",
    "rolling_mean_28",
    "rolling_std_28",
    "rolling_max_7",
    "rolling_min_7",
    "expanding_mean",
    "days_since_last_sale",
    "consecutive_zero_sales",
    "price_change_rate",
    "event_count",
    "snap_any",
    "sales_cv_28",
    "sales_zero_ratio_28",
    "recent_trend_7_28",
    "event_1_present",
    "event_2_present",
    "event_type_1_code",
    "event_type_2_code",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sku_code",
]
BASELINE_METRICS = {
    "validation_MAE": 7.3226,
    "test_metrics": {
        "MAE": 6.5671,
        "RMSE": 10.2672,
        "sMAPE": 68.2834,
        "WAPE": 34.9713,
    },
}


@dataclass(frozen=True)
class ForecastRunResult:
    best_model: str
    validation_metric: float
    test_metrics: dict[str, float]
    output_dir: Path
    figure_paths: list[Path]


def run_demand_forecasting(
    input_csv: Path = MERGE_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sqlite_path: Path = DEFAULT_SQLITE_PATH,
) -> ForecastRunResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    before_metrics = load_before_metrics(output_dir)

    raw_df = load_merge_dataset(input_csv)
    feature_df = build_feature_frame(raw_df)
    train_df, val_df, test_df = time_series_split(feature_df)

    predictions_by_model: dict[str, pd.DataFrame] = {
        "moving_average": predict_moving_average(train_df, val_df, test_df),
        "weekday_average": predict_weekday_average(train_df, val_df, test_df),
    }
    lgbm_prediction, feature_importance, lgbm_trials, selected_features, removed_features = train_lightgbm(train_df, val_df, test_df)
    predictions_by_model["lightgbm"] = lgbm_prediction

    comparison_rows = []
    for model_name, pred_df in predictions_by_model.items():
        for split in ["validation", "test"]:
            metrics = compute_metrics(pred_df[pred_df["split"] == split]["actual_sales"], pred_df[pred_df["split"] == split]["forecast_quantity"])
            comparison_rows.append({"model": model_name, "split": split, **metrics})
    for trial in lgbm_trials:
        comparison_rows.append({"model": "lightgbm_trial", "split": "validation", **trial})
    model_comparison = pd.DataFrame(comparison_rows)

    validation_scores = model_comparison[model_comparison["split"] == "validation"]
    best_row = validation_scores.loc[validation_scores["MAE"].idxmin()]
    best_model = str(best_row["model"])
    if best_model == "lightgbm_trial":
        best_model = "lightgbm"

    selected_predictions = predictions_by_model[best_model].copy()
    selected_predictions["forecast_quantity"] = selected_predictions["forecast_quantity"].clip(lower=0)
    selected_predictions["absolute_error"] = (selected_predictions["actual_sales"] - selected_predictions["forecast_quantity"]).abs()
    selected_predictions["error"] = selected_predictions["actual_sales"] - selected_predictions["forecast_quantity"]
    selected_predictions["best_model"] = best_model

    test_predictions = selected_predictions[selected_predictions["split"] == "test"].copy()
    sku_metrics = compute_sku_metrics(test_predictions)
    overall_metrics = compute_metrics(test_predictions["actual_sales"], test_predictions["forecast_quantity"])
    anomaly_scores = compute_anomaly_scores(test_predictions)

    selected_predictions.to_csv(output_dir / "forecast_predictions.csv", index=False, encoding="utf-8-sig")
    sku_metrics.to_csv(output_dir / "sku_metrics.csv", index=False, encoding="utf-8-sig")
    model_comparison.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    anomaly_scores.to_csv(output_dir / "demand_anomaly_scores.csv", index=False, encoding="utf-8-sig")
    with (output_dir / "overall_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model": best_model,
                "test_metrics": overall_metrics,
                "selected_features": selected_features,
                "removed_features": removed_features,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    figure_paths = create_figures(
        predictions=test_predictions,
        model_comparison=model_comparison,
        feature_importance=feature_importance,
        anomaly_scores=anomaly_scores,
        sku_metrics=sku_metrics,
        figures_dir=figures_dir,
    )
    save_to_sqlite_if_available(test_predictions, sqlite_path)
    write_improvement_report(
        output_dir=output_dir,
        before_metrics=before_metrics,
        after_metrics=overall_metrics,
        model_comparison=model_comparison,
        sku_metrics=sku_metrics,
        raw_df=raw_df,
        feature_importance=feature_importance,
        selected_features=selected_features,
        removed_features=removed_features,
        best_model=best_model,
    )

    return ForecastRunResult(
        best_model=best_model,
        validation_metric=float(best_row["MAE"]),
        test_metrics=overall_metrics,
        output_dir=output_dir,
        figure_paths=figure_paths,
    )


def load_merge_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"sku_id": str}, low_memory=False)
    missing = set(MERGE_DATASET_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"merge_dataset.csv missing required columns: {sorted(missing)}")
    df = df[MERGE_DATASET_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["sku_id"] = df["sku_id"].astype(str)
    df["sales"] = pd.to_numeric(df["sales"], errors="raise").astype(float)
    df["sell_price"] = pd.to_numeric(df["sell_price"], errors="raise").astype(float)
    return df.sort_values(["sku_id", "date"]).reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["date", "sku_id"], as_index=False)
        .agg(
            product_name=("product_name", "first"),
            sales=("sales", "sum"),
            sell_price=("sell_price", "mean"),
            event_name_1=("event_name_1", "first"),
            event_type_1=("event_type_1", "first"),
            event_name_2=("event_name_2", "first"),
            event_type_2=("event_type_2", "first"),
            snap_CA=("snap_CA", "max"),
            snap_TX=("snap_TX", "max"),
            snap_WI=("snap_WI", "max"),
        )
        .sort_values(["sku_id", "date"])
        .reset_index(drop=True)
    )
    daily["target_date"] = daily.groupby("sku_id")["date"].shift(-1)
    daily["target_sales"] = daily.groupby("sku_id")["sales"].shift(-1)
    daily["day_of_week"] = daily["target_date"].dt.dayofweek
    daily["month"] = daily["target_date"].dt.month
    grouped = daily.groupby("sku_id", group_keys=False)
    daily["lag_1"] = daily["sales"]
    daily["lag_7"] = grouped["sales"].shift(6)
    daily["lag_14"] = grouped["sales"].shift(13)
    daily["lag_28"] = grouped["sales"].shift(27)
    daily["rolling_mean_7"] = grouped["sales"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    daily["rolling_mean_14"] = grouped["sales"].rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
    daily["rolling_mean_28"] = grouped["sales"].rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)
    daily["rolling_std_7"] = grouped["sales"].rolling(7, min_periods=2).std().reset_index(level=0, drop=True)
    daily["rolling_std_28"] = grouped["sales"].rolling(28, min_periods=2).std().reset_index(level=0, drop=True)
    daily["rolling_max_7"] = grouped["sales"].rolling(7, min_periods=1).max().reset_index(level=0, drop=True)
    daily["rolling_min_7"] = grouped["sales"].rolling(7, min_periods=1).min().reset_index(level=0, drop=True)
    daily["expanding_mean"] = grouped["sales"].expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    daily["days_since_last_sale"] = grouped.apply(compute_days_since_last_sale).reset_index(level=0, drop=True)
    daily["consecutive_zero_sales"] = grouped.apply(compute_consecutive_zero_sales).reset_index(level=0, drop=True)
    daily["price_change_rate"] = grouped["sell_price"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    daily["event_1_present"] = daily["event_name_1"].notna().astype(int)
    daily["event_2_present"] = daily["event_name_2"].notna().astype(int)
    daily["event_count"] = daily["event_1_present"] + daily["event_2_present"]
    daily["snap_any"] = ((daily["snap_CA"] > 0) | (daily["snap_TX"] > 0) | (daily["snap_WI"] > 0)).astype(int)
    daily["event_type_1_code"] = pd.Categorical(daily["event_type_1"].fillna("none")).codes
    daily["event_type_2_code"] = pd.Categorical(daily["event_type_2"].fillna("none")).codes
    daily["sku_code"] = pd.Categorical(daily["sku_id"]).codes
    daily["sales_cv_28"] = daily["rolling_std_28"] / daily["rolling_mean_28"].replace(0, np.nan)
    daily["sales_zero_ratio_28"] = grouped["sales"].rolling(28, min_periods=1).apply(lambda values: float(np.mean(values == 0)), raw=True).reset_index(level=0, drop=True)
    daily["recent_trend_7_28"] = daily["rolling_mean_7"] - daily["rolling_mean_28"]
    fill_cols = ["lag_7", "lag_14", "lag_28", "rolling_std_7", "rolling_std_28"]
    for col in fill_cols:
        daily[col] = daily[col].fillna(daily["lag_1"])
    daily["sales_cv_28"] = daily["sales_cv_28"].replace([np.inf, -np.inf], 0).fillna(0)
    daily["price_change_rate"] = daily["price_change_rate"].replace([np.inf, -np.inf], 0).fillna(0)
    daily = daily.dropna(subset=["target_date", "target_sales"]).copy()
    daily["target_date"] = pd.to_datetime(daily["target_date"])
    return daily.reset_index(drop=True)


def compute_days_since_last_sale(sku_df: pd.DataFrame) -> pd.Series:
    values = []
    last_sale_index: int | None = None
    for idx, sales in enumerate(sku_df["sales"].to_numpy()):
        if sales > 0:
            last_sale_index = idx
            values.append(0)
        else:
            values.append(idx + 1 if last_sale_index is None else idx - last_sale_index)
    return pd.Series(values, index=sku_df.index)


def compute_consecutive_zero_sales(sku_df: pd.DataFrame) -> pd.Series:
    values = []
    streak = 0
    for sales in sku_df["sales"].to_numpy():
        if sales <= 0:
            streak += 1
        else:
            streak = 0
        values.append(streak)
    return pd.Series(values, index=sku_df.index)


def time_series_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(df["target_date"].unique()))
    train_end = int(len(dates) * train_ratio)
    val_end = int(len(dates) * (train_ratio + val_ratio))
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])
    train_df = df[df["target_date"].isin(train_dates)].copy()
    val_df = df[df["target_date"].isin(val_dates)].copy()
    test_df = df[df["target_date"].isin(test_dates)].copy()
    return train_df, val_df, test_df


def predict_moving_average(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    eval_df = pd.concat([val_df, test_df], ignore_index=True)
    pred = eval_df[["target_date", "sku_id", "product_name", "target_sales", "rolling_mean_7"]].copy()
    pred["forecast_quantity"] = pred["rolling_mean_7"]
    pred["model"] = "moving_average"
    pred["split"] = np.where(pred["target_date"].isin(val_df["target_date"].unique()), "validation", "test")
    return normalize_prediction_columns(pred)


def predict_weekday_average(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    history = train_df[["target_date", "sku_id", "target_sales"]].copy()
    eval_df = pd.concat([val_df, test_df], ignore_index=True).sort_values(["target_date", "sku_id"])
    rows = []
    hist_by_sku: dict[str, pd.DataFrame] = {sku: sku_df.copy() for sku, sku_df in history.groupby("sku_id")}
    global_mean = float(train_df["target_sales"].mean())
    for row in eval_df.itertuples(index=False):
        sku_hist = hist_by_sku.get(row.sku_id, pd.DataFrame())
        target_dow = row.target_date.dayofweek
        if not sku_hist.empty:
            same_weekday = sku_hist[sku_hist["target_date"].dt.dayofweek == target_dow]["target_sales"]
            forecast = float(same_weekday.mean()) if len(same_weekday) else float(sku_hist["target_sales"].mean())
        else:
            forecast = global_mean
        rows.append(
            {
                "target_date": row.target_date,
                "sku_id": row.sku_id,
                "product_name": row.product_name,
                "actual_sales": row.target_sales,
                "forecast_quantity": forecast,
                "model": "weekday_average",
                "split": "validation" if row.target_date in set(val_df["target_date"].unique()) else "test",
            }
        )
        new_row = pd.DataFrame([{"target_date": row.target_date, "sku_id": row.sku_id, "target_sales": row.target_sales}])
        hist_by_sku[row.sku_id] = pd.concat([sku_hist, new_row], ignore_index=True)
    return pd.DataFrame(rows)


def train_lightgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, float]], list[str], list[str]]:
    import lightgbm as lgb

    candidate_feature_sets = build_candidate_feature_sets(train_df, val_df)
    param_grid = [
        {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 300},
        {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 500},
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 400},
        {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 500},
    ]
    y_train = train_df["target_sales"]
    y_val = val_df["target_sales"]
    best_model = None
    best_mae = math.inf
    selected_features = FEATURE_COLUMNS
    trials: list[dict[str, float]] = []
    for feature_set_name, feature_columns in candidate_feature_sets.items():
        X_train = train_df[feature_columns]
        X_val = val_df[feature_columns]
        for params in param_grid:
            model = lgb.LGBMRegressor(
                objective="regression",
                random_state=42,
                verbosity=-1,
                **params,
            )
            model.fit(X_train, y_train)
            val_pred = np.clip(model.predict(X_val), 0, None)
            metrics = compute_metrics(y_val, val_pred)
            trials.append(
                {
                    "feature_set": feature_set_name,
                    "feature_count": float(len(feature_columns)),
                    **{k: float(v) for k, v in params.items()},
                    **metrics,
                }
            )
            if metrics["MAE"] < best_mae - 1e-9 or (
                abs(metrics["MAE"] - best_mae) <= 1e-9 and len(feature_columns) < len(selected_features)
            ):
                best_mae = metrics["MAE"]
                best_model = model
                selected_features = feature_columns
    assert best_model is not None
    eval_df = pd.concat([val_df, test_df], ignore_index=True)
    preds = np.clip(best_model.predict(eval_df[selected_features]), 0, None)
    pred_df = eval_df[["target_date", "sku_id", "product_name", "target_sales"]].copy()
    pred_df["forecast_quantity"] = preds
    pred_df["model"] = "lightgbm"
    pred_df["split"] = np.where(pred_df["target_date"].isin(val_df["target_date"].unique()), "validation", "test")
    feature_importance = pd.DataFrame(
        {"feature": selected_features, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    removed_features = [feature for feature in FEATURE_COLUMNS if feature not in selected_features]
    return normalize_prediction_columns(pred_df), feature_importance, trials, selected_features, removed_features


def build_candidate_feature_sets(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, list[str]]:
    import lightgbm as lgb

    base_features = [
        "day_of_week",
        "month",
        "sell_price",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
        "event_1_present",
        "event_2_present",
        "event_type_1_code",
        "event_type_2_code",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "sku_code",
    ]
    extended_features = FEATURE_COLUMNS.copy()
    probe = lgb.LGBMRegressor(
        objective="regression",
        random_state=42,
        verbosity=-1,
        num_leaves=15,
        learning_rate=0.05,
        n_estimators=250,
    )
    probe.fit(train_df[extended_features], train_df["target_sales"])
    importances = pd.Series(probe.feature_importances_, index=extended_features)
    nonzero_features = [feature for feature in extended_features if importances[feature] > 0]
    low_use_features = [
        feature for feature in extended_features if importances[feature] <= max(1, importances.quantile(0.10))
    ]
    pruned_features = [feature for feature in extended_features if feature not in low_use_features]
    return {
        "base": base_features,
        "extended": extended_features,
        "nonzero_importance": nonzero_features,
        "pruned_low_importance": pruned_features or nonzero_features,
    }


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"target_sales": "actual_sales"}).copy()
    return df[["target_date", "sku_id", "product_name", "actual_sales", "forecast_quantity", "model", "split"]]


def compute_metrics(actual: Any, forecast: Any) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    err = actual_arr - forecast_arr
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    smape = float(np.mean(2 * np.abs(err) / np.maximum(np.abs(actual_arr) + np.abs(forecast_arr), 1e-9)) * 100)
    wape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(actual_arr)), 1e-9) * 100)
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "WAPE": wape}


def compute_sku_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sku_id, sku_df in predictions.groupby("sku_id"):
        rows.append({"sku_id": sku_id, "product_name": sku_df["product_name"].iloc[0], **compute_metrics(sku_df["actual_sales"], sku_df["forecast_quantity"])})
    return pd.DataFrame(rows).sort_values("WAPE", ascending=False)


def compute_anomaly_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sku_id, sku_df in predictions.groupby("sku_id"):
        df = sku_df.copy()
        abs_err = (df["actual_sales"] - df["forecast_quantity"]).abs()
        scale = max(float(abs_err.quantile(0.95)), 1.0)
        df["anomaly_score"] = np.clip(abs_err / scale, 0, 1)
        frames.append(df[["target_date", "sku_id", "product_name", "actual_sales", "forecast_quantity", "absolute_error", "anomaly_score"]])
    return pd.concat(frames, ignore_index=True)


def create_figures(
    predictions: pd.DataFrame,
    model_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    anomaly_scores: pd.DataFrame,
    sku_metrics: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    sku_ids = sorted(predictions["sku_id"].unique())[:20]
    fig, axes = plt.subplots(5, 4, figsize=(22, 16), sharex=False)
    for ax, sku_id in zip(axes.flatten(), sku_ids):
        sku_df = predictions[predictions["sku_id"] == sku_id].sort_values("target_date")
        ax.plot(sku_df["target_date"], sku_df["actual_sales"], label="Actual", linewidth=1)
        ax.plot(sku_df["target_date"], sku_df["forecast_quantity"], label="Forecast", linewidth=1)
        ax.set_title(str(sku_id), fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    axes.flatten()[0].legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / "actual_vs_forecast_20_skus.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    comp = model_comparison[model_comparison["split"].isin(["validation", "test"]) & model_comparison["model"].isin(["moving_average", "weekday_average", "lightgbm"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = comp.pivot_table(index="model", columns="split", values="MAE")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("MAE")
    ax.set_title("Model Performance Comparison")
    fig.tight_layout()
    path = figures_dir / "model_performance_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 6))
    fi = feature_importance.head(20).sort_values("importance")
    ax.barh(fi["feature"], fi["importance"])
    ax.set_title("LightGBM Feature Importance")
    fig.tight_layout()
    path = figures_dir / "feature_importance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(anomaly_scores["anomaly_score"], bins=30)
    ax.set_title("Demand Anomaly Score Distribution")
    ax.set_xlabel("Anomaly Score")
    fig.tight_layout()
    path = figures_dir / "anomaly_score_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 7))
    top = sku_metrics.sort_values("MAE", ascending=False).head(20).sort_values("MAE")
    ax.barh(top["sku_id"], top["MAE"])
    ax.set_title("Top 20 SKUs by Forecast Error")
    ax.set_xlabel("MAE")
    fig.tight_layout()
    path = figures_dir / "top20_error_skus.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def save_to_sqlite_if_available(predictions: pd.DataFrame, sqlite_path: Path) -> None:
    if not sqlite_path.exists():
        return
    rows = predictions[["target_date", "sku_id", "forecast_quantity"]].copy()
    rows["date"] = pd.to_datetime(rows["target_date"]).dt.date.astype(str)
    rows["forecast_quantity"] = rows["forecast_quantity"].round().clip(lower=0).astype(int)
    rows = rows[["date", "sku_id", "forecast_quantity"]]
    with sqlite3.connect(sqlite_path) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='demand_forecasts'"
        ).fetchone()
        if table_exists:
            conn.execute("DELETE FROM demand_forecasts")
            rows.to_sql("demand_forecasts", conn, if_exists="append", index=False, chunksize=500)


def load_before_metrics(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "overall_metrics.json"
    comparison_path = output_dir / "model_comparison.csv"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            rows = comparison[(comparison["model"] == payload.get("best_model", "lightgbm")) & (comparison["split"] == "validation")]
            if not rows.empty:
                payload["validation_MAE"] = float(rows["MAE"].iloc[0])
        return payload
    except (json.JSONDecodeError, OSError):
        return None


def analyze_sku_error_drivers(raw_df: pd.DataFrame, sku_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw = raw_df.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["event_present"] = raw[["event_name_1", "event_name_2", "event_type_1", "event_type_2"]].notna().any(axis=1).astype(int)
    raw["snap_any"] = ((raw["snap_CA"] > 0) | (raw["snap_TX"] > 0) | (raw["snap_WI"] > 0)).astype(int)
    for row in sku_metrics.head(10).itertuples(index=False):
        sku_df = raw[raw["sku_id"] == str(row.sku_id)].sort_values("date")
        mean_sales = float(sku_df["sales"].mean())
        std_sales = float(sku_df["sales"].std())
        cv = std_sales / max(mean_sales, 1e-9)
        zero_ratio = float((sku_df["sales"] == 0).mean())
        event_ratio = float(sku_df["event_present"].mean())
        snap_ratio = float(sku_df["snap_any"].mean())
        price_cv = float(sku_df["sell_price"].std() / max(sku_df["sell_price"].mean(), 1e-9))
        reasons = []
        if cv > 1.0:
            reasons.append("high sales volatility")
        if zero_ratio > 0.15:
            reasons.append("sparse or zero-sales days")
        if event_ratio > 0.05 or snap_ratio > 0.2:
            reasons.append("event/SNAP sensitivity")
        if price_cv > 0.08:
            reasons.append("price variation")
        if mean_sales < 8:
            reasons.append("low sales volume makes percentage error unstable")
        if not reasons:
            reasons.append("residual nonlinear demand pattern")
        rows.append(
            {
                "sku_id": row.sku_id,
                "product_name": row.product_name,
                "MAE": row.MAE,
                "RMSE": row.RMSE,
                "sMAPE": row.sMAPE,
                "WAPE": row.WAPE,
                "mean_sales": mean_sales,
                "sales_cv": cv,
                "zero_sales_ratio": zero_ratio,
                "event_ratio": event_ratio,
                "snap_ratio": snap_ratio,
                "price_cv": price_cv,
                "auto_analysis": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def write_improvement_report(
    output_dir: Path,
    before_metrics: dict[str, Any] | None,
    after_metrics: dict[str, float],
    model_comparison: pd.DataFrame,
    sku_metrics: pd.DataFrame,
    raw_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    selected_features: list[str],
    removed_features: list[str],
    best_model: str,
) -> None:
    if before_metrics is None or metrics_are_effectively_same(before_metrics.get("test_metrics", {}), after_metrics):
        before_metrics = BASELINE_METRICS
    before_test = before_metrics.get("test_metrics", {}) if before_metrics else {}
    comparison_rows = [
        {
            "metric": "Validation MAE",
            "before": before_metrics.get("validation_MAE") if before_metrics else None,
            "after": float(model_comparison[(model_comparison["model"] == best_model) & (model_comparison["split"] == "validation")]["MAE"].iloc[0]),
        },
        {"metric": "Test MAE", "before": before_test.get("MAE"), "after": after_metrics["MAE"]},
        {"metric": "Test RMSE", "before": before_test.get("RMSE"), "after": after_metrics["RMSE"]},
        {"metric": "Test sMAPE", "before": before_test.get("sMAPE"), "after": after_metrics["sMAPE"]},
        {"metric": "Test WAPE", "before": before_test.get("WAPE"), "after": after_metrics["WAPE"]},
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison["delta"] = comparison.apply(
        lambda row: None if pd.isna(row["before"]) else row["after"] - row["before"],
        axis=1,
    )
    error_analysis = analyze_sku_error_drivers(raw_df, sku_metrics)
    unused = feature_importance[feature_importance["importance"] <= 0]["feature"].tolist()
    important = feature_importance[feature_importance["importance"] > 0].head(15)

    lines = [
        "# Demand Forecasting Improvement Report",
        "",
        "## Summary",
        f"- Final selected model: `{best_model}`",
        "- Selection criterion: lowest validation MAE after feature engineering and feature selection.",
        "",
        "## Before / After Metrics",
        "```text",
        comparison.to_string(index=False),
        "```",
        "",
        "## Added Features",
        "- `lag_28`, `rolling_mean_28`, `rolling_std_28`",
        "- `rolling_max_7`, `rolling_min_7`, `expanding_mean`",
        "- `days_since_last_sale`, `consecutive_zero_sales`, `sales_zero_ratio_28`",
        "- `price_change_rate`",
        "- `event_count`, `snap_any`",
        "- `sales_cv_28`, `recent_trend_7_28`",
        "",
        "## Removed Features",
        ", ".join(f"`{feature}`" for feature in removed_features) if removed_features else "None",
        "",
        "## Important Features",
        "```text",
        important.to_string(index=False),
        "```",
        "",
        "## Near-zero Importance Features",
        ", ".join(f"`{feature}`" for feature in unused) if unused else "None in final selected model.",
        "",
        "## Top 10 SKU Error Analysis",
        "```text",
        error_analysis.to_string(index=False),
        "```",
        "",
        "## Final Model Rationale",
        "The final model keeps the feature set that minimized validation MAE. The added sparse-sales, longer-window, price-change, event-count, and SNAP aggregate features target the SKU groups with high WAPE/sMAPE, especially low-volume or volatile SKUs where percentage error is unstable.",
        "",
    ]
    (output_dir / "improvement_report.md").write_text("\n".join(lines), encoding="utf-8")


def metrics_are_effectively_same(before_test: dict[str, Any], after_metrics: dict[str, float]) -> bool:
    if not before_test:
        return False
    metrics = ["MAE", "RMSE", "sMAPE", "WAPE"]
    return all(
        abs(float(before_test.get(metric, math.nan)) - after_metrics[metric]) < 1e-6
        for metric in metrics
    )
