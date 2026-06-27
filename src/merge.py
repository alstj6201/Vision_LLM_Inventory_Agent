import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES_PATH = os.path.join(ROOT, "data", "products", "m5_small.csv")
META_PATH = os.path.join(ROOT, "data", "embeddings", "metadata.csv")

OUTPUT_PATH = os.path.join(ROOT, "data", "products", "merge_dataset.csv")

def preprocess():

    df = pd.read_csv(SALES_PATH)
    meta = pd.read_csv(META_PATH)

    drop_cols = [
        "id", "dept_id", "cat_id", "store_id", "state_id"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"] + pd.DateOffset(years=10)

    items = df["item_id"].unique()[:20]
    skus = meta["sku_id"].unique()[:20]

    mapping = dict(zip(items, skus))

    df["sku_id"] = df["item_id"].map(mapping)

    meta_small = meta.drop_duplicates("sku_id")[
    ["sku_id", "product_name", "image_path", "filename", "height", "angle"]]
    df = df.merge(meta_small, on="sku_id", how="left")

    keep_cols = [
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
        "sell_price"
    ]

    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.sort_values(["sku_id", "date"])
    df.to_csv(OUTPUT_PATH, index=False)

    print("DONE")
    print(df.head())
    print("Shape:", df.shape)


if __name__ == "__main__":
    preprocess()