import pandas as pd
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES_PATH = os.path.join(ROOT, "data", "products", "m5_small.csv")
META_PATH = os.path.join(ROOT, "data", "embeddings", "metadata.csv")

OUTPUT_PATH = os.path.join(ROOT, "data", "products", "merge_dataset.csv")

def preprocess():

    # ======================
    # 1. LOAD
    # ======================
    df = pd.read_csv(SALES_PATH)
    meta = pd.read_csv(META_PATH)

    # ======================
    # 2. DROP USELESS COLS
    # ======================
    drop_cols = [
        "id", "dept_id", "cat_id", "store_id", "state_id"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ======================
    # 3. DATE +10 YEARS
    # ======================
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"] + pd.DateOffset(years=10)

    # ======================
    # 4. ITEM → SKU MAPPING
    # ======================
    # item_id 예: FOODS_2_019
    # metadata sku_id 예: 10060
    #
    # ⚠️ 여기서는 "강제 매핑 (20개 랜덤 or 순서 기반)"
    # 둘 다 20개라 했으니 순서 기반으로 안전하게 처리

    items = df["item_id"].unique()[:20]
    skus = meta["sku_id"].unique()[:20]

    mapping = dict(zip(items, skus))

    df["sku_id"] = df["item_id"].map(mapping)

    # product_name 매칭
    meta_small = meta.drop_duplicates("sku_id")[
    ["sku_id", "product_name", "image_path", "filename", "height", "angle"]]
    df = df.merge(meta_small, on="sku_id", how="left")

    # ======================
    # 5. CLEAN COLUMN ORDER
    # ======================
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

    # ======================
    # 6. SORT
    # ======================
    df = df.sort_values(["sku_id", "date"])

    # ======================
    # 7. SAVE
    # ======================
    df.to_csv(OUTPUT_PATH, index=False)

    print("DONE")
    print(df.head())
    print("Shape:", df.shape)


if __name__ == "__main__":
    preprocess()