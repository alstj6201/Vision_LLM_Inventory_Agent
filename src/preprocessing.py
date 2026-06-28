import pandas as pd

BASE_PATH = "/content"

SALES_PATH = f"{BASE_PATH}/sales_train_evaluation.csv"
CAL_PATH = f"{BASE_PATH}/calendar.csv"
PRICE_PATH = f"{BASE_PATH}/sell_prices.csv"

OUTPUT_PATH = f"{BASE_PATH}/m5_small.csv"

def preprocess():

    sales = pd.read_csv(SALES_PATH)
    calendar = pd.read_csv(CAL_PATH)
    prices = pd.read_csv(PRICE_PATH)

    id_cols = [
        "id", "item_id", "dept_id",
        "cat_id", "store_id", "state_id"
    ]

    sales_long = sales.melt(
        id_vars=id_cols,
        var_name="d",
        value_name="sales"
    )

    sales_long["d"] = sales_long["d"].str.replace("d_", "").astype(int)


    store = sales_long["store_id"].value_counts().idxmax()
    sales_long = sales_long[sales_long["store_id"] == store]

    item_stats = sales_long.groupby("item_id")["sales"].agg(["std", "mean"])
    item_stats["score"] = item_stats["std"] * item_stats["mean"]
    top_items = item_stats["score"].sort_values(ascending=False).head(20).index
    sales_long = sales_long[sales_long["item_id"].isin(top_items)]

    sales_long = sales_long[sales_long["item_id"].isin(top_items)]

    calendar["d"] = calendar["d"].str.replace("d_", "").astype(int)
    df = sales_long.merge(calendar, on="d", how="left")

    df = df.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    df = df.sort_values(["item_id", "d"])


    df["sell_price"] = df["sell_price"].fillna(method="ffill")
    df["sales"] = df["sales"].fillna(0)

    df.to_csv(OUTPUT_PATH, index=False)

    print("DONE")
    print("Store:", store)
    print("Items:", len(top_items))
    print("Shape:", df.shape)


if __name__ == "__main__":
    preprocess()