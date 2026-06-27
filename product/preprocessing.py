import pandas as pd

# 데이터 읽기
df = pd.read_csv("retail_sales.csv")

# 날짜 변환
df["Date"] = pd.to_datetime(df["Date"])


# Grocery만 선택
df = df[df["Category"] == "Groceries"].copy()


# 필요한 컬럼
use_cols = [
    "Date",
    "Product ID",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Weather Condition",
    "Holiday/Promotion",
    "Seasonality"
]

df = df[use_cols]


# Product ID + Date 기준 정렬
df = df.sort_values(
    ["Product ID", "Date"]
)


# 결측 확인
print(df.isnull().sum())


# 상품 개수 확인
print(
    "상품 개수:",
    df["Product ID"].nunique()
)

df.to_csv(
    "grocery_sales_processed.csv",
    index=False
)