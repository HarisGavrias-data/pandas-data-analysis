"""
Data Cleaning Script
Description: Cleans raw sales dataset and exports an analysis-ready CSV.

Input:  data/raw_sales_data.csv
Output: cleaned/clean_sales_data.csv
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


def standardize_city(city: str) -> str:
    """Normalize city names (trim, collapse spaces, title-case)."""
    if pd.isna(city):
        return np.nan
    s = str(city).strip()
    s = re.sub(r"\s+", " ", s)
    # common casing issues like PARIS, london -> Paris, London
    return s.title()


def standardize_name(name: str) -> str:
    """Normalize customer names (trim, collapse spaces, title-case)."""
    if pd.isna(name):
        return np.nan
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)
    return s.title()


def standardize_product(product: str) -> str:
    """Normalize product strings (trim, collapse spaces, title-case)."""
    if pd.isna(product):
        return np.nan
    s = str(product).strip()
    s = re.sub(r"\s+", " ", s)
    return s.title()


def main() -> None:
    root = Path(__file__).resolve().parents[1]  
    in_path = root / "data" / "raw_sales_data.csv"
    out_dir = root / "cleaned"
    out_path = out_dir / "clean_sales_data.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)

    print("=== Raw snapshot ===")
    print(df.head(3))
    print("\nRaw rows:", len(df))
    print("Raw duplicate rows:", int(df.duplicated().sum()))
    print("Raw missing values:\n", df.isna().sum())

    # 1) Strip/standardize text columns
    df["customer_name"] = df["customer_name"].apply(standardize_name)
    df["city"] = df["city"].apply(standardize_city)
    df["product"] = df["product"].apply(standardize_product)

    # 2) Coerce numerics
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # 3) Parse dates (mixed formats in the messy file)
    # dayfirst=True helps with formats like 03-06-2023 and 14-06-2023.
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", dayfirst=True)

    # 4) Handle missing values
    # Quantity: if missing, assume 1 (common assumption for line items) and flag it.
    df["quantity_was_missing"] = df["quantity"].isna()
    df["quantity"] = df["quantity"].fillna(1)

    # Price: if missing, keep NaN (can't infer safely). We'll drop rows with missing price later.
    # City/name/product: keep NaN, but you could also fill with "Unknown" if you want.

    # 5) Remove duplicate rows (exact duplicates)
    df = df.drop_duplicates()

    # 6) Validate & handle duplicate order_id situations
    dup_order_counts = df["order_id"].value_counts()
    remaining_dup_orders = dup_order_counts[dup_order_counts > 1]
    if not remaining_dup_orders.empty:
        print("\nNOTE: Some order_id values appear multiple times (could be multi-line orders):")
        print(remaining_dup_orders.head(10))

    # 7) Drop rows that are not usable for analysis (e.g., missing critical numeric/date)
    before = len(df)
    df = df.dropna(subset=["price", "quantity", "order_date"])
    dropped = before - len(df)
    if dropped:
        print(f"\nDropped {dropped} rows due to missing price/quantity/order_date.")

    # 8) Create derived columns
    df["revenue"] = df["price"] * df["quantity"]

    # 9) Final tidy: consistent column order
    ordered_cols = [
        "order_id",
        "customer_name",
        "city",
        "product",
        "price",
        "quantity",
        "order_date",
        "revenue",
        "quantity_was_missing",
    ]
    df = df[ordered_cols]

    # 10) Export
    df.to_csv(out_path, index=False)

    print("\n=== Clean snapshot ===")
    print(df.head(3))
    print("\nClean rows:", len(df))
    print("Clean duplicate rows:", int(df.duplicated().sum()))
    print("Clean missing values:\n", df.isna().sum())
    print(f"\nSaved cleaned dataset → {out_path}")


if __name__ == "__main__":
    main()
