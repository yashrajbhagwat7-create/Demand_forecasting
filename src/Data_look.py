import pandas as pd

import numpy as np


import matplotlib as plt

 
from datasets import load_dataset

datasets = load_dataset(
    "Dingdong-Inc/FreshRetailNet-50K",
    split="train[:50000]"
)

#print(datasets)
#print(datasets.column_names)
#print(datasets[0])
#
#
df=datasets.to_pandas()
#print(df.shape)
#print(df.head())
#print(df.describe())



# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


# --------------------------------------------------
# Unique values
# --------------------------------------------------

print("\n" + "=" * 60)
print("UNIQUE VALUES")
print("=" * 60)

for column in df.columns:

    if column in ["hours_sale", "hours_stock_status"]:
        print(f"{column}: list/array data")
    else:
        print(f"{column}: {df[column].nunique()} unique values")


# --------------------------------------------------
# Date information
# --------------------------------------------------

df["dt"] = pd.to_datetime(df["dt"])

print("\n" + "=" * 60)
print("DATE INFORMATION")
print("=" * 60)

print("Minimum date:", df["dt"].min())
print("Maximum date:", df["dt"].max())
print("Number of unique dates:", df["dt"].nunique())


# --------------------------------------------------
# Store / product information
# --------------------------------------------------

print("\n" + "=" * 60)
print("STORE / PRODUCT INFORMATION")
print("=" * 60)

print("Unique stores:", df["store_id"].nunique())
print("Unique products:", df["product_id"].nunique())
print("Unique cities:", df["city_id"].nunique())


# --------------------------------------------------
# Target information
# --------------------------------------------------

print("\n" + "=" * 60)
print("SALES INFORMATION")
print("=" * 60)

print(df["sale_amount"].describe())


# --------------------------------------------------
# Stockout information
# --------------------------------------------------

print("\n" + "=" * 60)
print("STOCKOUT INFORMATION")
print("=" * 60)

print(df["stock_hour6_22_cnt"].describe())

print(
    "\nRows with stockout hours > 0:",
    (df["stock_hour6_22_cnt"] > 0).sum()
)

print(
    "Percentage with stockout:",
    round(
        (df["stock_hour6_22_cnt"] > 0).mean() * 100,
        2
    ),
    "%"
)


# --------------------------------------------------
# Duplicate check
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

print(
    "Duplicate store-product-date combinations:",
    df.duplicated(
        subset=["store_id", "product_id", "dt"]
    ).sum()
)